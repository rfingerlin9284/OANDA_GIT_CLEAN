#!/usr/bin/env python3
"""
trade_engine.py
RBOTZILLA_OANDA_CLEAN — Phase 7
Label: NEW_CLEAN_REWRITE

OANDA Practice-Only Trade Engine.
No Phoenix path assumptions. No ML. No Coinbase. No live mode.

Execution flow per cycle:
    1.  Load env + initialize connector
    2.  Startup verification (account, endpoint, open trades)
    3.  Scan each configured pair → fetch M15 candles
    4.  Run scan_symbol() → optional AggregatedSignal
    5.  Dedup: skip if symbol already active this cycle or in broker
    6.  Log CANDIDATE_FOUND
    7.  Call check_broker_tradability() → may block here
    8.  Log gate result (ORDER_SUBMIT_ALLOWED or a *_BLOCK)
    9.  If allowed: log "→ Placing", place via OCO, confirm trade_id
    10. Log TRADE_OPENED only on confirmed broker trade_id
    11. Sleep scan_fast_seconds, loop
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Set

# ── Local imports (no Phoenix paths) ──────────────────────────────────────────
from brokers.oanda_connector import get_oanda_connector
from strategies.multi_signal_engine import scan_symbol, AggregatedSignal
from engine.broker_tradability_gate import (
    check_broker_tradability, validate_oco_payload,
    check_submit_response,
    set_cooldown,
)
from engine.trade_manager import TradeManager
from util.narration_logger import (
    log_event, log_trade_opened, log_gate_block, log_narration,
    CANDIDATE_FOUND, ORDER_SUBMIT_ALLOWED, SYMBOL_ALREADY_ACTIVE_BLOCK,
    TRADE_OPENED, TRADE_OPEN_FAILED, ENGINE_STARTED, ENGINE_STOPPED,
    SIGNAL_SCAN_COMPLETE, OCO_VALIDATION_BLOCK, ORDER_SUBMIT_BLOCK,
    MARGIN_GATE_BLOCKED, ATTACH_ONLY_BLOCK,
)

# ── Env-driven config ──────────────────────────────────────────────────────────
MAX_POSITIONS          = int(os.getenv("RBOT_MAX_POSITIONS",           "12"))
MAX_NEW_PER_CYCLE      = int(os.getenv("RBOT_MAX_NEW_TRADES_PER_CYCLE", "2"))
SCAN_FAST_SECONDS      = int(os.getenv("RBOT_SCAN_FAST_SECONDS",       "60"))
SCAN_SLOW_SECONDS      = int(os.getenv("RBOT_SCAN_SLOW_SECONDS",       "300"))
MIN_CONFIDENCE         = float(os.getenv("RBOT_MIN_SIGNAL_CONFIDENCE", "0.68"))
MIN_VOTES              = int(os.getenv("RBOT_MIN_VOTES",               "3"))
CANDLE_COUNT           = int(os.getenv("RBOT_CANDLE_COUNT",            "250"))
CANDLE_GRANULARITY     = os.getenv("RBOT_CANDLE_GRANULARITY",          "M15")
MIN_FREE_MARGIN_PCT    = float(os.getenv("RBOT_MIN_FREE_MARGIN_PCT",   "0.20"))  # block if <20% margin free

# ── Safety mode flags ─────────────────────────────────────────────────────────
# Both default to False so existing deployments are unaffected without .env change.
# Set ATTACH_ONLY=true when Phoenix is the active opener to prevent dual-placement.
ATTACH_ONLY          = os.getenv("ATTACH_ONLY",          "false").strip().lower() == "true"
DISABLE_NEW_ENTRIES  = os.getenv("DISABLE_NEW_ENTRIES",  "false").strip().lower() == "true"

# Pairs to scan — JPY and GBP_NZD excluded due to spread; configurable via env
_DEFAULT_PAIRS = (
    "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD,"
    "EUR_GBP,EUR_JPY,GBP_JPY,EUR_AUD,EUR_CAD,"
    "AUD_JPY,CHF_JPY,GBP_CAD,GBP_CHF,AUD_CAD,"
    "AUD_CHF,CAD_JPY,NZD_JPY,EUR_NZD,NZD_CAD"
)
TRADING_PAIRS = [
    p.strip() for p in os.getenv("RBOT_TRADING_PAIRS", _DEFAULT_PAIRS).split(",")
    if p.strip()
]


class TradeEngine:
    """
    Practice-only OANDA trade engine.
    Single async loop: scan → gate → place → sleep.
    """

    def __init__(self):
        self.connector        = get_oanda_connector()
        self.active_positions: Dict[str, dict] = {}   # trade_id → position dict
        self.manager          = TradeManager(self.connector)
        self.is_running     = False

    # ── Startup verification ─────────────────────────────────────────────────

    def print_startup_banner(self) -> None:
        """Print verified broker state at startup. Does NOT use cached values."""
        print("\n" + "=" * 60)
        print("  RBOTZILLA OANDA CLEAN — PRACTICE ENGINE")
        print("=" * 60)
        try:
            info = self.connector.get_account_info()
            trades = self.connector.get_trades()
            print(f"  Environment   : {os.getenv('OANDA_ENVIRONMENT', 'practice')}")
            print(f"  Endpoint      : {self.connector.api_base}")
            print(f"  Account ID    : {self.connector.account_id}")
            print(f"  Balance       : ${info.balance:,.2f}")
            print(f"  Margin Used   : ${info.margin_used:,.2f}")
            print(f"  Broker Trades : {len(trades)}")
            print(f"  OCO Enforced  : YES (SL+TP mandatory on all orders)")
            print(f"  Pairs Scanning: {len(TRADING_PAIRS)}")
            print(f"  Max Positions : {MAX_POSITIONS}")
            print(f"  Min Confidence: {MIN_CONFIDENCE:.0%}")
        except Exception as e:
            print(f"  WARNING: startup broker query failed: {e}")
        print("=" * 60 + "\n")

        log_event(ENGINE_STARTED, symbol="SYSTEM", venue="startup", details={
            "account_id":   self.connector.account_id,
            "endpoint":     self.connector.api_base,
            "pairs_count":  len(TRADING_PAIRS),
            "max_positions": MAX_POSITIONS,
        })

    # ── Broker dedup ─────────────────────────────────────────────────────────

    def _symbol_is_active(self, symbol: str) -> bool:
        """Return True if symbol already has an open position (local or broker)."""
        for pos in self.active_positions.values():
            if pos.get("symbol") == symbol:
                return True
        try:
            for t in (self.connector.get_trades() or []):
                if t.get("instrument") == symbol:
                    return True
        except Exception:
            pass
        return False

    # ── Single scan cycle ────────────────────────────────────────────────────

    async def _run_scan_cycle(self) -> int:
        """
        One full scan cycle.
        Returns: number of trades successfully placed.
        """
        # ── Live broker position count + full sync to active_positions ─────────
        try:
            broker_trades = self.connector.get_trades() or []
            broker_open   = len(broker_trades)
            # Rebuild broker truth: {trade_id: instrument}
            broker_by_id  = {t.get("id", t.get("tradeID", "")): t.get("instrument", "") for t in broker_trades}
            broker_symbols = set(broker_by_id.values())

            # Remove local positions no longer in broker
            stale = [tid for tid in list(self.active_positions) if tid not in broker_by_id]
            for tid in stale:
                self.active_positions.pop(tid, None)

            # Add broker positions not yet tracked locally
            for tid, instrument in broker_by_id.items():
                if tid and tid not in self.active_positions:
                    self.active_positions[tid] = {"symbol": instrument, "synced_from_broker": True}
        except Exception:
            broker_open   = len(self.active_positions)
            broker_symbols = set()

        slots_left  = max(0, MAX_POSITIONS - broker_open)
        cycle_limit = min(slots_left, MAX_NEW_PER_CYCLE)

        if cycle_limit == 0:
            return 0

        # ── ATTACH_ONLY hard block ─────────────────────────────────────────────
        # When ATTACH_ONLY=true or DISABLE_NEW_ENTRIES=true, no new trades are
        # submitted. Broker sync above still runs so active_positions stays accurate.
        # Phoenix is the verified source opener — this engine must not compete.
        if ATTACH_ONLY or DISABLE_NEW_ENTRIES:
            log_event(
                ATTACH_ONLY_BLOCK, symbol="SYSTEM", venue="engine",
                details={
                    "reason":               "ATTACH_ONLY mode active — new trade entries disabled",
                    "attach_only_flag":     ATTACH_ONLY,
                    "disable_entries_flag": DISABLE_NEW_ENTRIES,
                    "broker_open":          broker_open,
                    "slots_left":           slots_left,
                },
            )
            print(f"  [ATTACH_ONLY]  broker_open={broker_open}  no new entries this cycle")
            return 0

        # ── Live account for margin gate ──────────────────────────────────────
        try:
            acct         = self.connector.get_account_info()
            nav          = acct.balance + acct.unrealized_pl
            margin_used  = acct.margin_used
            free_margin  = nav - margin_used
            free_margin_pct = free_margin / nav if nav > 0 else 0.0
        except Exception:
            nav = free_margin_pct = 1.0   # fail-open on NAV query only

        # Collect qualifying signals
        qualified = []
        for symbol in TRADING_PAIRS:
            try:
                candles = self.connector.get_historical_data(
                    symbol, count=CANDLE_COUNT, granularity=CANDLE_GRANULARITY
                )
                if not candles or len(candles) < 50:
                    continue
                sig = scan_symbol(
                    symbol, candles,
                    min_confidence=MIN_CONFIDENCE,
                    min_votes=MIN_VOTES,
                )
                if sig:
                    qualified.append(sig)
            except Exception:
                continue  # log_event here would produce noise per-pair — skip

        qualified.sort(key=lambda s: s.confidence, reverse=True)

        log_event(SIGNAL_SCAN_COMPLETE, symbol="SYSTEM", venue="signal_scan", details={
            "pairs_scanned":    len(TRADING_PAIRS),
            "candidates_found": len(qualified),
            "cycle_limit":      cycle_limit,
            "open_slots":       slots_left,
        })

        # ── Placement loop ────────────────────────────────────────────────────
        placed_this_cycle: set = set()
        placed_count = 0

        for sig in qualified[:cycle_limit]:
            symbol = sig.symbol

            # ── Dedup: broker symbols + local + this cycle ─────────────────
            if (symbol.upper() in placed_this_cycle
                    or symbol in broker_symbols
                    or self._symbol_is_active(symbol)):
                log_gate_block(symbol, SYMBOL_ALREADY_ACTIVE_BLOCK, {"symbol": symbol})
                continue

            # ── Margin gate (live NAV) ──────────────────────────────────────
            if free_margin_pct < MIN_FREE_MARGIN_PCT:
                log_gate_block(symbol, MARGIN_GATE_BLOCKED, {
                    "free_margin_pct": round(free_margin_pct, 3),
                    "min_required":    MIN_FREE_MARGIN_PCT,
                })
                print(f"  BLOCKED    {symbol} — MARGIN_GATE_BLOCKED free={free_margin_pct:.1%}")
                break   # margin applies to all pairs this cycle

            # ── Phase 1: CANDIDATE_FOUND ───────────────────────────────────
            log_event(CANDIDATE_FOUND, symbol=symbol, venue="signal_scan", details={
                "symbol":     symbol,
                "direction":  sig.direction,
                "confidence": round(sig.confidence, 4),
                "votes":      sig.votes,
                "detectors":  sig.detectors_fired,
                "session":    sig.session,
            })
            print(
                f"  CANDIDATE  {symbol} {sig.direction} "
                f"conf={sig.confidence:.1%} ({sig.votes}v)"
            )

            # ── Phase 2: broker tradability gate ──────────────────────────
            gate = check_broker_tradability(
                self.connector, symbol,
                placed_this_cycle=placed_this_cycle,
            )
            if not gate["allowed"]:
                log_gate_block(symbol, gate["event"], gate["detail"])
                print(f"  BLOCKED    {symbol} — {gate['event']} {gate['detail']}")
                continue

            # Live mid-price from gate — used as entry_price for notional calc
            # (MARKET orders have no submitted price; Charter needs a real number)
            live_mid = (gate.get("live_price") or {}).get("mid") or 0.0

            # ── Phase 3: ORDER_SUBMIT_ALLOWED ─────────────────────────────
            log_event(ORDER_SUBMIT_ALLOWED, symbol=symbol, venue="tradability_gate", details={
                "symbol":     symbol,
                "direction":  sig.direction,
                "confidence": round(sig.confidence, 4),
            })
            print(f"  → Placing  {symbol} {sig.direction} conf={sig.confidence:.1%}")

            # ── Phase 4: OCO payload validation ───────────────────────────
            units = self._compute_units(symbol, sig, nav)
            oco_check = validate_oco_payload(
                symbol=symbol,
                direction=sig.direction,
                entry_price=live_mid,
                stop_loss=sig.sl,
                take_profit=sig.tp,
                units=units,
            )
            if not oco_check["valid"]:
                log_gate_block(symbol, OCO_VALIDATION_BLOCK, oco_check)
                print(f"  BLOCKED    {symbol} — OCO_VALIDATION_BLOCK {oco_check['reason']}")
                continue

            # ── Phase 5: place OCO order ───────────────────────────────────
            try:
                result = self.connector.place_oco_order(
                    instrument=symbol,
                    entry_price=live_mid,
                    stop_loss=sig.sl,
                    take_profit=sig.tp,
                    units=units,
                    order_type="MARKET",
                )

                # ── Phase 6: verify broker response ───────────────────────
                confirm = check_submit_response(result, symbol)
                if not confirm["confirmed"]:
                    log_event(ORDER_SUBMIT_BLOCK, symbol=symbol, venue="oanda", details={
                        "symbol": symbol, "error": confirm["error"]
                    })
                    print(f"  ✗ REJECTED {symbol} — {confirm['error']}")
                    continue

                trade_id = confirm["trade_id"]
                set_cooldown(symbol)
                placed_this_cycle.add(symbol.upper())
                self.active_positions[trade_id] = {
                    "symbol":      symbol,
                    "direction":   sig.direction,
                    "stop_loss":   sig.sl,
                    "take_profit": sig.tp,
                    "confidence":  sig.confidence,
                    "session":     sig.session,
                    "opened_at":   datetime.now(timezone.utc).isoformat(),
                }
                log_trade_opened(
                    symbol=symbol, direction=sig.direction, trade_id=trade_id,
                    entry=result.get("entry_price", live_mid),
                    stop_loss=sig.sl, take_profit=sig.tp,
                    size=units,
                    confidence=sig.confidence, votes=sig.votes,
                    detectors=sig.detectors_fired, session=sig.session,
                )
                placed_count += 1
                print(f"  ✓ OPENED   {symbol} trade_id={trade_id}")

            except Exception as e:
                log_event(TRADE_OPEN_FAILED, symbol=symbol, venue="oanda", details={
                    "symbol": symbol, "error": str(e)
                })
                print(f"  ✗ ERROR    {symbol} — {e}")

        return placed_count

    def _compute_units(self, symbol: str, sig: AggregatedSignal, nav: float = 0.0) -> int:
        """
        Compute position size.
        Uses RBOT_BASE_UNITS (env). Must be >= 20000 to meet Charter minimum.
        Positive = BUY, negative = SELL.
        """
        base_units = int(os.getenv("RBOT_BASE_UNITS", "20000"))
        return base_units if sig.direction == "BUY" else -base_units

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self.is_running = True
        self.print_startup_banner()

        while self.is_running:
            try:
                placed = await self._run_scan_cycle()
                # ── Trail management: runs every cycle regardless of ATTACH_ONLY ─
                await self.manager.tick()

                open_now = len(self.active_positions)

                if open_now >= MAX_POSITIONS:
                    print(f"  Positions full ({open_now}/{MAX_POSITIONS}) — waiting {SCAN_SLOW_SECONDS}s")
                    await asyncio.sleep(SCAN_SLOW_SECONDS)
                else:
                    print(f"  Slots open ({open_now}/{MAX_POSITIONS}) — rescanning in {SCAN_FAST_SECONDS}s")
                    await asyncio.sleep(SCAN_FAST_SECONDS)

            except KeyboardInterrupt:
                print("\n  Stopping engine…")
                self.is_running = False
                log_event(ENGINE_STOPPED, symbol="SYSTEM", venue="engine", details={
                    "reason": "keyboard_interrupt"
                })
            except Exception as e:
                print(f"  Engine cycle error: {e}")
                await asyncio.sleep(30)


def main():
    engine = TradeEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
