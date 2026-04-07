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
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Set

# Ensure repo root is on sys.path (engine/ lives one level below)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Local imports (no Phoenix paths) ──────────────────────────────────────────
from brokers.oanda_connector import get_oanda_connector
from strategies.multi_signal_engine import scan_symbol, AggregatedSignal
from engine.broker_tradability_gate import (
    check_broker_tradability, validate_oco_payload,
    check_submit_response,
    set_cooldown,
)
from engine.trade_manager import TradeManager
from engine.pre_market_scanner import PreMarketScanner
from engine.capital_router import CapitalRouter, compute_compounded_units, compute_watermark_compounded_units
# regime_detector intentionally not imported — vote+confidence gate is the quality filter
from engine.mean_reversion_scanner import scan_sideways_symbol
from engine.strategy_pipelines import (
    run_momentum_pipeline,
    run_reversal_pipeline,
    run_meanrev_pipeline,
    run_scalp_pipeline,
)
from foundation.rick_charter import RickCharter
from util.broker_clock import BrokerClock, broker_now, broker_now_eastern
from util.narration_logger import (
    log_event, log_trade_opened, log_gate_block, log_narration,
    CANDIDATE_FOUND, ORDER_SUBMIT_ALLOWED, SYMBOL_ALREADY_ACTIVE_BLOCK,
    TRADE_OPENED, TRADE_OPEN_FAILED, ENGINE_STARTED, ENGINE_STOPPED,
    SIGNAL_SCAN_COMPLETE, OCO_VALIDATION_BLOCK, ORDER_SUBMIT_BLOCK,
    MARGIN_GATE_BLOCKED, ATTACH_ONLY_BLOCK,
    CAPITAL_REALLOC_DECIDED, CAPITAL_REALLOC_FAILED,
)

# ── Env-driven config ──────────────────────────────────────────────────────────
MAX_POSITIONS          = int(os.getenv("RBOT_MAX_POSITIONS",           "12"))
MAX_NEW_PER_CYCLE      = int(os.getenv("RBOT_MAX_NEW_TRADES_PER_CYCLE", "4"))
SCAN_FAST_SECONDS      = int(os.getenv("RBOT_SCAN_FAST_SECONDS",       "60"))
SCAN_SLOW_SECONDS      = int(os.getenv("RBOT_SCAN_SLOW_SECONDS",       "300"))
MIN_CONFIDENCE         = float(os.getenv("RBOT_MIN_SIGNAL_CONFIDENCE", "0.75"))  # raised 0.70→0.75: Phoenix production standard  # PATCH#1
MIN_VOTES              = int(os.getenv("RBOT_MIN_VOTES",               "3"))
CANDLE_COUNT           = int(os.getenv("RBOT_CANDLE_COUNT",            "250"))
CANDLE_GRANULARITY     = os.getenv("RBOT_CANDLE_GRANULARITY",          "M15")
MIN_FREE_MARGIN_PCT    = float(os.getenv("RBOT_MIN_FREE_MARGIN_PCT",   "0.20"))  # block if <20% margin free
CHARTER_MIN_NOTIONAL_USD = float(os.getenv("RBOT_CHARTER_MIN_NOTIONAL_USD", "15000"))
CHARTER_TARGET_NOTIONAL_USD = float(os.getenv("RBOT_CHARTER_TARGET_NOTIONAL_USD", "17500"))
CHARTER_MIN_RR = float(os.getenv("RBOT_CHARTER_MIN_RR", "3.26"))
# ── Phoenix quality-picker sizing bounds ──────────────────────────────────────
CHARTER_MAX_NOTIONAL_USD = float(os.getenv("RBOT_MAX_NOTIONAL_USD", "30000"))

# ── TRANSCRIPT EDGE: Circuit Breaker + Directional Filters ────────────────────
DAILY_MAX_LOSS_USD     = float(os.getenv("RBOT_DAILY_MAX_LOSS_USD",    "150"))
DAILY_MAX_GAIN_USD     = float(os.getenv("RBOT_DAILY_MAX_GAIN_USD",    "300"))
EMA200_GATE_ENABLED    = os.getenv("RBOT_EMA200_GATE_ENABLED", "true").lower() == "true"
RSI_GATE_ENABLED       = os.getenv("RBOT_RSI_GATE_ENABLED", "true").lower() == "true"
RSI_OB_LEVEL           = float(os.getenv("RBOT_RSI_OB", "70"))
RSI_OS_LEVEL           = float(os.getenv("RBOT_RSI_OS", "30"))
KELLY_SIZING_ENABLED   = os.getenv("RBOT_KELLY_SIZING_ENABLED", "true").lower() == "true"
# Source: Decoded Liquidity Algorithm — volume confirmation on entry
VOLUME_GATE_ENABLED    = os.getenv("RBOT_VOLUME_GATE_ENABLED", "true").lower() == "true"
VOLUME_GATE_MULT       = float(os.getenv("RBOT_VOLUME_GATE_MULT", "1.2"))
# Source: DXY transcript — USD/CHF as DXY proxy for correlation filter
DXY_GATE_ENABLED       = os.getenv("RBOT_DXY_GATE_ENABLED", "true").lower() == "true"
# Source: MACD Divergence transcript — detect price/MACD disagreement
MACD_DIV_ENABLED       = os.getenv("RBOT_MACD_DIV_ENABLED", "true").lower() == "true"
# Source: Richie Nasser — London/NY sessions get confidence boost
SESSION_BOOST_ENABLED  = os.getenv("RBOT_SESSION_BOOST_ENABLED", "true").lower() == "true"

# ── Safety mode flags ─────────────────────────────────────────────────────────
# Both default to False so existing deployments are unaffected without .env change.
# Set ATTACH_ONLY=true when Phoenix is the active opener to prevent dual-placement.
ATTACH_ONLY          = os.getenv("ATTACH_ONLY",          "false").strip().lower() == "true"
DISABLE_NEW_ENTRIES  = os.getenv("DISABLE_NEW_ENTRIES",  "false").strip().lower() == "true"
PAIR_REENTRY_COOLDOWN_MINUTES = int(os.getenv("RBOT_PAIR_REENTRY_COOLDOWN_MINUTES", "60"))  # PATCH#3

# Pairs to scan — restricted to 10 major pairs for Phoenix-grade selectivity.
# Crosses (EUR_AUD, GBP_CAD, etc.) removed: lower liquidity + 0.90 off-session
# mult not strong enough to block them at the old 0.68 gate.
# Expand via RBOT_TRADING_PAIRS env var if you want crosses back.
_DEFAULT_PAIRS = (
    "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD,"
    "EUR_JPY,GBP_JPY,XAU_USD"  # PATCH#2
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
        # Charter compliance validation
        if hasattr(RickCharter, 'validate_pin'):
            if not RickCharter.validate_pin(841921):
                raise PermissionError("Invalid Charter PIN — cannot initialize trading engine")

        self.connector        = get_oanda_connector()
        self.active_positions: Dict[str, dict] = {}   # trade_id → position dict
        self.manager          = TradeManager(self.connector)
        self.is_running       = False
        self._pair_last_trade_ts: Dict[str, float] = {}
        self._pair_last_side: Dict[str, str] = {}
        self._cooldown_persist_path = Path("/tmp/rbotz_clean_cooldowns.json")
        self._load_cooldown_state()  # restore cooldowns saved by last run

        # ── Pre-market scanner (fires once per session boundary) ───────────────
        self.pre_scanner      = PreMarketScanner(self.connector)
        self._session_playbook = []   # ranked list from last pre-market scan

        # ── Capital router (snowball reallocation) — init_nav resolved at start ─
        self._initial_nav     = 0.0   # set in run() after first account query
        self._watermark_nav   = 0.0   # highest observed NAV since engine start
        self._router: Optional[CapitalRouter] = None

        print("  ✅ Foundation/Charter imported and validated")
        print(f"  ✅ TradeManager initialized (hard_stop=${self.manager._hard_stop_usd:.0f})")
        print("  ✅ PreMarketScanner initialized")
        print("  ✅ CapitalRouter ready (activates after first NAV read)")

        # ── Phoenix-mode: fixed SL/TP pips (0 = use signal's computed values) ──
        self._base_sl_pips = int(os.getenv("RBOT_SL_PIPS", "0"))
        self._sl_pips = self._base_sl_pips
        self._tp_pips = int(os.getenv("RBOT_TP_PIPS", "0"))

        # ── Chop Mode Config ────────────────────────────────────────────────
        self._chop_enabled = os.getenv("RBOT_CHOP_MODE_ENABLED", "false").lower() == "true"
        self._chop_start   = int(os.getenv("RBOT_CHOP_START_HOUR", "12"))
        self._chop_end     = int(os.getenv("RBOT_CHOP_END_HOUR", "3"))
        self._chop_units   = int(os.getenv("RBOT_CHOP_UNITS", "14000"))
        self._chop_sl_pips = int(os.getenv("RBOT_CHOP_SL_PIPS", "30"))
        self._chop_max_pos = int(os.getenv("RBOT_CHOP_MAX_POSITIONS", "12"))
        
        self.current_max_positions = int(os.getenv("RBOT_MAX_POSITIONS", "12"))
        self.is_chop_mode_active = False

        # ── QuantHedgeEngine (Phoenix port) — disabled by default ──────────
        # Hedge trades are placed with no signal gate or vote check, which adds
        # mechanical churn in ranging/cross-correlation-break markets.
        # Enable via RBOT_HEDGE_ENABLED=true in .env only after confirming edge.
        if os.getenv("RBOT_HEDGE_ENABLED", "false").strip().lower() == "true":
            try:
                from util.quant_hedge_engine import QuantHedgeEngine
                self._hedge_engine = QuantHedgeEngine()
                print("  ✅ QuantHedgeEngine armed")
            except Exception as e:
                print(f"  ⚠️ QuantHedgeEngine error: {e}")
                self._hedge_engine = None
        else:
            self._hedge_engine = None
            print("  ℹ️  QuantHedgeEngine disabled (RBOT_HEDGE_ENABLED=false)")

        # ── Transcript Edge: Daily P/L Circuit Breaker ─────────────────────
        self._daily_open_balance = None
        self._daily_trade_date = None
        self._circuit_breaker_tripped = False

    def _update_regime_state(self) -> None:
        """Evaluate if we are currently in the NY Afternoon / Tokyo Chop session."""
        if not self._chop_enabled:
            self.is_chop_mode_active = False
            self.current_max_positions = int(os.getenv("RBOT_MAX_POSITIONS", "12"))
            self._sl_pips = self._base_sl_pips
            return
            
        _now_et = broker_now_eastern()
        _h = _now_et.hour
        # If chop_start > chop_end (e.g., 12 PM to 3 AM), logically it wraps midnight
        if self._chop_start > self._chop_end:
            _in_chop = _h >= self._chop_start or _h < self._chop_end
        else:
            _in_chop = self._chop_start <= _h < self._chop_end
            
        self.is_chop_mode_active = _in_chop
        if _in_chop:
            self.current_max_positions = self._chop_max_pos
            self._sl_pips = self._chop_sl_pips if self._chop_sl_pips > 0 else self._base_sl_pips
        else:
            self.current_max_positions = int(os.getenv("RBOT_MAX_POSITIONS", "12"))
            self._sl_pips = self._base_sl_pips


    # ── Startup verification ─────────────────────────────────────────────────

    def print_startup_banner(self) -> None:
        """Print verified broker state at startup with timezone + time sync."""
        from util.narration_logger import dual_timestamp
        print("\n" + "=" * 60)
        print("  RBOTZILLA OANDA CLEAN — PRACTICE ENGINE")
        print("=" * 60)
        # ── Timezone + Broker Time Sync ──
        _now = dual_timestamp()
        print(f"  Engine Time   : {_now}")
        print(f"  Timezone      : US Eastern (NYC/NYSE)")
        try:
            _sync = self.connector.get_server_time()
            _drift = _sync.get("drift_ms", 0)
            _synced = _sync.get("synced", False)
            if _synced:
                _broker_ts = dual_timestamp(_sync["broker_utc"].isoformat())
                print(f"  Broker Time   : {_broker_ts}")
                if abs(_drift) < 2000:
                    print(f"  Clock Sync    : ✅ SYNCED (drift {_drift:+.0f}ms)")
                else:
                    print(f"  Clock Sync    : ⚠️  DRIFT DETECTED ({_drift:+.0f}ms)")
            else:
                print(f"  Clock Sync    : ⚠️  Could not verify broker time")
            # ── BrokerClock: set authoritative offset from this same sync result ──
            BrokerClock.instance().sync(self.connector)
        except Exception as _ts_err:
            print(f"  Clock Sync    : ⚠️  Time sync failed: {_ts_err}")
        try:
            info = self.connector.get_account_info()
            trades = self.connector.get_trades()
            print(f"  Environment   : {os.getenv('OANDA_ENVIRONMENT', 'practice')}")
            print(f"  Endpoint      : {self.connector.api_base}")
            print(f"  Account ID    : {self.connector.account_id}")
            print(f"  Balance       : ${info.balance:,.2f}")
            print(f"  Margin Used   : ${info.margin_used:,.2f}")
            print(f"  Broker Trades : {len(trades)}")
            print(f"  OCO Enforced  : YES (SL+TP+TS mandatory on all orders)")
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
        # ── Tick BrokerClock: auto-resyncs every 50 cycles ────────────────────
        BrokerClock.instance().tick(self.connector)

        # ── Live broker position count + full sync to active_positions ─────────
        try:
            broker_trades = self.connector.get_trades() or []
            broker_open   = len(broker_trades)
            # Rebuild broker truth: {trade_id: instrument}
            broker_by_id  = {t.get("id", t.get("tradeID", "")): t.get("instrument", "") for t in broker_trades}
            broker_symbols = set(broker_by_id.values())

            # ── Prevent Limit Order Spam: Add Pending Limit Orders to dedup list ──
            try:
                pending = self.connector.get_orders(state="PENDING") or []
                for p_ord in pending:
                    if p_ord.get("type") == "LIMIT" and p_ord.get("instrument"):
                        broker_symbols.add(p_ord["instrument"])
                        # Also count pending limit orders towards the max position slot limit
                        broker_open += 1
            except Exception:
                pass

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

        # Update dynamic sizing state before slot math
        self._update_regime_state()

        slots_left  = max(0, self.current_max_positions - broker_open)
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

        # ── TRANSCRIPT EDGE: Daily Circuit Breaker ─────────────────────────
        import datetime as _dt_mod
        _today = _dt_mod.date.today()
        if self._daily_trade_date != _today:
            self._daily_trade_date = _today
            self._circuit_breaker_tripped = False
            try:
                _acct_day = self.connector.get_account_info()
                self._daily_open_balance = _acct_day.balance
            except Exception:
                self._daily_open_balance = None

        if self._circuit_breaker_tripped:
            return 0

        if self._daily_open_balance is not None:
            try:
                _acct_cb = self.connector.get_account_info()
                _today_pl = _acct_cb.balance - self._daily_open_balance
                if _today_pl < 0 and abs(_today_pl) >= DAILY_MAX_LOSS_USD:
                    print(f"  🛑 CIRCUIT BREAKER — Today\'s loss ${abs(_today_pl):.2f} >= ${DAILY_MAX_LOSS_USD:.0f} limit. No new entries until tomorrow.")
                    self._circuit_breaker_tripped = True
                    return 0
                if _today_pl > 0 and _today_pl >= DAILY_MAX_GAIN_USD:
                    print(f"  🏆 DAILY TARGET HIT — Today\'s gain ${_today_pl:.2f} >= ${DAILY_MAX_GAIN_USD:.0f} target. Preserving profits.")
                    self._circuit_breaker_tripped = True
                    return 0
            except Exception:
                pass

        # ── TRANSCRIPT EDGE: DXY Proxy (USD/CHF 20-bar trend) ─────────────
        # Source: DXY transcript — USD/CHF has ~0.95 DXY correlation
        # Cached once per scan cycle, used inside per-pair loop
        _dxy_bias = None
        if DXY_GATE_ENABLED:
            try:
                _dxy_candles = self.connector.get_historical_data("USD_CHF", count=25, granularity="M15")
                _dxy_closes = [float(c.get("mid", {}).get("c", 0)) for c in _dxy_candles]
                if len(_dxy_closes) >= 20:
                    _dxy_sma20 = sum(_dxy_closes[-20:]) / 20.0
                    _dxy_now = _dxy_closes[-1]
                    if _dxy_now > _dxy_sma20 * 1.0005:  # USD strengthening
                        _dxy_bias = "USD_STRONG"
                    elif _dxy_now < _dxy_sma20 * 0.9995:  # USD weakening
                        _dxy_bias = "USD_WEAK"
            except Exception:
                pass

        # Collect qualifying signals
        # Pre-market playbook ordering: vetted symbols scan first
        if self._session_playbook:
            pb_symbols = [
                e.symbol for e in self._session_playbook
                if not e.vetoed and e.symbol in set(TRADING_PAIRS)
            ]
            pb_set = set(pb_symbols)
            scan_order = pb_symbols + [s for s in TRADING_PAIRS if s not in pb_set]
        else:
            scan_order = list(TRADING_PAIRS)

        qualified = []
        for symbol in scan_order:
            try:
                candles = self.connector.get_historical_data(
                    symbol, count=CANDLE_COUNT, granularity=CANDLE_GRANULARITY
                )
                if not candles or len(candles) < 50:
                    continue

                # ── Phase 9: Multi-Strategy Pipeline Scanner ───────────────────
                # Each pipeline runs its own detector subset independently.
                # Best qualifying signal for this pair wins.

                candidates = []

                # ── TRANSCRIPT EDGE: 200 EMA Directional Filter ────────────────
                _ema200_bias = None
                if EMA200_GATE_ENABLED:
                    _closes_m15 = [float(c.get("mid", {}).get("c", 0)) for c in candles]
                    if len(_closes_m15) >= 200:
                        _ema200 = sum(_closes_m15[:200]) / 200.0
                        _k200 = 2.0 / 201.0
                        for _px200 in _closes_m15[200:]:
                            _ema200 = (_px200 - _ema200) * _k200 + _ema200
                        _ema200_bias = "BUY" if _closes_m15[-1] > _ema200 else "SELL"

                # ── TRANSCRIPT EDGE: RSI Overbought/Oversold Gate ──────────────
                _rsi_block_buy = False
                _rsi_block_sell = False
                if RSI_GATE_ENABLED:
                    _cls_rsi = [float(c.get("mid", {}).get("c", 0)) for c in candles[-15:]]
                    if len(_cls_rsi) >= 15:
                        _gains = [max(_cls_rsi[i] - _cls_rsi[i-1], 0) for i in range(1, len(_cls_rsi))]
                        _losses = [max(_cls_rsi[i-1] - _cls_rsi[i], 0) for i in range(1, len(_cls_rsi))]
                        _avg_gain = sum(_gains) / 14.0
                        _avg_loss = sum(_losses) / 14.0
                        if _avg_loss > 0:
                            _rs = _avg_gain / _avg_loss
                            _rsi_val = 100.0 - (100.0 / (1.0 + _rs))
                        else:
                            _rsi_val = 100.0
                        if _rsi_val >= RSI_OB_LEVEL:
                            _rsi_block_buy = True
                        if _rsi_val <= RSI_OS_LEVEL:
                            _rsi_block_sell = True

                # ── TRANSCRIPT EDGE: MACD Divergence Detection ─────────────
                _macd_div_block_buy = False
                _macd_div_block_sell = False
                if MACD_DIV_ENABLED:
                    _cls_macd = [float(c.get("mid", {}).get("c", 0)) for c in candles[-30:]]
                    if len(_cls_macd) >= 26:
                        # EMA-12 and EMA-26
                        _e12 = sum(_cls_macd[:12]) / 12.0
                        _e26 = sum(_cls_macd[:26]) / 26.0
                        _k12 = 2.0 / 13.0
                        _k26 = 2.0 / 27.0
                        _macd_hist = []
                        for _i_m in range(26, len(_cls_macd)):
                            _e12 = (_cls_macd[_i_m] - _e12) * _k12 + _e12
                            _e26 = (_cls_macd[_i_m] - _e26) * _k26 + _e26
                            _macd_hist.append(_e12 - _e26)
                        if len(_macd_hist) >= 3:
                            # Bearish div: price rising, MACD histogram falling
                            if (_cls_macd[-1] > _cls_macd[-3] and
                                    _macd_hist[-1] < _macd_hist[-3]):
                                _macd_div_block_buy = True
                            # Bullish div: price falling, MACD histogram rising
                            if (_cls_macd[-1] < _cls_macd[-3] and
                                    _macd_hist[-1] > _macd_hist[-3]):
                                _macd_div_block_sell = True

                # ─ Pipeline 1: Momentum (SMA + EMA + Fib, 2-of-3, H4-confirmed) ─
                mom = run_momentum_pipeline(symbol, candles, min_confidence=MIN_CONFIDENCE)
                if mom and getattr(mom, 'signal_type', '') == 'trend':
                    try:
                        candles_h4 = self.connector.get_historical_data(symbol, count=100, granularity="H4")
                        closes_h4 = [float(c.get("mid", {}).get("c", 0)) for c in candles_h4]
                        if len(closes_h4) >= 55:
                            k = 2.0 / (56.0)
                            ema_h4 = sum(closes_h4[:55]) / 55.0
                            for px in closes_h4[55:]:
                                ema_h4 = (px - ema_h4) * k + ema_h4
                            price_h4 = closes_h4[-1]
                            if mom.direction == "BUY" and price_h4 < ema_h4:
                                mom.confidence -= 0.15
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM BUY scored -15% — H4 BEARISH")
                                log_gate_block(symbol, "MTF_SNIPER_PENALTY", {"h4_ema55": ema_h4, "price": price_h4})
                            elif mom.direction == "SELL" and price_h4 > ema_h4:
                                mom.confidence -= 0.15
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM SELL scored -15% — H4 BULLISH")
                                log_gate_block(symbol, "MTF_SNIPER_PENALTY", {"h4_ema55": ema_h4, "price": price_h4})
                            if mom:
                                mom._timeframe = "M15+H4"
                    except Exception:
                        pass  # Fail open if H4 fetch fails

                # ── Daily TF alignment gate ─────────────────────────────────────
                # Source insight: "2-of-3 higher TFs must agree — Weekly/Daily/4H"
                # We enforce H4 + Daily both aligned before momentum entry.
                # Adds ~20% fewer false signals vs H4-only gate per source analysis.
                if mom and getattr(mom, '_timeframe', '') == 'M15+H4':
                    try:
                        candles_d1 = self.connector.get_historical_data(symbol, count=50, granularity="D")
                        closes_d1 = [float(c.get("mid", {}).get("c", 0)) for c in candles_d1]
                        if len(closes_d1) >= 22:
                            ema_d1 = sum(closes_d1[:21]) / 21.0
                            k_d1 = 2.0 / 22.0
                            for px in closes_d1[21:]:
                                ema_d1 = (px - ema_d1) * k_d1 + ema_d1
                            price_d1 = closes_d1[-1]
                            if mom.direction == "BUY" and price_d1 < ema_d1:
                                mom.confidence -= 0.15
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM BUY scored -15% — DAILY BEARISH")
                                log_gate_block(symbol, "MTF_SNIPER_D1_PENALTY", {"d1_ema21": ema_d1, "price": price_d1})
                            elif mom.direction == "SELL" and price_d1 > ema_d1:
                                mom.confidence -= 0.15
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM SELL scored -15% — DAILY BULLISH")
                                log_gate_block(symbol, "MTF_SNIPER_D1_PENALTY", {"d1_ema21": ema_d1, "price": price_d1})
                            if mom:
                                mom._timeframe = "M15+H4+D1"
                    except Exception:
                        pass  # Fail open if D1 fetch fails

                if mom:
                    candidates.append(mom)

                # ─ Pipeline 2: Reversal (Trap + LiqSweep + RSI, 1-of-3) ────────
                rev = run_reversal_pipeline(symbol, candles)
                if rev:
                    candidates.append(rev)

                # ─ Pipeline 3: Mean Reversion (BB + S&D + RSI, 1-of-3) ─────────
                mr = run_meanrev_pipeline(symbol, candles)
                if mr:
                    candidates.append(mr)

                # ─ Pipeline 4: FVG Scalp (FVG + OrderBlock, 1-of-2) ───────────
                sc = run_scalp_pipeline(symbol, candles)
                if sc:
                    candidates.append(sc)

                # ── TRANSCRIPT EDGE: 200 EMA directional filter ───────────
                if _ema200_bias and candidates:
                    for _c in candidates:
                        # Stamp baseline confidence before any penalties (for cap mechanism)
                        if not hasattr(_c, '_pre_penalty_conf'):
                            _c._pre_penalty_conf = _c.confidence
                        if _c.direction != _ema200_bias:
                            _c.confidence -= 0.15
                            print(f"  [EMA200] {symbol} {_c.direction} penalizing confidence -15% vs EMA bias ({_ema200_bias})")
                        else:
                            _c.confidence += 0.05
                            print(f"  [EMA200] {symbol} {_c.direction} boosting confidence +5% with EMA bias")


                # ── TRANSCRIPT EDGE: DXY Correlation Filter ───────────────
                if DXY_GATE_ENABLED and _dxy_bias and candidates and symbol != "USD_CHF":
                    _base, _quote = symbol.split("_") if "_" in symbol else ("", "")
                    for _cs in candidates:
                        _block = False
                        if _dxy_bias == "USD_STRONG":
                            if _base == "USD" and _cs.direction == "SELL":
                                _block = True
                            if _quote == "USD" and _cs.direction == "BUY":
                                _block = True
                        elif _dxy_bias == "USD_WEAK":
                            if _base == "USD" and _cs.direction == "BUY":
                                _block = True
                            if _quote == "USD" and _cs.direction == "SELL":
                                _block = True
                        if _block:
                            _cs.confidence -= 0.10
                            print(f"  [DXY_GATE] {symbol} {_cs.direction} penalizing confidence -10% vs {_dxy_bias}")

                # ── TRANSCRIPT EDGE: Volume Confirmation Gate ─────────────
                if VOLUME_GATE_ENABLED and candidates:
                    try:
                        _vols = [float(c.get("volume", 0)) for c in candles[-20:]]
                        if len(_vols) >= 20 and _vols[-1] > 0:
                            _avg_vol = sum(_vols[:-1]) / len(_vols[:-1])
                            if _avg_vol > 0 and _vols[-1] < _avg_vol * VOLUME_GATE_MULT:
                                for _cs in candidates:
                                    _cs.confidence -= 0.10
                                    print(f"  [VOL_GATE] {symbol} {_cs.direction} penalizing confidence -10% — vol {_vols[-1]:.0f} < {_avg_vol * VOLUME_GATE_MULT:.0f} (1.2x avg)")
                    except Exception:
                        pass

                # ── TRANSCRIPT EDGE: MACD Divergence Filter ────────────────
                if MACD_DIV_ENABLED and candidates and (_macd_div_block_buy or _macd_div_block_sell):
                    for _cs in candidates:
                        if (_macd_div_block_buy and _cs.direction == "BUY") or (_macd_div_block_sell and _cs.direction == "SELL"):
                            _cs.confidence -= 0.15
                            print(f"  [MACD_DIV] {symbol} {_cs.direction} penalizing confidence -15% — divergence detected")

                # ── TRANSCRIPT EDGE: RSI overbought/oversold gate ──────────
                if candidates and (_rsi_block_buy or _rsi_block_sell):
                    for _cs in candidates:
                        if (_rsi_block_buy and _cs.direction == "BUY") or (_rsi_block_sell and _cs.direction == "SELL"):
                            _cs.confidence -= 0.20
                            print(f"  [RSI_GATE] {symbol} {_cs.direction} penalizing confidence -20% — RSI overbought/oversold")

                # ── PENALTY CAP: max -20% total from all stacked gates ─────────
                # Fix: EMA200(-15%) + MTF(-15%) + RSI(-20%) can stack to -70%.
                # At MIN_CONFIDENCE=0.65, a 0.80 signal becomes 0.10 → killed.
                # Cap ensures no signal loses more than 20% from ALL penalty gates.
                # (Claude audit + GPT audit + DeepSeek audit unanimous on this fix.)
                if candidates:
                    for _cs in candidates:
                        _start_conf = getattr(_cs, '_pre_penalty_conf', _cs.confidence)
                        if not hasattr(_cs, '_pre_penalty_conf'):
                            pass  # baseline not tracked — cap applied on next cycle
                        _penalty_applied = _start_conf - _cs.confidence
                        if _penalty_applied > 0.20:
                            _cs.confidence = _start_conf - 0.20  # hard cap at -20%

                # ── TRANSCRIPT EDGE: Session Confidence Boost ──────────────

                # Source: Richie Nasser — London/NY sessions have better fills
                if SESSION_BOOST_ENABLED and candidates:
                    try:
                        from util.time_utils import broker_now_eastern
                        _et_hour = broker_now_eastern().hour
                        _in_london_ny = 8 <= _et_hour <= 16  # 8am-4pm ET overlap
                        if _in_london_ny:
                            for _cs in candidates:
                                if not getattr(_cs, '_session_boosted', False):
                                    _cs.confidence = min(_cs.confidence + 0.03, 0.95)
                                    _cs._session_boosted = True
                    except Exception:
                        pass

                # ── Fib + S&D Confluence Boost ───────────────────────────────
                # Source: "COMPLETE S&D Course" — Fibonacci 61.8% + demand zone
                # overlapping = "maximum confidence, risk more."
                # When momentum pipeline (contains Fib) and mean-reversion
                # pipeline (contains S&D scanner) agree on same direction
                # → +5% confidence on aligned candidates (capped at 0.95).
                if mom and mr and mom.direction == mr.direction:
                    for _c in candidates:
                        if _c.direction == mom.direction and not getattr(_c, '_conf_boosted', False):
                            _c.confidence = min(_c.confidence + 0.05, 0.95)
                            _c._conf_boosted = True
                            print(f"  [CONFLUENCE] {symbol} {_c.direction} +5% conf — Fib+S&D aligned ({_c.confidence:.0%})")

                # ── "Look Left" Trend Exhaustion Filter ───────────────────── (DISABLED by Peer Review)
                # # Source: "15 Best Price Action Strategies" (15 years PA trading)
                # # "Fresh trends = high quality. Late exhausted trends = low quality."
                # # Block signals where price already traveled >65% of TP distance
                # # from the 30-candle swing. Prevents entering at the end of a move.
                # if candidates:
                # _pip_sz  = 0.01 if "JPY" in symbol.upper() else 0.0001
                # _tp_pips = float(os.getenv("RBOT_TP_PIPS", "150"))
                # _live_px = float(candles[-1].get("mid", {}).get("c", 0))
                # _h30     = max(float(c.get("mid", {}).get("h", 0)) for c in candles[-30:])
                # _l30     = min(float(c.get("mid", {}).get("l", 0)) for c in candles[-30:])
                # _thresh  = _tp_pips * 0.65 * _pip_sz
                # _fresh   = []
                # for _sig in candidates:
                # _travel = (_live_px - _l30) if _sig.direction == "BUY" else (_h30 - _live_px)
                # if _travel > _thresh:
                # _pips_t = round(_travel / _pip_sz, 0)
                # print(f"  [EXHAUST] {symbol} {_sig.direction} blocked — {_pips_t:.0f}p traveled (>{_tp_pips*0.65:.0f}p limit)")
                # log_gate_block(symbol, "EXHAUSTION_BLOCK", {
                # "travel_pips": _pips_t,
                # "threshold_pips": _tp_pips * 0.65,
                # "direction": _sig.direction,
                # })
                # else:
                # _fresh.append(_sig)
                # candidates = _fresh  # DISABLED

                # ─ Pick best signal for this pair ─────────────────────────────
                if candidates:
                    candidates = [c for c in candidates if c.confidence >= MIN_CONFIDENCE]
                    if candidates:
                        best = max(candidates, key=lambda s: s.confidence)
                        if not hasattr(best, '_timeframe'):
                            best._timeframe = "M15"
                        if not hasattr(best, '_strategy'):
                            best._strategy = getattr(best, 'signal_type', 'trend')
                        qualified.append(best)

                # ─ Per-pair scan diagnostic line ───────────────────────────────
                def _pfmt(sig, label):
                    if sig is None:
                        return f"{label}=✗"
                    return f"{label}={sig.confidence:.0%}✓"

                tag = ""
                plain = ""
                if candidates:
                    best_strat = getattr(candidates[0] if len(candidates)==1 else max(candidates, key=lambda s: s.confidence), '_strategy', '?')
                    best_sig   = max(candidates, key=lambda s: s.confidence)
                    dir_word   = "SELL" if best_sig.direction == "SELL" else "BUY"
                    strat_desc = {
                        "momentum":      "Strong trend detected",
                        "reversal":      "Reversal pattern spotted",
                        "mean_reversion":"Price stretched, snap-back expected",
                        "scalp":         "Quick entry opportunity found",
                    }.get(best_strat.lower(), "Signal detected")
                    tag   = f"  → QUEUED [{best_strat.upper()}]"
                    plain = f"  💬 {symbol}: {strat_desc} ({best_sig.confidence:.0%} confident) — queuing a {dir_word}"
                print(
                    f"  [SCAN] {symbol:<10}"
                    f"  MOM={_pfmt(mom, 'MOM')[4:]}"
                    f"  REV={_pfmt(rev, 'REV')[4:]}"
                    f"  MR={_pfmt(mr, 'MR')[3:]}"
                    f"  SC={_pfmt(sc, 'SC')[3:]}"
                    f"{tag}"
                )
                if plain:
                    print(plain)

            except Exception:
                continue  # Never let a single pair crash the scan loop


        # ─ Qualified summary ──────────────────────────────────────────────────
        if qualified:
            summary = "  , ".join(
                f"{s.symbol} {s.direction} {getattr(s,'_strategy','?')} {s.confidence:.0%}"
                for s in qualified
            )
            print(f"  📋 Qualified: {len(qualified)} signal(s) — {summary}")
            print(f"  💬 Scan complete: found {len(qualified)} tradeable opportunit{'y' if len(qualified)==1 else 'ies'} this minute. Best signals sorted by confidence — placing now.")

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

        # ── Quiet hours gate: no new opens 10pm–6am EDT (after NY close, before London) ──
        _quiet_enabled = os.getenv("RBOT_QUIET_HOURS_ENABLED", "false").lower() == "true"
        if _quiet_enabled:
            _qh_start = int(os.getenv("RBOT_QUIET_HOURS_START", "22"))
            _qh_end   = int(os.getenv("RBOT_QUIET_HOURS_END",   "6"))
            _now_et   = broker_now_eastern()
            _hour_et  = _now_et.hour
            _in_quiet = (_hour_et >= _qh_start) or (_hour_et < _qh_end)
            if _in_quiet:
                print(f"  🌙 QUIET HOURS ({_now_et.strftime('%I:%M%p ET')}) — no new opens until {_qh_end:02d}:00 ET. Existing trades running.")
                return 0

        # Pre-filter: don't waste cycle_limit slots on pairs already held at broker
        # (dedup guard inside the loop remains as safety net)
        eligible = [s for s in qualified if s.symbol not in broker_symbols and not self._symbol_is_active(s.symbol)]
        for sig in eligible[:cycle_limit]:
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
                continue   # margin tight for this symbol — try next

            # ── Correlation gate: block same-currency directional stacking ──
            if self._would_create_correlated_exposure(symbol, sig.direction):
                print(
                    f"  BLOCKED    {symbol} — CORRELATION_GATE "
                    f"({sig.direction} on same currency already open)"
                )
                print(f"  💬 {symbol}: Skipped — already in a very similar trade. Avoiding duplicate risk.")
                continue

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

            # ── Anti-churn gate: no immediate same-pair re-entry / flip ──────
            churn_reason = self._pair_cooldown_reason(symbol, sig.direction)
            if churn_reason:
                print(f"  BLOCKED    {symbol} — {churn_reason}")
                continue

            sig = self._enforce_rr_buffer(symbol, sig, live_mid)

            # ── Phase 3: ORDER_SUBMIT_ALLOWED ─────────────────────────────
            log_event(ORDER_SUBMIT_ALLOWED, symbol=symbol, venue="tradability_gate", details={
                "symbol":     symbol,
                "direction":  sig.direction,
                "confidence": round(sig.confidence, 4),
            })
            print(f"  → Placing  {symbol} {sig.direction} conf={sig.confidence:.1%}")
            print(f"  💬 {symbol}: Checks passed — submitting {'SELL (going short)' if sig.direction=='SELL' else 'BUY (going long)'} order to broker now.")

            _placed_entry = live_mid
            _ord_type = "MARKET"
            # Fibonacci LIMIT entry DISABLED — adverse-selection fills at 50% retracement
            # were starting trades from worse prices, compounding trailing stop sweeps.

            # ── Phoenix-mode: Strategy-Specific pipeline SL/TP override ─────────
            # Replaces signal's variable SL/TP with exact pip values based on strategy.
            # Reversals/Scalps need tighter exits to lock fast profits before reversion.
            # Momentum uses standard 15/35 to let trend run.
            if self._sl_pips > 0 and _placed_entry:
                _strategy  = getattr(sig, '_strategy',  getattr(sig, 'signal_type', 'trend')).lower()
                
                # Default to env-based majors sizing (15 SL, 35 TP)
                _trade_sl_pips = self._sl_pips
                _trade_tp_pips = self._tp_pips
                
                # All strategies use the same standard SL/TP geometry from .env.
                # Previously: reversal/scalp used 10/25 which was the same width as
                # the ATR trailing stop, giving zero room for the trade to breathe.
                _trade_sl_pips = self._sl_pips
                _trade_tp_pips = self._tp_pips
                print(f"  [STRATEGY EXIT] {_strategy} detected — using standard {_trade_sl_pips}/{_trade_tp_pips} exits")

                _pip = 0.01 if "JPY" in symbol.upper() else 0.0001
                _sl_dist = _trade_sl_pips * _pip
                _tp_dist = _trade_tp_pips * _pip
                if sig.direction == "BUY":
                    sig.sl = round(_placed_entry - _sl_dist, 5)
                    sig.tp = round(_placed_entry + _tp_dist, 5)
                else:
                    sig.sl = round(_placed_entry + _sl_dist, 5)
                    sig.tp = round(_placed_entry - _tp_dist, 5)

            # ── Phase 4: OCO payload validation ───────────────────────────
            units = self._compute_units(symbol, sig, nav)
            units = self._apply_min_notional_floor(symbol, units, _placed_entry)
            oco_check = validate_oco_payload(
                symbol=symbol,
                direction=sig.direction,
                entry_price=_placed_entry,
                stop_loss=sig.sl,
                take_profit=sig.tp,
                units=units,
            )
            if not oco_check["valid"]:
                log_gate_block(symbol, OCO_VALIDATION_BLOCK, oco_check)
                print(f"  BLOCKED    {symbol} — OCO_VALIDATION_BLOCK {oco_check['reason']}")
                set_cooldown(symbol)  # avoid retrying a structurally failing pair every cycle
                continue

            # ── Phase 5: place OCO order ───────────────────────────────────
            try:
                # OANDA hard minimum is 5 pips (0.0005 non-JPY, 0.05 JPY).
                # Setting floor to 10 pips and enforcing 2x buffer (20-pip worst-case)
                # prevents TRAILING_STOP_LOSS_ON_FILL_PRICE_DISTANCE_MINIMUM_NOT_MET
                # due to floating-point imprecision at or near the broker floor.
                pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
                min_ts_dist = 10.0 * pip_size          # 0.001 non-JPY, 0.10 JPY
                raw_sl_dist = abs(_placed_entry - sig.sl)
                # ── Adaptive trailing stop distance ─────────────────────────────
                # RBOT_TS_PIPS=N  → fixed N pips (override, current: 50)
                # RBOT_TS_PIPS=0  → ATR mode: RBOT_TS_ATR_MULT × ATR(14)
                # Source insight: ATR trail adapts to pair volatility vs fixed pips.
                # JPY pairs naturally get wider trail; rangy pairs get tighter.
                _ts_pips     = float(str(os.getenv("RBOT_TS_PIPS",     "0")).split("#", 1)[0].strip())
                _ts_atr_mult = float(str(os.getenv("RBOT_TS_ATR_MULT", "0")).split("#", 1)[0].strip())
                if _ts_pips > 0:
                    ts_dist = max(_ts_pips * pip_size, min_ts_dist * 2)
                elif _ts_atr_mult > 0:
                    # ATR(14) adaptive trail — scales to current pair volatility
                    try:
                        _atr_c = self.connector.get_historical_data(symbol, count=20, granularity="M15")
                        _ch = [float(c.get("mid", {}).get("h", 0)) for c in _atr_c]
                        _cl = [float(c.get("mid", {}).get("l", 0)) for c in _atr_c]
                        _cc = [float(c.get("mid", {}).get("c", 0)) for c in _atr_c]
                        if len(_ch) >= 15:
                            _trs = [max(_ch[i]-_cl[i], abs(_ch[i]-_cc[i-1]), abs(_cl[i]-_cc[i-1]))
                                    for i in range(1, len(_ch))]
                            _atr = sum(_trs[-14:]) / 14.0
                            ts_dist = max(_atr * _ts_atr_mult, min_ts_dist * 2)
                            print(f"  [ATR CALC] {symbol} 14-period M15 ATR calculated successfully.")
                        else:
                            ts_dist = max(raw_sl_dist, min_ts_dist * 2)
                    except Exception:
                        ts_dist = max(raw_sl_dist, min_ts_dist * 2)
                else:
                    ts_dist = max(raw_sl_dist, min_ts_dist * 2)

                result = self.connector.place_oco_order(
                    instrument=symbol,
                    entry_price=_placed_entry,
                    stop_loss=sig.sl,
                    take_profit=sig.tp,
                    units=units,
                    order_type=_ord_type,
                    trailing_stop_distance=ts_dist,
                )

                # ── Phase 6: verify broker response ───────────────────────
                confirm = check_submit_response(result, symbol)
                
                # Fallback: If broker said success=True but parsing failed to find trade_id
                if not confirm.get("confirmed") and result.get("success") and (result.get("live_api") or result.get("visible_in_oanda", True)):
                    import time
                    for _ in range(4):
                        time.sleep(1.0)  # Wait for LIMIT order cross
                        try:
                            trades = self.connector.get_trades()
                            matching = [t for t in trades if t.get("instrument") == symbol]
                            if matching:
                                newest = max(matching, key=lambda x: int(x.get("id", 0)))
                                confirm["confirmed"] = True
                                confirm["trade_id"] = str(newest.get("id"))
                                break
                        except Exception:
                            pass

                if not confirm.get("confirmed"):
                    log_event(ORDER_SUBMIT_BLOCK, symbol=symbol, venue="oanda", details={
                        "symbol": symbol, "error": confirm.get("error", "unknown")
                    })
                    print(f"  ✗ REJECTED {symbol} — {confirm.get('error', 'unknown')}")
                    continue

                trade_id = confirm["trade_id"]
                set_cooldown(symbol)
                placed_this_cycle.add(symbol.upper())
                _strategy  = getattr(sig, '_strategy',  getattr(sig, 'signal_type', 'trend'))
                _timeframe = getattr(sig, '_timeframe', 'M15')
                _detectors = sig.detectors_fired if hasattr(sig, 'detectors_fired') else []
                _det_str   = '+'.join(_detectors[:3]) if _detectors else _strategy
                self.active_positions[trade_id] = {
                    "symbol":      symbol,
                    "direction":   sig.direction,
                    "stop_loss":   sig.sl,
                    "take_profit": sig.tp,
                    "confidence":  sig.confidence,
                    "session":     sig.session,
                    "opened_at":   datetime.now(timezone.utc).isoformat(),
                    "stale_cycles": 0,
                    "strategy":    _strategy,
                    "timeframe":   _timeframe,
                    "detectors":   _detectors,
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
                self._mark_pair_trade(symbol, sig.direction)
                print(f"  ✓ OPENED   {symbol} {sig.direction} [{_strategy.upper()} {_timeframe}] {_det_str}  id={trade_id}")
                _rr = getattr(sig, 'rr', 0)
                _sl_pips = round(abs(sig.sl - live_mid) / (0.01 if 'JPY' in symbol else 0.0001))
                _tp_pips = round(abs(sig.tp - live_mid) / (0.01 if 'JPY' in symbol else 0.0001))
                print(f"  💬 ✅ Trade open! {symbol} {'selling' if sig.direction=='SELL' else 'buying'} {abs(units):,} units. "
                      f"Stop-loss in {_sl_pips} pips, take-profit in {_tp_pips} pips. "
                      f"Trailing stop active. OCO protection set.")

                # ── Hedge counter-trade (Phoenix QuantHedgeEngine) ──────────────────
                if self._hedge_engine and getattr(self, "is_chop_mode_active", False) and live_mid:
                    try:
                        _hedge = self._hedge_engine.execute_hedge(
                            primary_symbol=symbol,
                            primary_side=sig.direction,
                            position_size=units,
                            entry_price=live_mid,
                        )
                        if _hedge:
                            _h_prices = self.connector.get_live_prices([_hedge.symbol])
                            _h_mid = (_h_prices.get(_hedge.symbol) or {}).get("mid", 0.0)
                            if _h_mid:
                                _h_pip = 0.01 if "JPY" in _hedge.symbol.upper() else 0.0001
                                _h_sl_d = (self._sl_pips or 10) * _h_pip
                                _h_tp_d = (self._tp_pips or 32) * _h_pip
                                if _hedge.side == "BUY":
                                    _h_sl = round(_h_mid - _h_sl_d, 5)
                                    _h_tp = round(_h_mid + _h_tp_d, 5)
                                else:
                                    _h_sl = round(_h_mid + _h_sl_d, 5)
                                    _h_tp = round(_h_mid - _h_tp_d, 5)
                                _h_result = self.connector.place_oco_order(
                                    instrument=_hedge.symbol,
                                    entry_price=_h_mid,
                                    stop_loss=_h_sl,
                                    take_profit=_h_tp,
                                    units=int(_hedge.size),
                                    order_type="MARKET",
                                    trailing_stop_distance=_h_sl_d * 2,
                                    is_hedge=True,
                                )
                                if _h_result.get("success"):
                                    print(f"  🞡  HEDGE    {_hedge.symbol} {_hedge.side} {int(_hedge.size)}u @ {_h_mid} (counter to {symbol})")
                                else:
                                    print(f"  ⚠️  HEDGE    {_hedge.symbol} rejected: {_h_result.get('error', 'unknown')}")
                    except Exception as _he:
                        print(f"  ⚠️  HEDGE    error for {symbol}: {_he}")

            except Exception as e:
                log_event(TRADE_OPEN_FAILED, symbol=symbol, venue="oanda", details={
                    "symbol": symbol, "error": str(e)
                })
                print(f"  ✗ ERROR    {symbol} — {e}")

        # ── Gap 1 fix: CapitalRouter reallocation ──────────────────────────────
        # Runs ONLY when all position slots are full AND a candidate is genuinely
        # stronger than the weakest open position (controlled by UPGRADE_THRESHOLD).
        # One reallocation max per cycle. Passes through full OCO validation gate.
        if (
            self._router
            and qualified
            and not ATTACH_ONLY
            and not DISABLE_NEW_ENTRIES
        ):
            self._router.reset_cycle()
            try:
                acct_info = {"NAV": nav, "balance": nav}
                realloc = self._router.evaluate(
                    open_positions=self.active_positions,
                    candidates=qualified,
                    account_info=acct_info,
                )
            except Exception as _re:
                realloc = None
                print(f"  [ROUTER] evaluate() error: {_re}")

            if realloc:
                _rs = realloc.close_symbol
                _rt = realloc.close_trade_id
                log_event(CAPITAL_REALLOC_DECIDED, symbol=_rs, venue="capital_router",
                          details=realloc.as_dict())
                print(
                    f"  [ROUTER] REALLOC  close {_rs}({_rt[:6]}) "
                    f"→ open {realloc.open_symbol} {realloc.open_direction} "
                    f"({realloc.improvement_pct:.1%} improvement)"
                )

                # Step A: close the weak position
                try:
                    self.connector.close_trade(_rt)
                    self.active_positions.pop(_rt, None)
                except Exception as _ce:
                    log_event(CAPITAL_REALLOC_FAILED, symbol=_rs, venue="capital_router",
                              details={"step": "close", "error": str(_ce)})
                    print(f"  [ROUTER] ⚠️  Close failed for {_rs}: {_ce}")
                    realloc = None  # abort — do not open without closing

            if realloc:
                # Step B: find the matching signal from this cycle's qualified list
                _open_sig = next(
                    (s for s in qualified
                     if s.symbol == realloc.open_symbol
                     and s.direction == realloc.open_direction),
                    None,
                )
                if _open_sig and _open_sig.symbol not in placed_this_cycle:
                    _open_sym = _open_sig.symbol
                    # Re-run margin gate with fresh account state
                    if free_margin_pct < MIN_FREE_MARGIN_PCT:
                        log_event(CAPITAL_REALLOC_FAILED, symbol=_open_sym,
                                  venue="capital_router",
                                  details={"step": "open", "reason": "margin_gate"})
                        print(f"  [ROUTER] ⚠️  Realloc blocked — margin gate")
                    else:
                        # OCO validation — fresh gate on the replacement symbol
                        _realloc_gate = check_broker_tradability(
                            self.connector, _open_sym,
                            placed_this_cycle=placed_this_cycle,
                        )
                        _live_mid = (_realloc_gate.get("live_price") or {}).get("mid") or 0.0
                        _oco_check = validate_oco_payload(
                            symbol=_open_sym,
                            direction=_open_sig.direction,
                            entry_price=_live_mid,
                            stop_loss=_open_sig.sl,
                            take_profit=_open_sig.tp,
                            units=self._compute_units(_open_sym, _open_sig, nav),
                        )
                        if not _realloc_gate["allowed"] or not _oco_check["valid"]:
                            _reason = (
                                _realloc_gate.get("event", "gate_block")
                                if not _realloc_gate["allowed"]
                                else _oco_check.get("reason", "oco_invalid")
                            )
                            log_event(CAPITAL_REALLOC_FAILED, symbol=_open_sym,
                                      venue="capital_router",
                                      details={"step": "open", "reason": _reason})
                            print(f"  [ROUTER] ⚠️  Realloc open blocked: {_reason}")
                        else:
                            try:
                                _open_sig = self._enforce_rr_buffer(_open_sym, _open_sig, _live_mid)
                                _units = self._compute_units(_open_sym, _open_sig, nav)
                                _units = self._apply_min_notional_floor(_open_sym, _units, _live_mid)
                                _pip   = 0.01 if "JPY" in _open_sym.upper() else 0.0001
                                _min_ts = 10.0 * _pip
                                _ts_dist = max(abs(_live_mid - _open_sig.sl), _min_ts * 2)
                                _res = self.connector.place_oco_order(
                                    instrument=_open_sym,
                                    entry_price=_live_mid,
                                    stop_loss=_open_sig.sl,
                                    take_profit=_open_sig.tp,
                                    units=_units,
                                    order_type="MARKET",
                                    trailing_stop_distance=_ts_dist,
                                )
                                _conf = check_submit_response(_res, _open_sym)
                                if _conf.get("confirmed"):
                                    _tid = _conf["trade_id"]
                                    set_cooldown(_open_sym)
                                    placed_this_cycle.add(_open_sym.upper())
                                    self.active_positions[_tid] = {
                                        "symbol":      _open_sym,
                                        "direction":   _open_sig.direction,
                                        "stop_loss":   _open_sig.sl,
                                        "take_profit": _open_sig.tp,
                                        "confidence":  _open_sig.confidence,
                                        "session":     _open_sig.session,
                                        "opened_at":   datetime.now(timezone.utc).isoformat(),
                                        "stale_cycles": 0,
                                    }
                                    log_trade_opened(
                                        symbol=_open_sym, direction=_open_sig.direction,
                                        trade_id=_tid,
                                        entry=_res.get("entry_price", _live_mid),
                                        stop_loss=_open_sig.sl, take_profit=_open_sig.tp,
                                        size=_units, confidence=_open_sig.confidence,
                                        votes=_open_sig.votes,
                                        detectors=_open_sig.detectors_fired,
                                        session=_open_sig.session,
                                    )
                                    placed_count += 1
                                    self._mark_pair_trade(_open_sym, _open_sig.direction)
                                    print(f"  [ROUTER] ✓ REALLOC OPENED {_open_sym} id={_tid}")
                                else:
                                    log_event(CAPITAL_REALLOC_FAILED, symbol=_open_sym,
                                              venue="capital_router",
                                              details={"step": "open",
                                                       "error": _conf.get("error", "unknown")})
                                    print(f"  [ROUTER] ✗ Realloc open rejected: {_conf.get('error')}")
                            except Exception as _oe:
                                log_event(CAPITAL_REALLOC_FAILED, symbol=_open_sym,
                                          venue="capital_router",
                                          details={"step": "open", "error": str(_oe)})
                                print(f"  [ROUTER] ✗ Realloc open error: {_oe}")

        return placed_count

    def _pair_cooldown_reason(self, symbol: str, direction: str) -> Optional[str]:
        now = datetime.now(timezone.utc).timestamp()
        last_ts = self._pair_last_trade_ts.get(symbol)
        last_side = self._pair_last_side.get(symbol)
        if last_ts is not None:
            elapsed = now - last_ts
            cooldown = PAIR_REENTRY_COOLDOWN_MINUTES * 60
            if elapsed < cooldown:
                remain_sec = cooldown - elapsed
                remain_min = remain_sec / 60.0
                pct_hour = (remain_min / 60.0) * 100.0
                return f"PAIR_COOLDOWN_ACTIVE last_side={last_side} wait={remain_min:.1f}m ({pct_hour:.0f}% of hour)"

        for pos in self.active_positions.values():
            if str(pos.get("symbol", "")).upper() == symbol.upper():
                open_side = str(pos.get("direction", ""))
                return f"PAIR_ALREADY_OPEN open_side={open_side}"

        try:
            broker_trades = self.connector.get_trades() or []
            for t in broker_trades:
                inst = str(t.get("instrument") or "")
                units = float(t.get("currentUnits") or t.get("initialUnits") or 0)
                side = "BUY" if units > 0 else "SELL" if units < 0 else ""
                if inst.upper() == symbol.upper() and side:
                    return f"PAIR_ALREADY_OPEN open_side={side}"
        except Exception:
            pass

        return None

    def _load_cooldown_state(self) -> None:
        """Restore pair cooldown timestamps from disk (survives restarts)."""
        import json
        try:
            if self._cooldown_persist_path.exists():
                raw = json.loads(self._cooldown_persist_path.read_text())
                ts_map  = raw.get("ts", {})
                side_map = raw.get("side", {})
                self._pair_last_trade_ts.update({k: float(v) for k, v in ts_map.items()})
                self._pair_last_side.update(side_map)
                print(f"  ✅ Cooldown state restored ({len(ts_map)} pairs) from {self._cooldown_persist_path}")
        except Exception as _e:
            print(f"  ⚠️  Cooldown restore failed (starting fresh): {_e}")

    def _save_cooldown_state(self) -> None:
        """Persist pair cooldown timestamps to disk so restarts honour existing cooldowns."""
        import json
        try:
            payload = {
                "ts":   {k: v for k, v in self._pair_last_trade_ts.items()},
                "side": {k: v for k, v in self._pair_last_side.items()},
            }
            self._cooldown_persist_path.write_text(json.dumps(payload, indent=2))
        except Exception as _e:
            print(f"  ⚠️  Cooldown save failed: {_e}")

    def _mark_pair_trade(self, symbol: str, direction: str) -> None:
        self._pair_last_trade_ts[symbol] = datetime.now(timezone.utc).timestamp()
        self._pair_last_side[symbol] = direction
        self._save_cooldown_state()  # persist so restarts honour this cooldown

    def _would_create_correlated_exposure(self, symbol: str, direction: str) -> bool:
        """
        Return True if opening symbol+direction would double up on the same base or
        quote currency in the same direction as an existing open position.
        Examples blocked: EUR/CAD LONG when EUR/JPY LONG open;
                          USD/CAD SELL when USD/CHF SELL open.
        """
        if "_" not in symbol:
            return False
        new_base = symbol.split("_")[0]
        new_quote = symbol.split("_")[1]
        for pos in self.active_positions.values():
            s = pos.get("symbol", "")
            d = pos.get("direction", "")
            if "_" not in s or d != direction:
                continue
            b = s.split("_")[0]
            q = s.split("_")[1]
            if b == new_base or q == new_quote:
                return True
        return False

    def _apply_min_notional_floor(self, symbol: str, units: int, live_mid: float) -> int:
        """
        Enforce buffered Charter notional using broker USD-notional math.
        Fixes USD-base and cross-pair sizing where abs(units)*price is wrong.
        """
        try:
            px = float(live_mid or 0.0)
        except Exception:
            px = 0.0
        if px <= 0:
            return units

        side = 1 if int(units) >= 0 else -1
        abs_units = max(1, abs(int(units)))
        target = float(CHARTER_TARGET_NOTIONAL_USD)

        def usd_notional_for(test_units: int) -> float:
            try:
                if hasattr(self.connector, "get_usd_notional"):
                    return float(self.connector.get_usd_notional(test_units, symbol, px))
            except Exception:
                pass
            try:
                from brokers.oanda_connector import get_usd_notional as _g
                return float(_g(test_units, symbol, px))
            except Exception:
                pass

            base = (symbol or "").split("_")[0].upper()
            if base == "USD":
                return float(abs(test_units))
            return float(abs(test_units) * px)

        cur = abs_units
        cur_notional = usd_notional_for(cur)

        if cur_notional >= target:
            return cur * side

        if cur_notional > 0:
            cur = max(cur, int((cur * target / cur_notional) + 0.9999))
        else:
            cur = max(cur, int(target + 0.9999))

        step = max(100, int(cur * 0.02))
        for _ in range(40):
            cur_notional = usd_notional_for(cur)
            if cur_notional >= target:
                break
            cur += step

        while usd_notional_for(cur) < target:
            cur += 1

        return cur * side

    def _enforce_rr_buffer(self, symbol: str, sig, entry_price: float):
        """
        Push TP slightly farther so broker-side / Charter-side rounding cannot reject
        an exactly-3.20 setup. Keeps SL unchanged, only widens TP modestly.
        """
        try:
            entry = float(entry_price or 0.0)
            sl = float(sig.sl or 0.0)
            tp = float(sig.tp or 0.0)
        except Exception:
            return sig
        if entry <= 0 or sl <= 0 or tp <= 0:
            return sig

        side = str(sig.direction).upper()
        if side == "BUY":
            risk = entry - sl
            if risk > 0:
                min_tp = entry + (risk * CHARTER_MIN_RR)
                if tp < min_tp:
                    sig.tp = round(min_tp, 5)
        elif side == "SELL":
            risk = sl - entry
            if risk > 0:
                min_tp = entry - (risk * CHARTER_MIN_RR)
                if tp > min_tp:
                    sig.tp = round(min_tp, 5)
        return sig


    def _compute_units(self, symbol: str, sig: AggregatedSignal, nav: float = 0.0) -> int:
        """
        Sizing stack — Phoenix Quality-Picker (ported from oanda_trading_engine.py:1152):
        1) Power-curve confidence scaling: low-conf → Charter floor, high-conf → max notional
        2) Home-run bonus: conf >= 0.88 → +25% notional
        3) Watermark-based compounding on top
        4) Chop mode override (fixed units during chop session)
        5) Charter notional floor enforced last

        Replaces flat RBOT_BASE_UNITS + fake Kelly (confidence ≠ realized win-rate).
        Power-curve keeps 'meh' trades small and rewards genuine conviction.
        """
        import math

        # ── Chop mode: bypass quality-picker, use fixed conservative units ──
        if getattr(self, "is_chop_mode_active", False):
            chop_units = getattr(self, "_chop_units", int(os.getenv("RBOT_BASE_UNITS", "14000")))
            signed = chop_units if sig.direction == "BUY" else -chop_units
            live_prices = self.connector.get_live_prices([symbol]) or {}
            live_mid = float((live_prices.get(symbol) or {}).get("mid", 0.0) or 0.0)
            return int(self._apply_min_notional_floor(symbol, signed, live_mid))

        # ── Power-curve confidence sizing (Phoenix quality-picker) ────────
        conf_floor = float(os.getenv("RBOT_MIN_SIGNAL_CONFIDENCE", "0.75"))
        conf_cap   = 0.94
        min_notional = float(os.getenv("RBOT_CHARTER_MIN_NOTIONAL_USD", "15000"))
        max_notional = float(os.getenv("RBOT_MAX_NOTIONAL_USD", "30000"))
        if max_notional < min_notional:
            max_notional = min_notional

        conf = getattr(sig, 'confidence', conf_floor)
        conf = max(conf_floor, min(float(conf), conf_cap))

        scale_raw = (conf - conf_floor) / max(conf_cap - conf_floor, 1e-9)
        scale     = max(0.0, min(scale_raw, 1.0)) ** 1.8   # power curve: exponential conviction

        target_notional = min_notional + (max_notional - min_notional) * scale
        
        # ── Golden Turnaround Max Sizing ──────────────────────────────────
        if getattr(sig, "meta", {}).get("is_golden_turnaround"):
            golden_mult = float(os.getenv("OANDA_GOLDEN_MULTIPLIER", "2.0"))
            target_notional *= golden_mult
            print(f"  [GOLDEN TURNAROUND] 🔥 Sizing override applied: {golden_mult}x multiplier! Notional: ${target_notional:,.0f}")

        # Home-run bonus: conf >= 0.88 → +25% notional (capped at 1.5× max)
        if conf >= 0.88:
            target_notional *= 1.25
            target_notional  = min(target_notional, max_notional * 1.5)

        # ── Watermark compounding applied ON TOP of quality-picker ────────
        growth_exponent      = float(os.getenv("RBOT_COMPOUND_GROWTH_EXPONENT", "1.15"))
        drawdown_floor_ratio = float(os.getenv("RBOT_COMPOUND_DRAWDOWN_FLOOR_RATIO", "1.00"))
        max_growth_multiple  = float(os.getenv("RBOT_COMPOUND_MAX_GROWTH_MULTIPLE", "3.00"))
        if nav > 0 and self._initial_nav > 0:
            compounded_base = compute_watermark_compounded_units(
                base_units=int(min_notional),   # treat notional as pseudo-units for ratio math
                current_nav=nav,
                initial_nav=self._initial_nav,
                watermark_nav=(self._watermark_nav or self._initial_nav),
                growth_exponent=growth_exponent,
                drawdown_floor_ratio=drawdown_floor_ratio,
                max_growth_multiple=max_growth_multiple,
            )
            compound_ratio  = compounded_base / max(int(min_notional), 1)
            target_notional = min(target_notional * compound_ratio, max_notional * 2.0)

        # ── Convert notional → units for this symbol ──────────────────────
        live_prices = self.connector.get_live_prices([symbol]) or {}
        live_mid    = float((live_prices.get(symbol) or {}).get("mid", 0.0) or 0.0)

        parts = symbol.upper().split("_")
        base  = parts[0] if len(parts) == 2 else ""
        quote = parts[1] if len(parts) == 2 else ""

        entry = live_mid if live_mid > 0 else 1.0
        if quote == "USD":                    # EUR_USD, GBP_USD, AUD_USD
            raw_units = math.ceil(target_notional / entry)
        elif base == "USD":                   # USD_JPY, USD_CAD, USD_CHF
            raw_units = math.ceil(target_notional)
        else:                                 # crosses — conservative proxy
            raw_units = math.ceil(target_notional * 0.9)

        scaled = math.ceil(raw_units / 100) * 100  # round to nearest 100

        print(
            f"  [QUALITY-PICK] {symbol} conf={conf:.1%}  notional=${target_notional:,.0f}"
            f"  units={scaled}"
        )

        signed_units = scaled if sig.direction == "BUY" else -scaled
        signed_units = int(self._apply_min_notional_floor(symbol, signed_units, live_mid))
        return signed_units

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self.is_running = True
        self.print_startup_banner()

        # Activate TradeManager after is_running = True
        self.manager.activate()

        # ── Compound NAV persistence — survives network drops + restarts ────────
        # State file: logs/compound_state.json
        # initial_nav is NEVER reset on restart (preserves compound baseline)
        # watermark_nav restores to the last recorded all-time-high
        self._compound_state_path = str(Path(__file__).resolve().parent.parent / "logs" / "compound_state.json")
        try:
            _acct = self.connector.get_account_info()
            _live_nav = _acct.balance + _acct.unrealized_pl
            if os.path.exists(self._compound_state_path):
                with open(self._compound_state_path) as _csf:
                    _cs = json.load(_csf)
                self._initial_nav   = float(_cs.get("initial_nav",   _live_nav))
                self._watermark_nav = max(float(_cs.get("watermark_nav", _live_nav)), _live_nav)
                print(f"  ✅ CapitalRouter RESTORED  initial=${self._initial_nav:,.2f}  watermark=${self._watermark_nav:,.2f}  live=${_live_nav:,.2f}")
            else:
                self._initial_nav   = _live_nav
                self._watermark_nav = _live_nav
                os.makedirs(os.path.dirname(self._compound_state_path), exist_ok=True)
                with open(self._compound_state_path, "w") as _csf:
                    json.dump({"initial_nav": self._initial_nav, "watermark_nav": self._watermark_nav}, _csf)
                print(f"  ✅ CapitalRouter NEW       initial=${self._initial_nav:,.2f}  watermark=${self._watermark_nav:,.2f}")
            self._router = CapitalRouter(self.connector, initial_nav=self._initial_nav)
        except Exception as _nav_err:
            print(f"  ⚠️  CapitalRouter NAV init failed: {_nav_err} — compound scaling disabled")

        while self.is_running:
            try:
                # ── Pre-market session scan ────────────────────────────────────
                if self.pre_scanner.should_run_now():
                    self._session_playbook = self.pre_scanner.run_scan()
                    if self._session_playbook:
                        active_pb = [e for e in self._session_playbook if not e.vetoed]
                        print(f"  [PLAYBOOK] {len(active_pb)} active setups for upcoming session")

                try:
                    _acct_live = self.connector.get_account_info()
                    _nav_live = _acct_live.balance + _acct_live.unrealized_pl
                    if _nav_live > self._watermark_nav:
                        self._watermark_nav = _nav_live
                        # Persist new high to disk — survives network drop + restart
                        try:
                            with open(self._compound_state_path, "w") as _csf:
                                json.dump({"initial_nav": self._initial_nav, "watermark_nav": self._watermark_nav}, _csf)
                        except Exception:
                            pass
                except Exception:
                    pass

                placed = await self._run_scan_cycle()
                # ── Trail management: runs every cycle regardless of ATTACH_ONLY ─
                await self.manager.tick(engine_positions=self.active_positions)

                open_now = len(self.active_positions)
                
                # Retrieve current loop configs, honoring Chop Mode
                _max_p = getattr(self, "current_max_positions", MAX_POSITIONS)
                _mode_str = "[CHOP MODE 14k]" if getattr(self, "is_chop_mode_active", False) else "[SNIPER MODE 50k]"

                if open_now >= _max_p:
                    print(f"  {_mode_str} Positions full ({open_now}/{_max_p}) — waiting {SCAN_SLOW_SECONDS}s")
                    await asyncio.sleep(SCAN_SLOW_SECONDS)
                else:
                    print(f"  {_mode_str} Slots open ({open_now}/{_max_p}) — rescanning in {SCAN_FAST_SECONDS}s")
                    await asyncio.sleep(SCAN_FAST_SECONDS)

            except KeyboardInterrupt:
                print("\n  Stopping engine…")
                self.is_running = False
                self.manager.deactivate()
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
