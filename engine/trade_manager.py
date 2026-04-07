"""
engine/trade_manager.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Manages existing OANDA broker trades:
  - Position sync (new/closed detection)
  - Green-lock SL enforcement (profit protection)
  - Hard dollar stop ($30 default, env-configurable)
  - RBZ Two-Step SL lock + aggressive trailing (via trail_logic)
  - Stagnation kill-switch (tighten SL after N idle cycles)
  - SL rejection backoff (60s cooldown per price)
  - Heartbeat tracking

Logs to narration.jsonl via narration_logger.
Never opens new trades — that is the engine's job.
"""

import os
import time
import json
from pathlib import Path
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .trail_logic import policy_for, apply_tight_sl
from util.narration_logger import (
    log_event, log_narration,
    POSITION_SYNCED, POSITION_CLOSED,
    TRAIL_CANDIDATE, TRAIL_SUBMIT_ALLOWED,
    TRAIL_SL_SET, TRAIL_SL_REJECTED, TRAIL_NO_ACTION,
    BREAK_EVEN_SET,
    MANAGER_CYCLE_STARTED,
    TRADE_MANAGER_ACTIVATED, TRADE_MANAGER_DEACTIVATED,
    GREEN_LOCK_ENFORCED, HARD_DOLLAR_STOP, TRAIL_TIGHT_SET,
)


class TradeManager:
    """
    Manages existing trades: sync, trail, green-lock, hard-dollar-stop, stagnation.
    Initialized once by TradeEngine. Called every cycle via tick().
    """

    def __init__(self, broker) -> None:
        self.broker = broker
        self._managed: Dict[str, dict] = {}          # trade_id → last-known state
        self._sl_rejection_backoff: Dict[str, float] = {}  # "trade_id:sl_price" → timestamp

        # Heartbeat tracking
        self.trade_manager_active = False
        self.last_heartbeat: Optional[datetime] = None

        # Hard dollar stop config (env-overridable)
        self._hard_stop_usd = abs(float(os.getenv("RBOT_MAX_LOSS_USD_PER_TRADE", "45")))

        # Green-lock config
        self._green_lock_pips = float(os.getenv("RBOT_GREEN_LOCK_PIPS", "5.0"))
        self._green_lock_min_profit_pips = float(os.getenv("RBOT_GREEN_LOCK_MIN_PROFIT_PIPS", "5.0"))
        self._profit_target_pct = float(os.getenv("RBOT_PROFIT_TARGET_PCT", "75"))
        if self._profit_target_pct > 1.0:
            self._profit_target_pct = self._profit_target_pct / 100.0
        # Stagnation config
        self._stagnation_cycles = int(os.getenv("RBOT_STAGNATION_CYCLES", "5"))
        self._stagnation_pip_threshold = float(os.getenv("RBOT_STAGNATION_PIP_THRESHOLD", "2.0"))
        self._stagnation_tighten_pips = float(os.getenv("RBOT_STAGNATION_TIGHTEN_PIPS", "2.0"))

        # Profit target close config (full close at % of TP distance reached)
        # Safety net against reversals: exit cleanly before giving profits back

        # ── Transcript Edge: Per-pair win rate tracker ─────────────────────
        self._pair_stats_path = str(Path(__file__).resolve().parent.parent / "logs" / "pair_stats.json")
        self._pair_stats = self._load_pair_stats()

        # ── Transcript Edge: Per-pair win rate tracker ─────────────────────
        # Source: "Profitable Forex Trader" — track which pairs your algo works on
        self._pair_stats_path = str(Path(__file__).resolve().parent.parent / "logs" / "pair_stats.json")
        self._pair_stats = self._load_pair_stats()
    def _load_pair_stats(self) -> dict:
        """Load per-pair win/loss stats from disk."""
        try:
            if os.path.exists(self._pair_stats_path):
                with open(self._pair_stats_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_pair_stats(self) -> None:
        """Persist per-pair stats to disk."""
        try:
            with open(self._pair_stats_path, "w") as f:
                json.dump(self._pair_stats, f, indent=2)
        except Exception:
            pass

    def _record_trade_result(self, instrument: str, pnl: float) -> None:
        """Record a win or loss for a pair. Print rolling stats."""
        if instrument not in self._pair_stats:
            self._pair_stats[instrument] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        s = self._pair_stats[instrument]
        if pnl >= 0:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_pnl"] = round(s["total_pnl"] + pnl, 2)
        total = s["wins"] + s["losses"]
        win_rate = (s["wins"] / total * 100) if total > 0 else 0
        print(f"  [PAIR_STATS] {instrument}  {'WIN' if pnl >= 0 else 'LOSS'} ${pnl:+.2f}"
              f"  | W/L: {s['wins']}/{s['losses']} ({win_rate:.0f}%)  Net: ${s['total_pnl']:+.2f}")
        self._save_pair_stats()

    def _load_pair_stats(self) -> dict:
        try:
            if os.path.exists(self._pair_stats_path):
                with open(self._pair_stats_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_pair_stats(self) -> None:
        try:
            with open(self._pair_stats_path, "w") as f:
                json.dump(self._pair_stats, f, indent=2)
        except Exception:
            pass

    def _record_trade_result(self, instrument: str, pnl: float) -> None:
        if instrument not in self._pair_stats:
            self._pair_stats[instrument] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        s = self._pair_stats[instrument]
        if pnl >= 0:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_pnl"] = round(s["total_pnl"] + pnl, 2)
        total = s["wins"] + s["losses"]
        win_rate = (s["wins"] / total * 100) if total > 0 else 0
        print(f"  [PAIR_STATS] {instrument}  {'WIN' if pnl >= 0 else 'LOSS'} {pnl:+.1f}p"
              f"  | W/L: {s['wins']}/{s['losses']} ({win_rate:.0f}%)  Net: {s['total_pnl']:+.1f}p")
        self._save_pair_stats()

    def activate(self) -> None:
        """Mark manager as active. Called once after engine is_running = True."""
        self.trade_manager_active = True
        self.last_heartbeat = datetime.now(timezone.utc)
        print("  ✅ TRADE MANAGER ACTIVATED")
        log_event(TRADE_MANAGER_ACTIVATED, symbol="SYSTEM", venue="trade_manager", details={
            "status": "ACTIVE",
            "timestamp": self.last_heartbeat.isoformat(),
            "hard_stop_usd": self._hard_stop_usd,
        })

    def deactivate(self) -> None:
        """Mark manager as inactive."""
        self.trade_manager_active = False
        print("  ⚠️  TRADE MANAGER DEACTIVATED")
        log_event(TRADE_MANAGER_DEACTIVATED, symbol="SYSTEM", venue="trade_manager", details={
            "status": "INACTIVE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def tick(self, engine_positions: dict = None) -> None:
        """Called from engine run() loop every scan cycle."""
        self.manage_open_trades(engine_positions=engine_positions)
        self.last_heartbeat = datetime.now(timezone.utc)

    # ── Main management cycle ─────────────────────────────────────────────────

    def manage_open_trades(self, engine_positions: dict = None) -> None:
        """
        One full management cycle:
          1. Sync broker positions
          2. Hard dollar stop check
          3. Green-lock SL enforcement
          4. RBZ tight trailing (three-step SL progression)
          5. Stagnation kill-switch
          6. Clean up closed positions + free engine slots
        """
        log_event(MANAGER_CYCLE_STARTED, symbol="SYSTEM", venue="trade_manager", details={
            "managed_count": len(self._managed),
        })

        # ── 1. Fetch broker trades ────────────────────────────────────────────
        try:
            open_trades: List[dict] = self.broker.get_trades()
        except Exception as e:
            log_event(TRAIL_NO_ACTION, symbol="SYSTEM", venue="trade_manager",
                      details={"reason": f"get_trades failed: {e}"})
            return

        broker_ids = set()

        for trade in open_trades:
            trade_id = str(trade.get("id") or trade.get("tradeID") or "")
            instrument = str(trade.get("instrument") or "")
            if not trade_id or not instrument:
                continue

            broker_ids.add(trade_id)

            # Parse core fields from OANDA API
            entry = _safe_float(trade.get("price"))
            units = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
            direction = "BUY" if (units or 0) > 0 else "SELL" if (units or 0) < 0 else "UNKNOWN"
            sl_order = trade.get("stopLossOrder") or {}
            current_sl = _safe_float(sl_order.get("price")) if sl_order else None
            tp_order = trade.get("takeProfitOrder") or {}
            current_tp = _safe_float(tp_order.get("price")) if tp_order else None
            unrealized_pnl = _safe_float(trade.get("unrealizedPL")) or 0.0

            # ── Position sync: log newly discovered positions ─────────────────
            if trade_id not in self._managed:
                self._managed[trade_id] = {
                    "trade_id":    trade_id,
                    "instrument":  instrument,
                    "direction":   direction,
                    "entry":       entry,
                    "current_sl":  current_sl,
                    "meta":        {},     # trail step state
                    "stale_price": None,   # last observed price for stagnation
                    "stale_count": 0,      # how many cycles price hasn't moved
                }
                log_event(POSITION_SYNCED, symbol=instrument, venue="trade_manager",
                          details={
                              "trade_id":  trade_id,
                              "direction": direction,
                              "entry":     entry,
                              "sl":        current_sl,
                          })
                print(f"  [MANAGER] SYNCED  {instrument} {direction}"
                      f"  entry={entry}  sl={current_sl}  id={trade_id}")
            else:
                # Refresh SL from broker state
                self._managed[trade_id]["current_sl"] = current_sl

            # ── 2. Hard dollar stop ───────────────────────────────────────────
            if unrealized_pnl <= -self._hard_stop_usd:
                try:
                    self.broker.close_trade(trade_id)
                    self._managed.pop(trade_id, None)
                    log_event(HARD_DOLLAR_STOP, symbol=instrument, venue="trade_manager",
                              details={
                                  "trade_id": trade_id,
                                  "unrealized_pnl": unrealized_pnl,
                                  "limit": self._hard_stop_usd,
                              })
                    print(f"  [MANAGER] 🛑 HARD_DOLLAR_STOP  {instrument}"
                          f"  uPnL={unrealized_pnl:.2f} <= -${self._hard_stop_usd:.0f}  CLOSED")
                    continue
                except Exception as close_err:
                    print(f"  [MANAGER] ⚠️  Hard stop close failed for {instrument}: {close_err}")

            # ── 3. Get live mid price ─────────────────────────────────────────
            current_price = self._get_price(instrument)
            if not current_price or not entry:
                continue

            # ── 3a. Scale-Out at 1R ──────────────────────────────────────────
            managed_state = self._managed.get(trade_id, {})
            # Get initial sl (from start of trade)
            initial_sl = managed_state.get("initial_sl")
            if not initial_sl and current_sl:
                managed_state["initial_sl"] = current_sl
                initial_sl = current_sl

            # Provide the actual units
            current_units = abs(float(units))
            if initial_sl and self._try_scale_out_1r(
                trade_id, instrument, direction, entry, current_price, initial_sl, current_units
            ):
                pass # Continue managing the rest!

            # ── 3b. Profit target close (full exit at 75% of TP distance) ─────
            if current_tp and self._try_profit_target_close(
                trade_id, instrument, direction, entry, current_price, current_tp
            ):
                continue  # trade is closing — skip remainder this cycle


            # ── 4. Green-lock SL enforcement ──────────────────────────────────
            if current_sl and current_sl > 0:
                green_sl, green_applied, green_floor = self._enforce_green_sl(
                    symbol=instrument,
                    direction=direction,
                    entry_price=entry,
                    current_price=current_price,
                    candidate_sl=current_sl,
                )
                if green_applied:
                    improved = (
                        (direction == "BUY"  and green_sl > current_sl)
                        or (direction == "SELL" and green_sl < current_sl)
                    )
                    if improved:
                        sl_key = f"{trade_id}:{green_sl:.5f}"
                        last_reject = self._sl_rejection_backoff.get(sl_key, 0)
                        if (time.time() - last_reject) > 60:
                            try:
                                self.broker.set_trade_stop(trade_id, green_sl)
                                self._managed[trade_id]["current_sl"] = green_sl
                                current_sl = green_sl
                                self._sl_rejection_backoff.pop(sl_key, None)
                                log_event(GREEN_LOCK_ENFORCED, symbol=instrument,
                                          venue="trade_manager", details={
                                              "trade_id": trade_id,
                                              "entry": entry,
                                              "current_price": current_price,
                                              "old_sl": current_sl,
                                              "new_sl": green_sl,
                                              "green_floor": green_floor,
                                          })
                                print(f"  [MANAGER] 🔒 GREEN_LOCK  {instrument}"
                                      f"  SL → {green_sl:.5f}")
                            except Exception as gl_err:
                                self._sl_rejection_backoff[sl_key] = time.time()
                                print(f"  [MANAGER] ⚠️  Green-lock rejected"
                                      f" {instrument} @ {green_sl:.5f}: {gl_err}")

            # ── 5. RBZ Tight Trailing (Three-Step SL) ─────────────────────────
            managed_state = self._managed.get(trade_id, {})
            trail_trade = {
                "id":          trade_id,
                "trade_id":    trade_id,
                "symbol":      instrument,
                "instrument":  instrument,
                "side":        direction,
                "direction":   direction,
                "entry":       entry,
                "entry_price": entry,
                "sl":          current_sl or 0.0,
                "stop_loss":   current_sl or 0.0,
                "meta":        managed_state.get("meta", {}),
            }

            strat = managed_state.get("strategy_name")
            tags = set(managed_state.get("tags") or [])
            trail_policy = policy_for(instrument, strat, tags)

            try:
                # Dynamic ATR Trailing Calibration
                # Extrapolate 14-period M15 ATR to intelligently widen/tighten trailing stops
                _m15 = self.broker.get_historical_data(instrument, count=15, granularity="M15")
                _closes = [float(c.get("mid", {}).get("c", 0)) for c in _m15]
                _highs = [float(c.get("mid", {}).get("h", 0)) for c in _m15]
                _lows = [float(c.get("mid", {}).get("l", 0)) for c in _m15]
                if len(_closes) >= 15:
                    _tr = []
                    for i in range(1, len(_closes)):
                        _h, _l, _pc = _highs[i], _lows[i], _closes[i-1]
                        _tr.append(max(_h - _l, abs(_h - _pc), abs(_l - _pc)))
                    _atr = sum(_tr) / len(_tr)
                    
                    from .trail_logic import calibrate_from_atr
                    trail_policy = calibrate_from_atr(trail_policy, _atr / current_price)
            except Exception as e:
                pass  # Fall back to base trail logic if ATR fetch fails


            def _adjust_stop(tid: str, new_sl: float) -> None:
                # Profit-only guard: only tighten SL when trade is in profit.
                # This preserves the OANDA broker trailing stop while the trade
                # is unprofitable, preventing stagnation from squeezing losers.
                if entry and current_price:
                    in_profit = (
                        (direction == "BUY" and current_price > entry)
                        or (direction == "SELL" and current_price < entry)
                    )
                    if not in_profit:
                        return
                try:
                    self.broker.set_trade_stop(tid, new_sl)
                    if tid in self._managed:
                        self._managed[tid]["current_sl"] = new_sl
                except Exception as e:
                    print(f"  [MANAGER] ⚠️  Trail SL set failed: {e}")

            def _trail_log(msg: str) -> None:
                print(f"  [MANAGER] {msg}")

            # ── Detect counter-trend & Tight Trail (DISABLED) ──
            # Disabled by Operator: Let the trades breathe to hit the full 30-pip Take Profit.
            # Trailing stops were choking winners at +10 pips, ruining the 1:2 R:R math.
            pass

            # ── 6. Stagnation kill-switch ──────────────────────────────────────
            self._handle_stagnation(
                trade_id=trade_id,
                instrument=instrument,
                direction=direction,
                entry=entry,
                current_sl=current_sl,
                current_price=current_price,
                engine_positions=engine_positions,
            )

            # Calculate rich metrics for human live tail
            margin_used = _safe_float(trade.get("marginUsed")) or 1.0
            profit_pct = (unrealized_pnl / margin_used) * 100.0 if margin_used > 1.0 else 0.0
            pip_size = 0.01 if "JPY" in (instrument or "").upper() else 0.0001
            pips = ((current_price - entry) if direction == "BUY" else (entry - current_price)) / pip_size
            
            # SL Distance & Lock Status
            sl_dist_pips = abs(current_price - current_sl) / pip_size if current_sl else 0.0
            is_locked = (current_sl is not None and current_sl > entry) if direction == "BUY" else (current_sl is not None and current_sl > 0 and current_sl < entry)
            
            # Risk/Reward (using the initial 20 pip standard risk)
            initial_risk_pips = float(os.getenv("RBOT_SL_PIPS", "20.0"))
            rr_ratio = pips / initial_risk_pips if initial_risk_pips > 0 else 0.0

            strat = managed_state.get("strategy_name", "Trend Continuation")
            tf = "M15"
            if engine_positions is not None and trade_id in engine_positions:
                 ep = engine_positions[trade_id]
                 strat = ep.get("strategy", strat)
                 tf = ep.get("timeframe", tf)

            # ── Narration: TRAIL_CANDIDATE ─────────────────────────────────────
            log_event(TRAIL_CANDIDATE, symbol=instrument, venue="trade_manager",
                      details={
                          "trade_id":   trade_id,
                          "direction":  direction,
                          "entry":      entry,
                          "current_sl": self._managed.get(trade_id, {}).get("current_sl", current_sl),
                          "price":      current_price,
                          "pnl":        unrealized_pnl,
                          "profit_pct": profit_pct,
                          "pips":       pips,
                          "sl_dist":    sl_dist_pips,
                          "is_locked":  is_locked,
                          "rr_ratio":   rr_ratio,
                          "strategy":   strat,
                          "timeframe":  tf,
                          "rules":      f"TP: UNLIMITED | GreenLock: {self._green_lock_min_profit_pips}p",
                      })

        # ── 7. Remove closed positions from local state ───────────────────────
        for tid in list(self._managed):
            if tid not in broker_ids:
                inst = self._managed[tid].get("instrument", "")
                # ── Transcript Edge: record per-pair result ────────────────────
                _close_entry = self._managed[tid].get("entry", 0)
                _close_dir = self._managed[tid].get("direction", "")
                if _close_entry and inst:
                    try:
                        _close_price = self._get_price(inst)
                        if _close_price:
                            _pip_sz = 0.01 if "JPY" in inst.upper() else 0.0001
                            _close_pips = ((_close_price - _close_entry) if _close_dir == "BUY"
                                           else (_close_entry - _close_price)) / _pip_sz
                            _close_pnl = _close_pips  # approximate in pips
                            self._record_trade_result(inst, _close_pnl)
                    except Exception:
                        pass
                self._managed.pop(tid)
                # Free slot from engine's active_positions too
                if engine_positions is not None and tid in engine_positions:
                    engine_positions.pop(tid)
                log_event(POSITION_CLOSED, symbol=inst, venue="trade_manager",
                          details={"trade_id": tid, "reason": "not_in_broker_trades"})
                print(f"  [MANAGER] CLOSED  {inst}  id={tid}")

    # ── Profit target / Scale out closes ──────────────────────────────────────

    def _try_scale_out_1r(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry: float,
        current_price: float,
        initial_sl: float,
        current_units: float,
    ) -> bool:
        """
        Partial Exit at 1:1 Risk/Reward - STRIPPED.
        Disabled by Operator: The user requested full-size profits to match full-size losses.
        Scaling out 50% at 1R ruins the R:R math.
        """
        return False

    def _try_profit_target_close(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry: float,
        current_price: float,
        tp_price: float,
    ) -> bool:
        """
        Full close when price reaches RBOT_PROFIT_TARGET_PCT of the TP distance.
        One-shot per trade. Guards against giving profits back on reversals.

        Progress = (current - entry) / (tp - entry)   [BUY]
                 = (entry - current) / (entry - tp)   [SELL]
        """
        managed = self._managed.get(trade_id, {})
        if managed.get("meta", {}).get("profit_target_closed"):
            return False
        if not tp_price or not entry or not current_price:
            return False

        if direction == "BUY":
            total_dist = tp_price - entry
            current_dist = current_price - entry
        elif direction == "SELL":
            total_dist = entry - tp_price
            current_dist = entry - current_price
        else:
            return False

        if total_dist <= 0:
            return False

        progress = current_dist / total_dist
        if self._profit_target_pct <= 0 or progress <= 0 or progress < self._profit_target_pct:
            return False

        try:
            self.broker.close_trade(trade_id)
            managed.setdefault("meta", {})["profit_target_closed"] = True
            print(
                f"  [MANAGER] 🎯 PROFIT_TARGET_CLOSE  {instrument}"
                f"  progress={progress:.0%} of TP  threshold={self._profit_target_pct:.0%}"
                f"  price={current_price:.5f}  tp={tp_price:.5f}"
            )
            return True
        except Exception as e:
            print(f"  [MANAGER] ⚠️  Profit target close failed {instrument}: {e}")
            return False

    # ── Stagnation kill-switch ─────────────────────────────────────────────────

    def _handle_stagnation(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry: Optional[float],
        current_sl: Optional[float],
        current_price: float,
        engine_positions: Optional[dict],
    ) -> None:
        """
        Track price movement. If price has been range-bound for N cycles,
        tighten the SL by a small amount to accelerate natural exit.

        Guardrails (all must pass before any SL modification is submitted):
          BUY:
            - new_sl > current_sl        (never move SL backward)
            - new_sl < current_price     (never place SL above price)
            - new_sl >= entry (optional)  (break-even protection)
          SELL:
            - new_sl < current_sl        (never move SL backward)
            - new_sl > current_price     (never place SL below price)
            - new_sl <= entry (optional)  (break-even protection)

        Stale counter resets only when the SL change is actually applied.
        """
        if trade_id not in self._managed:
            return

        # Disable Stagnation Kill-Chain during Asian Session chop (5pm ET to 1am ET)
        try:
            from util.time_utils import broker_now_eastern
            _hour = broker_now_eastern().hour
            if _hour >= 17 or _hour < 1:
                return
        except Exception:
            pass

        pip_size = 0.01 if "JPY" in (instrument or "").upper() else 0.0001
        threshold = self._stagnation_pip_threshold * pip_size

        # Profit-only guard: never squeeze SL on a losing trade.
        # While unprofitable, the broker trailing stop (set at entry) gives adequate
        # protection. Tightening further only forces premature exits on reversals.
        if entry is not None and current_price:
            in_profit = (
                (direction == "BUY" and current_price > entry)
                or (direction == "SELL" and current_price < entry)
            )
            if not in_profit:
                return

        managed = self._managed[trade_id]
        stale_price: Optional[float] = managed.get("stale_price")
        stale_count: int = managed.get("stale_count", 0)

        # Update stale counter
        if stale_price is not None and abs(current_price - stale_price) < threshold:
            stale_count += 1
        else:
            stale_count = 0  # price moved enough — reset

        managed["stale_price"] = current_price
        managed["stale_count"] = stale_count

        # Propagate to engine_positions for eviction scoring
        if engine_positions is not None and trade_id in engine_positions:
            engine_positions[trade_id]["stale_cycles"] = stale_count

        # Not yet stagnant — nothing to do
        if stale_count < self._stagnation_cycles:
            return

        # Need a valid SL to tighten
        if not current_sl or current_sl <= 0:
            return

        tighten_by = self._stagnation_tighten_pips * pip_size

        if direction == "BUY":
            new_sl = current_sl + tighten_by

            # Guardrail 1: never move SL backward
            if new_sl <= current_sl:
                return
            # Guardrail 2: never place SL at or above current price
            if new_sl >= current_price:
                return
            # Guardrail 3: do not tighten below entry when entry is known
            if entry is not None and new_sl < entry:
                return

        elif direction == "SELL":
            new_sl = current_sl - tighten_by

            # Guardrail 1: never move SL backward
            if new_sl >= current_sl:
                return
            # Guardrail 2: never place SL at or below current price
            if new_sl <= current_price:
                return
            # Guardrail 3: do not tighten above entry when entry is known
            if entry is not None and new_sl > entry:
                return

        else:
            return  # Unknown direction — skip

        # All guardrails passed — submit SL tighten
        try:
            self.broker.set_trade_stop(trade_id, new_sl)
            managed["current_sl"] = new_sl
            managed["stale_count"] = 0  # reset ONLY on successful SL change
            print(
                f"  [MANAGER] ⌛ STAGNANT  {instrument} {direction}"
                f"  SL {current_sl:.5f} → {new_sl:.5f}"
                f"  (stale_cycles={stale_count})"
            )
        except Exception as e:
            # Do NOT reset stale_count on failure — try again next cycle
            print(f"  [MANAGER] ⚠️  Stagnation SL set failed {instrument}: {e}")

    # ── Green-lock SL enforcement ─────────────────────────────────────────────

    def _enforce_green_sl(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        candidate_sl: float,
    ) -> Tuple[float, bool, Optional[float]]:
        """
        If trade is in profit, force SL to remain in green by lock distance.
        Returns (adjusted_sl, was_applied, green_floor).

        Requires minimum profit distance before activating to avoid setting
        SL inside the spread on tiny gains.
        """
        if candidate_sl is None:
            return candidate_sl, False, None

        try:
            proposed = float(candidate_sl)
        except Exception:
            return candidate_sl, False, None

        d = (direction or "").upper()
        if not self._is_in_green(d, entry_price, current_price):
            return proposed, False, None

        pip_size = 0.01 if "JPY" in (symbol or "").upper() else 0.0001
        lock_distance = self._green_lock_pips * pip_size

        # Is the candidate SL already in profit?
        is_candidate_in_profit = False
        if d == "BUY":
            is_candidate_in_profit = proposed > entry_price
        else:
            is_candidate_in_profit = proposed > 0 and proposed < entry_price

        # Minimum profit gate
        min_profit_distance = self._green_lock_min_profit_pips * pip_size
        actual_profit_distance = abs(current_price - entry_price)

        if not is_candidate_in_profit and actual_profit_distance < min_profit_distance:
            return proposed, False, None

        # Calculate green floor
        if d == "BUY":
            green_floor = max(entry_price + lock_distance,
                              proposed if is_candidate_in_profit else 0)
            adjusted = max(proposed, green_floor)
        else:
            green_floor = min(entry_price - lock_distance,
                              proposed if is_candidate_in_profit else float("inf"))
            adjusted = min(proposed, green_floor)

        return adjusted, abs(adjusted - proposed) > 1e-12, green_floor

    @staticmethod
    def _is_in_green(direction: str, entry_price: float, current_price: float) -> bool:
        if direction == "BUY":
            return current_price > entry_price
        return current_price < entry_price

    # ── Price helper ──────────────────────────────────────────────────────────

    def _get_price(self, instrument: str) -> float:
        """Fetch live mid price using connector.get_live_prices()."""
        try:
            prices = self.broker.get_live_prices([instrument])
            quote = prices.get(instrument, {})
            mid = quote.get("mid")
            if mid is not None:
                return float(mid)
            bid = quote.get("bid")
            ask = quote.get("ask")
            if bid and ask:
                return (float(bid) + float(ask)) / 2.0
        except Exception:
            pass
        return 0.0

    def position_count(self) -> int:
        return len(self._managed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
