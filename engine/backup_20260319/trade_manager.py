"""
engine/trade_manager.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Manages existing OANDA broker trades in ATTACH_ONLY mode.
Uses trail_logic.TrailLogic for three-step SL progression.
Logs to narration.jsonl via narration_logger (not Python logging).

Bugs fixed vs earlier draft:
  1. Uses connector.get_trades() not get_open_trades()
  2. Uses connector.get_live_prices([symbol]) not get_current_price()
  3. Logs via narration_logger so events appear in narration.jsonl
"""

from typing import Dict, List, Optional

from .trail_logic import TrailLogic
from util.narration_logger import (
    log_event,
    POSITION_SYNCED, POSITION_CLOSED,
    TRAIL_CANDIDATE, TRAIL_SUBMIT_ALLOWED,
    TRAIL_SL_SET, TRAIL_SL_REJECTED, TRAIL_NO_ACTION,
    BREAK_EVEN_SET,
)


class TradeManager:
    """
    Manages existing trades in ATTACH_ONLY mode.
    Pulls open trades from OANDA, applies Phoenix trailing, calls broker.set_trade_stop.
    Logs every trail event via narration_logger. Phoenix remains the opener.
    """

    def __init__(self, broker) -> None:
        self.broker        = broker
        self.trail_logic   = TrailLogic()
        self._managed: Dict[str, dict] = {}   # trade_id → last-known state

    async def tick(self) -> None:
        """Called from engine run() loop every scan cycle."""
        self.manage_open_trades()

    def manage_open_trades(self) -> None:
        """
        One management cycle:
          1. Fetch all open broker trades
          2. Sync state (new/closed positions)
          3. Apply trailing-stop logic to each
        """
        try:
            # ── 1. Pull all open trades from broker ───────────────────────────
            open_trades: List[dict] = self.broker.get_trades()
        except Exception as e:
            log_event(TRAIL_NO_ACTION, symbol="SYSTEM", venue="trade_manager",
                      details={"reason": f"get_trades failed: {e}"})
            return

        broker_ids = set()

        for trade in open_trades:
            trade_id   = str(trade.get("id") or trade.get("tradeID") or "")
            instrument = str(trade.get("instrument") or "")
            if not trade_id or not instrument:
                continue

            broker_ids.add(trade_id)

            # Parse entry and current SL from OANDA API strings
            entry      = _safe_float(trade.get("price"))
            units      = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
            direction  = "BUY" if units > 0 else "SELL" if units < 0 else "UNKNOWN"
            sl_order   = trade.get("stopLossOrder") or {}
            current_sl = _safe_float(sl_order.get("price")) if sl_order else None

            # Log newly discovered positions
            if trade_id not in self._managed:
                self._managed[trade_id] = {
                    "trade_id":   trade_id,
                    "instrument": instrument,
                    "direction":  direction,
                    "entry":      entry,
                    "current_sl": current_sl,
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

            # ── 2. Get live mid price ─────────────────────────────────────────
            current_price = self._get_price(instrument)
            if not current_price or not entry:
                continue

            # ── 3. Compute trailing SL ────────────────────────────────────────
            log_event(TRAIL_CANDIDATE, symbol=instrument, venue="trade_manager",
                      details={
                          "trade_id":    trade_id,
                          "direction":   direction,
                          "entry":       entry,
                          "current_sl":  current_sl,
                          "price":       current_price,
                      })

            new_sl = self.trail_logic.calculate_new_sl(
                entry=entry,
                current_price=current_price,
                current_sl=current_sl,
                direction=direction,
                instrument=instrument,
            )

            if new_sl is None:
                log_event(TRAIL_NO_ACTION, symbol=instrument, venue="trade_manager",
                          details={"trade_id": trade_id, "price": current_price})
                continue

            # ── 4. Submit SL update to broker ─────────────────────────────────
            log_event(TRAIL_SUBMIT_ALLOWED, symbol=instrument, venue="trade_manager",
                      details={"trade_id": trade_id, "proposed_sl": new_sl})

            response = self.broker.set_trade_stop(trade_id, new_sl)
            success  = bool(response.get("success")) if isinstance(response, dict) else bool(response)

            if success:
                self._managed[trade_id]["current_sl"] = new_sl
                event = BREAK_EVEN_SET if _is_breakeven(new_sl, entry, direction) else TRAIL_SL_SET
                log_event(event, symbol=instrument, venue="trade_manager",
                          details={"trade_id": trade_id, "new_sl": new_sl, "price": current_price})
                print(f"  [MANAGER] {event:18s}  {instrument}  new_sl={new_sl:.5f}  price={current_price:.5f}")
            else:
                error = response.get("error", "unknown") if isinstance(response, dict) else str(response)
                log_event(TRAIL_SL_REJECTED, symbol=instrument, venue="trade_manager",
                          details={"trade_id": trade_id, "proposed_sl": new_sl, "error": error})
                print(f"  [MANAGER] TRAIL_SL_REJECTED  {instrument}  {error}")

        # ── 5. Remove closed positions from local state ───────────────────────
        for tid in list(self._managed):
            if tid not in broker_ids:
                inst = self._managed[tid].get("instrument", "")
                self._managed.pop(tid)
                log_event(POSITION_CLOSED, symbol=inst, venue="trade_manager",
                          details={"trade_id": tid, "reason": "not_in_broker_trades"})
                print(f"  [MANAGER] CLOSED  {inst}  id={tid}")

    def _get_price(self, instrument: str) -> float:
        """Fetch live mid price using connector.get_live_prices()."""
        try:
            prices = self.broker.get_live_prices([instrument])
            quote  = prices.get(instrument, {})
            mid    = quote.get("mid")
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


def _is_breakeven(new_sl: float, entry: float, direction: str) -> bool:
    """True if new_sl is at or near entry (Step 1 or Step 2 lock)."""
    if direction in ("BUY", "LONG"):
        return new_sl <= entry * 1.002   # within 20 pips of entry = BE territory
    else:
        return new_sl >= entry * 0.998
