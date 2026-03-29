#!/usr/bin/env python3
"""
broker_tradability_gate.py
RBOTZILLA_OANDA_CLEAN — Phase 5
Label: NEW_CLEAN_REWRITE
Runtime-verified gate pattern from Phoenix session 2026-03-17.

Call check_broker_tradability(connector, symbol) BEFORE any order submission.

Gate checks in order:
  1. Live quote fetch success from /pricing endpoint
  2. OANDA 'tradeable' boolean flag
  3. Quote timestamp freshness (stale = market likely closed)
  4. Ask-bid spread in pips vs threshold
  5. Symbol already active in broker (live dedup)
  6. Cooldown timer (symbol traded too recently)
  7. OCO payload integrity (SL and TP both present and logical)

Post-placement checks (called separately from trade_engine/manager):
  8. Account/NAV/margin checked live via get_account_info()
  9. Broker response after order submit checked for trade_id
  10. SL modification response checked before local state update

Env vars (all optional, safe defaults shown):
    RBOT_MAX_SPREAD_PIPS          = 8.0
    RBOT_MAX_STALE_QUOTE_SECONDS  = 120
    RBOT_COOLDOWN_SECONDS         = 3600   (1 hour per symbol)
"""

import os
from datetime import datetime, timezone as _tz
from typing import Dict, Any, Optional, Set

# ── Configurable thresholds via env ────────────────────────────────────────────
MAX_SPREAD_PIPS:         float = float(os.getenv("RBOT_MAX_SPREAD_PIPS",           "8.0"))
MAX_STALE_QUOTE_SECONDS: int   = int(os.getenv("RBOT_MAX_STALE_QUOTE_SECONDS",     "120"))
COOLDOWN_SECONDS:        int   = int(os.getenv("RBOT_COOLDOWN_SECONDS",            "3600"))

# ── Reason codes ───────────────────────────────────────────────────────────────
QUOTE_FETCH_FAILED            = "QUOTE_FETCH_FAILED"
INSTRUMENT_NOT_TRADABLE_BLOCK = "INSTRUMENT_NOT_TRADABLE_BLOCK"
STALE_QUOTE_BLOCK             = "STALE_QUOTE_BLOCK"
MARKET_CLOSED_BLOCK           = "MARKET_CLOSED_BLOCK"      # alias for stale
SPREAD_TOO_WIDE_BLOCK         = "SPREAD_TOO_WIDE_BLOCK"
SYMBOL_ALREADY_ACTIVE_BLOCK   = "SYMBOL_ALREADY_ACTIVE_BLOCK"
COOLDOWN_BLOCK                = "COOLDOWN_BLOCK"
OCO_VALIDATION_BLOCK          = "OCO_VALIDATION_BLOCK"
ORDER_SUBMIT_BLOCK            = "ORDER_SUBMIT_BLOCK"
ORDER_SUBMIT_ALLOWED          = "ORDER_SUBMIT_ALLOWED"
TRAIL_SL_REJECTED             = "TRAIL_SL_REJECTED"
GATE_ERROR                    = "TRADABILITY_CHECK_ERROR"

# ── Cooldown tracking (module-level, persists for process lifetime) ────────────
_cooldown_registry: Dict[str, datetime] = {}


def reset_cooldown(symbol: str) -> None:
    """Clear cooldown for a symbol (call after trade closes)."""
    _cooldown_registry.pop(symbol.upper(), None)


def set_cooldown(symbol: str) -> None:
    """Set cooldown for a symbol (call after successful placement)."""
    _cooldown_registry[symbol.upper()] = datetime.now(_tz.utc)


# ── OCO integrity check ────────────────────────────────────────────────────────

def validate_oco_payload(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    units: int,
) -> Dict[str, Any]:
    """
    Validate OCO payload integrity before submission.
    Checks:
      - SL and TP are both non-zero
      - SL is on the correct side of entry (loss side)
      - TP is on the correct side of entry (profit side)
      - Units are non-zero
    Returns: {"valid": True} or {"valid": False, "reason": str}
    """
    if not stop_loss or not take_profit:
        return {"valid": False, "reason": "stop_loss or take_profit is zero/None"}
    if units == 0:
        return {"valid": False, "reason": "units is zero"}

    direction = direction.upper()
    if direction == "BUY":
        if stop_loss >= entry_price and entry_price > 0:
            return {"valid": False, "reason": f"BUY stop_loss {stop_loss} must be < entry {entry_price}"}
        if take_profit <= entry_price and entry_price > 0:
            return {"valid": False, "reason": f"BUY take_profit {take_profit} must be > entry {entry_price}"}
        if units < 0:
            return {"valid": False, "reason": f"BUY requires positive units, got {units}"}
    elif direction == "SELL":
        if stop_loss <= entry_price and entry_price > 0:
            return {"valid": False, "reason": f"SELL stop_loss {stop_loss} must be > entry {entry_price}"}
        if take_profit >= entry_price and entry_price > 0:
            return {"valid": False, "reason": f"SELL take_profit {take_profit} must be < entry {entry_price}"}
        if units > 0:
            return {"valid": False, "reason": f"SELL requires negative units, got {units}"}
    else:
        return {"valid": False, "reason": f"Unknown direction '{direction}'"}

    return {"valid": True}


# ── Main gate ─────────────────────────────────────────────────────────────────

def check_broker_tradability(
    connector,
    symbol: str,
    active_symbols: Optional[Set[str]] = None,
    placed_this_cycle: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Hard pre-placement gate. Must pass before any order is submitted.

    Checks (in order):
        1. Live quote fetch from /pricing — mandatory
        2. OANDA 'tradeable' flag
        3. Quote timestamp freshness (MARKET_CLOSED_BLOCK if stale)
        4. Spread in pips (SPREAD_TOO_WIDE_BLOCK)
        5. Symbol not already active at broker (SYMBOL_ALREADY_ACTIVE_BLOCK)
        6. Symbol not in per-cycle dedup set (SYMBOL_ALREADY_ACTIVE_BLOCK)
        7. Symbol not in cooldown (COOLDOWN_BLOCK)

    Checks 8–10 are handled by trade_engine.py and trade_manager.py
    after placement, not here.

    Returns:
        {"allowed": True,  "event": ORDER_SUBMIT_ALLOWED, "detail": {...}, "live_price": {...}}
        {"allowed": False, "event": <REASON_CODE>,         "detail": {...}}
    """
    try:
        # ── 5. Per-cycle dedup ────────────────────────────────────────────────
        sym_upper = symbol.upper()
        if placed_this_cycle and sym_upper in placed_this_cycle:
            return {
                "allowed": False,
                "event":   SYMBOL_ALREADY_ACTIVE_BLOCK,
                "detail":  {"symbol": symbol, "reason": "already placed this cycle"},
            }

        # ── 6. Cooldown check ─────────────────────────────────────────────────
        if sym_upper in _cooldown_registry:
            age = (datetime.now(_tz.utc) - _cooldown_registry[sym_upper]).total_seconds()
            if age < COOLDOWN_SECONDS:
                return {
                    "allowed": False,
                    "event":   COOLDOWN_BLOCK,
                    "detail":  {
                        "symbol":         symbol,
                        "cooldown_seconds_remaining": round(COOLDOWN_SECONDS - age, 0),
                    },
                }

        # ── 1. Live quote fetch ───────────────────────────────────────────────
        endpoint = f"/v3/accounts/{connector.account_id}/pricing"
        resp = connector._make_request("GET", endpoint, params={"instruments": symbol})

        if not resp or not resp.get("success"):
            return {
                "allowed": False,
                "event":   QUOTE_FETCH_FAILED,
                "detail":  {"symbol": symbol, "response": str(resp)},
            }

        prices = (resp.get("data") or {}).get("prices") or []
        if not prices:
            return {
                "allowed": False,
                "event":   QUOTE_FETCH_FAILED,
                "detail":  {"symbol": symbol, "response": "empty prices list"},
            }

        price_obj = prices[0]

        # ── 2. Broker tradeable flag ──────────────────────────────────────────
        # Only hard-block if OANDA explicitly returns tradeable=False AND
        # status="non-tradeable". Missing/ambiguous fields default to allow.
        _tradeable = price_obj.get("tradeable")
        _status    = str(price_obj.get("status", "")).lower()
        if _tradeable is False and _status == "non-tradeable":
            return {
                "allowed": False,
                "event":   INSTRUMENT_NOT_TRADABLE_BLOCK,
                "detail":  {
                    "symbol":    symbol,
                    "tradeable": False,
                    "status":    _status,
                },
            }

        # ── 3. Quote freshness ────────────────────────────────────────────────
        quote_time_str = price_obj.get("time", "")
        if quote_time_str:
            try:
                # OANDA timestamps: "2026-03-17T05:50:23.602157664Z" — trim to μs
                quote_dt  = datetime.fromisoformat(
                    quote_time_str[:26].rstrip("Z")
                ).replace(tzinfo=_tz.utc)
                age_secs = (datetime.now(_tz.utc) - quote_dt).total_seconds()
                if age_secs > MAX_STALE_QUOTE_SECONDS:
                    return {
                        "allowed": False,
                        "event":   MARKET_CLOSED_BLOCK,
                        "detail":  {
                            "symbol":            symbol,
                            "quote_age_seconds": round(age_secs, 1),
                            "max_allowed":       MAX_STALE_QUOTE_SECONDS,
                            "quote_time":        quote_time_str,
                        },
                    }
            except Exception:
                pass  # unparseable timestamp — don't hard-block, proceed to spread

        # ── 4. Spread check ───────────────────────────────────────────────────
        bids = price_obj.get("bids") or []
        asks = price_obj.get("asks") or []
        spread_pips = None
        bid = ask = None

        if bids and asks:
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            pip_mult    = 100 if "JPY" in symbol else 10000
            spread_pips = round((ask - bid) * pip_mult, 1)
            if spread_pips > MAX_SPREAD_PIPS:
                return {
                    "allowed": False,
                    "event":   SPREAD_TOO_WIDE_BLOCK,
                    "detail":  {
                        "symbol":      symbol,
                        "spread_pips": spread_pips,
                        "max_allowed": MAX_SPREAD_PIPS,
                        "bid":         bid,
                        "ask":         ask,
                    },
                }

        # ── 5. Broker active position dedup ───────────────────────────────────
        if active_symbols and sym_upper in {s.upper() for s in active_symbols}:
            return {
                "allowed": False,
                "event":   SYMBOL_ALREADY_ACTIVE_BLOCK,
                "detail":  {"symbol": symbol, "reason": "symbol active at broker"},
            }

        # ── All checks passed ─────────────────────────────────────────────────
        return {
            "allowed":    True,
            "event":      ORDER_SUBMIT_ALLOWED,
            "detail":     {
                "symbol":      symbol,
                "tradeable":   price_obj.get("tradeable", True),
                "quote_time":  quote_time_str,
                "spread_pips": spread_pips,
            },
            "live_price": {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 5) if bid and ask else None},
        }

    except Exception as err:
        return {
            "allowed": False,
            "event":   GATE_ERROR,
            "detail":  {"symbol": symbol, "error": str(err)},
        }


def check_submit_response(result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Check 9: Verify broker order submit response contains a real trade_id.
    Call this after place_oco_order() returns.
    Returns {"confirmed": True, "trade_id": str} or {"confirmed": False, "error": str}
    """
    if not result or not result.get("success"):
        return {"confirmed": False, "error": result.get("error", "no success field")}
    trade_id = result.get("trade_id") or result.get("tradeID") or ""
    if not trade_id:
        return {"confirmed": False, "error": "success=True but no trade_id in response"}
    return {"confirmed": True, "trade_id": str(trade_id)}


def check_sl_update_response(result: Optional[Dict], proposed_sl: float) -> Dict[str, Any]:
    """
    Check 10: Verify broker confirmed a stop-loss update.
    Call this after set_trade_stop() returns.
    Do NOT update local state unless this returns confirmed=True.
    """
    if not result or not result.get("success"):
        error = (result or {}).get("error", "no success field in SL response")
        return {"confirmed": False, "error": error}
    return {"confirmed": True, "new_sl": proposed_sl}
