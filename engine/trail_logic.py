"""
engine/trail_logic.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Pair-aware Two-Step SL Lock + Aggressive Trailing Stop.
Extracted from Phoenix rbz_tight_trailing.py — ONLY the golden logic.
No monkey-patching, no engine wrapping, no console commands.

Three-stage progression per trade:
  Step 1  →  Lock SL to entry + small buffer (spread-safe)
  Step 2  →  Lock SL to breakeven (50% of trigger distance)
  Trail   →  Aggressive trailing stop follows price

All thresholds are pair-class + strategy aware.
SL NEVER moves backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Set


# ── Pair classification ───────────────────────────────────────────────────────

PAIR_CLASS = {
    "EUR_USD": "major", "GBP_USD": "major", "USD_JPY": "major", "USD_CHF": "major",
    "USD_CAD": "major", "AUD_USD": "major", "NZD_USD": "major",
    "EUR_GBP": "minor", "EUR_JPY": "minor", "GBP_JPY": "minor", "AUD_JPY": "minor",
    "CAD_JPY": "minor", "EUR_AUD": "minor", "EUR_NZD": "minor", "EUR_CAD": "minor",
    "GBP_AUD": "minor", "GBP_NZD": "minor", "AUD_NZD": "minor", "AUD_CAD": "minor",
    "AUD_CHF": "minor", "NZD_JPY": "minor", "CAD_CHF": "minor", "GBP_CAD": "minor",
    "GBP_CHF": "minor", "CHF_JPY": "minor", "NZD_CAD": "minor",
}


# ── TightSL policy dataclass ─────────────────────────────────────────────────

@dataclass
class TightSL:
    step1_trigger_pct: float = 0.0020
    step1_lock_pct:    float = -0.0003
    step2_trigger_pct: float = 0.0040
    trail_trigger_pct: float = 0.0070
    trail_pct:         float = 0.0020


DEFAULTS = {
    #                     Step1 Trig   Step1 Lock   Step2 Trig   Trail Trig   Trail Pct
    "major":  TightSL(      0.0004,     0.00010,      0.0008,     0.0014,     0.0008),
    "minor":  TightSL(      0.0006,     0.00015,      0.0011,     0.0018,     0.0010),
    "exotic": TightSL(      0.0012,     0.00015,      0.0020,     0.0030,     0.0015),
}


# ── Strategy-specific policy multipliers ──────────────────────────────────────

SCALP_TAGS = {"scalp", "micro", "hf", "intraday_fast"}
SWING_TAGS = {"swing", "position", "carry", "condor", "range_swing"}


@dataclass
class StrategyPolicy:
    is_swing: bool
    allow_tp: bool
    mult_step1_trig: float = 1.0
    mult_step1_lock: float = 1.0
    mult_step2_trig: float = 1.0
    mult_trail_trig: float = 1.0
    mult_trail_pct:  float = 1.0


STRATEGY_OVERRIDES = {
    "trap_reversal_scalper":       StrategyPolicy(False, False, 0.8, 1.0, 0.9, 0.9, 0.8),
    "liquidity_sweep_scalp":      StrategyPolicy(False, False, 0.9, 1.0, 0.9, 0.9, 0.9),
    "wolfpack_ema_trend_scalp":   StrategyPolicy(False, False, 1.0, 1.0, 1.0, 0.9, 0.9),
    "fvg_breakout_scalp":         StrategyPolicy(False, False, 0.9, 1.0, 0.9, 0.9, 0.9),
    "price_action_holy_grail_scalp": StrategyPolicy(False, False, 0.9, 1.0, 0.9, 0.9, 0.8),
    "holy_grail_swing":           StrategyPolicy(True, True, 1.2, 1.0, 1.2, 1.2, 1.3),
    "fvg_breakout_swing":         StrategyPolicy(True, True, 1.2, 1.0, 1.2, 1.2, 1.3),
    "institutional_sd_swing":     StrategyPolicy(True, True, 1.3, 1.0, 1.3, 1.3, 1.4),
    "iron_condor":                StrategyPolicy(True, True, 2.0, 1.0, 2.0, 2.0, 2.0),
}


# ── Public API ────────────────────────────────────────────────────────────────

def _pair_class(symbol: str) -> str:
    return PAIR_CLASS.get((symbol or "").upper(), "exotic")


def _strategy_policy(strategy_name: Optional[str], tags: Optional[Set[str]]) -> StrategyPolicy:
    n = (strategy_name or "").strip().lower()
    if n in STRATEGY_OVERRIDES:
        return STRATEGY_OVERRIDES[n]
    t = set((x or "").strip().lower() for x in (tags or set()))
    if t & SWING_TAGS:
        return StrategyPolicy(True, True)
    if t & SCALP_TAGS:
        return StrategyPolicy(False, False)
    return StrategyPolicy(False, False)


def policy_for(
    symbol: str,
    strategy_name: Optional[str] = None,
    tags: Optional[Set[str]] = None,
) -> TightSL:
    """Return the TightSL policy for a given pair + strategy combination."""
    base = DEFAULTS[_pair_class(symbol)]
    sp = _strategy_policy(strategy_name, tags)
    return TightSL(
        step1_trigger_pct=base.step1_trigger_pct * sp.mult_step1_trig,
        step1_lock_pct=base.step1_lock_pct * sp.mult_step1_lock,
        step2_trigger_pct=base.step2_trigger_pct * sp.mult_step2_trig,
        trail_trigger_pct=base.trail_trigger_pct * sp.mult_trail_trig,
        trail_pct=base.trail_pct * sp.mult_trail_pct,
    )


def should_allow_tp(strategy_name: Optional[str], tags: Optional[Set[str]] = None) -> bool:
    """Return True if the strategy permits a fixed TP order."""
    return bool(_strategy_policy(strategy_name, tags).allow_tp)


def tp_guard(
    strategy_name: Optional[str],
    tags: Optional[Set[str]] = None,
    proposed_tp: Optional[float] = None,
) -> Optional[float]:
    """Return proposed_tp if the strategy allows TP, else None."""
    return proposed_tp if should_allow_tp(strategy_name, tags) else None


def calibrate_from_atr(policy: TightSL, atr_pct_of_price: float) -> TightSL:
    """Scale policy thresholds by observed ATR relative to 0.3% baseline."""
    k = max(0.75, min(1.5, atr_pct_of_price / 0.003))
    return replace(
        policy,
        step1_trigger_pct=policy.step1_trigger_pct * k,
        step2_trigger_pct=policy.step2_trigger_pct * k,
        trail_trigger_pct=policy.trail_trigger_pct * k,
        trail_pct=policy.trail_pct * max(0.8, min(1.2, 1.0 / k)),
    )


def apply_tight_sl(
    *,
    policy: TightSL,
    trade: Dict[str, Any],
    price: float,
    adjust_stop_cb: Callable[[str, float], None],
    log: Callable[[str], None],
) -> None:
    """
    Apply Three-Step SL progression to a single trade.
    Mutates trade['meta'] to track step state.

    Args:
        policy:          TightSL for this pair+strategy
        trade:           dict with keys: id/trade_id, symbol/instrument,
                         side/direction, entry/entry_price, sl/stop_loss, meta
        price:           current mid price
        adjust_stop_cb:  fn(trade_id, new_sl) to submit SL change to broker
        log:             fn(msg) for narration output
    """
    side = (trade.get("side") or trade.get("direction") or "").upper()
    symbol = trade.get("symbol") or trade.get("instrument") or "UNKNOWN"
    entry = float(trade.get("entry") or trade.get("entry_price") or 0.0)
    sl = float(trade.get("sl") or trade.get("stop_loss") or 0.0)
    meta = dict(trade.get("meta") or {})
    trade_id = str(trade.get("id") or trade.get("trade_id") or "")

    if not trade_id or not entry or not price:
        return

    changed = False

    # ── Step 1: Lock SL near entry ────────────────────────────────────────
    if not meta.get("tight_step1", False):
        tgt = entry * (1.0 + policy.step1_trigger_pct) if side == "BUY" else entry * (1.0 - policy.step1_trigger_pct)
        if (side == "BUY" and price >= tgt) or (side == "SELL" and price <= tgt):
            # Minimum lock clears spread (3 pips for non-JPY, 30 pips for JPY)
            min_lock_pips = 0.0003 if "JPY" not in symbol else 0.03
            if side == "BUY":
                new_sl = max(entry * (1.0 + policy.step1_lock_pct), entry + min_lock_pips)
                if new_sl > sl:
                    adjust_stop_cb(trade_id, new_sl)
                    meta["tight_step1"] = True
                    changed = True
                    log(f"[TightSL] {symbol} STEP1 lock → SL {new_sl:.5f}")
            else:
                new_sl = min(entry * (1.0 - abs(policy.step1_lock_pct)), entry - min_lock_pips)
                if sl == 0.0 or new_sl < sl:
                    adjust_stop_cb(trade_id, new_sl)
                    meta["tight_step1"] = True
                    changed = True
                    log(f"[TightSL] {symbol} STEP1 lock → SL {new_sl:.5f}")

    # ── Step 2: Breakeven lock ────────────────────────────────────────────
    if meta.get("tight_step1", False) and not meta.get("tight_step2", False):
        tgt = entry * (1.0 + policy.step2_trigger_pct) if side == "BUY" else entry * (1.0 - policy.step2_trigger_pct)
        if (side == "BUY" and price >= tgt) or (side == "SELL" and price <= tgt):
            lock2_pct = policy.step2_trigger_pct * 0.50
            if side == "BUY":
                new_sl = max(sl, entry * (1.0 + lock2_pct))
            else:
                new_sl = min(sl if sl > 0 else entry, entry * (1.0 - lock2_pct))

            if (side == "BUY" and new_sl > sl) or (side == "SELL" and (sl == 0.0 or new_sl < sl)):
                adjust_stop_cb(trade_id, new_sl)
                log(f"[TightSL] {symbol} STEP2 breakeven → SL {new_sl:.5f}")
            meta["tight_step2"] = True
            changed = True

    # ── Trail: Aggressive trailing stop ───────────────────────────────────
    if meta.get("tight_step2", False):
        tgt = entry * (1.0 + policy.trail_trigger_pct) if side == "BUY" else entry * (1.0 - policy.trail_trigger_pct)
        if (side == "BUY" and price >= tgt) or (side == "SELL" and price <= tgt):
            if side == "BUY":
                new_sl = max(sl, price * (1.0 - policy.trail_pct))
                if new_sl > sl:
                    adjust_stop_cb(trade_id, new_sl)
                    changed = True
                    log(f"[TightSL] {symbol} TRAIL → SL {new_sl:.5f}")
            else:
                new_sl = min(sl if sl > 0 else price, price * (1.0 + policy.trail_pct))
                if sl == 0.0 or new_sl < sl:
                    adjust_stop_cb(trade_id, new_sl)
                    changed = True
                    log(f"[TightSL] {symbol} TRAIL → SL {new_sl:.5f}")

    if changed:
        trade["meta"] = meta
