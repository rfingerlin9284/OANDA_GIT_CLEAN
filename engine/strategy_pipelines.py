"""
engine/strategy_pipelines.py
RBOTZILLA_OANDA_CLEAN — Phase 9

Four independent strategy pipelines, each with its own detector subset,
vote threshold, and confidence gate. All read from the same candle array
fetched once per pair per cycle.

Pipelines:
  1. run_momentum_pipeline   — SMA/EMA/Fib trend following (H4-confirmed)
  2. run_reversal_pipeline   — Trap/LiqSweep/RSI reversal at key levels
  3. run_meanrev_pipeline    — Bollinger + S&D + RSI mean reversion
  4. run_scalp_pipeline      — FVG + Order Block intraday scalp

Each returns an AggregatedSignal (same type as scan_symbol) or None.
The trade_engine scan loop picks the highest-confidence qualifying signal
per pair and places it — strategy class is embedded in the signal label.
"""

from __future__ import annotations

import os
from typing import Optional, List

from strategies.multi_signal_engine import (
    AggregatedSignal,
    detect_momentum_sma,
    detect_ema_stack,
    detect_fibonacci,
    detect_liquidity_sweep,
    detect_trap_reversal,
    detect_rsi_extremes,
    detect_mean_reversion_bb,
    detect_fvg,
    detect_aggressive_shorting_ob,
)
from engine.mean_reversion_scanner import scan_sideways_symbol


# ── Confidence thresholds (env-overridable) ───────────────────────────────────

_MOMENTUM_CONF  = float(os.getenv("RBOT_MIN_SIGNAL_CONFIDENCE",      "0.75"))
_REVERSAL_CONF  = float(os.getenv("RBOT_REVERSAL_MIN_CONFIDENCE",    "0.72"))
_MEANREV_CONF   = float(os.getenv("RBOT_MEANREV_MIN_CONFIDENCE",     "0.70"))
_SCALP_CONF     = float(os.getenv("RBOT_SCALP_MIN_CONFIDENCE",       "0.72"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_direction(results) -> Optional[str]:
    """Return the majority direction among non-None detector results."""
    buys  = sum(1 for r in results if r and r.direction == "BUY")
    sells = sum(1 for r in results if r and r.direction == "SELL")
    if buys > sells:
        return "BUY"
    if sells > buys:
        return "SELL"
    return None


def _aggregate(
    symbol: str,
    detectors: list,   # list of SignalResult | None
    strategy_label: str,
    session_label: str,
    min_confidence: float,
    min_votes: int,
) -> Optional[AggregatedSignal]:
    """
    Aggregate a list of SignalResult objects into one AggregatedSignal.
    Picks the direction with the most votes, then uses the highest-confidence
    signal in that direction for SL/TP geometry.
    """
    fired = [d for d in detectors if d is not None]
    if not fired:
        return None

    direction = _best_direction(fired)
    if direction is None:
        return None

    aligned = [d for d in fired if d.direction == direction]
    if len(aligned) < min_votes:
        return None

    # Use highest-confidence aligned signal for SL/TP
    best = max(aligned, key=lambda d: d.confidence)
    if best.confidence < min_confidence:
        return None

    # R:R guard — must be at least 1.5:1
    entry = getattr(best, 'entry', 0.0) or 0.0
    sl    = getattr(best, 'sl',    0.0) or 0.0
    tp    = getattr(best, 'tp',    0.0) or 0.0
    if sl == 0 or tp == 0:
        return None
    if entry != 0:
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or (reward / risk) < 1.5:
            return None

    # Use current price as entry if detector didn't set one
    actual_entry = entry if entry != 0 else sl  # fallback — trade_engine rebases anyway

    sig = AggregatedSignal(
        symbol          = symbol,
        direction       = direction,
        confidence      = round(best.confidence, 4),
        entry           = round(actual_entry, 5),
        sl              = round(sl, 5),
        tp              = round(tp, 5),
        votes           = len(aligned),
        detectors_fired = [d.detector for d in aligned],
        all_results     = [d for d in fired],
        session         = session_label,
        session_mult    = 1.0,
        signal_type     = strategy_label,
    )
    sig._strategy  = strategy_label
    sig._timeframe = "M15"
    return sig



# ── Pipeline 1: MOMENTUM ──────────────────────────────────────────────────────

def run_momentum_pipeline(
    symbol: str,
    candles: list,
    min_confidence: Optional[float] = None,
) -> Optional[AggregatedSignal]:
    """
    SMA crossover + EMA stack + Fibonacci retracement.
    Needs 2 of 3 detectors to agree.
    H4 MTF alignment is handled by the caller (trade_engine).
    """
    conf = min_confidence if min_confidence is not None else _MOMENTUM_CONF
    results = [
        detect_momentum_sma(symbol, candles),
        detect_ema_stack(symbol, candles),
        detect_fibonacci(symbol, candles),
    ]
    sig = _aggregate(symbol, results, "momentum", "Momentum [M15]", conf, min_votes=2)
    if sig:
        sig.signal_type = "trend"   # flag for H4 MTF gate in trade_engine
    return sig


# ── Pipeline 2: REVERSAL / TRAP ───────────────────────────────────────────────

def run_reversal_pipeline(
    symbol: str,
    candles: list,
    min_confidence: Optional[float] = None,
) -> Optional[AggregatedSignal]:
    """
    Trap reversal (pin bar / engulfing) + Liquidity sweep + RSI extreme.
    Needs 1 of 3 detectors to qualify (reversals are single-pattern setups).
    """
    conf = min_confidence if min_confidence is not None else _REVERSAL_CONF
    results = [
        detect_trap_reversal(symbol, candles),
        detect_liquidity_sweep(symbol, candles),
        detect_rsi_extremes(symbol, candles),
    ]
    sig = _aggregate(symbol, results, "reversal", "Reversal [M15]", conf, min_votes=1)
    if sig:
        sig.signal_type = "reversal"
    return sig


# ── Pipeline 3: MEAN REVERSION ────────────────────────────────────────────────

def run_meanrev_pipeline(
    symbol: str,
    candles: list,
    min_confidence: Optional[float] = None,
) -> Optional[AggregatedSignal]:
    """
    Bollinger Band pierce + RSI extreme + institutional S&D zone retest.
    Needs 1 of 3 detectors (each is independently self-sufficient).
    S&D scanner already vetted via scan_sideways_symbol — here we check
    BB and RSI independently to cover setups the S&D scanner misses.
    """
    conf = min_confidence if min_confidence is not None else _MEANREV_CONF
    results = [
        detect_mean_reversion_bb(symbol, candles),
        detect_rsi_extremes(symbol, candles),
    ]
    # Also try the dedicated S&D scanner as a third input
    sd_sig = scan_sideways_symbol(symbol, candles, min_confidence=conf)
    if sd_sig:
        # Wrap into a pseudo-SignalResult-compatible object
        class _SD:
            detector  = "institutional_sd"
            direction = sd_sig.direction
            confidence = sd_sig.confidence
            entry     = 0.0
            sl        = sd_sig.sl
            tp        = sd_sig.tp
        results.append(_SD())

    sig = _aggregate(symbol, results, "mean_reversion", "MeanRev [M15]", conf, min_votes=1)
    if sig:
        sig.signal_type = "mean_reversion"
    return sig


# ── Pipeline 4: FVG SCALP ─────────────────────────────────────────────────────

def run_scalp_pipeline(
    symbol: str,
    candles: list,
    min_confidence: Optional[float] = None,
) -> Optional[AggregatedSignal]:
    """
    Fair Value Gap retest + Bearish Order Block displacement.
    Needs 1 of 2 detectors (FVG is standalone; OB is standalone).
    """
    conf = min_confidence if min_confidence is not None else _SCALP_CONF
    results = [
        detect_fvg(symbol, candles),
        detect_aggressive_shorting_ob(symbol, candles),
    ]
    sig = _aggregate(symbol, results, "scalp", "FVG Scalp [M15]", conf, min_votes=1)
    if sig:
        sig.signal_type = "scalp"
    return sig
