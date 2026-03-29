#!/usr/bin/env python3
"""
engine/mean_reversion_scanner.py

Extracted from Phoenix `institutional_sd.py`.
Scans for Supply & Demand institutional zones and returns an AggregatedSignal
if price is retesting a fresh zone. Configured explicitly for SIDEWAYS regimes.
"""

from typing import Optional, List, Dict, Any
import sys
import os
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.multi_signal_engine import AggregatedSignal

def _f(c: dict, *keys) -> float:
    for k in keys:
        v = c.get(k)
        if v is not None:
            return float(v)
    return 0.0

def _ohlc(candles: List[Any]):
    O, H, L, C = [], [], [], []
    for c in candles:
        mid = c.get("mid", {}) if isinstance(c, dict) and "mid" in c else c
        O.append(_f(mid, "o"))
        H.append(_f(mid, "h"))
        L.append(_f(mid, "l"))
        C.append(_f(mid, "c"))
    return O, H, L, C

def _pip(sym: str) -> float: 
    return 0.01 if "JPY" in sym.upper() else 0.0001

def _ema(seq: List[float], n: int) -> float:
    if not seq: return 0.0
    k = 2.0 / (n + 1)
    v = seq[0]
    for p in seq[1:]:
        v = p * k + v * (1 - k)
    return v

def _detect_sd_zones(highs, lows, closes, opens, lookback=60, min_move_mult=2.0) -> Dict[str, List[Dict]]:
    if len(closes) < lookback + 5:
        return {"demand": [], "supply": []}

    slice_o = opens[-lookback:]
    slice_h = highs[-lookback:]
    slice_l = lows[-lookback:]
    slice_c = closes[-lookback:]

    bodies = [abs(slice_c[i] - slice_o[i]) for i in range(len(slice_c))]
    avg_body = sum(bodies) / max(len(bodies), 1) if bodies else 1e-9

    demand_zones: List[Dict] = []
    supply_zones: List[Dict] = []

    for i in range(len(slice_c) - 3):
        body_i = abs(slice_c[i] - slice_o[i])
        # Base candle: indecision (small body)
        if body_i > 0.5 * avg_body:
            continue

        # Next candle: strong impulse
        impulse_body = abs(slice_c[i+1] - slice_o[i+1])
        if impulse_body < min_move_mult * avg_body:
            continue

        # Demand: base then bullish impulse
        if slice_c[i+1] > slice_o[i+1]:
            lower = slice_l[i]
            upper = max(slice_o[i], slice_c[i])
            if upper <= lower: upper = lower + avg_body
            fresh = not any(slice_l[j] < upper for j in range(i+2, len(slice_l)))
            demand_zones.append({"lower": lower, "upper": upper, "fresh": fresh,
                                  "origin_idx": i, "strength": impulse_body / avg_body})

        # Supply: base then bearish impulse
        elif slice_c[i+1] < slice_o[i+1]:
            upper = slice_h[i]
            lower = min(slice_o[i], slice_c[i])
            if lower >= upper: lower = upper - avg_body
            fresh = not any(slice_h[j] > lower for j in range(i+2, len(slice_h)))
            supply_zones.append({"lower": lower, "upper": upper, "fresh": fresh,
                                  "origin_idx": i, "strength": impulse_body / avg_body})

    return {"demand": demand_zones, "supply": supply_zones}

def scan_sideways_symbol(symbol: str, candles: List[Any], min_confidence: float = 0.68) -> Optional[AggregatedSignal]:
    """
    Evaluates raw candles for fresh Supply & Demand zone retests.
    Returns an AggregatedSignal structured exactly like the Momentum engine if a valid S&D retest is found.
    """
    if len(candles) < 70:
        return None

    O, H, L, C = _ohlc(candles)
    pip   = _pip(symbol)
    price = C[-1]
    ema55 = _ema(C[-70:], 55)

    zones = _detect_sd_zones(H, L, C, O, lookback=60)
    
    # Sideways regimes require tighter targets physically inside the box range
    target_rr = float(os.getenv("RBOT_CHOP_TARGET_RR", "1.5"))

    # BUY Logic: Price dipping into a fresh demand zone (Support)
    if price > ema55:
        for z in sorted(zones["demand"], key=lambda x: -x["strength"]):
            if not z["fresh"]:
                continue
            if z["lower"] <= price <= z["upper"]:
                sl_   = z["lower"] - 2 * pip
                risk  = price - sl_
                if risk <= 0: continue
                tp    = price + risk * target_rr
                conf  = min(0.90, 0.68 + min(z["strength"] * 0.04, 0.18))
                
                if conf >= min_confidence:
                    return AggregatedSignal(
                        symbol=symbol,
                        direction="BUY",
                        confidence=conf,
                        votes=1,
                        detectors_fired=["institutional_sd"],
                        session="S&D Scalp",
                        sl=round(sl_, 5),
                        tp=round(tp, 5)
                    )

    # SELL Logic: Price popping into a fresh supply zone (Resistance)
    if price < ema55:
        for z in sorted(zones["supply"], key=lambda x: -x["strength"]):
            if not z["fresh"]:
                continue
            if z["lower"] <= price <= z["upper"]:
                sl_   = z["upper"] + 2 * pip
                risk  = sl_ - price
                if risk <= 0: continue
                tp    = price - risk * target_rr
                conf  = min(0.90, 0.68 + min(z["strength"] * 0.04, 0.18))
                
                if conf >= min_confidence:
                    return AggregatedSignal(
                        symbol=symbol,
                        direction="SELL",
                        confidence=conf,
                        votes=1,
                        detectors_fired=["institutional_sd"],
                        session="S&D Scalp",
                        sl=round(sl_, 5),
                        tp=round(tp, 5)
                    )

    return None
