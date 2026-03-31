"""
Multi-Signal Engine — RBOTzilla Phoenix
Runs every candle-based detector in parallel, votes, and returns the
highest-confidence aggregated signal across ALL strategies for a symbol.

Detectors (all run from raw OANDA candles, zero external deps):
  1. MomentumSMA   — SMA20/50 crossover + rate-of-change
  2. EMAStack      — EMA8/21/55 stacked trend + crossover
  3. FVG           — Fair Value Gap (3-candle imbalance zone)
  4. Fibonacci     — Swing-high/low retracements (0.236–0.786)
  5. LiqSweep      — Liquidity sweep of equal highs/lows + reversal
  6. TrapReversal  — Pin-bar / engulfing trap at key levels
  7. SessionBias   — Caps confidence outside high-probability sessions

Incremental trade-management signals:
  - "SCALE_OUT_HALF"   when price crosses 1R profit
  - "TRAIL_TIGHT"      when price crosses 2R profit
  - "CLOSE_ALL"        when price reverts past trailing level or session ends

PIN: 841921
"""

from __future__ import annotations
import math
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _closes(candles: list) -> list[float]:
    out = []
    for c in candles:
        if isinstance(c, dict):
            v = c.get("mid", {}).get("c") or c.get("close") or c.get("c")
            if v:
                out.append(float(v))
    return out


def _highs(candles: list) -> list[float]:
    out = []
    for c in candles:
        if isinstance(c, dict):
            v = c.get("mid", {}).get("h") or c.get("high") or c.get("h")
            if v:
                out.append(float(v))
    return out


def _lows(candles: list) -> list[float]:
    out = []
    for c in candles:
        if isinstance(c, dict):
            v = c.get("mid", {}).get("l") or c.get("low") or c.get("l")
            if v:
                out.append(float(v))
    return out


def _sma(seq: list[float], n: int) -> float:
    if len(seq) < n:
        return sum(seq) / max(len(seq), 1)
    return sum(seq[-n:]) / n


def _ema(seq: list[float], n: int) -> float:
    if not seq:
        return 0.0
    k = 2.0 / (n + 1)
    val = seq[0]
    for p in seq[1:]:
        val = p * k + val * (1 - k)
    return val


def _roc(seq: list[float], n: int = 10) -> float:
    if len(seq) <= n:
        return 0.0
    a, b = float(seq[-n - 1]), float(seq[-1])
    return (b - a) / max(abs(a), 1e-9) * 100.0


def _swing_high(highs: list[float], lookback: int = 30) -> float:
    return max(highs[-lookback:]) if highs else 0.0


def _swing_low(lows: list[float], lookback: int = 30) -> float:
    return min(lows[-lookback:]) if lows else 0.0


def _pip_size(symbol: str) -> float:
    """Return pip size for symbol (0.0001 for most FX, 0.01 for JPY pairs)."""
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Session Awareness
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS = {
    "tokyo":   (0,  9),    # 00:00–09:00 UTC
    "london":  (7,  16),   # 07:00–16:00 UTC
    "new_york":(12, 21),   # 12:00–21:00 UTC
}

# Which sessions are high-probability for which pairs
PAIR_SESSIONS: Dict[str, List[str]] = {
    "EUR_USD": ["london", "new_york"],
    "GBP_USD": ["london", "new_york"],
    "USD_JPY": ["tokyo", "london", "new_york"],
    "USD_CHF": ["london", "new_york"],
    "AUD_USD": ["tokyo", "london"],
    "NZD_USD": ["tokyo", "london"],
    "USD_CAD": ["new_york", "london"],
    "GBP_JPY": ["tokyo", "london"],
    "EUR_JPY": ["tokyo", "london"],
    "XAU_USD": ["london", "new_york"],
}
_DEFAULT_SESSIONS = ["london", "new_york"]


def session_bias(symbol: str, utc_now: Optional[datetime] = None) -> Tuple[str, float]:
    """
    Returns (active_session_name, confidence_multiplier).
    Multiplier: 1.0 = prime session, 0.90 = off-session (slight dampening;
    still allows strong setups through — prevents near-total afternoon blackout).
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    h = utc_now.hour
    preferred = PAIR_SESSIONS.get(symbol.upper(), _DEFAULT_SESSIONS)

    active = []
    for name, (start, end) in SESSIONS.items():
        if start <= h < end:
            active.append(name)
    # overlap = London+NY (07–16 UTC) is highest quality
    if len(active) >= 2:
        return ("overlap", 1.0)
    for sess in active:
        if sess in preferred:
            return (sess, 1.0)
    if active:
        return (active[0], 0.90)
    return ("off_session", 0.90)


# ─────────────────────────────────────────────────────────────────────────────
# Signal result type
# ─────────────────────────────────────────────────────────────────────────────

class SignalResult:
    def __init__(
        self,
        detector: str,
        direction: Optional[str],   # "BUY" | "SELL" | None
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        meta: Dict[str, Any],
    ):
        self.detector = detector
        self.direction = direction
        self.confidence = round(confidence, 4)
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 1e-9 else 0.0
        self.meta = meta

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "rr": round(self.rr, 2),
            "meta": self.meta,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Momentum SMA Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_momentum_sma(symbol: str, candles: list) -> Optional[SignalResult]:
    closes = _closes(candles)[-100:]
    if len(closes) < 52:
        return None
    s20 = _sma(closes, 20)
    s50 = _sma(closes, 50)
    roc = _roc(closes, 10)
    pip = _pip_size(symbol)
    price = closes[-1]

    direction = None
    if s20 > s50 and roc > 0.15:
        direction = "BUY"
    elif s20 < s50 and roc < -0.15:
        direction = "SELL"
    if direction is None:
        return None

    conf = min(0.90, abs(roc) / 2.0)
    sl_dist = 10 * pip
    tp_dist = 30 * pip
    if direction == "BUY":
        sl, tp = price - sl_dist, price + tp_dist
    else:
        sl, tp = price + sl_dist, price - tp_dist

    return SignalResult("momentum_sma", direction, conf, price, sl, tp,
                        {"sma20": s20, "sma50": s50, "roc": roc})


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMA Stack Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_ema_stack(symbol: str, candles: list) -> Optional[SignalResult]:
    closes = _closes(candles)[-120:]
    if len(closes) < 60:
        return None
    e8  = _ema(closes, 8)
    e21 = _ema(closes, 21)
    e55 = _ema(closes, 55)
    price = closes[-1]
    pip = _pip_size(symbol)

    # Bullish: price > e8 > e21 > e55 (stack)
    # Bearish: price < e8 < e21 < e55
    if price > e8 > e21 > e55:
        direction = "BUY"
        sep = (e8 - e55) / e55 * 100   # stack separation %
        conf = min(0.88, 0.55 + sep * 5)
    elif price < e8 < e21 < e55:
        direction = "SELL"
        sep = (e55 - e8) / e55 * 100
        conf = min(0.88, 0.55 + sep * 5)
    else:
        return None

    # ── Candle body momentum quality check ────────────────────────────────────
    # Source: "15 Best Price Action Strategies" — growing bodies = momentum gain,
    # shrinking bodies as price approaches level = exhaustion / momentum loss.
    try:
        _bodies = [
            abs(float(c.get("mid", {}).get("c", 0)) - float(c.get("mid", {}).get("o", 0)))
            for c in candles[-6:] if isinstance(c, dict) and c.get("mid")
        ]
        if len(_bodies) >= 4:
            _growing   = sum(1 for i in range(1, len(_bodies)) if _bodies[i] > _bodies[i-1])
            _shrinking = sum(1 for i in range(1, len(_bodies)) if _bodies[i] < _bodies[i-1])
            if _shrinking > _growing:
                conf = max(conf - 0.10, 0.50)   # momentum loss → reduce confidence
            elif _growing > _shrinking:
                conf = min(conf + 0.03, 0.88)   # momentum gain → slight boost
    except Exception:
        pass  # Fail silent — never block a signal due to this check

    sl_dist = 12 * pip
    tp_dist = 36 * pip
    if direction == "BUY":
        sl, tp = price - sl_dist, price + tp_dist
    else:
        sl, tp = price + sl_dist, price - tp_dist

    return SignalResult("ema_stack", direction, conf, price, sl, tp,
                        {"ema8": e8, "ema21": e21, "ema55": e55})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fair Value Gap (FVG) Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_fvg(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Bullish FVG: candle[i].low > candle[i+2].high  (gap left above)
    Bearish FVG: candle[i].high < candle[i+2].low  (gap left below)
    Price then pulls back INTO the gap → high-probability continuation.
    """
    highs  = _highs(candles)
    lows   = _lows(candles)
    closes = _closes(candles)
    if len(closes) < 10 or len(highs) < 10 or len(lows) < 10:
        return None

    price = closes[-1]
    pip   = _pip_size(symbol)
    best: Optional[SignalResult] = None

    # Scan last 20 candles for FVG zones, pick most recent unfilled one
    n = min(len(closes), 20)
    for i in range(len(closes) - n, len(closes) - 2):
        h0, l0 = highs[i], lows[i]
        h2, l2 = highs[i + 2], lows[i + 2]

        # Bullish FVG
        if l0 > h2:
            gap_top, gap_bot = l0, h2
            gap_size = gap_top - gap_bot
            # Price is retesting inside the gap
            if gap_bot <= price <= gap_top:
                # Strength = gap size relative to pip
                strength = gap_size / pip
                conf = min(0.85, 0.55 + strength * 0.01)
                sl = gap_bot - 3 * pip
                tp = price + 3 * gap_size   # ~3R extension
                r = SignalResult("fvg", "BUY", conf, price, sl, tp,
                                 {"gap_top": gap_top, "gap_bot": gap_bot,
                                  "gap_pips": round(strength, 1), "type": "bullish_fvg"})
                if best is None or conf > best.confidence:
                    best = r

        # Bearish FVG
        elif h0 < l2:
            gap_top, gap_bot = l2, h0
            gap_size = gap_top - gap_bot
            if gap_bot <= price <= gap_top:
                strength = gap_size / pip
                conf = min(0.85, 0.55 + strength * 0.01)
                sl = gap_top + 3 * pip
                tp = price - 3 * gap_size
                r = SignalResult("fvg", "SELL", conf, price, sl, tp,
                                 {"gap_top": gap_top, "gap_bot": gap_bot,
                                  "gap_pips": round(strength, 1), "type": "bearish_fvg"})
                if best is None or conf > best.confidence:
                    best = r

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fibonacci Retracement Detector
# ─────────────────────────────────────────────────────────────────────────────

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
KEY_FIBS   = {0.500, 0.618, 0.786}   # higher bonus


def detect_fibonacci(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Identify swing H/L over last 50 candles.
    Check if current price is within tolerance of a key fib retracement.
    Use EMA55 trend bias to determine direction.
    """
    highs  = _highs(candles)[-50:]
    lows   = _lows(candles)[-50:]
    closes = _closes(candles)[-60:]
    if len(closes) < 55:
        return None

    price  = closes[-1]
    pip    = _pip_size(symbol)
    sh     = _swing_high(highs, 50)
    sl_    = _swing_low(lows, 50)
    swing  = sh - sl_
    if swing < 10 * pip:   # too small a swing to be meaningful
        return None

    # Trend bias from EMA55
    e55 = _ema(closes, 55)
    trend_up = price > e55

    tol = 0.003   # within 0.3% of price counts as "at fib"

    best_dist = float("inf")
    best_lvl  = None
    best_fib_price = None

    for lvl in FIB_LEVELS:
        # In uptrend: retracement = bull pull-back = sh - swing*lvl
        # In downtrend: retracement = bear pull-back = sl_ + swing*lvl
        if trend_up:
            fib_price = sh - swing * lvl
        else:
            fib_price = sl_ + swing * lvl

        dist = abs(price - fib_price) / price
        if dist < best_dist:
            best_dist = dist
            best_lvl  = lvl
            best_fib_price = fib_price

    if best_dist > tol:
        return None

    direction = "BUY" if trend_up else "SELL"
    proximity_score = 1.0 - (best_dist / tol)
    bonus = 0.15 if best_lvl in KEY_FIBS else 0.0
    conf  = min(0.90, 0.55 + proximity_score * 0.25 + bonus)

    if direction == "BUY":
        sl = sl_ - 3 * pip
        tp = sh + swing * 0.272    # extension toward 1.272
    else:
        sl = sh + 3 * pip
        tp = sl_ - swing * 0.272

    return SignalResult("fibonacci", direction, conf, price, sl, tp,
                        {"fib_level": best_lvl, "fib_price": round(best_fib_price, 5),
                         "swing_high": round(sh, 5), "swing_low": round(sl_, 5),
                         "distance_pct": round(best_dist * 100, 3)})


# ─────────────────────────────────────────────────────────────────────────────
# 5. Liquidity Sweep Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_liquidity_sweep(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Equal highs (within 2 pips) swept then reversed → SELL.
    Equal lows swept then recovered → BUY.
    Requires a bullish/bearish close AFTER the sweep candle.
    """
    highs  = _highs(candles)
    lows   = _lows(candles)
    closes = _closes(candles)
    if len(closes) < 15:
        return None

    pip   = _pip_size(symbol)
    price = closes[-1]
    tol   = 2 * pip

    # Check last 3 candles for sweep
    h3 = highs[-4:-1]
    l3 = lows[-4:-1]
    c3 = closes[-4:-1]

    # Equal-highs sweep → bear reversal
    max_h = max(h3)
    prev_h_avg = sum(h3[:-1]) / max(len(h3[:-1]), 1)
    if abs(max_h - prev_h_avg) < tol and highs[-1] > max_h and closes[-1] < highs[-1]:
        # wick spiked above equal highs then closed lower → sweep + reversal
        conf = 0.72
        sl   = highs[-1] + 3 * pip
        tp   = price - (sl - price) * 3
        return SignalResult("liq_sweep", "SELL", conf, price, sl, tp,
                            {"type": "equal_highs_sweep", "swept": round(max_h, 5)})

    # Equal-lows sweep → bull reversal
    min_l = min(l3)
    prev_l_avg = sum(l3[:-1]) / max(len(l3[:-1]), 1)
    if abs(min_l - prev_l_avg) < tol and lows[-1] < min_l and closes[-1] > lows[-1]:
        conf = 0.72
        sl   = lows[-1] - 3 * pip
        tp   = price + (price - sl) * 3
        return SignalResult("liq_sweep", "BUY", conf, price, sl, tp,
                            {"type": "equal_lows_sweep", "swept": round(min_l, 5)})

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Trap Reversal (Pin Bar / Engulfing) Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_trap_reversal(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Bearish pin bar at recent swing high → SELL trap.
    Bullish pin bar at recent swing low → BUY trap.
    Also catches bullish/bearish engulfing.
    """
    highs  = _highs(candles)
    lows   = _lows(candles)
    closes = _closes(candles)
    if len(closes) < 20:
        return None

    pip   = _pip_size(symbol)
    price = closes[-1]

    # Get last candle OHLC
    c1 = candles[-1]
    if not isinstance(c1, dict):
        return None

    def _v(c, keys) -> float:
        for k in keys:
            v = c.get("mid", {}).get(k) or c.get(k)
            if v:
                return float(v)
        return 0.0

    o1 = _v(c1, ["o", "open"])
    h1 = _v(c1, ["h", "high"])
    l1 = _v(c1, ["l", "low"])
    c_close = closes[-1]
    if not all([o1, h1, l1, c_close]):
        return None

    body = abs(c_close - o1)
    full_range = h1 - l1 if h1 > l1 else 1e-9
    upper_wick  = h1 - max(o1, c_close)
    lower_wick  = min(o1, c_close) - l1
    body_ratio  = body / full_range

    sh = _swing_high(highs, 20)
    sl_ = _swing_low(lows, 20)
    at_high = abs(price - sh) < 10 * pip
    at_low  = abs(price - sl_) < 10 * pip

    # Bearish pin bar at high
    if at_high and upper_wick > 2 * body and body_ratio < 0.35:
        conf = 0.68 + min(0.15, (upper_wick / full_range) * 0.3)
        sl   = h1 + 3 * pip
        tp   = price - (sl - price) * 3
        return SignalResult("trap_reversal", "SELL", conf, price, sl, tp,
                            {"pattern": "bearish_pin_bar",
                             "upper_wick_pips": round(upper_wick / pip, 1)})

    # Bullish pin bar at low
    if at_low and lower_wick > 2 * body and body_ratio < 0.35:
        conf = 0.68 + min(0.15, (lower_wick / full_range) * 0.3)
        sl   = l1 - 3 * pip
        tp   = price + (price - sl) * 3
        return SignalResult("trap_reversal", "BUY", conf, price, sl, tp,
                            {"pattern": "bullish_pin_bar",
                             "lower_wick_pips": round(lower_wick / pip, 1)})

    # Bullish engulfing
    if len(candles) >= 2:
        prev_c = closes[-2]
        prev_h = highs[-2] if len(highs) >= 2 else 0
        if (c_close > o1                         # bullish close
                and o1 < prev_c                  # opened below prior close
                and c_close > prev_h             # closed above prior high
                and at_low):
            conf = 0.70
            sl   = l1 - 3 * pip
            tp   = price + (price - sl) * 3
            return SignalResult("trap_reversal", "BUY", conf, price, sl, tp,
                                {"pattern": "bullish_engulfing"})

    # Bearish engulfing
    if len(candles) >= 2:
        prev_c = closes[-2]
        prev_l = lows[-2] if len(lows) >= 2 else float("inf")
        if (c_close < o1                         # bearish close
                and o1 > prev_c                  # opened above prior close
                and c_close < prev_l             # closed below prior low
                and at_high):
            conf = 0.70
            sl   = h1 + 3 * pip
            tp   = price - (sl - price) * 3
            return SignalResult("trap_reversal", "SELL", conf, price, sl, tp,
                                {"pattern": "bearish_engulfing"})

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. RSI Divergence + Oversold/Overbought
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(closes: list[float], n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    gains, losses = [], []
    for i in range(len(closes) - n, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = sum(gains) / n; al = sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - 100 / (1 + rs)


def detect_rsi_extremes(symbol: str, candles: list) -> Optional[SignalResult]:
    closes = _closes(candles)[-80:]
    if len(closes) < 20:
        return None
    rsi_now  = _rsi(closes, 14)
    rsi_prev = _rsi(closes[:-5], 14)
    pip      = _pip_size(symbol)
    price    = closes[-1]

    if rsi_now < 28:  # oversold
        # divergence bonus: price lower but RSI higher
        div_bonus = 0.10 if rsi_now > rsi_prev else 0.0
        conf = min(0.80, 0.55 + (28 - rsi_now) * 0.01 + div_bonus)
        sl   = price - 12 * pip
        tp   = price + 36 * pip
        return SignalResult("rsi_extreme", "BUY", conf, price, sl, tp,
                            {"rsi": round(rsi_now, 2), "type": "oversold",
                             "divergence": div_bonus > 0})

    if rsi_now > 72:  # overbought
        div_bonus = 0.10 if rsi_now < rsi_prev else 0.0
        conf = min(0.80, 0.55 + (rsi_now - 72) * 0.01 + div_bonus)
        sl   = price + 12 * pip
        tp   = price - 36 * pip
        return SignalResult("rsi_extreme", "SELL", conf, price, sl, tp,
                            {"rsi": round(rsi_now, 2), "type": "overbought",
                             "divergence": div_bonus > 0})

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Mean Reversion (Bollinger Bands) Detector
# ─────────────────────────────────────────────────────────────────────────────

def _stddev(seq: list[float], mean: float) -> float:
    if not seq:
        return 0.0
    var = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(var)

def detect_mean_reversion_bb(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Mean Reversion using Bollinger Bands.
    Buy when price drops below lower BB and starts rising.
    Sell when price spikes above upper BB and starts dropping.
    """
    closes = _closes(candles)[-22:]
    if len(closes) < 21:
        return None
        
    current_price = closes[-1]
    prev_price = closes[-2]
    
    # Calculate BB for the previous candle to check for piercing
    basis_seq = closes[-21:-1]
    basis = _sma(basis_seq, 20)
    dev = _stddev(basis_seq, basis)
    upper_bb = basis + (2.0 * dev)
    lower_bb = basis - (2.0 * dev)
    
    pip = _pip_size(symbol)
    
    direction = None
    conf = 0.0
    
    # Buy signal: previous candle pierced lower band, current price is moving up
    if prev_price <= lower_bb and current_price > prev_price:
        direction = "BUY"
        # Confidence scales with how far outside the bands it went
        pierce_dist = lower_bb - prev_price
        conf = min(0.85, 0.60 + (pierce_dist / (10 * pip)) * 0.05)
        
    # Sell signal: previous candle pierced upper band, current price is moving down
    elif prev_price >= upper_bb and current_price < prev_price:
        direction = "SELL"
        pierce_dist = prev_price - upper_bb
        conf = min(0.85, 0.60 + (pierce_dist / (10 * pip)) * 0.05)
        
    if not direction:
        return None
        
    # Targets for mean reversion: TP is the middle band (basis), SL is tight
    dist_to_basis = abs(current_price - basis)
    if dist_to_basis < 8 * pip: 
        return None # Not enough meat on the bone
        
    sl_dist = 12 * pip
    # Minimum TP distance for Charter 3.2:1 R:R — use 3.5x for margin
    min_tp_dist = sl_dist * 3.5
    if direction == "BUY":
        sl = current_price - sl_dist
        tp = max(basis, current_price + min_tp_dist)  # extend beyond basis if needed
    else:
        sl = current_price + sl_dist
        tp = min(basis, current_price - min_tp_dist)  # extend beyond basis if needed
        
    return SignalResult("mean_reversion_bb", direction, conf, current_price, sl, tp,
                        {"upper_bb": upper_bb, "lower_bb": lower_bb, "basis": basis})

# ─────────────────────────────────────────────────────────────────────────────
# 9. Aggressive Shorting (Order Block) Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_aggressive_shorting_ob(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    Aggressively hunts for Bearish Order Blocks to short into.
    Looks for a rapid down-move (displacement) breaking structure, then shorts the retracement.
    """
    highs = _highs(candles)
    lows = _lows(candles)
    closes = _closes(candles)
    
    if len(closes) < 30 or len(highs) < 30 or len(lows) < 30:
        return None
        
    pip = _pip_size(symbol)
    price = closes[-1]
    
    # Look back 20 candles for a large bearish displacement
    best_ob_high = 0.0
    best_ob_low = 0.0
    highest_displacement = 0.0
    
    n_candles = len(closes)
    for i in range(n_candles - 25, n_candles - 5):
        # Look for a strong down candle
        c = candles[i]
        c_open_val = None
        if isinstance(c, dict):
            c_open_val = c.get("mid", {}).get("o") or c.get("o")
        
        if not c_open_val:
            continue
        c_open = float(c_open_val)
        c_close = closes[i]
        
        # Check if it's a strong bear candle
        if c_close < c_open:
            body = c_open - c_close
            
            # Require minimum 15 pip displacement
            if body > 15 * pip and body > highest_displacement:
                # The Order Block is the last up-candle before this down-move
                # Try finding it in the 3 preceding candles
                for j in range(1, 4):
                    if i-j >= 0:
                        prev_c = candles[i-j]
                        j_open_raw = None
                        if isinstance(prev_c, dict):
                            j_open_raw = prev_c.get("mid", {}).get("o") or prev_c.get("o")
                        
                        if j_open_raw:
                            j_open = float(j_open_raw)
                            j_close = closes[i-j]
                            
                            if j_close > j_open: # Bullish candle = OB
                                best_ob_high = float(highs[i-j])
                                best_ob_low = float(lows[i-j])
                                highest_displacement = body
                                break
                                
    # If we found a valid Bearish Order Block and price is retracing into it
    if best_ob_high > 0 and best_ob_low <= price <= best_ob_high:
        conf = 0.88 # Very high confidence for OB retracements
        sl = best_ob_high + 5 * pip # Stop just above the OB
        tp = price - (highest_displacement * 1.5) # Target 1.5x the displacement 
        
        return SignalResult("aggressive_short_ob", "SELL", conf, price, sl, tp,
                            {"ob_high": best_ob_high, "ob_low": best_ob_low, "displacement": highest_displacement})
                            
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 10. EMA 100/200 Scalper — Dec 1 Backtest #1 Performer (Sharpe 3.32)
# Requires EMA100 to cross EMA200 with price momentum confirmation.
# ATR-based SL/TP gives adaptive sizing across all volatility regimes.
# ─────────────────────────────────────────────────────────────────────────────

def _atr(candles: list, n: int = 14) -> float:
    """Average True Range over last n candles."""
    highs  = _highs(candles)
    lows   = _lows(candles)
    closes = _closes(candles)
    if len(closes) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs)


def detect_ema_scalper_200(symbol: str, candles: list) -> Optional[SignalResult]:
    """
    EMA 100/200 crossover scalper — backtested #1 OANDA performer.
    Entry: EMA100 crosses above/below EMA200, with price momentum.
    SL: 1.5 × ATR14.  TP: 1.5 × ATR14 × 4.0  (4:1 R:R, backtest-optimal).
    Requires at least 220 candles for reliable 200-period EMA.
    """
    closes = _closes(candles)
    if len(closes) < 220:
        return None

    price  = closes[-1]
    e100_n = _ema(closes, 100)           # current  EMA100
    e100_p = _ema(closes[:-1], 100)      # previous EMA100
    e200   = _ema(closes, 200)
    atr    = _atr(candles, 14)
    pip    = _pip_size(symbol)

    if atr < pip * 3:                    # too quiet — skip
        return None

    crossed_up   = e100_p <= e200 < e100_n   # EMA100 just crossed above EMA200
    crossed_down = e100_p >= e200 > e100_n   # EMA100 just crossed below EMA200

    # Also accept being in a clear trend (EMA100 > EMA200 + momentum)
    roc = _roc(closes, 10)
    trending_up   = e100_n > e200 and roc >  0.10
    trending_down = e100_n < e200 and roc < -0.10

    if crossed_up or trending_up:
        direction = "BUY"
        sep_bonus = min(0.10, ((e100_n - e200) / e200) * 200)  # bonus for separation
        conf = min(0.88, 0.68 + sep_bonus + (0.05 if crossed_up else 0.0))
    elif crossed_down or trending_down:
        direction = "SELL"
        sep_bonus = min(0.10, ((e200 - e100_n) / e200) * 200)
        conf = min(0.88, 0.68 + sep_bonus + (0.05 if crossed_down else 0.0))
    else:
        return None

    sl_dist = atr * 1.5
    tp_dist = atr * 1.5 * 4.0     # 4:1 R:R — backtest optimal
    if direction == "BUY":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist

    return SignalResult(
        "ema_scalper_200", direction, conf, price, sl, tp,
        {"ema100": round(e100_n, 5), "ema200": round(e200, 5),
         "atr": round(atr, 5), "crossed": crossed_up or crossed_down},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Signal Aggregator
# ─────────────────────────────────────────────────────────────────────────────

DETECTORS = [
    detect_momentum_sma,
    detect_ema_stack,
    detect_fvg,
    detect_fibonacci,
    detect_liquidity_sweep,
    detect_trap_reversal,
    detect_rsi_extremes,
    detect_mean_reversion_bb,
    detect_aggressive_shorting_ob,
    detect_ema_scalper_200,       # 10th: Dec 1 backtested #1 performer
]

# Minimum individual confidence to count as a "vote" (raised from 0.55 → 0.62)
# Each detector must individually score ≥0.62 to cast a vote.
_MIN_VOTE_CONF = 0.62
# Minimum votes in the same direction to fire a trade (raised from 2 → 3)
# Requires 3 of 9 detectors independently agreeing — restores Dec 10 standard.
_MIN_VOTES = 3


class AggregatedSignal:
    """Final result returned to the trading engine."""
    def __init__(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        votes: int,
        detectors_fired: List[str],
        all_results: List[SignalResult],
        session: str,
        session_mult: float,
        signal_type: str = "trend",   # FIX #4: "trend" | "mean_reversion" | "reversal"
    ):
        self.symbol         = symbol
        self.direction      = direction
        self.confidence     = round(confidence, 4)
        self.entry          = entry
        self.sl             = sl
        self.tp             = tp
        # ── Store raw pip distances so place_trade can rebase to live price ──
        self.risk_dist      = abs(sl - entry)    # distance entry→SL in price
        self.reward_dist    = abs(tp - entry)    # distance entry→TP in price
        self.rr             = self.reward_dist / self.risk_dist if self.risk_dist > 1e-9 else 0.0
        self.votes          = votes
        self.detectors_fired = detectors_fired
        self.all_results    = all_results
        self.session        = session
        self.session_mult   = session_mult
        self.signal_type    = signal_type   # FIX #4: used by manage_open_trade

    def as_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "direction":       self.direction,
            "confidence":      self.confidence,
            "entry":           self.entry,
            "sl":              self.sl,
            "tp":              self.tp,
            "rr":              round(self.rr, 2),
            "votes":           self.votes,
            "detectors":       self.detectors_fired,
            "session":         self.session,
            "session_mult":    self.session_mult,
            "signal_type":     self.signal_type,
        }


def scan_symbol(
    symbol: str,
    candles: list,
    utc_now: Optional[datetime] = None,
    min_confidence: float = 0.75,
    min_votes: int = _MIN_VOTES,
) -> Optional[AggregatedSignal]:
    """
    Run all detectors on `candles` for `symbol`.
    Return the best AggregatedSignal if enough detectors agree,
    or None if there is no tradeable setup.

    The winning direction is decided by VOTE COUNT first,
    then TIE-BREAK by highest confidence sum.
    """
    session_name, session_mult = session_bias(symbol, utc_now)

    # Run every detector, collect results
    all_results: List[SignalResult] = []
    for fn in DETECTORS:
        try:
            r = fn(symbol, candles)
            if r and r.confidence >= _MIN_VOTE_CONF and r.direction:
                all_results.append(r)
        except Exception:
            pass  # one detector failing must never kill the scan

    if not all_results:
        return None

    # Group by direction
    by_dir: Dict[str, List[SignalResult]] = {}
    for r in all_results:
        by_dir.setdefault(r.direction, []).append(r)

    # Pick direction with most votes, tie-break by sum of confidence
    best_dir = max(by_dir, key=lambda d: (len(by_dir[d]),
                                          sum(r.confidence for r in by_dir[d])))
    voted = by_dir[best_dir]
    if len(voted) < min_votes:
        return None

    # Aggregate SL/TP:
    #   SL: FIX #3 — use MEDIAN SL (not worst-case) so early BE trigger isn't
    #       made noisier by one wide-SL detector dragging the group
    #   TP: most aggressive (furthest from entry) for full reward capture
    #   Entry: last close (market order)
    entry = voted[0].entry  # all share same candle close
    if best_dir == "BUY":
        # FIX #3: median of SLs, not the minimum (furthest away)
        sls_sorted = sorted(r.sl for r in voted)
        sl = sls_sorted[len(sls_sorted) // 2]   # median SL
        tp = max(r.tp for r in voted)            # highest TP aims highest
    else:
        sls_sorted = sorted((r.sl for r in voted), reverse=True)
        sl = sls_sorted[len(sls_sorted) // 2]   # median SL for short
        tp = min(r.tp for r in voted)            # lowest TP aims lowest

    # FIX #4: classify signal type so trade manager applies correct logic
    # Mean reversion alone → "mean_reversion"; reversal detectors only → "reversal"
    # Any trend detector in the mix → "trend" (trend overrides)
    _TREND_DETECTORS = {"momentum_sma", "ema_stack", "fibonacci", "fvg"}
    _REVERSAL_DETECTORS = {"liq_sweep", "trap_reversal"}
    fired_set = {r.detector for r in voted}
    if fired_set & _TREND_DETECTORS:
        signal_type = "trend"
    elif fired_set & _REVERSAL_DETECTORS:
        signal_type = "reversal"
    elif "mean_reversion_bb" in fired_set:
        signal_type = "mean_reversion"
        # For pure mean reversion: TP is closer (to the middle band),
        # so use the most conservative TP to avoid over-extending
        if best_dir == "BUY":
            tp = min(r.tp for r in voted)
        else:
            tp = max(r.tp for r in voted)
    else:
        signal_type = "trend"  # safe default

    # Weighted confidence = mean of voting detectors × session multiplier
    raw_conf = sum(r.confidence for r in voted) / len(voted)
    final_conf = min(0.99, raw_conf * session_mult)

    if final_conf < min_confidence:
        return None

    # ── ML Regime Gate (Dec 10 standard) ──────────────────────────────────────
    # Trending regimes: pass at standard min_confidence (0.68)
    # Sideways/Unknown: require higher bar (0.72) to block ranging chop
    # Model falls back gracefully if no .pkl or import fails.
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from ml_learning.regime_detector import RegimeDetector, MarketRegime
        _rd = getattr(scan_symbol, '_regime_detector', None)
        if _rd is None:
            _rd = RegimeDetector()
            scan_symbol._regime_detector = _rd          # cache on function
        regime = _rd.detect_regime(candles[-60:])
        regime_conf = _rd.get_confidence()
        _is_sideways = regime in (MarketRegime.SIDEWAYS, MarketRegime.UNKNOWN)
        _regime_threshold = 0.72 if _is_sideways else min_confidence
        if final_conf < _regime_threshold:
            return None   # blocked by ML regime gate
        # Attach regime metadata to signal for logging
        _regime_label = regime.value if hasattr(regime, 'value') else str(regime)
    except Exception:
        _regime_label = "unknown"
        # Regime gate failed — allow the signal through (fail open)

    return AggregatedSignal(
        symbol          = symbol,
        direction       = best_dir,
        confidence      = final_conf,
        entry           = entry,
        sl              = sl,
        tp              = tp,
        votes           = len(voted),
        detectors_fired = [r.detector for r in voted],
        all_results     = all_results,
        session         = session_name,
        session_mult    = session_mult,
        signal_type     = signal_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Position Management
# ─────────────────────────────────────────────────────────────────────────────

class TradeAction:
    HOLD           = "HOLD"
    SCALE_OUT_HALF = "SCALE_OUT_HALF"   # close 50% at configured R
    TRAIL_TIGHT    = "TRAIL_TIGHT"      # tighten trail at configured R
    CLOSE_ALL      = "CLOSE_ALL"        # full exit
    MOVE_BE        = "MOVE_BE"          # move SL to breakeven


def manage_open_trade(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    current_price: float,
    symbol: str,
    scaled_out: bool = False,
    trail_active: bool = False,
    session: Optional[str] = None,
    signal_type: str = "trend",   # FIX #4: "trend"|"mean_reversion"|"reversal"
) -> Tuple[str, Dict[str, Any]]:
    """
    Given current price on an open trade, returns (action, detail_dict).

        Logic:
            • BE threshold R profit     → Move SL to breakeven
            • SCALE threshold R profit  → Scale out 50% (if not already done)
            • TRAIL threshold R profit  → Activate tight trail (trail at TRAIL_DIST_R × risk)
            • Price crosses back through prior trail level → Close all
            • Session ends (off_session) → Close all (prevent overnight drift)

        FIX #1: R thresholds raised to be above M15 noise level:
            BE: 0.30R → 0.75R  |  Scale: 1.0R → 1.5R  |  Trail: 1.25R → 2.0R
        FIX #4: mean_reversion trades use tighter trail (1.0R) — they don't run far
    """
    # FIX #1: raised defaults — old values caused BE at 3 pips (M15 noise)
    be_r_threshold    = float(os.getenv("RBOT_BE_TRIGGER_R",         "0.75"))  # was 0.30
    scale_r_threshold = float(os.getenv("RBOT_SCALE_OUT_TRIGGER_R",  "1.50"))  # was 1.00
    trail_r_threshold = float(os.getenv("RBOT_TRAIL_TRIGGER_R",      "2.00"))  # was 1.25
    trail_dist_r      = float(os.getenv("RBOT_TRAIL_DISTANCE_R",     "0.40"))  # was 0.25

    # FIX #4: mean_reversion trades don't run far — tighten their trail trigger
    if signal_type == "mean_reversion":
        trail_r_threshold = min(trail_r_threshold, 1.00)
        scale_r_threshold = min(scale_r_threshold, 0.80)

    profit_extraction_mode = os.getenv("RBOT_PROFIT_EXTRACTION_MODE", "1") == "1"
    session_name = str(session or "").lower()

    if profit_extraction_mode:
        if session_name in {"overlap", "london", "new_york"}:
            be_r_threshold *= 0.85
            scale_r_threshold *= 0.85
            trail_r_threshold *= 0.90
            trail_dist_r *= 0.90
        elif session_name == "tokyo":
            be_r_threshold *= 0.90
            scale_r_threshold *= 0.90
            trail_r_threshold *= 0.95
        elif session_name == "off_session":
            be_r_threshold *= 0.75
            scale_r_threshold *= 0.80
            trail_r_threshold *= 0.85
            trail_dist_r *= 0.85

    be_r_threshold = max(0.05, be_r_threshold)
    scale_r_threshold = max(be_r_threshold, scale_r_threshold)
    trail_r_threshold = max(scale_r_threshold, trail_r_threshold)
    trail_dist_r = max(0.05, min(trail_dist_r, 1.0))

    risk = abs(entry - sl)
    if risk < 1e-9:
        return (TradeAction.HOLD, {})

    if direction == "BUY":
        pnl_r = (current_price - entry) / risk
    else:
        pnl_r = (entry - current_price) / risk

    detail: Dict[str, Any] = {
        "pnl_r": round(pnl_r, 3),
        "current": current_price,
        "entry": entry,
        "sl": sl,
        "be_r_threshold": round(be_r_threshold, 3),
        "scale_r_threshold": round(scale_r_threshold, 3),
        "trail_r_threshold": round(trail_r_threshold, 3),
        "trail_dist_r": round(trail_dist_r, 3),
        "profit_extraction_mode": profit_extraction_mode,
        "session": session_name or "unknown",
    }

    # Session ended → protect profits
    if session == "off_session" and pnl_r > be_r_threshold:
        detail["reason"] = "session_ended_with_profit"
        return (TradeAction.CLOSE_ALL, detail)

    # Already losing beyond SL (staleness guard)
    if pnl_r < -1.05:
        detail["reason"] = "stop_breach"
        return (TradeAction.CLOSE_ALL, detail)

    # Configured R+ → tight trail (with progressive tightening)
    if pnl_r >= trail_r_threshold:
        try:
            from util.momentum_trailing import get_trailing_distance
            
            # Use original risk as a proxy for ATR in this framework
            dynamic_trail_dist = get_trailing_distance(pnl_r, risk, momentum_active=False)
            candidate_sl = current_price - dynamic_trail_dist if direction == "BUY" else current_price + dynamic_trail_dist
            
            # Prevent API spam: only update if SL improves by at least 10% of risk space
            material_improvement = False
            if direction == "BUY" and candidate_sl > (sl + 0.1 * risk):
                material_improvement = True
            elif direction == "SELL" and candidate_sl < (sl - 0.1 * risk):
                material_improvement = True
                
            if material_improvement or not trail_active:
                detail["reason"] = f"dynamic_trail_{pnl_r:.1f}R"
                detail["new_sl"] = candidate_sl
                return (TradeAction.TRAIL_TIGHT, detail)
                
        except ImportError:
            # Fallback: Static trail if module missing
            if not trail_active:
                detail["reason"] = f"{trail_r_threshold:.2f}R_trail_activate"
                detail["new_sl"] = current_price - trail_dist_r * risk if direction == "BUY" \
                                   else current_price + trail_dist_r * risk
                return (TradeAction.TRAIL_TIGHT, detail)

    # Configured R → scale out
    if pnl_r >= scale_r_threshold and not scaled_out:
        detail["reason"] = f"{scale_r_threshold:.2f}R_scale_out"
        return (TradeAction.SCALE_OUT_HALF, detail)

    # Configured R → move to breakeven
    if pnl_r >= be_r_threshold and abs(sl - entry) > 1e-9:
        detail["reason"] = f"{be_r_threshold:.2f}R_breakeven"
        detail["new_sl"] = entry
        return (TradeAction.MOVE_BE, detail)

    return (TradeAction.HOLD, detail)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — drop-in replacement for momentum_signals.generate_signal
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(symbol: str, candles: list):
    """
    Backward-compatible wrapper.
    Returns (signal, confidence, meta) 3-tuple.
    """
    result = scan_symbol(symbol, candles)
    if result is None:
        return (None, 0.0, {"reason": "no_edge"})
    return (
        result.direction,
        result.confidence,
        {
            "detectors": result.detectors_fired,
            "votes":     result.votes,
            "session":   result.session,
            "rr":        round(result.rr, 2),
            "sl":        result.sl,
            "tp":        result.tp,
        },
    )
