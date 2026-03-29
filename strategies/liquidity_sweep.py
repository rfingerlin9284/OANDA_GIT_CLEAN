from __future__ import annotations
# strategies/liquidity_sweep.py
# RBOTZILLA_OANDA_CLEAN
# Label: EXTRACTED_UNVERIFIED (copied from Phoenix — no logic changes)
from typing import Optional

from strategies.base import BaseStrategy, StrategyContext, ProposedTrade

# ── helpers ───────────────────────────────────────────────────────────────────

def _f(c, *keys) -> float:
    for k in keys:
        v = c.get(k)
        if v is not None:
            return float(v)
    return 0.0

def _ohlc(candles):
    H, L, C = [], [], []
    for c in candles:
        if not isinstance(c, dict): continue
        H.append(_f(c, "high", "h")); L.append(_f(c, "low", "l")); C.append(_f(c, "close", "c"))
    return H, L, C

def _pip(sym): return 0.01 if "JPY" in sym.upper() else 0.0001

def _ema(seq, n):
    if not seq: return 0.0
    k = 2.0 / (n + 1); v = seq[0]
    for p in seq[1:]: v = p * k + v * (1 - k)
    return v

def _eq_highs(highs, lookback, tol):
    w = highs[-lookback:]
    for i in range(len(w)-1):
        for j in range(i+1, len(w)):
            if abs(w[i]-w[j]) <= tol:
                return (w[i]+w[j])/2.0
    return None

def _eq_lows(lows, lookback, tol):
    w = lows[-lookback:]
    for i in range(len(w)-1):
        for j in range(i+1, len(w)):
            if abs(w[i]-w[j]) <= tol:
                return (w[i]+w[j])/2.0
    return None

# ── strategy ─────────────────────────────────────────────────────────────────

class LiquiditySweepReversalStrategy(BaseStrategy):
    """
    Liquidity Sweep + Market Structure Shift — computed from raw candles.

    SELL: cluster of equal highs detected → last candle wicked ABOVE and closed
          BELOW the level (sweep + rejection) + EMA21 downward bias.
    BUY : mirror — equal lows swept, closed back above, EMA21 upward.

    SL: beyond the wick  |  TP: target_rr × risk
    """

    def decide_entry(self, ctx: StrategyContext) -> Optional[ProposedTrade]:
        if ctx.timeframe not in self.metadata.base_timeframes:
            return None
        if len(ctx.candles) < 35:
            return None

        H, L, C = _ohlc(ctx.candles)
        pip   = _pip(ctx.symbol)
        tol   = 3 * pip
        price = C[-1]
        ema21 = _ema(C[-50:], 21)

        # ── SELL ──────────────────────────────────────────────────────────────
        eq_h = _eq_highs(H[:-1], 30, tol)
        if eq_h and price < ema21:
            if H[-1] > eq_h and C[-1] < eq_h:
                sl   = H[-1] + 3 * pip
                risk = sl - price
                if risk <= 0: return None
                tp   = price - risk * self.metadata.target_rr
                conf = min(0.85, 0.70 + (H[-1] - eq_h) / (10*pip) * 0.05)
                return ProposedTrade(
                    strategy_code=self.metadata.code, symbol=ctx.symbol,
                    direction="SELL", entry_type="market", entry_price=None,
                    stop_loss_price=round(sl, 5), take_profit_price=round(tp, 5),
                    target_rr=self.metadata.target_rr, confidence=conf,
                    notes={"reason": "equal_highs_swept", "eq_level": round(eq_h,5),
                           "wick_high": round(H[-1],5), "ema21": round(ema21,5)})

        # ── BUY ───────────────────────────────────────────────────────────────
        eq_l = _eq_lows(L[:-1], 30, tol)
        if eq_l and price > ema21:
            if L[-1] < eq_l and C[-1] > eq_l:
                sl   = L[-1] - 3 * pip
                risk = price - sl
                if risk <= 0: return None
                tp   = price + risk * self.metadata.target_rr
                conf = min(0.85, 0.70 + (eq_l - L[-1]) / (10*pip) * 0.05)
                return ProposedTrade(
                    strategy_code=self.metadata.code, symbol=ctx.symbol,
                    direction="BUY", entry_type="market", entry_price=None,
                    stop_loss_price=round(sl, 5), take_profit_price=round(tp, 5),
                    target_rr=self.metadata.target_rr, confidence=conf,
                    notes={"reason": "equal_lows_swept", "eq_level": round(eq_l,5),
                           "wick_low": round(L[-1],5), "ema21": round(ema21,5)})

        return None
