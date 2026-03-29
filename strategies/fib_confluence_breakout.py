from __future__ import annotations
# strategies/fib_confluence_breakout.py
# RBOTZILLA_OANDA_CLEAN
# Label: EXTRACTED_UNVERIFIED (copied from Phoenix — import path only change)
from typing import Optional

from strategies.base import BaseStrategy, StrategyContext, ProposedTrade

def _f(c, *keys) -> float:
    for k in keys:
        v = c.get(k)
        if v is not None: return float(v)
    return 0.0

def _ohlc(candles):
    H, L, C = [], [], []
    for c in candles:
        if not isinstance(c, dict): continue
        H.append(_f(c,"high","h")); L.append(_f(c,"low","l")); C.append(_f(c,"close","c"))
    return H, L, C

def _pip(sym): return 0.01 if "JPY" in sym.upper() else 0.0001
def _ema(seq, n):
    if not seq: return 0.0
    k = 2.0/(n+1); v = seq[0]
    for p in seq[1:]: v = p*k + v*(1-k)
    return v

FIB_LEVELS   = [0.236, 0.382, 0.500, 0.618, 0.786]
FIB_EXTS     = [1.272, 1.618]
KEY_FIBS     = {0.500, 0.618, 0.786}   # golden pocket + confluence bonus


class FibConfluenceBreakoutStrategy(BaseStrategy):
    """
    Fibonacci Retracement Confluence — computed from raw candles.

    1. Find 50-bar swing high and swing low.
    2. Determine trend via EMA55 (price above → uptrend → look for BUY on pull-back).
    3. Compute fib retracement levels (0.236 – 0.786).
    4. If current price is within ±0.3% of a key level AND closing back in
       the trend direction after a pull-back → enter.
    5. SL just beyond the next deeper fib.  TP at 1.272 extension.

    Confidence bonus for 0.500, 0.618, 0.786 (golden pocket).
    """

    def decide_entry(self, ctx: StrategyContext) -> Optional[ProposedTrade]:
        if ctx.timeframe not in self.metadata.base_timeframes:
            return None
        if len(ctx.candles) < 60:
            return None

        H, L, C = _ohlc(ctx.candles)
        pip   = _pip(ctx.symbol)
        price = C[-1]
        ema55 = _ema(C[-70:], 55)

        # Swing reference over last 50 bars
        sh  = max(H[-50:])
        sl_ = min(L[-50:])
        swing = sh - sl_
        if swing < 10 * pip:
            return None                 # range too tight

        trend_up = price > ema55

        tol = 0.003                     # 0.3% proximity to fib level
        best_dist = float("inf")
        best_lvl  = None
        best_fp   = None

        for lvl in FIB_LEVELS:
            fp = sh - swing * lvl if trend_up else sl_ + swing * lvl
            d  = abs(price - fp) / max(price, 1e-9)
            if d < best_dist:
                best_dist = d; best_lvl = lvl; best_fp = fp

        if best_dist > tol or best_lvl is None:
            return None

        # Require a reversal candle closing back toward trend
        prev_C = C[-2]
        if trend_up  and C[-1] <= prev_C: return None
        if not trend_up and C[-1] >= prev_C: return None

        direction = "BUY" if trend_up else "SELL"

        # SL: just beyond the next deeper fib
        deeper_fibs = [l for l in FIB_LEVELS if l > best_lvl]
        if deeper_fibs:
            next_lvl = deeper_fibs[0]
            sl_fib   = sh - swing * next_lvl if trend_up else sl_ + swing * next_lvl
        else:
            sl_fib   = sl_ if trend_up else sh

        if direction == "BUY":
            sl_price   = sl_fib - 3 * pip
            risk       = price - sl_price
            tp         = sh + swing * (1.272 - 1.0)
        else:
            sl_price   = sl_fib + 3 * pip
            risk       = sl_price - price
            tp         = sl_ - swing * (1.272 - 1.0)

        if risk <= 0:
            return None

        proximity_score = 1.0 - (best_dist / tol)
        bonus = 0.12 if best_lvl in KEY_FIBS else 0.0
        conf  = min(0.90, 0.60 + proximity_score * 0.20 + bonus)

        return ProposedTrade(
            strategy_code=self.metadata.code, symbol=ctx.symbol,
            direction=direction, entry_type="market", entry_price=None,
            stop_loss_price=round(sl_price,5), take_profit_price=round(tp,5),
            target_rr=round(abs(tp-price)/risk, 2),
            confidence=conf,
            notes={"fib_level": best_lvl, "fib_price": round(best_fp,5),
                   "swing_high": round(sh,5), "swing_low": round(sl_,5),
                   "dist_pct": round(best_dist*100,3), "ema55": round(ema55,5)})
