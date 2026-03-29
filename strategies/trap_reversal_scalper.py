from __future__ import annotations
# strategies/trap_reversal_scalper.py
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
    O, H, L, C = [], [], [], []
    for c in candles:
        if not isinstance(c, dict): continue
        O.append(_f(c,"open","o")); H.append(_f(c,"high","h"))
        L.append(_f(c,"low","l"));  C.append(_f(c,"close","c"))
    return O, H, L, C

def _pip(sym): return 0.01 if "JPY" in sym.upper() else 0.0001
def _ema(seq, n):
    if not seq: return 0.0
    k = 2.0/(n+1); v = seq[0]
    for p in seq[1:]: v = p*k + v*(1-k)
    return v


class TrapReversalScalperStrategy(BaseStrategy):
    """
    Trap Reversal Scalper — computed from raw candles.

    A "trap" is when price runs an obvious swing high/low (sweeping retail stops),
    reverses sharply, and closes against the sweep direction.

    Patterns detected (raw candle math only):
      • Bearish pin bar at 20-bar swing high  → SELL
      • Bullish pin bar at 20-bar swing low   → BUY
      • Bearish engulfing near swing high     → SELL
      • Bullish engulfing near swing low      → BUY

    Pin bar rule:
      upper_wick > 2 × body  (bearish pin)
      lower_wick > 2 × body  (bullish pin)
      body < 40% of full range

    EMA55 provides trend filter; a trap against the trend gets a bonus.
    """

    def decide_entry(self, ctx: StrategyContext) -> Optional[ProposedTrade]:
        if ctx.timeframe not in self.metadata.base_timeframes:
            return None
        if len(ctx.candles) < 25:
            return None

        O, H, L, C = _ohlc(ctx.candles)
        pip          = _pip(ctx.symbol)
        price        = C[-1]
        ema55        = _ema(C[-70:], 55)

        # 20-bar swing reference
        sh = max(H[-21:-1])
        sl_ = min(L[-21:-1])
        at_high = abs(price - sh) < 12 * pip
        at_low  = abs(price - sl_) < 12 * pip

        o1, h1, l1, c1 = O[-1], H[-1], L[-1], C[-1]
        body       = abs(c1 - o1)
        full_range = h1 - l1 + 1e-9
        upper_wick = h1 - max(o1, c1)
        lower_wick = min(o1, c1) - l1
        body_ratio = body / full_range

        # ── Bearish pin bar at swing high ─────────────────────────────────────
        if at_high and upper_wick > 2 * body and body_ratio < 0.40:
            sl_price = h1 + 3 * pip
            risk     = sl_price - price
            if risk <= 0: return None
            tp       = price - risk * self.metadata.target_rr
            trend_bonus = 0.08 if price < ema55 else 0.0
            conf     = min(0.88, 0.66 + (upper_wick/full_range)*0.20 + trend_bonus)
            return ProposedTrade(
                strategy_code=self.metadata.code, symbol=ctx.symbol,
                direction="SELL", entry_type="market", entry_price=None,
                stop_loss_price=round(sl_price,5), take_profit_price=round(tp,5),
                target_rr=self.metadata.target_rr, confidence=conf,
                notes={"pattern":"bearish_pin_bar","wick_pips":round(upper_wick/pip,1),
                       "sh":round(sh,5),"ema55":round(ema55,5)})

        # ── Bullish pin bar at swing low ──────────────────────────────────────
        if at_low and lower_wick > 2 * body and body_ratio < 0.40:
            sl_price = l1 - 3 * pip
            risk     = price - sl_price
            if risk <= 0: return None
            tp       = price + risk * self.metadata.target_rr
            trend_bonus = 0.08 if price > ema55 else 0.0
            conf     = min(0.88, 0.66 + (lower_wick/full_range)*0.20 + trend_bonus)
            return ProposedTrade(
                strategy_code=self.metadata.code, symbol=ctx.symbol,
                direction="BUY", entry_type="market", entry_price=None,
                stop_loss_price=round(sl_price,5), take_profit_price=round(tp,5),
                target_rr=self.metadata.target_rr, confidence=conf,
                notes={"pattern":"bullish_pin_bar","wick_pips":round(lower_wick/pip,1),
                       "sl":round(sl_,5),"ema55":round(ema55,5)})

        # ── Bearish engulfing near swing high ─────────────────────────────────
        if len(C) >= 2 and at_high:
            prev_h, prev_l = H[-2], L[-2]
            if c1 < o1 and o1 > C[-2] and c1 < prev_l:
                sl_price = h1 + 3 * pip
                risk     = sl_price - price
                if risk > 0:
                    tp   = price - risk * self.metadata.target_rr
                    return ProposedTrade(
                        strategy_code=self.metadata.code, symbol=ctx.symbol,
                        direction="SELL", entry_type="market", entry_price=None,
                        stop_loss_price=round(sl_price,5), take_profit_price=round(tp,5),
                        target_rr=self.metadata.target_rr, confidence=0.72,
                        notes={"pattern":"bearish_engulfing","sh":round(sh,5)})

        # ── Bullish engulfing near swing low ──────────────────────────────────
        if len(C) >= 2 and at_low:
            prev_l = L[-2]; prev_h = H[-2]
            if c1 > o1 and o1 < C[-2] and c1 > prev_h:
                sl_price = l1 - 3 * pip
                risk     = price - sl_price
                if risk > 0:
                    tp   = price + risk * self.metadata.target_rr
                    return ProposedTrade(
                        strategy_code=self.metadata.code, symbol=ctx.symbol,
                        direction="BUY", entry_type="market", entry_price=None,
                        stop_loss_price=round(sl_price,5), take_profit_price=round(tp,5),
                        target_rr=self.metadata.target_rr, confidence=0.72,
                        notes={"pattern":"bullish_engulfing","sl":round(sl_,5)})

        return None
