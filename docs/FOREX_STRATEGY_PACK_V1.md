# FOREX_STRATEGY_PACK_V1.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-17 | Label: NEW_CLEAN_REWRITE
Source ground truth: Phoenix narration session 2026-03-17, detectors in multi_signal_engine.py

---

## Strategy 1 — EMA Stack Trend Continuation

| Property | Value |
|---|---|
| Type | Composite (ema_stack + ema_scalper_200) |
| Market | Trending |
| Session preference | London, New York |
| Timeframe | M15 (primary), M30 |
| Runtime evidence | Fired on EUR_USD, GBP_USD, GBP_CAD, EUR_CAD, AUD_CAD, AUD_USD |

**Long entry rules:**
1. EMA 9 > EMA 21 > EMA 50 (stacked bullish)
2. Price above 200 EMA (ema_scalper_200 confirmation)
3. Last close above EMA 9
4. Session multiplier ≥ 0.90 (London or NY only)

**Short entry rules:**
1. EMA 9 < EMA 21 < EMA 50 (stacked bearish)
2. Price below 200 EMA
3. Last close below EMA 9

**Invalidation:**
- Any EMA crosses to opposite direction intrabar
- Spread > 4.0 pips at entry time (tight threshold for trend trades)
- Price > 1.5% from 200 EMA (overextended)

**Minimum confluence:** 2 votes (ema_stack + ema_scalper_200)
**Confidence logic:** base 0.70 + session multiplier (0.90–1.0) = 0.62–0.70 minimum
**Pair restrictions:** All 22 configured pairs. Avoid GBP exotics (spread risk).
**Spread restriction:** Max 4.0 pips for this strategy (tighter than global 8.0 limit)
**When to skip:** Asia session only, no NY/London overlap confirmed

**False positive notes:**
- Fires heavily in choppy sideways periods — requires fib or FVG confirmation to filter
- Best results when 200 EMA slope confirms direction (trending, not flat)

---

## Strategy 2 — Fibonacci Confluence Reversal

| Property | Value |
|---|---|
| Type | Composite (fibonacci + ema_stack or rsi_extreme) |
| Market | Retracement in trend |
| Session preference | London open, NY open |
| Timeframe | M15 (primary), H1 for swing |

**Long entry rules:**
1. Swing low identified in last 30 candles
2. Price retraced to 61.8% or 50% fib level
3. Bullish candle close at or above fib level
4. EMA stack confirms bullish direction (optional +1 vote)

**Short entry rules:**
1. Swing high identified in last 30 candles
2. Price retraced to 61.8% or 50% fib level
3. Bearish candle close at or below fib level
4. EMA stack confirms bearish direction

**Invalidation:**
- Price closes beyond 78.6% level (invalidates the retracement thesis)
- No clear swing point identifiable in last 30 bars

**Minimum confluence:** 2 votes (fibonacci + 1 confirming detector)
**Confidence:** 0.68 base, +0.05 if rsi_extreme confirms, +0.05 if 200 EMA aligns
**Pair restrictions:** All major pairs. Avoid pairs with spread > 4.0 pips.
**When to skip:** First 30 minutes of any session (candle structure forming)

**False positive notes:**
- 38.2% fib level fires too often in ranging markets — require 50% minimum
- Strong news releases invalidate fib levels instantly

---

## Strategy 3 — Fair Value Gap (FVG) Fill

| Property | Value |
|---|---|
| Type | Standalone or composite |
| Market | Post-impulse retracement |
| Session preference | Any (best at session opens) |
| Timeframe | M15 |

**Long entry rules:**
1. Bullish FVG identified: gap between candle[−3] high and candle[−1] low
2. Current price retracing into the gap
3. EMA stack direction = bullish (confirmation)

**Short entry rules:**
1. Bearish FVG identified: gap between candle[−3] low and candle[−1] high
2. Current price retracing into the gap
3. EMA stack direction = bearish

**Invalidation:**
- Price fully fills and breaks beyond the FVG
- Gap is > 25 pips wide (institutional gap — behavior unpredictable)

**Minimum confluence:** 1 vote (can be standalone at 68%+), better with ema_stack
**Confidence:** 0.73 with ema_stack confirmation
**When to skip:** Choppy ranging days, weekend gaps, high-spread periods

---

## Strategy 4 — Liquidity Sweep Reversal

| Property | Value |
|---|---|
| Type | Reversal |
| Market | Equal highs/lows hunted, reversal expected |
| Session preference | London, NY |
| Timeframe | M15 |

**Long entry rules:**
1. Equal lows (within tolerance) identified in last 20 candles
2. Sweep: price pierces below those lows and recovers (wick below, close above)
3. Buy on recovery candle close

**Short entry rules:**
1. Equal highs identified in last 20 candles
2. Sweep: price pierces above those highs and fails (wick above, close below)
3. Sell on failure candle close

**Stop loss:** Below sweep wick (long) or above sweep wick (short) + 5 pip buffer
**Take profit:** Previous structure (next swing) = 2:1 minimum R

**Minimum confluence:** 1 vote (standalone at 70%+), +1 if trap_reversal confirms
**Pair restrictions:** GBP_NZD excluded (spread). All majors preferred.
**When to skip:** Trending impulse (sweeps in trending markets have lower probability)

---

## Strategy 5 — Trap Reversal Scalper

| Property | Value |
|---|---|
| Type | Reversal / scalp |
| Market | Short-distance reversal after bull/bear trap |
| Session preference | London, NY — avoid Asia |
| Timeframe | M15 |

**Long entry rules:**
1. Recent bearish move (EMA aligned bearish)
2. Bullish trap: bearish candle fails to hold (price spikes then recovers)
3. Close above the "trap" candle's midpoint
4. Scalp target: 10–20 pip gain

**Short entry rules:**
1. Recent bullish move (EMA aligned bullish)
2. Bearish trap: bullish candle fails (spike high then close below midpoint)
3. Scalp target: 10–20 pip gain

**Minimum confluence:** 2 votes required (trap + one confirmation)
**Confidence:** 0.68–0.72 range typical
**Note:** This is classified as a scalping mode in Phoenix — requires tighter spread < 3.0 pips

---

## Strategy 6 — RSI Extreme Reversal (Confirmation-Only)

| Property | Value |
|---|---|
| Type | Confirmation detector |
| Market | Oversold/overbought |
| Timeframe | M15 |

**Rules:**
- RSI < 30 → potential long confluence
- RSI > 70 → potential short confluence
- NEVER use as standalone entry — add as vote +1 to other strategies

**Pair restrictions:** None (universal indicator)
**False positives:** RSI can remain extreme for extended periods in strong trends — must not use alone

---

## BEST 3 — Forex Day-Trade Strategies

| Rank | Strategy | Why |
|---|---|---|
| 1 | EMA Stack + 200 EMA | Highest fire rate, confirmed in runtime. Best in trending sessions (London/NY). |
| 2 | Liquidity Sweep + Trap Reversal | Catches stop-hunt reversals. Two detectors used together — more reliable than standalone. |
| 3 | FVG Fill + EMA Stack | Post-impulse retracement fill — common institutional behavior in M15. |

## BEST 3 — Forex Swing Strategies

| Rank | Strategy | Why |
|---|---|---|
| 1 | Fibonacci Confluence + EMA Stack | H1/H4 timeframe fib retracements with trend confirmation. High R:R. |
| 2 | EMA Stack + RSI Extreme | Overbought/oversold entries in trending direction — strong reversals from key levels. |
| 3 | Liquidity Sweep + Fibonacci | Sweep at or near fib level = institutional accumulation signal. Low-false-positive combination. |
