# WOLFPACK_COMBOS_V1.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-17 | Label: NEW_CLEAN_REWRITE
Only KEEP-rated detectors used. No ML. No Hive. No session-gate fakes.

---

## How wolf-pack combinations work in scan_symbol()

`scan_symbol()` aggregates votes from all detectors that fire on a given symbol.
A wolf-pack = a named set of detectors that, when they all fire together,
produce a high-confidence signal better than any one of them alone.

Each combo below defines:
- which detectors must fire (lead + confirmations)
- minimum vote threshold
- confidence threshold to attempt placement
- preferred conditions

---

## Combo 1 — Trend Continuation Pack

**Name:** TREND_CONT  
**Minimum votes:** 3  
**Confidence threshold:** 0.68

| Role | Detector |
|---|---|
| Lead | `ema_stack` |
| Confirmation 1 | `ema_scalper_200` |
| Confirmation 2 | `fibonacci` (retracement entry in trend) |

**Preferred pairs:** EUR_USD, GBP_USD, AUD_USD, USD_CAD  
**Preferred timeframe:** M15  
**Session:** London, NY  
**Max spread:** 4.0 pips

**Entry logic:**
1. ema_stack fires with clear EMA stack in trend direction
2. ema_scalper_200 confirms price is on correct side of 200 EMA
3. fibonacci detects a pullback to 50–61.8% level within the trend
4. All 3 fire → place market OCO in trend direction

**Kill conditions:**
- Any candle close that crosses the 200 EMA against the trade
- Spread exceeds 4.0 pips at trigger time
- Session changes to Asia before entry confirms

**Why stronger than standalone:**  
ema_stack alone fires in choppy markets. Adding 200 EMA filters chop.  
Adding fibonacci ensures an entry point with tight stop (below fib invalidation).  
Result: better R:R with fewer false fires.

---

## Combo 2 — Liquidity Sweep Reversal Pack

**Name:** LIQ_SWEEP_REV  
**Minimum votes:** 2  
**Confidence threshold:** 0.70

| Role | Detector |
|---|---|
| Lead | `liq_sweep` |
| Confirmation | `trap_reversal` |
| Optional +1 | `rsi_extreme` |

**Preferred pairs:** EUR_USD, GBP_USD, GBP_JPY, AUD_USD  
**Preferred timeframe:** M15  
**Session:** London open (08:00–10:00 UTC), NY open (13:00–15:00 UTC)  
**Max spread:** 3.5 pips

**Entry logic:**
1. liq_sweep identifies equal highs/lows hunt (stop grab)
2. trap_reversal confirms failure to hold after the sweep
3. Optional: rsi_extreme adds confirmation at extreme level
4. Enter on candle close confirming reversal

**Kill conditions:**
- Price continues beyond the sweep level (sweep invalidation)
- Spread > 3.5 pips (reversal setups need tight spreads)
- No recovery candle within 2 bars of sweep

**Why stronger than standalone:**  
liq_sweep fires at any equal high/low hunt but some do not reverse.  
trap_reversal confirms the actual failure. Together they filter continuation sweeps.

---

## Combo 3 — Fibonacci Breakout Pack

**Name:** FIB_BREAK  
**Minimum votes:** 3  
**Confidence threshold:** 0.72

| Role | Detector |
|---|---|
| Lead | `fibonacci` |
| Confirmation 1 | `ema_stack` |
| Confirmation 2 | `fvg` (gap confirms momentum imbalance) |

**Preferred pairs:** EUR_USD, EUR_GBP, GBP_USD, USD_JPY  
**Preferred timeframe:** M15, M30  
**Session:** London, NY  
**Max spread:** 4.0 pips

**Entry logic:**
1. fibonacci identifies key swing + fib level
2. ema_stack confirms trend direction aligns with fib entry direction
3. fvg fires, showing an imbalance gap in the fib retracement zone
4. All 3 → enter at fib level with stop below/above next fib level (78.6%)

**Kill conditions:**
- Price breaks through 78.6% level on close basis
- EMA stack reverses direction on the entry bar

**Why stronger than standalone:**  
fibonacci without trend confirmation enters against trends.  
Adding ema_stack ensures direction alignment.  
fvg confirmation adds institutional imbalance signal — fewer fake fib bounces.

---

## Combo 4 — High-Confidence Institutional Pack

**Name:** INST_GRADE  
**Minimum votes:** 4  
**Confidence threshold:** 0.76

| Role | Detector |
|---|---|
| Lead | `ema_stack` |
| Confirmation 1 | `ema_scalper_200` |
| Confirmation 2 | `fibonacci` |
| Confirmation 3 | `fvg` or `liq_sweep` |

**Preferred pairs:** EUR_USD, GBP_USD, USD_JPY only  
**Preferred timeframe:** M15  
**Session:** London/NY overlap (13:00–16:00 UTC)  
**Max spread:** 3.0 pips (strictest threshold)

**Entry logic:**
1. All 4 detectors fire on same symbol/candle
2. Direction unanimously agrees
3. Gate passes (spread < 3.0, fresh quote, tradeable confirmed)
4. Place max size allowed by dynamic sizing

**Kill conditions:**
- Any detector fires in opposite direction before entry
- Spread > 3.0 pips at entry time
- Confidence drops below 0.76 after session multiplier

**Why stronger:**  
4-vote minimum restricts firing to highest-confidence institutional confluences.  
In the Phoenix runtime session, these 4-detector signals produced AUD_USD and EUR_USD at 78–80% confidence.

---

## Combo 5 — Scalper Reversal Pack

**Name:** SCALP_REV  
**Minimum votes:** 2  
**Confidence threshold:** 0.68

| Role | Detector |
|---|---|
| Lead | `trap_reversal` |
| Confirmation | `rsi_extreme` |

**Preferred pairs:** EUR_USD, GBP_USD, USD_JPY, USD_CHF  
**Preferred timeframe:** M15  
**Session:** First 60 min of any major session open  
**Max spread:** 2.5 pips (scalps require tight spreads)

**Entry logic:**
1. rsi_extreme fires (RSI < 30 or > 70)
2. trap_reversal fires on same bar (trap candle confirmed)
3. Enter at close of the trigger bar
4. Target: 10–20 pips

**Kill conditions:**
- RSI moves further extreme without reversal within 2 bars
- Spread > 2.5 pips
- London/NY overlap not active

**Why stronger:**  
rsi_extreme alone enters into trending moves.  
trap_reversal alone has no momentum confirmation.  
Together they catch exhaustion + structure failure simultaneously.
