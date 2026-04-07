---
description: Range Bounce & Breakout — Strategies for ranging/sideways/choppy markets using key level bounces, narrow range breakout, and market structure shift detection
---

# Range Bounce & Breakout Workflow

## Source
Extracted from: "Most Effective RANGE Trading Strategy for Crypto, Forex & Stocks (Sideways/Choppy Market Strategy)"

## Overview
Markets range ~70% of the time and trend only ~30%. This workflow detects when a market has shifted from trending to ranging, identifies key support/resistance zones, and provides two distinct execution methods: bounce trading within the range, and breakout trading when the range compresses.

---

## CRITICAL INSIGHT
> **Most indicators (MACD, standard MAs) produce FALSE SIGNALS in ranging markets.**  
> The engine MUST detect regime type before applying any signal. Trending indicators in a range = losses.

**Engine mapping:** `regime_detector.py` → when state = SIDEWAYS / RANGING, switch to this workflow instead of trend-following signals.

---

## STEP 1 — Detect Range Formation via Market Structure Shift

**Rules for identifying that a trend has ended and range has begun:**

### From Uptrend → Range:
1. Price was making HH and HL (uptrend)
2. Price fails to make a new HH (equal high or lower high)
3. Price fails AGAIN to make a new HH
4. **Two consecutive HH failures = trend is dead, range is forming**

### From Downtrend → Range:
1. Price was making LH and LL (downtrend)
2. Price fails to make a new LL (equal low or higher low)
3. Price fails AGAIN to make a new LL
4. **Two consecutive LL failures = trend is dead, range is forming**

**Key parameter:** `range_detection_lookback` = number of swings to evaluate (minimum 4)

**Engine mapping:** `regime_detector.py` → `detect_state()` → should classify as SIDEWAYS when this pattern appears

---

## STEP 2 — Draw Key Levels (Support & Resistance)

**Once range is detected:**

1. Draw **Resistance** at the cluster of recent highs (the ceiling)
2. Draw **Support** at the cluster of recent lows (the floor)
3. These are ZONES, not exact lines — expect wicks above/below

**Rules:**
- S/R are general areas, not exact prices
- Allow candle wicks to slightly exceed the levels (normal behavior)
- The range must be WIDE ENOUGH to trade inside (see narrow range exception below)
- Monitor for NEW key levels forming inside the range (price may start respecting a new level)

**Engine mapping:** S/R zone calculation → used by `multi_signal_engine.py` for entry filtering

---

## STRATEGY A — Range Bounce (Wide Range)

### Conditions
- Range is wide enough that price can move meaningfully between S and R
- At least 2-3 prior bounces at each level confirming the zones are respected

### Entry Rules

**BUY (at support):**
1. Price touches or enters the support zone
2. A rejection candle forms (bullish pin bar, hammer, engulfing)
3. Enter LONG with SL below the support zone

**SELL (at resistance):**
1. Price touches or enters the resistance zone
2. A rejection candle forms (bearish pin bar, shooting star, engulfing)
3. Enter SHORT with SL above the resistance zone

### Stop Loss
- Below support zone (for longs) or above resistance zone (for shorts)
- Add buffer: OANDA = +5 pips, Coinbase = +0.1% of price

### Take Profit
- Opposite side of the range (support → target resistance, and vice versa)
- Or middle of range if conviction is lower

### Exit Rules
- If price breaks through a key level with a **momentum candle** = EXIT immediately (range is broken)
- Do NOT fight a breakout — the range strategy is over

---

## STRATEGY B — Narrow Range Breakout (Compression)

### Conditions
- Range is TOO NARROW for bounce trading
- Multiple small candles moving sideways erratically (consolidation)
- Best found on lower timeframes (M1 to M15)

### Detection Rules
- Candle bodies are consistently small
- Range height < 50% of average daily range (ADR)
- This is a compression → expect expansion (breakout)

### Entry Rules

**Step 1:** Draw tight S/R around the narrow range (with enough spacing to avoid false triggers)

**Step 2:** Wait for a **momentum candle** to break out of either level

**Momentum candle criteria:**
- A single BIG candle that closes clearly beyond the level, OR
- 3+ consecutive same-color medium candles breaking through

**NOT a momentum candle:**
- A small candle barely poking above/below the level = likely false breakout → SKIP

**Step 3:** Enter in the direction of the breakout AFTER the momentum candle closes

### Stop Loss
- Opposite side of the narrow range

### Take Profit
- Measured move: height of the range projected from the breakout point
- Or next significant S/R level

---

## Range Width Classification

| Classification | Range Height vs ADR | Strategy |
|---|---|---|
| Wide range | >75% of ADR | Strategy A (Bounce) |
| Medium range | 50-75% of ADR | Strategy A with tighter TP |
| Narrow range | <50% of ADR | Strategy B (Breakout) |
| Ultra-narrow | <25% of ADR | Wait — imminent breakout, use B with wider TP |

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| Range detection | 2x failed HH or 2x failed LL | Same |
| Min bounces to confirm range | 2-3 per level | 2-3 per level |
| S/R zone buffer | 5-10 pips | 0.1-0.2% of price |
| Momentum candle min body | >70% of candle range | >70% of candle range |
| Narrow range threshold | <50% of 14-period ADR | <50% of 14-period ADR |
| Breakout false-out filter | Close beyond level (not just wick) | Close beyond level |
| SL for bounce | Below/above S/R zone + buffer | Below/above S/R zone + buffer |
| TP for bounce | Opposite S/R level | Opposite S/R level |
| TP for breakout | Range height projected | Range height projected |

---

## When Grid/DCA Bots Apply (Automated Range Capture)

Per the transcript, grid bots work best in:
- ✅ Ranging / sideways markets
- ✅ Weak uptrend markets (gradual rise with pullbacks)
- ❌ Strong downtrend markets (bot keeps buying into losses)
- ❌ Strong uptrend markets (bot sells too early, misses gains)

**Engine integration:** If `regime_detector` classifies market as SIDEWAYS for >24h (crypto) or >2 sessions (forex), flag as "grid-eligible" zone for automated range capture.

---

## Integration with Engine Regime Detection

```
if regime == TRENDING_UP or TRENDING_DOWN:
    → Use: Order Block Sniper, 9 EMA Continuation
    → SKIP: Range Bounce

if regime == SIDEWAYS / RANGING:
    → Use: Range Bounce (wide) or Breakout (narrow)
    → SKIP: Trend-following signals
    → Flag: Most indicators produce false signals

if regime == TRIAGE / UNKNOWN:
    → SKIP ALL — wait for clarity
```

**This is the most important integration point.** The engine must NOT apply trending strategies in a range, and must NOT apply range strategies in a trend.
