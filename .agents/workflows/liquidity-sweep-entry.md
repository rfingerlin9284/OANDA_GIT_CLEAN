---
description: Liquidity Sweep Entry — ICT-based strategy for detecting institutional stop hunts, confirming via displacement + market structure shift, and entering on FVG/OB retracement
---

# Liquidity Sweep Entry Workflow

## Source
Synthesized from: ICT (Inner Circle Trader) methodology — liquidity concepts, market structure shifts, and smart money execution

## Overview
Institutional algorithms drive price INTO known liquidity pools (clusters of retail stop-losses) to fill large orders. This workflow detects the sweep, waits for confirmation that the hunt is complete, and enters on the retracement — riding the institutional move AFTER retail has been stopped out.

---

## STEP 1 — Map Liquidity Pools (Pre-Session)

**Action:** Before the session, identify where stop-losses are likely clustered.

### External Liquidity (High Priority)
| Level | Description |
|-------|------------|
| Previous Day High (PDH) | Buy-side liquidity (stop-losses of shorts above) |
| Previous Day Low (PDL) | Sell-side liquidity (stop-losses of longs below) |
| Weekly High / Low | Larger liquidity pools |
| Session High / Low | Asian range H/L, London H/L |

### Internal / Structural Liquidity
| Level | Description |
|-------|------------|
| Equal Highs | Multiple touches at same price = stop cluster above |
| Equal Lows | Multiple touches at same price = stop cluster below |
| Obvious swing highs/lows | Where most retail traders place stops |

**Rule:** Mark these levels. Do NOT trade AT these levels — wait for the SWEEP.

**Engine mapping:** Auto-calculate PDH/PDL, session H/L from candle history. Flag equal H/L clusters.

---

## STEP 2 — Wait for the Sweep (Stop Hunt)

**Action:** Watch for price to trade THROUGH one of the mapped liquidity levels.

**Sweep criteria:**
- Price trades beyond the liquidity level (triggers the stops)
- Price does NOT show strong continuation beyond the level
- Price quickly REJECTS back below/above the level
- Often appears as a LONG WICK through the level followed by close on the opposite side

**Timing filter (Killzones — when sweeps are most likely):**

| Killzone | Time (New York / ET) | Forex | Crypto |
|----------|---------------------|-------|--------|
| Asian | 7:00 PM – 10:00 PM | Range-setting, low vol | Low vol |
| London Open | 2:00 AM – 5:00 AM | HIGH probability sweeps | Moderate |
| NY Open | 7:00 AM – 10:00 AM | HIGHEST probability | HIGH |
| London Close | 11:00 AM – 1:00 PM | Retracement / reversal | Moderate |

**OANDA:** Focus on London + NY killzones
**COINBASE:** Sweeps can happen any time (24/7), but volume peaks at NY open

**Engine mapping:** Session time filter in `multi_signal_engine.py`

---

## STEP 3 — Confirm Displacement + Market Structure Shift (MSS)

**Action:** After the sweep, look for TWO confirmations:

### A. Displacement
- An AGGRESSIVE, impulsive candle (or series of candles) in the OPPOSITE direction of the sweep
- Must show "intent" — strong body, minimal wick on the impulsive side
- This is NOT a soft reversal — it's a SHARP, fast move

### B. Market Structure Shift
- The displacement must BREAK a recent short-term swing point
- For bullish MSS: displacement breaks above the most recent lower high
- For bearish MSS: displacement breaks below the most recent higher low

**Both must be present. Sweep alone is NOT a trade signal.**

**Engine mapping:** Break-of-structure detection + candle momentum analysis in signal detectors

---

## STEP 4 — Identify Entry Zone (FVG or Order Block)

**Action:** The displacement move creates entry zones:

### Fair Value Gap (FVG)
- Look at the 3-candle pattern during displacement:
  - Candle 1: Last candle before the impulsive move
  - Candle 2: The impulsive candle (large body)
  - Candle 3: First candle after the impulsive move
- **FVG = gap between Candle 1 wick and Candle 3 wick** (where they don't overlap)
- This imbalance zone is where price is likely to retrace before continuing

### Order Block (OB)
- The last candle BEFORE the displacement began (same as order-block-sniper.md)
- Marks where institutions loaded their orders

**Entry options:**
1. **Conservative:** Limit order at the 50% point of the FVG (Consequent Encroachment)
2. **Moderate:** Limit order at the edge of the FVG zone
3. **Aggressive:** Enter as soon as price touches the OB/FVG zone

---

## STEP 5 — Stop Loss & Take Profit

**Stop Loss:** Beyond the extreme of the sweep
- For bullish entry (sweep was below): SL below the sweep low
- For bearish entry (sweep was above): SL above the sweep high
- Add buffer: OANDA = +3-5 pips, COINBASE = +0.1%

**Take Profit — Target the OPPOSING liquidity:**
| Priority | Target |
|----------|--------|
| 1st | Next opposing liquidity pool (if sweep was below, target buy-side liquidity above) |
| 2nd | Previous significant swing H/L |
| 3rd | Opposing FVG from a prior timeframe |

**Minimum R:R:** 2:1, ideal 3:1+

---

## STEP 6 — Trade Management

| Stage | Action |
|-------|--------|
| Entry | Position opened at FVG/OB zone |
| 1R profit | Move SL to breakeven |
| 2R profit | Trail using 2-bar trail (from ai-edge-infrastructure.md) |
| Target reached | Close full position |
| Opposing sweep detected | Close early (new liquidity event starting) |

---

## Complete Flow Example (Bullish)

```
1. Pre-session: Mark PDL at 1.0850 (sell-side liquidity below)
2. London Open: Price sweeps below 1.0850 → stops triggered
3. Sweep wick: Low at 1.0842, but candle closes at 1.0858
4. Displacement: 3 strong bullish candles break above recent LH at 1.0880
5. FVG created: Gap between 1.0865 and 1.0875
6. Entry: Limit buy at 1.0870 (mid-FVG)
7. SL: 1.0840 (below sweep low + buffer)
8. TP: 1.0920 (buy-side liquidity / PDH)
9. R:R = 30 pip SL / 50 pip TP = 1.67:1 → expand TP or tighten to meet 2:1
```

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| Liquidity levels | PDH/PDL, session H/L, equal H/L | PDH/PDL, equal H/L |
| Sweep detection | Wick > 3 pips beyond level + close back inside | Wick > 0.1% beyond level + close back |
| Displacement candle | Body > 70% of range, > 1.5x avg bar size | Same |
| MSS confirmation | Break of last swing H/L on M15/M5 | Same |
| FVG identification | 3-candle gap pattern | Same |
| Entry | 50% of FVG (Consequent Encroachment) | Same |
| SL | Beyond sweep extreme + 3-5 pip buffer | + 0.1% buffer |
| TP | Next opposing liquidity pool | Same |
| Killzone filter | London (2-5 AM ET), NY (7-10 AM ET) | Any, prefer NY open |

---

## Relationship to Other Workflows

| Connection | How They Link |
|------------|--------------|
| `order-block-sniper.md` | OB detection is SHARED — this workflow adds liquidity context on top |
| `range-bounce-breakout.md` | Equal H/L in a range ARE liquidity pools — sweeps at range edges |
| `prop-desk-scalps.md` | "Rubber Band" = mean reversion after a sweep-like extension |
| `scalp-math-rules.md` | All entries must pass scalp math validation |
| `workflow-router.md` | Active in TRENDING regime primarily; can fire in SIDEWAYS during range sweeps |
