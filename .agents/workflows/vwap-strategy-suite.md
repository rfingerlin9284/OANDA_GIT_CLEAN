---
description: VWAP Strategy Suite — Institutional VWAP bounce, 2-sigma mean reversion, and anchored VWAP setups with professional execution rules
---

# VWAP Strategy Suite

## Source
Synthesized from: Professional VWAP scalping methodology (prop trading desk execution rules)

## Overview
VWAP (Volume Weighted Average Price) is the institutional benchmark for "fair value" during a session. Algorithms execute large orders around VWAP. This suite provides three VWAP-based strategies: trend-following pullback, overextension mean reversion, and anchored VWAP support/resistance.

---

## CORE PRINCIPLE

**VWAP = Where institutions consider "fair value" for the session.**

| Price Position | Bias | Institutional Behavior |
|---------------|------|----------------------|
| Price > VWAP | Bullish bias | Institutions buying dips to VWAP |
| Price < VWAP | Bearish bias | Institutions selling rallies to VWAP |
| Price = VWAP | Neutral | Wait for directional commitment |

**Rule:** Do NOT fade VWAP bias unless in explicit mean-reversion mode with overextension confirmed.

---

## STRATEGY 1 — VWAP Trend Pullback (Highest Probability)

### Context
The most popular institutional setup. Price is trending, pulls back to VWAP, and bounces — confirming the trend is still alive.

### Setup Requirements
1. Clear trend established (HH/HL for uptrend, LH/LL for downtrend)
2. Price is consistently on one side of VWAP
3. VWAP is sloping in the trend direction

### Entry
- **Long:** Price pulls back TO VWAP from above → rejection candle forms (hammer, bullish engulfing, pin bar) → enter LONG
- **Short:** Price rallies TO VWAP from below → rejection candle forms (shooting star, bearish engulfing) → enter SHORT
- Volume on the rejection candle should be above average (confirms institutional defense of VWAP)

### Stop Loss
- Below the rejection candle low (for longs) / above rejection candle high (for shorts)
- Alternative: below 1st standard deviation band

### Take Profit
- Recent trend extreme (high of day for longs, low of day for shorts)
- 1st or 2nd standard deviation band in trend direction

### Engine Mapping
- VWAP calculation: Session-based (forex: London/NY open; crypto: midnight UTC or rolling 24h)
- Signal: Price touches VWAP + rejection candle pattern + volume confirmation

---

## STRATEGY 2 — 2-Sigma Mean Reversion (Counter-Trend)

### Context
Price has extended far beyond VWAP, reaching the 2nd standard deviation band. Statistically overextended — high probability of snap-back to the mean.

### Setup Requirements
1. Price reaches or exceeds 2σ (2nd standard deviation) from VWAP
2. Exhaustion candle forms at the extreme (doji, long-wick reversal, pinbar)
3. Volume spike at the extreme (capitulation / exhaustion signal)

### Entry
- **Long (at lower 2σ):** Exhaustion candle at or below lower 2σ → enter LONG
- **Short (at upper 2σ):** Exhaustion candle at or above upper 2σ → enter SHORT

### Stop Loss
- Beyond the extreme of the exhaustion candle + buffer
- OANDA: +5 pips; COINBASE: +0.15%

### Take Profit
- The VWAP line itself (mean reversion target)
- Partial exit at 1σ band, remainder to VWAP

### Risk Warning
- **Bandwalking:** Sometimes price rides along the 2σ band without reverting. If price stays at 2σ for 3+ bars without reverting → CUT the trade, the trend is too strong
- This strategy has LOWER win rate than Strategy 1 but HIGHER R:R

### Engine Mapping
- Calculate VWAP + standard deviation bands (1σ, 2σ)
- Exhaustion pattern detection at extreme bands
- Bandwalk detection: 3+ consecutive closes beyond 2σ = exit early

---

## STRATEGY 3 — Anchored VWAP (Historical Reference)

### Context
Standard VWAP resets daily. Anchored VWAP (AVWAP) starts from a specific event — giving the average cost basis of everyone who traded since that event.

### Anchor Points (Where to Start the AVWAP)
| Event | Why Anchor Here |
|-------|----------------|
| Previous major swing high/low | Tracks avg cost of traders who entered at the turn |
| High-volume reversal bar | Tracks avg cost of the reversal participants |
| Earnings / news candle | Tracks avg cost since the catalyst |
| Gap fill level | Tracks avg cost of gap traders |
| Week / month open | Tracks weekly/monthly "fair value" |

### How to Use
1. Calculate AVWAP from the anchor point
2. When price approaches the AVWAP from below → potential resistance
3. When price approaches the AVWAP from above → potential support
4. Combine with standard VWAP for confluence zones

### Entry
- When standard session VWAP AND anchored VWAP converge at the same level → STRONG support/resistance
- Entry on rejection candle at the confluence zone
- This is a CONFLUENCE tool, not a standalone entry

### Engine Mapping
- Calculate AVWAP from configurable anchor points (prior swing H/L, prior week open)
- Flag when session VWAP and AVWAP are within 0.1% of each other → "VWAP confluence zone"

---

## VWAP Calculation Methods

| Market | VWAP Reset | Notes |
|--------|-----------|-------|
| OANDA (Forex) | London open (3:00 AM ET) or NY open (9:30 AM ET) | Session-based; reset each session |
| COINBASE (Crypto) | Midnight UTC (rolling 24h) | 24/7 market; can also use exchange-specific candle times |
| Standard Deviation | 1σ = ~68% of price within; 2σ = ~95% | Used for band calculations |

---

## Volume Rules (Non-Negotiable)

| Rule | Requirement |
|------|------------|
| VWAP valid only with volume | In low-volume / thin markets, VWAP loses gravitational pull — SKIP |
| Rejection must have volume | If price "bounces" at VWAP but on low volume → weak signal, reduce size |
| Volume spike at 2σ | Required for mean reversion entry — no spike = no exhaustion confirmation |
| High-volume session preference | Trade VWAP strategies during London/NY overlap (forex) or US hours (crypto) |

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| VWAP reset time | 3:00 AM ET (London) or 9:30 AM ET (NY) | Midnight UTC |
| 1σ band | Auto-calculated | Auto-calculated |
| 2σ band | Auto-calculated | Auto-calculated |
| Rejection candle body ratio | >60% body vs total range | >60% body vs total range |
| Volume confirmation | >1.5x 20-bar average | >1.5x 20-bar average |
| Bandwalk exit | 3+ consecutive closes beyond 2σ | Same |
| Anchored VWAP source | Prior swing H/L, week open | Prior swing H/L, week open |
| VWAP confluence zone | VWAP + AVWAP within 5 pips | VWAP + AVWAP within 0.1% |

---

## Integration with Other Workflows

| Workflow | VWAP Role |
|----------|----------|
| `prop-desk-scalps.md` | VWAP is the TARGET for rubber band and fashionably late setups |
| `ema9-continuation-scalp.md` | VWAP as confirming indicator (price above VWAP = long bias for 9 EMA scalps) |
| `liquidity-sweep-entry.md` | VWAP proximity during killzones = high probability sweep zones |
| `swing-trade-ladder.md` | Session VWAP as intraday "mean" for stretch/snap detection |
| `range-bounce-breakout.md` | VWAP inside the range = midpoint reference |
