---
description: 9 EMA Continuation Scalp — High-probability scalp on trending instruments using 9-period EMA pullback-and-reject with progressive stop game
---

# 9 EMA Continuation Scalp Workflow

## Source
Extracted from: "Step-by-Step Tutorial to Make Consistent Daily Profits in 2026" (SMB Capital / Jeff Holden)

## Overview
A sniper scalp strategy: wait for a strong directional move, let price pull back to the 9 EMA, wait for a **distinct and noticeable change** (rejection), enter with tight risk, and ride with progressive trailing stops ("the stop game").

---

## STEP 1 — Pre-Conditions (Filter)

**Rules — ALL must be true before scanning:**

| Filter | Rule | Engine Mapping |
|--------|------|----------------|
| Clear trend | Instrument has made a distinct move in one direction from the session open / last major swing | `regime_detector.py` → TRENDING state |
| Volume | Volume trailing off on the pullback (not increasing) | Volume analysis in `multi_signal_engine.py` |
| Avoid open | Do NOT trade the first 5 minutes of session (9:30-9:35 for stocks). For forex: avoid first 15 min of London/NY. For crypto: no restriction | Session filter |
| Max names | Watch only 2-3 instruments at a time | `watchlist` / `pre_market_scanner.py` |

---

## STEP 2 — Setup: Price Pulls Back to 9 EMA

**Action:** Wait for the price action to retrace INTO the 9-period Exponential Moving Average.

**Rules:**
- The 9 EMA should be clearly sloping in the trend direction (up for longs, down for shorts)
- Price must actually REACH the 9 EMA — do NOT enter early on anticipation
- Volume should be DECLINING on the pullback (not spiking)
- Do NOT chase breakouts without confirmation

**Engine mapping:** EMA calculation in `multi_signal_engine.py` detector functions

---

## STEP 3 — Trigger: Distinct and Noticeable Change

**Action:** Wait for a **clear rejection** at the 9 EMA level.

**For LONG (uptrend):**
- Buyers step in aggressively at/near the 9 EMA
- Candle prints a bullish rejection (long lower wick, strong close)
- Bid activity suddenly increases (tape reading confirmation)

**For SHORT (downtrend):**
- Sellers step in aggressively at/near the 9 EMA
- Candle prints a bearish rejection (long upper wick, strong close)
- Ask activity suddenly increases

**Critical rule:** NO CHANGE = NO TRADE. If price just hovers at the 9 EMA without a clear rejection, skip it.

**Engine mapping:** Candle analysis in signal detectors (`multi_signal_engine.py`)

---

## STEP 4 — Entry

**Entry point:** Immediately after the distinct and noticeable change is confirmed.

**Stop Loss:** Below the pullback low (for longs) or above the pullback high (for shorts).

**Risk per trade:** 
- Target: $50 risk or ~1% of account equity per trade
- OANDA: Fixed pip-based SL (15-25 pips depending on pair volatility)
- Coinbase: 0.1% - 0.3% of price depending on pair volatility

---

## STEP 5 — The Stop Game (Progressive Trailing)

**This is the critical edge.** Do NOT predict — just keep moving your stop up (or down for shorts).

**Rules:**

| Stage | Trigger | New SL Location |
|-------|---------|-----------------|
| Entry | Trade opened | Below pullback low |
| Stage 1 | Price clears the prior minor swing H/L | Move SL to entry (breakeven) |
| Stage 2 | Price makes a new higher high / lower low | Move SL to the base of the last minor swing |
| Stage 3 | Price makes ANOTHER new HH/LL | Move SL up again to latest minor swing |
| Stage N | Repeat | Keep raising stop with each new swing |
| Exit | Price reverses and hits trailing SL | Stopped out with profit locked |

**Alternative stop method:** Use the 9 EMA itself as a dynamic trailing stop. Exit when price closes below/above the 9 EMA.

**Engine mapping:**
- `trade_manager.py` → green_lock (Stage 1 breakeven)
- `trail_logic.py` → progressive 3-step trail (Stages 2-N)

---

## STEP 6 — Exit Triggers (Sell Rules)

**Exit the remaining position when ANY of these occur:**

1. **Momentum climax:** Sudden acceleration/volume spike (exhaustion candle) → take profit on the blow-off
2. **9 EMA cross:** Price closes clearly on the wrong side of the 9 EMA
3. **Trailing stop hit:** Progressive SL catches a reversal → automatic exit with profit
4. **End of session:** Forex: close before session end. Crypto: no forced close but review at 4h intervals

---

## STEP 7 — "Second Chance Scalp" Variant

When a breakout occurs and then pulls back to re-test the breakout level:

1. Price breaks out of a range/consolidation
2. Price pulls back to the breakout level
3. Wait for the **distinct and noticeable change** (same criteria as Step 3)
4. Enter with stop below the re-test low
5. Manage with the Stop Game

**Key advantage:** Superior R:R because the re-test confirms the breakout is real.

---

## Risk Management Rules (Mandatory)

| Rule | Value | Notes |
|------|-------|-------|
| Max risk per trade | 1R = $50 or 1% account | Scale with account size |
| Daily stop loss | 5R = $250 or 5% account | Stop trading for the day if hit |
| Min R:R target | 3:1 (aim for 4:1) | Never enter below 2:1 |
| Max concurrent trades | 2 | Sniper, not scatter gun |
| Trade journaling | MANDATORY | Review entry, trigger, and post-entry action |

---

## Consistency Protocol (60-Day Rule)

From SMB Capital's development curve:

1. **Rule 1:** Consistency
2. **Rule 2:** Consistency  
3. **Rule 3:** Consistency

**Process:**
- Target a SMALL daily goal (e.g., $200/day or 2R)
- String together 60 consecutive trading days of hitting the target
- Do NOT chase larger targets until the 60-day streak is established
- Scale position size only AFTER proving consistency

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| EMA period | 9 | 9 |
| Timeframe | M5 / M15 | M5 / M15 |
| Min trend move before pullback | 30+ pips | 0.5%+ of price |
| SL distance | Below pullback low (15-25 pips) | Below pullback low (0.15-0.3%) |
| Trail method | Progressive swing-based | Progressive swing-based |
| Session filter | Skip first 15 min of London/NY | None (24/7) |
| Volume decline on pullback | Required | Required |
| Rejection candle body ratio | >60% body vs total range | >60% body vs total range |

---

## When NOT to Take This Trade

- Price is CHOPPY / SIDEWAYS → regime detector shows RANGING/TRIAGE
- Volume is INCREASING on the pullback (contra-trend pressure)
- No clear 9 EMA slope → flat EMA = no trend
- First 5 minutes of session open (noise)
- Already at daily stop loss limit
