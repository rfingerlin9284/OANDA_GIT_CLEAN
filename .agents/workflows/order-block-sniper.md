---
description: Order Block Sniper Entry — Institutional-grade entry using smart money OB detection, market shift confirmation, and MTF refinement
---

# Order Block Sniper Entry Workflow

## Source
Extracted from: "Order Blocks Explained — Trade Like the Banks (No BS Guide)"

## Overview
Identify where institutional smart money is loading orders (Order Blocks), wait for a market structure shift to confirm direction, then refine entry on a lower timeframe for sniper risk:reward.

---

## STEP 1 — Identify Market Structure & Trend Direction (H4/H1)

**Action:** On the H4 or H1 chart, identify the current trend.

**Rules:**
- **Uptrend:** Price making Higher Highs (HH) and Higher Lows (HL)
- **Downtrend:** Price making Lower Highs (LH) and Lower Lows (LL)
- Mark the **last significant swing high/low** — this is the structure level that must break

**Engine mapping:** `regime_detector.py` → trend classification (TRENDING_UP / TRENDING_DOWN / SIDEWAYS)

---

## STEP 2 — Wait for Market Shift (Break of Structure)

**Action:** Wait for price to break the last significant swing high (in a downtrend) or swing low (in an uptrend).

**Rules:**
- **Bearish shift:** Price breaks the LAST HIGHER LOW → trend reversal signal
- **Bullish shift:** Price breaks the LAST LOWER HIGH → trend reversal signal
- The break must be a **clean close** beyond the level, not just a wick

**Key parameter:** `market_shift_threshold` — minimum % price must exceed the swing level to confirm the shift

**Engine mapping:** Break-of-structure detection in `multi_signal_engine.py` detectors

---

## STEP 3 — Identify the Origin Order Block

**Action:** Find the last opposite candle BEFORE the impulsive move (inefficiency) that caused the market shift.

**Rules:**
- **For a bearish OB (sell setup):** Find the last bullish (green) candle before the sharp down move
- **For a bullish OB (buy setup):** Find the last bearish (red) candle before the sharp up move
- Mark the **high and low** of that candle as the Order Block zone

**Validation — Three Rules of a Valid OB:**

| Rule | Requirement | Invalid If |
|------|------------|------------|
| 1. Inefficiency present | Gap between candle before move and candle after move | No gap = no institutional intervention |
| 2. Creates Break of Structure | The move from the OB must break a prior HH or LL | No BOS = weak signal |
| 3. Unmitigated (one-time use) | Price has NOT returned to touch the OB zone yet | Already touched = used/invalidated |

---

## STEP 4 — Wait for Price to Return to the Order Block

**Action:** Patiently wait for price to retrace back to the identified OB zone.

**Rules:**
- Do NOT enter before price reaches the OB zone
- Price usually returns to fill the inefficiency (gap) created by the impulsive move
- Set alert or limit order at the OB zone boundary

**Engine mapping:** Limit order or pending entry logic in `trade_engine.py`

---

## STEP 5 — Drop to Lower Timeframe for Sniper Entry (M15 / M5)

**Action:** When price reaches the H1/H4 OB zone, drop to M15 or M5 to look for a SECOND market shift.

**Rules:**
- On M15/M5, look for a mini trend forming INTO the OB zone
- Wait for a break of that mini trend's last swing → lower timeframe market shift
- This gives a sniper entry with a much tighter stop loss

**Example flow (bearish):**
1. H1: Market shift down → identify bullish OB above
2. Price retraces up to OB zone
3. M15: Price making HH/HL on approach
4. M15: Price breaks last HL → mini market shift confirmed
5. Enter SHORT at the mini OB on M15
6. SL above the M15 OB (very tight)
7. TP at the next unmitigated OB below

---

## STEP 6 — Entry, Stop Loss, and Take Profit

**Entry:** At the lower-timeframe OB zone (M15/M5) OR at the H1 OB zone if skipping MTF refinement

**Stop Loss placement:**
- **Refined (sniper):** Above/below the M15 order block → tight SL → possible 5:1+ R:R
- **Standard:** Above/below the H1 order block → wider SL → possible 1:1 to 2:1 R:R

**Take Profit targets (priority order):**
1. Next unmitigated OB in the direction of the trade (PREFERRED — higher hit rate)
2. Next key demand/supply zone
3. ~~Extreme OB~~ (furthest OB — lower hit rate, often not reached)

**Key lesson:** Always target the NEAREST unmitigated OB, NOT the extreme one. The extreme OB is often not reached.

---

## STEP 7 — Trade Management

**Rules:**
- Once in profit by 1R, move SL to breakeven
- Do NOT hold over high-impact news events (NFP, rate decisions) unless conviction is extreme
- Do NOT hold over weekends for forex (gaps will destroy the setup)
- Crypto: 24/7 market, position can be held but monitor for sudden liquidation cascades

**Engine mapping:** 
- `trade_manager.py` → green_lock (breakeven SL)
- `trail_logic.py` → progressive trail after breakeven

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| HTF for structure | H4 / H1 | H4 / H1 |
| LTF for sniper | M15 / M5 | M15 / M5 |
| BOS threshold | 3+ pips beyond swing | 0.15%+ beyond swing |
| OB zone width | Candle body H-L | Candle body H-L |
| TP target | Nearest unmitigated OB | Nearest unmitigated OB |
| SL placement | Above/below OB + spread | Above/below OB + 0.1% buffer |
| Min R:R required | 2:1 | 2:1 |
| Max hold (forex) | Close before weekend | N/A (24/7) |

---

## Signal Score Integration
This strategy produces highest confidence when combined with:
- Regime detector confirming trend (TRENDING state, not SIDEWAYS)
- Session bias alignment (London/NY for forex, any session for crypto)
- Volume spike on the impulsive move (confirms institutional participation)
