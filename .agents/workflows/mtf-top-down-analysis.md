---
description: Multi-Timeframe Top-Down Analysis — Systematic HTF→MTF→LTF framework for establishing directional bias before entry
---

# Multi-Timeframe Top-Down Analysis

## Source
Synthesized from: Institutional top-down analysis methodology

## Overview
Before taking ANY trade, the engine must establish a directional bias from the highest relevant timeframe down to the execution timeframe. This prevents the #1 killer of bot profitability: **taking valid signals in the WRONG direction.**

---

## THE THREE-TIMEFRAME STACK

| Layer | Purpose | Analogy | Forex (OANDA) | Crypto (COINBASE) |
|-------|---------|---------|---------------|-------------------|
| **HTF (Higher)** | Direction / Bias | "The Map" | Weekly / Daily | Daily / H4 |
| **MTF (Medium)** | Setup Zone | "The Route" | H4 / H1 | H4 / H1 |
| **LTF (Lower)** | Entry Trigger | "Street View" | M15 / M5 | M15 / M5 |

**Rule:** Pick ONE set of timeframes and stick to it. Do NOT jump between different TF sets.

**Recommended multiplier:** 4x to 6x between each layer (e.g., D1 → H4 → M15)

---

## STEP 1 — HTF Analysis (The Map)

**Goal:** Determine the DIRECTION you will trade. Nothing else.

### What to identify:
1. **Trend structure:** Is price making HH/HL (bullish) or LH/LL (bearish)?
2. **Key levels:** Major support/resistance zones, order blocks, liquidity pools
3. **Bias statement:** Write it down: "The Daily chart is BULLISH, I will ONLY look for LONGS"

### Rules:
- If HTF is BULLISH → only look for LONG entries on LTF
- If HTF is BEARISH → only look for SHORT entries on LTF
- If HTF is UNCLEAR/SIDEWAYS → reduce size by 50% OR skip entirely
- **NEVER trade against the HTF bias** (this is the #1 rule of MTF analysis)

### Engine Mapping:
- `regime_detector.py` on Daily/H4 → outputs TRENDING_UP, TRENDING_DOWN, or SIDEWAYS
- This bias direction feeds ALL signal workflows as a mandatory filter

---

## STEP 2 — MTF Analysis (The Route)

**Goal:** Find WHERE within the HTF trend you should be looking for entries.

### What to identify:
1. **Pullback zones:** Where is price likely to retrace to? (MA, VWAP, OB, FVG, S/R)
2. **Structure alignment:** Is the MTF trend aligned with HTF?
3. **Setup zone:** Mark the specific area where an entry setup COULD form

### Alignment Check:

| HTF Bias | MTF Behavior | Action |
|----------|-------------|--------|
| Bullish | Making HH/HL (aligned) | ✅ PROCEED to LTF |
| Bullish | Pulling back to support (retracement) | ✅ PROCEED — this IS the entry opportunity |
| Bullish | Making LH/LL (counter-trend) | ⚠️ WAIT — MTF needs to realign before entering |
| Bearish | Making LH/LL (aligned) | ✅ PROCEED to LTF |
| Bearish | Rallying to resistance (retracement) | ✅ PROCEED — entry opportunity |
| Bearish | Making HH/HL (counter-trend) | ⚠️ WAIT — MTF needs to realign |

### Engine Mapping:
- H4/H1 structure analysis → confirm alignment with Daily bias
- Mark setup zones (S/R, OB, FVG, MA levels) for LTF monitoring

---

## STEP 3 — LTF Execution (Street View)

**Goal:** Find the precise ENTRY TRIGGER within the setup zone identified on MTF.

### Entry triggers (from existing workflows):
| Trigger Type | Source Workflow |
|-------------|---------------|
| 9 EMA pullback + rejection | `ema9-continuation-scalp.md` |
| Order Block retest | `order-block-sniper.md` |
| Liquidity sweep + MSS + FVG | `liquidity-sweep-entry.md` |
| VWAP bounce | `vwap-strategy-suite.md` |
| Range breakout momentum candle | `range-bounce-breakout.md` |
| Prop desk setups (rubber band, etc.) | `prop-desk-scalps.md` |

### Rules:
- Only take LTF entries that are IN THE DIRECTION of the HTF bias
- Only take LTF entries WITHIN the setup zone identified on MTF
- If the LTF trigger fires but is NOT in a setup zone → SKIP (it's noise)
- SL placement based on LTF structure (tight), TP based on MTF/HTF structure (wide)

---

## COMPLETE TOP-DOWN CHECKLIST

Run this checklist BEFORE every trade:

```
☐ 1. HTF BIAS
     What is the Daily/H4 telling me?
     → BULLISH / BEARISH / UNCLEAR
     → If UNCLEAR → SKIP or reduce size 50%

☐ 2. HTF KEY LEVELS
     Where are the major S/R, OBs, liquidity pools?
     → Mark on chart

☐ 3. MTF ALIGNMENT
     Is the H4/H1 aligned with HTF bias?
     → YES → proceed
     → NO → WAIT for realignment

☐ 4. MTF SETUP ZONE
     Has MTF price reached a valid setup zone?
     → S/R zone, OB, FVG, MA, VWAP level
     → If not at a zone → WAIT, don't force

☐ 5. LTF ENTRY TRIGGER
     Is there a specific entry trigger at the setup zone?
     → 9 EMA rejection, OB retest, FVG fill, VWAP bounce, etc.
     → If no clear trigger → SKIP

☐ 6. DIRECTION MATCH
     Does the LTF trigger direction match HTF bias?
     → MATCH → EXECUTE
     → MISMATCH → REJECT (log: "counter-HTF signal")

☐ 7. RISK/REWARD
     SL at LTF invalidation, TP at MTF/HTF target
     → R:R ≥ 2:1 → EXECUTE
     → R:R < 2:1 → SKIP or adjust
```

---

## Common Mistakes the Bot Must Avoid

| Mistake | Why It's Bad | Prevention |
|---------|-------------|------------|
| Trading LTF signals without HTF bias | High-probability LTF setups can fail against the HTF trend | MANDATORY HTF check before every signal |
| Forcing trades when MTF is misaligned | Counter-trend pullbacks look like entry zones but aren't | Alignment check must pass |
| Using too many timeframes | Analysis paralysis, conflicting signals | Stick to exactly 3 TFs |
| Changing TF set mid-session | Inconsistency, signal confusion | Lock TF set at session start |
| Ignoring "UNCLEAR" readings | Trading in uncertainty increases losses | SKIP or reduce when unclear |

---

## Engine Integration Flow

```
regime_detector.py (HTF bias)
    ↓
    ├─ BULLISH → only process LONG signals
    ├─ BEARISH → only process SHORT signals
    └─ UNCLEAR → reduce max position 50%, raise confidence threshold +15%
    
multi_signal_engine.py (MTF zone detection)
    ↓
    ├─ Price at setup zone → ACTIVATE LTF monitoring
    └─ Price NOT at setup zone → IDLE (no signals evaluated)
    
signal detectors (LTF triggers)
    ↓
    ├─ Trigger matches HTF direction → PASS to compound gate
    └─ Trigger opposes HTF direction → REJECT (log reason)
    
compound gate (ai-edge-infrastructure.md)
    ↓
session filter (session-killzone-filter.md)
    ↓
scalp math (scalp-math-rules.md)
    ↓
EXECUTE
```

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| HTF | Daily (Weekly for context) | Daily (H4 for alts) |
| MTF | H4 | H4 / H1 |
| LTF | M15 / M5 | M15 / M5 |
| TF multiplier | ~4-6x between layers | Same |
| HTF bias update frequency | Once per session (London, NY) | Every 4 hours |
| Counter-HTF signal handling | REJECT | REJECT |
| Unclear HTF size reduction | 50% of standard | 50% of standard |
