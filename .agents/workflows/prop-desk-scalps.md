---
description: Prop Desk Scalps — 6 institutional scalp setups from SMB Capital's live trading desk with exact entry/exit/stop rules and tape reading triggers
---

# Prop Desk Scalps Workflow

## Source
Extracted from: "How to Scalp Like an Elite Prop Trader (Inside Look)" — Mike Belafurie, SMB Capital

## Overview
Six battle-tested scalp setups from a top-10 US prop firm. Each setup has a specific structure, entry trigger, stop placement, target, and failure response. The key unifying principle: **losses are information, not emotion.** When a setup fails, it often sets up the NEXT trade.

---

## GROUND RULES (Apply to ALL Setups)

| Rule | Requirement |
|------|------------|
| No revenge trades | If a trade fails, move on from the trade, NOT the symbol |
| Always ask "What's setting up next?" | Failed trades create information for the next setup |
| Scalp radar / alert confirmation | Enter FOCUS MODE when conditions align, not ACTION mode |
| Stop precision | 2 cents below/above the key level (equities); 2-3 pips (forex); 0.05% (crypto) |
| Trail method | 2-bar trail during momentum phase; LH/LL trail during grind phase |

---

## SETUP 1 — VWAP Continuation (Failed Breakout Variant)

**Context:** Mega-cap stocks (or high-volume instruments) that run up into resistance, look like they'll break out, then STUFF right at the highs.

**Structure:**
1. Instrument makes first move UP from session open
2. Approaches a key resistance level (range high, HOD, round number)
3. Creates a "stuff scalp" — price pushes into resistance, buyers step in, then it FAILS
4. Price drops BELOW VWAP

**Entry trigger:**
- Wait for a "change in character" — candle closes below the prior 2 candles' low
- Enter SHORT after the change-in-character candle closes

**Stop:** Above the high of the failed breakout move

**Target:** Measured move — distance from the stuff high down to VWAP, projected below VWAP

**Management:**
- Use 2-bar trail during momentum phase
- If move happens "too quickly," partial exit into the flush is acceptable but NOT ideal

**Engine mapping:**
- `multi_signal_engine.py` → failed breakout detector
- VWAP calculated from session open (forex: London/NY; crypto: rolling 24h or midnight UTC)

---

## SETUP 2 — Day 2 Puppy Dog (Continuation)

**Context:** Instrument had a strong move on Day 1 (sustained buy/sell program in the afternoon). Day 2 opens with a small consolidation above/below the prior day's key level.

**Structure:**
1. Day 1: Strong sustained directional move in the afternoon session
2. Day 2: Small consolidation forms near pre-market high/low (the "puppy dog" pattern)
3. Key correlation: consolidation must be ABOVE a round number (for longs) or BELOW (for shorts)

**Entry trigger:**
- Break of the puppy dog consolidation in the trend direction
- Candle closes above pre-market high (for longs) or below pre-market low (for shorts)

**Stop:** Below the low of the puppy dog consolidation (for longs); above the high (for shorts)

**Target:** Use 9 EMA as dynamic exit — first close on the wrong side of the 9 EMA = exit

**Critical filter:** If the consolidation is on the WRONG side of the round number → NO TRADE

**Engine mapping:**
- Day 2 continuation logic → requires prior-session data
- Round number proximity check as a filter

---

## SETUP 3 — Rubber Band (Mean Reversion at Extension)

**Context:** Instrument has extended significantly from its mean (VWAP). When it reaches a key support/resistance level, look for a sudden tape change that signals the move is exhausted.

**Structure:**
1. Strong directional move creates overextension from VWAP
2. Price reaches a KEY SUPPORT/RESISTANCE level
3. Watch for "distinct and noticeable change" on the tape

**The tape change signals:**
- Sudden flurry of bids (for long rubber band) or offers (for short)
- Activity shifts visibly — "like the whole sky lit up"
- Buyer/seller absorption pattern visible

**Entry trigger:**
- After the tape change is confirmed
- Enter in the mean-reversion direction (long if extended down, short if extended up)
- "Size up" — this is a high-conviction setup due to tape confirmation

**Stop:** Below low of day (for long rubber band); above high of day (for short)

**Target:** VWAP — the mean reversion target

**Post-entry validation:**
- Price should NOT hold near the extreme → it should immediately start normalizing
- If price HOLDS at the extreme after the tape change → EXIT early, the setup is failing
- "If it's going to hold at the lows, this is a bad sign"

**Bonus — Higher Low Add:**
- After the initial rubber band move, if price pulls back and forms a HIGHER LOW
- Then prints a re-bid (buyers step back in)
- This is a TRAP for late shorts → ADD to position
- New stop: below the higher low
- Target remains VWAP

**Engine mapping:**
- `trade_manager.py` → overextension detection
- VWAP distance calculation for extension measurement
- Tape proxy: volume spike + candle rejection at key level

---

## SETUP 4 — Big Dog Consolidation (Failed Rubber Band → Breakdown)

**Context:** The Rubber Band setup (Setup 3) FAILS. Instead of reverting to VWAP, price cannot make new highs/lows. This failure creates a new setup.

**Structure:**
1. Rubber band entry taken (Setup 3)
2. Price tries to revert 1x, 2x, 3x → but FAILS each time to make new progress
3. Three failed attempts to advance = "fake strength"
4. A time-based consolidation forms (flat, grinding, going nowhere)

**Failure detection (exit rubber band):**
- After "time goes by" and price has tried 3x to continue but can't → SCRATCH the rubber band
- Do NOT wait for full stop-out — use the failed attempts as information

**Entry trigger (new trade):**
- Wait for the consolidation to build
- When sellers/buyers step back in with a "distinct change" → enter the BREAKDOWN/BREAKUP
- Entry: on the break of the consolidation range

**Stop:** Above the high of the consolidation (for short breakdown); below the low (for long breakup)

**Target:** Two measured moves:
1. First target: width of the consolidation range projected from breakout point
2. Second target: 2x the width of the consolidation range

**Key lesson:** "This is the scalp I'd show a rookie. Simple, clean, repeatable."

**Engine mapping:**
- Failed trade → information pipeline
- Consolidation timer + range compression detection

---

## SETUP 5 — Fashionably Late (9 EMA Cross + Volume)

**Context:** You almost missed the move. A "buy program" is detected via consistent, sustained buying. The entry comes when 9 EMA crosses VWAP with elevated volume.

**Structure:**
1. Instrument shows consistent, sustained buying (or selling) from a dip
2. Price recovers and approaches VWAP from below (for long)
3. Volume increases as price nears VWAP

**Entry trigger:**
- 9 EMA crosses ABOVE VWAP (for long) → enter LONG
- Volume on the cross bar must be elevated (above 20-period average)

**Stop:** 1/3 of the prior range below the cross point

**Target:** Measured move = height of the range from the low to the EMA/VWAP cross point

**Trail:** As price approaches target, trail stop up. Watch for buyers on dips — if they step in, stay in.

**Engine mapping:**
- 9 EMA + VWAP crossover detector
- Volume confirmation filter

---

## SETUP 6 — Above the Clouds (Backside Fail → Trend Continuation)

**Context:** Instrument was bought all morning, pulls back toward VWAP in a "grindy" fashion, looks like it's going to roll over (backside forming), but the backside FAILS.

**Structure:**
1. All-morning (or multi-hour) sustained buying
2. Grindy pullback toward VWAP starts
3. Expected "backside" (trend reversal) starts forming
4. Backside FAILS — price holds above VWAP and key levels
5. Price breaks back above the consolidation highs → "above the clouds"

**Entry trigger:**
- When price breaks the consolidation high with VOLUME confirmation (large volume increase on the break bar)
- Often preceded by a hold above key highs ("if this holds above these highs, it's going to go")

**Stop:** Below the low of the consolidation

**Target:** Trail until blow-off — wait for price to put in a clear Lower High + Lower Low

**Exit method:**
- Trail using LH/LL structure
- If no LH/LL forms, trail until end of session
- On blow-off move: watch for tape acceleration → buyer vanishes = EXIT

**Blow-off detection:**
- Price accelerates sharply into a round number / resistance
- Tape shows massive buyer → then buyer suddenly VANISHES
- "Watch him vanish. That's your cue."

**Engine mapping:**
- Backside failure detection
- Volume-confirmed breakout from consolidation
- Session-end trail timer

---

## Failure → Information Pipeline

This is the CRITICAL meta-rule from this transcript:

```
TRADE FAILS
    ↓
DO NOT REVENGE TRADE
    ↓
ASK: "What is this failure telling me?"
    ↓
Three possible outcomes:
    1. Setup A fails → Setup B forms (Big Dog from Failed Rubber Band)
    2. Near-stop survival → the trade is still valid, stay in (BILL example)
    3. Full stop hit → move on, no revenge
```

**Rule:** "How we take losses matters a ton because losses can be frustrating or they can be information."

**Engine mapping:** `trade_manager.py` → failed trade flag → feed into next signal evaluation cycle

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| VWAP base | Session VWAP (London / NY) | Rolling 24h or midnight UTC VWAP |
| Stop buffer | 2-3 pips beyond key level | 0.05-0.1% beyond key level |
| Momentum trail | 2-bar trail (M5 bars) | 2-bar trail (M5 bars) |
| Tape change proxy | Volume spike + rejection candle | Volume spike + rejection candle |
| 9 EMA period | 9 | 9 |
| Timeframe | M2 / M5 for scalps | M5 for scalps (M2 too noisy in crypto) |
| Round number significance | X.000, X.500 | $1000 increments (BTC), $100 (ETH), $1 (alts) |
| Session | London open / NY open | No session restriction (24/7) |
| Max concurrent scalps | 2-3 | 2-3 |
