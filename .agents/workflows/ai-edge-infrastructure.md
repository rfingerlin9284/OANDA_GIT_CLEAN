---
description: AI Edge Infrastructure — Compound signal gating, pre-session ranking, self-performance analytics, two-bar trail, and post-trade autopsy loop for autonomous bot adaptation
---

# AI Edge Infrastructure Workflow

## Source
Extracted from: "How to Use Claude to Gain a Huge Day Trading Edge" — SMB Capital

## Overview
This is NOT a trading strategy. This is an **operational infrastructure layer** that makes the bot smarter over time. It implements five systems that give the engine capabilities previously only available to hedge funds with dedicated quant teams. The key insight: **"The edge in trading isn't what you trade anymore. It's how efficiently you operate."**

---

## SYSTEM 1 — Compound Signal Gate (Multi-Condition Alert Architecture)

### Problem Solved
Basic signals (single indicator crossover, single breakout, single volume spike) produce too many false positives. The bot enters too often with marginal edge.

### Solution
Every signal must pass through a COMPOUND GATE — multiple independent conditions must ALL be true simultaneously before the signal fires.

### Architecture

```
Signal fires ONLY when ALL conditions met:

┌──────────────────────────────────────┐
│ CONDITION 1: Price action trigger    │ ← Breakout, rejection, OB retest, etc.
├──────────────────────────────────────┤
│ CONDITION 2: Volume confirmation     │ ← Bar volume ≥ 1.5x average volume
├──────────────────────────────────────┤
│ CONDITION 3: VWAP/MA position        │ ← Price on correct side of VWAP or MA
├──────────────────────────────────────┤
│ CONDITION 4: Time/session filter     │ ← Within allowed trading window
├──────────────────────────────────────┤
│ CONDITION 5: Regime alignment        │ ← regime_detector confirms state
├──────────────────────────────────────┤
│ CONDITION 6: Scalp math validated    │ ← From scalp-math-rules.md
└──────────────────────────────────────┘
         ALL TRUE → SIGNAL APPROVED
         ANY FALSE → SIGNAL REJECTED (log which condition failed)
```

### Implementation Rules

| Rule | Description |
|------|------------|
| Minimum conditions | Every signal must pass at least 4 independent conditions |
| Condition logging | Log WHICH conditions passed and which failed for every evaluated signal |
| False positive tracking | Track how many signals were rejected and why — this feeds System 3 |
| Refinement cycle | Every 2 weeks, review rejected signals — were they correctly rejected or missed opportunities? |

### Concrete Example — Opening Range Breakout

From the transcript, this is the exact compound gate for an ORB signal:

```
1. Price breaks above 30-min opening range high (close, not wick)
2. Bar volume ≥ 1.5x average volume of the opening 30-min range
3. Price is currently above VWAP
4. Within first 2 hours of session (before 11:30 ET / 2 hours post-London)
5. Breakout bar closes in top 50% of its range (strong buying pressure)
6. Regime = TRENDING (not SIDEWAYS or TRIAGE)
```

**All 6 must be true. Miss one → no trade.**

### Engine Mapping
- `multi_signal_engine.py` → implement compound gate wrapper around all signal detectors
- Each detector returns a dict of `{condition_name: True/False}` instead of a single score
- Final signal only fires when all conditions = True

---

## SYSTEM 2 — Pre-Session Instrument Ranking (Automated Game Plan)

### Problem Solved
The bot scans all instruments equally, wasting cycles on low-probability instruments while potentially missing the best setups.

### Solution
Before each trading session, run an automated ranking of all watched instruments. Sort by priority. Focus execution capacity on the top 2-3.

### Ranking Algorithm

For each instrument on the watchlist, score across these dimensions:

| Dimension | Weight | How to Score |
|-----------|--------|-------------|
| Catalyst strength | 30% | News event, earnings, data release proximity (0-10) |
| Pre-session price action | 25% | Gap direction, pre-market volume, range expansion (0-10) |
| Setup potential | 25% | Does current price action match any active workflow setup? (0-10) |
| Historical win rate on this instrument | 20% | Bot's own track record on this pair/ticker (0-10) |

### Output Format

```
Priority Ranking (Session: London Open / NY Open / Crypto 24h):
──────────────────────────────────────────────────
 1. [HIGH]   EUR/USD  — Score 8.5 — Setup: 9 EMA pullback forming, regime TRENDING
 2. [HIGH]   GBP/JPY  — Score 7.8 — Setup: OB retest approaching, NFP catalyst
 3. [MED]    AUD/USD  — Score 5.2 — Setup: Range forming, no catalyst
 4. [LOW]    USD/CHF  — Score 3.1 — Setup: None identified, regime TRIAGE
──────────────────────────────────────────────────
 → Focus on: EUR/USD, GBP/JPY (max 2-3 concurrent)
 → Skip: USD/CHF (low score)
```

### Implementation Rules

| Rule | Description |
|------|------------|
| Max focus instruments | 2-3 simultaneous (sniper, not scatter gun) |
| Ranking frequency | Once per session start (London, NY for forex; every 4h for crypto) |
| Re-rank trigger | If top instrument hits daily stop or all setups invalidated → re-rank |
| Log the ranking | Store each session's ranking for post-analysis (feeds System 3) |

### Engine Mapping
- `pre_market_scanner.py` or new `session_ranker.py` module
- Runs automatically at session start
- Feeds ranked list into `trade_engine.py` → instruments outside top 3 get deprioritized

---

## SYSTEM 3 — Self-Performance Analytics (The Bot Watches Itself)

### Problem Solved
The bot has no awareness of its own performance patterns. It doesn't know which setups work best, which time periods are profitable, or when its edge has disappeared.

### Solution
Continuous self-analysis of the bot's own trade history. Automatically detect performance patterns and adjust behavior.

### Analysis Dimensions

| Analysis | What It Measures | Frequency |
|----------|-----------------|-----------|
| **By setup type** | Win rate, avg win, avg loss, total P&L per workflow/setup | Weekly |
| **By time of day** | Win rate per hourly block (which hours is the bot profitable?) | Weekly |
| **By day of week** | Win rate Mon-Fri (forex) or daily (crypto) | Weekly |
| **By instrument** | Win rate per pair/ticker | Weekly |
| **Streak detection** | Identify consecutive loss streaks and what preceded them | Daily |
| **Edge decay** | Is overall win rate trending down over 20+ trade windows? | Daily |

### Auto-Adjustment Rules (The Critical Edge)

From the transcript: *"Your edge disappears completely after 11:30."*

The bot must detect patterns like this automatically and respond:

| Pattern Detected | Automatic Response |
|---|---|
| Win rate < 50% in specific hour block (10+ trades sample) | **Reduce position size 50% in that hour** |
| Win rate < 40% on specific setup type (15+ trades sample) | **Disable that setup type until manual review** |
| Win rate < 50% on specific instrument (10+ trades sample) | **Remove instrument from active watchlist** |
| 3+ consecutive losses in current session | **Pause trading for 1 hour** |
| Daily P&L hits -5% of equity | **Stop trading for the day** |
| Weekly win rate drops below 55% (all setups combined) | **Switch from scalp mode to swing-only mode** |
| Setup produces 80%+ win rate over 20+ trades | **Flag as "A+ setup" — allow increased position size** |

### The "One Most Important Thing" Protocol

From the transcript: *"Help me prioritize. Help me define the one most important thing I need to focus on."*

At the end of each week, the analytics system should produce a single actionable recommendation:

```
WEEKLY SELF-ANALYSIS SUMMARY:
────────────────────────────
Total trades: 47
Win rate: 62%
Best setup: 9 EMA Scalp (78% win, 23 trades)
Worst setup: Range Bounce (41% win, 12 trades)
Best hour: 09:00-10:00 (71% win)
Worst hour: 14:00-15:00 (38% win)

→ ONE THING TO FIX: Range Bounce setup is below threshold.
  RECOMMENDATION: Disable Range Bounce until regime detection
  accuracy improves for SIDEWAYS classification.
────────────────────────────
```

### Engine Mapping
- New module: `self_analytics.py` or integrated into existing logging
- Reads from trade log / P&L history
- Outputs to narration log + adjusts runtime parameters
- **Does NOT require external AI** — this is pure data analysis (pandas-grade math)

---

## SYSTEM 4 — Two-Bar Trailing Stop (Structural Exit Logic)

### Problem Solved
Exits based on fixed pip/percentage targets or emotions. No structural awareness.

### Solution
A trailing stop that locks in profit based on PRICE STRUCTURE, not arbitrary numbers.

### Rules

```
ENTRY:
  → Place stop at lowest low of past 2 bars (for longs)
  → Place stop at highest high of past 2 bars (for shorts)

EVERY NEW BAR:
  → IF current_high > highest_high_of_past_2_bars:
       new_stop = lowest_low_of_past_2_bars
       IF new_stop > current_stop:
           current_stop = new_stop  ← RATCHET UP ONLY
  → Stop NEVER moves down (for longs) / NEVER moves up (for shorts)

EXIT:
  → Price touches or crosses the trailing stop → EXIT
```

### Visual Trail Example (Long Trade)

```
Bar 1: Entry at 100.50, Stop at 100.20 (low of past 2 bars)
Bar 2: New high 100.80 → Stop moves to 100.40 (new lowest of past 2)
Bar 3: New high 101.10 → Stop moves to 100.65 (new lowest of past 2)
Bar 4: Price pulls back to 100.90 → Stop stays at 100.65 (no new high)
Bar 5: New high 101.30 → Stop moves to 100.80
Bar 6: Price drops to 100.80 → STOPPED OUT with profit
```

### Configuration

| Parameter | Default | Adjustable |
|-----------|---------|------------|
| Lookback period | 2 bars | Can adjust to 3, 5, 7 via config |
| Bar timeframe | M5 (scalps) / H1 (swings) | Per-workflow setting |
| Activation | After 1R profit (combine with green_lock) | Immediate or delayed |
| Direction | Only ratchets in profit direction | Non-negotiable |

### Integration with Existing Trail Logic

The two-bar trail is a REPLACEMENT or ALTERNATIVE to fixed-pip trailing:

| Scenario | Use Fixed Trail | Use 2-Bar Trail |
|---|---|---|
| Scalp in fast momentum | ✅ Fixed (tight) | ⚠️ May be too loose |
| Trend continuation | ⚠️ Gets stopped too early | ✅ 2-bar (respects structure) |
| Swing trade | ❌ Way too tight | ✅ 2-bar or wider |
| Rubber band / mean reversion | ✅ Fixed to VWAP target | ❌ Not applicable |

### Engine Mapping
- `trail_logic.py` → add `two_bar_trail` method alongside existing fixed and progressive trails
- Activated based on workflow type (scalp = fixed, trend = two-bar, swing = two-bar)
- Lookback period configurable in `config/trail_config.json`

---

## SYSTEM 5 — Post-Trade Autopsy Loop (Rule Violation Detection)

### Problem Solved
The bot can't learn from its own mistakes. It repeats the same errors (premature exits, oversized entries, counter-trend trades in ranging markets) without knowing.

### Solution
After every closed trade, run a structured self-analysis that checks: Did the bot follow its own rules?

### Autopsy Checklist (Evaluated Per Trade)

| Check | Question | Data Source |
|---|---|---|
| Entry rules compliance | Did the entry match a defined workflow setup? | Signal log |
| Compound gate compliance | Were all conditions in the compound gate met? | Condition log |
| Regime alignment | Was the trade direction aligned with the regime? | regime_detector state |
| Risk sizing | Was position size within 1-2% risk limit? | Position log |
| Stop placement | Was initial SL at the correct structural level? | Order log |
| Exit compliance | Was exit triggered by the defined exit rule (trail, TP, structural)? | Exit log |
| Emotional override? | Was the trade exited early/late vs plan? | Compare planned TP/SL vs actual exit |
| Time filter | Was the trade within the allowed session/hour? | Timestamp |

### Violation Scoring

| Severity | Violation | Consequence |
|---|---|---|
| 🔴 Critical | Traded against regime (trend signal in SIDEWAYS) | Flag for setup review, count toward weekly audit |
| 🔴 Critical | Risk exceeded 3% of equity on single trade | Reduce max position size automatically |
| 🟡 Major | Exited before SL or TP hit (premature exit) | Log but no auto-action (may be valid discretion) |
| 🟡 Major | Entered without full compound gate approval | Count violations; 3+ in a day → pause 2 hours |
| 🟢 Minor | Missed optimal entry by 1-2 bars | Log for optimization, no action |
| 🟢 Minor | Trail stopped out just before target | Log, consider wider trail for this setup |

### Weekly Autopsy Summary

```
WEEKLY TRADE AUTOPSY:
────────────────────────────
Trades reviewed: 47
Rules compliance rate: 83% (39/47 fully compliant)

Violations:
  🔴 Critical: 2 (regime mismatch — both Range Bounce trades in TRENDING market)
  🟡 Major: 4 (premature exits — avg missed profit: 12 pips)
  🟢 Minor: 2 (entry timing — 1-2 bars late)

Pattern detected: 
  → ALL critical violations occurred during Range Bounce setup
  → Regime detector may be mis-classifying transitions

Recommendation:
  → Add confirmation delay to regime transitions (require 3+ bars of SIDEWAYS 
    before activating range strategies)
────────────────────────────
```

### Engine Mapping
- New module: `trade_autopsy.py` or integrated into `trade_manager.py` post-close hook
- Runs automatically after every trade closes
- Writes to narration log + `logs/autopsy/`
- Weekly summary aggregation for operator review

---

## Integration Map

```
PRE-SESSION:
  System 2 (Instrument Ranking) → ranks watchlist → top 3 fed to engine

DURING SESSION:
  System 1 (Compound Gate) → filters all signals → only multi-confirmed pass
  System 4 (Two-Bar Trail) → manages open positions structurally

POST-TRADE:
  System 5 (Autopsy) → evaluates each closed trade against rules

WEEKLY:
  System 3 (Self-Analytics) → analyzes all trades → detects patterns → auto-adjusts
  
FEEDBACK LOOP:
  System 3 findings → adjust System 1 gate conditions
  System 5 violations → adjust System 2 rankings
  System 3 win rates → adjust System 4 trail parameters
```

---

## The Non-Negotiable Rule

From the transcript: **"AI does not replace your thinking. It removes the bottlenecks that prevent you from thinking clearly."**

**For the bot:** These systems do not replace the trading strategy. They remove the bottlenecks that prevent the strategy from executing cleanly. The strategy decides WHAT to trade. This infrastructure decides HOW EFFICIENTLY to execute it.
