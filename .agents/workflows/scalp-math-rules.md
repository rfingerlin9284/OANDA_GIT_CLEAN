---
description: Scalp Math & Execution Rules — Al Brooks scalping fundamentals covering minimum sizes, trader's equation, entry types, scale-in protocol, and probability requirements
---

# Scalp Math & Execution Rules

## Source
Extracted from: "Scalping Series #01 — Rules for Scalping" — Al Brooks

## Overview
The mathematical and structural rules that determine whether a scalping strategy is viable. This is NOT a setup — it's a **validation layer** that sits on top of all scalp workflows. Any scalp setup must pass these rules before execution.

---

## RULE 1 — Minimum Scalp Size (Non-Negotiable)

| Market | Minimum Scalp Target | Below This = Guaranteed Loss |
|--------|---------------------|------------------------------|
| E-Mini S&P | 1 point ($50/contract) | ❌ Never scalp for less |
| Forex (OANDA) | 10 pips | ❌ Never scalp for less |
| Stocks | $0.10 / share | ❌ Never scalp for less |
| Crypto (Coinbase) | 0.15% of price (derived) | ❌ Never scalp for less |

**Why:** Commission, slippage, and spread eat sub-minimum scalps. You mathematically cannot win.

**Engine mapping:** Minimum profit target validation in `trade_engine.py` → reject any signal with TP below minimum

---

## RULE 2 — Scalp Risk / Reward Math

**The fundamental scalp equation:**

```
Scalp reward ≈ ½ average bar height
Scalp risk   ≈ 1x average bar height
Risk:Reward  = 2:1 AGAINST you
```

**Therefore:**

| Win Rate Required | To Break Even | To Profit |
|---|---|---|
| 67% | Break even (before costs) | ❌ Not enough |
| 70% | Break even (after costs) | ❌ Marginal |
| 80% | Profitable | ✅ Good |
| 90% | Very profitable | ✅ Elite |

**Engine mapping:** `risk_manager.py` → scalp mode requires minimum 70% historical win rate on the setup type before allowing execution

---

## RULE 3 — Trader's Equation (Must Be Positive)

```
(Win% × Avg Win) > (Loss% × Avg Loss)
```

**Every trade must satisfy this equation.** If your average win is smaller than your average loss (typical in scalping), your win rate MUST compensate.

**Practical implication:** Scalping is HARDER than swing trading for most traders/bots because:
- Swing: Can win 30-40% and still profit (large wins absorb small losses)
- Scalp: Must win 70%+ (small wins require high frequency success)

**Engine mapping:** Weekly performance audit → if scalp win rate drops below 65%, automatically switch to swing/trend workflows

---

## RULE 4 — Entry Type Selection

### Stop Orders (Trend-Following)
- **Use when:** Betting on a successful breakout / trend continuation
- **Direction:** Enter IN the direction price is already moving
- **Example (short):** Below moving average + bear bar closing on low → sell stop 1 tick below low
- **Example (long):** Above moving average + bull bar closing on high → buy stop 1 tick above high

### Limit Orders (Counter-Trend / Mean Reversion)
- **Use when:** Betting a breakout will FAIL and price will reverse
- **Direction:** Enter AGAINST the current direction
- **Example:** Sell at the high of a bar, betting it won't continue up

**CRITICAL RULE FOR BOTS:**

| Trader Type | Allowed Entry Types | Scale-In | Wide Stops |
|---|---|---|---|
| Beginners / Bots (default) | Stop orders ONLY | ❌ NO | ❌ NO |
| Professional scalpers | Both stop and limit | ✅ Yes | ✅ Yes |

**Engine mapping:** Default bot configuration = STOP ORDERS ONLY, NO SCALE-IN. This is safer and matches the mathematical requirements.

---

## RULE 5 — MA Direction Bias

**Use 10-period EMA on the scalp timeframe (M2 / M5) for direction bias:**

| Price vs 10 EMA | Bias | Allowed Scalps |
|---|---|---|
| Above 10 EMA | Bullish | LONG scalps only |
| Below 10 EMA | Bearish | SHORT scalps only |
| At 10 EMA | Neutral | Wait for directional commitment |

**Signal bar requirements:**
- **Long:** Bull bar closing near its HIGH, turning UP from the 10 EMA
- **Short:** Bear bar closing near its LOW, turning DOWN from the 10 EMA

**Engine mapping:** 10 EMA bias filter in `multi_signal_engine.py` → reject counter-EMA scalps unless explicitly in mean-reversion mode

---

## RULE 6 — The Jackrabbit Problem

**Charts at end of day look EASY. Real-time execution is HARD.**

**Why:** Signal bars can change dramatically in their final 1-2 seconds:
- A bear bar can close bullish in the last second
- A bull bar can close bearish in the last second
- By the time you see the final bar, price may already be several ticks beyond your planned entry

**Bot advantage:** Automated execution removes this problem. Bots can place orders instantly when bars close.

**Bot risk:** False signals from bars that looked good mid-bar but closed differently.

**Engine rule:** ONLY evaluate signals on BAR CLOSE, never mid-bar. Wait for the candle to fully close before evaluating and entering.

**Engine mapping:** `trade_engine.py` → signal evaluation strictly on closed candles, no intra-bar entries

---

## RULE 7 — Scale-In Protocol (Advanced — Disabled by Default)

**For professional scalpers ONLY. Disabled for bot by default.**

**How it works:**
1. Enter at price level A
2. If price moves AGAINST you to level B → buy MORE at B
3. If price moves further to level C → buy MORE at C
4. When price returns to level A → exit all for net profit

**Requirements for scale-in:**
- Position size must be VERY SMALL at each level
- Must use WIDE STOP (or no stop)
- Total risk across all scale-in entries must not exceed standard single-trade risk

**Why it's dangerous for bots:** If the reversal doesn't come, the scaled-in position bleeds heavily. Al Brooks: "Beginners have an incredible talent at exiting right before the trade goes their way."

**Engine mapping:** `risk_manager.py` → `LOCK_SCALE_IN = true` by default. Override only with explicit operator approval.

---

## RULE 8 — Swing > Scalp (Default Preference)

Al Brooks' core advice: **"Most traders should swing trade and not scalp."**

**Why:**
- Swing traders can be sloppy (enter late, exit early) and still profit
- Scalpers must be PERFECT on every tick
- "Anyone can hit a great golf shot occasionally, but it's really hard to hit 72 in a row"

**Bot implication:** The bot should DEFAULT to swing/trend mode and only activate scalp mode when:
1. Regime detector shows STRONG TRENDING (scalp with trend)
2. High-confidence setup from Prop Desk Scalps or 9 EMA workflow
3. Historical scalp win rate > 70% for the current instrument

**Engine mapping:** `regime_detector.py` + `workflow-router.md` → scalp activation requires trending regime + high confidence

---

## Validation Checklist (Run Before Every Scalp)

| Check | Pass | Fail |
|---|---|---|
| Target > minimum scalp size? | ✅ Proceed | ❌ Skip |
| Risk ≤ 1x average bar? | ✅ Proceed | ❌ Reduce size or skip |
| Price vs 10 EMA aligned? | ✅ Proceed | ❌ Skip (counter-trend) |
| Signal bar confirmed on CLOSE? | ✅ Proceed | ❌ Wait for close |
| Trader's equation positive? | ✅ Proceed | ❌ Skip |
| Win rate > 70% this week? | ✅ Proceed | ⚠️ Switch to swing mode |
| Entry type = STOP ORDER? | ✅ Proceed | ⚠️ Need override for limit |

---

## Integration with Workflow Router

This file is a **filter layer**, not a standalone strategy. It validates scalps from:
- `ema9-continuation-scalp.md` → 9 EMA pullback scalps
- `prop-desk-scalps.md` → 6 institutional setups
- `range-bounce-breakout.md` → narrow range breakout (scalp variant)

```
Signal from any scalp workflow
    ↓
Run Scalp Math validation (this file)
    ↓
PASS → Execute
FAIL → Log reason, skip, check for swing alternative
```
