---
description: Position Sizing Engine — Kelly criterion, volatility-adjusted sizing, fractional risk models, and automatic size reduction rules
---

# Position Sizing Engine

## Source
Synthesized from: Kelly Criterion research, professional risk management methodology, ATR-based sizing

## Overview
Position sizing is the SINGLE MOST IMPORTANT factor in long-term survival. This workflow replaces gut-feel sizing with mathematical models that automatically adjust position size based on edge quality, volatility, and recent performance.

---

## MODEL 1 — Fixed Fractional (Default / Baseline)

**The simplest and safest model. DEFAULT for both bots.**

### Rule
Risk a fixed percentage of account equity on every trade.

```
Position Size = (Account Equity × Risk%) / (Entry - Stop Loss)
```

### Parameters

| Parameter | Conservative | Standard | Aggressive |
|-----------|-------------|----------|------------|
| Risk % per trade | 0.5% | 1.0% | 2.0% |
| Max concurrent risk | 2.0% | 4.0% | 6.0% |
| Daily stop | 2.0% | 5.0% | 8.0% |

### Example (OANDA)
```
Account: $10,000
Risk: 1% = $100
Entry: 1.0850
SL: 1.0825 (25 pips)
Pip value: $10/pip (1 standard lot)
Position size = $100 / (25 × $10) = 0.4 lots
```

### Example (COINBASE)
```
Account: $5,000
Risk: 1% = $50
Entry: BTC at $67,000
SL: $66,800 (0.3% = $200 away)
Position size = $50 / $200 = 0.25 units
→ In dollar terms: 0.25 × $67,000 = $16,750 notional
→ Requires leverage or adjust%
```

---

## MODEL 2 — Kelly Criterion (Edge-Adjusted Sizing)

**Advanced. Use ONLY after 50+ trade sample on a specific setup.**

### Formula
```
f* = (p × b - q) / b

Where:
  f* = fraction of capital to allocate
  p  = probability of winning (historical win rate)
  q  = probability of losing (1 - p)
  b  = win/loss ratio (average win / average loss)
```

### Example
```
Win rate: 65% (p = 0.65, q = 0.35)
Average win: $150
Average loss: $100
b = 150/100 = 1.5

f* = (0.65 × 1.5 - 0.35) / 1.5
f* = (0.975 - 0.35) / 1.5
f* = 0.625 / 1.5
f* = 0.417 → Kelly says risk 41.7% of capital

THIS IS WAY TOO AGGRESSIVE.
```

### Fractional Kelly (What to Actually Use)

| Fraction | Risk Level | Use Case |
|----------|-----------|----------|
| Full Kelly (100%) | ❌ NEVER USE | Theoretical only — drawdowns of 50%+ |
| Half Kelly (50%) | ⚠️ Aggressive | Only for A+ setups with 100+ trade sample |
| Quarter Kelly (25%) | ✅ Recommended | Good balance of growth and safety |
| Eighth Kelly (12.5%) | ✅ Conservative | Early stage, proving edge |

### Engine Implementation
```python
def kelly_position_size(win_rate, avg_win, avg_loss, account_equity, fraction=0.25):
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    kelly = max(kelly, 0)  # never negative
    adjusted = kelly * fraction  # fractional Kelly
    max_risk = 0.02  # hard cap at 2%
    risk_pct = min(adjusted, max_risk)
    return account_equity * risk_pct
```

### Requirements for Kelly Activation
| Requirement | Minimum |
|-------------|---------|
| Trade sample size | 50+ trades on this specific setup |
| Win rate stability | <5% variation over rolling 20-trade windows |
| Data recency | Calculated from last 90 days only |
| Recalibration | Weekly |

---

## MODEL 3 — Volatility-Adjusted Sizing (ATR-Based)

**Adjusts position size based on how volatile the instrument is RIGHT NOW.**

### Formula
```
Position Size = (Account Equity × Risk%) / (ATR × ATR_Multiplier)

Where:
  ATR = Average True Range (14-period)
  ATR_Multiplier = how many ATRs your stop is away (typically 1.5-2.0)
```

### Why This Matters
- A 25-pip stop on EUR/USD in LOW volatility is a wide stop
- A 25-pip stop on GBP/JPY in HIGH volatility is a tight stop
- The SAME pip distance has DIFFERENT risk profiles depending on volatility
- ATR-based sizing normalizes this automatically

### Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| ATR period | 14 | 14 |
| ATR multiplier (scalp) | 1.0 | 1.0 |
| ATR multiplier (swing) | 2.0 | 2.0 |
| Recalculation | Every new candle | Every new candle |

### Engine Mapping
- `dynamic_sizing.py` → add ATR-adjusted sizing method
- ATR calculated from the ENTRY timeframe (M5 for scalps, H1 for swings)

---

## AUTOMATIC SIZE REDUCTION RULES

These override ALL models above when triggered:

| Trigger | Size Reduction | Duration |
|---------|---------------|----------|
| 2 consecutive losses | Reduce to 75% of standard | Until next winner |
| 3 consecutive losses | Reduce to 50% of standard | Until 2 consecutive winners |
| Daily P&L at -3% | Reduce to 50% | Rest of day |
| Daily P&L at -5% | STOP TRADING | Rest of day |
| Weekly win rate < 50% | Reduce to 50% | Until weekly review |
| Trading outside A+ killzone | Reduce to 50% | During B/C session |
| HTF bias unclear | Reduce to 50% | Until bias clarifies |
| New setup (< 20 trades history) | Cap at 0.5% risk | Until 50+ trade sample |
| Correlated positions open | Cap total at 3% combined | While correlated |

---

## POSITION SIZE LADDER (Conviction-Based)

From `ai-edge-infrastructure.md` compound gate — signals with higher confidence get larger size:

| Compound Gate Score | Position Size |
|--------------------|---------------|
| 4/6 conditions met | 50% of standard (minimum viable) |
| 5/6 conditions met | 75% of standard |
| 6/6 conditions met | 100% of standard |
| 6/6 + A+ killzone + HTF aligned | 125% of standard (max) |

**Hard cap:** NEVER exceed 2% risk per trade regardless of conviction.

---

## INTEGRATION FLOW

```
Signal approved by compound gate
    ↓
Check conviction score (conditions met / total)
    ↓
Select sizing model:
    ├─ Default: Fixed Fractional (1%)
    ├─ If 50+ trade history on setup: Kelly (Quarter)
    └─ Apply ATR adjustment for volatility
    ↓
Apply reduction rules:
    ├─ Consecutive losses? → reduce
    ├─ Daily P&L limit? → reduce or stop
    ├─ Session priority? → reduce if B/C
    └─ Correlation check? → cap combined
    ↓
Final position size → send to trade_engine.py
```

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| Default risk % | 1.0% | 1.0% |
| Max risk % | 2.0% | 2.0% |
| Max concurrent risk | 4.0% | 4.0% |
| Daily stop | 5.0% | 5.0% |
| Kelly fraction | 0.25 (Quarter) | 0.25 (Quarter) |
| Kelly min sample | 50 trades | 50 trades |
| ATR period | 14 | 14 |
| ATR multiplier (scalp) | 1.0 | 1.0 |
| ATR multiplier (swing) | 2.0 | 2.0 |
| Loss streak reduction | 75% after 2, 50% after 3 | Same |
| Conviction scaling | 50/75/100/125% | Same |
| New setup cap | 0.5% max | 0.5% max |
