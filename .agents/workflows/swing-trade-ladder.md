---
description: Swing Trade Ladder — Structured swing trading using moving average stretch/snap, laddered entry/exit, core floor position, and sector-awareness pause triggers
---

# Swing Trade Ladder Workflow

## Source
Extracted from: "How I Swing Trade for Continuous Realized Gains — My Full Strategy Revealed"

## Overview
A disciplined swing trading strategy that exploits the natural stretch-and-snap cycle around a moving average. Uses laddered buys on dips and laddered sells on extensions, while maintaining a core floor position that is never violated. Incorporates sector/macro awareness to know when to PAUSE swing trading during breakouts or macro events.

---

## CORE PRINCIPLES

1. **Never fully exit.** Maintain a core floor position at all times.
2. **Never try to nail the exact top or bottom.** Ladder in and out.
3. **The MA is a GUIDE, not an exact trigger.** Combine with sector knowledge.
4. **Breakouts override the strategy.** If the instrument is breaking out, STOP swing trading and let it run.
5. **Cash is ammunition.** Never be fully invested — reserve capital for dip buys.

---

## STEP 1 — Establish Core Floor Position

**Before any swing trading:**

1. Define your MAXIMUM position weight (% of portfolio or max units/lots)
2. Define your FLOOR position (minimum units you will NEVER sell below)
3. The difference between max and floor = your **swing capacity**

| Parameter | Definition | Example |
|-----------|-----------|---------|
| Max Weight | Maximum position allowed | 15% of portfolio / 10 lots |
| Floor Position | Minimum always held | 60% of max weight / 6 lots |
| Swing Capacity | Available for active trading | Max - Floor = 4 lots |

**Rule:** If the instrument has a major breakout while you're trimmed down to floor, you still participate via your floor position. This prevents the nightmare scenario of watching a 50%+ move from the sidelines.

**Engine mapping:** `trade_engine.py` position sizing → enforce floor/ceiling limits

---

## STEP 2 — Configure Moving Average

**Recommended:** 10-period Moving Average (10 MA) on the Daily timeframe

| MA Period | Behavior | Best For |
|-----------|----------|----------|
| 10 MA | Tight/sensitive, more frequent signals | Active swing traders |
| 20 MA | Balanced | Standard swing trading |
| 30 MA | Smooth, fewer signals | Conservative / lower frequency |
| 50 MA | Very smooth | Position trading (not swing) |

**OANDA:** Use 10 MA on Daily for major/minor pairs
**COINBASE:** Use 10 MA on Daily for BTC, ETH; consider 20 MA for altcoins (more volatile)

**Engine mapping:** MA calculation in `multi_signal_engine.py`

---

## STEP 3 — Identify Stretch Zones

**A "stretch" occurs when price moves significantly away from the MA.**

### Sell Zone (Price > MA, stretched above):
- Price has pulled far ABOVE the moving average
- The larger the stretch, the higher the probability of a snap-back
- This is where you TRIM (sell portions)

### Buy Zone (Price < MA, stretched below):
- Price has pulled far BELOW the moving average
- The larger the stretch, the higher the probability of a recovery
- This is where you ADD (buy portions)

### Neutral Zone (Price ≈ MA):
- Price is at or near the MA
- No action — wait for a stretch to form

**Stretch measurement:**

| Stretch Level | OANDA (Forex) | COINBASE (Crypto) |
|--------------|--------------|-------------------|
| Minor | 0.5-1.0% from MA | 3-5% from MA |
| Moderate | 1.0-2.0% from MA | 5-10% from MA |
| Major | >2.0% from MA | >10% from MA |

---

## STEP 4 — Laddered Selling (Trim on Strength)

**When price stretches above the MA:**

**Rules:**
1. Do NOT sell everything at once — LADDER your trims
2. Sell small portions at different price levels on the way up
3. Never sell below your floor position

**Ladder example (4 lots swing capacity):**

| Price Level | Action | Remaining Swing |
|------------|--------|----------------|
| Minor stretch | Sell 1 lot | 3 lots |
| Moderate stretch | Sell 1 lot | 2 lots |
| Major stretch | Sell 1 lot | 1 lot |
| Extreme stretch (or running out of steam) | Sell final 1 lot | 0 lots (floor only) |

**"Running out of steam" signals:**
- Momentum slowing (candles getting smaller)
- Volume declining
- Sector/macro catalyst exhausting
- Price failing to make new highs

**Engine mapping:** `trade_manager.py` → scale-out logic at progressive profit levels

---

## STEP 5 — Laddered Buying (Add on Weakness)

**When price stretches below the MA:**

**Rules:**
1. Do NOT buy everything at once — LADDER your buys
2. Buy small portions at different price levels on the way down
3. Never exceed your maximum position weight

**Ladder example (rebuilding from floor):**

| Price Level | Action | Swing Position |
|------------|--------|----------------|
| Minor dip below MA | Buy 1 lot | 1 lot (+ floor) |
| Moderate dip | Buy 1 lot | 2 lots (+ floor) |
| Major dip | Buy 1 lot | 3 lots (+ floor) |
| Extreme dip (capitulation) | Buy final 1 lot | 4 lots = MAX (+ floor) |

**Critical rule:** You MUST have cash/margin available to buy dips. If you're fully invested, you can't execute this strategy.

**Engine mapping:** `trade_engine.py` → position add logic with max weight check

---

## STEP 6 — Breakout Pause Rule

**When the instrument is breaking out with real momentum, STOP swing trading.**

**Breakout detection:**
- Price making consecutive new highs/lows
- Volume significantly above average
- Sector/macro catalyst driving sustained directional move
- Moving average sharply accelerating

**Rules during breakout:**
- Do NOT trim — let the breakout run
- Benefit from your floor + any remaining swing position
- Resume swing trading only when the breakout shows signs of exhaustion

**"I know the sector" edge:** If you understand the fundamental catalyst driving the breakout (e.g., oil price surge for energy stocks, BTC halving narrative for crypto), you can make informed decisions about whether to hold through or resume trimming.

**Engine mapping:** `regime_detector.py` → BREAKOUT state should override swing logic

---

## STEP 7 — Sector & Macro Awareness

**The most critical edge in swing trading is understanding WHY price is moving.**

### Forex (OANDA):
| Factor | Impact |
|--------|--------|
| Interest rate decisions | Major directional shifts |
| NFP / Employment data | Short-term volatility |
| Geopolitical events | Safe-haven flows (JPY, CHF, USD) |
| Central bank speeches | Medium-term sentiment |

### Crypto (COINBASE):
| Factor | Impact |
|--------|--------|
| BTC dominance shifts | Altcoin rotation |
| Regulatory news | Market-wide drops |
| Exchange hacks/failures | Liquidation cascades |
| Macro risk-on/off | Correlation with equities |
| Token unlocks | Single-asset sell pressure |

**Rule:** If a macro event occurs that fundamentally changes the thesis, re-evaluate ALL positions. Do not blindly follow the ladder — adapt.

---

## STEP 8 — Patience Protocol

**Swing trading is NOT day trading.** Some cycles take days, weeks, or months.

**Rules:**
- Do NOT panic at every red candle
- Do NOT force trades when no stretch exists
- Accept periods of NO activity (price near MA = sit on hands)
- The profit comes from REPEATING the cycle many times, not from one big trade

**Expected activity frequency:**

| Market | Typical Cycle Length | Trades per Month |
|--------|---------------------|-----------------|
| OANDA major pairs | 5-15 trading days | 2-4 round trips |
| OANDA crosses | 3-10 trading days | 3-6 round trips |
| BTC/ETH | 5-20 days | 2-4 round trips |
| Altcoins | 2-10 days | 4-8 round trips |

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| MA period | 10 (Daily) | 10 (Daily), 20 for alts |
| Minor stretch | 0.5% from MA | 3% from MA |
| Moderate stretch | 1.0% from MA | 5% from MA |
| Major stretch | 2.0% from MA | 10% from MA |
| Ladder steps | 3-5 trims/buys | 3-5 trims/buys |
| Floor position | 60% of max weight | 60% of max weight |
| Max position | Defined per pair | Defined per pair |
| Breakout pause trigger | 3+ consecutive new HH/LL | 3+ consecutive new HH/LL |
| Cash reserve minimum | 20% of trading capital | 20% of trading capital |

---

## Integration with Other Workflows

| Regime | Primary Strategy | Swing Ladder Role |
|--------|-----------------|-------------------|
| TRENDING | Order Block / 9 EMA Scalp | Ladder trims if overstretched |
| SIDEWAYS | Range Bounce | Ladder buys at range support |
| BREAKOUT | Let it run | PAUSE all swing activity |
| TRIAGE | No trading | PAUSE all swing activity |

**Bottom line:** The Swing Ladder is a **position management** strategy that works ON TOP of signal-based entries. It governs HOW MUCH you hold, not WHEN you enter.
