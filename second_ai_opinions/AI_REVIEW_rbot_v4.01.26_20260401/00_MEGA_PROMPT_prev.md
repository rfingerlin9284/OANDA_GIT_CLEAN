# RBOTZILLA Quantitative Trading System — Peer Review Audit Request

## TO: Senior Quantitative Trading Engineers / AI Agents
## FROM: RBOTZILLA Development Team
## DATE: March 31, 2026
## SUBJECT: System Not Producing Consistent Profits — Need Fresh Eyes

---

## SITUATION BRIEF

We have a **fully autonomous forex trading bot** (RBOTZILLA, "Rbot") running on OANDA's practice account via REST API. It's written in Python, runs on WSL2/Linux, scans 10 currency pairs on M15 candles every 60 seconds, and places MARKET orders with OCO brackets (SL + TP + trailing stop).

**The problem:** Despite implementing 12+ institutional-grade filters derived from professional trading transcripts, the system is **not generating consistent or meaningful profit**. It either:
1. Takes almost no trades (over-filtered), OR
2. When it does trade, generates tiny P/L that gets eaten by spread/slippage, OR
3. Previously generated $50-$75 winning trades but **gave it all back** via yo-yo reversals

**The account:** $11,200 practice account, 50:1 leverage, started with $21,735 → currently $11,200 (net -$10,500 realized loss over ~2 weeks of development iterations).

**What we need from you:** A brutally honest technical review of the system architecture, signal generation, position sizing, exit management, and filter chain. Identify the root causes of underperformance and suggest specific, implementable fixes.

---

## SYSTEM ARCHITECTURE

### Core Files (read these in order)
1. `01_ENV_CONFIG.txt` — All runtime configuration (.env file)
2. `02_STRATEGY_PIPELINES.py` — The 4 signal generation pipelines
3. `03_SIGNAL_DETECTORS.py` — The individual technical detectors (SMA, EMA, Fib, RSI, etc.)
4. `04_TRADE_ENGINE_SCAN.py` — The per-pair scan loop with all 12 filters
5. `05_TRADE_ENGINE_PLACEMENT.py` — Order construction and submission
6. `06_TRADE_MANAGER.py` — Position management (trail, green-lock, hard stop)
7. `07_TRAIL_LOGIC.py` — Three-step trailing stop progression
8. `08_CAPITAL_ROUTER.py` — Snowball reallocation engine (excerpt)
9. `09_RECENT_LOGS.txt` — Actual engine output showing what happens per cycle
10. `10_DIFF_SINCE_BASELINE.patch` — All code changes since stable baseline

### Signal Flow (Critical Path)
```
M15 Candles (250 bars) fetched per pair
    ↓
4 Strategy Pipelines run independently:
  1. Momentum (SMA + EMA + Fib, needs 2/3 agree)
  2. Reversal (TrapReversal + LiqSweep + RSI, needs 1/3)
  3. Mean Reversion (BB + RSI + S&D zones, needs 1/3)
  4. Scalp (FVG + Order Block, needs 1/2)
    ↓
Each pipeline produces AggregatedSignal or None
    ↓
12-GATE FILTER CHAIN (sequential, any gate can kill signal):
  1. ✅ H4 EMA-55 MTF alignment (momentum only)
  2. ✅ Daily EMA-21 MTF alignment (momentum only)
  3. ✅ 200 EMA directional filter (all signals)
  4. ✅ DXY correlation filter via USD/CHF proxy (USD pairs)
  5. ✅ Volume confirmation (current vol > 1.2x avg)
  6. ✅ MACD divergence detection
  7. ✅ RSI 70/30 overbought/oversold gate
  8. ✅ Session confidence boost (+3% during London/NY)
  9. ✅ Fib+S&D confluence boost (+5%)
  10. ✅ Trend exhaustion filter (>65% of TP already traveled)
  11. ✅ Kelly Criterion sizing (0.5x-1.25x based on confidence)
  12. ✅ Daily circuit breaker ($500 loss / $300 gain limit)
    ↓
ADDITIONAL PLACEMENT GATES:
  - 80% minimum confidence
  - 3+ detector votes required
  - R:R must be >= 1.5:1 (pipeline level) AND >= 3.26:1 (Charter level)
  - Symbol not already active
  - Pair reentry cooldown (15 min)
  - Correlation gate (no same-currency same-direction double-up)
  - Margin gate (>20% free margin)
  - Session gate (3am-5pm ET only)
  - Max 1 new trade per cycle
  - Max 6 open positions
```

### Position Management (After Entry)
```
Trade Manager runs every 60 seconds on all open positions:
  1. Hard dollar stop: close if unrealized P/L < -$120
  2. Green-lock: move SL to breakeven after 10+ pips profit
  3. Three-step trailing stop:
     - Step 1: Lock SL near entry at +6 pips (major pairs)
     - Step 2: Move SL to breakeven at +12 pips
     - Trail: Activate at +25 pips, trail by 15 pips
  4. Stagnation kill: tighten SL after 10 idle cycles
  5. Pair stats tracking: log W/L ratio per pair to disk
```

---

## KNOWN PROBLEMS

### Problem 1: Over-Filtering (The "Nothing Passes" Problem)
With 12 sequential gates, the probability of ANY signal surviving is extremely low. If each gate independently passes 70% of signals:
- `0.70^12 = 1.4%` chance of any signal getting through
- In practice, the engine sits in "CHOP MODE" showing `Slots open (0/12)` for hours
- Most scan cycles show: `MOM=✗ REV=✗ MR=✗ SC=✗` across all 10 pairs

### Problem 2: Position Sizing Mismatch
- Current: 14,000 units → 15 pip move = $21
- Previous (working): 50,000 units → 15 pip move = $75
- The SL/TP math was designed for 14k units but the user needs $50+ trades to be meaningful

### Problem 3: TP Target Unrealistic
- TP = 150 pips on M15 timeframe
- Average daily range for EUR/USD: ~50-80 pips
- 150-pip TP may take 2-5 days to hit — most trades get stopped out or stagnate first
- Meanwhile, SL (ATR-based) fires at 4-9 pips = asymmetric but WRONG direction

### Problem 4: ATR-Based SL Too Tight
- `RBOT_SL_PIPS=0` (ATR mode)
- ATR on M15 for major pairs: ~4-9 pips
- At 14k units, that's $5-$12 risk per trade
- But M15 noise/spread alone can be 2-3 pips → SL fires on random market noise
- Combined with 150-pip TP: risk/reward is 4:150 on paper but 4:never in practice

### Problem 5: Yo-Yo Reversal (The Original Problem)
- Before filters were added: engine generated $50-$75 winners
- But no trailing stop protection → trade would run +$75 then reverse to -$20
- Trailing stop logic was added but thresholds set too aggressively tight initially
- Current trail triggers at +25 pips — but trades rarely GET to +25 pips because SL fires first at 4-9 pips

### Problem 6: Filter Interaction Effects
- **Volume gate** kills valid signals — forex tick volume from OANDA is NOT real exchange volume
- **MACD divergence** on 4 bars of M15 data is statistically meaningless
- **Exhaustion filter** uses TP distance (150 pips × 0.65 = 97.5 pips) — almost never triggers because 30-candle range is usually <50 pips
- **DXY gate** forces USD/CHF correlation assumption — may block valid counter-trend trades
- The gates were designed independently but STACK multiplicatively

---

## CRITICAL QUESTIONS FOR REVIEWERS

1. **Signal Quality**: Are the 4 pipelines generating genuine edge, or are they curve-fitted noise detectors? The detection logic uses fixed thresholds (SMA 20/50, EMA 12/26/50, BB 2σ, RSI 30/70) — are these appropriate for M15 forex?

2. **Filter Architecture**: Should the 12 gates be sequential (kill chain) or should they be scored (each contributes ±weight to a composite confidence)? Current system: any single gate kills the signal entirely.

3. **Position Sizing**: What's the optimal unit size for an $11k account trading 50:1 leverage on M15? Is Kelly Criterion appropriate here, or is fixed fractional better?

4. **Exit Strategy**: Is the 3-step trailing stop correct for M15 forex? The thresholds (activate at +25 pips, trail by 15 pips) were set for "don't trail too early" — but combined with tight ATR SL (4-9 pips), very few trades survive to the trail activation point.

5. **Risk/Reward**: The Charter enforces 3.26:1 minimum R:R. With ATR SL at ~5 pips, that requires TP at ~16.3 pips minimum. But the system has TP at 150 pips. Is this creating a structural mismatch where SL fires 30x more often than TP?

6. **The Fundamental Question**: Should this system be a **scalper** (small SL/TP, tight trail, high frequency) or a **swing trader** (wide SL/TP, patient trail, few trades)? Currently it's neither — it has scalper SLs with swing TPs.

---

## ACCOUNT PERFORMANCE DATA

| Metric | Value |
|--------|-------|
| Starting Balance | $21,735 |
| Current Balance | $11,200 |
| Net Realized P/L | -$10,535 |
| Time Period | ~2 weeks |
| Current Open | 1 trade (USD/JPY SELL) |
| Leverage | 50:1 |
| Environment | OANDA Practice (paper money) |
| Avg Win (when winning) | $5-$75 (highly variable) |
| Avg Loss | $14-$120 (hard stop) |

---

## WHAT WE WANT BACK

1. **Root Cause** — What is the #1 reason this system loses money?
2. **Architecture Fix** — Should the filter chain be restructured? How?
3. **Parameter Fix** — Give us exact values for: SL pips, TP pips, trailing stop activation, trail width, position size, confidence gate
4. **Strategy Fix** — Are the right detectors being used? Should we drop/add any?
5. **Quick Win** — What single change would have the biggest impact right now?

---

## FILES IN THIS PACKAGE

| File | Contents |
|------|----------|
| `00_MEGA_PROMPT.md` | This document (system overview + questions) |
| `01_ENV_CONFIG.txt` | Complete .env configuration |
| `02_STRATEGY_PIPELINES.py` | 4 pipeline runner functions |
| `03_SIGNAL_DETECTORS.py` | Core technical analysis detectors |
| `04_TRADE_ENGINE_SCAN.py` | Per-pair scan loop + 12-gate filter chain |
| `05_TRADE_ENGINE_PLACEMENT.py` | Order construction + broker submission |
| `06_TRADE_MANAGER.py` | Position management (trail, SL, green lock) |
| `07_TRAIL_LOGIC.py` | Three-step trailing stop progression |
| `08_CAPITAL_ROUTER.py` | Snowball reallocation (excerpt) |
| `09_RECENT_LOGS.txt` | Raw engine output from live scanning |
| `10_DIFF_SINCE_BASELINE.patch` | All code changes since stable baseline |
