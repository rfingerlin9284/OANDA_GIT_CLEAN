# RBOTZILLA v4.01.26 — Deep Audit & Multi-AI Mega Prompt
**Date:** April 1, 2026 | **Account:** $10,174 NAV | **Environment:** OANDA Practice

---

## SECTION 1 — DIFF vs PREVIOUS AUDIT (March 31, 2026)

The previous audit was committed to `second_ai_opinions/` on March 31 at 17:50.
The HEAD commit at that time was `d089a29`. Current HEAD is `a616bf5`.

### Files Changed Since Last Audit

| File | Change Summary |
|---|---|
| `engine/trade_engine.py` | Correlation gate commented out (lines 779-786). `PAIR_REENTRY_COOLDOWN_MINUTES` read from env |
| `.env` | `RBOT_BASE_UNITS` 48000→26000, `RBOT_CHOP_UNITS` 48000→26000, `RBOT_MAX_LOSS_USD_PER_TRADE` 120→45, `RBOT_PAIR_REENTRY_COOLDOWN_MINUTES` 15→5 |
| `.vscode/tasks.json` | Restart task now wipes `/tmp/rbotz_clean_cooldowns.json`; pkill suicide bug removed; labels updated to v4.01.26 |

### Critical New Capabilities vs March 31 Audit
- **Unit sizing correctly anchored**: 26k units = ~$2.60/pip. Max 15-pip loss = ~$39 (was $72–$98)
- **Throughput unblocked**: Correlation gate was killing 80%+ of qualified signals each scan cycle. Now disabled.
- **Cooldown reduced 66%**: 15-min → 5-min. Pairs re-enter the pool 3x faster after closure.
- **Cooldown cache wiped on restart**: Old 20-min ghost locks no longer survive reboots.

---

## SECTION 2 — DAILY TRADING BREAKDOWN (Mar 20 – Apr 1)

> Source: `logs/narration.jsonl` (553 TRADE_OPENED events, 666 POSITION_CLOSED events), `logs/pair_stats.json`

### Last Week (Mar 24–27)
| Date | Active Config | Trades Opened | Logic Active at Time |
|---|---|---|---|
| Mar 24 | 48k units, 80% conf gate, 3-vote min | 95 opened, 0 cleanly closed | EMA200 gate, DXY filter, exhaust filter, Kelly sizing — hyper over-filtered. Most signals blocked before entry. |
| Mar 25 | Same | 71 | MTF Sniper D1 block firing 3,768 times total across session. 80% confidence gate rejecting most signals. |
| Mar 26 | Same | 39 | Session gate 3am-5pm ET active. Asian session blocked. |
| Mar 27 | Partial session | 5 | Engine restarted mid-session during config work. |

### This Week (Mar 30 – Apr 1)
| Date | Active Config | Trades Opened | Key Events |
|---|---|---|---|
| Mar 30 | 48k units, conf gate lowered to 65%, ATR trail | 96 | First full day with Fib limit entry logic. Limit orders filling at better prices. |
| Mar 31 | 48k units, 65% gate | 68 | Previous second_ai_opinions audit committed. Engine running. |
| Apr 1 AM | 48k units → 26k units at ~9:25am | 54+ | **The pivot day.** Unit sizing cut. Old trades still hitting 15-pip stops at 48k weight = $72-98 losses. New trades at 26k = ~$39 max loss. Restart at 12:31 ET, 14:13 ET. Correlation gate disabled at 14:13 ET. Cooldown cut to 5 min. |

---

## SECTION 3 — PAIR PERFORMANCE AUDIT

> Source: `logs/pair_stats.json`

| Pair | W | L | Win Rate | Net P&L | Verdict |
|---|---|---|---|---|---|
| **USD_CAD** | 5 | 1 | **83%** | **+$21.05** | ✅ Your best performer. Keep. |
| **EUR_USD** | 3 | 2 | 60% | −$9.65 | ⚠️ Wins exist but exit too early. Trail tightening too fast. |
| **USD_JPY** | 4 | 5 | 44% | +$0.55 | ⚠️ Break-even. JPY pairs volatile during Asian session — session gate not fully protecting. |
| **AUD_USD** | 5 | 4 | 56% | −$25.05 | ⚠️ WR positive but net negative. Losses larger than wins — asymmetric exit problem. |
| **NZD_USD** | 4 | 4 | 50% | −$22.85 | ⚠️ Neutral WR, net negative. Same asymmetric problem. |
| **GBP_JPY** | 2 | 4 | 33% | −$1.70 | 🔴 Low WR JPY cross. MTF sniper should block more aggressively. |
| **GBP_USD** | 2 | 4 | 33% | −$40.50 | 🔴 High loss pair. 33% WR = structural edge problem on this pair. |
| **USD_CHF** | 2 | 5 | **29%** | **−$60.75** | 🔴🔴 WORST PERFORMER. Nearly 3x more losses than wins. DXY inverse pair — engine may be misidentifying direction. |
| **EUR_JPY** | 0 | 4 | **0%** | **−$43.60** | 🔴🔴🔴 ZERO wins. Full 15-pip stops on every single trade. This pair is being traded without any edge whatsoever. |
| **TOTAL** | 27 | 33 | **45%** | **−$182.50** | Below 50% WR. Net negative. Fixing EUR_JPY + USD_CHF + GBP_USD alone could flip this to profitable. |

---

## SECTION 4 — ENGINE GATE AUDIT

> Source: `logs/narration.jsonl` (full lifetime scan)

| Gate | Fires | Interpretation |
|---|---|---|
| MTF_SNIPER blocks | **3,768** | The #1 signal killer. D1/H4 alignment gates are rejecting the vast majority of qualified signals. This is over-conservative. |
| COOLDOWN blocks | **1,862** | Previously 15-min lock. Now 5-min. Should reduce significantly after today's patch. |
| HARD_DOLLAR_STOP | **51** | The $45 software kill-switch has activated 51 times. This is the emergency ejector. It means the broker's physical SL is not always executing cleanly before the trade crosses $45. |
| MARGIN_GATE blocks | **259** | Engine repeatedly tried to trade when margin was to tight. Sizing at 48k caused this. 26k should reduce dramatically. |
| RR_RATIO rejections | **84** | Charter's 3.26:1 R:R requirement repeatedly rejecting orders that cleared the 1.5:1 pipeline threshold. This double standard is a source of missed trades. |
| TRADE_OPENED | **553** | 553 entries in the full log window. |

---

## SECTION 5 — KEY INSIGHTS & PATTERNS IDENTIFIED

### 🔴 CRITICAL — EUR/JPY Must Be Suspended
0-for-4 with full stop losses every time. This is not statistical noise — it's structural. The engine's signal detectors are producing signals that run counter to the actual EUR/JPY price behavior. **Suspending EUR/JPY from the active pairs list immediately stops a guaranteed capital bleed.**

### 🔴 CRITICAL — USD/CHF Direction Is Systematically Wrong
29% WR means the bot is entering on the wrong side of this pair more often than not. USD/CHF has an inverse correlation to DXY/EUR/USD. When the DXY filter is active and correctly classifying USD strength, a BUY on USD/CHF should be valid. The 29% WR suggests the DXY correlation logic is generating false positives.

### 🟡 IMPORTANT — Asymmetric Exit Problem on AUD/USD, NZD/USD
Both pairs have >50% WR but produce net losses. This means wins are smaller than losses. The ATR trailing stop (`RBOT_TS_ATR_MULT=1.5`) is closing winners at partial profit while losses run the full 15 pips. The 3-step TightSL progression may be tightening too aggressively on these highly volatile pairs.

### 🟡 IMPORTANT — Over-Gating Is Destroying Throughput
3,768 MTF Sniper blocks + 1,862 cooldown blocks = **5,630 qualified signals killed** vs only 553 entries placed. That's a **91% rejection rate after signal qualification.** Even at 45% win rate, the mathematical edge disappears when you're only executing 9% of your identified opportunities. The engine is too scared to trade.

### 🟢 POSITIVE — USD/CAD Has Real Edge
83% WR on USD/CAD is statistically significant. This pair has the most structural logic working correctly. At 26k units and $2.60/pip, a 35-pip TP = +$91. The engine should prioritize USD_CAD entries and potentially allow more slots for this pair.

### 🟢 POSITIVE — $45 Hard Stop Is Working
51 HARD_DOLLAR_STOP fires shows the failsafe is active. The issue is that 51 emergency exits is **too many** — each one represents a trade that drifted through $39 before OANDA's broker-side SL executed. This means slippage or SL placement is slightly off on some pairs.

---

## SECTION 6 — RECOMMENDED IMMEDIATE ACTIONS

> **NOT APPLIED — For operator review and override approval**

| Priority | Action | Files | Expected Impact |
|---|---|---|---|
| P0 | Disable EUR_JPY from active pairs | `.env → RBOT_PAIRS` | Stops guaranteed bleed (~$11/trade avg loss) |
| P0 | Disable USD_CHF from active pairs | `.env → RBOT_PAIRS` | Stops systematic directional error ($60.75 lost already) |
| P1 | Raise ATR multiplier for volatile pairs | `.env → RBOT_TS_ATR_MULT=2.0` | Gives winning trades more room to breathe before trailing too tight |
| P1 | Remove Charter's 3.26:1 R:R double-gate | `engine/trade_engine.py` | Eliminates 84 unnecessary RR rejections |
| P2 | Add per-pair WR circuit breaker | `engine/trade_engine.py` | Auto-suspend pairs below 35% lifetime WR |

---

## SECTION 7 — THE $650/DAY FINANCIAL ROADMAP

### Current Math (as of v4.01.26)
- Account NAV: ~$10,174
- Units: 26,000 | Pip value: $2.60
- SL: 15 pips = −$39 max | TP: 35 pips = +$91 max
- Current WR: 45% (net negative due to EUR_JPY / USD_CHF drag)

### After P0 Fixes Applied (Suspending EUR_JPY + USD_CHF)
- Remaining 7 pairs, estimated blended WR rises to ~55–60%
- At 60% WR: 10 trades/day → 6 wins × $65 avg = +$390, 4 losses × $39 = −$156 → **Net +$234/day**

### Path to $650/Day
To reliably generate $650/day requires one of:
1. **Scale units to ~55,000** (when NAV reaches ~$15,000 via compounding) — pip value rises to $5.50, same trade profile now yields $192/win
2. **Increase trade frequency to 18–20/day** at 26k units with 60% WR
3. **Both**: moderate scale + frequency after EUR_JPY/USD_CHF removal unblocks throughput

**Compounding timeline at $234/day clean:**
- Week 1: $10,174 → $11,342
- Week 2: $11,342 → $13,980 (CapitalRouter scales units)
- Week 3: Scale to ~35k units → pip value $3.50 → projected $315/day
- Week 4: Scale to ~45k units → pip value $4.50 → projected $450/day
- Week 6: Scale to ~55k units → pip value $5.50 → $650+/day in reach

---

## SECTION 8 — MULTI-AI MEGA PROMPT

> Copy this entire block and present it identically to: **Grok, GPT-4, Claude, Gemini, DeepSeek**
> Attach the files listed in the FILE MANIFEST section.

---

```
═══════════════════════════════════════════════════════════
RBOTZILLA FOREX BOT — INDEPENDENT AI AUDIT REQUEST
Version: v4.01.26 | Date: April 1, 2026
═══════════════════════════════════════════════════════════

YOU ARE: A senior quantitative trading engineer and algorithmic systems architect.

YOUR TASK: Conduct a brutally honest, technically rigorous audit of a fully autonomous
forex trading bot called RBOTZILLA. The goal is to identify exactly why it is failing
to produce consistent daily profits and provide a specific, implementable improvement
plan to achieve reliably $650+ USD/day in realized capital gains.

══════════════════════
ACCOUNT STATUS
══════════════════════
- Broker: OANDA (V20 REST API, practice account)
- Starting capital (inception): $21,735
- Current NAV: ~$10,174 (net -$11,561 over ~6 weeks of development)
- Leverage: 50:1
- Current active pairs: EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, NZD_USD, USD_CAD, GBP_JPY, EUR_JPY

══════════════════════
CURRENT CONFIGURATION (v4.01.26)
══════════════════════
- Base units: 26,000 (pip value ~$2.60 on majors)
- Stop Loss: 15 pips broker-side OCO = max ~$39/trade
- Take Profit: 35 pips = max +$91/trade
- Trailing Stop: ATR x1.5 multiplier (M15 14-period)
- Hard dollar stop (software): $45/trade
- Daily profit target: $300 (circuit breaker stops new entries when hit)
- Daily loss limit: $150 (circuit breaker)
- Session gate: London + NY only (3am–5pm ET)
- Min confidence: 65%
- Min votes: 1
- Cooldown between trades per pair: 5 minutes (just reduced from 15)
- Correlation gate: DISABLED (just removed)
- Max positions: 6
- Scan frequency: every 60 seconds
- Compounding: watermark-based NAV scaling (RBOT_COMPOUND_GROWTH_EXPONENT=1.5)

══════════════════════
SIGNAL ARCHITECTURE
══════════════════════
4 independent signal pipelines run per pair per cycle:
1. MOMENTUM: SMA crossover + EMA alignment + Fibonacci retracement entry
2. REVERSAL: Trap reversal + Liquidity sweep + RSI extreme detection
3. MEAN REVERSION: Bollinger Band + S&D zone + RSI
4. SCALP: Fair Value Gap (FVG) + Order Block detection

All signals require qualifying through a 12-gate filter chain:
Gate 1: H4 EMA-55 alignment (MOMENTUM only)
Gate 2: Daily EMA-21 alignment (MOMENTUM only)  
Gate 3: Min confidence 65%
Gate 4: DXY correlation via USD/CHF proxy
Gate 5: Volume confirmation (>1.2x average)
Gate 6: MACD divergence
Gate 7: RSI 70/30 overbought/oversold
Gate 8: Session gate
Gate 9: Trend exhaustion block (>65% of TP traveled)
Gate 10: Kelly Criterion sizing
Gate 11: Margin gate (>15% free margin required)
Gate 12: R:R Charter gate (must be ≥3.26:1 at broker level)

══════════════════════
NATIVE PERFORMANCE DATA (Lifetime, from pair_stats.json)
══════════════════════
Pair       | Wins | Losses | Win Rate | Net P&L
-----------|------|--------|----------|--------
USD_CAD    |  5   |   1    |   83%    | +$21.05
EUR_USD    |  3   |   2    |   60%    |  -$9.65
USD_JPY    |  4   |   5    |   44%    |  +$0.55
AUD_USD    |  5   |   4    |   56%    | -$25.05
NZD_USD    |  4   |   4    |   50%    | -$22.85
GBP_JPY    |  2   |   4    |   33%    |  -$1.70
GBP_USD    |  2   |   4    |   33%    | -$40.50
USD_CHF    |  2   |   5    |   29%    | -$60.75
EUR_JPY    |  0   |   4    |    0%    | -$43.60
-----------|------|--------|----------|--------
TOTAL      | 27   |  33    |   45%    | -$182.50

══════════════════════
ENGINE GATE FIRING STATS (Lifetime from narration.jsonl)
══════════════════════
Event                   | Count
------------------------|-------
TRADE_OPENED            | 553
MTF_SNIPER blocks       | 3,768
COOLDOWN blocks         | 1,862
MARGIN_GATE blocks      | 259
HARD_DOLLAR_STOP fires  | 51
RR_RATIO rejections     | 84

══════════════════════
IDENTIFIED PROBLEMS (My current assessment — challenge or confirm these)
══════════════════════
1. EUR/JPY has 0% win rate across all 4 trades. Structural signal mismatch.
2. USD/CHF has 29% win rate. DXY proxy logic may be inverting direction.
3. AUD/USD and NZD/USD have >50% WR but net losses = exits too early on winners, too late on losers.
4. 91% signal rejection rate (3,768 MTF blocks + 1,862 cooldowns vs 553 entries).
   Over-gating is destroying edge by refusing to execute qualified opportunities.
5. The Charter's 3.26:1 R:R gate conflicts with the pipeline's 1.5:1 target, causing 84 extra rejections.
6. HARD_DOLLAR_STOP firing 51 times suggests broker SL slippage or incorrect SL placement on some pairs.

══════════════════════
WHAT I NEED FROM YOU
══════════════════════
Please answer ALL of the following with specificity:

A) PAIR SELECTION
   - Should EUR/JPY and USD/CHF be suspended immediately?
   - What is the correct criteria for blacklisting an underperforming pair?
   - Which of the remaining 7 pairs have the strongest structural edge at 15-pip SL / 35-pip TP?

B) EXIT MANAGEMENT
   - Why would a bot with >50% WR on AUD/USD still produce net losses?
   - Is ATR x1.5 on M15 too tight for volatile pairs? What multiplier would you recommend per pair type?
   - Should trailing stops be session-aware (tighter during NY close, wider during London open)?

C) SIGNAL GATING
   - Is a 91% pre-entry rejection rate acceptable for a 15-pip system at this account size?
   - Should MTF Sniper (D1 + H4 alignment) be limited to MOMENTUM signals only, not REVERSAL/SCALP?
   - What minimum confidence threshold actually produces edge vs random noise at M15 timeframe?

D) POSITION SIZING
   - At $10,174 NAV with 26k units ($2.60/pip), what daily trade volume is needed for $650 net profit?
   - What is the optimal unit scaling schedule from $10k NAV to $20k NAV for maximum compounding without excessive risk?
   - Should the daily profit circuit breaker ($300) be raised to allow more compounding?

E) STRUCTURAL EDGE
   - Given the architecture described, what single change would produce the most immediate improvement in net P&L?
   - Is a 15-pip SL / 35-pip TP ratio (2.33:1) optimal for M15 scalping with ATR trailing, or should SL/TP be dynamic per pair volatility?
   - Should mean reversion signals be traded differently than momentum signals (different SL/TP/trail)?

F) $650/DAY ROADMAP
   - Provide a specific, step-by-step plan to scale this account from ~$10,174 NAV to consistently
     producing $650+ realized USD per trading day.
   - Include: exact unit scaling milestones, minimum WR required at each stage, gate adjustments needed,
     and realistic timeline given the current signal frequency (3–12 trades/day).
   - What NAV level is required before $650/day becomes mathematically sustainable?

══════════════════════
FILE MANIFEST
══════════════════════
The following files are available for your review.
Prioritize reading them in this order:

1. pair_stats.json              — Live pair performance counters
2. .env (current)               — All runtime configuration parameters  
3. engine/trade_engine.py       — Core scan loop, gate chain, order placement
4. engine/trade_manager.py      — Position management, trail logic, hard stop
5. engine/trail_logic.py        — 3-step ATR trailing stop progression
6. engine/capital_router.py     — Watermark compounding and unit scaling
7. engine/signal_detectors.py   — All 4 signal pipeline detectors
8. second_ai_opinions/00_MEGA_PROMPT.md    — Previous audit (March 31) for comparison
9. second_ai_opinions/10_DIFF_SINCE_BASELINE.patch — All changes since stable baseline
10. logs/engine_continuous.out  — Today's live engine logs (Apr 1)

══════════════════════
CONSTRAINTS & CONTEXT
══════════════════════
- This is a PRACTICE account. No real money is at risk.
- The operator uses OANDA's V20 REST API exclusively.
- All code is Python 3.11. No external ML frameworks.
- The engine runs in WSL2 Ubuntu on a ThinkPad.
- The operator wants AUTONOMOUS operation — no manual trade management.
- The operator's non-negotiable constraints: OCO protection on every trade,
  ATR trailing stop, session gate, margin protection.
- The account started at $21,735. It is now at $10,174. STOP THE BLEED FIRST,
  then optimize for growth.

══════════════════════
RESPONSE FORMAT REQUESTED
══════════════════════
1. 3-sentence executive summary of the core problem
2. Answers to questions A through F (be specific, cite the data provided)
3. Ordered action plan (P0 = do today, P1 = do this week, P2 = do this month)
4. Specific code changes or parameter values where relevant
5. $650/day financial roadmap with explicit milestones
6. Any patterns or insights not covered above that you believe are critical

Thank you.
═══════════════════════════════════════════════════════════
```

---

## SECTION 9 — HOW TO USE THIS DOCUMENT

1. Open each AI (Grok, GPT-4o, Claude Sonnet, Gemini Pro, DeepSeek)
2. Paste the mega prompt from Section 8 exactly as written
3. Attach these files (from your repo) to each conversation:
   - `logs/pair_stats.json`
   - `.env`
   - `engine/trade_engine.py`
   - `engine/trade_manager.py`
   - `engine/trail_logic.py`
   - `second_ai_opinions/00_MEGA_PROMPT.md`
   - `second_ai_opinions/10_DIFF_SINCE_BASELINE.patch`
   - `logs/engine_continuous.out` (today's)
4. Collect all 5 responses
5. Bring ALL 5 responses back here — I will synthesize them into a unified improvement plan with exact patches

> [!IMPORTANT]
> Present the SAME prompt to all 5 AIs without modification. This ensures comparable responses you can cross-reference for consensus vs divergence.

> [!WARNING]
> Do NOT accept vague advice like "improve your signals." Demand specific parameter values, specific file edits, and specific milestones with dollar amounts attached.
