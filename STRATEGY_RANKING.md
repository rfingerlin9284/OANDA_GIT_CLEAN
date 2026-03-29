# STRATEGY_RANKING.md — RBOTZILLA_OANDA_CLEAN
# Phase 6 — Strategy Audit and Ranking
Generated: 2026-03-17
Source of runtime evidence: Phoenix narration.jsonl session 2026-03-17

---

## Ranking legend

| Rating | Meaning |
|---|---|
| **KEEP** | Runtime evidence confirms it fires real candidates. Carry to clean repo. |
| **MAYBE KEEP** | Logic is sound but lacks runtime confirmation in clean repo. Inspect before trusting. |
| **DISCARD** | Fake data dependency, crypto-only, deprecated, ghosting, or noise-producing. |

---

## Section 1 — Signal Detectors (from strategies/multi_signal_engine.py)

All detectors are internal functions in `scan_symbol()`. They are NOT standalone files.

### ema_stack
- **Rating: KEEP**
- Fired in every confirmed candidate scan (EUR_USD, GBP_USD, GBP_CAD, EUR_CAD, AUD_CAD, AUD_USD)
- Consistent 73–80% confidence with 3–4 votes
- Pure candle math — no external dependency
- OANDA compatible: uses OHLCV data only

### fibonacci
- **Rating: KEEP**
- Fired alongside ema_stack in all high-confidence trades this session
- Swing high/swing low math — no external dependency
- Detects price reactions off fib levels in M15 data
- OANDA compatible

### ema_scalper_200
- **Rating: KEEP**
- Fired in most scan passes alongside ema_stack + fibonacci
- 200 EMA directional bias — well-established institutional reference
- Adds 4th vote when combined with ema_stack + fibonacci + fvg (seen on AUD_USD)
- OANDA compatible

### fvg (Fair Value Gap)
- **Rating: KEEP**
- Fired on AUD_USD and AUD_CAD at 74–78% confidence
- Detects price imbalance gaps in M15 candles
- OANDA compatible
- Pairs well with fibonacci for confluence

### liq_sweep (Liquidity Sweep)
- **Rating: KEEP (with pair restriction)**
- Runtime evidence: fired on GBP_NZD at 74% confidence
- Signal is genuinely present — but GBP_NZD was blocked by spread gate (9.1–9.4 pips)
- Apply to major pairs only; GBP exotics have structural spread problems
- Keep detector; discard GBP_NZD from pair list

### trap_reversal
- **Rating: KEEP**
- Fires with liq_sweep as a confirmation
- Detects exhaustion after a liquidity grab
- OANDA compatible
- Low false-positive rate when requiring 3+ votes

### rsi_extreme
- **Rating: KEEP**
- Fires with liq_sweep + trap_reversal
- 14-period RSI extremes (overbought/oversold)
- OANDA compatible
- Works as a third-vote confirmation only — not standalone

### momentum_sma (detect_momentum_sma)
- **Rating: MAYBE KEEP**
- Not observed firing in runtime session
- Simple SMA crossover — valid logic but not producing candidates today
- May fire in different market conditions — keep in engine, monitor

### mean_reversion_bb (detect_mean_reversion_bb)
- **Rating: MAYBE KEEP**
- Not observed firing in runtime session
- Bollinger Band mean reversion — noisy in trending markets
- Mark as low-priority signal; requires 3-vote minimum to reduce false fires

### aggressive_shorting_ob (detect_aggressive_shorting_ob)
- **Rating: MAYBE KEEP**
- Not observed in runtime output
- Order block logic — potentially useful in range-to-trend breakouts
- Requires inspection of position sizing assumptions before enabling

---

## Section 2 — Strategy Files (strategies/ folder)

These are class-based strategies (separate from the detector functions in multi_signal_engine.py).
They were NOT confirmed as active in the Phoenix runtime scan this session.
Their relationship to the scan_symbol() detectors is UNVERIFIED.

| File | Rating | Reason |
|---|---|---|
| `strategies/base.py` | MAYBE KEEP | Required base class |
| `strategies/liquidity_sweep.py` | MAYBE KEEP | Maps conceptually to liq_sweep detector; inspect if used |
| `strategies/trap_reversal_scalper.py` | MAYBE KEEP | Maps conceptually to trap_reversal; inspect if used |
| `strategies/fib_confluence_breakout.py` | MAYBE KEEP | Maps to fibonacci; inspect if used |
| `strategies/institutional_sd.py` | MAYBE KEEP | Supply/demand logic — inspect quality |
| `strategies/price_action_holy_grail.py` | MAYBE KEEP | Inspect before deploying |
| `strategies/bullish_wolf.py` | MAYBE KEEP | Wolf-pattern — historically produced trades; no runtime evidence this session |
| `strategies/bearish_wolf.py` | MAYBE KEEP | Same as bullish_wolf |
| `strategies/sideways_wolf.py` | DISCARD | Ranging-market noise; too many false fires in non-ranging conditions |
| `strategies/crypto_breakout.py` | DISCARD | Crypto-only; no OANDA use case |
| `strategies/registry.py` | MAYBE KEEP | Loader needed if strategy files are used |

---

## Section 3 — Session Bias (session_bias() in multi_signal_engine.py)

- **Rating: METADATA ONLY — not a gate**
- `session_bias()` returns (session_name, multiplier) based on UTC clock ranges
- Used legitimately as a confidence multiplier (0.90–1.0)
- **MUST NOT be used as a market-open gate**
- Clock math does not confirm broker is serving live, tradable prices
- OANDA `tradeable` flag + spread gate (broker_tradability_gate.py) is the correct market-open gate

---

## Section 4 — Hive Mind (hive/rick_hive_mind.py)

- **Rating: MAYBE KEEP (inspect dependency type)**
- Runtime evidence: `hive_conflict` blocked EUR_USD (77%), GBP_USD (76%), AUD_USD (76%) in scan at 08:47:29
- This is a significant filter — rejected 6 high-confidence signals in one cycle
- UNKNOWN whether Hive is rule-based or LLM-dependent
- If LLM-dependent: **DISCARD from production engine** — non-deterministic
- If rule-based: inspect rules, document behavior, then include with explicit narration
- **For clean repo: exclude by default until dependency is confirmed**
- **Action: read rick_hive_mind.py and report what drives the consensus before including**

---

## Section 5 — Detectors Not in Clean Repo Scope

| Source | Rating | Reason |
|---|---|---|
| `_source_repos/rick_clean_live` wolf_packs/ | DISCARD | Prototype, not runtime-proven |
| `_source_repos/rick_clean_live` logic/smart_logic.py | DISCARD | Archive only |
| `_source_repos/rick_clean_live` swarm/swarm_bot.py | DISCARD | Not OANDA trading logic |
| `ml_learning/` all files | DISCARD | ML unavailable at runtime ("basic mode" confirmed) |

---

## Final recommendation for clean repo

**Include in first deployment:**
- ema_stack, fibonacci, ema_scalper_200, fvg, liq_sweep, trap_reversal, rsi_extreme

**Disable by default, enable optionally:**
- momentum_sma, mean_reversion_bb, aggressive_shorting_ob

**Exclude until audited:**
- Hive Mind consensus gate
- All standalone strategy class files
- sideways_wolf, crypto_breakout

**Never include:**
- ML models
- Ghost/simulation engines
- Fake spread or session-based market-open gates
