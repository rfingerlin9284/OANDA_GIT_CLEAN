# PHASE7_IMPLEMENTATION_BLUEPRINT.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-17 | Label: NEW_CLEAN_REWRITE

---

## Strategy file inspection results

| File | Lines | Assessment | Verdict |
|---|---|---|---|
| `strategies/liquidity_sweep.py` | 105 | Pure candle math. `LiquiditySweepReversalStrategy`. Clean `decide_entry()`. No ML, no fakes. | **PORT NOW** |
| `strategies/trap_reversal_scalper.py` | 135 | Pure candle math. `TrapReversalScalperStrategy`. Two entry modes (base + scalper). No ML. | **PORT NOW** |
| `strategies/fib_confluence_breakout.py` | 122 | Pure candle math. `FibConfluenceBreakoutStrategy`. Swing high/low + fib retracement. No ML. | **PORT NOW** |
| `strategies/bullish_wolf.py` | 463 | pandas/RSI/BB/MACD. Technically sound, standalone signal scorer. No ML or external deps. | **HOLD FOR LATER** — needs `pandas` which is now installed; separate from scan_symbol() pipeline |
| `strategies/bearish_wolf.py` | 487 | Same as bullish_wolf — MACD/RSI/SMA, clean pandas math. | **HOLD FOR LATER** — same reason |
| `hive/rick_hive_mind.py` | 157 | Delegates to GPT, GROK, DeepSeek agents via `rick_hive_browser`. Fallback is hash-seeded (time+symbol). External LLM required. | **DISCARD** — non-deterministic, external AI dependency |

---

## The 3 strategies to implement first

**Priority order:** confidence + vote frequency in runtime session + implementation simplicity

| Priority | Strategy | Why first |
|---|---|---|
| 1 | `EMA Stack + 200 EMA Confluence` | Highest fire rate in runtime evidence. ema_stack + ema_scalper_200 fired together on every confirmed trade. Provides direction bias. |
| 2 | `Fibonacci Confluence Breakout` | Fired alongside ema_stack in all high-confidence trades. `fib_confluence_breakout.py` is 122 lines of clean candle math. Ready to port. |
| 3 | `Liquidity Sweep Reversal` | 105 lines, clean `decide_entry()`. Runtime evidence: fired on GBP_NZD (blocked by spread — signal was valid). Provides reversal entries after stop hunts. |

---

## Implementation order and rationale

### Step 1 — EMA Stack Core (already in multi_signal_engine.py)

The `detect_ema_stack()` and `detect_ema_scalper_200()` detectors already exist inside
`strategies/multi_signal_engine.py` and are called by `scan_symbol()`.

**No new file needed for Step 1.** The detectors are already pulling candidates.
Required action: verify they are wired correctly to `scan_symbol()` via `voted` list.

### Step 2 — Port `fib_confluence_breakout.py` → `strategies/fib_confluence_breakout.py`

- Source: `RBOTZILLA_PHOENIX/strategies/fib_confluence_breakout.py`
- Requires: `brokers/oanda_connector.py` (for candle data)
- Requires: `foundation/rick_charter.py` (for PIN validation)
- New dependency: `strategies/base.py` (BaseStrategy, StrategyContext, ProposedTrade)
- Action: copy `strategies/base.py` from Phoenix, then copy `fib_confluence_breakout.py`

### Step 3 — Port `liquidity_sweep.py` + `trap_reversal_scalper.py`

Both files depend on:
- `strategies/base.py` (same base class)
- Pure candle math — no other deps

Copy both after `base.py` is confirmed clean.

---

## Existing clean repo files these strategies depend on

| Dependency | Status |
|---|---|
| `strategies/multi_signal_engine.py` | PRESENT — contains `scan_symbol()` |
| `brokers/oanda_connector.py` | PRESENT — EXTRACTED_UNVERIFIED in clean repo |
| `foundation/rick_charter.py` | PRESENT — imports confirmed OK |
| `strategies/base.py` | **MISSING** — must be ported from Phoenix |
| `engine/broker_tradability_gate.py` | PRESENT — gate ready |
| `util/narration_logger.py` | PRESENT — logger ready |

---

## New files that must be created next

| File | Action | Source |
|---|---|---|
| `strategies/base.py` | COPY from Phoenix/strategies/base.py | EXTRACTED_UNVERIFIED |
| `strategies/fib_confluence_breakout.py` | COPY from Phoenix | EXTRACTED_UNVERIFIED |
| `strategies/liquidity_sweep.py` | COPY from Phoenix | EXTRACTED_UNVERIFIED |
| `strategies/trap_reversal_scalper.py` | COPY from Phoenix | EXTRACTED_UNVERIFIED |
| `strategies/strategy_runner.py` | NEW_CLEAN_REWRITE — bridges scan_symbol() and class-based strategies | NEW |
| `tests/test_strategy_signals.py` | NEW_CLEAN_REWRITE — unit tests with mock candle data | NEW |

---

## Files that must NOT be trusted yet for first deployment

| File | Do not use because |
|---|---|
| `risk/dynamic_sizing.py` | Pandas-heavy, not integrated into trade_engine.py yet; not tested |
| `risk/margin_correlation_gate.py` | `MarginCorrelationGate` class must be wired to live account NAV — not done yet |
| `risk/oco_validator.py` | `OCOValidator` class not yet wired into placement flow |
| `strategies/bullish_wolf.py` | Pandas signal scorer, not integrated with scan_symbol() |
| `strategies/bearish_wolf.py` | Same |

---

## What must remain disabled for first deployment

1. **Hive Mind** — DISCARDED. LLM + browser dependency. Non-deterministic.
2. **dynamic_sizing.py** — Wiring not verified. Use flat RBOT_BASE_UNITS env var.
3. **margin_correlation_gate.py** — Not wired to live NAV in clean engine yet.
4. **Wolf pack classes (bullish/bearish_wolf.py)** — Not wired to scan_symbol().
5. **ML features** — Not present. Do not add.
6. **session_bias() as a gate** — Clock math only. Use broker tradability gate.
