# AUDIT_LEDGER.md — RBOTZILLA_OANDA_CLEAN
Last updated: 2026-03-17

---

## ENFORCED WORKSPACE RULES

1. I cannot see your terminal unless you paste output.
2. I cannot execute code in your terminal. I propose commands only.
3. Never label anything VERIFIED without your pasted runtime proof.
4. Labels: EXTRACTED_VERIFIED | EXTRACTED_UNVERIFIED | NEW_CLEAN_REWRITE | DISCARDED | INFERRED
5. No Phoenix paths in any new clean repo file.
6. Never claim broker/runtime success without pasted evidence.
7. If manual action is required, one copy-paste block only.

---

## PHASE 1 AUDIT — 2026-03-17 (COMPLETED)

### Verified filesystem truth

All files confirmed present via `find` + `py_compile` + `unittest` run.
**.venv installed** (Python 3.12, requests + urllib3 installed).
**.env present** (credentials need manual edit before first run).

### FOUND — All project files present and syntax-clean

| File | Label | Syntax | Notes |
|---|---|---|---|
| `README.md` | NEW_CLEAN_REWRITE | — | Present |
| `.env.template` | NEW_CLEAN_REWRITE | — | Present |
| `.env` | — | — | Present — needs credentials filled |
| `requirements.txt` | NEW_CLEAN_REWRITE | — | Present |
| `.gitignore` | NEW_CLEAN_REWRITE | — | Present |
| `DISCARD_LIST.md` | NEW_CLEAN_REWRITE | — | Present |
| `AUDIT_LEDGER.md` | NEW_CLEAN_REWRITE | — | This file |
| `docs/FINAL_REBUILD_REPORT.md` | NEW_CLEAN_REWRITE | — | Present |
| `logs/.gitkeep` | — | — | Present |
| `logs/engine.log` | — | — | Present (from previous run attempt) |
| `brokers/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `brokers/oanda_connector.py` | EXTRACTED_UNVERIFIED | OK | Runtime-verified in Phoenix; not yet tested in clean repo isolation |
| `engine/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `engine/broker_tradability_gate.py` | NEW_CLEAN_REWRITE | OK | 15/15 unit tests PASSING |
| `engine/startup_sequence.py` | EXTRACTED_UNVERIFIED | OK | No Phoenix paths found; needs clean-repo import wiring check |
| `engine/trade_engine.py` | NEW_CLEAN_REWRITE | OK | Runtime UNVERIFIED |
| `engine/trade_manager.py` | NEW_CLEAN_REWRITE | OK | Runtime UNVERIFIED |
| `foundation/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `foundation/rick_charter.py` | EXTRACTED_UNVERIFIED | OK | Logic intact; not isolated-tested |
| `risk/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `risk/dynamic_sizing.py` | EXTRACTED_UNVERIFIED | OK | Not isolated-tested |
| `risk/margin_correlation_gate.py` | EXTRACTED_UNVERIFIED | OK | Live NAV fix was applied in Phoenix source |
| `risk/oco_validator.py` | EXTRACTED_UNVERIFIED | OK | Not isolated-tested |
| `strategies/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `strategies/multi_signal_engine.py` | EXTRACTED_UNVERIFIED | OK | Detectors runtime-verified in Phoenix; not tested in clean repo |
| `tests/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `tests/test_pricing_gate.py` | NEW_CLEAN_REWRITE | OK | **15/15 PASSING** |
| `tests/test_narration.py` | NEW_CLEAN_REWRITE | OK | Syntax OK; runtime UNVERIFIED |
| `util/__init__.py` | NEW_CLEAN_REWRITE | OK | Present |
| `util/mode_manager.py` | NEW_CLEAN_REWRITE | OK | Practice lock |
| `util/narration_logger.py` | NEW_CLEAN_REWRITE | OK | Syntax OK; runtime UNVERIFIED |
| `scripts/start.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths confirmed |
| `scripts/stop.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths |
| `scripts/restart.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths |
| `scripts/tail_engine.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths |
| `scripts/tail_narration.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths |
| `scripts/health_check.sh` | NEW_CLEAN_REWRITE | — | Clean repo paths |
| `.vscode/tasks.json` | NEW_CLEAN_REWRITE | valid JSON | 9 tasks, clean paths |

### MISSING — Must be created in upcoming phases

| File | Phase | Priority |
|---|---|---|
| `STRATEGY_RANKING.md` | Phase 6 | HIGH |

### NEEDS REWRITE or INSPECTION

| File | Issue | Action |
|---|---|---|
| `engine/startup_sequence.py` | Extracted from Phoenix — has Phoenix-specific init logic, class wiring not yet tested in new repo | Phase 4: inspect and patch or replace |
| `risk/margin_correlation_gate.py` | Depends on live account NAV wiring — needs clean import test | Phase 4: verify imports resolve in clean repo |
| `util/narration_logger.py` | `test_narration.py` uses `importlib.reload()` — reload behavior needs verification | Phase 11 |

---

## RUNTIME VERIFIED (from prior Phoenix session)

| Item | Evidence |
|---|---|
| OANDA practice account 101-001-31210531-001 | Pasted output 2026-03-17 |
| `/pricing` response fields: `tradeable`, `time`, `bids[0].price`, `asks[0].price` | Pasted JSON 2026-03-17 |
| CANDIDATE_FOUND → SPREAD_TOO_WIDE_BLOCK flow | Narration JSON pasted 2026-03-17 |
| CANDIDATE_FOUND → ORDER_SUBMIT_ALLOWED flow | Narration JSON pasted 2026-03-17 |
| 60–71s scan cycle timing | Timestamps in pasted narration 2026-03-17 |
| pairs_scanned: 22 (GBP_NZD removed) | Narration SIGNAL_SCAN_RESULTS pasted |

---

## WHAT REMAINS UNVERIFIED

| Item | Needed |
|---|---|
| trade_engine.py runtime | Start engine + paste first cycle narration |
| trade_manager.py trail SL path | Need open trade to reach 2R + paste TRAIL_SL_SET or TRAIL_SL_REJECTED |
| narration_logger.py runtime in clean repo | Start engine + paste any narration.jsonl entry |
| startup_sequence.py in clean repo context | Run engine + observe startup banner |
| risk/ and foundation/ imports in clean repo | Run: `.venv/bin/python -c "from risk.margin_correlation_gate import pre_trade_gate"` |
