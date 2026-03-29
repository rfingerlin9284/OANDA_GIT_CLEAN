# DISCARD_LIST.md — Phase 2
# RBOTZILLA_OANDA_CLEAN — Items that must NOT enter the new repo

Generated: 2026-03-17
Rule: If it appears here, it may NOT be copied into RBOTZILLA_OANDA_CLEAN under any label.

---

## CATEGORY 1 — Fake Data / Simulation

| Item | Location | Reason |
|---|---|---|
| `evaluate_news_spread()` | `oanda_trading_engine.py` line 3757 | Code comment: "generic spread simulation until dynamic spread API available" — not real broker data |
| `session_bias()` as a tradability gate | `systems/multi_signal_engine.py` lines 106-151 | Clock-only math. Does not confirm market is open. Must NEVER be used as a trade gate. |
| Any hardcoded spread constant not from live broker | Any | Spread must always come from `/pricing` endpoint |
| `ghost_trading_engine.py` | `rick_clean_live/RICK_LIVE_PROTOTYPE/` | Explicitly ghost/simulated fills |
| `live_ghost_engine.py` | `rick_clean_live/RICK_LIVE_PROTOTYPE/` | Ghost/simulated fills |

---

## CATEGORY 2 — Ghost Trail Logic

| Item | Location | Reason |
|---|---|---|
| Old `set_trade_stop()` call without response check | Pre-patch Phoenix engine | Logged TRAIL_TIGHT_SET without confirming broker accepted stop. |
| Any trail SL update that does not check API response | Any | Must verify `success` field in response before updating local state |

---

## CATEGORY 3 — Margin Gate Defaults that Don't Use Live Values

| Item | Location | Reason |
|---|---|---|
| Hardcoded NAV fallback `$1,970` | Pre-patch Phoenix line ~1435 | Would allow over-margin trades when account value changed |
| `margin_used=0` passed into gate without broker confirmation | Pre-patch Phoenix margin gate | Gate would always pass with zero margin used as default |

---

## CATEGORY 4 — Misleading "Placing" Logs

| Item | Reason |
|---|---|
| Any `→ Placing` print before broker tradability check | Gate must pass first. CANDIDATE_FOUND → gate → ORDER_SUBMIT_ALLOWED → Placing |
| `TRAIL_TIGHT_SET` without confirming API response | Must verify before narrating success |
| `TRADE_OPENED` narration before trade ID is returned | Only narrate on confirmed broker response |

---

## CATEGORY 5 — Stale Price / Closed Market Trading

| Item | Reason |
|---|---|
| `get_historical_data()` candle timestamp used as "live price" | Candle close is not a live tradeable price |
| Any trade placement without first calling `/pricing` | Required for freshness + spread check |
| Stale candle check only (old >1 hour check) | OANDA serves weekend cached candles without flagging them as stale |

---

## CATEGORY 6 — Silent API Failures

| Item | Reason |
|---|---|
| `except Exception: pass` without narration | Swallowed errors hide ghost fills |
| `set_trade_stop()` without return value check | Previously caused ghost trail SL |
| `place_oco_order()` without trade ID confirmation | Must confirm returned `tradeID` |
| Any API request that uses `try/except` without logging the failure | Every failure must appear in narration |

---

## CATEGORY 7 — Duplicated Repo Junk

| Item | Reason |
|---|---|
| `_source_repos/` subfolder contents (wholesale) | Source reference only — not production |
| `dec4_dec10/` | Old proposals, not production |
| `ROLLBACK_SNAPSHOTS/` | Snapshots only |
| `_archive_scripts/` | Archived |
| `.progress_backups/`, `.operator_fix_backups/` | Backup scaffolding |
| `canary_trading_engine_OLD_DEPRECATED.py` | Explicitly deprecated |
| `backups/`, `charter_backups/` | Backup folders |

---

## CATEGORY 8 — Dead Tasks and Broken Scripts

| Item | Reason |
|---|---|
| `train_initial_ml_model.py` | ML unavailable at runtime |
| `audit_today_trades_et.py`, `edge_kpi_report_et.py` | ET timezone, not portable |
| `coinbase_headless.py` | Coinbase, not OANDA |
| `build_prototype_backup.sh` | Repo management |
| `qc_sl_test.py` | One-off QC script |
| `fix_oanda.py` | One-off fix script |
| `test_der.py`, `test_hmac_api.py`, `test_margin.py` | One-off test scripts |

---

## CATEGORY 9 — Non-OANDA Brokers

| Item | Reason |
|---|---|
| `brokers/alpaca_connector.py` | Not OANDA |
| `brokers/ib_connector.py` | Not OANDA |
| `brokers/coinbase_connector.py` | Not OANDA |
| `brokers/coinbase_advanced_connector.py` | Not OANDA |
| `connectors/futures/` | Futures, not forex |
| `connectors/yahoo_cryptopanic_connector.py` | News, not needed |
| `ibkr_gateway/` | IBKR |
| `coinbase_advanced/` | Coinbase |

---

## CATEGORY 10 — Unverified ML / AI / Browser Systems

| Item | Reason |
|---|---|
| `ml_learning/` (all files) | ML not available at runtime. Engine falls back to "basic mode". |
| `hive/browser_ai_connector.py`, `hive/rick_hive_browser.py` | Browser automation. Not trading logic. |
| `hive/hive_llm_orchestrator.py` | LLM dependency. Non-deterministic. |
| `hive/rick_local_ai.py` | Local AI. Not verified in production. |
| `hive/rick_hive_browser.py` | Browser automation. |
| `maintenance_agent.py`, `audit_self_optimization.py` | Agent scaffolding. |

---

## CATEGORY 11 — Multi-Broker Orchestration (Out of Scope)

| Item | Reason |
|---|---|
| `hedge_fund_orchestrator.py`, `orchestrator_start.py` | Multi-broker routing. Not needed. |
| `systems/hedge_fund_orchestrator.py` | Same. |
| `autonomous_boot.py` | Multi-system boot. Not needed for OANDA-only. |
| `headless_runtime.py` | Headless multi-system mode. |

---

## NOTE: Hive Mind (hive/rick_hive_mind.py)

The `hive_conflict` gate is rejecting legitimate high-confidence signals.
Status: **DO NOT DISCARD YET** — inspect first.
In Phase 6, the `hive_conflict` rejection pattern will be reviewed.
If the Hive is rule-based (not LLM-dependent), it may be kept with documentation.
If LLM-dependent, it must be excluded from the clean engine.
