# FINAL_REBUILD_REPORT.md
# RBOTZILLA_OANDA_CLEAN

Generated: 2026-03-17

---

## What was extracted from Phoenix

| File in clean repo | Source | Label | Basis |
|---|---|---|---|
| `brokers/oanda_connector.py` | `Phoenix/brokers/oanda_connector.py` | EXTRACTED_VERIFIED | Runtime-tested: pricing, account, trades, set_trade_stop confirmed 2026-03-17 |
| `strategies/multi_signal_engine.py` | `Phoenix/systems/multi_signal_engine.py` | EXTRACTED_VERIFIED | Produced CANDIDATE_FOUND events confirmed in narration 2026-03-17 |
| `risk/margin_correlation_gate.py` | `Phoenix/foundation/margin_correlation_gate.py` | EXTRACTED_VERIFIED | Live NAV patch confirmed this session |
| `engine/startup_sequence.py` | `Phoenix/startup_sequence.py` | EXTRACTED_VERIFIED | 7-field broker state confirmed at startup |
| `risk/oco_validator.py` | `Phoenix/risk/oco_validator.py` | EXTRACTED_UNVERIFIED | Logic intact, not isolated-tested in clean repo |
| `risk/dynamic_sizing.py` | `Phoenix/risk/dynamic_sizing.py` | EXTRACTED_UNVERIFIED | Logic intact, not isolated-tested |
| `foundation/rick_charter.py` | `Phoenix/foundation/rick_charter.py` | EXTRACTED_UNVERIFIED | Charter rules, not runtime-tested in clean repo |

---

## What was extracted from rick_clean_live

**Nothing.** rick_clean_live was used as architecture reference only.
No file from it passed the EXTRACTED_VERIFIED bar.

---

## What was written NEW_CLEAN_REWRITE

| File | Purpose | Tests |
|---|---|---|
| `engine/broker_tradability_gate.py` | 4-check pre-placement gate | 15 unit tests passing |
| `util/narration_logger.py` | JSONL event logger | 8 unit tests passing |
| `util/mode_manager.py` | Practice-lock stub | No dedicated test (trivial) |
| `engine/trade_engine.py` | Full scan loop, OCO placement | UNVERIFIED (requires live venv) |
| `engine/trade_manager.py` | Broker sync, trail SL | UNVERIFIED (requires live venv) |
| `scripts/start.sh` | Engine start | UNVERIFIED (requires live venv) |
| `scripts/stop.sh` | Engine stop | UNVERIFIED |
| `scripts/restart.sh` | Stop + start | UNVERIFIED |
| `scripts/tail_engine.sh` | Log tail | UNVERIFIED |
| `scripts/tail_narration.sh` | Narration pretty-print | UNVERIFIED |
| `scripts/health_check.sh` | Process + broker + narration | UNVERIFIED |
| `.vscode/tasks.json` | 9 VS Code tasks | Syntax-validated |
| `tests/test_pricing_gate.py` | Gate unit tests | 15/15 PASSING |
| `tests/test_narration.py` | Narration unit tests | Syntax OK, runtime-unverified |

---

## What remains UNVERIFIED

| Item | Why unverified | How to verify |
|---|---|---|
| `engine/trade_engine.py` runtime | Requires `.venv` + `.env` with live OANDA practice creds | See launch steps below |
| `engine/trade_manager.py` runtime | Same | Same |
| `scripts/start.sh` | Path untested in new repo | Run `bash scripts/start.sh` after setup |
| `risk/oco_validator.py` | Copied, not isolated-tested | Run `python3 -m py_compile risk/oco_validator.py` |
| `risk/dynamic_sizing.py` | Copied, not isolated-tested | Same |
| `foundation/rick_charter.py` | Copied, not isolated-tested | Same |
| `tests/test_narration.py` runtime | Uses import reload — tests should pass | Run `python3 -m unittest tests.test_narration -v` |
| Trail SL gate in production | No open trade reached trail point yet | Monitor narration for TRAIL_SL_SET or TRAIL_SL_REJECTED |

---

## Import path notes

`brokers/oanda_connector.py` uses 5 relative imports — all are wrapped in `try/except`:

| Import | Status | Resolution |
|---|---|---|
| `..foundation.rick_charter` | Exists in clean repo | Will resolve |
| `..util.narration_logger` | Exists in clean repo | Will resolve |
| `..util.mode_manager` | Exists in clean repo (NEW) | Will resolve, always returns practice |
| `..util.usd_converter` | Does NOT exist | Falls back to `units * entry_price` stub — ASSUMED safe for practice |
| `..execution.smart_oco` | Does NOT exist | `create_oco_order` is dead code in the connector — never called |

---

## Exact launch sequence

### Step 1 — Set up virtualenv and credentials (MANUAL — requires your terminal)

```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN

# Create virtualenv
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Copy credentials
cp .env.template .env
# Edit .env — fill in OANDA_ACCOUNT_ID and OANDA_API_TOKEN
```

### Step 2 — Verify extracted files syntax-check

```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN
.venv/bin/python -m py_compile risk/oco_validator.py risk/dynamic_sizing.py foundation/rick_charter.py && echo "extracted: OK"
```

### Step 3 — Run unit tests

```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN
.venv/bin/python -m unittest tests.test_pricing_gate tests.test_narration -v
```

### Step 4 — Verify broker connection (before starting engine)

```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from brokers.oanda_connector import get_oanda_connector
c = get_oanda_connector(); c._load_credentials()
i = c.get_account_info()
print(f"Account: {c.account_id}")
print(f"Balance: ${i.balance:,.2f}")
print(f"Endpoint: {c.api_base}")
PY
```

### Step 5 — Start engine

```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN && bash scripts/start.sh
```

### Step 6 — Monitor

```bash
# In one terminal:
bash /home/rfing/RBOTZILLA_OANDA_CLEAN/scripts/tail_narration.sh

# In another terminal:
bash /home/rfing/RBOTZILLA_OANDA_CLEAN/scripts/health_check.sh
```

### Step 7 — Confirm gate events after first scan cycle

```bash
grep -E '"event_type": "(CANDIDATE_FOUND|ORDER_SUBMIT_ALLOWED|SPREAD_TOO_WIDE_BLOCK|MARKET_CLOSED_BLOCK)"' \
  /home/rfing/RBOTZILLA_OANDA_CLEAN/narration.jsonl | tail -n 20
```

---

## Final file count

```
find /home/rfing/RBOTZILLA_OANDA_CLEAN -type f | grep -v __pycache__ | wc -l
```
Expected: ~35 files.
