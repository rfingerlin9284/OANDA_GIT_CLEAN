#!/usr/bin/env bash
# =============================================================================
# ALL_SCRIPTS_REFERENCE.sh — RBOTZILLA_OANDA_CLEAN
# Snapshot Date: 2026-03-23
# Purpose: Single-file reference copy of every script linked to a VS Code
#          Run Task in this version. DO NOT run this file directly.
#          Each section is the exact script for its task.
# =============================================================================

# =============================================================================
# TASK: 🟢 Start Engine
# File: scripts/start.sh
# Command: bash scripts/start.sh
# =============================================================================
: <<'START_ENGINE'
#!/usr/bin/env bash
# start.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE (paths correct for this repo)
set -euo pipefail

REPO="/home/rfing/RBOTZILLA_OANDA_CLEAN"
ENGINE="engine/trade_engine.py"
LOG="logs/engine_continuous.out"
VENV=".venv/bin/python"

cd "$REPO"
mkdir -p logs

if [ ! -x "$VENV" ]; then
  echo "FAIL: missing $REPO/.venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Kill any stale instance
pkill -f "engine.trade_engine|trade_engine.py" >/dev/null 2>&1 || true
sleep 1

export PYTHONPATH="$REPO"

# Load .env so all operator-configured values reach the engine process
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

nohup "$VENV" -u -m engine.trade_engine > "$LOG" 2>&1 &

sleep 2
if pgrep -af "engine.trade_engine|trade_engine.py" >/dev/null; then
  echo "OK: RBOTZILLA_OANDA_CLEAN engine started (log: $LOG)"
else
  echo "FAIL: engine not running — check $LOG"
  exit 1
fi
START_ENGINE

# =============================================================================
# TASK: 🔴 Stop Engine
# File: scripts/stop.sh
# Command: bash scripts/stop.sh
# =============================================================================
: <<'STOP_ENGINE'
#!/usr/bin/env bash
# stop.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
set -euo pipefail

pkill -f "trade_engine.py" >/dev/null 2>&1 && echo "OK: RBOTZILLA_OANDA_CLEAN engine stopped" || echo "OK: engine was not running"
STOP_ENGINE

# =============================================================================
# TASK: 🔄 Restart Engine
# File: scripts/restart.sh
# Command: bash scripts/restart.sh
# =============================================================================
: <<'RESTART_ENGINE'
#!/usr/bin/env bash
# restart.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/stop.sh"
sleep 1
bash "$SCRIPT_DIR/start.sh"
echo "OK: RBOTZILLA_OANDA_CLEAN engine restarted"
RESTART_ENGINE

# =============================================================================
# TASK: 📡 LIVE TAIL — Engine Output (engine_continuous.out)
# File: scripts/tail_engine.sh  (+ inline command in tasks.json)
# Command: tail -f /home/rfing/RBOTZILLA_OANDA_CLEAN/logs/engine_continuous.out
# =============================================================================
: <<'TAIL_ENGINE'
#!/usr/bin/env bash
# tail_engine.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
set -euo pipefail
REPO="/home/rfing/RBOTZILLA_OANDA_CLEAN"
tail -f "$REPO/logs/engine.log"
TAIL_ENGINE

# =============================================================================
# TASK: 📜 LIVE TAIL — Narration Events (narration.jsonl)
# File: scripts/tail_narration.sh  (+ inline command in tasks.json)
# Command: tail -f .../narration.jsonl | python3 -c "..."
# =============================================================================
: <<'TAIL_NARRATION'
#!/usr/bin/env bash
# tail_narration.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
set -euo pipefail
REPO="/home/rfing/RBOTZILLA_OANDA_CLEAN"
tail -f "$REPO/narration.jsonl" | while IFS= read -r line; do
  echo "$line" | python3 -c "
import sys, json
try:
  e = json.loads(sys.stdin.read())
  ts  = e.get('timestamp','')[-12:-7]
  evt = e.get('event_type','?')
  sym = e.get('symbol','?')
  det = e.get('details',{})
  print(f'  {ts}  {evt:<38}  {sym:<10}  {det}')
except Exception:
  pass
"
done
TAIL_NARRATION

# =============================================================================
# TASK: 🏥 Health Check
# File: scripts/health_check.sh
# Command: bash scripts/health_check.sh
# =============================================================================
: <<'HEALTH_CHECK'
#!/usr/bin/env bash
# health_check.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
# Checks: engine process, recent narration events, broker open trades
set -euo pipefail

REPO="/home/rfing/RBOTZILLA_OANDA_CLEAN"
cd "$REPO"

echo ""
echo "=== ENGINE PROCESS ==="
if pgrep -af "trade_engine.py" >/dev/null; then
  echo "RUNNING"
else
  echo "NOT RUNNING"
fi

echo ""
echo "=== LAST 5 NARRATION EVENTS ==="
if [ -f narration.jsonl ]; then
  tail -n 5 narration.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
  try:
    e = json.loads(line)
    print(f\"  {e.get('timestamp','')[-25:-7]}  {e.get('event_type',''):38}  {e.get('symbol','')}\")
  except Exception:
    pass
"
else
  echo "narration.jsonl not found"
fi

echo ""
echo "=== OPEN BROKER TRADES ==="
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, ".")
from brokers.oanda_connector import get_oanda_connector
try:
    c = get_oanda_connector()
    c._load_credentials()
    trades = c.get_trades() or []
    print(f"  Broker reports {len(trades)} open trade(s)")
    for t in trades:
        sym  = t.get("instrument","?")
        tid  = t.get("id","?")
        upl  = t.get("unrealizedPL","?")
        units = t.get("currentUnits","?")
        print(f"    {sym}  id={tid}  units={units}  UPL={upl}")
except Exception as e:
    print(f"  ERROR: {e}")
PY

echo ""
HEALTH_CHECK

# =============================================================================
# TASK: 🔒 Lock All Code (Read-Only)
# Inline command in tasks.json (no separate script file)
# Command:
#   find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.py' ! -path '*/logs/*' -exec chmod 444 {} \;
#   find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.sh' ! -path '*/logs/*' -exec chmod 444 {} \;
# =============================================================================
: <<'LOCK_CODE'
find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.py' ! -path '*/logs/*' -exec chmod 444 {} \; \
  && find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.sh' ! -path '*/logs/*' -exec chmod 444 {} \; \
  && echo '✅ All .py and .sh files locked to read-only (444)'
LOCK_CODE

# =============================================================================
# TASK: 🔓 Unlock All Code (Writable)
# Inline command in tasks.json (no separate script file)
# Command:
#   find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.py' ! -path '*/logs/*' -exec chmod 644 {} \;
#   find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.sh' ! -path '*/logs/*' -exec chmod 755 {} \;
# =============================================================================
: <<'UNLOCK_CODE'
find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.py' ! -path '*/logs/*' -exec chmod 644 {} \; \
  && find /home/rfing/RBOTZILLA_OANDA_CLEAN -name '*.sh' ! -path '*/logs/*' -exec chmod 755 {} \; \
  && echo '✅ All .py and .sh files unlocked (644/755)'
UNLOCK_CODE

echo "This file is a reference only. Do not execute directly."
