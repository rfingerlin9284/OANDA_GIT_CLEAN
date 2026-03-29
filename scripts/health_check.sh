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
