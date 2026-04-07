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

# ── Clock sync status (chrony runs continuously — no sudo needed) ─────────────
if command -v chronyc > /dev/null 2>&1; then
  CHRONY=$(chronyc tracking 2>/dev/null | grep -E "System time|RMS offset|Stratum" | tr '\n' '  ' | xargs || true)
  echo "  🕐 Clock: ${CHRONY:-chrony active}"
elif command -v timedatectl > /dev/null 2>&1; then
  NTP_STATUS=$(timedatectl status 2>/dev/null | grep -E "synchronized|NTP service" | tr '\n' ' ' | xargs || true)
  echo "  🕐 Clock: ${NTP_STATUS:-NTP active}"
fi

# Kill any stale instance
pkill -f "engine.trade_engine" || true; pkill -f "trade_engine.py" > /dev/null 2>&1 || true
sleep 1

# Clear Python bytecache so code changes always take effect
find "$REPO" -name "*.pyc" -delete 2>/dev/null || true
find "$REPO" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Prevent Python from EVER creating bytecache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO"

# Load .env — strip full-line comments, blank lines, and inline comments
# before bash exports them. Prevents inline # comments from poisoning float() calls.
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source <(grep -v '^\s*#' "$REPO/.env" | grep -v '^\s*$' | sed 's/[[:space:]]*#[^"]*$//')
  set +a
fi

# Truncate log to avoid binary content from previous runs
: > "$LOG"

nohup "$VENV" -B -u -m engine.trade_engine > "$LOG" 2>&1 &

sleep 2
if pgrep -af "engine.trade_engine|trade_engine.py" > /dev/null; then
  echo "OK: RBOTZILLA_OANDA_CLEAN engine started (log: $LOG)"
  echo ""
  sleep 3
  tail -n 25 "$LOG" | grep -E -A 15 "RBOTZILLA OANDA CLEAN — PRACTICE" || true
  echo ""
else
  echo "FAIL: engine not running — check $LOG"
  exit 1
fi
