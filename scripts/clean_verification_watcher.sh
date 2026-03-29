#!/usr/bin/env bash
set -euo pipefail

cd /home/rfing/RBOTZILLA_OANDA_CLEAN

echo "READ-ONLY CLEAN verification watcher starting..."
echo "No code changes. No Phoenix. No patching. Just truth."

while true; do
  clear
  echo "============================================================"
  echo "RBOTZILLA CLEAN - READ ONLY LIVE VERIFICATION WATCHER"
  echo "============================================================"
  date
  echo

  pid="$(pgrep -f 'engine.trade_engine|trade_engine.py' | tail -n 1 || true)"
  echo "=== ENGINE PID ==="
  if [ -n "${pid:-}" ]; then
    echo "pid=$pid"
  else
    echo "FAIL: engine not running"
  fi
  echo

  echo "=== START LINE ==="
  start_line="$(grep -n 'RBOTZILLA OANDA CLEAN — PRACTICE ENGINE' logs/engine_continuous.out | tail -n 1 | cut -d: -f1 || true)"
  if [ -z "${start_line:-}" ]; then
    start_line=1
  fi
  echo "$start_line"
  echo

  echo "=== LIVE ENV HARD STOP ==="
  if [ -n "${pid:-}" ] && [ -r "/proc/$pid/environ" ]; then
    tr '\0' '\n' < "/proc/$pid/environ" | grep '^RBOT_MAX_LOSS_USD_PER_TRADE=' || echo "not found in live env"
  else
    echo "proc env unavailable"
  fi
  echo

  echo "=== CURRENT BROKER OPEN TRADES ==="
  .venv/bin/python - <<'PY'
from brokers.oanda_connector import get_oanda_connector, get_usd_notional

TARGET = 17500.0
c = get_oanda_connector()
trades = c.get_trades() or []
prices = c.get_live_prices([str(t.get("instrument")) for t in trades]) or {}

print(f"open_trades={len(trades)}")
for t in trades:
    inst = str(t.get("instrument"))
    units = int(float(t.get("currentUnits") or t.get("initialUnits") or 0))
    side = "BUY" if units > 0 else "SELL" if units < 0 else "FLAT"
    px = float(((prices.get(inst) or {}).get("mid")) or float(t.get("price") or 0.0) or 0.0)
    usd_notional = float(get_usd_notional(abs(units), inst, px)) if px > 0 else 0.0
    delta = usd_notional - TARGET
    kind = "HEDGE_OR_LEGACY" if abs(units) <= 1000 else "MAIN"
    print(f"{inst:10s} {side:4s} units={units:>7d} usd_notional={usd_notional:>10.2f} delta_vs_17500={delta:>8.2f} kind={kind} id={t.get('id')}")
PY
  echo

  echo "=== POST-RESTART COUNTS ==="
  reject_count="$(sed -n "${start_line},\$p" logs/engine_continuous.out | grep -c 'ORDER REJECTED' || true)"
  open_count="$(sed -n "${start_line},\$p" logs/engine_continuous.out | grep -c '✓ OPENED' || true)"
  close_count="$(sed -n "${start_line},\$p" logs/engine_continuous.out | grep -c '\[MANAGER\] CLOSED' || true)"
  hedge_count="$(sed -n "${start_line},\$p" logs/engine_continuous.out | grep -c 'HEDGE' || true)"
  hard_stop_count="$(sed -n "${start_line},\$p" logs/engine_continuous.out | grep -c 'HARD_DOLLAR_STOP' || true)"
  echo "reject_count=$reject_count"
  echo "open_count=$open_count"
  echo "close_count=$close_count"
  echo "hedge_count=$hedge_count"
  echo "hard_stop_count=$hard_stop_count"
  echo

  echo "=== POST-RESTART WATERMARK / ROUTER / HARD STOP LINES ==="
  sed -n "${start_line},\$p" logs/engine_continuous.out | \
    grep -E 'CapitalRouter ACTIVE|watermark=|ROUTER|HARD_DOLLAR_STOP|uPnL=.*CLOSED|ORDER REJECTED' | tail -n 40 || true
  echo

  echo "=== POST-RESTART OPEN / CLOSE / HEDGE LINES ==="
  sed -n "${start_line},\$p" logs/engine_continuous.out | \
    grep -E 'OPENED|HEDGE|CLOSED' | tail -n 60 || true
  echo

  echo "=== TRUTH VERDICT ==="
  if [ "$reject_count" -eq 0 ]; then
    echo "reject_status=CLEAN"
  else
    echo "reject_status=HAS_REJECTS"
  fi

  if sed -n "${start_line},\$p" logs/engine_continuous.out | grep -q 'watermark='; then
    echo "watermark_logic=INSTALLED"
  else
    echo "watermark_logic=NOT_VISIBLE"
  fi

  if sed -n "${start_line},\$p" logs/engine_continuous.out | grep -q 'HARD_DOLLAR_STOP'; then
    echo "hard_stop_live=SEEN_IN_LOG"
  else
    echo "hard_stop_live=NOT_TRIGGERED_SINCE_RESTART"
  fi

  mains_above_target="$(
    .venv/bin/python - <<'PY'
from brokers.oanda_connector import get_oanda_connector, get_usd_notional
TARGET = 17500.0
c = get_oanda_connector()
trades = c.get_trades() or []
prices = c.get_live_prices([str(t.get("instrument")) for t in trades]) or {}
count = 0
for t in trades:
    units = int(float(t.get("currentUnits") or t.get("initialUnits") or 0))
    if abs(units) <= 1000:
        continue
    inst = str(t.get("instrument"))
    px = float(((prices.get(inst) or {}).get("mid")) or float(t.get("price") or 0.0) or 0.0)
    usd_notional = float(get_usd_notional(abs(units), inst, px)) if px > 0 else 0.0
    if usd_notional > TARGET + 250:
        count += 1
print(count)
PY
  )"
  echo "main_positions_above_17500_plus_buffer=$mains_above_target"

  if [ "$mains_above_target" -gt 0 ]; then
    echo "compounding_live_verdict=POSSIBLE_BUT_CONFIRM_WITH_WATERMARK_RISE"
  else
    echo "compounding_live_verdict=NOT_YET_PROVEN"
  fi

  echo
  echo "Refreshing in 30s... old-school heartbeat, no nonsense."
  sleep 30
done
