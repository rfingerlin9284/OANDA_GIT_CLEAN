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
