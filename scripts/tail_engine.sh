#!/usr/bin/env bash
# tail_engine.sh — RBOTZILLA_OANDA_CLEAN
# Label: NEW_CLEAN_REWRITE
set -euo pipefail
REPO="/home/rfing/RBOTZILLA_OANDA_CLEAN"
tail -f "$REPO/logs/engine.log"
