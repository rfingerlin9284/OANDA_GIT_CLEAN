#!/usr/bin/env bash
# scripts/install_sentiment_patch.sh
set -euo pipefail

# Patch broker_tradability_gate.py to support SENTIMENT_VETO_BLOCK
GATE="engine/broker_tradability_gate.py"

if ! grep -q "SENTIMENT_VETO_BLOCK" "$GATE"; then
  echo "Patching Trade Gate signature..."
  # Add reason code
  sed -i 's/GATE_ERROR                    = "TRADABILITY_CHECK_ERROR"/SENTIMENT_VETO_BLOCK          = "SENTIMENT_VETO_BLOCK"\nGATE_ERROR                    = "TRADABILITY_CHECK_ERROR"/g' "$GATE"
  
  # Alter signature to accept direction
  sed -i 's/placed_this_cycle: Optional\[Set\[str\]\] = None,/placed_this_cycle: Optional\[Set\[str\]\] = None,\n    direction: Optional\[str\] = None,/g' "$GATE"

  # Inject JSON reader BEFORE Step 5
  python3 -c '
import sys
content = open("engine/broker_tradability_gate.py").read()
patch = """    try:
        # ── 0. LLM Sentiment Edge Veto ────────────────────────────────────────
        if direction:
            import json, os
            sentiment_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "market_sentiment.json")
            if os.path.exists(sentiment_file):
                try:
                    with open(sentiment_file, "r") as f:
                        brain = json.load(f)
                    sym_data = brain.get(symbol.upper())
                    if sym_data:
                        score = sym_data.get("sentiment", 0.0)
                        conf = sym_data.get("confidence", 0.0)
                        catalyst = sym_data.get("catalyst", "Unknown")
                        if conf >= 0.5:
                            if direction.upper() == "BUY" and score <= -0.5:
                                return {"allowed": False, "event": "SENTIMENT_VETO_BLOCK", "detail": {"symbol": symbol, "reason": f"Opposing sentiment (Score: {score}) - {catalyst}"}}
                            elif direction.upper() == "SELL" and score >= 0.5:
                                return {"allowed": False, "event": "SENTIMENT_VETO_BLOCK", "detail": {"symbol": symbol, "reason": f"Opposing sentiment (Score: {score}) - {catalyst}"}}
                except Exception:
                    pass

        # ── 5. Per-cycle dedup ────────────────────────────────────────────────"""
content = content.replace("    try:\n        # ── 5. Per-cycle dedup ────────────────────────────────────────────────", patch)
open("engine/broker_tradability_gate.py", "w").write(content)
'
fi

# Patch trade_engine.py to pass direction
ENGINE="engine/trade_engine.py"
if grep -q "placed_this_cycle=placed_this_cycle," "$ENGINE"; then
  echo "Patching Trade Engine calls..."
  sed -i 's/placed_this_cycle=placed_this_cycle,/placed_this_cycle=placed_this_cycle,\n                direction=sig.direction,/g' "$ENGINE"
fi

echo "✅ Sentinel Node Integration Complete!"
