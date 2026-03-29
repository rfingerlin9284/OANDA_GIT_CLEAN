# PAPER HARD STOP PROOF MISSION PROMPT

Use this workflow when the mission is to verify hard-stop behavior on the CLEAN paper account.

## Activation Rule
Treat all logs, screenshots, terminal output, and instructions in the same message as the live mission payload.

## Core Rules
- CLEAN only
- Paper account only
- Phoenix untouched
- Prefer read-only proof first
- Never claim proof unless log output confirms it
- Distinguish code defaults from live process env and live log behavior

## Required Response Structure
1. Mission Extraction
2. Evidence Split
3. Governor Map
4. Truth Verdict
5. Exact terminal-safe next step

## What must be verified
- practice account really active
- live process env value for RBOT_MAX_LOSS_USD_PER_TRADE
- code value in trade_manager.py
- post-restart log evidence of HARD_DOLLAR_STOP
- whether live behavior matches 45, not stale 16

## Output Style
- direct
- evidence-based
- no fluff
- no fake confidence
