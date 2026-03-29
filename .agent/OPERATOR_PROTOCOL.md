# OPERATOR PROTOCOL — RBOTZILLA_OANDA_CLEAN Agent

**Last updated:** 2026-03-22

---

## WHO I AM TALKING TO

The operator (rfing) is the **human middleman** between this agent and the terminal.

- He has **no coding experience**
- He **manually pastes** every command I give him into the terminal
- He **manually pastes** the terminal output back to me
- I **cannot** access or see his terminals, VSCode integrated terminal, or any running process
- I **cannot** assume any command was run unless he pastes the output proving it

---

## MANDATORY RESPONSE RULES

### 1 — ALWAYS give exact copy-paste command blocks

Every command I want him to run must be in a fenced code block:

```bash
<exact command here — nothing abbreviated>
```

Never say "run the usual command" or "run the restart script." Always show the full exact string.

### 2 — ALWAYS tell him exactly what output to paste back

After every command block, I must say:

> **Paste the terminal output back to me so I can confirm it worked.**

Or for specific things:

> **Paste back the last 20 lines of the output so I can confirm XYZ.**

### 3 — NEVER claim something worked without output proof

If he has not pasted terminal output, I must say:

```
NOT VERIFIED YET — please paste the terminal output.
```

### 4 — NEVER give compound commands without explaining each step

If a command has multiple parts (e.g., `&&` chains), briefly label each step in plain English before the code block:

> This command does three things:
> 1. Moves into the repo folder
> 2. Checks the Python syntax of two files
> 3. Prints OK if no errors

```bash
<command>
```

### 5 — NEVER assume he knows what to do with a file path

If I reference a file, I say the full absolute path. Never say "open the file" — say:

> Open this file in VSCode: `/home/rfing/RBOTZILLA_OANDA_CLEAN/engine/trade_engine.py`

---

## STANDARD VERIFY PATTERN

After every patch I apply, I must give him this exact verify sequence:

**Step 1 — Syntax check (paste this into your terminal):**
```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN && \
PYTHONPATH=. .venv/bin/python -m py_compile <file1>.py && echo "<file1>: OK" && \
PYTHONPATH=. .venv/bin/python -m py_compile <file2>.py && echo "<file2>: OK"
```

> **Paste the output back to me. I'm looking for lines that say "OK" with no errors above them.**

**Step 2 — Restart engine (only after syntax check passes):**
```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN && bash scripts/restart.sh
```

> **Paste back the first 30 lines of output so I can confirm the engine started cleanly.**

**Step 3 — Confirm it's running:**
```bash
tail -20 /home/rfing/RBOTZILLA_OANDA_CLEAN/logs/engine_continuous.out
```

> **Paste everything that prints. I'm looking for TRADE_MANAGER_ACTIVATED and no ERROR lines.**

---

## WHAT STAYS LOCKED BY DEFAULT

See `RULE[user_global]` in the global agent settings. In short:
- No autonomous terminal execution
- No broad refactors
- No changes to live trading, OCO, SL/TP, risk, or recovery without explicit narrow scope
- No runtime claims without pasted proof

---
