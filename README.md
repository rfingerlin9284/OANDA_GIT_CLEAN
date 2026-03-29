# RBOTZILLA OANDA CLEAN - SYSTEM HANDOFF & README

## 🤖 FOR AI AGENTS (CHATGPT / CLAUDE / ETC)
**If you are reading this document, your task is to act as the autonomous operator and monitor for RBOTZILLA.** 
This is a fully assimilated, clean, OANDA-specific Python algorithmic trading engine. It has been stripped of legacy code and stabilized for high-frequency scanning and continuous execution.

**Your Ongoing Objectives:**
1. Ensure the engine runs continuously without crashing.
2. Read the standard output and `logs/narration.jsonl` to monitor trade lifecycles.
3. Obey the strictly enforced **Charter Rules** inside `foundation/rick_charter.py`.
4. Do **not** hallucinate or execute random broad refactors. If a bug occurs, identify the exact line, summarize the logic failure, and patch only the exact file needed.

---

## 🏗️ SYSTEM ARCHITECTURE & MODULE MAP

### 1. The Core Engine (`engine/trade_engine.py`)
- **Role:** The heartbeat of the system. It runs an asynchronous `while self.is_running` loop.
- **Cycle Flow:**
  - Queries `OANDA` for active NAV, balance, and margin.
  - Queries historical candles for all pairs in `TRADING_PAIRS`.
  - Passes data to the signal engine.
  - Filters candidates based on `MIN_CONFIDENCE`, `RickCharter` notional USD limits, and margin gates.
  - Submits valid signals to OANDA as **MARKET OCO Orders**, logging every step in the terminal.

### 2. Trade Manager & Trailing Stops (`engine/trade_manager.py` & `engine/trail_logic.py`)
- **Role:** Autonomous lifecycle management of active positions.
- **Mechanics:** 
  - Runs seamlessly every cycle via `await self.manager.tick()`.
  - Continuously syncs open broker trades into local memory.
  - Enforces the **Green-Lock** (locking the SL into profit once a minimum threshold is reached).
  - Enforces the **Hard Stop** (cutting trades instantly if USD negative PnL exceeds the allowed threshold).
  - Executes **Trailing Stops** incrementally as price moves in favor, using threshold-based logic.

### 3. The Signal Engine (`strategies/multi_signal_engine.py`)
- **Role:** Analyzes price action using 10 specialized technical detectors.
- **Detectors Included:**
  - `momentum_sma`, `ema_stack`, `fvg` (Fair Value Gap), `fibonacci`, `liq_sweep` (Liquidity Sweep), `trap_reversal`, `rsi_extreme`, `mean_reversion_bb`, `aggressive_short_ob`, `ema_scalper_200`.
- **Output:** Returns an `AggregatedSignal` object containing trade direction, calculated entry, dynamic Stop Loss (SL), and Take Profit (TP) bounds.

### 4. OANDA Connector (`brokers/oanda_connector.py`)
- **Role:** Handles all REST API communication with the OANDA v20 pricing and ordering endpoints.
- **Crucial Detail:** The `place_oco_order` function natively injects OANDA's `trailingStopLossOnFill` into every single order payload using the initial SL risk distance. **Native OANDA TS is mandatory on all orders.** 
- **Endpoint:** Strictly locked to `api-fxpractice.oanda.com` to prevent real money loss. 

---

## ⚙️ CONFIGURATION & ENVIRONMENT (`.env`)

The engine's personality and risk profile are entirely governed by the `.env` file at the repository root. 

**Key Variables you must strictly monitor:**
- `RBOT_BASE_UNITS`: The standard position size calculation factor. It must be high enough (e.g., `25000`) so that low-value quote currencies (like AUD or NZD) clear the `$15,000` notional minimum gate required by `RickCharter`.
- `RBOT_MAX_POSITIONS`: Maximum concurrent open broker trades before the engine throttles scanning.
- `RBOT_SCAN_FAST_SECONDS`: Delay between scan cycles when there are open slots in the portfolio.
- `RBOT_MIN_SIGNAL_CONFIDENCE`: The rigid minimum agreement percentage required from detectors to fire a signal.

---

## 🚦 MANDATORY RULES OF ENGAGEMENT

When operating this bot, you are strictly bound to the operator's golden rules:
1. **PRACTICE ONLY:** Do not attempt to route this engine to live money endpoints without explicit operator override. It is currently hardcoded for safety.
2. **NATIVE TS ENFORCED:** Every OCO order submitted to OANDA MUST natively include SL, TP, and TS in the JSON payload. Do not remove this from the connector.
3. **READ-ONLY NATIVE TRUTH:** Base all your diagnostics strictly on terminal output or logs pasted by the human operator. Do not guess execution states.
4. **NARRATION LOGGING:** `logs/narration.jsonl` contains structured JSON events for every major state transition (`CANDIDATE_FOUND`, `ORDER_SUBMIT_ALLOWED`, `OCO_PLACED`, `MANAGER_SYNC`). Rely on this for complex debugging.

---

## 🚀 HOW TO START AND DIAGNOSE 

**To launch the engine sequentially in the foreground (Recommended for live AI monitoring):**
```bash
cd /home/rfing/RBOTZILLA_OANDA_CLEAN
pkill -f "trade_engine.py" || true
.venv/bin/python -u engine/trade_engine.py
```

**Diagnostic Guide for standard Terminal Output:**
- Open positions are synced and stated every cycle via `[MANAGER] SYNCED [PAIR]`.
- Blocked trade candidates will show `✗ REJECTED` alongside the exact gate that blocked them.
  - *Example:* `ORDER_REJECTED: Notional $14,163.80 below Charter minimum $15,000`. (Fix: Increase `RBOT_BASE_UNITS` in `.env`).
- Successfully filled trades will instantly output `✓ OPENED [PAIR] trade_id=[ID]`.
