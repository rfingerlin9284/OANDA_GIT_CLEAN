# RBOTZILLA_OANDA_CLEAN — Agent Context Brief
**Generated:** 2026-03-22 19:50 EDT

---

## SYSTEM ROLE

You are the engineer for **RBOTZILLA_OANDA_CLEAN** — a clean, modular OANDA practice trading system.

**Your repo:** `/home/rfing/RBOTZILLA_OANDA_CLEAN`
**DO NOT touch:** `/home/rfing/RBOTZILLA_PHOENIX` (legacy source — read only reference)

---

## WHAT THIS SYSTEM IS AND DOES

RBOTZILLA_OANDA_CLEAN is a **practice-only** OANDA trading engine that:
1. Scans 22 liquid FX pairs every cycle using a multi-detector signal engine
2. Places OCO orders (SL + TP + native OANDA trailing stop) on qualifying signals
3. Manages all open trades with a programmatic trailing stop manager (separate from broker TS)
4. **NEVER trades live money** — hard-locked to OANDA practice endpoint (account 101-001-31210531-002)

---

## CURRENT STATUS (as of 2026-03-22 19:50 EDT)

### ✅ Engine Is Running
```
Process: engine/trade_engine.py
Started: 2026-03-22 ~19:29 EDT  (via scripts/restart.sh)
Log: logs/engine_continuous.out
Narration: logs/narration.jsonl
```

### ✅ TradeManager Is Active
Confirmed from live narration.jsonl:
- `MANAGER_CYCLE_STARTED managed_count=6` — running every cycle
- `POSITION_SYNCED` events firing — picking up Phoenix-opened sandbox trades
- `TRAIL_CANDIDATE` events firing on all 6 open positions:
  - EUR_JPY BUY id=136973 @ 184.103, SL=183.849
  - AUD_CAD SELL id=136966 @ 0.96153, SL=0.96310
  - EUR_CAD BUY id=136957 @ 1.58619, SL=1.58301
  - AUD_JPY SELL id=136901 @ 111.584, SL=111.785
  - USD_CHF SELL id=136889 @ 0.78689, SL=0.78808
  - EUR_AUD BUY id=136847 @ 1.64992, SL=1.64794
  - EUR_GBP BUY id=136824 @ 0.86754, SL=0.86638

### ⚠️ No TRAIL_SL_SET events yet
Because the open positions are mostly underwater or at breakeven. TrailLogic Step 1 triggers when price moves ~4 pips into profit (major pair). Once any position hits Step 1 distance, the manager will push a new SL to OANDA and log `TRAIL_SL_SET`.

---

## ARCHITECTURE — WHAT EACH FILE DOES

| File | Role |
|---|---|
| `engine/trade_engine.py` | Main loop. Scans signals, places OCO orders, calls TradeManager each cycle |
| `engine/trade_manager.py` | Manages open trades. Syncs broker state, applies trail_logic, green-lock, hard stop |
| `engine/trail_logic.py` | Pure math: 3-step SL progression (Step1=entry lock, Step2=breakeven, Trail=follow price) |
| `engine/broker_tradability_gate.py` | Pre-trade gate: Charter rules, margin, spread, correlation |
| `engine/startup_sequence.py` | Boot validation and account check |
| `strategies/multi_signal_engine.py` | 10 detectors: EMA stack, Fibonacci, FVG, Liquidity Sweep, Trap Reversal, RSI, BB, OB, EMA200, Momentum |
| `brokers/oanda_connector.py` | OANDA API interface (practice endpoint locked) |
| `foundation/rick_charter.py` | Hard trading rules: notional min, position limits, OCO mandatory |
| `scripts/restart.sh` | Start/stop/restart engine |
| `logs/narration.jsonl` | Structured event log (TRADE_OPENED, TRAIL_SL_SET, etc.) |
| `logs/engine_continuous.out` | Live engine stdout |

---

## KEY RULES — NEVER VIOLATE

1. **Practice-only lock is permanent.** The broker connector is hard-coded to `api-fxtrade.oanda.com/v3` (practice). Do not change endpoints.
2. **OCO is mandatory.** Every new order must include SL + TP. No naked orders.
3. **TradeManager NEVER opens trades.** Only `trade_engine.py` opens. TradeManager only modifies SL.
4. **Phoenix repo is read-only.** If you need to reference Phoenix logic, grep it. Do not modify it.
5. **SL never moves backwards.** trail_logic enforces this — never override it.

---

## SIGNAL ENGINE — WHAT FINDS THE TRADES

File: `strategies/multi_signal_engine.py` — **untouched since March 17, fully working**

- **Minimum confidence:** 0.75 (per `scan_symbol` call in trade_engine)
- **Minimum votes:** 3 detectors must agree on direction
- **Top detector combos** (from 1,167 Phoenix trade history):
  - EMA stack + Fibonacci (most common, ~37%)
  - EMA stack + Fibonacci + EMA scalper 200 (second, ~31%)
  - EMA stack + Fibonacci + Liquidity Sweep (third, ~10%)
- **`AggregatedSignal.sl` and `.tp`** are always populated — SL is the median of all voting detector SLs

---

## TRAILING STOP — HOW IT WORKS

### Broker-Level (OCO placement in trade_engine.py)
```python
pip_size = 0.01 if "JPY" in symbol else 0.0001
min_ts_dist = 10.0 * pip_size          # 10-pip minimum
raw_sl_dist = abs(live_mid - sig.sl)
ts_dist = max(raw_sl_dist, min_ts_dist * 2)  # never below 20-pip distance
trailing_stop_distance=ts_dist          # passed to place_oco_order
```
OANDA requires minimum 5 pips on non-JPY, ~50 pips on JPY. The 20-pip floor clears both.

### Programmatic Trail (trade_manager.py → trail_logic.py)
Three-step progression per trade:
```
Step 1: price moves 4 pips (major) / 6 pips (minor) into profit
        → Lock SL to entry + small buffer (just above entry = cannot lose)

Step 2: price moves 8 pips into profit
        → Lock SL to 50% of gain (partial breakeven protection)

Trail:  price moves 14+ pips into profit
        → Aggressive trail: SL follows price at trail_pct distance
        → SL NEVER moves backwards
```

---

## WHAT PHOENIX DOES (for reference — DO NOT MODIFY)

Phoenix (`/home/rfing/RBOTZILLA_PHOENIX/oanda_trading_engine.py`) is the production bot that:
- Opens all trades autonomously on the sandbox account
- Manages SLs with its own TRAIL_TIGHT system (programmatic SL changes every ~5 min)
- Places native OANDA trailing stops on every new OCO entry
- Last touched: **March 17** — do not modify

Both processes run simultaneously against the same sandbox account. Phoenix opens. Clean engine monitors and applies additional trailing management.

---

## KNOWN ISSUES TO WATCH

| Issue | Status | Notes |
|---|---|---|
| `TRAILING_STOP_LOSS_ON_FILL_PRICE_DISTANCE_MINIMUM_NOT_MET` | **Fixed** | 20-pip minimum enforced in trade_engine.py lines 363-366 |
| Old errors showing in `engine.log` | **Expected** | Log not rotated. Real live output is in `engine_continuous.out` |
| `capital_router.py` and `pre_market_scanner.py` | **Newly added** | Added 2026-03-22 — verify these integrate cleanly with trade_engine.py |
| No `TRAIL_SL_SET` events yet | **Expected** | Will fire once any position hits Step 1 profit distance |

---

## WHAT STILL NEEDS TO BE DONE

1. **Verify `capital_router.py` and `pre_market_scanner.py`** are wired into `trade_engine.py` correctly — they were created today (March 22) and need validation
2. **Watch for first `TRAIL_SL_SET`** — confirms end-to-end trail path works
3. **Create `backup_20260322/`** — the current upgraded engine/ files have no backup yet
4. **Watch for `TRADE_OPENED` events** from clean engine — confirms OCO orders land correctly with the 20-pip TS floor

---

## HOW TO CHECK SYSTEM STATUS

```bash
# Live narration events
tail -f /home/rfing/RBOTZILLA_OANDA_CLEAN/logs/narration.jsonl

# Live engine output
tail -f /home/rfing/RBOTZILLA_OANDA_CLEAN/logs/engine_continuous.out

# Check what's running
ps aux | grep trade_engine

# Restart engine
cd /home/rfing/RBOTZILLA_OANDA_CLEAN && bash scripts/restart.sh

# Syntax check all engine files
cd /home/rfing/RBOTZILLA_OANDA_CLEAN && \
PYTHONPATH=. .venv/bin/python -m py_compile engine/trail_logic.py && echo "trail_logic: OK" && \
PYTHONPATH=. .venv/bin/python -m py_compile engine/trade_manager.py && echo "trade_manager: OK" && \
PYTHONPATH=. .venv/bin/python -m py_compile engine/trade_engine.py && echo "trade_engine: OK"
```
