# SOURCE_BEHAVIOR_MAP.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-19 | Based on workspace inspection only.
Labels: VERIFIED (runtime-pasted proof) | INFERRED (code inspection) | UNVERIFIED (assumed)

---

## 1. Source Repo Identity

| Property | Value | Label |
|---|---|---|
| Repo path | `/home/rfing/RBOTZILLA_PHOENIX` | INFERRED |
| Primary engine file | `oanda_trading_engine.py` | INFERRED |
| Launcher script | `scripts/task_restart_practice.sh` | INFERRED |
| Start script | `scripts/task_start_practice.sh` | INFERRED |
| Stop script | `scripts/task_stop_practice.sh` | INFERRED |
| Python binary | `.venv/bin/python` | INFERRED |
| Log file | `logs/practice_session.log` | INFERRED |
| Process is currently running | UNKNOWN | UNVERIFIED — needs terminal proof |

> Basis: launcher script content confirmed by workspace read.  
> RBOTZILLA_OANDA_CLEAN trade_engine.py has never successfully placed a trade (blocked by notional). All broker trades confirmed in screenshots (tickets 41023–47193) pre-date the clean repo engine start.

---

## 2. Signal Generation

| Property | Value | Label |
|---|---|---|
| Signal file | `strategies/multi_signal_engine.py` | INFERRED |
| Strategy registry | `strategies/registry.py` | INFERRED |
| Entry function | `scan_symbol()` | INFERRED |
| Detector stack | ema_stack, fibonacci, ema_scalper_200, fvg, liq_sweep, trap_reversal, rsi_extreme | INFERRED |
| Aggregation | vote-weighted confidence scoring | INFERRED |
| Min confidence threshold | 0.68 (configurable) | INFERRED |
| Min votes | 3 | INFERRED |

---

## 3. Session Handling

| Property | Value | Label |
|---|---|---|
| Session gate | UTC clock-based (London/NY/Asia) | INFERRED |
| Session multiplier applied | YES — in `scan_symbol()` | INFERRED |
| Fake session label used as open gate | NO (runtime-confirmed March 2026) | INFERRED |

---

## 4. Pair Universe

| Property | Value | Label |
|---|---|---|
| Configured pairs | EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD, EUR_GBP, EUR_JPY, GBP_JPY, EUR_AUD, EUR_CAD, AUD_JPY, CHF_JPY, GBP_CAD, GBP_CHF, AUD_CAD, AUD_CHF, CAD_JPY, NZD_JPY, EUR_NZD, NZD_CAD + GBP_NZD | INFERRED |
| Configurable via env | YES (`RBOT_TRADING_PAIRS`) | INFERRED |

---

## 5. OCO Placement

| Property | Value | Label |
|---|---|---|
| Order file | `brokers/oanda_connector.py` | INFERRED |
| Key function | `place_oco_order(instrument, entry_price, stop_loss, take_profit, units, ttl_hours, order_type)` | INFERRED |
| SL required | YES — mandatory | INFERRED |
| TP required | YES — mandatory | INFERRED |
| Order type used | `MARKET` | INFERRED |
| Charter minimum notional enforced | YES — $15,000 USD | INFERRED |
| Broker confirms trade_id before logging TRADE_OPENED | UNKNOWN — depends on connector impl | UNVERIFIED |

---

## 6. Dedup Logic

| Property | Value | Label |
|---|---|---|
| Per-cycle symbol dedup | YES — `placed_this_cycle` set | INFERRED |
| Broker-live position dedup | YES — `get_trades()` checked | INFERRED |
| Cooldown period after placement | PRESENT in Phoenix engine | INFERRED |

---

## 7. Trailing Stop Behavior

| Property | Value | Label |
|---|---|---|
| Trailing stop code present | YES — `rbz_tight_trailing.py`, `util/momentum_trailing.py`, `SmartTrailingSystem` | INFERRED (code read) |
| `RBZ_TRAILING_AVAILABLE` flag | Set at import time — fails silently to `False` if import fails | INFERRED (code read) |
| Trailing start condition | After 3 pips profit (`trailing_start_pips=3`) | INFERRED |
| Trail distance | 5 pips (`trailing_dist_pips=5`) | INFERRED |
| Default management profile | `"OCO SL/TP; 0.75R→BE; 1.5R→scale50%; 2R→trail; hard_stop_usd"` | INFERRED (code read) |
| TS column in OANDA broker UI | **BLANK on all 6 trades** | VERIFIED (screenshots 9:17 AM and 12:41 PM both confirmed) |
| Trailing stop actually active | **NO — TS blank on all trades in broker UI** | INFERRED from screenshot VERIFIED |
| Root cause of trail not active | UNKNOWN — import failure, logic gate, or dormant code | UNVERIFIED — needs log proof |

---

## 8. Trade Manager

| Property | Value | Label |
|---|---|---|
| Trade manager present | YES — within `oanda_trading_engine.py` | INFERRED |
| Trail submit to broker | PRESENT in code | INFERRED |
| Trail confirmed before log | UNKNOWN | UNVERIFIED |
| Sync from broker on restart | UNKNOWN | UNVERIFIED |

---

## 9. Map: Keep / Extract / Discard

| Component | Source File | Action |
|---|---|---|
| Signal detectors (ema_stack, fib, fvg, liq_sweep, trap_rev, rsi_extreme) | `strategies/multi_signal_engine.py` | ALREADY EXTRACTED |
| `scan_symbol()` entry aggregator | `strategies/multi_signal_engine.py` | ALREADY EXTRACTED |
| OCO placement | `brokers/oanda_connector.py` | ALREADY EXTRACTED |
| Base strategy classes | `strategies/base.py` | ALREADY EXTRACTED |
| Charter minimum enforcement | `brokers/oanda_connector.py` | KEEP AS-IS |
| `rbz_tight_trailing.py` | Root of Phoenix | EXTRACT TO CLEAN REPO (pending runtime proof it works) |
| `util/momentum_trailing.py` | Phoenix/util | EXTRACT OR REWRITE — depends on log proof |
| `SmartTrailingSystem` | `util/momentum_trailing.py` | EXTRACT OR REWRITE |
| Hive Mind | `hive/rick_hive_mind.py` | DISCARD — LLM-dependent |
| Crypto / Coinbase logic | Various | DISCARD |
| ML / neural / model files | Various | DISCARD |

---

## 10. Open Questions (require terminal proof)

1. Is the Phoenix engine process currently running?
2. Which PID / launch method?
3. Do Phoenix logs show any TRAIL attempts or broker SL updates?
4. Does `rbz_tight_trailing.py` import successfully in the Phoenix `.venv`?
5. Are there periodic SL modification API calls appearing in any log?
