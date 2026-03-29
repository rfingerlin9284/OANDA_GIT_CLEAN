# RUNTIME_OWNERSHIP_REPORT.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-19T14:15 | Based on user-pasted terminal evidence + workspace inspection.

---

## 1. Source Opener Process

| Property | Value | Label |
|---|---|---|
| Repo | `/home/rfing/RBOTZILLA_PHOENIX` | VERIFIED |
| Process | `.venv/bin/python oanda_trading_engine.py` | VERIFIED |
| Launcher | Direct process — no tmux | VERIFIED |
| Signal engine | `strategies/multi_signal_engine.py` → `scan_symbol()` | INFERRED |
| Order placement | `brokers/oanda_connector.py` → `place_oco_order()` | INFERRED |
| Trade manager | Within `oanda_trading_engine.py` | INFERRED |
| Status | **ACTIVE — DO NOT MODIFY** | VERIFIED |

---

## 2. Source Trailing-Stop Process

| Property | Value | Label |
|---|---|---|
| Trail logic file | `rbz_tight_trailing.py` (root of Phoenix) | INFERRED |
| Trail runtime events | Multiple `TRAIL_TIGHT USD_JPY trail SL updates` in Phoenix log | VERIFIED |
| Trail logic is active | YES | VERIFIED |
| TS column in broker UI | Blank — OANDA does not show programmatic SL updates as "TS" | VERIFIED |
| Trail method | Programmatic SL modification via broker API (not OANDA native trailing) | INFERRED |

> ⚠️ The trailing logic IS running. The blank TS column in the OANDA UI is expected — Phoenix
> updates SL via trade modify calls, which do not activate OANDA's built-in trailing stop column.
> What appeared to be "no trailing" was actually programmatic trailing that OANDA doesn't surface as "TS".

---

## 3. Auth Failure Evidence

| Property | Value | Label |
|---|---|---|
| Error pattern | `401 Unauthorized` on candle requests | VERIFIED |
| Affected repo | RBOTZILLA_PHOENIX | VERIFIED |
| Cause | INFERRED: API token expired, or rate-limit on candle fetch endpoint | INFERRED |
| Impact | Candle data unavailable for affected cycles → no signals for those pairs | INFERRED |
| Trade placement impact | OCO placement uses a different endpoint — may still succeed | INFERRED |

---

## 4. Dual-Process Risk Assessment

| Risk | Detail | Severity |
|---|---|---|
| Both repos submitting new trades | Phoenix opens + clean repo tries to open on same account | 🔴 HIGH |
| Double position sizing | Same pair could get two entries | 🔴 HIGH |
| Conflicting SL modifications | Clean repo SL update + Phoenix trail update race | 🔴 HIGH |
| Margin exhaustion | Two engines scanning → more orders → margin gate fails | 🟡 MEDIUM |
| Clean repo `active_positions` desynced from broker | Already mitigated by broker sync loop | 🟢 LOW |

---

## 5. Immediate Action Taken

- `ATTACH_ONLY=true` added to clean repo `.env` — **NEW ENTRIES DISABLED**
- `DISABLE_NEW_ENTRIES=true` added to clean repo `.env`
- Hard placement block added to `engine/trade_engine.py` — logs `ATTACH_ONLY_BLOCK` and returns 0
- Phoenix NOT touched

---

## 6. Next Objective

Extract Phoenix `rbz_tight_trailing.py` behavior into clean repo `engine/trail_manager.py` as a read-only port.
Before doing that, need user to paste:

```
cat /home/rfing/RBOTZILLA_PHOENIX/rbz_tight_trailing.py | head -n 100
```
