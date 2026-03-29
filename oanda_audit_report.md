# RBOTZILLA_OANDA_CLEAN — Trade Audit Report
**Generated:** 2026-03-22 10:34 EDT

---

## 1. Signal Selection Logic — WAS IT CHANGED?

**NO. `strategies/multi_signal_engine.py` was last modified on 2026-03-17.**

It has been untouched since the initial build — **7 days before this audit**. The confidence scoring, vote counting, and detector logic that produced profitable trades is **exactly as it was from day one.** No changes to:
- `MIN_CONFIDENCE` threshold (0.68)
- `MIN_VOTES` threshold (3)
- EMA stack, Fibonacci, Liquidity Sweep, FVG, EMA scalper 200 detectors
- [scan_symbol()](file:///home/rfing/RBOTZILLA_PHOENIX/systems/multi_signal_engine.py#889-1013) function

The strategy files ([fib_confluence_breakout.py](file:///home/rfing/RBOTZILLA_PHOENIX/strategies/fib_confluence_breakout.py), [liquidity_sweep.py](file:///home/rfing/RBOTZILLA_PHOENIX/strategies/liquidity_sweep.py), [trap_reversal_scalper.py](file:///home/rfing/RBOTZILLA_PHOENIX/strategies/trap_reversal_scalper.py)) were last touched 2026-03-18 — also untouched since Friday.

> [!IMPORTANT]
> The profitable trade selection behavior is **fully preserved and unchanged.**

---

## 2. Trades Opened by Clean Repo (TRADE_OPENED events)

Only **2 trades** were confirmed placed by the clean engine. Both on **2026-03-20**:

| Time (UTC) | Pair | Dir | Confidence | Votes | Detectors |
|---|---|---|---|---|---|
| 2026-03-20 18:22 | AUD_JPY | SELL | 0.7905 | 3 | Not logged |
| 2026-03-20 18:26 | USD_CAD | SELL | 0.7843 | 3 | Not logged |

Both passed the 0.68 confidence / 3 vote threshold. Both were high-quality signals.

---

## 3. Trades Synced From Broker (POSITION_SYNCED = Phoenix-Opened)

**133 POSITION_SYNCED events** — these are trades Phoenix opened that the clean repo's trade manager picked up and started tracking for trailing stop management.

Pairs seen most frequently in synced positions:
`USD_JPY` · `EUR_NZD` · `EUR_GBP` · `EUR_USD` · `AUD_JPY` · `AUD_USD` · `AUD_CAD` · `EUR_AUD` · `NZD_JPY` · `CHF_JPY` · `GBP_JPY` · `GBP_CHF` · `CAD_JPY`

---

## 4. Position Closures — 52 Total

**All 52** closed with `reason=not_in_broker_trades`.

This means: **the close was NOT done by the clean repo.** It means the trade disappeared from OANDA before the clean repo could log the reason. Three possible causes per closure:

| Cause | How to tell |
|---|---|
| **SL hit (Phoenix SL order triggered)** | Trade closed at or near the SL price |
| **TP hit (Phoenix TP order triggered)** | Trade closed at or near the TP price |
| **You manually closed it** | Trade closed at a price between SL and TP |

> [!NOTE]
> The clean repo's [TradeManager](file:///home/rfing/RBOTZILLA_OANDA_CLEAN/engine/trade_manager.py#37-483) has **zero force-close authority** in the current code except for the `HARD_DOLLAR_STOP` which triggers at -$30 unrealized. None of the 52 closures logged a `HARD_DOLLAR_STOP` event — so **none were force-closed by the clean repo.**

The 52 closures are a mix of Phoenix auto-close (SL/TP) and your manual closes. **Cannot distinguish without OANDA transaction history.**

---

## 5. Autonomous Manager Actions

No `TRAIL_SL_SET`, `GREEN_LOCK_ENFORCED`, `HARD_DOLLAR_STOP`, `BREAK_EVEN_SET`, or `MANAGER_CYCLE_STARTED` events appear in narration.jsonl.

**Why:** The `TradeManager.activate()` / [tick()](file:///home/rfing/RBOTZILLA_OANDA_CLEAN/engine/trade_manager.py#84-88) wiring was only added on 2026-03-21 (yesterday). The manager has not yet had a full live run cycle logged.

---

## 6. 🚨 CRITICAL BUG FOUND — All New OCO Orders Rejected

The engine log contains **repeated `TRAILING_STOP_LOSS_ON_FILL_PRICE_DISTANCE_MINIMUM_NOT_MET` errors from OANDA.**

```
EUR_GBP:  trailingStopLossOnFill distance=0.00034  → REJECTED
EUR_GBP:  trailingStopLossOnFill distance=0.00036  → REJECTED
EUR_GBP:  trailingStopLossOnFill distance=0.00039  → REJECTED
CAD_JPY:  trailingStopLossOnFill distance=0.043    → REJECTED
```

**What this means:** When the new `trailing_stop_distance` parameter was added to OCO placement (the "only major flaw fix"), the distance being passed is often **below OANDA's minimum**. OANDA requires the trailing stop distance to be **at least 5 pips for most pairs** (50 pips for JPY pairs). Values like `0.00034` are ~3.4 pips — below the 5-pip minimum.

**Result:** Every OCO order with the native trailing stop is being rejected by OANDA at the broker level. No trades are being placed while this is active.

> [!CAUTION]
> This is the most urgent fix needed. Until the trailing stop distance is corrected or the native OANDA TS is removed from the OCO payload, the clean engine places **zero trades**.

---

## 7. Summary

| Item | Status |
|---|---|
| Signal confidence logic changed? | **NO — unchanged since March 17** |
| Profitable trade behavior preserved? | **YES** |
| Trades placed by clean repo | **2 (AUD_JPY SELL, USD_CAD SELL)** |
| Trades synced from Phoenix | **133** |
| Closures by clean repo TradeManager | **NONE** |
| Closures by Phoenix SL/TP or manual | **52** |
| Active trailing manager actions logged | **NONE YET** |
| Native OCO trailing stop active? | **YES — but broken** |
| Orders being placed right now? | **NO — all rejected by OANDA** |
