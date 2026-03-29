# STOP_MANAGEMENT_PROFILES.md
# RBOTZILLA_OANDA_CLEAN
Generated: 2026-03-17 | Label: NEW_CLEAN_REWRITE
These profiles govern SL/TP logic for each strategy family.
All stop updates must be confirmed by broker (TRAIL_SL_REJECTED if not).

---

## Profile 1 — Trend Following

**Applies to:** EMA Stack Pack (TREND_CONT), EMA Scalper 200

| Property | Rule |
|---|---|
| Initial stop | Below/above swing low/high used as entry reference, + 5 pip buffer |
| Max initial stop | 20 pips from entry |
| Breakeven trigger | When profit = initial risk (1R) |
| Partial take profit | 50% position at 1.5R |
| Trail trigger | 2R profit |
| Trail method | Move SL to 10 pips behind current price in trade direction |
| Hard stop fail-safe | Initial SL never moves against direction — only in favourable direction |
| Spread fail-safe | Do not trail if live spread > RBOT_MAX_SPREAD_PIPS at trail time |
| Do NOT use when | M5 timeframe or scalping mode — too loose for short targets |

**AUDIT NOTE:** Trailing stop update must go through `check_sl_update_response()`.
Local `pos['stop_loss']` only updated if broker confirms success.

---

## Profile 2 — Reversal

**Applies to:** Liquidity Sweep (LIQ_SWEEP_REV), Fibonacci Reversal, RSI Extreme + Trap

| Property | Rule |
|---|---|
| Initial stop | Above/below the sweep wick or fib invalidation level (78.6%) + 5 pips |
| Max initial stop | 25 pips from entry |
| Breakeven trigger | At 1R |
| Partial take profit | 50% at 1.5R (reversal often runs to target quickly) |
| Trail trigger | 2R |
| Trail method | Move SL to entry price (breakeven) then trail by 15 pips |
| Hard stop fail-safe | If price returns to entry bar open, close trade |
| Spread fail-safe | Block SL modification if spread > max spread at modification time |
| Do NOT use when | Strong institutional trend with no sign of exhaustion |

---

## Profile 3 — Liquidity Sweep

**Applies to:** `liq_sweep` standalone, Combo LIQ_SWEEP_REV

| Property | Rule |
|---|---|
| Initial stop | Beyond sweep wick + 8 pip buffer (wick must be respected) |
| Max initial stop | 30 pips (sweeps can have wide wicks) |
| Breakeven trigger | At 0.8R (sooner than other profiles — sweep moves are sharp) |
| Partial take profit | 40% at 1.5R, 40% at 2R, hold 20% to 3R |
| Trail trigger | 1.5R |
| Trail method | 8 pip trail (tight — sweep reversals are fast) |
| Hard stop fail-safe | Close if price re-tests sweep level — thesis invalidated |
| Spread fail-safe | Do not execute if spread > 3.5 pips at entry time |
| Do NOT use when | No clear equal highs/lows structure — no sweep without structure |

---

## Profile 4 — Breakout (Fib Confluence Breakout)

**Applies to:** FIB_BREAK combo, `fib_confluence_breakout.py`

| Property | Rule |
|---|---|
| Initial stop | Below 78.6% fib level (entry invalidation) + 5 pips |
| Max initial stop | 20 pips |
| Breakeven trigger | At 1R |
| Partial take profit | 50% at 2R (breakouts often run) |
| Trail trigger | 2.5R |
| Trail method | Trail 20 pips behind running price |
| Hard stop fail-safe | Close if price returns below fib entry level on close |
| Spread fail-safe | Block new entries if spread > 4.0 pips |
| Do NOT use when | Range-bound market — no clear swing structure for fib reference |

---

## Profile 5 — Scalp Reversal

**Applies to:** Trap Reversal Scalper (SCALP_REV), short-timeframe trap entries

| Property | Rule |
|---|---|
| Initial stop | 10–15 pip max (scalp profile — wide stop kills the edge) |
| Max initial stop | 15 pips absolute |
| Breakeven trigger | At 0.5R (move fast — scalps lose edge quickly) |
| Partial take profit | 100% at 1.5R target (scalp — exit clean) |
| Trail trigger | Do not trail — scalp to target |
| Trail method | N/A — close at target |
| Hard stop fail-safe | Initial stop is final stop — no manual adjustments in scalp mode |
| Spread fail-safe | Skip if spread > 2.5 pips at trigger time |
| Do NOT use when | Slow Asian session, no session open catalyst present |

---

## Strategy to Profile Mapping

| Strategy | Combo Name | Stop Profile |
|---|---|---|
| EMA Stack + 200 EMA | TREND_CONT | Profile 1 — Trend Following |
| Fibonacci Confluence | FIB_BREAK | Profile 4 — Breakout |
| Liquidity Sweep | LIQ_SWEEP_REV | Profile 3 — Liquidity Sweep |
| Trap Reversal (standard) | LIQ_SWEEP_REV | Profile 2 — Reversal |
| Trap Reversal Scalper | SCALP_REV | Profile 5 — Scalp Reversal |
| RSI Extreme | Confirmation only | Profile 2 — Reversal |
| FVG Fill | FIB_BREAK | Profile 4 — Breakout |
| High-Confidence 4-vote | INST_GRADE | Profile 1 — Trend Following |

---

## Implementation notes

**All trail SL updates must:**
1. Call `set_trade_stop()` on broker
2. Pass through `check_sl_update_response()` in `engine/broker_tradability_gate.py`
3. Only update `pos['stop_loss']` locally if `confirmed == True`
4. Emit `TRAIL_SL_REJECTED` via `narration_logger` if not confirmed

**Hard stops must:**
1. Be set at OCO placement time (not after)
2. Never be removed once placed
3. Be confirmed by broker response with a stop order ID

**Partial take profits:**
Not yet implemented in `trade_manager.py`.
Mark as Phase 7C or later work item.
