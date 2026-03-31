# EXCERPT FROM: engine/trade_engine.py — Lines 726-1050


        # ─ Qualified summary ──────────────────────────────────────────────────
        if qualified:
            summary = "  , ".join(
                f"{s.symbol} {s.direction} {getattr(s,'_strategy','?')} {s.confidence:.0%}"
                for s in qualified
            )
            print(f"  📋 Qualified: {len(qualified)} signal(s) — {summary}")
            print(f"  💬 Scan complete: found {len(qualified)} tradeable opportunit{'y' if len(qualified)==1 else 'ies'} this minute. Best signals sorted by confidence — placing now.")

        qualified.sort(key=lambda s: s.confidence, reverse=True)

        log_event(SIGNAL_SCAN_COMPLETE, symbol="SYSTEM", venue="signal_scan", details={
            "pairs_scanned":    len(TRADING_PAIRS),
            "candidates_found": len(qualified),
            "cycle_limit":      cycle_limit,
            "open_slots":       slots_left,
        })

        # ── Placement loop ────────────────────────────────────────────────────
        placed_this_cycle: set = set()
        placed_count = 0

        # ── Quiet hours gate: no new opens 10pm–6am EDT (after NY close, before London) ──
        _quiet_enabled = os.getenv("RBOT_QUIET_HOURS_ENABLED", "false").lower() == "true"
        if _quiet_enabled:
            _qh_start = int(os.getenv("RBOT_QUIET_HOURS_START", "22"))
            _qh_end   = int(os.getenv("RBOT_QUIET_HOURS_END",   "6"))
            _now_et   = broker_now_eastern()
            _hour_et  = _now_et.hour
            _in_quiet = (_hour_et >= _qh_start) or (_hour_et < _qh_end)
            if _in_quiet:
                print(f"  🌙 QUIET HOURS ({_now_et.strftime('%I:%M%p ET')}) — no new opens until {_qh_end:02d}:00 ET. Existing trades running.")
                return 0

        # Pre-filter: don't waste cycle_limit slots on pairs already held at broker
        # (dedup guard inside the loop remains as safety net)
        eligible = [s for s in qualified if s.symbol not in broker_symbols and not self._symbol_is_active(s.symbol)]
        for sig in eligible[:cycle_limit]:
            symbol = sig.symbol

            # ── Dedup: broker symbols + local + this cycle ─────────────────
            if (symbol.upper() in placed_this_cycle
                    or symbol in broker_symbols
                    or self._symbol_is_active(symbol)):

                log_gate_block(symbol, SYMBOL_ALREADY_ACTIVE_BLOCK, {"symbol": symbol})
                continue

            # ── Margin gate (live NAV) ──────────────────────────────────────
            if free_margin_pct < MIN_FREE_MARGIN_PCT:
                log_gate_block(symbol, MARGIN_GATE_BLOCKED, {
                    "free_margin_pct": round(free_margin_pct, 3),
                    "min_required":    MIN_FREE_MARGIN_PCT,
                })
                print(f"  BLOCKED    {symbol} — MARGIN_GATE_BLOCKED free={free_margin_pct:.1%}")
                continue   # margin tight for this symbol — try next

            # ── Correlation gate: no same-currency same-direction double-up ──
            if self._would_create_correlated_exposure(symbol, sig.direction):
                print(
                    f"  BLOCKED    {symbol} — CORRELATION_GATE "
                    f"({sig.direction} on same currency already open)"
                )
                print(f"  💬 {symbol}: Skipped — already in a very similar trade. Avoiding duplicate risk.")
                continue

            # ── Phase 1: CANDIDATE_FOUND ───────────────────────────────────
            log_event(CANDIDATE_FOUND, symbol=symbol, venue="signal_scan", details={
                "symbol":     symbol,
                "direction":  sig.direction,
                "confidence": round(sig.confidence, 4),
                "votes":      sig.votes,
                "detectors":  sig.detectors_fired,
                "session":    sig.session,
            })
            print(
                f"  CANDIDATE  {symbol} {sig.direction} "
                f"conf={sig.confidence:.1%} ({sig.votes}v)"
            )

            # ── Phase 2: broker tradability gate ──────────────────────────
            gate = check_broker_tradability(
                self.connector, symbol,
                placed_this_cycle=placed_this_cycle,
            )
            if not gate["allowed"]:
                log_gate_block(symbol, gate["event"], gate["detail"])
                print(f"  BLOCKED    {symbol} — {gate['event']} {gate['detail']}")
                continue

            # Live mid-price from gate — used as entry_price for notional calc
            # (MARKET orders have no submitted price; Charter needs a real number)
            live_mid = (gate.get("live_price") or {}).get("mid") or 0.0

            # ── Anti-churn gate: no immediate same-pair re-entry / flip ──────
            churn_reason = self._pair_cooldown_reason(symbol, sig.direction)
            if churn_reason:
                print(f"  BLOCKED    {symbol} — {churn_reason}")
                continue

            sig = self._enforce_rr_buffer(symbol, sig, live_mid)

            # ── Phase 3: ORDER_SUBMIT_ALLOWED ─────────────────────────────
            log_event(ORDER_SUBMIT_ALLOWED, symbol=symbol, venue="tradability_gate", details={
                "symbol":     symbol,
                "direction":  sig.direction,
                "confidence": round(sig.confidence, 4),
            })
            print(f"  → Placing  {symbol} {sig.direction} conf={sig.confidence:.1%}")
            print(f"  💬 {symbol}: Checks passed — submitting {'SELL (going short)' if sig.direction=='SELL' else 'BUY (going long)'} order to broker now.")

            # ── Phoenix-mode: fixed pip SL/TP override ──────────────────────────
            # Replaces signal's variable SL/TP (10–100+ pips) with exact pip values,
            # guaranteeing 3.2:1 R:R and predictable per-trade dollar risk.
            if self._sl_pips > 0 and live_mid:
                _pip = 0.01 if "JPY" in symbol.upper() else 0.0001
                _sl_dist = self._sl_pips * _pip
                _tp_dist = self._tp_pips * _pip
                if sig.direction == "BUY":
                    sig.sl = round(live_mid - _sl_dist, 5)
                    sig.tp = round(live_mid + _tp_dist, 5)
                else:
                    sig.sl = round(live_mid + _sl_dist, 5)
                    sig.tp = round(live_mid - _tp_dist, 5)

            # ── Phase 4: OCO payload validation ───────────────────────────
            units = self._compute_units(symbol, sig, nav)
            units = self._apply_min_notional_floor(symbol, units, live_mid)
            oco_check = validate_oco_payload(
                symbol=symbol,
                direction=sig.direction,
                entry_price=live_mid,
                stop_loss=sig.sl,
                take_profit=sig.tp,
                units=units,
            )
            if not oco_check["valid"]:
                log_gate_block(symbol, OCO_VALIDATION_BLOCK, oco_check)
                print(f"  BLOCKED    {symbol} — OCO_VALIDATION_BLOCK {oco_check['reason']}")
                set_cooldown(symbol)  # avoid retrying a structurally failing pair every cycle
                continue

            # ── Phase 5: place OCO order ───────────────────────────────────
            try:
                # OANDA hard minimum is 5 pips (0.0005 non-JPY, 0.05 JPY).
                # Setting floor to 10 pips and enforcing 2x buffer (20-pip worst-case)
                # prevents TRAILING_STOP_LOSS_ON_FILL_PRICE_DISTANCE_MINIMUM_NOT_MET
                # due to floating-point imprecision at or near the broker floor.
                pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
                min_ts_dist = 10.0 * pip_size          # 0.001 non-JPY, 0.10 JPY
                raw_sl_dist = abs(live_mid - sig.sl)
                # ── Adaptive trailing stop distance ─────────────────────────────
                # RBOT_TS_PIPS=N  → fixed N pips (override, current: 50)
                # RBOT_TS_PIPS=0  → ATR mode: RBOT_TS_ATR_MULT × ATR(14)
                # Source insight: ATR trail adapts to pair volatility vs fixed pips.
                # JPY pairs naturally get wider trail; rangy pairs get tighter.
                _ts_pips     = float(str(os.getenv("RBOT_TS_PIPS",     "0")).split("#", 1)[0].strip())
                _ts_atr_mult = float(str(os.getenv("RBOT_TS_ATR_MULT", "0")).split("#", 1)[0].strip())
                if _ts_pips > 0:
                    ts_dist = max(_ts_pips * pip_size, min_ts_dist * 2)
                elif _ts_atr_mult > 0:
                    # ATR(14) adaptive trail — scales to current pair volatility
                    try:
                        _atr_c = self.connector.get_historical_data(symbol, count=20, granularity="M15")
                        _ch = [float(c.get("mid", {}).get("h", 0)) for c in _atr_c]
                        _cl = [float(c.get("mid", {}).get("l", 0)) for c in _atr_c]
                        _cc = [float(c.get("mid", {}).get("c", 0)) for c in _atr_c]
                        if len(_ch) >= 15:
                            _trs = [max(_ch[i]-_cl[i], abs(_ch[i]-_cc[i-1]), abs(_cl[i]-_cc[i-1]))
                                    for i in range(1, len(_ch))]
                            _atr = sum(_trs[-14:]) / 14.0
                            ts_dist = max(_atr * _ts_atr_mult, min_ts_dist * 2)
                        else:
                            ts_dist = max(raw_sl_dist, min_ts_dist * 2)
                    except Exception:
                        ts_dist = max(raw_sl_dist, min_ts_dist * 2)
                else:
                    ts_dist = max(raw_sl_dist, min_ts_dist * 2)

                result = self.connector.place_oco_order(
                    instrument=symbol,
                    entry_price=live_mid,
                    stop_loss=sig.sl,
                    take_profit=sig.tp,
                    units=units,
                    order_type="MARKET",
                    trailing_stop_distance=ts_dist,
                )

                # ── Phase 6: verify broker response ───────────────────────
                confirm = check_submit_response(result, symbol)
                
                # Fallback: If broker said success=True but parsing failed to find trade_id
                if not confirm.get("confirmed") and result.get("success") and (result.get("live_api") or result.get("visible_in_oanda", True)):
                    import time
                    for _ in range(4):
                        time.sleep(1.0)  # Wait for LIMIT order cross
                        try:
                            trades = self.connector.get_trades()
                            matching = [t for t in trades if t.get("instrument") == symbol]
                            if matching:
                                newest = max(matching, key=lambda x: int(x.get("id", 0)))
                                confirm["confirmed"] = True
                                confirm["trade_id"] = str(newest.get("id"))
                                break
                        except Exception:
                            pass

                if not confirm.get("confirmed"):
                    log_event(ORDER_SUBMIT_BLOCK, symbol=symbol, venue="oanda", details={
                        "symbol": symbol, "error": confirm.get("error", "unknown")
                    })
                    print(f"  ✗ REJECTED {symbol} — {confirm.get('error', 'unknown')}")
                    continue

                trade_id = confirm["trade_id"]
                set_cooldown(symbol)
                placed_this_cycle.add(symbol.upper())
                _strategy  = getattr(sig, '_strategy',  getattr(sig, 'signal_type', 'trend'))
                _timeframe = getattr(sig, '_timeframe', 'M15')
                _detectors = sig.detectors_fired if hasattr(sig, 'detectors_fired') else []
                _det_str   = '+'.join(_detectors[:3]) if _detectors else _strategy
                self.active_positions[trade_id] = {
                    "symbol":      symbol,
                    "direction":   sig.direction,
                    "stop_loss":   sig.sl,
                    "take_profit": sig.tp,
                    "confidence":  sig.confidence,
                    "session":     sig.session,
                    "opened_at":   datetime.now(timezone.utc).isoformat(),
                    "stale_cycles": 0,
                    "strategy":    _strategy,
                    "timeframe":   _timeframe,
                    "detectors":   _detectors,
                }
                log_trade_opened(
                    symbol=symbol, direction=sig.direction, trade_id=trade_id,
                    entry=result.get("entry_price", live_mid),
                    stop_loss=sig.sl, take_profit=sig.tp,
                    size=units,
                    confidence=sig.confidence, votes=sig.votes,
                    detectors=sig.detectors_fired, session=sig.session,
                )
                placed_count += 1
                self._mark_pair_trade(symbol, sig.direction)
                print(f"  ✓ OPENED   {symbol} {sig.direction} [{_strategy.upper()} {_timeframe}] {_det_str}  id={trade_id}")
                _rr = getattr(sig, 'rr', 0)
                _sl_pips = round(abs(sig.sl - live_mid) / (0.01 if 'JPY' in symbol else 0.0001))
                _tp_pips = round(abs(sig.tp - live_mid) / (0.01 if 'JPY' in symbol else 0.0001))
                print(f"  💬 ✅ Trade open! {symbol} {'selling' if sig.direction=='SELL' else 'buying'} {abs(units):,} units. "
                      f"Stop-loss in {_sl_pips} pips, take-profit in {_tp_pips} pips. "
                      f"Trailing stop active. OCO protection set.")

                # ── Hedge counter-trade (Phoenix QuantHedgeEngine) ──────────────────
                if self._hedge_engine and getattr(self, "is_chop_mode_active", False) and live_mid:
                    try:
                        _hedge = self._hedge_engine.execute_hedge(
                            primary_symbol=symbol,
                            primary_side=sig.direction,
                            position_size=units,
                            entry_price=live_mid,
                        )
                        if _hedge:
                            _h_prices = self.connector.get_live_prices([_hedge.symbol])
                            _h_mid = (_h_prices.get(_hedge.symbol) or {}).get("mid", 0.0)
                            if _h_mid:
                                _h_pip = 0.01 if "JPY" in _hedge.symbol.upper() else 0.0001
                                _h_sl_d = (self._sl_pips or 10) * _h_pip
                                _h_tp_d = (self._tp_pips or 32) * _h_pip
                                if _hedge.side == "BUY":
                                    _h_sl = round(_h_mid - _h_sl_d, 5)
                                    _h_tp = round(_h_mid + _h_tp_d, 5)
                                else:
                                    _h_sl = round(_h_mid + _h_sl_d, 5)
                                    _h_tp = round(_h_mid - _h_tp_d, 5)
                                _h_result = self.connector.place_oco_order(
                                    instrument=_hedge.symbol,
                                    entry_price=_h_mid,
                                    stop_loss=_h_sl,
                                    take_profit=_h_tp,
                                    units=int(_hedge.size),
                                    order_type="MARKET",
                                    trailing_stop_distance=_h_sl_d * 2,
                                    is_hedge=True,
                                )
                                if _h_result.get("success"):
                                    print(f"  🞡  HEDGE    {_hedge.symbol} {_hedge.side} {int(_hedge.size)}u @ {_h_mid} (counter to {symbol})")
                                else:
                                    print(f"  ⚠️  HEDGE    {_hedge.symbol} rejected: {_h_result.get('error', 'unknown')}")
                    except Exception as _he:
                        print(f"  ⚠️  HEDGE    error for {symbol}: {_he}")

            except Exception as e:
                log_event(TRADE_OPEN_FAILED, symbol=symbol, venue="oanda", details={
                    "symbol": symbol, "error": str(e)
                })
                print(f"  ✗ ERROR    {symbol} — {e}")

        # ── Gap 1 fix: CapitalRouter reallocation ──────────────────────────────
        # Runs ONLY when all position slots are full AND a candidate is genuinely
        # stronger than the weakest open position (controlled by UPGRADE_THRESHOLD).
        # One reallocation max per cycle. Passes through full OCO validation gate.
        if (
            self._router
            and qualified
            and not ATTACH_ONLY
            and not DISABLE_NEW_ENTRIES
        ):
            self._router.reset_cycle()
            try:
                acct_info = {"NAV": nav, "balance": nav}
                realloc = self._router.evaluate(
                    open_positions=self.active_positions,
                    candidates=qualified,
                    account_info=acct_info,
                )
            except Exception as _re:
                realloc = None
                print(f"  [ROUTER] evaluate() error: {_re}")

            if realloc:
                _rs = realloc.close_symbol
                _rt = realloc.close_trade_id
