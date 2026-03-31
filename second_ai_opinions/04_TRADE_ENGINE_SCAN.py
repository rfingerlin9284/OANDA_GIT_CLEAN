# EXCERPT FROM: engine/trade_engine.py — Lines 320-725

        if cycle_limit == 0:
            return 0

        # ── ATTACH_ONLY hard block ─────────────────────────────────────────────
        # When ATTACH_ONLY=true or DISABLE_NEW_ENTRIES=true, no new trades are
        # submitted. Broker sync above still runs so active_positions stays accurate.
        # Phoenix is the verified source opener — this engine must not compete.
        if ATTACH_ONLY or DISABLE_NEW_ENTRIES:
            log_event(
                ATTACH_ONLY_BLOCK, symbol="SYSTEM", venue="engine",
                details={
                    "reason":               "ATTACH_ONLY mode active — new trade entries disabled",
                    "attach_only_flag":     ATTACH_ONLY,
                    "disable_entries_flag": DISABLE_NEW_ENTRIES,
                    "broker_open":          broker_open,
                    "slots_left":           slots_left,
                },
            )
            print(f"  [ATTACH_ONLY]  broker_open={broker_open}  no new entries this cycle")
            return 0

        # ── Live account for margin gate ──────────────────────────────────────
        try:
            acct         = self.connector.get_account_info()
            nav          = acct.balance + acct.unrealized_pl
            margin_used  = acct.margin_used
            free_margin  = nav - margin_used
            free_margin_pct = free_margin / nav if nav > 0 else 0.0
        except Exception:
            nav = free_margin_pct = 1.0   # fail-open on NAV query only

        # ── TRANSCRIPT EDGE: Daily Circuit Breaker ─────────────────────────
        import datetime as _dt_mod
        _today = _dt_mod.date.today()
        if self._daily_trade_date != _today:
            self._daily_trade_date = _today
            self._circuit_breaker_tripped = False
            try:
                _acct_day = self.connector.get_account_info()
                self._daily_open_balance = _acct_day.balance
            except Exception:
                self._daily_open_balance = None

        if self._circuit_breaker_tripped:
            return 0

        if self._daily_open_balance is not None:
            try:
                _acct_cb = self.connector.get_account_info()
                _today_pl = _acct_cb.balance - self._daily_open_balance
                if _today_pl < 0 and abs(_today_pl) >= DAILY_MAX_LOSS_USD:
                    print(f"  🛑 CIRCUIT BREAKER — Today\'s loss ${abs(_today_pl):.2f} >= ${DAILY_MAX_LOSS_USD:.0f} limit. No new entries until tomorrow.")
                    self._circuit_breaker_tripped = True
                    return 0
                if _today_pl > 0 and _today_pl >= DAILY_MAX_GAIN_USD:
                    print(f"  🏆 DAILY TARGET HIT — Today\'s gain ${_today_pl:.2f} >= ${DAILY_MAX_GAIN_USD:.0f} target. Preserving profits.")
                    self._circuit_breaker_tripped = True
                    return 0
            except Exception:
                pass

        # ── TRANSCRIPT EDGE: DXY Proxy (USD/CHF 20-bar trend) ─────────────
        # Source: DXY transcript — USD/CHF has ~0.95 DXY correlation
        # Cached once per scan cycle, used inside per-pair loop
        _dxy_bias = None
        if DXY_GATE_ENABLED:
            try:
                _dxy_candles = self.connector.get_historical_data("USD_CHF", count=25, granularity="M15")
                _dxy_closes = [float(c.get("mid", {}).get("c", 0)) for c in _dxy_candles]
                if len(_dxy_closes) >= 20:
                    _dxy_sma20 = sum(_dxy_closes[-20:]) / 20.0
                    _dxy_now = _dxy_closes[-1]
                    if _dxy_now > _dxy_sma20 * 1.0005:  # USD strengthening
                        _dxy_bias = "USD_STRONG"
                    elif _dxy_now < _dxy_sma20 * 0.9995:  # USD weakening
                        _dxy_bias = "USD_WEAK"
            except Exception:
                pass

        # Collect qualifying signals
        # Pre-market playbook ordering: vetted symbols scan first
        if self._session_playbook:
            pb_symbols = [
                e.symbol for e in self._session_playbook
                if not e.vetoed and e.symbol in set(TRADING_PAIRS)
            ]
            pb_set = set(pb_symbols)
            scan_order = pb_symbols + [s for s in TRADING_PAIRS if s not in pb_set]
        else:
            scan_order = list(TRADING_PAIRS)

        qualified = []
        for symbol in scan_order:
            try:
                candles = self.connector.get_historical_data(
                    symbol, count=CANDLE_COUNT, granularity=CANDLE_GRANULARITY
                )
                if not candles or len(candles) < 50:
                    continue

                # ── Phase 9: Multi-Strategy Pipeline Scanner ───────────────────
                # Each pipeline runs its own detector subset independently.
                # Best qualifying signal for this pair wins.

                candidates = []

                # ── TRANSCRIPT EDGE: 200 EMA Directional Filter ────────────────
                _ema200_bias = None
                if EMA200_GATE_ENABLED:
                    _closes_m15 = [float(c.get("mid", {}).get("c", 0)) for c in candles]
                    if len(_closes_m15) >= 200:
                        _ema200 = sum(_closes_m15[:200]) / 200.0
                        _k200 = 2.0 / 201.0
                        for _px200 in _closes_m15[200:]:
                            _ema200 = (_px200 - _ema200) * _k200 + _ema200
                        _ema200_bias = "BUY" if _closes_m15[-1] > _ema200 else "SELL"

                # ── TRANSCRIPT EDGE: RSI Overbought/Oversold Gate ──────────────
                _rsi_block_buy = False
                _rsi_block_sell = False
                if RSI_GATE_ENABLED:
                    _cls_rsi = [float(c.get("mid", {}).get("c", 0)) for c in candles[-15:]]
                    if len(_cls_rsi) >= 15:
                        _gains = [max(_cls_rsi[i] - _cls_rsi[i-1], 0) for i in range(1, len(_cls_rsi))]
                        _losses = [max(_cls_rsi[i-1] - _cls_rsi[i], 0) for i in range(1, len(_cls_rsi))]
                        _avg_gain = sum(_gains) / 14.0
                        _avg_loss = sum(_losses) / 14.0
                        if _avg_loss > 0:
                            _rs = _avg_gain / _avg_loss
                            _rsi_val = 100.0 - (100.0 / (1.0 + _rs))
                        else:
                            _rsi_val = 100.0
                        if _rsi_val >= RSI_OB_LEVEL:
                            _rsi_block_buy = True
                        if _rsi_val <= RSI_OS_LEVEL:
                            _rsi_block_sell = True

                # ── TRANSCRIPT EDGE: MACD Divergence Detection ─────────────
                _macd_div_block_buy = False
                _macd_div_block_sell = False
                if MACD_DIV_ENABLED:
                    _cls_macd = [float(c.get("mid", {}).get("c", 0)) for c in candles[-30:]]
                    if len(_cls_macd) >= 26:
                        # EMA-12 and EMA-26
                        _e12 = sum(_cls_macd[:12]) / 12.0
                        _e26 = sum(_cls_macd[:26]) / 26.0
                        _k12 = 2.0 / 13.0
                        _k26 = 2.0 / 27.0
                        _macd_hist = []
                        for _i_m in range(26, len(_cls_macd)):
                            _e12 = (_cls_macd[_i_m] - _e12) * _k12 + _e12
                            _e26 = (_cls_macd[_i_m] - _e26) * _k26 + _e26
                            _macd_hist.append(_e12 - _e26)
                        if len(_macd_hist) >= 3:
                            # Bearish div: price rising, MACD histogram falling
                            if (_cls_macd[-1] > _cls_macd[-3] and
                                    _macd_hist[-1] < _macd_hist[-3]):
                                _macd_div_block_buy = True
                            # Bullish div: price falling, MACD histogram rising
                            if (_cls_macd[-1] < _cls_macd[-3] and
                                    _macd_hist[-1] > _macd_hist[-3]):
                                _macd_div_block_sell = True

                # ─ Pipeline 1: Momentum (SMA + EMA + Fib, 2-of-3, H4-confirmed) ─
                mom = run_momentum_pipeline(symbol, candles, min_confidence=MIN_CONFIDENCE)
                if mom and getattr(mom, 'signal_type', '') == 'trend':
                    try:
                        candles_h4 = self.connector.get_historical_data(symbol, count=100, granularity="H4")
                        closes_h4 = [float(c.get("mid", {}).get("c", 0)) for c in candles_h4]
                        if len(closes_h4) >= 55:
                            k = 2.0 / (56.0)
                            ema_h4 = sum(closes_h4[:55]) / 55.0
                            for px in closes_h4[55:]:
                                ema_h4 = (px - ema_h4) * k + ema_h4
                            price_h4 = closes_h4[-1]
                            if mom.direction == "BUY" and price_h4 < ema_h4:
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM BUY blocked — H4 BEARISH")
                                log_gate_block(symbol, "MTF_SNIPER_BLOCK", {"h4_ema55": ema_h4, "price": price_h4})
                                mom = None
                            elif mom.direction == "SELL" and price_h4 > ema_h4:
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM SELL blocked — H4 BULLISH")
                                log_gate_block(symbol, "MTF_SNIPER_BLOCK", {"h4_ema55": ema_h4, "price": price_h4})
                                mom = None
                            elif mom:
                                mom._timeframe = "M15+H4"
                    except Exception:
                        pass  # Fail open if H4 fetch fails

                # ── Daily TF alignment gate ─────────────────────────────────────
                # Source insight: "2-of-3 higher TFs must agree — Weekly/Daily/4H"
                # We enforce H4 + Daily both aligned before momentum entry.
                # Adds ~20% fewer false signals vs H4-only gate per source analysis.
                if mom and getattr(mom, '_timeframe', '') == 'M15+H4':
                    try:
                        candles_d1 = self.connector.get_historical_data(symbol, count=50, granularity="D")
                        closes_d1 = [float(c.get("mid", {}).get("c", 0)) for c in candles_d1]
                        if len(closes_d1) >= 22:
                            ema_d1 = sum(closes_d1[:21]) / 21.0
                            k_d1 = 2.0 / 22.0
                            for px in closes_d1[21:]:
                                ema_d1 = (px - ema_d1) * k_d1 + ema_d1
                            price_d1 = closes_d1[-1]
                            if mom.direction == "BUY" and price_d1 < ema_d1:
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM BUY blocked — DAILY BEARISH")
                                log_gate_block(symbol, "MTF_SNIPER_D1_BLOCK", {"d1_ema21": ema_d1, "price": price_d1})
                                mom = None
                            elif mom.direction == "SELL" and price_d1 > ema_d1:
                                print(f"  [MTF_SNIPER] {symbol} MOMENTUM SELL blocked — DAILY BULLISH")
                                log_gate_block(symbol, "MTF_SNIPER_D1_BLOCK", {"d1_ema21": ema_d1, "price": price_d1})
                                mom = None
                            elif mom:
                                mom._timeframe = "M15+H4+D1"
                    except Exception:
                        pass  # Fail open if D1 fetch fails

                if mom:
                    candidates.append(mom)

                # ─ Pipeline 2: Reversal (Trap + LiqSweep + RSI, 1-of-3) ────────
                rev = run_reversal_pipeline(symbol, candles)
                if rev:
                    candidates.append(rev)

                # ─ Pipeline 3: Mean Reversion (BB + S&D + RSI, 1-of-3) ─────────
                mr = run_meanrev_pipeline(symbol, candles)
                if mr:
                    candidates.append(mr)

                # ─ Pipeline 4: FVG Scalp (FVG + OrderBlock, 1-of-2) ───────────
                sc = run_scalp_pipeline(symbol, candles)
                if sc:
                    candidates.append(sc)

                # ── TRANSCRIPT EDGE: 200 EMA directional filter ───────────
                if _ema200_bias and candidates:
                    _pre_ema = len(candidates)
                    candidates = [c for c in candidates if c.direction == _ema200_bias]
                    if len(candidates) < _pre_ema:
                        print(f"  [EMA200] {symbol} filtered {_pre_ema - len(candidates)} signal(s) against 200 EMA bias ({_ema200_bias})")

                # ── TRANSCRIPT EDGE: DXY Correlation Filter ───────────────
                if DXY_GATE_ENABLED and _dxy_bias and candidates and symbol != "USD_CHF":
                    _base, _quote = symbol.split("_") if "_" in symbol else ("", "")
                    _pre_dxy = len(candidates)
                    _dxy_filtered = []
                    for _cs in candidates:
                        _block = False
                        if _dxy_bias == "USD_STRONG":
                            # USD strong: block SELL on USD/*, block BUY on */USD
                            if _base == "USD" and _cs.direction == "SELL":
                                _block = True
                            if _quote == "USD" and _cs.direction == "BUY":
                                _block = True
                        elif _dxy_bias == "USD_WEAK":
                            # USD weak: block BUY on USD/*, block SELL on */USD
                            if _base == "USD" and _cs.direction == "BUY":
                                _block = True
                            if _quote == "USD" and _cs.direction == "SELL":
                                _block = True
                        if not _block:
                            _dxy_filtered.append(_cs)
                    candidates = _dxy_filtered
                    if len(candidates) < _pre_dxy:
                        print(f"  [DXY_GATE] {symbol} blocked {_pre_dxy - len(candidates)} signal(s) — {_dxy_bias}")

                # ── TRANSCRIPT EDGE: Volume Confirmation Gate ─────────────
                if VOLUME_GATE_ENABLED and candidates:
                    try:
                        _vols = [float(c.get("volume", 0)) for c in candles[-20:]]
                        if len(_vols) >= 20 and _vols[-1] > 0:
                            _avg_vol = sum(_vols[:-1]) / len(_vols[:-1])
                            if _avg_vol > 0 and _vols[-1] < _avg_vol * VOLUME_GATE_MULT:
                                _vol_ok = False
                                print(f"  [VOL_GATE] {symbol} blocked — vol {_vols[-1]:.0f} < {_avg_vol * VOLUME_GATE_MULT:.0f} (1.2x avg)")
                                candidates = []
                    except Exception:
                        pass

                # ── TRANSCRIPT EDGE: Volume Confirmation Gate ─────────────
                if VOLUME_GATE_ENABLED and candidates:
                    try:
                        _vols = [float(c.get("volume", 0)) for c in candles[-20:]]
                        if len(_vols) >= 20 and _vols[-1] > 0:
                            _avg_vol = sum(_vols[:-1]) / len(_vols[:-1])
                            if _avg_vol > 0 and _vols[-1] < _avg_vol * VOLUME_GATE_MULT:
                                print(f"  [VOL_GATE] {symbol} blocked — vol {_vols[-1]:.0f} < {_avg_vol * VOLUME_GATE_MULT:.0f} (1.2x avg)")
                                candidates = []
                    except Exception:
                        pass

                # ── TRANSCRIPT EDGE: MACD Divergence Filter ────────────────
                if MACD_DIV_ENABLED and candidates and (_macd_div_block_buy or _macd_div_block_sell):
                    _pre_macd = len(candidates)
                    candidates = [c for c in candidates
                                  if not (_macd_div_block_buy and c.direction == "BUY")
                                  and not (_macd_div_block_sell and c.direction == "SELL")]
                    if len(candidates) < _pre_macd:
                        print(f"  [MACD_DIV] {symbol} blocked {_pre_macd - len(candidates)} signal(s) — divergence detected")

                # ── TRANSCRIPT EDGE: RSI overbought/oversold gate ──────────
                if candidates and (_rsi_block_buy or _rsi_block_sell):
                    _pre_rsi = len(candidates)
                    candidates = [c for c in candidates
                                  if not (_rsi_block_buy and c.direction == "BUY")
                                  and not (_rsi_block_sell and c.direction == "SELL")]
                    if len(candidates) < _pre_rsi:
                        print(f"  [RSI_GATE] {symbol} blocked {_pre_rsi - len(candidates)} signal(s) — RSI overbought/oversold")

                # ── TRANSCRIPT EDGE: Session Confidence Boost ──────────────
                # Source: Richie Nasser — London/NY sessions have better fills
                if SESSION_BOOST_ENABLED and candidates:
                    try:
                        from util.time_utils import broker_now_eastern
                        _et_hour = broker_now_eastern().hour
                        _in_london_ny = 8 <= _et_hour <= 16  # 8am-4pm ET overlap
                        if _in_london_ny:
                            for _cs in candidates:
                                if not getattr(_cs, '_session_boosted', False):
                                    _cs.confidence = min(_cs.confidence + 0.03, 0.95)
                                    _cs._session_boosted = True
                    except Exception:
                        pass

                # ── Fib + S&D Confluence Boost ───────────────────────────────
                # Source: "COMPLETE S&D Course" — Fibonacci 61.8% + demand zone
                # overlapping = "maximum confidence, risk more."
                # When momentum pipeline (contains Fib) and mean-reversion
                # pipeline (contains S&D scanner) agree on same direction
                # → +5% confidence on aligned candidates (capped at 0.95).
                if mom and mr and mom.direction == mr.direction:
                    for _c in candidates:
                        if _c.direction == mom.direction and not getattr(_c, '_conf_boosted', False):
                            _c.confidence = min(_c.confidence + 0.05, 0.95)
                            _c._conf_boosted = True
                            print(f"  [CONFLUENCE] {symbol} {_c.direction} +5% conf — Fib+S&D aligned ({_c.confidence:.0%})")

                # ── "Look Left" Trend Exhaustion Filter ─────────────────────
                # Source: "15 Best Price Action Strategies" (15 years PA trading)
                # "Fresh trends = high quality. Late exhausted trends = low quality."
                # Block signals where price already traveled >65% of TP distance
                # from the 30-candle swing. Prevents entering at the end of a move.
                if candidates:
                    _pip_sz  = 0.01 if "JPY" in symbol.upper() else 0.0001
                    _tp_pips = float(os.getenv("RBOT_TP_PIPS", "150"))
                    _live_px = float(candles[-1].get("mid", {}).get("c", 0))
                    _h30     = max(float(c.get("mid", {}).get("h", 0)) for c in candles[-30:])
                    _l30     = min(float(c.get("mid", {}).get("l", 0)) for c in candles[-30:])
                    _thresh  = _tp_pips * 0.65 * _pip_sz
                    _fresh   = []
                    for _sig in candidates:
                        _travel = (_live_px - _l30) if _sig.direction == "BUY" else (_h30 - _live_px)
                        if _travel > _thresh:
                            _pips_t = round(_travel / _pip_sz, 0)
                            print(f"  [EXHAUST] {symbol} {_sig.direction} blocked — {_pips_t:.0f}p traveled (>{_tp_pips*0.65:.0f}p limit)")
                            log_gate_block(symbol, "EXHAUSTION_BLOCK", {
                                "travel_pips": _pips_t,
                                "threshold_pips": _tp_pips * 0.65,
                                "direction": _sig.direction,
                            })
                        else:
                            _fresh.append(_sig)
                    candidates = _fresh

                # ─ Pick best signal for this pair ─────────────────────────────
                if candidates:
                    best = max(candidates, key=lambda s: s.confidence)
                    if not hasattr(best, '_timeframe'):
                        best._timeframe = "M15"
                    if not hasattr(best, '_strategy'):
                        best._strategy = getattr(best, 'signal_type', 'trend')
                    qualified.append(best)

                # ─ Per-pair scan diagnostic line ───────────────────────────────
                def _pfmt(sig, label):
                    if sig is None:
                        return f"{label}=✗"
                    return f"{label}={sig.confidence:.0%}✓"

                tag = ""
                plain = ""
                if candidates:
                    best_strat = getattr(candidates[0] if len(candidates)==1 else max(candidates, key=lambda s: s.confidence), '_strategy', '?')
                    best_sig   = max(candidates, key=lambda s: s.confidence)
                    dir_word   = "SELL" if best_sig.direction == "SELL" else "BUY"
                    strat_desc = {
                        "momentum":      "Strong trend detected",
                        "reversal":      "Reversal pattern spotted",
                        "mean_reversion":"Price stretched, snap-back expected",
                        "scalp":         "Quick entry opportunity found",
                    }.get(best_strat.lower(), "Signal detected")
                    tag   = f"  → QUEUED [{best_strat.upper()}]"
                    plain = f"  💬 {symbol}: {strat_desc} ({best_sig.confidence:.0%} confident) — queuing a {dir_word}"
                print(
                    f"  [SCAN] {symbol:<10}"
                    f"  MOM={_pfmt(mom, 'MOM')[4:]}"
                    f"  REV={_pfmt(rev, 'REV')[4:]}"
                    f"  MR={_pfmt(mr, 'MR')[3:]}"
                    f"  SC={_pfmt(sc, 'SC')[3:]}"
                    f"{tag}"
                )
                if plain:
                    print(plain)

            except Exception:
                continue  # Never let a single pair crash the scan loop
