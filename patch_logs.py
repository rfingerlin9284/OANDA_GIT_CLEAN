import sys

path1 = "engine/regime_detector.py"
try:
    with open(path1, "r") as f:
        text1 = f.read()

    target1 = """    return {
        'regime': result.regime.value,
        'vol': result.volatility,
        'trend': result.trend_strength
    }"""
    replace1 = """    return {
        'regime': result.regime.value,
        'confidence': result.confidence,
        'vol': result.volatility,
        'trend': result.trend_strength
    }"""

    if target1 in text1:
        with open(path1, "w") as f:
            f.write(text1.replace(target1, replace1))
        print("✓ Patched " + path1)
    elif replace1 in text1:
        print("✓ Already patched: " + path1)
    else:
        print("✗ Target block not found in " + path1)
except Exception as e:
    print(f"✗ Error reading {path1}: {e}")

path2 = "engine/trade_engine.py"
try:
    with open(path2, "r") as f:
        text2 = f.read()

    target2 = """                    if len(closes) >= 50:
                        regime_data = detect_market_regime(closes, symbol)
                        current_regime = regime_data.get('regime', 'TRIAGE')
                        
                        # If chop/sideways is mathematically detected, pivot to Mean-Reversion S&D logic
                        if current_regime.upper() in ('CRASH', 'TRIAGE'):
                            print(f"  [REGIME]   {symbol} search blocked — MARKET IS {current_regime.upper()}")
                            log_gate_block(symbol, "REGIME_BLOCK", {"regime": current_regime.upper()})
                            continue
                        elif current_regime.upper() == 'SIDEWAYS':
                            sig = scan_sideways_symbol(symbol, candles, min_confidence=MIN_CONFIDENCE)
                            if sig:
                                sig.session = "S&D Scalp [SIDEWAYS]"
                                qualified.append(sig)
                            else:
                                print(f"  [REGIME]   {symbol} momentum blocked — MARKET IS {current_regime.upper()} (No S&D Zone Found)")
                            continue"""

    replace2 = """                    if len(closes) >= 50:
                        regime_data = detect_market_regime(closes, symbol)
                        current_regime = regime_data.get('regime', 'TRIAGE')
                        conf = regime_data.get('confidence', 0.0)
                        
                        # If chop/sideways is mathematically detected, pivot to Mean-Reversion S&D logic
                        if current_regime.upper() in ('CRASH', 'TRIAGE'):
                            print(f"  [REGIME]   {symbol} search blocked — MARKET IS {current_regime.upper()} (conf={conf:.1%})")
                            log_gate_block(symbol, "REGIME_BLOCK", {"regime": current_regime.upper(), "confidence": round(conf, 4)})
                            continue
                        elif current_regime.upper() == 'SIDEWAYS':
                            sig = scan_sideways_symbol(symbol, candles, min_confidence=MIN_CONFIDENCE)
                            if sig:
                                sig.session = "S&D Scalp [SIDEWAYS]"
                                qualified.append(sig)
                            else:
                                print(f"  [REGIME]   {symbol} momentum blocked — MARKET IS {current_regime.upper()} (conf={conf:.1%}) (No S&D Zone Found)")
                            continue"""

    if target2 in text2:
        with open(path2, "w") as f:
            f.write(text2.replace(target2, replace2))
        print("✓ Patched " + path2)
    elif replace2 in text2:
        print("✓ Already patched: " + path2)
    else:
        print("✗ Target block not found in " + path2)
except Exception as e:
    print(f"✗ Error reading {path2}: {e}")
