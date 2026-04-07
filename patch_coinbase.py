import os
import re

ENGINE_DIR = "/home/rfing/RBOTZILLA_COINBASE_CLEAN/engine"

def patch_trade_engine():
    path = os.path.join(ENGINE_DIR, "trade_engine.py")
    with open(path, "r") as f:
        content = f.read()

    # 1. MTF Sniper - replace hard block with penalty, add baseline
    old_sniper = '''                                if sig.direction == "BUY" and price_h4 < ema_h4:
                                    print(f"  [MTF_SNIPER] {symbol} BUY blocked — H4 trend is BEARISH")
                                    log_gate_block(symbol, "MTF_SNIPER_BLOCK", {"h4_ema55": ema_h4, "price": price_h4})
                                    continue
                                elif sig.direction == "SELL" and price_h4 > ema_h4:
                                    print(f"  [MTF_SNIPER] {symbol} SELL blocked — H4 trend is BULLISH")
                                    log_gate_block(symbol, "MTF_SNIPER_BLOCK", {"h4_ema55": ema_h4, "price": price_h4})
                                    continue'''
    
    new_sniper = '''                                if sig.direction == "BUY" and price_h4 < ema_h4:
                                    print(f"  [MTF_SNIPER] {symbol} BUY penalized -15% — H4 BEARISH")
                                    sig.confidence -= 0.15
                                elif sig.direction == "SELL" and price_h4 > ema_h4:
                                    print(f"  [MTF_SNIPER] {symbol} SELL penalized -15% — H4 BULLISH")
                                    sig.confidence -= 0.15'''

    if old_sniper in content:
        content = content.replace(old_sniper, new_sniper)
        print("Patched MTF Sniper")
    else:
        print("Could not find old MTF Sniper logic")

    # Limit confidence drop to -20% max total penalty
    confidence_cap = '''                    if sig.signal_type == "trend":
                        sig._baseline_confidence = sig.confidence'''
    if 'sig._baseline_confidence' not in content:
        content = content.replace('                    if sig.signal_type == "trend":', confidence_cap)
    
    cap_logic = '''                            pass  # Fail open if H4 fetch fails
                            
                    # Prevent Penalty death stack (cap at -20%)
                    if hasattr(sig, "_baseline_confidence"):
                        if sig.confidence < (sig._baseline_confidence - 0.20):
                            sig.confidence = sig._baseline_confidence - 0.20
                            print(f"  [INFO] Penalty capped at -20% for {symbol}")

                    qualified.append(sig)'''
    content = content.replace('''                            pass  # Fail open if H4 fetch fails
                            
                    qualified.append(sig)''', cap_logic)

    # 2. Strategy Specific Exits mapped to Crypto Math (Percentage Based)
    # 0.1% = 0.001
    old_sl_tp = '''            if self._sl_pips > 0 and live_mid:
                _pip = live_mid * 0.001   # crypto: 0.1% of price per pip-equiv
                _sl_dist = self._sl_pips * _pip
                _tp_dist = self._tp_pips * _pip'''
    
    new_sl_tp = '''            if self._sl_pips > 0 and live_mid:
                _strategy  = getattr(sig, '_strategy',  getattr(sig, 'signal_type', 'trend')).lower()
                
                # Default to env-based sizing
                _trade_sl_pips = self._sl_pips
                _trade_tp_pips = self._tp_pips
                
                if any(x in _strategy for x in ["reversal", "mean_rev", "scalp", "yt_macd"]):
                    _trade_sl_pips = 12
                    _trade_tp_pips = 24
                    print(f"  [STRATEGY EXIT] {_strategy} detected — using tighter {_trade_sl_pips}/{_trade_tp_pips} exits")
                else:
                    print(f"  [STRATEGY EXIT] {_strategy} detected — using standard {_trade_sl_pips}/{_trade_tp_pips} exits")

                _pip = live_mid * 0.001   # crypto: 0.1% of price per pip-equiv
                _sl_dist = _trade_sl_pips * _pip
                _tp_dist = _trade_tp_pips * _pip'''
    
    if old_sl_tp in content:
        content = content.replace(old_sl_tp, new_sl_tp)
        print("Patched Strategy Exits")
    else:
        print("Could not find old SL/TP logic")
    
    with open(path, "w") as f:
        f.write(content)

def patch_trade_manager():
    path = os.path.join(ENGINE_DIR, "trade_manager.py")
    with open(path, "r") as f:
        content = f.read()
    
    # Enable fallback variables inside the code in case Coinbase .env lacks them
    old_env_fallbacks = '''        # Profit target close config (full close at % of TP distance reached)'''
    new_env_fallbacks = '''        # Profit target close config (full close at % of TP distance reached)
        self._profit_target_pct = float(os.getenv("RBOT_PROFIT_TARGET_PCT", "75"))
        if self._profit_target_pct > 1.0:
            self._profit_target_pct = self._profit_target_pct / 100.0
            
        self._green_lock_pips = float(os.getenv("RBOT_GREEN_LOCK_PIPS", "2.0"))
        self._green_lock_min_profit_pips = float(os.getenv("RBOT_GREEN_LOCK_MIN_PROFIT_PIPS", "8.0"))'''
    
    if 'self._profit_target_pct' not in content:
        content = content.replace(old_env_fallbacks, new_env_fallbacks)
        print("Patched Manager Config")
    else:
        print("Manager config already patched")

    with open(path, "w") as f:
        f.write(content)

patch_trade_engine()
patch_trade_manager()
print("Done patching Coinbase execution engine files.")
