import os

# 1. Update .env values
env_path = ".env"
with open(env_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("RBOT_TS_PIPS="):
        new_lines.append("RBOT_TS_PIPS=0\n")
    elif line.startswith("RBOT_GREEN_LOCK_PIPS="):
        new_lines.append("RBOT_GREEN_LOCK_PIPS=3.0\n")
    elif line.startswith("RBOT_GREEN_LOCK_MIN_PROFIT_PIPS="):
        new_lines.append("RBOT_GREEN_LOCK_MIN_PROFIT_PIPS=12.0\n")
    else:
        new_lines.append(line)

with open(env_path, "w") as f:
    f.writelines(new_lines)

# 2. Update trade_engine.py Reversal Targets (10/25)
engine_path = "engine/trade_engine.py"
with open(engine_path, "r") as f:
    engine_code = f.read()

old_logic = """if any(x in _strategy for x in ["reversal", "mean_rev", "scalp"]):
                    _trade_sl_pips = 12
                    _trade_tp_pips = 24"""

new_logic = """if any(x in _strategy for x in ["reversal", "mean_rev", "scalp"]):
                    _trade_sl_pips = 10
                    _trade_tp_pips = 25"""

if old_logic in engine_code:
    with open(engine_path, "w") as f:
        f.write(engine_code.replace(old_logic, new_logic))
    print("✓ Successfully patched trade_engine.py Reversal targets to 10-SL / 25-TP")
else:
    print("⚠️ Could not find exact Reversal/Scalp string in trade_engine.py")

print("✓ Successfully unbound OANDA Native Trail and widened Green Lock to 12.0 triggers!")
