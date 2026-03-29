import sys

path1 = "engine/regime_detector.py"
try:
    with open(path1, "r") as f:
        text1 = f.read()

    target1 = """    def _calculate_regime_probabilities(self, vol: float, trend: float) -> Dict[str, float]:
        \"\"\"Calculate regime probabilities using softmax\"\"\"
        # Base scores for each regime"""

    replace1 = """    def _calculate_regime_probabilities(self, vol: float, trend: float) -> Dict[str, float]:
        \"\"\"Calculate regime probabilities using softmax\"\"\"
        # --- FOREX M15 SCALE ADJUSTMENT (OPERATOR PATCH) ---
        # Multiply by 1000.0 so the hardcoded scoring multipliers (x10, x20) function 
        # correctly and allow trending markets to mathematically outscore TRIAGE.
        trend = trend * 1000.0

        # Base scores for each regime"""

    if target1 in text1:
        with open(path1, "w") as f:
            f.write(text1.replace(target1, replace1))
        print("✓ Math logic patched in " + path1)
    elif "trend * 1000.0" in text1:
        print("✓ Already patched: " + path1)
    else:
        print("✗ Target block not found in " + path1)
except Exception as e:
    print(f"✗ Error reading {path1}: {e}")
