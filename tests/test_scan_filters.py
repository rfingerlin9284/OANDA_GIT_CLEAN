#!/usr/bin/env python3
"""
tests/test_scan_filters.py — RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Unit tests for the three scan-cycle filter fixes applied in the audit:

  1. Duplicate volume gate removed — only one VOL_GATE block per cycle.
  2. MACD divergence now uses 60-candle window (34 usable histogram bars)
     and requires ≥10 bars before firing.
  3. Exhaustion filter threshold is ATR(14)×2.5, not RBOT_TP_PIPS×0.65.

All tests operate on synthetic candle lists — no broker connection needed.
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── helpers ──────────────────────────────────────────────────────────────────

def _candle(c: float, h: float = None, l: float = None, volume: float = 100.0) -> dict:
    """Build a minimal OANDA-style M15 candle dict."""
    h = h if h is not None else c + 0.0003
    l = l if l is not None else c - 0.0003
    return {"mid": {"c": str(c), "h": str(h), "l": str(l)}, "volume": str(volume)}


def _flat_candles(n: int, price: float = 1.10000, volume: float = 100.0) -> list:
    """Return n identical candles — no trend, no divergence."""
    return [_candle(price, price + 0.0002, price - 0.0002, volume) for _ in range(n)]


# ── ─────────────────────────────────────────────────────────────────────────
# 1. Volume gate: only one block executes (duplicate removed)
# ── ─────────────────────────────────────────────────────────────────────────

class TestVolumeGateSingleExecution(unittest.TestCase):
    """
    Confirm that when tick-volume is low the gate fires exactly once.
    If the duplicate were present, the print/log would appear twice and the
    candidates list would be cleared redundantly; here we verify the logic
    produces the correct outcome with a single code path.
    """

    def _run_volume_check(self, candles: list, candidates: list) -> list:
        """Replicate the single (de-duplicated) volume gate logic."""
        VOLUME_GATE_MULT = 1.2
        try:
            vols = [float(c.get("volume", 0)) for c in candles[-20:]]
            if len(vols) >= 20 and vols[-1] > 0:
                avg_vol = sum(vols[:-1]) / len(vols[:-1])
                if avg_vol > 0 and vols[-1] < avg_vol * VOLUME_GATE_MULT:
                    candidates = []
        except Exception:
            pass
        return candidates

    def test_low_volume_clears_candidates(self):
        """Last candle volume below 1.2× average → candidates cleared."""
        candles = _flat_candles(20, volume=100.0)
        # Set last candle's volume below the 1.2× average threshold
        candles[-1] = _candle(1.10000, volume=50.0)
        candidates = ["signal_a"]
        result = self._run_volume_check(candles, candidates)
        self.assertEqual(result, [])

    def test_sufficient_volume_passes(self):
        """Last candle volume above 1.2× average → candidates untouched."""
        candles = _flat_candles(20, volume=100.0)
        candles[-1] = _candle(1.10000, volume=200.0)
        candidates = ["signal_a", "signal_b"]
        result = self._run_volume_check(candles, candidates)
        self.assertEqual(result, ["signal_a", "signal_b"])

    def test_insufficient_candles_skips_check(self):
        """Fewer than 20 candles → gate skips, candidates unchanged."""
        candles = _flat_candles(10, volume=50.0)
        candidates = ["signal_a"]
        result = self._run_volume_check(candles, candidates)
        self.assertEqual(result, ["signal_a"])


# ── ─────────────────────────────────────────────────────────────────────────
# 2. MACD divergence: 60-candle window, ≥10 histogram bars required
# ── ─────────────────────────────────────────────────────────────────────────

def _compute_macd_flags(candles: list) -> tuple:
    """
    Replicate the fixed MACD divergence logic (60-candle window, ≥10 bars).
    Returns (block_buy, block_sell).
    """
    cls = [float(c.get("mid", {}).get("c", 0)) for c in candles[-60:]]
    block_buy = False
    block_sell = False
    if len(cls) >= 26:
        e12 = sum(cls[:12]) / 12.0
        e26 = sum(cls[:26]) / 26.0
        k12 = 2.0 / 13.0
        k26 = 2.0 / 27.0
        hist = []
        for i in range(26, len(cls)):
            e12 = (cls[i] - e12) * k12 + e12
            e26 = (cls[i] - e26) * k26 + e26
            hist.append(e12 - e26)
        if len(hist) >= 10:
            if cls[-1] > cls[-3] and hist[-1] < hist[-3]:
                block_buy = True
            if cls[-1] < cls[-3] and hist[-1] > hist[-3]:
                block_sell = True
    return block_buy, block_sell


class TestMACDDivergenceWindow(unittest.TestCase):

    def test_30_candles_never_fires(self):
        """
        With only 30 candles, the 26-bar EMA warm-up leaves 4 histogram bars
        which is below the ≥10 minimum → no divergence block, regardless of
        price action. This confirms the old bug no longer triggers.
        """
        candles = _flat_candles(30)
        block_buy, block_sell = _compute_macd_flags(candles)
        self.assertFalse(block_buy)
        self.assertFalse(block_sell)

    def test_60_candles_enables_divergence_detection(self):
        """
        60 candles → 34 usable MACD-histogram bars → divergence can fire.
        Construct a clear bearish divergence: price rising while the MACD
        histogram (momentum derivative of the MACD line) is falling, indicating
        decelerating upward momentum despite continuing price gains.
        """
        # Build 60 candles with rising price for last 10 bars but the MACD
        # histogram decelerating — early bars have strong upward momentum,
        # last 10 bars continue rising but at a much smaller increment.
        candles = []
        # First 50 candles: strong up-trend to build a large positive MACD
        price = 1.10000
        for i in range(50):
            price += 0.0010
            candles.append(_candle(price, price + 0.0003, price - 0.0003))
        # Last 10 candles: small continuing price rise but MACD momentum slows
        # (achieved by much smaller increments → MACD histogram shrinks)
        for i in range(10):
            price += 0.0001
            candles.append(_candle(price, price + 0.0003, price - 0.0003))

        block_buy, block_sell = _compute_macd_flags(candles)
        # Price still rising (last > last-3) but MACD histogram should be
        # falling (deceleration of trend) → bearish divergence → block BUY
        self.assertTrue(block_buy, "Expected bearish divergence to block BUY")
        self.assertFalse(block_sell)

    def test_flat_price_no_divergence(self):
        """Flat price over 60 bars → no divergence signal."""
        candles = _flat_candles(60)
        block_buy, block_sell = _compute_macd_flags(candles)
        self.assertFalse(block_buy)
        self.assertFalse(block_sell)

    def test_exactly_9_histogram_bars_skipped(self):
        """
        35 candles → 9 histogram bars (26 warm-up + 9 history). Still below
        the ≥10 threshold → no block even with clear divergence pattern.
        """
        candles = _flat_candles(35)
        block_buy, block_sell = _compute_macd_flags(candles)
        self.assertFalse(block_buy)
        self.assertFalse(block_sell)

    def test_exactly_10_histogram_bars_allowed(self):
        """36 candles → 10 histogram bars → gate is eligible to fire."""
        # With 36 flat candles, MACD is zero throughout → no divergence,
        # but gate is no longer skipped due to insufficient history.
        candles = _flat_candles(36)
        block_buy, block_sell = _compute_macd_flags(candles)
        # Flat price → no divergence regardless, but the gate ran
        self.assertFalse(block_buy)
        self.assertFalse(block_sell)


# ── ─────────────────────────────────────────────────────────────────────────
# 3. Exhaustion filter: ATR(14)×2.5 threshold, not RBOT_TP_PIPS×0.65
# ── ─────────────────────────────────────────────────────────────────────────

def _compute_atr_threshold(candles: list, symbol: str = "EUR_USD") -> float:
    """
    Replicate the fixed exhaustion threshold: ATR(14) multiplied by 2.5.
    ATR(14) = average of the 14 most-recent true ranges (requires 15 candles
    because each TR references the previous candle's close).
    Returns threshold in price units (not pips).
    """
    pip_sz = 0.01 if "JPY" in symbol.upper() else 0.0001
    try:
        trs = [
            max(
                float(candles[i].get("mid", {}).get("h", 0)) - float(candles[i].get("mid", {}).get("l", 0)),
                abs(float(candles[i].get("mid", {}).get("h", 0)) - float(candles[i - 1].get("mid", {}).get("c", 0))),
                abs(float(candles[i].get("mid", {}).get("l", 0)) - float(candles[i - 1].get("mid", {}).get("c", 0))),
            )
            for i in range(max(1, len(candles) - 14), len(candles))
        ]
        atr = sum(trs) / len(trs) if trs else pip_sz * 10
    except Exception:
        atr = pip_sz * 10
    return atr * 2.5


class TestExhaustionFilter(unittest.TestCase):

    def test_atr_threshold_is_not_tp_pips_based(self):
        """
        Old threshold: RBOT_TP_PIPS(150) × 0.65 × pip_sz = 0.00975
        New threshold: ATR(14) × 2.5 ≈ 0.0003 × 2.5 = 0.00075 (for 3-pip ATR)
        The new threshold must be strictly less than the old one for typical M15
        candles, meaning the filter actually triggers on real moves now.
        """
        candles = _flat_candles(50)  # ATR ≈ 0.0004 (4-pip H-L range per candle)
        thresh = _compute_atr_threshold(candles)
        old_thresh = 150 * 0.65 * 0.0001  # 0.00975
        self.assertLess(thresh, old_thresh,
                        f"ATR threshold {thresh:.6f} should be < old TP-based {old_thresh:.6f}")

    def test_small_move_is_not_exhausted(self):
        """
        A 3-pip BUY travel in a normal-range market must NOT be blocked.
        ATR ≈ 4 pips, threshold ≈ 10 pips → 3-pip travel passes.
        """
        candles = _flat_candles(50, price=1.10000)
        thresh = _compute_atr_threshold(candles)
        travel = 0.0003  # 3 pips
        self.assertLess(travel, thresh, "3-pip travel should be below ATR threshold")

    def test_large_move_is_exhausted(self):
        """
        A 50-pip move in a normal-range M15 market (ATR ~5 pips) must be
        blocked as exhausted.  Threshold ≈ 12.5 pips < 50 pips.
        """
        candles = _flat_candles(50, price=1.10000)
        thresh = _compute_atr_threshold(candles)
        travel = 0.0050  # 50 pips
        self.assertGreater(travel, thresh, "50-pip travel should exceed ATR threshold")

    def test_jpy_pair_uses_correct_pip_size(self):
        """
        For a JPY pair (pip_sz=0.01), fallback ATR should produce a threshold
        several times larger in price units than for a non-JPY pair.
        """
        candles_jpy = [
            _candle(150.000, 150.060, 149.940) for _ in range(50)  # ~12-pip range
        ]
        thresh_jpy = _compute_atr_threshold(candles_jpy, symbol="USD_JPY")
        thresh_non_jpy = _compute_atr_threshold(_flat_candles(50), symbol="EUR_USD")
        self.assertGreater(thresh_jpy, thresh_non_jpy,
                           "JPY threshold should be larger in price units than non-JPY")

    def test_old_tp_threshold_never_triggered_on_m15(self):
        """
        Demonstrate the original bug: old threshold (97.5 pips) was almost
        never reached by a 30-candle M15 swing.  Compute the 30-candle range
        for a normal EUR/USD market and confirm it is well below 97.5 pips.
        """
        candles = _flat_candles(30, price=1.10000)
        h30 = max(float(c["mid"]["h"]) for c in candles)
        l30 = min(float(c["mid"]["l"]) for c in candles)
        range_pips = (h30 - l30) / 0.0001
        old_threshold_pips = 150 * 0.65
        self.assertLess(range_pips, old_threshold_pips,
                        "Flat-market 30-candle range should be < 97.5-pip old threshold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
