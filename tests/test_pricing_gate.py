#!/usr/bin/env python3
"""
test_pricing_gate.py — RBOTZILLA_OANDA_CLEAN Phase 9
Label: NEW_CLEAN_REWRITE

Tests for engine/broker_tradability_gate.py.
Covers:
  - Spread math for JPY and non-JPY pairs
  - Timestamp parsing for OANDA nanosecond format
  - Structured result shape
  - All 4 block reason codes

These tests use mock connector objects — no real OANDA calls.
Runtime verification against live OANDA requires your terminal.
"""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.broker_tradability_gate import (
    check_broker_tradability,
    ORDER_SUBMIT_ALLOWED, SPREAD_TOO_WIDE_BLOCK,
    MARKET_CLOSED_BLOCK, INSTRUMENT_NOT_TRADABLE_BLOCK, QUOTE_FETCH_FAILED,
)


# ── Mock connector helpers ────────────────────────────────────────────────────

def _fresh_ts() -> str:
    """Return an OANDA-format timestamp that is <10 seconds old."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _stale_ts(seconds: int = 200) -> str:
    """Return an OANDA-format timestamp that is `seconds` old."""
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


class MockConnector:
    account_id = "TEST-ACCOUNT-001"

    def __init__(self, prices_payload):
        self._payload = prices_payload

    def _make_request(self, method, endpoint, params=None):
        return self._payload


def _make_price(bid, ask, tradeable=True, ts=None):
    return {
        "success": True,
        "data": {
            "prices": [{
                "tradeable": tradeable,
                "time":      ts or _fresh_ts(),
                "bids":      [{"price": str(bid), "liquidity": 500000}],
                "asks":      [{"price": str(ask), "liquidity": 500000}],
                "status":    "tradeable" if tradeable else "halted",
            }]
        }
    }


# ── Test cases ────────────────────────────────────────────────────────────────

class TestSpreadMathNonJPY(unittest.TestCase):
    def test_within_limit(self):
        # EUR_USD spread: 1.6 pips — should pass
        c = MockConnector(_make_price(1.14840, 1.14856))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertTrue(r["allowed"], f"Expected allowed, got {r}")
        self.assertEqual(r["event"], ORDER_SUBMIT_ALLOWED)

    def test_exceed_limit(self):
        # 9.3 pip spread — should block
        c = MockConnector(_make_price(2.28496, 2.28589))
        r = check_broker_tradability(c, "GBP_NZD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], SPREAD_TOO_WIDE_BLOCK)
        self.assertAlmostEqual(r["detail"]["spread_pips"], 9.3, places=0)

    def test_exact_limit(self):
        # Exactly 8.0 pips — should pass (boundary is strictly > MAX)
        c = MockConnector(_make_price(1.10000, 1.10080))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertTrue(r["allowed"], f"Expected allowed at boundary, got {r}")


class TestSpreadMathJPY(unittest.TestCase):
    def test_jpy_within_limit(self):
        # USD_JPY spread: 1.0 pip (pip_mult=100) — should pass
        c = MockConnector(_make_price(149.500, 149.510))
        r = check_broker_tradability(c, "USD_JPY")
        self.assertTrue(r["allowed"])

    def test_jpy_exceed_limit(self):
        # USD_JPY spread: 9.0 pips — should block
        c = MockConnector(_make_price(149.500, 149.590))
        r = check_broker_tradability(c, "USD_JPY")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], SPREAD_TOO_WIDE_BLOCK)
        self.assertAlmostEqual(r["detail"]["spread_pips"], 9.0, places=0)

    def test_cad_jpy_identified_correctly(self):
        # CAD_JPY contains JPY — should use pip_mult=100
        c = MockConnector(_make_price(108.000, 108.090))
        r = check_broker_tradability(c, "CAD_JPY")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], SPREAD_TOO_WIDE_BLOCK)


class TestTradableFlag(unittest.TestCase):
    def test_instrument_not_tradable(self):
        c = MockConnector(_make_price(1.14840, 1.14842, tradeable=False))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], INSTRUMENT_NOT_TRADABLE_BLOCK)


class TestQuoteFreshness(unittest.TestCase):
    def test_stale_quote_blocks(self):
        c = MockConnector(_make_price(1.14840, 1.14842, ts=_stale_ts(200)))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], MARKET_CLOSED_BLOCK)
        self.assertGreater(r["detail"]["quote_age_seconds"], 120)

    def test_fresh_quote_passes(self):
        c = MockConnector(_make_price(1.14840, 1.14842, ts=_fresh_ts()))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertTrue(r["allowed"])

    def test_nanosecond_timestamp_parses(self):
        # Exact format OANDA returns in production (confirmed 2026-03-17)
        ts = "2026-03-17T05:50:23.602157664Z"
        c = MockConnector(_make_price(1.14840, 1.14842, ts=ts))
        r = check_broker_tradability(c, "EUR_USD")
        # Will block (stale) — but should not raise/error on timestamp parsing
        self.assertIn("allowed", r)
        self.assertNotEqual(r["event"], "TRADABILITY_CHECK_ERROR")


class TestQuoteFetchFailure(unittest.TestCase):
    def test_empty_prices_list(self):
        c = MockConnector({"success": True, "data": {"prices": []}})
        r = check_broker_tradability(c, "EUR_USD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], QUOTE_FETCH_FAILED)

    def test_request_failure(self):
        c = MockConnector({"success": False})
        r = check_broker_tradability(c, "EUR_USD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], QUOTE_FETCH_FAILED)

    def test_none_response(self):
        c = MockConnector(None)
        r = check_broker_tradability(c, "EUR_USD")
        self.assertFalse(r["allowed"])
        self.assertEqual(r["event"], QUOTE_FETCH_FAILED)


class TestResultShape(unittest.TestCase):
    def test_allowed_result_has_required_keys(self):
        c = MockConnector(_make_price(1.14840, 1.14842))
        r = check_broker_tradability(c, "EUR_USD")
        self.assertIn("allowed", r)
        self.assertIn("event", r)
        self.assertIn("detail", r)

    def test_blocked_result_has_required_keys(self):
        c = MockConnector(_make_price(2.28496, 2.28589))
        r = check_broker_tradability(c, "GBP_NZD")
        self.assertIn("allowed", r)
        self.assertIn("event", r)
        self.assertIn("detail", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
