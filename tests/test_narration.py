#!/usr/bin/env python3
"""
test_narration.py — RBOTZILLA_OANDA_CLEAN Phase 9
Label: NEW_CLEAN_REWRITE

Tests for util/narration_logger.py.
Covers: file writing, JSON schema, event constants, wrappers.
Uses a temp file — no interference with live narration.jsonl.
"""

import sys
import os
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestNarrationLogger(unittest.TestCase):

    def setUp(self):
        # Point at a temp file for each test
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmpfile.close()
        os.environ["RBOT_NARRATION_FILE"] = self.tmpfile.name
        # Re-import to pick up patched env
        import importlib
        import util.narration_logger as nm
        importlib.reload(nm)
        self.nm = nm

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def _read_events(self):
        with open(self.tmpfile.name, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_log_event_writes_jsonl(self):
        self.nm.log_event("TEST_EVENT", symbol="EUR_USD", venue="test", details={"k": "v"})
        events = self._read_events()
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["event_type"], "TEST_EVENT")
        self.assertEqual(e["symbol"], "EUR_USD")
        self.assertEqual(e["venue"], "test")
        self.assertEqual(e["details"]["k"], "v")
        self.assertIn("timestamp", e)

    def test_log_event_appends(self):
        self.nm.log_event("EVENT_1")
        self.nm.log_event("EVENT_2")
        events = self._read_events()
        self.assertEqual(len(events), 2)

    def test_log_trade_opened_schema(self):
        self.nm.log_trade_opened(
            symbol="GBP_USD", direction="BUY", trade_id="123",
            entry=1.2500, stop_loss=1.2450, take_profit=1.2650,
            size=10000, confidence=0.78, votes=3,
            detectors=["ema_stack", "fibonacci"], session="london",
        )
        events = self._read_events()
        self.assertEqual(events[0]["event_type"], self.nm.TRADE_OPENED)
        d = events[0]["details"]
        self.assertEqual(d["trade_id"], "123")
        self.assertEqual(d["direction"], "BUY")
        self.assertEqual(d["stop_loss"], 1.2450)

    def test_log_gate_block_uses_correct_event_type(self):
        self.nm.log_gate_block("GBP_NZD", self.nm.SPREAD_TOO_WIDE_BLOCK, {"spread_pips": 9.3})
        events = self._read_events()
        self.assertEqual(events[0]["event_type"], "SPREAD_TOO_WIDE_BLOCK")
        self.assertEqual(events[0]["venue"], "tradability_gate")

    def test_log_trail_rejected_schema(self):
        self.nm.log_trail_rejected("EUR_USD", "456", 1.0800, "no success field")
        events = self._read_events()
        self.assertEqual(events[0]["event_type"], self.nm.TRAIL_SL_REJECTED)
        self.assertEqual(events[0]["details"]["trade_id"], "456")

    def test_phoenix_alias_works(self):
        self.nm.log_narration("CANDIDATE_FOUND", {"symbol": "EUR_USD"}, symbol="EUR_USD")
        events = self._read_events()
        self.assertEqual(events[0]["event_type"], "CANDIDATE_FOUND")

    def test_never_raises_on_bad_path(self):
        # Even with an unwritable path, must not raise
        os.environ["RBOT_NARRATION_FILE"] = "/root/cannot_write_here/narration.jsonl"
        import importlib
        import util.narration_logger as nm2
        importlib.reload(nm2)
        try:
            nm2.log_event("WONT_CRASH")  # must not raise
        except Exception as e:
            self.fail(f"log_event raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
