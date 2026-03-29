"""
engine/pre_market_scanner.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_FEATURE

Pre-Market and Mid-Session Daily Planner.
Runs automatically before each FX session open and at mid-session.

Behaviour (mirrors how a top-tier prop-desk prepares the day):
  1. Detect which FX session is ABOUT TO open (look-ahead window = 60 min)
  2. Score every watchlist symbol across all 10 signal detectors
  3. Apply session-quality multipliers (London open > NY open > overlap > Tokyo)
  4. Apply economic calendar veto (high-impact events within 30 min → block)
  5. Rank symbols by (vote_count × confidence) composite score
  6. Write a structured SESSION_PLAN to the log
  7. Return a ranked PlayBook that trade_engine.py can optionally use to
     prioritise which symbols to scan first when the session opens

Integration:
  - Called from trade_engine.py run() loop once per session boundary
  - Can also be called manually: python -m engine.pre_market_scanner

Read-only — does NOT place orders. All data from broker candle API.
"""

from __future__ import annotations

import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

from strategies.multi_signal_engine import scan_symbol, AggregatedSignal, session_bias

log = logging.getLogger("pre_market_scanner")

# ── Session windows (UTC) ────────────────────────────────────────────────────

SESSION_OPENS_UTC = {
    "weekly_open": (21,  0),    # 21:00 UTC Sunday = 17:00 ET = FX week start
    "tokyo":       ( 0,  0),    # 00:00 UTC Monday
    "london":      ( 7,  0),    # 07:00 UTC Mon–Fri
    "new_york":    (12,  0),    # 12:00 UTC Mon–Fri (8:00 AM ET)
}


# How far ahead (minutes) to trigger a pre-session scan
PRE_SESSION_LOOKAHEAD_MIN = int(os.getenv("RBOT_PRE_SESSION_LOOKAHEAD_MIN", "60"))

# High-impact event types that should veto a signal
HIGH_IMPACT_KEYWORDS = {
    "nonfarm", "cpi", "fomc", "gdp", "pmi flash", "rate decision",
    "interest rate", "boe", "rba", "rbnz", "boc", "snb",
    "ecb", "fed chair", "unemployment", "payroll",
}

# Default watchlist — mirrors CLEAN engine's 22-pair scan list
DEFAULT_WATCHLIST = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "CAD_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD",
    "GBP_AUD", "GBP_NZD", "AUD_NZD", "AUD_CAD", "AUD_CHF",
    "NZD_JPY", "GBP_CHF",
]


# ── PlayBook entry ───────────────────────────────────────────────────────────

class PlayBookEntry:
    """One ranked entry in the session playbook."""

    def __init__(
        self,
        rank: int,
        symbol: str,
        direction: str,
        confidence: float,
        votes: int,
        detectors: List[str],
        session: str,
        sl: float,
        tp: float,
        rr: float,
        vetoed: bool = False,
        veto_reason: str = "",
    ):
        self.rank = rank
        self.symbol = symbol
        self.direction = direction
        self.confidence = confidence
        self.votes = votes
        self.detectors = detectors
        self.session = session
        self.sl = sl
        self.tp = tp
        self.rr = rr
        self.vetoed = vetoed
        self.veto_reason = veto_reason

    def as_dict(self) -> dict:
        return {
            "rank":        self.rank,
            "symbol":      self.symbol,
            "direction":   self.direction,
            "confidence":  round(self.confidence, 4),
            "votes":       self.votes,
            "detectors":   self.detectors,
            "session":     self.session,
            "sl":          round(self.sl, 5),
            "tp":          round(self.tp, 5),
            "rr":          round(self.rr, 2),
            "vetoed":      self.vetoed,
            "veto_reason": self.veto_reason,
        }

    def __repr__(self) -> str:
        status = f"VETOED({self.veto_reason})" if self.vetoed else "ACTIVE"
        return (
            f"#{self.rank:2d} {self.symbol:8s} {self.direction:4s} "
            f"conf={self.confidence:.1%} ({self.votes}v) "
            f"RR={self.rr:.1f}  [{status}]"
        )


# ── Economic calendar stub ───────────────────────────────────────────────────

class EconomicCalendar:
    """
    Lightweight economic event veto gate.

    In production: integrate with ForexFactory / TradingEconomics API.
    As shipped: reads from logs/economic_events.json if present,
    otherwise fails open (no veto applied).

    File format (list of events):
    [
      {"time_utc": "2026-03-22T12:30:00", "currency": "USD",
       "event": "Nonfarm Payroll", "impact": "high"},
      ...
    ]
    """

    def __init__(self, calendar_path: Optional[str] = None):
        self._events: List[Dict] = []
        _path = calendar_path or os.path.join(
            os.path.dirname(__file__), "..", "logs", "economic_events.json"
        )
        try:
            with open(_path, "r") as f:
                self._events = json.load(f)
            log.info(f"[Calendar] Loaded {len(self._events)} events from {_path}")
        except FileNotFoundError:
            log.info("[Calendar] No economic_events.json found — calendar veto disabled")
        except Exception as e:
            log.warning(f"[Calendar] Failed to load events: {e} — calendar veto disabled")

    def is_high_impact_soon(
        self,
        currency: str,
        now_utc: Optional[datetime] = None,
        window_minutes: int = 30,
    ) -> Tuple[bool, str]:
        """
        Return (True, event_name) if a high-impact event involving `currency`
        is within `window_minutes` of now. Otherwise (False, '').
        """
        if not self._events:
            return False, ""
        now = now_utc or datetime.now(timezone.utc)
        window = timedelta(minutes=window_minutes)

        for ev in self._events:
            try:
                ev_time = datetime.fromisoformat(
                    ev["time_utc"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                continue

            # Currency match (both currencies in the pair)
            ev_ccy = str(ev.get("currency", "")).upper()
            if ev_ccy not in currency.upper():
                continue

            impact = str(ev.get("impact", "")).lower()
            event_name = str(ev.get("event", "")).lower()

            is_high = impact == "high" or any(
                kw in event_name for kw in HIGH_IMPACT_KEYWORDS
            )
            if not is_high:
                continue

            # Is it soon?
            delta = ev_time - now
            if timedelta(0) <= delta <= window:
                return True, ev.get("event", "high_impact_event")

        return False, ""


# ── Main scanner ─────────────────────────────────────────────────────────────

class PreMarketScanner:
    """
    Produces a ranked PlayBook of trade opportunities for the upcoming session.

    Usage:
        scanner = PreMarketScanner(broker_connector)
        playbook = scanner.run_scan(session_hint="london")
        for entry in playbook:
            print(entry)
    """

    def __init__(
        self,
        connector,                            # OandaConnector instance
        watchlist: Optional[List[str]] = None,
        candle_count: int = 250,
        granularity: str = "M15",
        min_confidence: float = 0.68,
        min_votes: int = 2,
    ):
        self.connector = connector
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.candle_count = candle_count
        self.granularity = granularity
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.calendar = EconomicCalendar()

        # Track last scan time per session to avoid duplicate scans
        self._last_scan: Dict[str, datetime] = {}

    # ── Session detection ────────────────────────────────────────────────────

    @staticmethod
    def upcoming_session(now_utc: Optional[datetime] = None) -> Optional[str]:
        """
        Return the name of the FX session that is opening within
        PRE_SESSION_LOOKAHEAD_MIN minutes, or None if no session is imminent.
        """
        now = now_utc or datetime.now(timezone.utc)
        h, m = now.hour, now.minute
        current_mins = h * 60 + m

        for session, (oh, om) in SESSION_OPENS_UTC.items():
            open_mins = oh * 60 + om
            delta = (open_mins - current_mins) % (24 * 60)   # wrap midnight
            if 0 <= delta <= PRE_SESSION_LOOKAHEAD_MIN:
                return session
        return None

    @staticmethod
    def mid_session_check(now_utc: Optional[datetime] = None) -> Optional[str]:
        """
        Return active session name at mid-session (2–3 h after open), else None.
        Used for mid-session re-ranking.
        """
        now = now_utc or datetime.now(timezone.utc)
        h = now.hour
        # Tokyo mid = 04:00-05:00 UTC, London mid = 10:00-11:00 UTC, NY mid = 15:00-16:00 UTC
        if 4 <= h < 5:
            return "tokyo_mid"
        if 10 <= h < 11:
            return "london_mid"
        if 15 <= h < 16:
            return "new_york_mid"
        return None

    # ── Core scan ────────────────────────────────────────────────────────────

    def run_scan(
        self,
        session_hint: Optional[str] = None,
        now_utc: Optional[datetime] = None,
    ) -> List[PlayBookEntry]:
        """
        Fetch candles for every symbol in watchlist, run all signal detectors,
        rank by composite score, apply calendar veto, return ordered PlayBook.

        Args:
            session_hint: override auto-detected session ("london", "new_york", etc.)
            now_utc:      override current time (for testing)

        Returns:
            List[PlayBookEntry] sorted by rank (best first).
        """
        now = now_utc or datetime.now(timezone.utc)
        session_label = session_hint or self.upcoming_session(now) or "pre_scan"

        print(f"\n{'='*60}")
        print(f"  PRE-MARKET SCAN  —  {session_label.upper()}  session")
        print(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")

        raw_signals: List[Tuple[AggregatedSignal, float]] = []   # (signal, score)

        for symbol in self.watchlist:
            try:
                candles = self.connector.get_historical_data(
                    symbol,
                    count=self.candle_count,
                    granularity=self.granularity,
                )
                if not candles or len(candles) < 60:
                    continue

                sig = scan_symbol(
                    symbol,
                    candles,
                    utc_now=now,
                    min_confidence=self.min_confidence,
                    min_votes=self.min_votes,
                )
                if sig is None:
                    continue

                # Composite score: votes weighted by confidence
                # A 3-vote 78% signal scores higher than a 2-vote 85% signal
                # because conviction breadth matters more than single-detector height
                score = sig.votes * sig.confidence
                raw_signals.append((sig, score))

            except Exception as e:
                log.debug(f"[Scan] {symbol} error: {e}")

        # Sort descending by score
        raw_signals.sort(key=lambda x: x[1], reverse=True)

        # Build PlayBook with calendar veto
        playbook: List[PlayBookEntry] = []
        rank = 0

        for sig, score in raw_signals:
            rank += 1
            vetoed = False
            veto_reason = ""

            # Calendar veto: does the pair's currencies have high-impact news soon?
            is_news, news_name = self.calendar.is_high_impact_soon(sig.symbol, now)
            if is_news:
                vetoed = True
                veto_reason = f"HIGH_IMPACT_EVENT:{news_name}"

            entry = PlayBookEntry(
                rank=rank,
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=sig.confidence,
                votes=sig.votes,
                detectors=sig.detectors_fired,
                session=sig.session,
                sl=sig.sl,
                tp=sig.tp,
                rr=sig.rr,
                vetoed=vetoed,
                veto_reason=veto_reason,
            )
            playbook.append(entry)
            status = f"  ❌ VETOED  {veto_reason}" if vetoed else "  ✅ ACTIVE"
            print(f"  #{rank:02d} {sig.symbol:8s} {sig.direction:4s} "
                  f"conf={sig.confidence:.1%} ({sig.votes}v) RR={sig.rr:.1f}{status}")

        # Session summary
        active = [e for e in playbook if not e.vetoed]
        print(f"\n  SUMMARY: {len(active)} active setups / {len(playbook)} total")
        if active:
            top = active[0]
            print(f"  TOP PICK: {top.symbol} {top.direction} "
                  f"conf={top.confidence:.1%} ({top.votes}v) RR={top.rr:.1f}")
        print(f"{'='*60}\n")

        # Log session plan to narration
        self._log_session_plan(session_label, now, playbook)

        return playbook

    # ── Should-scan gate ─────────────────────────────────────────────────────

    def should_run_now(self, now_utc: Optional[datetime] = None) -> bool:
        """
        Return True if a pre-market or mid-session scan is due.
        Prevents scanning more than once per session window.
        """
        now = now_utc or datetime.now(timezone.utc)
        upcoming = self.upcoming_session(now)
        mid = self.mid_session_check(now)
        trigger = upcoming or mid
        if not trigger:
            return False

        last = self._last_scan.get(trigger)
        if last and (now - last).total_seconds() < 3600:
            return False   # Already scanned this session

        self._last_scan[trigger] = now
        return True

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_session_plan(
        self,
        session: str,
        now: datetime,
        playbook: List[PlayBookEntry],
    ) -> None:
        try:
            from util.narration_logger import log_event
            log_event(
                "SESSION_PLAN",
                symbol="SYSTEM",
                venue="pre_market_scanner",
                details={
                    "session":       session,
                    "scan_time_utc": now.isoformat(),
                    "total_signals": len(playbook),
                    "active":        sum(1 for e in playbook if not e.vetoed),
                    "vetoed":        sum(1 for e in playbook if e.vetoed),
                    "playbook":      [e.as_dict() for e in playbook[:10]],  # top 10
                },
            )
        except Exception:
            pass   # Never let logging crash the scanner


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from brokers.oanda_connector import OandaConnector

    connector = OandaConnector()
    scanner = PreMarketScanner(connector)
    playbook = scanner.run_scan()
    print(json.dumps([e.as_dict() for e in playbook], indent=2))
