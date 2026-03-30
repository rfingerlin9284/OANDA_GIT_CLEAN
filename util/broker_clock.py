"""
util/broker_clock.py
RBOTZILLA_OANDA_CLEAN — Professional-grade broker-authoritative time

Professional trading firms do not trust the OS clock for market decisions.
They query the exchange/broker for authoritative time, store the delta,
and apply it to every session check, candle timestamp, and cooldown calculation.

Architecture:
  - On engine startup: fetch OANDA broker time, compute offset vs local clock
  - Every 50 cycles: re-sync automatically to absorb OS drift
  - BrokerClock.now() → datetime in UTC, adjusted to broker's authoritative clock
  - BrokerClock.now_eastern() → same, in US/Eastern for session display
  - Fallback: if broker unreachable, uses local clock (never blocks the engine)

OS-level recommendation (run once as root):
  sudo apt-get install -y chrony
  sudo systemctl disable systemd-timesyncd
  sudo systemctl enable --now chrony
  chronyc tracking  # verify sync quality
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

_log = logging.getLogger("broker_clock")

# Eastern time helpers (no pytz dependency)
_EST = timezone(timedelta(hours=-5))   # UTC-5  (EST)
_EDT = timezone(timedelta(hours=-4))   # UTC-4  (EDT, Mar–Nov)


def _eastern_tz() -> timezone:
    """Return EDT in summer, EST in winter (simple DST rule)."""
    now_utc = datetime.now(timezone.utc)
    # DST: 2nd Sunday March → 1st Sunday November
    year = now_utc.year
    # 2nd Sunday of March
    dst_start = datetime(year, 3, 8, 2, 0, tzinfo=timezone.utc)
    dst_start += timedelta(days=(6 - dst_start.weekday()) % 7)
    # 1st Sunday of November
    dst_end = datetime(year, 11, 1, 2, 0, tzinfo=timezone.utc)
    dst_end += timedelta(days=(6 - dst_end.weekday()) % 7)
    return _EDT if dst_start <= now_utc < dst_end else _EST


class BrokerClock:
    """
    Singleton. Holds a running offset between local clock and OANDA broker clock.
    All engine time decisions should call BrokerClock.now() instead of datetime.now().
    """

    _instance: Optional["BrokerClock"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._offset_ms: float = 0.0          # broker_time - local_time in ms
        self._last_sync: Optional[datetime] = None
        self._sync_count: int = 0
        self._connector = None                 # set on first sync
        self._cycle_counter: int = 0
        self._resync_interval: int = 50        # re-sync every N engine cycles

    @classmethod
    def instance(cls) -> "BrokerClock":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Sync ─────────────────────────────────────────────────────────────────

    def sync(self, connector) -> dict:
        """
        Fetch OANDA broker time and compute offset.
        Returns status dict for logging.
        """
        self._connector = connector
        try:
            result = connector.get_server_time()
            broker_utc: datetime = result["broker_utc"]
            local_utc:  datetime = result["local_utc"]

            # Store offset: how many ms to ADD to local clock to get broker clock
            self._offset_ms = (broker_utc - local_utc).total_seconds() * 1000
            self._last_sync = datetime.now(timezone.utc)
            self._sync_count += 1

            return {
                "synced":     True,
                "offset_ms":  round(self._offset_ms, 1),
                "broker_utc": broker_utc,
                "local_utc":  local_utc,
                "drift_ms":   result.get("drift_ms", 0.0),
            }
        except Exception as e:
            _log.warning(f"BrokerClock sync failed — using local clock: {e}")
            return {"synced": False, "offset_ms": 0.0}

    def tick(self, connector=None) -> None:
        """
        Call once per engine cycle. Auto-resyncs every self._resync_interval cycles.
        """
        self._cycle_counter += 1
        if self._cycle_counter % self._resync_interval == 0:
            c = connector or self._connector
            if c:
                self.sync(c)

    # ── Time accessors ────────────────────────────────────────────────────────

    def now(self) -> datetime:
        """
        Current time adjusted by broker offset. UTC.
        If never synced, returns plain UTC (safe fallback).
        """
        local_now = datetime.now(timezone.utc)
        if self._offset_ms == 0.0:
            return local_now
        return local_now + timedelta(milliseconds=self._offset_ms)

    def now_eastern(self) -> datetime:
        """Current broker time converted to US Eastern (EDT/EST auto)."""
        return self.now().astimezone(_eastern_tz())

    def offset_ms(self) -> float:
        return self._offset_ms

    def last_sync_age_s(self) -> Optional[float]:
        if self._last_sync is None:
            return None
        return (datetime.now(timezone.utc) - self._last_sync).total_seconds()

    def status_line(self) -> str:
        """One-line status for the engine banner."""
        age = self.last_sync_age_s()
        age_str = f"{age:.0f}s ago" if age is not None else "never"
        drift = abs(self._offset_ms)
        icon = "✅" if drift < 2000 else "⚠️ "
        return (
            f"{icon} BROKERC LOCK  offset={self._offset_ms:+.0f}ms  "
            f"syncs={self._sync_count}  last={age_str}"
        )


# ── Module-level convenience ──────────────────────────────────────────────────

def broker_now() -> datetime:
    """Convenience: broker-authoritative UTC time."""
    return BrokerClock.instance().now()


def broker_now_eastern() -> datetime:
    """Convenience: broker-authoritative Eastern time."""
    return BrokerClock.instance().now_eastern()
