#!/usr/bin/env python3
"""
narration_logger.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Writes append-only structured events to:
  logs/narration.jsonl  — all trading events
  logs/pnl.jsonl        — P&L summaries only (separate for easy reporting)

Import event type constants from here. Never use raw strings in callers.

Usage:
    from util.narration_logger import log_event, log_trade_opened, CANDIDATE_FOUND
    log_event(CANDIDATE_FOUND, symbol="EUR_USD", venue="scan", details={...})
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── Output paths ───────────────────────────────────────────────────────────────
_REPO_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOG_DIR          = os.path.join(_REPO_ROOT, "logs")
_DEFAULT_NAR      = os.path.join(_LOG_DIR, "narration.jsonl")
_DEFAULT_PNL      = os.path.join(_LOG_DIR, "pnl.jsonl")
_NARRATION_FILE   = os.getenv("RBOT_NARRATION_FILE", _DEFAULT_NAR)
_PNL_FILE         = os.getenv("RBOT_PNL_FILE",       _DEFAULT_PNL)

# ── Event type constants ───────────────────────────────────────────────────────
# Signal gate
CANDIDATE_FOUND               = "CANDIDATE_FOUND"
ORDER_SUBMIT_ALLOWED          = "ORDER_SUBMIT_ALLOWED"

# Tradability gate blocks
MARKET_CLOSED_BLOCK           = "MARKET_CLOSED_BLOCK"
INSTRUMENT_NOT_TRADABLE_BLOCK = "INSTRUMENT_NOT_TRADABLE_BLOCK"
STALE_QUOTE_BLOCK             = "STALE_QUOTE_BLOCK"
SPREAD_TOO_WIDE_BLOCK         = "SPREAD_TOO_WIDE_BLOCK"
QUOTE_FETCH_FAILED            = "QUOTE_FETCH_FAILED"
ORDER_SUBMIT_BLOCK            = "ORDER_SUBMIT_BLOCK"
OCO_VALIDATION_BLOCK          = "OCO_VALIDATION_BLOCK"
SYMBOL_ALREADY_ACTIVE_BLOCK   = "SYMBOL_ALREADY_ACTIVE_BLOCK"

# Trade lifecycle
TRADE_OPENED                  = "TRADE_OPENED"
TRADE_OPEN_FAILED             = "TRADE_OPEN_FAILED"
TRADE_CLOSED                  = "TRADE_CLOSED"
OCO_ERROR                     = "OCO_ERROR"

# Trail SL
TRAIL_SL_SET                  = "TRAIL_SL_SET"
TRAIL_SL_REJECTED             = "TRAIL_SL_REJECTED"
TRAIL_CANDIDATE               = "TRAIL_CANDIDATE"
TRAIL_SUBMIT_ALLOWED          = "TRAIL_SUBMIT_ALLOWED"
TRAIL_NO_ACTION               = "TRAIL_NO_ACTION"
BREAK_EVEN_SET                = "BREAK_EVEN_SET"

# Position management
POSITION_SYNCED               = "POSITION_SYNCED"
POSITION_CLOSED               = "POSITION_CLOSED"

# Engine lifecycle
ENGINE_STARTED                = "ENGINE_STARTED"
ENGINE_STOPPED                = "ENGINE_STOPPED"
SIGNAL_SCAN_COMPLETE          = "SIGNAL_SCAN_COMPLETE"

# Risk gate
MARGIN_GATE_BLOCKED           = "MARGIN_GATE_BLOCKED"
MARGIN_GATE_PASSED            = "MARGIN_GATE_PASSED"

# Safety mode
ATTACH_ONLY_BLOCK             = "ATTACH_ONLY_BLOCK"

# Trade manager lifecycle
MANAGER_CYCLE_STARTED         = "MANAGER_CYCLE_STARTED"
TRADE_MANAGER_ACTIVATED       = "TRADE_MANAGER_ACTIVATED"
TRADE_MANAGER_DEACTIVATED     = "TRADE_MANAGER_DEACTIVATED"

# Green-lock enforcement
GREEN_LOCK_ENFORCED           = "GREEN_LOCK_ENFORCED"

# Hard dollar stop
HARD_DOLLAR_STOP              = "HARD_DOLLAR_STOP"

# RBZ tight trailing
TRAIL_TIGHT_SET               = "TRAIL_TIGHT_SET"

# Capital router reallocation
CAPITAL_REALLOC_DECIDED       = "CAPITAL_REALLOC_DECIDED"
CAPITAL_REALLOC_FAILED        = "CAPITAL_REALLOC_FAILED"


# ── Internal writer ────────────────────────────────────────────────────────────

def _write_jsonl(path: str, record: dict) -> None:
    """
    Append one JSONL record to path.
    On failure: write one short fallback line to stderr. Never raises.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        try:
            sys.stderr.write(f"[narration_logger] WRITE_FAIL path={path} err={exc}\n")
        except Exception:
            pass


# ── Core public function ───────────────────────────────────────────────────────

def log_event(
    event_type: str,
    symbol: str = "SYSTEM",
    venue: str = "engine",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one JSONL record to narration.jsonl.
    Never raises — engine must not crash on a log failure.

    Schema (runtime-confirmed, 2026-03-17):
        {
            "timestamp":  "2026-03-17T08:18:20.671331+00:00",
            "event_type": "CANDIDATE_FOUND",
            "symbol":     "EUR_USD",
            "venue":      "signal_scan",
            "details":    { ... }
        }
    """
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "symbol":     symbol,
        "venue":      venue,
        "details":    details or {},
    }
    _write_jsonl(_NARRATION_FILE, record)


# ── Structured convenience wrappers ───────────────────────────────────────────

def log_trade_opened(
    symbol:      str,
    direction:   str,
    trade_id:    str,
    entry:       float,
    stop_loss:   float,
    take_profit: float,
    size:        int,
    confidence:  float,
    votes:       int,
    detectors:   List[str],
    session:     str,
) -> None:
    """
    Call ONLY after broker returns a confirmed trade_id.
    Schema uses Phoenix-compatible field names for downstream tooling.
    """
    log_event(
        event_type=TRADE_OPENED,
        symbol=symbol,
        venue="oanda",
        details={
            "trade_id":          trade_id,
            "direction":         direction,
            "entry_price":       entry,
            "stop_loss":         stop_loss,
            "take_profit":       take_profit,
            "size":              size,
            "signal_confidence": round(confidence, 4),
            "signal_votes":      votes,
            "signal_detectors":  detectors,
            "signal_session":    session,
        },
    )


def log_trade_closed(
    symbol:   str,
    trade_id: str,
    pnl_usd:  float,
    reason:   str,
    detail:   Optional[Dict[str, Any]] = None,
) -> None:
    """Log trade close to narration.jsonl."""
    log_event(
        event_type=TRADE_CLOSED,
        symbol=symbol,
        venue="oanda",
        details={"trade_id": trade_id, "pnl_usd": pnl_usd, "reason": reason, **(detail or {})},
    )


def log_gate_block(
    symbol:     str,
    event_type: str,
    detail:     Optional[Dict[str, Any]] = None,
) -> None:
    """Log any gate rejection. event_type must be one of the *_BLOCK constants."""
    log_event(event_type=event_type, symbol=symbol, venue="tradability_gate", details=detail or {})


def log_trail_rejected(
    symbol:      str,
    trade_id:    str,
    proposed_sl: float,
    error:       str,
) -> None:
    """
    Call when broker does NOT confirm a stop-loss update.
    After calling this, do NOT update local pos['stop_loss'].
    """
    log_event(
        event_type=TRAIL_SL_REJECTED,
        symbol=symbol,
        venue="trade_manager",
        details={"trade_id": trade_id, "proposed_sl": proposed_sl, "error": error},
    )


def log_pnl(
    symbol:   str,
    trade_id: str,
    pnl_usd:  float,
    details:  Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write P&L event to pnl.jsonl (separate file from narration.jsonl).
    Also writes to narration for full audit trail.
    """
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "event_type": TRADE_CLOSED,
        "symbol":     symbol,
        "venue":      "oanda",
        "details":    {
            "trade_id": trade_id,
            "pnl_usd":  pnl_usd,
            **(details or {}),
        },
    }
    _write_jsonl(_PNL_FILE, record)
    _write_jsonl(_NARRATION_FILE, record)


# ── Phoenix backward-compat alias ─────────────────────────────────────────────

def log_narration(
    event_type: str,
    details: dict,
    symbol: str = "SYSTEM",
    venue: str = "engine",
) -> None:
    """Alias used by EXTRACTED_VERIFIED files from Phoenix. Do not use in new code."""
    log_event(event_type=event_type, symbol=symbol, venue=venue, details=details)
