#!/usr/bin/env python3
"""
scripts/human_tail.py — RBOTZILLA_OANDA_CLEAN
Plain-English live feed from narration.jsonl.
Run via tail_human.sh or the VS Code task.
"""
import json
import os
import sys
import time

LOG = "/home/rfing/RBOTZILLA_OANDA_CLEAN/logs/narration.jsonl"

# ── Event translation table ────────────────────────────────────────────────
# None = skip entirely (too noisy / internal)
LABELS = {
    "ENGINE_STARTED":              "🚀  Engine is ON",
    "ENGINE_STOPPED":              "⛔  Engine is OFF",
    "TRADE_MANAGER_ACTIVATED":     "🟢  Trade manager is watching",
    "TRADE_MANAGER_DEACTIVATED":   "🔴  Trade manager stopped",

    "CANDIDATE_FOUND":             "💡  Signal found",
    "ORDER_SUBMIT_ALLOWED":        "→   Placing order",
    "TRADE_OPENED":                "✅  Trade OPENED",
    "TRADE_OPEN_FAILED":           "❌  Trade FAILED to open",

    "POSITION_SYNCED":             "🔍  Now tracking trade",
    "POSITION_CLOSED":             "📋  Trade closed",

    "HARD_DOLLAR_STOP":            "🛑  MAX LOSS HIT — closed trade",
    "GREEN_LOCK_ENFORCED":         "🔒  Breakeven lock applied",
    "BREAK_EVEN_SET":              "🔒  Stop moved to breakeven",
    "TRAIL_TIGHT_SET":             "📈  Trailing stop tightened",
    "TRAIL_SL_SET":                "📈  Stop loss updated",
    "TRAIL_SL_REJECTED":           "⚠️   Stop loss update rejected by broker",

    "SYMBOL_ALREADY_ACTIVE_BLOCK": "⏸️   Skipped — already in that trade",
    "MARGIN_GATE_BLOCKED":         "⚠️   Skipped — margin too low",
    "OCO_VALIDATION_BLOCK":        "⚠️   Skipped — invalid order params",
    "ORDER_SUBMIT_BLOCK":          "❌  Order rejected by broker",
    "ATTACH_ONLY_BLOCK":           "🔒  No new trades (read-only mode)",
    "REGIME_BLOCK":                "🛑  Skipped — Market Regime (Chop/Crash) blocked entry",

    "CAPITAL_REALLOC_DECIDED":     "🔄  Reallocating capital to stronger trade",
    "CAPITAL_REALLOC_FAILED":      "⚠️   Reallocation failed",

    # Skip these — they fire every cycle and add no value to a human feed
    "MANAGER_CYCLE_STARTED":       None,
    "TRAIL_CANDIDATE":             "💹  Live Trade Status",
    "TRAIL_NO_ACTION":             None,
    "TRAIL_SUBMIT_ALLOWED":        None,
    "SIGNAL_SCAN_COMPLETE":        None,
    "CHARTER_VIOLATION":           "🚫  Charter rule blocked an order",
}


GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def fmt(e: dict) -> str | None:
    etype = e.get("event_type", "")
    label = LABELS.get(etype, f"[{etype}]")
    if label is None:
        return None

    sym = e.get("symbol", "")
    det = e.get("details") or {}
    ts = (e.get("timestamp") or "")
    # Dual timezone: EST date/time + [UTC date/time] with yellow UTC
    YELLOW = '\033[93m'
    try:
        from util.narration_logger import dual_timestamp
        ts_short = dual_timestamp(ts)
        bracket_start = ts_short.rfind("[")
        if bracket_start >= 0:
            ts_short = ts_short[:bracket_start] + YELLOW + ts_short[bracket_start:] + RESET
    except Exception:
        ts_short = ts[11:19] if len(ts) >= 19 else ts

    sym_str = f"  {sym}" if sym and sym != "SYSTEM" else ""

    # Determine default color based on event type
    color = ""
    if etype in {"TRADE_OPEN_FAILED", "HARD_DOLLAR_STOP", "TRAIL_SL_REJECTED", 
                 "SYMBOL_ALREADY_ACTIVE_BLOCK", "MARGIN_GATE_BLOCKED", 
                 "OCO_VALIDATION_BLOCK", "ORDER_SUBMIT_BLOCK", 
                 "ATTACH_ONLY_BLOCK", "CAPITAL_REALLOC_FAILED", 
                 "CHARTER_VIOLATION", "REGIME_BLOCK"}:
        color = RED
    elif etype in {"TRADE_OPENED", "GREEN_LOCK_ENFORCED", "BREAK_EVEN_SET", 
                   "TRAIL_TIGHT_SET", "TRAIL_SL_SET", "CAPITAL_REALLOC_DECIDED"}:
        color = GREEN

    msg = ""
    # ── Richer formatting for key events ──────────────────────────────────
    if etype == "TRADE_OPENED":
        direction = det.get("direction", "")
        entry     = det.get("entry", "")
        sl        = det.get("stop_loss", "")
        tp        = det.get("take_profit", "")
        conf      = det.get("confidence", "")
        conf_str  = f"  conf:{float(conf):.0%}" if conf != "" else ""
        msg = (
            f"{ts_short}  {label}{sym_str}  {direction} @ {entry}"
            f"  │  SL:{sl}  TP:{tp}{conf_str}"
        )

    elif etype == "CANDIDATE_FOUND":
        direction = det.get("direction", "")
        conf      = det.get("confidence", 0)
        votes     = det.get("votes", "")
        msg = (
            f"{ts_short}  {label}{sym_str}  {direction}"
            f"  conf:{float(conf):.0%}  ({votes} detectors agree)"
        )

    elif etype == "HARD_DOLLAR_STOP":
        pnl = det.get("unrealized_pnl", "")
        lim = det.get("limit", "")
        msg = f"{ts_short}  {label}{sym_str}  P&L: ${pnl}  (limit: -${lim})"

    elif etype == "GREEN_LOCK_ENFORCED":
        old_sl = det.get("old_sl", "")
        new_sl = det.get("new_sl", "")
        msg = f"{ts_short}  {label}{sym_str}  stop moved {old_sl} → {new_sl}"

    elif etype == "POSITION_CLOSED":
        reason = det.get("reason", "")
        reason_str = f"  ({reason})" if reason else ""
        msg = f"{ts_short}  {label}{sym_str}{reason_str}"
        
        # Color POSITION_CLOSED dynamically based on reason
        if any(r in reason.upper() for r in ["STOP_LOSS", "HARD_STOP", "REJECTED"]):
            color = RED
        elif any(r in reason.upper() for r in ["PROFIT_TARGET", "TRAILING_STOP", "GREEN_LOCK", "PROFIT"]):
            color = GREEN

    elif etype in ("TRAIL_SL_SET", "TRAIL_TIGHT_SET"):
        new_sl = det.get("new_sl", "")
        sl_str = f"  new stop: {new_sl}" if new_sl else ""
        msg = f"{ts_short}  {label}{sym_str}{sl_str}"

    elif etype == "TRAIL_CANDIDATE":
        direction = det.get("direction", "")
        entry = det.get("entry", "")
        price = det.get("price", "")
        pips = det.get("pips", 0.0)
        pnl = det.get("pnl", 0.0)
        profit_pct = det.get("profit_pct", 0.0)
        strategy = det.get("strategy", "Trend Continuation")
        tf = det.get("timeframe", "M15")
        rules = det.get("rules", "")

        is_locked = det.get("is_locked", False)
        sl_dist = det.get("sl_dist", 0.0)
        rr_ratio = det.get("rr_ratio", 0.0)

        # ANSI Badges
        BG_GREEN = '\033[42m\033[30m'  # Black on Green
        BG_YELLOW = '\033[43m\033[30m' # Black on Yellow
        BOLD_CYAN = '\033[1m\033[96m'

        lock_badge = f"{BG_GREEN} 🔒 GREEN-LOCKED {RESET}" if is_locked else f"{BG_YELLOW} ⚠️ AT RISK {RESET}"
        rr_str = f"{GREEN}+{rr_ratio:.1f}R{RESET}" if rr_ratio > 0 else f"{RED}{rr_ratio:.1f}R{RESET}"
        pnl_str = f"{GREEN}+${pnl:.2f} (+{profit_pct:.2f}%){RESET}" if pnl > 0 else f"{RED}-${abs(pnl):.2f} ({profit_pct:.2f}%){RESET}"
        pips_str = f"{GREEN}{pips:+.1f}p{RESET}" if pips > 0 else f"{RED}{pips:+.1f}p{RESET}"

        msg = (
            f"{ts_short}  {label} {BOLD_CYAN}{sym}{RESET} {direction}  [{strategy} {tf}]\n"
            f"          ├─ Status: {lock_badge}  |  {rr_str}  |  {pips_str}  |  {pnl_str}\n"
            f"          └─ SL Dist: {sl_dist:.1f} pips  |  Entry: {entry} → Now: {price:.5f}  |  {rules}"
        )

    elif etype == "CAPITAL_REALLOC_DECIDED":
        close_sym = det.get("close_symbol", "")
        open_sym  = det.get("open_symbol", "")
        msg = (
            f"{ts_short}  {label}  "
            f"closing {close_sym} → opening {open_sym}"
        )

    elif sym_str:
        msg = f"{ts_short}  {label}{sym_str}"
    else:
        msg = f"{ts_short}  {label}"
        
    return f"{color}{msg}{RESET}" if color else msg


def main() -> None:
    print("═" * 62)
    print("  RBOTZILLA — PLAIN ENGLISH FEED")
    print("  (reading narration.jsonl — q to quit)")
    print("═" * 62)
    sys.stdout.flush()

    while not os.path.exists(LOG):
        print(f"  Waiting for log file: {LOG}")
        time.sleep(2)

    with open(LOG, "r") as fh:
        fh.seek(0, 2)  # jump to end — only show new events
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.4)
                continue
            try:
                event = json.loads(line.strip())
                msg = fmt(event)
                if msg:
                    print(msg, flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
