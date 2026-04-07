#!/usr/bin/env python3
"""
RBOTzilla Narrator — Real-Time Storyteller Dashboard
Translates raw engine logs into human-readable narrative.
Monitors both OANDA and Coinbase engines simultaneously.
"""

import os
import re
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
OANDA_LOG = Path("/home/rfing/RBOTZILLA_OANDA_CLEAN/logs/engine_continuous.out")
COINBASE_LOG = Path("/home/rfing/RBOTZILLA_COINBASE_CLEAN/logs/engine_continuous.out")
COINBASE_TRADES = Path("/home/rfing/RBOTZILLA_COINBASE_CLEAN/data/active_trades.json")

# ── ANSI Colors ───────────────────────────────────────────────────────────────
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
C = "\033[96m"   # cyan
B = "\033[94m"   # blue
M = "\033[95m"   # magenta
W = "\033[97m"   # white
D = "\033[2m"    # dim
BOLD = "\033[1m"
RST = "\033[0m"

# ── State ─────────────────────────────────────────────────────────────────────
portfolio = {
    "oanda_trades": {},
    "coinbase_trades": {},
    "oanda_events": [],
    "coinbase_events": [],
    "last_summary": 0,
}

SUMMARY_INTERVAL = 30  # seconds between portfolio summary prints


def ts():
    return datetime.now().strftime("%H:%M:%S")


def narrate(engine: str, line: str):
    """Translate a single log line into human-readable narrative."""
    line = line.strip()
    if not line:
        return

    tag = f"{C}[{engine}]{RST}"

    # ── Trade Opened ──────────────────────────────────────────
    m = re.search(r"✓ OPENED\s+(\S+)\s+trade_id=(\S+)", line)
    if m:
        print(f"  {tag} {G}🟢 NEW TRADE{RST} — Opened {BOLD}{m.group(1)}{RST} position (ID: {D}{m.group(2)[:8]}…{RST})")
        return

    # ── Trade Placement ───────────────────────────────────────
    m = re.search(r"→ Placing\s+(\S+)\s+(BUY|SELL)\s+conf=(\S+)", line)
    if m:
        direction = f"{G}BUY ↑{RST}" if m.group(2) == "BUY" else f"{R}SELL ↓{RST}"
        print(f"  {tag} 🎯 Placing {BOLD}{m.group(1)}{RST} {direction} — confidence {Y}{m.group(3)}{RST}")
        return

    # ── Qualified Signal ──────────────────────────────────────
    m = re.search(r"📋 Qualified: (\d+) signal", line)
    if m:
        count = m.group(1)
        print(f"  {tag} {M}📋 Found {count} qualified trade(s){RST} — placing best signals now")
        return

    # ── Scan complete narrative ───────────────────────────────
    m = re.search(r"💬 Scan complete: found (\d+) tradeable", line)
    if m:
        print(f"  {tag} {B}🔍 Scan found {m.group(1)} opportunity(s){RST}")
        return

    # ── Signal queued ─────────────────────────────────────────
    m = re.search(r"💬 (\S+): (.+) — queuing a (BUY|SELL)", line)
    if m:
        direction = f"{G}BUY{RST}" if m.group(3) == "BUY" else f"{R}SELL{RST}"
        print(f"  {tag} 💡 {BOLD}{m.group(1)}{RST}: {m.group(2)} → queuing {direction}")
        return

    # ── QUEUED ────────────────────────────────────────────────
    m = re.search(r"→ QUEUED \[(\w+)\]", line)
    if m:
        return  # handled by the signal queued line above

    # ── Scan line ─────────────────────────────────────────────
    m = re.search(r"\[SCAN\]\s+(\S+)\s+(.*)", line)
    if m:
        pair = m.group(1)
        signals = m.group(2)
        # Only narrate if something qualified (✓)
        hits = re.findall(r"(\w+)=(\d+%✓)", signals)
        if hits:
            readable = ", ".join(f"{h[0]} at {h[1][:-1]}" for h in hits)
            print(f"  {tag} 🔎 {pair}: signals detected — {readable}")
        return

    # ── VOL_GATE ──────────────────────────────────────────────
    m = re.search(r"\[VOL_GATE\] (\S+) (\w+) penalizing.*vol (\d+) < (\d+)", line)
    if m:
        # Only print once per pair per cycle (suppress duplicates)
        return  # These are noisy, skip in narrator

    # ── Manager SYNCED ────────────────────────────────────────
    m = re.search(r"\[MANAGER\] SYNCED\s+(\S+)\s+(BUY|SELL)\s+entry=(\S+)\s+sl=(\S+)\s+id=(\S+)", line)
    if m:
        pair, direction, entry, sl, tid = m.groups()
        direction_str = f"{G}LONG{RST}" if direction == "BUY" else f"{R}SHORT{RST}"
        sl_str = f"SL at {sl}" if sl != "None" else f"{Y}no stop loss{RST}"
        print(f"  {tag} 📊 Managing {BOLD}{pair}{RST} {direction_str} — entry {entry}, {sl_str}")
        key = f"{engine}_{pair}"
        portfolio[f"{engine.lower()}_trades"][pair] = {
            "direction": direction, "entry": float(entry),
            "sl": None if sl == "None" else float(sl), "id": tid
        }
        return

    # ── Trail SL update ───────────────────────────────────────
    m = re.search(r"\[TightSL\]\s+(\S+)\s+TRAIL\s+→\s+SL\s+(\S+)", line)
    if m:
        print(f"  {tag} {G}🔒 TRAILING STOP{RST} — {BOLD}{m.group(1)}{RST} stop moved to {m.group(2)}")
        return

    # ── Scale-out ─────────────────────────────────────────────
    m = re.search(r"💰 1:1 SCALE-OUT.*?(\S+USD|\S+_\S+)", line)
    if m:
        print(f"  {tag} {G}💰 PROFIT SECURED{RST} — Sold 50% of {BOLD}{m.group(1)}{RST} at 1:1 risk/reward. Stop moved to breakeven.")
        return

    # ── Hard dollar stop ──────────────────────────────────────
    m = re.search(r"HARD_DOLLAR_STOP\s+(\S+).*uPnL=(\S+)", line)
    if m:
        print(f"  {tag} {R}🛑 EMERGENCY STOP{RST} — {BOLD}{m.group(1)}{RST} hit hard limit, loss {m.group(2)}. Position closed.")
        return

    # ── Local SL hit ──────────────────────────────────────────
    m = re.search(r"LOCAL_SL_HIT\s+(\S+).*price=(\S+).*SL=(\S+)", line)
    if m:
        print(f"  {tag} {R}🛑 STOP LOSS HIT{RST} — {BOLD}{m.group(1)}{RST} price {m.group(2)} crossed SL {m.group(3)}. Closed.")
        return

    # ── Profit target ─────────────────────────────────────────
    m = re.search(r"TP_HIT\s+(\S+).*price=(\S+)", line)
    if m:
        print(f"  {tag} {G}🎯 TAKE PROFIT{RST} — {BOLD}{m.group(1)}{RST} hit target at {m.group(2)}! Full exit.")
        return

    # ── Counter-trend warning ─────────────────────────────────
    m = re.search(r"⚡ (\S+) counter-trend.*trail (\S+)", line)
    if m:
        print(f"  {tag} {Y}⚡ CAUTION{RST} — {BOLD}{m.group(1)}{RST} fighting the higher timeframe. Trail speed reduced to {m.group(2)}.")
        return

    # ── BLOCKED ───────────────────────────────────────────────
    m = re.search(r"BLOCKED\s+(\S+)\s+—\s+(.*)", line)
    if m:
        reason = m.group(2)
        readable = {
            "LONG_ONLY_MODE (Shorting disabled for Spot)": "Short signal ignored — spot market doesn't support shorts",
            "CORRELATION_GATE": "Position overlap — already have exposure in this currency",
        }
        r = readable.get(reason, reason)
        # Only some blocks are interesting
        if "LONG_ONLY" not in reason:
            print(f"  {tag} {Y}⛔ BLOCKED{RST} {BOLD}{m.group(1)}{RST}: {r}")
        return

    # ── Slots ─────────────────────────────────────────────────
    m = re.search(r"Slots open \((\d+)/(\d+)\)", line)
    if m:
        return  # handled in summary

    # ── CHOP / SNIPER mode ────────────────────────────────────
    if "CHOP MODE" in line or "SNIPER MODE" in line:
        return  # skip mode headers

    # ── CapitalRouter ─────────────────────────────────────────
    m = re.search(r"CapitalRouter.*initial.*\$(\S+).*watermark.*\$(\S+)", line)
    if m:
        print(f"  {tag} 💼 Portfolio — starting balance ${m.group(1)}, high-water mark ${m.group(2)}")
        return

    # ── Engine restarted ──────────────────────────────────────
    if "engine restarted" in line.lower() or "engine started" in line.lower():
        print(f"  {tag} {G}🚀 ENGINE STARTED{RST}")
        return

    # ── Errors ────────────────────────────────────────────────
    if "502 Server Error" in line or "Bad Gateway" in line:
        m = re.search(r"instruments/(\S+)/", line)
        pair = m.group(1) if m else "unknown"
        # Only print once, not the duplicate ERROR line
        if "OANDA candles error" in line:
            print(f"  {tag} {Y}⚠️  API hiccup{RST} — broker returned 502 for {pair} (retrying next cycle)")
        return

    if "Scale-out failed" in line:
        m = re.search(r"Scale-out failed (\S+): (.*)", line)
        if m:
            print(f"  {tag} {Y}⚠️  Scale-out failed{RST} for {m.group(1)}: {m.group(2)}")
        return


def print_summary():
    """Print a portfolio summary dashboard."""
    now = datetime.now().strftime("%H:%M:%S ET")

    oanda_count = len(portfolio["oanda_trades"])
    coinbase_count = len(portfolio["coinbase_trades"])

    print(f"\n  {BOLD}{'═' * 60}{RST}")
    print(f"  {BOLD}📊 PORTFOLIO DASHBOARD{RST}  —  {D}{now}{RST}")
    print(f"  {'─' * 60}")

    if oanda_count == 0 and coinbase_count == 0:
        print(f"  {D}  No managed trades currently tracked{RST}")
    else:
        if oanda_count > 0:
            print(f"  {C}  OANDA ({oanda_count} open):{RST}")
            for pair, info in portfolio["oanda_trades"].items():
                d = f"{G}LONG{RST}" if info["direction"] == "BUY" else f"{R}SHORT{RST}"
                sl = f"SL {info['sl']}" if info['sl'] else f"{Y}no SL{RST}"
                print(f"    • {BOLD}{pair}{RST}  {d}  entry={info['entry']}  {sl}")

        if coinbase_count > 0:
            print(f"  {C}  COINBASE ({coinbase_count} open):{RST}")
            for pair, info in portfolio["coinbase_trades"].items():
                d = f"{G}LONG{RST}" if info["direction"] == "BUY" else f"{R}SHORT{RST}"
                sl = f"SL {info['sl']}" if info['sl'] else f"{Y}no SL{RST}"
                print(f"    • {BOLD}{pair}{RST}  {d}  entry={info['entry']}  {sl}")

    # Read Coinbase active_trades.json for extra detail
    try:
        if COINBASE_TRADES.exists():
            trades = json.loads(COINBASE_TRADES.read_text())
            if trades:
                print(f"  {D}  Coinbase local positions: {len(trades)}{RST}")
    except Exception:
        pass

    print(f"  {BOLD}{'═' * 60}{RST}\n")


def main():
    print(f"\n  {BOLD}{M}╔══════════════════════════════════════════════════════════╗{RST}")
    print(f"  {BOLD}{M}║   RBOTzilla NARRATOR — Real-Time Storyteller Dashboard   ║{RST}")
    print(f"  {BOLD}{M}║   Translating engine signals into human language...      ║{RST}")
    print(f"  {BOLD}{M}╚══════════════════════════════════════════════════════════╝{RST}\n")

    # Start tailing both log files
    procs = []
    labels = []

    if OANDA_LOG.exists():
        p = subprocess.Popen(
            ["tail", "-n", "0", "-f", str(OANDA_LOG)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1
        )
        procs.append(p)
        labels.append("OANDA")
    else:
        print(f"  {Y}⚠️  OANDA log not found at {OANDA_LOG}{RST}")

    if COINBASE_LOG.exists():
        p = subprocess.Popen(
            ["tail", "-n", "0", "-f", str(COINBASE_LOG)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1
        )
        procs.append(p)
        labels.append("COINBASE")
    else:
        print(f"  {Y}⚠️  Coinbase log not found at {COINBASE_LOG}{RST}")

    if not procs:
        print(f"  {R}No log files found. Start the engines first.{RST}")
        sys.exit(1)

    import select

    try:
        while True:
            # Check each process for new output
            readable = select.select(
                [p.stdout for p in procs], [], [], 1.0
            )[0]

            for fd in readable:
                idx = [p.stdout for p in procs].index(fd)
                line = fd.readline()
                if line:
                    narrate(labels[idx], line)

            # Periodic summary
            now = time.time()
            if now - portfolio["last_summary"] > SUMMARY_INTERVAL:
                print_summary()
                portfolio["last_summary"] = now

    except KeyboardInterrupt:
        print(f"\n  {D}Narrator stopped.{RST}")
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
