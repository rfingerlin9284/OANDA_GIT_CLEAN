#!/usr/bin/env python3
"""
scripts/narration_buddy.py — RBOTZILLA_OANDA_CLEAN
Your Trading Buddy: A plain-English, colleague-style narration feed.
Reads engine_continuous.out and translates each line into human-friendly updates.
Think: professional colleague who's watching your trades with you over Slack.

Run: .venv/bin/python scripts/narration_buddy.py
"""
import os
import re
import sys
import time
from datetime import datetime

LOG = "/home/rfing/RBOTZILLA_OANDA_CLEAN/logs/engine_continuous.out"

# ── ANSI Colors ──────────────────────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
RESET  = '\033[0m'
BLUE   = '\033[94m'
MAG    = '\033[95m'

def now_stamp():
    return datetime.now().strftime("%I:%M:%S %p")

def translate(line: str) -> str | None:
    """Translate a raw engine log line into plain English."""
    line = line.strip()
    if not line:
        return None

    ts = now_stamp()

    # ── Engine lifecycle ──────────────────────────────────────────────────
    if "RBOTZILLA OANDA CLEAN" in line and "PRACTICE ENGINE" in line:
        return f"\n{BOLD}{CYAN}{'═'*60}{RESET}\n  {ts}  {GREEN}🚀 Hey! The trading engine just fired up. Let's get to work.{RESET}\n{BOLD}{CYAN}{'═'*60}{RESET}"

    if "Engine Time" in line:
        t = line.split(":", 1)[-1].strip() if ":" in line else line
        return f"  {DIM}⏰ Engine clock: {t}{RESET}"

    if "Clock Sync" in line and "SYNCED" in line:
        return f"  {GREEN}✅ Broker clock is synced — we're good to go.{RESET}"
    if "Clock Sync" in line and "DRIFT" in line:
        return f"  {YELLOW}⚠️  Heads up — clock drift detected. Watching it.{RESET}"

    if "Balance" in line and "$" in line:
        bal = re.search(r'\$[\d,.]+', line)
        return f"  {BOLD}💰 Account balance: {bal.group() if bal else 'N/A'}{RESET}"

    if "Broker Trades" in line:
        n = re.search(r':\s*(\d+)', line)
        count = n.group(1) if n else "?"
        return f"  📊 Currently holding {BOLD}{count}{RESET} open trade(s)"

    if "OCO Enforced" in line and "YES" in line:
        return f"  {GREEN}🛡️  Safety harness on — every trade has a stop-loss and take-profit.{RESET}"

    if "TRADE MANAGER ACTIVATED" in line:
        return f"\n  {ts}  {GREEN}👀 Trade Manager is awake — monitoring your positions now.{RESET}"

    if "CapitalRouter RESTORED" in line:
        wm = re.search(r'watermark=\$([\d,.]+)', line)
        live = re.search(r'live=\$([\d,.]+)', line)
        return (f"  {ts}  {BLUE}💼 Capital tracker restored."
                f" High watermark: ${wm.group(1) if wm else '?'}"
                f" | Live equity: ${live.group(1) if live else '?'}{RESET}")

    if "Min Confidence" in line:
        pct = re.search(r'(\d+)%', line)
        return f"  🎯 Only taking trades with {BOLD}{pct.group(1) if pct else '80'}%+{RESET} confidence."

    if "Max Positions" in line:
        n = re.search(r':\s*(\d+)', line)
        return f"  📐 Max simultaneous trades: {BOLD}{n.group(1) if n else '6'}{RESET}"

    if "Pairs Scanning" in line:
        n = re.search(r':\s*(\d+)', line)
        return f"  🔎 Scanning {BOLD}{n.group(1) if n else '10'}{RESET} currency pairs"

    # ── PRE-MARKET SCAN ───────────────────────────────────────────────────
    if "PRE-MARKET SCAN" in line:
        return f"\n  {ts}  {MAG}📋 Pre-market scan starting — let's see what's cooking today...{RESET}"

    if re.match(r'\s*#\d+\s+\w+', line) and ("ACTIVE" in line or "conf=" in line):
        parts = line.strip().split()
        pair = parts[1] if len(parts) > 1 else "?"
        direction = parts[2] if len(parts) > 2 else ""
        conf_match = re.search(r'conf=([\d.]+)%', line)
        rr_match = re.search(r'RR=([\d.]+)', line)
        conf = conf_match.group(1) if conf_match else "?"
        rr = rr_match.group(1) if rr_match else "?"
        emoji = "📈" if direction == "BUY" else "📉" if direction == "SELL" else "📊"
        return f"  {emoji} {BOLD}{pair}{RESET} looking like a {direction} — {conf}% confident, {rr}:1 reward"

    if "TOP PICK" in line:
        parts = line.strip().split(":")
        pick = parts[-1].strip() if parts else line
        return f"\n  {ts}  {GREEN}{BOLD}⭐ Best opportunity right now: {pick}{RESET}"

    if "SUMMARY:" in line:
        match = re.search(r'(\d+)\s+active', line)
        n = match.group(1) if match else "?"
        return f"  {ts}  {CYAN}📊 Found {BOLD}{n}{RESET}{CYAN} potential setups on the watchlist.{RESET}"

    if "PLAYBOOK" in line:
        match = re.search(r'(\d+)\s+active', line)
        n = match.group(1) if match else "?"
        return f"  {ts}  {BLUE}📒 Playbook loaded — {n} setups queued for this session.{RESET}"

    # ── SCANNING ──────────────────────────────────────────────────────────
    if "[SCAN]" in line:
        pair_match = re.search(r'\[SCAN\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        # Check for signals found
        signals = []
        for code, name in [("MOM", "momentum"), ("REV", "reversal"),
                           ("MR", "mean-reversion"), ("SC", "scalp")]:
            patt = f"{code}=(\\d+)%✓"
            m = re.search(patt, line)
            if m:
                signals.append(f"{name} {m.group(1)}%")
        if signals:
            return f"  {ts}  {GREEN}🎯 {BOLD}{pair}{RESET}{GREEN} — Signal! {', '.join(signals)}{RESET}"
        # No signals — skip (too noisy)
        return None

    # ── GATES & FILTERS (the interesting stuff) ───────────────────────────
    if "[MTF_SNIPER]" in line and "blocked" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        reason = "higher timeframe disagrees"
        if "H4 BEARISH" in line: reason = "4-hour chart is bearish"
        elif "H4 BULLISH" in line: reason = "4-hour chart is bullish"
        elif "DAILY BEARISH" in line: reason = "daily chart is bearish"
        elif "DAILY BULLISH" in line: reason = "daily chart is bullish"
        return f"  {ts}  {YELLOW}🚫 {pair} — Nope. Blocked because {reason}. Smart move.{RESET}"

    if "[EMA200]" in line and "filtered" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        bias_match = re.search(r'bias \((\w+)\)', line)
        bias = bias_match.group(1) if bias_match else "?"
        return f"  {ts}  {YELLOW}📏 {pair} — 200 EMA says the trend is {bias}. Filtered out conflicting signals.{RESET}"

    if "[VOL_GATE]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {YELLOW}📊 {pair} — Volume too low. Not enough market conviction. Skipping.{RESET}"

    if "[RSI_GATE]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {YELLOW}⚡ {pair} — RSI says this pair is overextended. Let's wait for a better entry.{RESET}"

    if "[DXY_GATE]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        bias = "USD is strong" if "USD_STRONG" in line else "USD is weak" if "USD_WEAK" in line else "USD bias"
        return f"  {ts}  {YELLOW}💵 {pair} — DXY filter kicked in ({bias}). Would be fighting the dollar.{RESET}"

    if "[MACD_DIV]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {YELLOW}📉 {pair} — MACD divergence detected. Momentum is fading — staying out.{RESET}"

    if "[CONFLUENCE]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {GREEN}🎯 {pair} — Multiple strategies agree! Confidence boosted. Nice setup.{RESET}"

    if "[EXHAUST]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {YELLOW}😴 {pair} — This move is exhausted. We'd be showing up late to the party.{RESET}"

    # ── TRADE EXECUTION ───────────────────────────────────────────────────
    if "Qualified:" in line:
        match = re.search(r'(\d+)\s+signal', line)
        n = match.group(1) if match else "?"
        return f"\n  {ts}  {CYAN}📋 {BOLD}{n}{RESET}{CYAN} signal(s) passed all filters. Evaluating...{RESET}"

    if "QUEUED" in line:
        pair_match = re.search(r'(\w+_\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        direction = "BUY" if "BUY" in line else "SELL" if "SELL" in line else "?"
        return f"  {ts}  {GREEN}📥 {BOLD}{pair}{RESET}{GREEN} {direction} queued up! Getting ready to pull the trigger.{RESET}"

    if "Quick entry opportunity" in line:
        return f"  {ts}  {GREEN}💬 Found a quick opportunity — confidence is high.{RESET}"

    if "placing now" in line.lower() or "Placing now" in line:
        return f"  {ts}  {GREEN}{BOLD}�� Placing trade(s) now!{RESET}"

    # ── MANAGER UPDATES ───────────────────────────────────────────────────
    if "[MANAGER] SYNCED" in line:
        parts = line.split()
        pair = ""
        direction = ""
        entry = ""
        for i, p in enumerate(parts):
            if p == "SYNCED":
                if i + 1 < len(parts): pair = parts[i + 1]
                if i + 2 < len(parts): direction = parts[i + 2]
            if p.startswith("entry="):
                entry = p.split("=")[1]
        return f"  {ts}  {BLUE}🔗 Now tracking {BOLD}{pair}{RESET}{BLUE} ({direction} from {entry}){RESET}"

    if "[MANAGER] CLOSED" in line:
        pair_match = re.search(r'CLOSED\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {MAG}📋 {pair} just closed. Checking the results...{RESET}"

    if "[MANAGER]" in line and "HARD_DOLLAR_STOP" in line:
        pair_match = re.search(r'HARD_DOLLAR_STOP\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {RED}{BOLD}🛑 {pair} hit the max loss limit. Closed it. No emotions — that's what risk management is for.{RESET}"

    if "[MANAGER]" in line and "GREEN_LOCK" in line:
        pair_match = re.search(r'GREEN_LOCK\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        return f"  {ts}  {GREEN}🔒 {pair} — Stop-loss moved to protect profits. We're playing with house money now! 🎰{RESET}"

    if "[TightSL]" in line:
        pair_match = re.search(r'\]\s+(\w+)', line)
        pair = pair_match.group(1) if pair_match else "?"
        if "STEP1" in line:
            return f"  {ts}  {GREEN}📈 {pair} — First target hit! Locking in some protection.{RESET}"
        if "STEP2" in line or "breakeven" in line.lower():
            return f"  {ts}  {GREEN}🔐 {pair} — Moved to breakeven. Risk-free trade now. Love to see it.{RESET}"
        if "TRAIL" in line:
            return f"  {ts}  {GREEN}🏃 {pair} — Trailing stop tightened. Locking in more profit as we ride this wave!{RESET}"

    if "[PAIR_STATS]" in line:
        return f"  {ts}  {CYAN}📊 {line.strip().replace('[PAIR_STATS]', 'Stats:')}{RESET}"

    # ── CHOP MODE / WAITING ───────────────────────────────────────────────
    if "CHOP MODE" in line:
        slots_match = re.search(r'\((\d+)/(\d+)\)', line)
        if slots_match:
            used, total = slots_match.group(1), slots_match.group(2)
            return f"  {ts}  {DIM}⏳ Waiting for the next opportunity... ({used} of {total} slots open). Patience pays.{RESET}"
        return f"  {ts}  {DIM}⏳ Scanning... patience is the game.{RESET}"

    if "rescanning" in line.lower():
        return None  # Too noisy

    # ── CIRCUIT BREAKER ───────────────────────────────────────────────────
    if "CIRCUIT_BREAKER" in line or "circuit breaker" in line.lower():
        if "LOSS" in line.upper():
            return f"  {ts}  {RED}{BOLD}🔴 Daily loss limit reached. Engine is pausing new entries. Living to fight another day.{RESET}"
        if "GAIN" in line.upper():
            return f"  {ts}  {GREEN}{BOLD}🟢 Daily profit target hit! Taking the rest of the day off. Well played! 🎉{RESET}"

    # ── Separator lines ───────────────────────────────────────────────────
    if line.startswith("═") or line.startswith("─") or line.startswith("="):
        return None  # Skip decorative lines

    if "started" in line.lower() and "engine" in line.lower():
        return None  # Already handled by PRACTICE ENGINE

    # ── Catch-all for unrecognized but non-empty lines ────────────────────
    # Only show if it's not a decorative/empty line
    if len(line) > 3 and not line.startswith("  ") and not line.startswith("OK:"):
        return f"  {ts}  {DIM}ℹ️  {line}{RESET}"

    return None


def main() -> None:
    print()
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"  {BOLD}RBOTZILLA — Your Trading Buddy{RESET}")
    print(f"  {DIM}Plain English translation of what the engine is doing.{RESET}")
    print(f"  {DIM}Think of me as your colleague watching the screens with you.{RESET}")
    print(f"  {DIM}Press Ctrl+C to stop.{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print()
    sys.stdout.flush()

    while not os.path.exists(LOG):
        print(f"  {YELLOW}⏳ Waiting for engine log to appear...{RESET}")
        time.sleep(2)

    with open(LOG, "r") as fh:
        # Start from beginning to catch the startup banner
        fh.seek(0, 2)  # Jump to end — only new lines
        while True:
            raw = fh.readline()
            if not raw:
                time.sleep(0.3)
                continue
            try:
                msg = translate(raw)
                if msg:
                    print(msg, flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
