#!/usr/bin/env python3
"""
scripts/system_status.py — RBOTZILLA_OANDA_CLEAN
Run Task: "📊 System Status — All Features ON/OFF"

Reads .env, engine code, and broker state to produce a full audit of:
  1. DEFAULT ON — always-active features when engine starts
  2. DEFAULT OFF / DISABLED — coded but intentionally disabled
  3. All configurable env vars with current values
  4. Interactive menu to toggle features or exit

Safe to run while engine is running — READ-ONLY by default.
Only writes to .env if user explicitly chooses to toggle a feature.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env"

# ── ANSI Colors ──────────────────────────────────────────────────────────────
GREEN   = '\033[92m'
RED     = '\033[91m'
YELLOW  = '\033[93m'
CYAN    = '\033[96m'
BOLD    = '\033[1m'
DIM     = '\033[2m'
RESET   = '\033[0m'
BG_GREEN = '\033[42m\033[30m'
BG_RED   = '\033[41m\033[97m'

SEP = "═" * 65


def load_env() -> dict:
    """Load .env file into a dict (does not export to os.environ)."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip("'\"")
    return env


def engine_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "trade_engine.py"],
            capture_output=True, text=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def print_header(env: dict):
    from util.narration_logger import dual_timestamp
    ts = dual_timestamp()

    print(f"\n{SEP}")
    print(f"  {BOLD}📊  RBOTZILLA SYSTEM STATUS DASHBOARD{RESET}")
    print(SEP)
    print(f"  Time     : {ts}")
    print(f"  Timezone : US Eastern (NYC/NYSE)")

    running = engine_running()
    status = f"{BG_GREEN} RUNNING {RESET}" if running else f"{BG_RED} STOPPED {RESET}"
    print(f"  Engine   : {status}")

    mode = env.get("OANDA_ENVIRONMENT", "practice")
    print(f"  Mode     : {BOLD}{mode.upper()}{RESET}")
    print(SEP)


def print_section(title: str):
    print(f"\n  {BOLD}{CYAN}── {title} ──{RESET}")


def on_off(val: str, invert=False) -> str:
    """Green ✅ ON or Red ❌ OFF badge."""
    is_on = val.lower() in ("true", "1", "yes", "on")
    if invert:
        is_on = not is_on
    if is_on:
        return f"{GREEN}✅ ON{RESET}"
    else:
        return f"{RED}❌ OFF{RESET}"


def print_default_on(env: dict):
    """Features that are ALWAYS ON when engine is started."""
    print_section("DEFAULT ON — Active When Engine Runs")

    features = [
        ("OCO Enforcement (SL+TP mandatory)",      "RBOT_REQUIRE_OCO",    env.get("RBOT_REQUIRE_OCO", "1")),
        ("Practice-Only Lock",                      "RBOT_PRACTICE_ONLY",  env.get("RBOT_PRACTICE_ONLY", "1")),
        ("Regime Detection (Bull/Bear/Sideways)",   "—",                   "1"),
        ("10-Detector Signal Voting (scan_symbol)", "—",                   "1"),
        ("Session Bias (Tokyo/London/NY/Overlap)",  "—",                   "1"),
        ("Broker Tradability Gate (spread/stale)",  "—",                   "1"),
        ("Margin Gate (20% free minimum)",          "—",                   "1"),
        ("Correlation Gate (same-base blocking)",   "—",                   "1"),
        ("Pair Cooldown (re-entry timer)",          "RBOT_PAIR_REENTRY_COOLDOWN_MINUTES", env.get("RBOT_PAIR_REENTRY_COOLDOWN_MINUTES", "60")),
        ("Green-Lock SL (profit protection)",       "RBOT_GREEN_LOCK_PIPS", env.get("RBOT_GREEN_LOCK_PIPS", "5.0")),
        ("Hard Dollar Stop ($max loss/trade)",      "RBOT_MAX_LOSS_USD_PER_TRADE", f"${env.get('RBOT_MAX_LOSS_USD_PER_TRADE', '45')}"),
        ("Stagnation SL Tighten",                   "RBOT_STAGNATION_CYCLES", f"{env.get('RBOT_STAGNATION_CYCLES', '5')} cycles"),
        ("RBZ Trailing Stop Logic",                 "—",                   "1"),
        ("Charter R:R Enforcement (≥3.26:1)",       "RBOT_CHARTER_MIN_RR", f"{env.get('RBOT_CHARTER_MIN_RR', '3.26')}:1"),
        ("Charter Min Notional ($15k+)",            "RBOT_CHARTER_MIN_NOTIONAL_USD", f"${env.get('RBOT_CHARTER_MIN_NOTIONAL_USD', '15000')}"),
        ("Capital Router (eviction + realloc)",     "—",                   "1"),
        ("Compounding Position Sizing",             "RBOT_BASE_UNITS",     f"{env.get('RBOT_BASE_UNITS', '14000')} units"),
        ("MTF Sniper Gate (H4 alignment)",          "—",                   "1"),
        ("Profit Extraction Mode",                  "RBOT_PROFIT_EXTRACTION_MODE", env.get("RBOT_PROFIT_EXTRACTION_MODE", "1")),
        ("Narration Logging (narration.jsonl)",     "—",                   "1"),
        ("Time Sync (broker clock drift check)",    "—",                   "1"),
    ]

    for name, var, val in features:
        if var == "—":
            status = f"{GREEN}✅ ON{RESET}  {DIM}(hardcoded){RESET}"
        elif val.startswith("$") or val.endswith("units") or val.endswith("cycles") or ":1" in val:
            status = f"{GREEN}✅ ON{RESET}  = {YELLOW}{val}{RESET}"
        else:
            status = on_off(val)
        print(f"    {status}  {name}")


def print_default_off(env: dict):
    """Features that are coded but DEFAULT OFF or intentionally disabled."""
    print_section("DEFAULT OFF — Coded But Disabled")

    features = [
        ("ATTACH_ONLY (read-only mode)",            "ATTACH_ONLY",         env.get("ATTACH_ONLY", "false")),
        ("DISABLE_NEW_ENTRIES (no new trades)",      "DISABLE_NEW_ENTRIES", env.get("DISABLE_NEW_ENTRIES", "false")),
        ("Live Trading (real money)",                "RBOT_ALLOW_LIVE",     env.get("RBOT_ALLOW_LIVE", "0")),
        ("Native OANDA Trailing Stop on Fill",       "—disabled—",         "false"),
        ("ML Regime Gate (ml_learning.regime_detector)", "—missing—",      "false"),
        ("Profit Target Close (% of TP)",            "RBOT_PROFIT_TARGET_PCT", env.get("RBOT_PROFIT_TARGET_PCT", "75")),
    ]

    for name, var, val in features:
        if var in ("—disabled—", "—missing—"):
            status = f"{RED}❌ OFF{RESET}  {DIM}(code disabled){RESET}"
        elif var == "RBOT_PROFIT_TARGET_PCT":
            pct = float(val) if val else 75
            if pct == 0:
                status = f"{RED}❌ OFF{RESET}  = {YELLOW}0 (disabled){RESET}"
            else:
                pct_display = pct if pct <= 1.0 else pct / 100
                status = f"{GREEN}✅ ON{RESET}   = {YELLOW}{pct_display:.0%}{RESET}"
        else:
            status = on_off(val)
        print(f"    {status}  {name}")


def print_toggleable(env: dict):
    """Features the user can toggle via .env."""
    print_section("CONFIGURABLE — Current .env Values")

    configs = [
        ("Signal Confidence Gate",    "RBOT_MIN_SIGNAL_CONFIDENCE", "0.75"),
        ("Min Votes Required",        "RBOT_MIN_VOTES",             "3"),
        ("Max Open Positions",        "RBOT_MAX_POSITIONS",         "12"),
        ("Max New Trades/Cycle",      "RBOT_MAX_NEW_TRADES_PER_CYCLE", "4"),
        ("Scan Interval (fast)",      "RBOT_SCAN_FAST_SECONDS",    "60"),
        ("Scan Interval (slow)",      "RBOT_SCAN_SLOW_SECONDS",    "300"),
        ("SL Pips (fixed override)",  "RBOT_SL_PIPS",              "0"),
        ("TP Pips (fixed override)",  "RBOT_TP_PIPS",              "0"),
        ("Base Position Size",        "RBOT_BASE_UNITS",            "14000"),
        ("Hard Stop $/trade",         "RBOT_MAX_LOSS_USD_PER_TRADE","45"),
        ("Max Spread (pips)",         "RBOT_MAX_SPREAD_PIPS",       "8.0"),
        ("Stale Quote (seconds)",     "RBOT_MAX_STALE_QUOTE_SECONDS","120"),
        ("Green Lock (pips)",         "RBOT_GREEN_LOCK_PIPS",       "5.0"),
        ("Green Lock Min Profit",     "RBOT_GREEN_LOCK_MIN_PROFIT_PIPS", "5.0"),
        ("Stagnation Cycles",        "RBOT_STAGNATION_CYCLES",     "5"),
        ("Stagnation Pip Threshold",  "RBOT_STAGNATION_PIP_THRESHOLD", "2.0"),
        ("Stagnation Tighten Pips",   "RBOT_STAGNATION_TIGHTEN_PIPS", "2.0"),
        ("Pair Re-entry Cooldown",    "RBOT_PAIR_REENTRY_COOLDOWN_MINUTES", "60"),
        ("Profit Target %",          "RBOT_PROFIT_TARGET_PCT",     "75"),
        ("Candle Count",             "RBOT_CANDLE_COUNT",           "250"),
        ("Candle Granularity",       "RBOT_CANDLE_GRANULARITY",    "M15"),
        ("Chop Target R:R",          "RBOT_CHOP_TARGET_RR",        "1.5"),
        ("Hedge Engine",             "RBOT_HEDGE_ENABLED",          "false"),
    ]

    for label, key, default in configs:
        current = env.get(key, default)
        is_default = current == default
        marker = f"{DIM}(default){RESET}" if is_default else f"{YELLOW}(custom){RESET}"
        print(f"    {key:42s} = {BOLD}{current}{RESET}  {marker}   {DIM}# {label}{RESET}")


def print_active_scripts():
    """List scripts that are part of the operational system."""
    print_section("OPERATIONAL SCRIPTS")

    scripts = [
        ("scripts/start.sh",          "Start the engine (nohup background)"),
        ("scripts/stop.sh",           "Kill the engine process"),
        ("scripts/restart.sh",        "Stop + Start"),
        ("scripts/health_check.sh",   "Process + narration + broker status"),
        ("scripts/human_tail.py",     "Plain-English narration feed (EST/EDT + UTC)"),
        ("scripts/reassess_trades.py","Force TradeManager cycle on all open trades"),
        ("scripts/system_status.py",  "This dashboard"),
    ]

    for path, desc in scripts:
        full = REPO / path
        exists = full.exists()
        badge = f"{GREEN}✅{RESET}" if exists else f"{RED}❌ MISSING{RESET}"
        print(f"    {badge}  {path:35s}  {DIM}{desc}{RESET}")


def interactive_menu():
    """Post-report menu — READ-ONLY, no changes possible."""
    print(f"\n{SEP}")
    print(f"  {BOLD}OPTIONS{RESET}")
    print(SEP)
    print(f"    {BOLD}1{RESET}  Exit")
    print(f"    {BOLD}2{RESET}  Refresh — re-run this status report")
    print()
    print(f"  {DIM}This dashboard is READ-ONLY. To change configs,{RESET}")
    print(f"  {DIM}edit .env manually and restart the engine.{RESET}")
    print()

    try:
        choice = input(f"  {CYAN}Choose [1-2]: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    return choice


def main():
    while True:
        env = load_env()
        print_header(env)
        print_default_on(env)
        print_default_off(env)
        print_toggleable(env)
        print_active_scripts()

        choice = interactive_menu()

        if choice == "2":
            continue  # re-run
        else:
            print(f"\n  {DIM}Exiting — no changes made.{RESET}\n")
            break


if __name__ == "__main__":
    main()

