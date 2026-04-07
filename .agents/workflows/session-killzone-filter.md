---
description: Session Killzone Filter — Time-based trading windows that identify optimal execution periods and restrict trading during low-probability hours
---

# Session Killzone Filter

## Source
Synthesized from: ICT Killzone methodology + session timing research

## Overview
NOT all hours are equal. Institutional algorithms execute during specific windows, creating predictable volatility and liquidity patterns. This workflow defines WHEN to trade and WHEN to sit on hands. It acts as a TIME GATE applied before any signal from other workflows.

---

## FOREX (OANDA) — Session Killzones

All times in New York (Eastern Time). Adjust for DST.

### Primary Sessions

| Session | Time (ET) | Volatility | Best For | Priority |
|---------|-----------|-----------|----------|----------|
| **London Open** | 2:00 AM – 5:00 AM | 🔥 HIGH | Trend initiation, liquidity sweeps | ⭐ A+ |
| **NY Open** | 7:00 AM – 10:00 AM | 🔥🔥 HIGHEST | News moves, trend continuation, reversals | ⭐ A+ |
| **London/NY Overlap** | 8:00 AM – 12:00 PM | 🔥🔥 HIGHEST | Maximum volume, best R:R setups | ⭐ A+ |
| **London Close** | 11:00 AM – 1:00 PM | 🔥 MODERATE | Retracements, profit-taking | ⭐ B |
| **Asian Session** | 7:00 PM – 10:00 PM | ❄️ LOW | Range formation, bias detection | ⭐ C |

### Dead Zones (DO NOT TRADE)

| Window | Time (ET) | Why |
|--------|-----------|-----|
| Post-NY / Pre-Asia | 1:00 PM – 7:00 PM | Lowest volume, widest spreads, false signals |
| Late Asia | 10:00 PM – 2:00 AM | Transition period, low conviction |
| First 15 min of any session | Variable | Noise, fake moves, stop hunts not yet complete |

### Pair-Specific Adjustments

| Pair Type | Best Session | Avoid |
|-----------|-------------|-------|
| EUR/USD, GBP/USD, EUR/GBP | London + NY overlap | Asia |
| USD/JPY, EUR/JPY, GBP/JPY | London + Asia overlap | Dead zone |
| AUD/USD, NZD/USD | Asia (their session) + NY | London pre-open |
| USD/CAD | NY session (oil correlation) | Asia |
| Gold (XAU/USD) | London + NY | Asia (thin) |

---

## CRYPTO (COINBASE) — Session Windows

Crypto trades 24/7. There are no "closed" periods, but volume patterns still create optimal windows.

### Volume-Based Windows

| Window | Time (ET) | Volume | Best For | Priority |
|--------|-----------|--------|----------|----------|
| **US Market Hours** | 9:00 AM – 4:00 PM | 🔥🔥 HIGH | Highest volume, best execution | ⭐ A+ |
| **EU + US Overlap** | 9:00 AM – 12:00 PM | 🔥🔥 HIGHEST | Peak activity | ⭐ A+ |
| **Asian Hours** | 8:00 PM – 4:00 AM | 🔥 MODERATE | BTC/ETH moves, altcoin quiet | ⭐ B |
| **EU Hours** | 3:00 AM – 9:00 AM | 🔥 MODERATE | Moderate volume | ⭐ B |
| **US Evening** | 4:00 PM – 8:00 PM | ❄️ LOW | Thin orderbooks, avoid for scalps | ⭐ C |

### Crypto-Specific Timing Rules

| Rule | Description |
|------|------------|
| Major token unlocks | Check unlock schedules — sell pressure window for specific tokens |
| Sunday evening (ET) | Often sees large BTC moves as Asian markets open for the week |
| US market close (4 PM ET) | Crypto often correlates; watch for directional shift |
| FOMC / NFP days | Crypto correlated with equities — expect volatility spillover |

---

## Implementation as Engine Filter

### Pre-Signal Gate

```
BEFORE evaluating any signal:
    1. Check current time (ET)
    2. Check instrument type (forex pair / crypto pair)
    3. Determine session window
    4. Apply filter:

    if session == DEAD_ZONE:
        → REJECT signal (log: "outside killzone")
        → Exception: swing trades already open can be MANAGED but not ENTERED

    if session == C_PRIORITY:
        → Reduce position size by 50%
        → Require higher confidence threshold (+10%)
        → Only allow signals from swing-trade-ladder.md

    if session == B_PRIORITY:
        → Standard position size
        → Standard confidence threshold

    if session == A_PRIORITY:
        → Standard or increased position size
        → All workflows active
```

### News Event Override

| Event Type | Action | Duration |
|------------|--------|----------|
| FOMC rate decision | PAUSE all new entries 30 min before → 15 min after | ~45 min |
| NFP (Non-Farm Payrolls) | PAUSE all new entries 15 min before → 10 min after | ~25 min |
| CPI / PPI | PAUSE all new entries 10 min before → 5 min after | ~15 min |
| Central bank speech | MONITOR — reduce size 50% during | Variable |
| Crypto: SEC announcement | PAUSE crypto entries until initial volatility settles | 30 min |

---

## Automatable Parameters

| Parameter | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| A+ killzone start | 2:00 AM ET (London) / 7:00 AM ET (NY) | 9:00 AM ET |
| A+ killzone end | 12:00 PM ET (overlap end) | 12:00 PM ET |
| Dead zone start | 1:00 PM ET | 4:00 PM ET |
| Dead zone end | 7:00 PM ET | 8:00 PM ET |
| First-bar skip | 15 minutes after session open | Not applicable |
| News pause buffer | 15-30 min pre-event | 15 min pre-event |
| Weekend filter | CLOSE all positions Friday before close | Not applicable (24/7) |
| Low-priority size reduction | 50% of standard | 50% of standard |

---

## Integration with Workflow Router

```
workflow-router.md
    ↓
session-killzone-filter.md  ← THIS FILE (time gate)
    ↓
[signal workflows]  ← only receive signals that pass the time gate
    ↓
scalp-math-rules.md  ← math validation
    ↓
EXECUTE or REJECT
```

**This filter sits BETWEEN the router and the strategy workflows.** The router selects WHICH strategy based on regime. This filter determines WHETHER to execute based on TIME.
