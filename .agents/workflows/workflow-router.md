---
description: Master workflow router — Selects the correct trading workflow based on regime detection, market conditions, and instrument type
---

# Workflow Router — Strategy Selection Engine

## Purpose
This is the MASTER ROUTER that determines which trading workflow to activate based on the current market regime, instrument characteristics, and time conditions. The engine MUST check this router before executing any signal.

---

## Full Signal Processing Pipeline

```
┌──────────────────────────────────────────────────────┐
│  LAYER 1: MTF Top-Down (mtf-top-down-analysis.md)    │
│  Daily/H4 → establish directional BIAS               │
│  Only process signals matching HTF direction          │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────┐
│  LAYER 2: Session Killzone (session-killzone-filter)  │
│  Is current time in an A+/B/C/Dead zone?              │
│  Dead zone → REJECT all new entries                   │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────┐
│  LAYER 3: Regime Router (regime_detector.py)           │
│                                                       │
│  TRENDING ──→ Order Block Sniper                      │
│            ──→ 9 EMA Continuation Scalp               │
│            ──→ Prop Desk Scalps (6 setups)             │
│            ──→ Liquidity Sweep Entry                   │
│            ──→ VWAP Trend Pullback                     │
│            ──→ Swing Ladder: trim on stretch           │
│                                                       │
│  SIDEWAYS  ──→ Range Bounce (wide) / Breakout (narrow)│
│            ──→ VWAP 2-Sigma Mean Reversion             │
│            ──→ Swing Ladder: buy on dips               │
│            ──✗ DO NOT use trend indicators             │
│                                                       │
│  BREAKOUT  ──→ HOLD core, ride momentum                │
│            ──✗ DO NOT trim into genuine breakout       │
│                                                       │
│  TRIAGE    ──→ NO TRADING, wait for clarity            │
│            ──→ Reduce to floor minimum                 │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────┐
│  LAYER 4: Compound Gate (ai-edge-infrastructure.md)   │
│  ALL conditions must pass (4+ of 6 minimum)           │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────┐
│  LAYER 5: Scalp Math (scalp-math-rules.md)            │
│  Min size, trader's equation, 70%+ win rate gate      │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────┐
│  LAYER 6: Position Sizing (position-sizing-engine.md) │
│  Fixed fractional / Kelly / ATR-adjusted              │
│  Auto-reduction on loss streaks / session / conviction│
└──────────────────┬───────────────────────────────────┘
                   │
              ✅ EXECUTE
```

---

## Confidence Score Boosters

When evaluating any signal from any workflow, boost confidence when:

| Factor | Boost | Source |
|--------|-------|--------|
| Multiple timeframe agreement (H4 + H1 + M15) | +20% | Order Block Sniper |
| Volume confirming the move | +15% | All workflows |
| Sector/macro catalyst identified | +15% | Swing Trade Ladder |
| 9 EMA clearly sloping in trade direction | +10% | 9 EMA Scalp |
| Price at unmitigated Order Block | +20% | Order Block Sniper |
| Range has 3+ prior bounces at level | +10% | Range Bounce |
| Breakout momentum candle (>70% body ratio) | +15% | Range Breakout |
| Tape change / volume shift at key level | +20% | Prop Desk Scalps (Rubber Band) |
| VWAP cross with 9 EMA + volume | +15% | Prop Desk Scalps (Fashionably Late) |
| Failed setup → new setup forms (information) | +10% | Prop Desk Scalps (Big Dog) |
| Scalp math validation passed (70%+ win rate) | +10% | Scalp Math Rules |
| Liquidity sweep + displacement + MSS confirmed | +25% | Liquidity Sweep Entry |
| VWAP + Anchored VWAP confluence zone | +20% | VWAP Strategy Suite |
| Inside A+ killzone (London/NY overlap) | +15% | Session Killzone Filter |
| HTF + MTF + LTF all aligned in same direction | +25% | MTF Top-Down Analysis |
| Kelly-validated edge (50+ trade sample) | +10% | Position Sizing Engine |

---

## Venue-Specific Adjustments

| Adjustment | OANDA (Forex) | COINBASE (Crypto) |
|-----------|--------------|-------------------|
| Session filter | London/NY preferred, skip Asia for most majors | 24/7, no session filter |
| Pip sizing | Fixed (0.0001/0.01) | Price-proportional (0.1% of price) |
| Weekend holding | AVOID (gaps kill setups) | OK (24/7 market) |
| Spread impact | Check during high-vol sessions | Check orderbook depth |
| Volatility baseline | ATR 14-period | ATR 14-period |
| MA stretch thresholds | 0.5% / 1% / 2% | 3% / 5% / 10% |

---

## Risk Stack (Applied Across All Workflows)

| Guard | Rule |
|-------|------|
| Max risk per trade | 1-2% of account equity |
| Max concurrent trades | 2-3 |
| Daily stop | 5% of account equity |
| Correlation gate | No more than 2 same-direction positions on correlated pairs |
| Floor position | Never sell below core floor (swing-trade-ladder) |
| Breakeven lock | Move SL to entry after 1R profit |
| No revenge trading | After 2 consecutive losses, pause 1 hour |

---

## Workflow Files Reference (13 Total)

### Signal Workflows (Entry Strategies)
| File | Strategy | Best Regime |
|------|----------|-------------|
| `order-block-sniper.md` | Institutional OB entry with MTF refinement | TRENDING |
| `ema9-continuation-scalp.md` | Pullback-to-9EMA with stop game | TRENDING |
| `prop-desk-scalps.md` | 6 prop firm setups (VWAP, rubber band, puppy dog, etc.) | TRENDING |
| `liquidity-sweep-entry.md` | ICT liquidity sweep + displacement + FVG entry | TRENDING |
| `vwap-strategy-suite.md` | VWAP pullback, 2σ mean reversion, anchored VWAP | ALL |
| `range-bounce-breakout.md` | Key level bounce + compression breakout | SIDEWAYS |

### Position Management
| File | Strategy | Best Regime |
|------|----------|-------------|
| `swing-trade-ladder.md` | MA stretch ladder + core floor position | ALL (overlay) |

### Filters & Validation
| File | Strategy | Applies To |
|------|----------|------------|
| `mtf-top-down-analysis.md` | HTF→MTF→LTF directional bias gating | ALL signals |
| `session-killzone-filter.md` | Time-based execution windows + dead zones | ALL signals |
| `scalp-math-rules.md` | Min sizes, trader's equation, 70%+ win rate | Scalp signals |
| `position-sizing-engine.md` | Kelly, ATR-adjusted, conviction-based sizing | ALL trades |

### Infrastructure
| File | Strategy | Applies To |
|------|----------|------------|
| `ai-edge-infrastructure.md` | Compound gate, self-analytics, 2-bar trail, autopsy | ALL |
| `workflow-router.md` | THIS FILE — master pipeline | N/A |
