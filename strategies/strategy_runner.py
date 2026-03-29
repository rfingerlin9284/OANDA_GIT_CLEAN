#!/usr/bin/env python3
"""
strategies/strategy_runner.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Bridges the scan_symbol() detector pipeline with the class-based strategy files.

Usage:
    from strategies.strategy_runner import run_class_strategies
    proposals = run_class_strategies(symbol, candles, timeframe="M15")

Returns a list of ProposedTrade objects (may be empty).
The caller (trade_engine) decides which proposals to submit via the gate.
"""

import time
from typing import List, Optional

from strategies.base import (
    BaseStrategy, StrategyContext, StrategyMetadata, ProposedTrade
)
from strategies.fib_confluence_breakout import FibConfluenceBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from strategies.trap_reversal_scalper import TrapReversalScalperStrategy

# ── Strategy registry ─────────────────────────────────────────────────────────
# Add new strategies here. Each entry is (StrategyClass, StrategyMetadata).

_STRATEGIES: List[BaseStrategy] = [

    FibConfluenceBreakoutStrategy(StrategyMetadata(
        name="Fibonacci Confluence Breakout",
        code="FIB_BREAK",
        priority="gold",
        markets=["FX"],
        base_timeframes=["M15", "M30"],
        max_hold_minutes=240,
        target_rr=2.0,
        est_win_rate=0.52,
    )),

    LiquiditySweepReversalStrategy(StrategyMetadata(
        name="Liquidity Sweep Reversal",
        code="LIQ_SWEEP",
        priority="gold",
        markets=["FX"],
        base_timeframes=["M15"],
        max_hold_minutes=180,
        target_rr=2.0,
        est_win_rate=0.54,
    )),

    TrapReversalScalperStrategy(StrategyMetadata(
        name="Trap Reversal Scalper",
        code="TRAP_REV",
        priority="silver",
        markets=["FX"],
        base_timeframes=["M15"],
        max_hold_minutes=120,
        target_rr=1.5,
        est_win_rate=0.50,
    )),

]


def run_class_strategies(
    symbol: str,
    candles: list,
    timeframe: str = "M15",
    min_confidence: float = 0.68,
    venue: str = "oanda_practice",
) -> List[ProposedTrade]:
    """
    Run all registered class-based strategies against the given candles.

    Returns: list of ProposedTrade objects that meet min_confidence.
             Empty list if no signals fire.

    This does NOT submit orders. Caller is responsible for gate + submission.
    """
    if not candles or len(candles) < 30:
        return []

    ctx = StrategyContext(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        higher_tf_context={},
        indicators={},
        venue=venue,
        now_ts=time.time(),
    )

    proposals: List[ProposedTrade] = []
    for strategy in _STRATEGIES:
        try:
            trade = strategy.decide_entry(ctx)
            if trade and trade.confidence >= min_confidence:
                proposals.append(trade)
        except Exception:
            pass  # strategy errors must not crash the engine

    # Sort by confidence descending
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    return proposals


def get_strategy_names() -> List[str]:
    """Return list of registered strategy names for logging."""
    return [s.metadata.name for s in _STRATEGIES]
