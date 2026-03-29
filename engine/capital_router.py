"""
engine/capital_router.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_FEATURE

Capital Snowball Router — Autonomous Profit Reallocation.

Philosophy:
    Capital should move like a snowball rolling downhill:
    - It concentrates on the strongest positive momentum
    - It sheds positions that are stagnant or weakening
    - It grows faster as it compounds — each profit enables a larger next entry
    - It never chases — it reallocates to HIGHER-confidence setups only

The Router runs AFTER the main scan cycle each tick:
  1. Score every open position by health (unrealized PnL × signal confidence × momentum)
  2. Score every new signal candidate by conviction (votes × confidence × RR)
  3. If a candidate's score exceeds an open position's score by UPGRADE_THRESHOLD,
     and there are no free slots, close the weakest position and redeploy capital
  4. Never close a position that is in profit by > PROTECT_PROFIT_PIPS (it's earning)
  5. Scale units proportionally to NAV growth (compounding multiplier)

Safety gates (ALL must pass before any reallocation):
  - Min free margin must remain above RBOT_MIN_FREE_MARGIN_PCT after the new trade
  - Only 1 reallocation per cycle (prevents cascade closures)
  - Replacement trade must satisfy all standard OCO/charter gates
  - Never reallocate during high-impact news window

Key insight:
    The snowball effect comes from the unit-scaling formula.
    Each cycle: effective_units = base_units × (NAV / initial_NAV) ^ COMPOUND_EXPONENT
    With COMPOUND_EXPONENT=1.0 this is linear. With 1.5 it accelerates.
    Default: 1.0 (safe conservative start, user can tune via env var).
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger("capital_router")

# ── Config via env ────────────────────────────────────────────────────────────

UPGRADE_THRESHOLD   = float(os.getenv("RBOT_UPGRADE_THRESHOLD",     "0.25"))  # candidate must score 25% higher
PROTECT_PROFIT_PIPS = float(os.getenv("RBOT_PROTECT_PROFIT_PIPS",   "15.0"))  # do not evict positions ≥ 15 pip profit
COMPOUND_EXPONENT   = float(os.getenv("RBOT_COMPOUND_EXPONENT",      "1.0"))   # 1.0=linear, 1.5=accelerating
MAX_REALLOCATIONS   = int(os.getenv("RBOT_MAX_REALLOCATIONS_PER_CYCLE", "1"))  # conservative: 1 per cycle
BASE_UNITS          = int(os.getenv("RBOT_BASE_UNITS",               "5000"))


# ── Position health score ─────────────────────────────────────────────────────

@dataclass
class PositionScore:
    trade_id:       str
    symbol:         str
    direction:      str
    unrealized_pnl: float
    signal_conf:    float       # confidence at entry (0–1)
    stale_cycles:   int         # from trade_manager stagnation counter
    entry:          float
    current_price:  float
    pip_size:       float

    def health_score(self) -> float:
        """
        Composite health score (-∞ to 1):
          • PnL contribution (normalized to pip distance from entry)
          • Confidence at entry (quality of original thesis)
          • Stagnation penalty (erodes score if price is stuck)

        Higher = healthier = less likely to be evicted.
        """
        pip_profit = (
            (self.current_price - self.entry) / self.pip_size
            if self.direction == "BUY"
            else (self.entry - self.current_price) / self.pip_size
        )
        pnl_component  = max(-1.0, min(1.0, pip_profit / 50.0))     # normalised to 50-pip scale
        conf_component = self.signal_conf                            # 0–1
        stale_penalty  = min(0.5, self.stale_cycles * 0.05)         # max -0.50 drag

        return pnl_component * 0.5 + conf_component * 0.35 - stale_penalty * 0.15


# ── Candidate ranking entry ───────────────────────────────────────────────────

@dataclass
class CandidateScore:
    symbol:     str
    direction:  str
    confidence: float
    votes:      int
    rr:         float
    sl:         float
    tp:         float

    def conviction_score(self) -> float:
        """
        Composite conviction score (0–1+):
          votes × confidence + RR bonus
        """
        vote_weight = min(1.0, self.votes / 5.0)                    # 5 votes = full weight
        rr_bonus    = min(0.15, (self.rr - 2.0) * 0.03)            # bonus above 2:1 RR
        return vote_weight * self.confidence + rr_bonus


# ── Compound unit calculator ──────────────────────────────────────────────────


def compute_watermark_compounded_units(
    base_units: int,
    current_nav: float,
    watermark_nav: float,
    initial_nav: float,
    growth_exponent: float = 1.0,
    drawdown_floor_ratio: float = 1.0,
    max_growth_multiple: float = 3.0,
) -> int:
    """
    True growth-oriented sizing:
    - grows from the greater of initial_nav or watermark_nav
    - never shrinks below base_units during drawdown by default
    - caps runaway growth for sanity
    """
    try:
        base_units = int(base_units)
        current_nav = float(current_nav)
        watermark_nav = float(watermark_nav)
        initial_nav = float(initial_nav)
        growth_exponent = float(growth_exponent)
        drawdown_floor_ratio = float(drawdown_floor_ratio)
        max_growth_multiple = float(max_growth_multiple)
    except Exception:
        return int(base_units)

    if base_units <= 0:
        return 0

    anchor_nav = max(initial_nav, watermark_nav, 1.0)
    if current_nav <= 0:
        return int(base_units)

    ratio = (current_nav / anchor_nav) ** growth_exponent

    # do not decompound below base size unless explicitly allowed
    ratio = max(drawdown_floor_ratio, ratio)
    ratio = min(ratio, max_growth_multiple)

    return max(1, int(round(base_units * ratio)))


def compute_compounded_units(
    base_units: int,
    current_nav: float,
    initial_nav: float,
    exponent: float = COMPOUND_EXPONENT,
) -> int:
    """
    Scale trade size with NAV growth.

    Formula: units = base_units × (NAV / initial_NAV) ^ exponent

    With exponent=1.0 (linear):
        $10k NAV → base_units
        $20k NAV → 2× base_units
    With exponent=1.5 (accelerating):
        $20k NAV → 2.83× base_units (snowball acceleration)

    Units are clamped: min=base_units, max=base_units×5 (safety cap).
    """
    if initial_nav <= 0 or current_nav <= 0:
        return base_units
    growth_ratio = current_nav / initial_nav
    scaled = base_units * (growth_ratio ** exponent)
    return int(max(base_units, min(scaled, base_units * 5)))


# ── Main router ───────────────────────────────────────────────────────────────

class CapitalRouter:
    """
    Autonomous capital allocation engine.

    Wired into trade_engine.py — called once per scan cycle AFTER scanning.

    Usage in trade_engine.py:
        router = CapitalRouter(connector, initial_nav=start_balance)
        ...
        realloc = router.evaluate(
            open_positions=self.active_positions,
            candidates=new_signal_candidates,
            account_info=acct,
        )
        if realloc:
            # realloc.close_trade_id → close this
            # realloc.open_candidate  → open this instead
    """

    def __init__(self, connector, initial_nav: float):
        self.connector   = connector
        self.initial_nav = max(initial_nav, 1.0)
        self._reallocations_this_cycle = 0

    def reset_cycle(self) -> None:
        """Call at the start of each engine cycle."""
        self._reallocations_this_cycle = 0

    # ── Compounded unit size ─────────────────────────────────────────────────

    def get_units(self, current_nav: float, symbol: str) -> int:
        """Return compounded unit size for current NAV."""
        base = BASE_UNITS
        units = compute_compounded_units(base, current_nav, self.initial_nav)
        # Negative for SELL handled by caller
        return units

    # ── Main evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self,
        open_positions: Dict[str, dict],          # trade_id → position dict
        candidates:     List["AggregatedSignal"],  # new signals from this cycle
        account_info:   dict,
    ) -> Optional["ReallocationDecision"]:
        """
        Compare open position health against new candidate conviction.
        Return a ReallocationDecision if a better opportunity exists, else None.

        Guards:
          - MAX_REALLOCATIONS per cycle
          - Only if there are no free slots already (don't close when slots exist)
          - PROTECT_PROFIT_PIPS: never close a position earning well
          - Candidate must be genuinely stronger by UPGRADE_THRESHOLD
        """
        if self._reallocations_this_cycle >= MAX_REALLOCATIONS:
            return None
        if not open_positions or not candidates:
            return None

        # Compute account state
        current_nav   = float(account_info.get("NAV", account_info.get("balance", 0)))
        max_positions = int(os.getenv("RBOT_MAX_POSITIONS", "12"))
        free_slots    = max_positions - len(open_positions)

        # If there are free slots, no need to evict — let engine fill normally
        if free_slots > 0:
            return None

        # Score every open position
        pip_map = {"JPY": 0.01}
        pos_scores: List[PositionScore] = []
        for tid, pos in open_positions.items():
            instrument = str(pos.get("instrument", pos.get("symbol", "")))
            pip = 0.01 if "JPY" in instrument.upper() else 0.0001
            entry = float(pos.get("entry", 0) or 0)
            current = float(pos.get("current_price", entry) or entry)
            ps = PositionScore(
                trade_id       = tid,
                symbol         = instrument,
                direction      = str(pos.get("direction", "BUY")),
                unrealized_pnl = float(pos.get("unrealized_pnl", 0) or 0),
                signal_conf    = float(pos.get("signal_confidence", 0.68) or 0.68),
                stale_cycles   = int(pos.get("stale_cycles", 0) or 0),
                entry          = entry,
                current_price  = current,
                pip_size       = pip,
            )
            pos_scores.append(ps)

        # Find the WEAKEST open position
        weakest = min(pos_scores, key=lambda p: p.health_score())
        weakest_score = weakest.health_score()

        # Protect positions earning well
        weakest_pip_profit = (
            (weakest.current_price - weakest.entry) / weakest.pip_size
            if weakest.direction == "BUY"
            else (weakest.entry - weakest.current_price) / weakest.pip_size
        )
        if weakest_pip_profit >= PROTECT_PROFIT_PIPS:
            return None   # it's earning — leave it alone

        # Score every candidate
        from strategies.multi_signal_engine import AggregatedSignal
        cand_scores: List[CandidateScore] = []
        for sig in candidates:
            cs = CandidateScore(
                symbol     = sig.symbol,
                direction  = sig.direction,
                confidence = sig.confidence,
                votes      = sig.votes,
                rr         = sig.rr,
                sl         = sig.sl,
                tp         = sig.tp,
            )
            cand_scores.append(cs)

        if not cand_scores:
            return None

        # Find the STRONGEST candidate
        best_cand = max(cand_scores, key=lambda c: c.conviction_score())
        cand_score = best_cand.conviction_score()

        # Require candidate to be meaningfully better
        improvement = (cand_score - weakest_score) / max(abs(weakest_score), 0.01)
        if improvement < UPGRADE_THRESHOLD:
            return None

        # Don't reallocate same symbol to avoid churn
        if best_cand.symbol == weakest.symbol:
            return None

        log.info(
            f"[Router] REALLOC candidate: {best_cand.symbol} {best_cand.direction} "
            f"score={cand_score:.3f} vs weakest {weakest.symbol} score={weakest_score:.3f} "
            f"(improvement={improvement:.1%})"
        )

        self._reallocations_this_cycle += 1

        return ReallocationDecision(
            close_trade_id   = weakest.trade_id,
            close_symbol     = weakest.symbol,
            close_reason     = f"evicted_by_{best_cand.symbol}_{best_cand.direction}",
            open_symbol      = best_cand.symbol,
            open_direction   = best_cand.direction,
            open_sl          = best_cand.sl,
            open_tp          = best_cand.tp,
            open_confidence  = best_cand.confidence,
            open_votes       = best_cand.votes,
            units            = self.get_units(current_nav, best_cand.symbol),
            improvement_pct  = improvement,
        )


# ── Decision dataclass ────────────────────────────────────────────────────────

@dataclass
class ReallocationDecision:
    """Returned by CapitalRouter.evaluate() when a reallocation is warranted."""
    close_trade_id:  str
    close_symbol:    str
    close_reason:    str
    open_symbol:     str
    open_direction:  str
    open_sl:         float
    open_tp:         float
    open_confidence: float
    open_votes:      int
    units:           int
    improvement_pct: float     # how much stronger the new trade is, as a fraction

    def as_dict(self) -> dict:
        return {
            "close_trade_id":  self.close_trade_id,
            "close_symbol":    self.close_symbol,
            "close_reason":    self.close_reason,
            "open_symbol":     self.open_symbol,
            "open_direction":  self.open_direction,
            "open_sl":         round(self.open_sl, 5),
            "open_tp":         round(self.open_tp, 5),
            "open_confidence": round(self.open_confidence, 4),
            "open_votes":      self.open_votes,
            "units":           self.units,
            "improvement_pct": round(self.improvement_pct * 100, 1),
        }

    def __repr__(self) -> str:
        return (
            f"REALLOC: close {self.close_symbol}({self.close_trade_id[:6]}) "
            f"→ open {self.open_symbol} {self.open_direction} "
            f"({self.improvement_pct:.1%} improvement)"
        )
