#!/usr/bin/env python3
"""
util/quant_hedge_engine.py — RBOTZILLA_OANDA_CLEAN
Ported from RBOTZILLA_PHOENIX/util/quant_hedge_engine.py

Lightweight Quant Hedge Engine.
Provides correlation-based hedge suggestions and sizing.
Called after a primary trade opens to place an inverse-correlated counter-trade
at 50% position size, reducing directional exposure on the primary currency.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HedgePosition:
    symbol: str
    side: str
    size: float
    hedge_ratio: float
    correlation: float
    entry_price: float


class QuantHedgeEngine:
    """
    Inverse-correlation hedge engine.

    Hedge map: for each primary symbol, defines the best negatively-correlated
    counter symbol and the historical correlation coefficient (negative = inverse).
    """

    def __init__(self) -> None:
        self.default_hedge_ratio = 0.5   # hedge at 50% of primary size

        # Primary → {inverse-correlated hedge symbol, correlation}
        self.hedge_map: Dict[str, Dict[str, float]] = {
            "EUR_USD": {"symbol": "USD_CAD", "correlation": -0.62},
            "GBP_USD": {"symbol": "USD_CAD", "correlation": -0.58},
            "AUD_USD": {"symbol": "USD_CAD", "correlation": -0.65},
            "NZD_USD": {"symbol": "USD_CAD", "correlation": -0.60},
            "USD_JPY": {"symbol": "EUR_USD", "correlation": -0.54},
            "USD_CAD": {"symbol": "AUD_USD", "correlation": -0.65},
            "USD_CHF": {"symbol": "EUR_USD", "correlation": -0.70},
            "EUR_JPY": {"symbol": "USD_JPY", "correlation": -0.48},
            "GBP_JPY": {"symbol": "USD_JPY", "correlation": -0.46},
            "AUD_JPY": {"symbol": "USD_JPY", "correlation": -0.44},
        }

    def evaluate_hedge_opportunity(self, symbol: str) -> Dict:
        symbol = (symbol or "").upper()
        target = self.hedge_map.get(symbol)
        if not target:
            return {
                "hedge_available": False,
                "hedge_symbol": None,
                "correlation": 0.0,
                "reason": "No hedge mapping for symbol",
            }
        return {
            "hedge_available": True,
            "hedge_symbol": target["symbol"],
            "correlation": float(target["correlation"]),
            "reason": "Mapped inverse-correlation hedge",
        }

    def execute_hedge(
        self,
        primary_symbol: str,
        primary_side: str,
        position_size: float,
        entry_price: float,
        hedge_ratio: Optional[float] = None,
    ) -> Optional[HedgePosition]:
        opp = self.evaluate_hedge_opportunity(primary_symbol)
        if not opp.get("hedge_available"):
            return None

        ratio = self.default_hedge_ratio if hedge_ratio is None else float(hedge_ratio)
        ratio = max(0.1, min(ratio, 1.0))

        primary_side = (primary_side or "BUY").upper()
        hedge_side = "SELL" if primary_side == "BUY" else "BUY"
        hedge_size = max(1000.0, round(float(position_size) * ratio, -3))  # round to nearest 1000 units

        return HedgePosition(
            symbol=opp["hedge_symbol"],
            side=hedge_side,
            size=hedge_size,
            hedge_ratio=ratio,
            correlation=float(opp["correlation"]),
            entry_price=float(entry_price),
        )
