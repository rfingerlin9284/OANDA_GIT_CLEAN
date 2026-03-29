import logging
from typing import Optional, Tuple

class TrailLogic:
    """
    Standalone trailing stop logic extracted and cleaned from Phoenix.
    Implements three-stage progression with pair-specific distances.
    SL NEVER moves backwards.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Stage thresholds (profit % from entry)
        self.STAGE1_PCT = 0.0006   # 0.06% - initial move to BE+
        self.STAGE2_PCT = 0.0012   # 0.12% - partial lock
        self.STAGE3_PCT = 0.0018   # 0.18% - begin dynamic trailing
        # Early stage locks
        self.BE_BUFFER_PCT = 0.0002
        self.PARTIAL_LOCK_PCT = 0.0006

    def _classify_pair(self, instrument: str) -> Tuple[str, float]:
        """Returns (pair_type, trailing_distance_decimal)"""
        inst = instrument.replace('/', '_').upper().replace(' ', '_')
        majors = {"EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"}
        minors = {"EUR_GBP", "EUR_AUD", "GBP_JPY", "EUR_JPY", "AUD_JPY", "GBP_AUD", "AUD_NZD", 
                 "EUR_CAD", "GBP_CAD", "AUD_CAD", "NZD_JPY"}
        if inst in majors:
            return "major", 0.0008  # 0.08%
        elif inst in minors or any(base in inst for base in ["EUR", "GBP", "AUD", "NZD"]):
            return "minor", 0.0010  # 0.10%
        else:
            return "exotic", 0.0015  # 0.15%

    def calculate_new_sl(self, entry: float, current_price: float, current_sl: Optional[float], 
                        direction: str, instrument: str) -> Optional[float]:
        """
        Main Phoenix logic. Returns new SL if improvement possible, else None.
        direction: 'BUY' or 'SELL'
        """
        if not entry or not current_price:
            return None
        direction = direction.upper().strip()
        pair_type, trail_dist = self._classify_pair(instrument)
        # LONG / BUY
        if direction in ("BUY", "LONG"):
            profit_pct = (current_price - entry) / entry if entry else 0
            if profit_pct >= self.STAGE1_PCT:
                proposed = entry * (1 + self.BE_BUFFER_PCT)
                if current_sl is None or proposed > current_sl:
                    self.logger.debug(f"BE lock triggered for {instrument} LONG")
                    return proposed
            if profit_pct >= self.STAGE2_PCT:
                proposed = entry * (1 + self.PARTIAL_LOCK_PCT)
                if current_sl is None or proposed > current_sl:
                    self.logger.debug(f"Partial lock triggered for {instrument} LONG")
                    return proposed
            if profit_pct >= self.STAGE3_PCT:
                proposed = current_price * (1 - trail_dist)
                if current_sl is None or proposed > current_sl:
                    self.logger.debug(f"Trailing update for {instrument} LONG at {proposed}")
                    return proposed
        # SHORT / SELL
        elif direction in ("SELL", "SHORT"):
            profit_pct = (entry - current_price) / entry if entry else 0
            if profit_pct >= self.STAGE1_PCT:
                proposed = entry * (1 - self.BE_BUFFER_PCT)
                if current_sl is None or proposed < current_sl:
                    self.logger.debug(f"BE lock triggered for {instrument} SHORT")
                    return proposed
            if profit_pct >= self.STAGE2_PCT:
                proposed = entry * (1 - self.PARTIAL_LOCK_PCT)
                if current_sl is None or proposed < current_sl:
                    self.logger.debug(f"Partial lock triggered for {instrument} SHORT")
                    return proposed
            if profit_pct >= self.STAGE3_PCT:
                proposed = current_price * (1 + trail_dist)
                if current_sl is None or proposed < current_sl:
                    self.logger.debug(f"Trailing update for {instrument} SHORT at {proposed}")
                    return proposed
        return None
