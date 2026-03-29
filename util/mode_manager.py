#!/usr/bin/env python3
"""
mode_manager.py
RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_REWRITE

Minimal stub required by brokers/oanda_connector.py (EXTRACTED_VERIFIED).
The connector imports get_connector_environment() to determine live vs practice.
In this repo the answer is always "practice" — enforced here and in .env.
"""

import os


def get_connector_environment(broker: str = "oanda") -> str:
    """
    Always returns the environment declared in OANDA_ENVIRONMENT env var.
    Default: "practice". Never returns "live" unless explicitly set.
    """
    env = os.getenv("OANDA_ENVIRONMENT", "practice").lower().strip()
    if env not in ("practice", "paper"):
        # Hard lock — refuse to return live from this repo
        raise RuntimeError(
            f"RBOTZILLA_OANDA_CLEAN is PRACTICE ONLY. "
            f"OANDA_ENVIRONMENT='{env}' is not allowed. "
            f"Set OANDA_ENVIRONMENT=practice in your .env file."
        )
    return "practice"
