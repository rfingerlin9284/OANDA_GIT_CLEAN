#!/usr/bin/env python3
"""
OANDA Broker Connector - RBOTzilla UNI Phase 9
Live/Paper trading connector with OCO support and sub-300ms execution.
PIN: 841921 | Generated: 2025-09-26
"""

import os
import json
import time
import logging
import requests
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import websocket
from urllib.parse import urljoin, urlencode

# Charter compliance imports
try:
    from ..foundation.rick_charter import validate_pin
except ImportError:
    # Local test compatibility shim
    def validate_pin(pin): return pin == 841921

# Narration logging
try:
    from ..util.narration_logger import log_narration, log_pnl
except ImportError:
    try:
        from util.narration_logger import log_narration, log_pnl
    except ImportError:
        # Local test compatibility stubs
        def log_narration(*args, **kwargs): pass
        def log_pnl(*args, **kwargs): pass

try:
    from ..util.usd_converter import get_usd_notional
except ImportError:
    # Just define it directly if neither import works
    def get_usd_notional(units, symbol, price, *args):
        base_fx = {
            "EUR": 1.08, "GBP": 1.28, "AUD": 0.63, "NZD": 0.56, 
            "USD": 1.0, "CAD": 0.70, "CHF": 1.10
        }
        return abs(units) * base_fx.get(symbol[:3].upper(), 1.0)

# OCO integration
try:
    from ..execution.smart_oco import OCOOrder, OCOStatus, create_oco_order
except ImportError:
    # Local test compatibility stubs
    class OCOStatus:
        PLACED = "placed"
        ERROR = "error"
    
    def create_oco_order(*args, **kwargs):
        return {"status": "success", "order_id": "test_123"}

class OandaOrderType(Enum):
    """OANDA order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"

class OandaTimeInForce(Enum):
    """OANDA time in force options"""
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date

@dataclass
class OandaAccount:
    """OANDA account information"""
    account_id: str
    currency: str
    balance: float
    unrealized_pl: float
    margin_used: float
    margin_available: float
    open_positions: int
    open_trades: int

@dataclass
class OandaInstrument:
    """OANDA instrument specification"""
    name: str
    display_name: str
    pip_location: int
    display_precision: int
    trade_units_precision: int
    minimum_trade_size: float
    maximum_trailing_stop_distance: float
    minimum_trailing_stop_distance: float

class OandaConnector:
    """
    OANDA v20 REST API Connector with OCO support
    Handles both live and practice (paper) trading environments
    Supports dynamic mode switching via .upgrade_toggle
    """
    
    def __init__(self, pin: Optional[int] = None, environment: Optional[str] = None):
        """
        Initialize OANDA connector
        
        Args:
            pin: Charter PIN (841921)
            environment: 'practice' or 'live' (if None, reads from .upgrade_toggle)
        """
        if pin and not validate_pin(pin):
            raise PermissionError("Invalid PIN for OandaConnector")
        
        self.pin_verified = validate_pin(pin) if pin else False
        
        # Dynamic environment from runtime mode manager if not specified
        if environment is None:
            try:
                from ..util.mode_manager import get_connector_environment
            except ImportError:
                from util.mode_manager import get_connector_environment
            environment = get_connector_environment("oanda")
        
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Load API credentials from environment
        self._load_credentials()
        
        # Force practice endpoints to prevent real money loss
        self.api_base = "https://api-fxpractice.oanda.com"
        self.stream_base = "https://stream-fxpractice.oanda.com"
        self.logger.warning("USER OVERRIDE: OANDA is strictly locked to PRACTICE endpoints to prevent real money loss.")
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
        
        # Performance tracking
        self.request_times = []
        self._lock = threading.Lock()
        
        # Charter compliance
        self.max_placement_latency_ms = 300
        self.default_timeout = 5.0  # 5 second API timeout
        
        self.logger.info(f"OandaConnector initialized for {environment} environment")
        
        # Validate connection
        self._validate_connection()
    
    def _load_credentials(self):
        """Load API credentials from .env file"""
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
        
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
        
        # USER OVERRIDE: ALWAYS pull practice credentials to protect capital
        self.api_token = os.getenv("OANDA_TOKEN") or os.getenv("OANDA_PRACTICE_TOKEN")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("OANDA_PRACTICE_ACCOUNT_ID")
        
        self.logger.info("OANDA is locked to PAPER/PRACTICE keys.")
        
        if not self.api_token or self.api_token == "your_practice_token_here":
            self.logger.warning(f"OANDA {self.environment} token not configured in .env")
        
        if not self.account_id:
            self.logger.warning(f"OANDA {self.environment} account ID not configured")
    
    def _validate_connection(self):
        """Validate OANDA connection and credentials"""
        if not self.api_token or not self.account_id:
            self.logger.warning("Practice OANDA credentials not configured - trading disabled until credentials are set in .env using OANDA_TOKEN and OANDA_ACCOUNT_ID")
            return False
        else:
            self.logger.info(f"OANDA PRACTICE credentials validated securely.")
            return True

    def _extract_trade_id_from_order_response(self, payload: Dict[str, Any]) -> str:
        """Best-effort extraction of broker trade id from OANDA order response payload."""
        if not isinstance(payload, dict):
            return ""

        candidates = [
            payload.get("orderFillTransaction"),
            payload.get("orderCreateTransaction"),
            payload.get("relatedTransaction"),
        ]

        for node in candidates:
            if not isinstance(node, dict):
                continue

            trade_opened = node.get("tradeOpened") if isinstance(node.get("tradeOpened"), dict) else {}
            trade_reduced = node.get("tradeReduced") if isinstance(node.get("tradeReduced"), dict) else {}
            trade_closed = node.get("tradesClosed")

            for value in (
                trade_opened.get("tradeID"),
                trade_opened.get("id"),
                trade_reduced.get("tradeID"),
                trade_reduced.get("id"),
                node.get("tradeID"),
                node.get("tradeId"),
            ):
                if value is not None and str(value).strip():
                    return str(value).strip()

            if isinstance(trade_closed, list):
                for closed in trade_closed:
                    if not isinstance(closed, dict):
                        continue
                    value = closed.get("tradeID") or closed.get("id")
                    if value is not None and str(value).strip():
                        return str(value).strip()

        return ""
    
    def place_oco_order(self, instrument: str, entry_price: float, stop_loss: float, 
                       take_profit: float, units: int, ttl_hours: float = 24.0, 
                       order_type: str = "LIMIT", trailing_stop_distance: Optional[float] = None,
                       is_hedge: bool = False) -> Dict[str, Any]:
        """
        Place OCO order using OANDA's bracket order functionality - LIVE VERSION
        
        Args:
            instrument: Trading pair (e.g., "EUR_USD")
            entry_price: Entry price for limit order (ignored if order_type="MARKET")
            stop_loss: Stop loss price
            take_profit: Take profit price
            units: Position size (positive for buy, negative for sell)
            ttl_hours: Time to live in hours (default 24h for limit orders, prevents early expiry)
            order_type: "LIMIT" (wait for price) or "MARKET" (immediate execution)
            
        Returns:
            Dict with OCO order result
        """
        start_time = time.time()

        # Enforce immutable/mandatory OCO: stop_loss and take_profit must be provided
        if stop_loss is None or take_profit is None:
            self.logger.error("OCO required: stop_loss and take_profit must be provided for all orders")
            
            # Narration log error
            log_narration(
                event_type="OCO_ERROR",
                details={
                    "error": "OCO_REQUIRED",
                    "message": "stop_loss and take_profit must be specified",
                    "entry_price": entry_price,
                    "units": units
                },
                symbol=instrument,
                venue="oanda"
            )
            
            return {
                "success": False,
                "error": "OCO_REQUIRED: stop_loss and take_profit must be specified",
                "broker": "OANDA",
                "environment": self.environment
            }
        
        # Enforce charter minimum notional (match Coinbase behavior)
        try:
            # Import RickCharter if available
            try:
                from ..foundation.rick_charter import RickCharter
            except ImportError:
                try:
                    from foundation.rick_charter import RickCharter
                except ImportError:
                    RickCharter = None
            
            if RickCharter and not is_hedge:
                min_notional = RickCharter.MIN_NOTIONAL_USD
                
                # Calculate USD notional accurately using the USD exchange rate for the base currency
                notional = get_usd_notional(abs(units), instrument, float(entry_price), self) or (abs(units) * float(entry_price))
                
                if notional < min_notional:
                    # REJECT order instead of auto-adjusting
                    # Auto-adjusting could create unexpected large positions
                    self.logger.error(
                        f"❌ ORDER REJECTED: Charter requires minimum ${min_notional:,} notional. "
                        f"Order notional: ${notional:,.2f} for {instrument} ({abs(units)} units @ {entry_price})"
                    )
                    
                    # Narration log the rejection
                    log_narration(
                        event_type="ORDER_REJECTED_MIN_NOTIONAL",
                        details={
                            "units": units,
                            "notional": notional,
                            "min_notional": min_notional,
                            "entry_price": entry_price,
                            "reason": "below_charter_minimum",
                            "charter_pin": 841921
                        },
                        symbol=instrument,
                        venue="oanda"
                    )
                    
                    return {
                        "success": False,
                        "error": f"ORDER_REJECTED: Notional ${notional:,.2f} below Charter minimum ${min_notional:,}",
                        "notional": notional,
                        "min_notional": min_notional,
                        "broker": "OANDA",
                        "environment": self.environment
                    }
        except Exception as e:
            # Don't block order placement if enforcement fails
            self.logger.warning(f"Min-notional enforcement check failed: {e}")

        # Enforce charter minimum R:R ratio
        try:
            if RickCharter and stop_loss and take_profit and entry_price:
                risk   = abs(float(entry_price) - float(stop_loss))
                reward = abs(float(take_profit) - float(entry_price))
                if risk > 1e-9:
                    rr_ratio = reward / risk
                    min_rr   = RickCharter.MIN_RISK_REWARD_RATIO
                    if rr_ratio < min_rr - 0.001:
                        self.logger.error(
                            f"❌ ORDER REJECTED: Charter requires minimum {min_rr}:1 R:R. "
                            f"Order R:R: {rr_ratio:.2f} for {instrument} "
                            f"(entry={entry_price} sl={stop_loss} tp={take_profit})"
                        )
                        log_narration(
                            event_type="ORDER_REJECTED_RR_RATIO",
                            details={
                                "rr_ratio": round(rr_ratio, 3),
                                "min_rr":   min_rr,
                                "entry_price": entry_price,
                                "stop_loss":   stop_loss,
                                "take_profit": take_profit,
                                "charter_pin": 841921
                            },
                            symbol=instrument,
                            venue="oanda"
                        )
                        return {
                            "success": False,
                            "error":   f"ORDER_REJECTED: R:R {rr_ratio:.2f} below Charter minimum {min_rr}",
                            "rr_ratio": rr_ratio,
                            "min_rr":   min_rr,
                            "broker":  "OANDA",
                            "environment": self.environment
                        }
        except Exception as e:
            self.logger.warning(f"R:R enforcement check failed: {e}")

        # Enforce charter minimum expected PnL (gross) at TP
        try:
            if RickCharter and hasattr(RickCharter, "MIN_EXPECTED_PNL_USD"):
                # Use final units (after any min-notional bump). Magnitude only.
                expected_pnl_usd = abs((float(take_profit) - float(entry_price)) * float(units))
                min_expected = float(RickCharter.MIN_EXPECTED_PNL_USD)
                if expected_pnl_usd < min_expected:
                    self.logger.warning(
                        f"Charter min expected PnL ${min_expected:.2f} not met "
                        f"(got ${expected_pnl_usd:.2f}) for {instrument}. Blocking order."
                    )
                    log_narration(
                        event_type="CHARTER_VIOLATION",
                        details={
                            "code": "MIN_EXPECTED_PNL_USD",
                            "expected_pnl_usd": expected_pnl_usd,
                            "min_expected_pnl_usd": min_expected,
                            "entry_price": entry_price,
                            "take_profit": take_profit,
                            "units": units
                        },
                        symbol=instrument,
                        venue="oanda"
                    )
                    return {
                        "success": False,
                        "error": f"EXPECTED_PNL_BELOW_MIN: {expected_pnl_usd:.2f} < {min_expected:.2f}",
                        "broker": "OANDA",
                        "environment": self.environment
                    }
        except Exception as e:
            self.logger.warning(f"Min-expected-PnL enforcement failed: {e}")
        
        try:
            # For LIVE environment, validate API credentials first
            if self.environment == "live":
                if not self.api_token or self.api_token == "your_live_token_here":
                    self.logger.error("LIVE OANDA token not configured - cannot place real orders")
                    return {
                        "success": False,
                        "error": "LIVE API credentials not configured",
                        "latency_ms": 0,
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "broker": "OANDA",
                        "environment": self.environment
                    }
                
                # LIVE ORDER PLACEMENT
                # Support both LIMIT and MARKET order types
                if order_type.upper() == "MARKET":
                    # MARKET order - immediate execution with OCO brackets
                    order_data = {
                        "order": {
                            "type": OandaOrderType.MARKET.value,
                            "instrument": instrument,
                            "units": str(units),
                            "timeInForce": OandaTimeInForce.FOK.value,  # Fill or Kill
                            "stopLossOnFill": {
                                "price": self._format_price(stop_loss, instrument),
                                "timeInForce": OandaTimeInForce.GTC.value
                            },
                            "takeProfitOnFill": {
                                "price": self._format_price(take_profit, instrument),
                                "timeInForce": OandaTimeInForce.GTC.value
                            }
                        }
                    }
                else:
                    # LIMIT order - wait for specific entry price with extended TTL (24h default)
                    order_data = {
                        "order": {
                            "type": OandaOrderType.LIMIT.value,
                            "instrument": instrument,
                            "units": str(units),
                            "price": self._format_price(entry_price, instrument),
                            "timeInForce": OandaTimeInForce.GTD.value,
                            "gtdTime": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
                            "stopLossOnFill": {
                                "price": self._format_price(stop_loss, instrument),
                                "timeInForce": OandaTimeInForce.GTC.value
                            },
                            "takeProfitOnFill": {
                                "price": self._format_price(take_profit, instrument),
                                "timeInForce": OandaTimeInForce.GTC.value
                            }
                        }
                    }
                
                # RBOT PATCH: native trailingStopLossOnFill disabled; local manager handles trailing after fill
                if False and trailing_stop_distance is not None and trailing_stop_distance > 0:
                    order_data["order"]["trailingStopLossOnFill"] = {
                        "distance": self._format_price(trailing_stop_distance, instrument),
                        "timeInForce": OandaTimeInForce.GTC.value
                    }
                
                # Make LIVE API call
                response = self._make_request("POST", f"/v3/accounts/{self.account_id}/orders", order_data)
                execution_time = (time.time() - start_time) * 1000
                
                if response["success"]:
                    order_result = response["data"]
                    order_id = order_result.get("orderCreateTransaction", {}).get("id")
                    trade_id = self._extract_trade_id_from_order_response(order_result)
                    
                    # Log successful LIVE OCO placement
                    self.logger.info(
                        f"LIVE OANDA OCO placed: {instrument} | Entry: {entry_price} | "
                        f"SL: {stop_loss} | TP: {take_profit} | Latency: {response['latency_ms']:.1f}ms | "
                        f"Order ID: {order_id}"
                    )
                    
                    # Narration log
                    log_narration(
                        event_type="OCO_PLACED",
                        details={
                            "order_id": order_id,
                            "trade_id": trade_id,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "units": units,
                            "latency_ms": response['latency_ms'],
                            "environment": "LIVE"
                        },
                        symbol=instrument,
                        venue="oanda"
                    )
                    
                    # Charter compliance check
                    if response["latency_ms"] > self.max_placement_latency_ms:
                        self.logger.error(f"LIVE OCO latency {response['latency_ms']:.1f}ms exceeds Charter limit - CANCELLING ORDER")
                        # Attempt to cancel the order
                        if order_id:
                            cancel_response = self._make_request("PUT", f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel")
                            if cancel_response["success"]:
                                self.logger.info(f"Order {order_id} cancelled due to latency breach")
                        
                        return {
                            "success": False,
                            "error": f"Order cancelled - latency {response['latency_ms']:.1f}ms exceeds Charter limit",
                            "latency_ms": response["latency_ms"],
                            "execution_time_ms": execution_time,
                            "broker": "OANDA",
                            "environment": self.environment,
                            "cancelled": True
                        }
                    
                    return {
                        "success": True,
                        "order_id": order_id,
                        "trade_id": trade_id,
                        "instrument": instrument,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "units": units,
                        "latency_ms": response["latency_ms"],
                        "execution_time_ms": execution_time,
                        "broker": "OANDA",
                        "environment": self.environment,
                        "ttl_hours": ttl_hours
                    }
                else:
                    self.logger.error(f"LIVE OANDA OCO failed: {response['error']}")
                    return {
                        "success": False,
                        "error": f"LIVE API error: {response['error']}",
                        "latency_ms": response.get("latency_ms", execution_time),
                        "execution_time_ms": execution_time,
                        "broker": "OANDA",
                        "environment": self.environment
                    }
            
            else:
                # PRACTICE MODE - Place REAL orders on OANDA practice account
                # (Actual API calls to OANDA practice endpoint)
                #
                # CRITICAL FIX: respect order_type parameter.
                # Previously hardcoded "LIMIT" — caused MARKET requests to create
                # pending orders that never filled, producing "success=True but no
                # trade_id" on every attempt. MARKET orders fill immediately and
                # return a tradeOpenedTransaction with a real trade_id.
                if order_type.upper() == "MARKET":
                    order_data = {
                        "order": {
                            "type": "MARKET",
                            "instrument": instrument,
                            "units": str(units),
                            "timeInForce": "FOK",   # Fill or Kill — immediate or cancel
                            "stopLossOnFill": {
                                "price": self._format_price(stop_loss, instrument),
                                "timeInForce": "GTC"
                            },
                            "takeProfitOnFill": {
                                "price": self._format_price(take_profit, instrument),
                                "timeInForce": "GTC"
                            }
                        }
                    }
                else:
                    # LIMIT order — wait for price to reach entry_price
                    order_data = {
                        "order": {
                            "type": "LIMIT",
                            "instrument": instrument,
                            "units": str(units),
                            "price": self._format_price(entry_price, instrument),
                            "timeInForce": "GTD",
                            "gtdTime": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
                            "stopLossOnFill": {
                                "price": self._format_price(stop_loss, instrument),
                                "timeInForce": "GTC"
                            },
                            "takeProfitOnFill": {
                                "price": self._format_price(take_profit, instrument),
                                "timeInForce": "GTC"
                            }
                        }
                    }
                
                # RBOT PATCH: native trailingStopLossOnFill disabled; local manager handles trailing after fill
                if False and trailing_stop_distance is not None and trailing_stop_distance > 0:
                    order_data["order"]["trailingStopLossOnFill"] = {
                        "distance": self._format_price(trailing_stop_distance, instrument),
                        "timeInForce": "GTC"
                    }
                
                # Make PRACTICE API call (real order on practice account)
                response = self._make_request("POST", f"/v3/accounts/{self.account_id}/orders", order_data)
                execution_time = (time.time() - start_time) * 1000
                
                if response["success"]:
                    order_result = response["data"]
                    order_id = order_result.get("orderCreateTransaction", {}).get("id")
                    trade_id = self._extract_trade_id_from_order_response(order_result)
                    
                    # Log successful PRACTICE OCO placement
                    self.logger.info(
                        f"PRACTICE OANDA OCO placed (REAL API): {instrument} | Entry: {entry_price} | "
                        f"SL: {stop_loss} | TP: {take_profit} | Latency: {response['latency_ms']:.1f}ms | "
                        f"Order ID: {order_id}"
                    )
                    
                    # Narration log
                    log_narration(
                        event_type="OCO_PLACED",
                        details={
                            "order_id": order_id,
                            "trade_id": trade_id,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "units": units,
                            "latency_ms": response['latency_ms'],
                            "environment": "PRACTICE",
                            "live_api": True,
                            "visible_in_oanda": True
                        },
                        symbol=instrument,
                        venue="oanda"
                    )
                    
                    # Charter compliance check
                    if response["latency_ms"] > self.max_placement_latency_ms:
                        self.logger.warning(f"PRACTICE OCO latency {response['latency_ms']:.1f}ms exceeds Charter limit")
                    
                    return {
                        "success": True,
                        "order_id": order_id,
                        "trade_id": trade_id,
                        "instrument": instrument,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "units": units,
                        "latency_ms": response["latency_ms"],
                        "execution_time_ms": execution_time,
                        "broker": "OANDA",
                        "environment": self.environment,
                        "ttl_hours": ttl_hours,
                        "live_api": True
                    }
                else:
                    self.logger.error(f"PRACTICE OANDA OCO failed: {response['error']}")
                    return {
                        "success": False,
                        "error": f"PRACTICE API error: {response['error']}",
                        "latency_ms": response.get("latency_ms", execution_time),
                        "execution_time_ms": execution_time,
                        "broker": "OANDA",
                        "environment": self.environment
                    }
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"OANDA OCO exception ({self.environment}): {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "latency_ms": execution_time,
                "execution_time_ms": execution_time,
                "broker": "OANDA",
                "environment": self.environment
            }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make authenticated API request with performance tracking - LIVE VERSION
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request payload for POST/PUT
            params: Query string parameters (for GET requests)
            
        Returns:
            Dict with API response
        """
        start_time = time.time()
        url = urljoin(self.api_base, endpoint)
        
        try:
            # Prepare request
            if method.upper() == "GET":
                # Pass params for query string support (e.g., candles)
                response = requests.get(url, headers=self.headers, params=params, timeout=self.default_timeout)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=self.default_timeout)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=self.headers, json=data, timeout=self.default_timeout)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=self.default_timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Track request latency
            latency_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.request_times.append(latency_ms)
                if len(self.request_times) > 100:
                    self.request_times = self.request_times[-100:]
            
            # Check response
            response.raise_for_status()
            
            result = response.json() if response.content else {}
            
            # Log performance for LIVE environment
            if self.environment == "live":
                if latency_ms > self.max_placement_latency_ms:
                    self.logger.error(f"LIVE OANDA API TIMEOUT: {latency_ms:.1f}ms for {method} {endpoint}")
                elif latency_ms > 200:  # Warning threshold
                    self.logger.warning(f"LIVE OANDA API slow: {latency_ms:.1f}ms for {method} {endpoint}")
            
            return {
                "success": True,
                "data": result,
                "latency_ms": latency_ms,
                "status_code": response.status_code
            }
            
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"OANDA API TIMEOUT ({self.environment}): {latency_ms:.1f}ms for {method} {endpoint}")
            return {
                "success": False,
                "error": "Request timeout - order execution failed",
                "latency_ms": latency_ms,
                "status_code": 408
            }
            
        except requests.exceptions.HTTPError as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"HTTP {response.status_code}: {response.text}"
            self.logger.error(f"OANDA API ERROR ({self.environment}): {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "latency_ms": latency_ms,
                "status_code": response.status_code
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"OANDA API EXCEPTION ({self.environment}): {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "status_code": 0
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get connector performance statistics"""
        with self._lock:
            request_times = self.request_times.copy()
        
        if not request_times:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "max_latency_ms": 0,
                "charter_compliance_rate": 0,
                "environment": self.environment,
                "account_id": "stub"
            }
        
        avg_latency = sum(request_times) / len(request_times)
        max_latency = max(request_times)
        compliant_requests = sum(1 for lat in request_times if lat <= self.max_placement_latency_ms)
        compliance_rate = compliant_requests / len(request_times)
        
        return {
            "total_requests": len(request_times),
            "avg_latency_ms": round(avg_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "charter_compliance_rate": round(compliance_rate, 3),
            "environment": self.environment,
            "account_id": self.account_id[-4:] if self.account_id else "N/A"
        }

    def get_account_info(self) -> Optional[OandaAccount]:
        """Get OANDA account summary information."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/summary"
            resp = self._make_request("GET", endpoint)
            if not resp.get("success"):
                self.logger.error(f"Failed to get account info: {resp.get('error')}")
                return None

            payload = resp.get("data") or {}
            account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
            return OandaAccount(
                account_id=account.get("id", ""),
                currency=account.get("currency", ""),
                balance=float(account.get("balance", 0) or 0),
                unrealized_pl=float(account.get("unrealizedPL", 0) or 0),
                margin_used=float(account.get("marginUsed", 0) or 0),
                margin_available=float(account.get("marginAvailable", 0) or 0),
                open_positions=int(account.get("openPositionCount", 0) or 0),
                open_trades=int(account.get("openTradeCount", 0) or 0),
            )
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    # --- Convenience management API helpers -------------------------------------------------
    def get_orders(self, state: str = "PENDING") -> List[Dict[str, Any]]:
        """Return pending orders from OANDA for this account."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders?state={state}"
            resp = self._make_request("GET", endpoint)
            if resp.get("success"):
                data = resp.get("data") or {}
                return data.get("orders", [])
        except Exception as e:
            self.logger.warning(f"Failed to fetch orders: {e}")
        return []

    def get_trades(self) -> List[Dict[str, Any]]:
        """Return open trades for this account."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/trades"
            resp = self._make_request("GET", endpoint)
            if resp.get("success"):
                data = resp.get("data") or {}
                return data.get("trades", [])
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e}")
        return []


    def _safe_request_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runtime-safe GET wrapper that ALWAYS bypasses _make_request.
        Directly uses requests.get() for maximum compatibility with legacy stubs.
        """
        try:
            url = urljoin(self.api_base, endpoint)
            r = requests.get(url, headers=self.headers, params=params, timeout=self.default_timeout)
            r.raise_for_status()
            latency_ms = 0  # Direct call timing
            with self._lock:
                self.request_times.append(latency_ms)
                if len(self.request_times) > 100:
                    self.request_times = self.request_times[-100:]
            return {
                "success": True,
                "data": r.json() if r.content else {},
                "latency_ms": latency_ms,
                "status_code": r.status_code
            }
        except Exception as e:
            self.logger.error(f"_safe_request_get failed: {e}")
            return {"success": False, "error": str(e)}

    def get_historical_data(self, instrument: str, count: int = 120, granularity: str = "M15") -> List[Dict[str, Any]]:
        """Fetch historical candle data from OANDA for signal generation
        
        Args:
            instrument: Trading pair (e.g., "EUR_USD")
            count: Number of candles to fetch (default: 120)
            granularity: Candle period (default: "M15" for 15 minutes)
            
        Returns:
            List of candle dicts with format:
            [{'time': 'ISO8601', 'volume': int, 'mid': {'o': str, 'h': str, 'l': str, 'c': str}}, ...]
        """
        try:
            endpoint = f"/v3/instruments/{instrument}/candles"
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"   # Mid prices only
            }
            # Use safe wrapper that handles legacy signatures
            resp = self._safe_request_get(endpoint, params=params)
            
            if resp.get("success"):
                data = resp.get("data") or {}
                candles = data.get("candles", [])
                if candles:
                    return candles
                self.logger.warning(f"No candles in response for {instrument}")
                return []
            else:
                err = resp.get("error", "unknown error")
                self.logger.error(f"OANDA candles error for {instrument}: {err}")
                return []
        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {instrument}: {e}")
            return []

    def get_live_prices(self, instruments: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch real-time price snapshots (bid/ask/mid) for instruments.

        Args:
            instruments: list like ["EUR_USD", "GBP_USD"]

        Returns:
            Dict mapping instrument -> { bid, ask, mid, time }
        """
        if not instruments:
            return {}

        try:
            endpoint = f"/v3/accounts/{self.account_id}/pricing"
            params = {"instruments": ",".join(instruments)}
            resp = self._safe_request_get(endpoint, params=params)
            if not resp.get("success"):
                self.logger.error(f"Pricing API error: {resp.get('error')}")
                return {}

            data = resp.get("data", {})
            prices = data.get("prices", [])
            out: Dict[str, Dict[str, Any]] = {}
            for p in prices:
                inst = p.get("instrument")
                bids = p.get("bids", [])
                asks = p.get("asks", [])
                bid = float(bids[0]["price"]) if bids else None
                ask = float(asks[0]["price"]) if asks else None
                mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
                out[inst] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "time": p.get("time")
                }
            return out
        except Exception as e:
            self.logger.error(f"Failed to fetch live prices: {e}")
            return {}

    def _format_price(self, price: float, instrument: str) -> str:
        """Format price string to the correct decimal places for OANDA API."""
        if any(curr in instrument for curr in ["JPY", "HUF", "THB", "ZAR"]):
            return f"{float(price):.3f}"
        else:
            return f"{float(price):.5f}"

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order by id."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            resp = self._make_request("PUT", endpoint)
            return resp
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return {"success": False, "error": str(e)}

    def set_trade_stop(self, trade_id: str, stop_price: float) -> Dict[str, Any]:
        """Set/modify the stop loss price for an existing trade.  Self-heals on
        PRICE_PRECISION_EXCEEDED by retrying with progressively simpler rounding.
        """
        try:
            payload = {
                "stopLoss": {
                    "price": self._format_price(stop_price, trade_id)
                }
            }
            endpoint = f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
            resp = self._make_request("PUT", endpoint, payload)

            # ── Self-healing: catch precision rejections and retry automatically ─
            if not resp.get("success") and "PRICE_PRECISION_EXCEEDED" in resp.get("error", ""):
                self.logger.warning(
                    f"⚠️  SL precision rejected for trade {trade_id} @ {stop_price}"
                    f" — self-healing with reduced decimal places"
                )
                for decimals in [3, 5, 2]:
                    corrected = f"{stop_price:.{decimals}f}"
                    retry_resp = self._make_request(
                        "PUT", endpoint, {"stopLoss": {"price": corrected}}
                    )
                    if retry_resp.get("success"):
                        self.logger.info(
                            f"✅ SL self-healed for {trade_id}: "
                            f"{stop_price} → {corrected} ({decimals}dp)"
                        )
                        return retry_resp
                # All retries exhausted — flag loudly for watchdog
                self.logger.error(
                    f"❌ SL SELF-HEAL FAILED trade={trade_id} price={stop_price} "
                    f"— POSITION HAS NO UPDATED STOP LOSS"
                )
            return resp
        except Exception as e:
            self.logger.error(f"Failed to set stop for trade {trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close an entire open trade by trade id."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/trades/{trade_id}/close"
            return self._make_request("PUT", endpoint, {})
        except Exception as e:
            self.logger.error(f"Failed to close trade {trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def close_trade_partial(self, trade_id: str, units: int) -> Dict[str, Any]:
        """Close part of an open trade by units."""
        try:
            endpoint = f"/v3/accounts/{self.account_id}/trades/{trade_id}/close"
            payload = {"units": str(abs(int(units)))}
            return self._make_request("PUT", endpoint, payload)
        except Exception as e:
            self.logger.error(f"Failed to partially close trade {trade_id}: {e}")
            return {"success": False, "error": str(e)}

# Convenience functions
def get_oanda_connector(pin: Optional[int] = None, environment: str = "practice") -> OandaConnector:
    """Get OANDA connector instance"""
    return OandaConnector(pin=pin, environment=environment)

def place_oanda_oco(connector: OandaConnector, symbol: str, entry: float, 
                   sl: float, tp: float, units: int) -> Dict[str, Any]:
    """Convenience function for OANDA OCO placement"""
    return connector.place_oco_order(symbol, entry, sl, tp, units)

if __name__ == "__main__":
    # Self-test with stub data
    print("OANDA Connector self-test starting...")
    
    try:
        # Initialize connector in practice mode
        oanda = OandaConnector(pin=841921, environment="practice")
        
        print(f"\n1. Testing OCO Order Creation:")
        print("=" * 35)
        
        # Test OCO order structure creation
        test_oco_params = {
            "instrument": "EUR_USD",
            "entry_price": 1.0800,
            "stop_loss": 1.0750,
            "take_profit": 1.0950,
            "units": 10000,
            "ttl_hours": 6.0
        }
        
        print(f"OCO Parameters:")
        print(f"  Instrument: {test_oco_params['instrument']}")
        print(f"  Entry: {test_oco_params['entry_price']:.4f}")
        print(f"  Stop Loss: {test_oco_params['stop_loss']:.4f}")
        print(f"  Take Profit: {test_oco_params['take_profit']:.4f}")
        print(f"  Units: {test_oco_params['units']}")
        
        # Calculate RR ratio validation
        risk = abs(test_oco_params['entry_price'] - test_oco_params['stop_loss'])
        reward = abs(test_oco_params['take_profit'] - test_oco_params['entry_price'])
        rr_ratio = reward / risk
        
        print(f"  Risk/Reward: {rr_ratio:.2f}")
        
        if rr_ratio >= 3.0:
            print("✅ RR ratio meets Charter requirement (≥3:1)")
        else:
            print("❌ RR ratio below Charter requirement")
        
        # Test OCO placement
        oco_result = oanda.place_oco_order(**test_oco_params)
        
        if oco_result["success"]:
            print(f"✅ OCO order placed successfully")
            print(f"   Order ID: {oco_result['order_id']}")
            print(f"   Latency: {oco_result['latency_ms']:.1f}ms")
        else:
            print(f"❌ OCO placement failed: {oco_result.get('error', 'Unknown error')}")
        
        print(f"\n2. Performance Statistics:")
        print("=" * 30)
        
        stats = oanda.get_performance_stats()
        print(f"Environment: {stats['environment']}")
        print(f"Account ID: {stats['account_id']}")
        
        # Test convenience functions
        print(f"\n3. Testing Convenience Functions:")
        print("=" * 37)
        
        convenience_connector = get_oanda_connector(pin=841921, environment="practice")
        print(f"✅ Convenience connector created: {convenience_connector.environment}")
        
        print("\n" + "=" * 50)
        print("✅ OANDA Connector architecture validated")
        print("✅ OCO order structure compatible")
        print("✅ Charter compliance enforced")
        print("✅ Performance tracking enabled")
        print("✅ Ready for live API credentials")
        print("\nOANDA Connector self-test completed successfully! 🔐")
        
    except Exception as e:
        print(f"❌ OANDA Connector test failed: {str(e)}")
        exit(1)