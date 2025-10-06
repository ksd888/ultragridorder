#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models and Structures - Fixed and Enhanced
Version: 2.0.1

Key fixes:
- Added input validation in Position.update_pnl()
- Added safe division checks
- Enhanced error handling
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal
from enum import Enum
import json
import logging

# Setup logging
logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


@dataclass
class Position:
    """
    Trading position with full lifecycle tracking
    
    ✅ FIXED: Added comprehensive validation and error handling
    """
    position_id: str
    symbol: str
    entry_price: float
    size_coin: float
    size_usd: float
    
    # Take Profit
    tp_price: float
    tp_order_id: Optional[str] = None
    
    # Stop Loss
    sl_price: Optional[float] = None
    sl_type: Optional[str] = None
    sl_order_id: Optional[str] = None
    trailing_activated: bool = False
    highest_price: float = 0.0
    
    # Timestamps
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: Optional[datetime] = None
    
    # Metadata
    entry_reason: str = ""
    exit_reason: str = ""
    status: PositionStatus = PositionStatus.OPEN
    
    # Market context at entry
    market_phase: str = "unknown"
    volatility_at_entry: float = 0.0
    cvd_z_at_entry: float = 0.0
    
    # Performance tracking
    max_profit_pct: float = 0.0
    max_loss_pct: float = 0.0
    current_pnl: float = 0.0
    
    def __post_init__(self):
        """✅ FIXED: Validate position data"""
        # Validate entry price
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}")
        
        # Validate size
        if self.size_coin <= 0:
            raise ValueError(f"Invalid size_coin: {self.size_coin}")
        
        if self.size_usd <= 0:
            raise ValueError(f"Invalid size_usd: {self.size_usd}")
        
        # Validate TP price
        if self.tp_price <= 0:
            raise ValueError(f"Invalid tp_price: {self.tp_price}")
        
        if self.tp_price <= self.entry_price:
            logger.warning(
                f"TP price {self.tp_price} <= entry {self.entry_price} "
                f"for position {self.position_id}"
            )
        
        # Validate SL price if set
        if self.sl_price is not None:
            if self.sl_price <= 0:
                raise ValueError(f"Invalid sl_price: {self.sl_price}")
            
            if self.sl_price >= self.entry_price:
                raise ValueError(
                    f"SL price {self.sl_price} >= entry {self.entry_price}"
                )
        
        # Initialize highest price
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
    
    def update_pnl(self, current_price: float):
        """
        Update current P&L
        
        ✅ FIXED: Added comprehensive validation
        """
        # Validate current price
        if current_price is None:
            logger.warning(f"Received None as current_price for {self.position_id}")
            return
        
        if current_price <= 0:
            logger.warning(
                f"Invalid current_price: {current_price} for {self.position_id}"
            )
            return
        
        # Validate entry price (safety check)
        if self.entry_price <= 0:
            logger.error(
                f"Invalid entry_price: {self.entry_price} for {self.position_id}"
            )
            return
        
        try:
            # Calculate P&L
            self.current_pnl = (current_price - self.entry_price) * self.size_coin
            
            # Calculate profit percentage (safe division)
            profit_pct = (current_price / self.entry_price) - 1
            
            # Update max profit/loss
            self.max_profit_pct = max(self.max_profit_pct, profit_pct)
            self.max_loss_pct = min(self.max_loss_pct, profit_pct)
            
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Update timestamp
            self.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(
                f"Error updating P&L for {self.position_id}: {e}",
                exc_info=True
            )
    
    def should_timeout(self, max_hours: int = 48) -> bool:
        """
        Check if position should be force-closed due to timeout
        
        ✅ FIXED: Made max_hours configurable (was hard-coded 24)
        """
        try:
            age_hours = (
                datetime.now(timezone.utc) - self.entry_time
            ).total_seconds() / 3600
            
            return age_hours > max_hours
        
        except Exception as e:
            logger.error(
                f"Error checking timeout for {self.position_id}: {e}",
                exc_info=True
            )
            return False
    
    def get_profit_pct(self, current_price: float) -> float:
        """
        Get current profit percentage
        
        ✅ NEW: Helper method with safe calculation
        """
        if self.entry_price <= 0:
            return 0.0
        
        if current_price <= 0:
            return 0.0
        
        return (current_price / self.entry_price) - 1
    
    def is_profitable(self, current_price: float) -> bool:
        """✅ NEW: Check if position is profitable"""
        return self.get_profit_pct(current_price) > 0
    
    def to_dict(self) -> Dict:
        """Convert to dict for serialization"""
        try:
            data = asdict(self)
            # Convert enum
            data['status'] = self.status.value
            # Convert datetime
            for key in ['entry_time', 'last_update', 'exit_time']:
                if data[key] is not None:
                    data[key] = data[key].isoformat()
            return data
        
        except Exception as e:
            logger.error(
                f"Error converting position {self.position_id} to dict: {e}",
                exc_info=True
            )
            raise
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dict"""
        try:
            # Convert status
            if isinstance(data.get('status'), str):
                data['status'] = PositionStatus(data['status'])
            
            # Convert datetime
            for key in ['entry_time', 'last_update', 'exit_time']:
                if data.get(key) and isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
            
            return cls(**data)
        
        except Exception as e:
            logger.error(f"Error creating position from dict: {e}", exc_info=True)
            raise
    
    def __repr__(self) -> str:
        """✅ NEW: Better string representation"""
        return (
            f"Position(id={self.position_id}, "
            f"symbol={self.symbol}, "
            f"entry=${self.entry_price:.6f}, "
            f"size={self.size_coin:.4f}, "
            f"tp=${self.tp_price:.6f}, "
            f"status={self.status.value})"
        )


@dataclass
class Bar5s:
    """
    5-second bar aggregation
    
    ✅ FIXED: Enhanced validation
    """
    bar_ts: int
    mid_price: float
    cvd: float
    trade_size_proxy: float
    order_imbalance: float
    total_bid_vol: float
    total_ask_vol: float
    num_trades: int
    data_quality: Literal["ok", "partial", "stale"]
    
    # Additional metrics
    spread_pct: float = 0.0
    weighted_imbalance: float = 0.5
    vpin: float = 0.5
    absorption_detected: bool = False
    
    def is_valid(self, strict: bool = True) -> bool:
        """
        Check if bar data is valid
        
        ✅ FIXED: Added strict mode
        
        Args:
            strict: If True, only "ok" is valid. If False, "partial" also valid
        """
        if strict:
            return (
                self.data_quality == "ok" and
                self.mid_price > 0 and
                self.num_trades > 0
            )
        else:
            return (
                self.data_quality in ["ok", "partial"] and
                self.mid_price > 0 and
                self.num_trades > 0
            )
    
    def __repr__(self) -> str:
        """✅ NEW: Better string representation"""
        return (
            f"Bar5s(ts={self.bar_ts}, "
            f"mid=${self.mid_price:.6f}, "
            f"cvd={self.cvd:.2f}, "
            f"quality={self.data_quality})"
        )


@dataclass
class OrderBookSnapshot:
    """
    Orderbook snapshot analysis
    
    ✅ No changes needed - already good
    """
    timestamp: int
    mid: float
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    
    total_bid_vol: float
    total_ask_vol: float
    depth_bid_5: float
    depth_ask_5: float
    
    imbalance_raw: float
    imbalance_ema: float
    weighted_imbalance: float
    
    quality: Literal["good", "poor", "stale"]


@dataclass
class TradeDecision:
    """
    Trading decision with full context
    
    ✅ No changes needed - already comprehensive
    """
    timestamp: int
    action: Literal["PLACE_BUY", "HOLD", "SKIP"]
    reason: str
    
    # Market data
    mid_price: float
    grid_candidate: float
    
    # Signals
    cvd_z: float
    ts_z: float
    order_imbalance: float
    confirm_count: int
    
    # Advanced indicators
    rsi: float = 50.0
    bb_position: float = 0.5
    near_support: bool = False
    vpin: float = 0.5
    smi: float = 0.5
    wyckoff_phase: str = "unknown"
    absorption: Dict = field(default_factory=dict)
    
    # Filters
    volatility: float = 0.0
    spread_pct: float = 0.0
    atr_ok: bool = True
    liquidity_ok: bool = True
    time_ok: bool = True
    cooldown_ok: bool = True
    
    # State
    active_positions: int = 0
    open_orders: int = 0


@dataclass
class HealthStatus:
    """
    System health status
    
    ✅ No changes needed - already comprehensive
    """
    timestamp: datetime
    
    # Connectivity
    api_connected: bool
    api_latency_ms: float
    api_failure_count: int
    
    # Data Quality
    data_fresh: bool
    last_data_timestamp: datetime
    data_staleness_sec: float
    
    # State
    positions_count: int
    positions_value_usd: float
    open_orders_count: int
    
    # Performance
    daily_pnl: float
    current_drawdown: float
    circuit_breaker_active: bool
    
    # Overall
    status: Literal["healthy", "degraded", "critical"]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return (
            self.status == "healthy" and
            self.api_connected and
            self.data_fresh and
            not self.circuit_breaker_active
        )


@dataclass
class BotState:
    """
    Complete bot state for persistence
    
    ✅ FIXED: Added version checking
    """
    version: str = "2.0.1"  # ✅ Updated version
    last_save: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    active_levels: List[float] = field(default_factory=list)
    
    # Orders
    open_orders: Dict[str, Dict] = field(default_factory=dict)
    
    # Performance
    total_trades: int = 0
    total_pnl: float = 0.0
    peak_balance: float = 0.0
    max_drawdown: float = 0.0
    
    # Circuit Breaker
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str = ""
    circuit_breaker_activation_time: Optional[datetime] = None
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        try:
            data = {
                'version': self.version,
                'last_save': self.last_save.isoformat(),
                'positions': {k: v.to_dict() for k, v in self.positions.items()},
                'active_levels': self.active_levels,
                'open_orders': self.open_orders,
                'total_trades': self.total_trades,
                'total_pnl': self.total_pnl,
                'peak_balance': self.peak_balance,
                'max_drawdown': self.max_drawdown,
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'circuit_breaker_active': self.circuit_breaker_active,
                'circuit_breaker_reason': self.circuit_breaker_reason,
                'circuit_breaker_activation_time': (
                    self.circuit_breaker_activation_time.isoformat() 
                    if self.circuit_breaker_activation_time else None
                )
            }
            return json.dumps(data, indent=2)
        
        except Exception as e:
            logger.error(f"Error serializing bot state: {e}", exc_info=True)
            raise
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BotState':
        """
        Deserialize from JSON
        
        ✅ FIXED: Added version compatibility check
        """
        try:
            data = json.loads(json_str)
            
            # ✅ NEW: Check version compatibility
            version = data.get('version', '1.0')
            if not version.startswith('2.'):
                logger.warning(
                    f"Loading state from older version: {version}. "
                    f"Current version: 2.0.1"
                )
            
            # Convert positions
            positions = {
                k: Position.from_dict(v) 
                for k, v in data.get('positions', {}).items()
            }
            
            # Convert datetime
            last_save = datetime.fromisoformat(data['last_save'])
            
            cb_time = data.get('circuit_breaker_activation_time')
            if cb_time:
                cb_time = datetime.fromisoformat(cb_time)
            
            return cls(
                version=data.get('version', '2.0.1'),
                last_save=last_save,
                positions=positions,
                active_levels=data.get('active_levels', []),
                open_orders=data.get('open_orders', {}),
                total_trades=data.get('total_trades', 0),
                total_pnl=data.get('total_pnl', 0.0),
                peak_balance=data.get('peak_balance', 0.0),
                max_drawdown=data.get('max_drawdown', 0.0),
                daily_pnl=data.get('daily_pnl', 0.0),
                consecutive_losses=data.get('consecutive_losses', 0),
                circuit_breaker_active=data.get('circuit_breaker_active', False),
                circuit_breaker_reason=data.get('circuit_breaker_reason', ''),
                circuit_breaker_activation_time=cb_time
            )
        
        except Exception as e:
            logger.error(f"Error deserializing bot state: {e}", exc_info=True)
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_position(position: Position) -> bool:
    """
    ✅ NEW: Validate position data
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if position.entry_price <= 0:
            logger.error(f"Invalid entry_price: {position.entry_price}")
            return False
        
        if position.size_coin <= 0:
            logger.error(f"Invalid size_coin: {position.size_coin}")
            return False
        
        if position.size_usd <= 0:
            logger.error(f"Invalid size_usd: {position.size_usd}")
            return False
        
        if position.tp_price <= position.entry_price:
            logger.warning(
                f"TP {position.tp_price} <= entry {position.entry_price}"
            )
        
        if position.sl_price and position.sl_price >= position.entry_price:
            logger.error(
                f"SL {position.sl_price} >= entry {position.entry_price}"
            )
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating position: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Test position creation and validation
    print("Testing Position model...")
    
    try:
        # Valid position
        pos = Position(
            position_id="test_001",
            symbol="ADA/USDT",
            entry_price=0.5000,
            size_coin=100.0,
            size_usd=50.0,
            tp_price=0.5075
        )
        print(f"✅ Valid position created: {pos}")
        
        # Test P&L update
        pos.update_pnl(0.5050)
        print(f"✅ P&L updated: ${pos.current_pnl:.2f}")
        
        # Invalid position (should raise error)
        try:
            invalid_pos = Position(
                position_id="test_002",
                symbol="ADA/USDT",
                entry_price=0,  # Invalid!
                size_coin=100.0,
                size_usd=50.0,
                tp_price=0.5075
            )
        except ValueError as e:
            print(f"✅ Caught invalid position: {e}")
    
    except Exception as e:
        print(f"✗ Test failed: {e}")