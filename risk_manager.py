#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Risk Management System
"""

import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import Position
from config import TradingConfig


class StopLossManager:
    """Advanced Stop Loss Management"""

    def __init__(self, config: TradingConfig, executor=None):
        self.cfg = config
        self.active_stops: Dict[str, Dict] = {}
        self.executor = executor  # [OK] NEW: Reference to ExecutionLayer
    
    def create_stop_loss(self, position: Position, market_phase: str, 
                        volatility: float) -> Optional[float]:
        """Create appropriate stop loss for position"""
        if not self.cfg.safety.enable_stop_loss:
            return None
        
        sl_price = None
        entry = position.entry_price
        
        if self.cfg.safety.stop_loss_type == "fixed":
            sl_price = entry * (1 - self.cfg.safety.fixed_sl_pct)
        
        elif self.cfg.safety.stop_loss_type == "atr":
            vol = max(volatility, 0.001)
            atr_distance = vol * self.cfg.safety.atr_sl_multiplier

            # [OK] ENHANCED: Adjust minimum SL based on volatility regime
            if volatility > 0.03:  # High volatility (>3%)
                min_sl = 0.015  # 1.5% minimum
            elif volatility > 0.02:  # Medium volatility
                min_sl = 0.012  # 1.2% minimum
            else:  # Low volatility
                min_sl = self.cfg.safety.min_sl_distance_pct  # 1% minimum

            # Enforce minimum SL distance
            atr_distance = max(atr_distance, min_sl)

            # Tighter in bearish phases (but still respect minimum)
            if market_phase in ['distribution', 'markdown']:
                atr_distance = max(atr_distance * 0.9, min_sl)

            sl_price = entry * (1 - atr_distance)
        
        elif self.cfg.safety.stop_loss_type == "trailing":
            # Start with fixed, will trail later
            sl_price = entry * (1 - self.cfg.safety.fixed_sl_pct)
        
        if sl_price:
            self.active_stops[position.position_id] = {
                'entry_price': entry,
                'current_sl': sl_price,
                'highest_price': entry,
                'type': self.cfg.safety.stop_loss_type,
                'activated': False
            }
            
            # Update position
            position.sl_price = sl_price
            position.sl_type = self.cfg.safety.stop_loss_type
        
        return sl_price
    
    def update_trailing_stop(self, position: Position, current_price: float, engine=None):
        """Update trailing stop if applicable"""
        pos_id = position.position_id

        if pos_id not in self.active_stops:
            return

        stop_info = self.active_stops[pos_id]

        if stop_info['type'] != 'trailing':
            return

        # Update highest price
        if current_price > stop_info['highest_price']:
            stop_info['highest_price'] = current_price

        # Check if trailing should activate
        profit_pct = (current_price - stop_info['entry_price']) / stop_info['entry_price']

        if profit_pct >= self.cfg.safety.trailing_sl_activation:
            stop_info['activated'] = True

            # [OK] FIX: Use config value (default 0.7%)
            new_sl = stop_info['highest_price'] * (1 - self.cfg.safety.trailing_sl_distance)

            # Only move stop up
            if new_sl > stop_info['current_sl']:
                old_sl = stop_info['current_sl']
                stop_info['current_sl'] = new_sl
                position.sl_price = new_sl

                # [OK] NEW: Log trailing stop updates
                print(f"  [INFO] Trailing SL: ${new_sl:.6f} (profit locked: {(new_sl/stop_info['entry_price']-1)*100:.2f}%)")

                # [OK] ENHANCED: Update SL order on exchange with verification
                if self.executor and position.sl_order_id and not self.cfg.dry_run:
                    try:
                        # Cancel old SL order
                        self.executor.exchange.cancel_order(position.sl_order_id, position.symbol)
                        print(f"  [OK] Cancelled old SL order #{position.sl_order_id}")

                        # Place new SL order
                        new_sl_order = self.executor.place_stop_loss_order(position, engine)
                        if new_sl_order:
                            new_sl_order_id = new_sl_order.get('id')

                            # [OK] VERIFY: Check order was actually placed
                            try:
                                import time
                                time.sleep(0.5)  # Brief delay for order to propagate
                                verify_order = self.executor.exchange.fetch_order(new_sl_order_id, position.symbol)

                                if verify_order and verify_order.get('status') in ['open', 'new']:
                                    position.sl_order_id = new_sl_order_id
                                    print(f"  [OK] New SL order verified: #{new_sl_order_id} @ ${new_sl:.6f}")
                                else:
                                    print(f"  [WARNING] SL order verification failed, tracking locally only")
                                    position.sl_order_id = None

                            except Exception as verify_error:
                                print(f"  [WARNING] SL order verification error: {verify_error}, assuming success")
                                position.sl_order_id = new_sl_order_id
                        else:
                            print(f"  [WARNING] Failed to place new SL order, tracking locally only")
                            position.sl_order_id = None

                    except Exception as e:
                        print(f"  [WARNING] Error updating SL order on exchange: {e}")
                        position.sl_order_id = None

    def check_stop_triggered(self, position: Position, current_price: float) -> bool:
        """Check if stop loss triggered"""
        pos_id = position.position_id
        
        if pos_id not in self.active_stops:
            return False
        
        stop_info = self.active_stops[pos_id]
        
        # Conservative: use low of bar or current price
        return current_price <= stop_info['current_sl']
    
    def remove_stop(self, position_id: str):
        """Remove stop when position closed"""
        if position_id in self.active_stops:
            del self.active_stops[position_id]


class CircuitBreaker:
    """System-wide circuit breaker for risk management"""
    
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.active = False
        self.activation_time: Optional[datetime] = None
        self.activation_reason: str = ""
        
        # Daily tracking
        self.daily_pnl: float = 0.0
        self.daily_reset_time: datetime = self._get_daily_reset_time()
        
        # Loss streak tracking
        self.consecutive_losses: int = 0
        
        # Emergency conditions
        self.last_balance_check: float = 0.0
        self.api_failure_count: int = 0
        self.last_price: float = 0.0
        self.price_history = deque(maxlen=60)  # 5 minutes of data
    
    def _get_daily_reset_time(self) -> datetime:
        """Get daily reset time (00:00 UTC)"""
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def reset_daily_stats(self):
        """Reset daily statistics at midnight UTC"""
        now = datetime.now(timezone.utc)
        
        if now.date() > self.daily_reset_time.date():
            print(f"\nResetting daily stats. Previous daily P&L: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_reset_time = self._get_daily_reset_time()
    
    def check_and_activate(self, current_state: Dict) -> Tuple[bool, str]:
        """
        Check all conditions and activate if needed
        Returns: (should_activate, reason)
        """
        if not self.cfg.safety.enable_circuit_breaker:
            return False, ""
        
        # Check if already active and cooldown not expired
        if self.active:
            if self.activation_time:
                hours_passed = (
                    datetime.now(timezone.utc) - self.activation_time
                ).total_seconds() / 3600
                
                if hours_passed < self.cfg.safety.circuit_breaker_cooldown_hours:
                    return True, self.activation_reason
                else:
                    self.deactivate()
        
        # Reset daily stats if needed
        self.reset_daily_stats()
        
        # Check daily loss limit
        if self.daily_pnl < -self.cfg.budget_usdt * self.cfg.safety.daily_loss_limit_pct:
            return self._activate(
                f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}"
            )
        
        # Check consecutive losses
        if self.consecutive_losses >= self.cfg.safety.max_consecutive_losses:
            return self._activate(
                f"Max consecutive losses: {self.consecutive_losses}"
            )
        
        # Check max drawdown
        current_dd = current_state.get('current_drawdown', 0)
        if current_dd > self.cfg.safety.max_drawdown_stop_pct:
            return self._activate(
                f"Max drawdown exceeded: {current_dd*100:.2f}%"
            )
        
        # Check emergency conditions
        if self.cfg.safety.enable_emergency_exit:
            emergency_check = self._check_emergency_conditions(current_state)
            if emergency_check[0]:
                return self._activate(f"Emergency: {emergency_check[1]}")
        
        return False, ""
    
    # Line 175: _check_emergency_conditions()
    def _check_emergency_conditions(self, state: Dict):
        # [OK] FIX: Track exact timestamps instead of using deque indices
        current_price = state.get('current_price', 0)
        current_time = time.time()
        
        if current_price > 0:
            self.price_history.append((current_time, current_price))
            
            # Remove old data (>5 minutes)
            while self.price_history and current_time - self.price_history[0][0] > 300:
                self.price_history.popleft()
            
            if len(self.price_history) >= 2:
                time_5min_ago, price_5min_ago = self.price_history[0]
                actual_duration = current_time - time_5min_ago
                
                # Check if we have 5 minutes of data
                if actual_duration >= 300:  # 5 minutes
                    price_change = (current_price - price_5min_ago) / price_5min_ago
                    
                    if price_change < -self.cfg.safety.flash_crash_threshold_pct:
                        return True, "flash_crash"
        
        # API failure detection
        if state.get('data_quality') == 'stale':
            self.api_failure_count += 1
            if self.api_failure_count >= 60:  # 5 minutes of stale data
                return True, "api_failure"
        else:
            self.api_failure_count = 0
        
        # Balance mismatch detection
        current_balance = state.get('current_balance', 0)
        
        if current_balance > 0:
            if self.last_balance_check > 0:
                balance_diff = abs(current_balance - self.last_balance_check)
                balance_diff_pct = balance_diff / self.last_balance_check
                
                if balance_diff_pct > 0.10:  # 10% mismatch
                    return True, "balance_mismatch"
            
            self.last_balance_check = current_balance
        
        return False, ""
    
    def _activate(self, reason: str) -> Tuple[bool, str]:
        """Activate circuit breaker"""
        self.active = True
        self.activation_time = datetime.now(timezone.utc)
        self.activation_reason = reason

        print(f"\n{'='*60}")
        print(f"[WARNING] CIRCUIT BREAKER ACTIVATED [WARNING]")
        print(f"{'='*60}")
        print(f"Reason: {reason}")
        print(f"Time: {self.activation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Cooldown: {self.cfg.safety.circuit_breaker_cooldown_hours} hours")
        print(f"{'='*60}\n")
        
        return True, reason
    
    def deactivate(self):
        """Deactivate circuit breaker"""
        self.active = False
        self.activation_time = None
        self.activation_reason = ""
        print(f"\n[OK] Circuit breaker deactivated\n")
    
    def record_trade(self, pnl: float):
        """Record trade result for tracking"""
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


class PositionSizer:
    """Advanced position sizing"""
    
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.trade_history = deque(maxlen=100)
    
    def calculate_size(self, 
                      available_balance: float,
                      confidence: float = 1.0,
                      volatility: float = 0.02,
                      current_drawdown: float = 0.0) -> float:
        """
        Calculate optimal position size
        
        Args:
            available_balance: Available capital
            confidence: Signal confidence (0-1)
            volatility: Current market volatility
            current_drawdown: Current drawdown percentage
        
        Returns:
            Position size in USD
        """
        # Base size
        base_size = available_balance / self.cfg.max_grid_levels
        
        # Apply sizing method
        if self.cfg.position_sizing == "fixed":
            size = base_size
        
        elif self.cfg.position_sizing == "volatility":
            # Adjust for volatility (inverse relationship)
            target_vol = 0.02  # 2% target volatility
            vol = max(volatility, 0.001)  # Prevent division by zero
            vol_mult = np.clip(target_vol / vol, 0.5, 1.5)
            size = base_size * vol_mult
        
        elif self.cfg.position_sizing == "kelly":
            kelly = self._calculate_kelly_fraction()
            size = available_balance * kelly / self.cfg.max_grid_levels
        
        else:
            size = base_size
        
        # Apply confidence adjustment
        size *= np.clip(confidence, 0.5, 1.0)
        
        # Apply drawdown reduction
        if current_drawdown > 0.05:
            size *= 0.7  # Reduce 30%
        elif current_drawdown > 0.03:
            size *= 0.85  # Reduce 15%
        
        # Constraints
        min_size = 20.0  # Minimum notional
        max_size = available_balance * self.cfg.safety.max_position_size_pct
        
        return float(np.clip(size, min_size, max_size))
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly fraction from trade history"""
        if len(self.trade_history) < 20:
            return 0.1  # Conservative default
        
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        if not wins or not losses:
            return 0.1
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.1
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        # Use fractional Kelly (25% of full Kelly for safety)
        kelly_fractional = np.clip(kelly * 0.25, 0.02, 0.25)
        
        return float(kelly_fractional)
    
    def record_trade(self, pnl: float, size: float):
        """Record trade for Kelly calculation"""
        self.trade_history.append({
            'pnl': pnl,
            'size': size,
            'timestamp': datetime.now(timezone.utc)
        })


class PortfolioRiskManager:
    """Manage portfolio-level risk across multiple pairs"""
    
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.pair_allocations: Dict[str, float] = {}
        self.pair_correlations: Dict[Tuple[str, str], float] = {}
    
    def set_allocation(self, symbol: str, allocation_pct: float):
        """Set allocation for a trading pair"""
        if allocation_pct < 0 or allocation_pct > 1:
            raise ValueError("Allocation must be between 0 and 1")
        
        self.pair_allocations[symbol] = allocation_pct
    
    def check_total_exposure(self, positions: Dict[str, Position]) -> bool:
        """Check if total exposure is within limits"""
        total_exposure = sum(p.size_usd for p in positions.values())
        max_exposure = self.cfg.budget_usdt * self.cfg.safety.max_total_exposure_pct
        
        return total_exposure <= max_exposure
    
    def rebalance_allocations(self, performance: Dict[str, float]):
        """
        Rebalance allocations based on performance
        Increase winners, decrease losers
        """
        if not performance:
            return
        
        # Calculate new allocations
        new_allocs = {}
        
        for symbol, alloc in self.pair_allocations.items():
            perf = performance.get(symbol, 0)
            
            if perf > 0.02:  # +2% performance
                # Increase allocation by 10%
                new_allocs[symbol] = min(alloc * 1.1, 0.5)  # Max 50% per pair
            elif perf < -0.02:  # -2% performance
                # Decrease allocation by 10%
                new_allocs[symbol] = max(alloc * 0.9, 0.05)  # Min 5% per pair
            else:
                new_allocs[symbol] = alloc
        
        # Normalize to ensure total = 1.0
        total = sum(new_allocs.values())
        if total > 0:
            for symbol in new_allocs:
                new_allocs[symbol] /= total
        
        self.pair_allocations = new_allocs
    
    def calculate_portfolio_var(self, positions: Dict[str, Position],
                                volatilities: Dict[str, float],
                                confidence_level: float = 0.95) -> float:
        """
        Calculate Portfolio Value at Risk (VaR)
        
        Returns: VaR in USD
        """
        if not positions:
            return 0.0
        
        # Get position values and volatilities
        values = []
        vols = []
        
        for pos in positions.values():
            values.append(pos.size_usd)
            vols.append(volatilities.get(pos.symbol, 0.02))
        
        values = np.array(values)
        vols = np.array(vols)
        
        # Simple VaR calculation (assuming independence)
        # VaR = Portfolio Value * Volatility * Z-score
        portfolio_value = values.sum()
        portfolio_vol = np.sqrt((values * vols).sum() / portfolio_value)
        
        # Z-score for 95% confidence
        z_score = 1.645 if confidence_level == 0.95 else 2.326  # 99%
        
        var = portfolio_value * portfolio_vol * z_score
        
        return float(var)