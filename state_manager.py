#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Persistence and Recovery
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
import threading

from models import Position, BotState


class StatePersistence:
    """
    Persistent state storage with SQLite
    Handles crash recovery and state restoration
    """
    
    def __init__(self, db_path: str = "state/bot_state.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size_coin REAL NOT NULL,
                    size_usd REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    tp_order_id TEXT,
                    sl_price REAL,
                    sl_type TEXT,
                    sl_order_id TEXT,
                    entry_time TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    exit_time TEXT,
                    entry_reason TEXT,
                    exit_reason TEXT,
                    status TEXT NOT NULL,
                    market_phase TEXT,
                    volatility_at_entry REAL,
                    cvd_z_at_entry REAL,
                    data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status 
                ON positions(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp 
                ON state_snapshots(timestamp DESC)
            """)
            
            conn.commit()
    
    def save_position(self, position: Position):
        """Save or update position"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                data = position.to_dict()
                
                conn.execute("""
                    INSERT OR REPLACE INTO positions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    position.position_id,
                    position.symbol,
                    position.entry_price,
                    position.size_coin,
                    position.size_usd,
                    position.tp_price,
                    position.tp_order_id,
                    position.sl_price,
                    position.sl_type,
                    position.sl_order_id,
                    position.entry_time.isoformat(),
                    position.last_update.isoformat(),
                    position.exit_time.isoformat() if position.exit_time else None,
                    position.entry_reason,
                    position.exit_reason,
                    position.status.value,
                    position.market_phase,
                    position.volatility_at_entry,
                    position.cvd_z_at_entry,
                    json.dumps(data)
                ))
                conn.commit()
    
    def load_position(self, position_id: str) -> Optional[Position]:
        """Load single position"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT data FROM positions WHERE position_id = ?",
                (position_id,)
            )
            row = cursor.fetchone()
            
            if row:
                data = json.loads(row['data'])
                return Position.from_dict(data)
        
        return None
    
    def load_open_positions(self) -> Dict[str, Position]:
        """Load all open positions"""
        positions = {}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT data FROM positions WHERE status = 'open'"
            )
            
            for row in cursor:
                data = json.loads(row['data'])
                pos = Position.from_dict(data)
                positions[pos.position_id] = pos
        
        return positions
    
    def close_position(self, position_id: str, exit_reason: str):
        """Mark position as closed"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE positions 
                    SET status = 'closed',
                        exit_time = ?,
                        exit_reason = ?,
                        last_update = ?
                    WHERE position_id = ?
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    exit_reason,
                    datetime.now(timezone.utc).isoformat(),
                    position_id
                ))
                conn.commit()
    
    def save_state_snapshot(self, state: BotState):
        """Save complete state snapshot"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO state_snapshots (timestamp, state_json)
                    VALUES (?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    state.to_json()
                ))
                conn.commit()
                
                # Keep only last 100 snapshots
                conn.execute("""
                    DELETE FROM state_snapshots
                    WHERE id NOT IN (
                        SELECT id FROM state_snapshots
                        ORDER BY timestamp DESC
                        LIMIT 100
                    )
                """)
                conn.commit()
    
    def load_latest_state(self) -> Optional[BotState]:
        """Load latest state snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT state_json FROM state_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                return BotState.from_json(row['state_json'])
        
        return None
    
    def checkpoint(self, checkpoint_type: str, data: Dict):
        """Save checkpoint for recovery"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO checkpoints (timestamp, checkpoint_type, data)
                    VALUES (?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    checkpoint_type,
                    json.dumps(data)
                ))
                conn.commit()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) as total FROM positions")
            stats['total_positions'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) as open FROM positions WHERE status = 'open'")
            stats['open_positions'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) as closed FROM positions WHERE status = 'closed'")
            stats['closed_positions'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) as snapshots FROM state_snapshots")
            stats['snapshots'] = cursor.fetchone()[0]
            
            return stats
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old closed positions"""
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM positions
                    WHERE status = 'closed'
                    AND exit_time < ?
                """, (cutoff_iso,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                return deleted


class PositionReconciler:
    """
    [OK] ENHANCED: Reconcile bot state with exchange reality
    Fix discrepancies and report differences

    Improvements:
    - Rebuild tracking maps from scratch (like template bot)
    - Price tolerance matching for grid levels
    - Sync active_levels with exchange state
    - Handle filled orders detection
    """

    def __init__(self, exchange, symbol: str, state_manager: StatePersistence):
        self.exchange = exchange
        self.symbol = symbol
        self.state_manager = state_manager

        # [OK] Get market info for tolerance calculation
        self.market_info = exchange.markets.get(symbol, {})
        self.tick_size = float(self.market_info.get('precision', {}).get('price', 1e-8))

    def _price_tolerance(self, price: float) -> float:
        """
        [OK] NEW: Calculate price tolerance for matching
        Use max of percentage tolerance and tick size
        """
        pct_tol = price * 0.0015  # 0.15%
        tick_tol = self.tick_size * 2
        return max(pct_tol, tick_tol)

    def reconcile(self, engine=None, grid_levels: list = None) -> Dict:
        """
        [OK] ENHANCED: Compare local state with exchange and fix issues

        Args:
            engine: Signal engine (to rebuild active_levels)
            grid_levels: List of grid levels for matching

        Returns:
            Reconciliation report
        """
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'discrepancies': [],
            'fixes_applied': [],
            'positions_synced': 0,
            'active_levels_rebuilt': False
        }
        
        try:
            # Get local positions
            local_positions = self.state_manager.load_open_positions()

            # Get exchange open orders
            exchange_orders = self.exchange.fetch_open_orders(self.symbol)
            exchange_order_ids = {o['id'] for o in exchange_orders}

            # [OK] ENHANCED: Rebuild active_levels from exchange (like template bot)
            if engine is not None:
                new_active_levels = self._rebuild_active_levels(
                    exchange_orders,
                    grid_levels or []
                )

                if new_active_levels != engine.active_levels:
                    old_count = len(engine.active_levels)
                    engine.active_levels = new_active_levels

                    report['active_levels_rebuilt'] = True
                    report['fixes_applied'].append({
                        'fix': 'rebuilt_active_levels',
                        'old_count': old_count,
                        'new_count': len(new_active_levels)
                    })

            # Check each local position
            for pos_id, position in local_positions.items():
                # Check if TP order exists
                if position.tp_order_id:
                    if position.tp_order_id not in exchange_order_ids:
                        report['discrepancies'].append({
                            'type': 'missing_tp_order',
                            'position_id': pos_id,
                            'tp_order_id': position.tp_order_id
                        })

                        # [OK] ENHANCED: Check if order was filled
                        filled = self._check_if_filled(position.tp_order_id)

                        if filled:
                            # Order was filled - close position
                            report['fixes_applied'].append({
                                'fix': 'detected_filled_tp',
                                'position_id': pos_id,
                                'reason': 'tp_filled'
                            })
                            # Position should be closed by main loop
                        else:
                            # Try to recreate TP order
                            try:
                                self._recreate_tp_order(position)
                                report['fixes_applied'].append({
                                    'fix': 'recreated_tp_order',
                                    'position_id': pos_id
                                })
                            except Exception as e:
                                report['discrepancies'].append({
                                    'type': 'failed_tp_recreation',
                                    'position_id': pos_id,
                                    'error': str(e)
                                })

                # Check if SL order exists (if applicable)
                if position.sl_order_id:
                    if position.sl_order_id not in exchange_order_ids:
                        report['discrepancies'].append({
                            'type': 'missing_sl_order',
                            'position_id': pos_id,
                            'sl_order_id': position.sl_order_id
                        })

                        # [OK] ENHANCED: Check if SL was filled
                        filled = self._check_if_filled(position.sl_order_id)

                        if filled:
                            report['fixes_applied'].append({
                                'fix': 'detected_filled_sl',
                                'position_id': pos_id,
                                'reason': 'sl_filled'
                            })
            
            report['positions_synced'] = len(local_positions)
            
            # Check for orphaned exchange orders
            local_order_ids = set()
            for pos in local_positions.values():
                if pos.tp_order_id:
                    local_order_ids.add(pos.tp_order_id)
                if pos.sl_order_id:
                    local_order_ids.add(pos.sl_order_id)
            
            orphaned = exchange_order_ids - local_order_ids
            if orphaned:
                report['discrepancies'].append({
                    'type': 'orphaned_orders',
                    'count': len(orphaned),
                    'order_ids': list(orphaned)[:10]  # [OK] Limit to 10 for logging
                })

                # [OK] ENHANCED: Optionally cancel orphaned orders
                # (Commented out for safety - enable if needed)
                # for order_id in list(orphaned)[:5]:  # Cancel up to 5
                #     try:
                #         self.exchange.cancel_order(order_id, self.symbol)
                #         report['fixes_applied'].append({
                #             'fix': 'cancelled_orphaned_order',
                #             'order_id': order_id
                #         })
                #     except Exception:
                #         pass
        
        except Exception as e:
            report['error'] = str(e)
        
        return report
    
    def _recreate_tp_order(self, position: Position):
        """Recreate missing TP order"""
        # Round quantity to exchange precision
        from decimal import Decimal, ROUND_FLOOR
        
        market_info = self.exchange.markets.get(position.symbol, {})
        amount_precision = market_info.get('precision', {}).get('amount', 8)
        
        step = 10 ** (-amount_precision)
        rounded_size = float(Decimal(str(position.size_coin)).quantize(
            Decimal(str(step)), rounding=ROUND_FLOOR
        ))
        
        # Create TP order
        order = self.exchange.create_limit_sell_order(
            position.symbol,
            rounded_size,
            position.tp_price
        )
        
        # Update position
        position.tp_order_id = order['id']
        self.state_manager.save_position(position)

    def _rebuild_active_levels(self, exchange_orders: list, grid_levels: list) -> set:
        """
        [OK] NEW: Rebuild active_levels from exchange orders

        Match exchange orders to grid levels by price tolerance

        Args:
            exchange_orders: List of open orders from exchange
            grid_levels: List of configured grid levels

        Returns:
            Set of active grid levels
        """
        active = set()

        for order in exchange_orders:
            price = float(order.get('price', 0))
            side = (order.get('side') or '').lower()

            if price <= 0:
                continue

            # Match to grid level
            matched_level = self._match_to_grid_level(price, side, grid_levels)

            if matched_level:
                active.add(matched_level)

        return active

    def _match_to_grid_level(self, order_price: float, side: str,
                            grid_levels: list) -> Optional[float]:
        """
        [OK] NEW: Match order price to grid level

        For BUY orders: match directly to buy_price
        For SELL orders: match to TP price, return corresponding buy_price

        Args:
            order_price: Order price from exchange
            side: 'buy' or 'sell'
            grid_levels: List of grid buy prices

        Returns:
            Matched grid level (buy_price) or None
        """
        if not grid_levels:
            return None

        tolerance = self._price_tolerance(order_price)

        # For BUY orders: direct match
        if side == 'buy':
            for level in grid_levels:
                if abs(level - order_price) <= tolerance:
                    return level

        # For SELL orders: estimate buy_price from TP
        # Assume TP = buy_price * (1 + tp_pct)
        # Therefore: buy_price ~= order_price / (1 + tp_pct)
        else:
            # Use a default TP of 1.5% if not available
            tp_factor = 1.015  # Could be made configurable
            estimated_buy = order_price / tp_factor

            for level in grid_levels:
                if abs(level - estimated_buy) <= self._price_tolerance(level):
                    return level

        return None

    def _check_if_filled(self, order_id: str) -> bool:
        """
        [OK] NEW: Check if order was filled

        Args:
            order_id: Order ID to check

        Returns:
            True if order was filled, False otherwise
        """
        try:
            order = self.exchange.fetch_order(order_id, self.symbol)
            status = order.get('status', '').lower()

            return status in ['filled', 'closed']

        except Exception:
            # If we can't fetch order, assume it wasn't filled
            return False


class StateRecovery:
    """Handle crash recovery and state restoration"""
    
    def __init__(self, state_manager: StatePersistence):
        self.state_manager = state_manager
    
    def recover_from_crash(self) -> Optional[BotState]:
        """
        Recover state after crash
        Returns restored state or None
        """
        print("\n" + "="*60)
        print("CRASH RECOVERY")
        print("="*60)
        
        # Load latest state
        state = self.state_manager.load_latest_state()
        
        if state:
            print(f"Found state snapshot from: {state.last_save}")
            print(f"Open positions: {len(state.positions)}")
            print(f"Active levels: {len(state.active_levels)}")
            print(f"Circuit breaker: {'ACTIVE' if state.circuit_breaker_active else 'OK'}")
            
            # Validate state
            validation = self._validate_state(state)

            if validation['valid']:
                print("[OK] State validation passed")
                return state
            else:
                print("[ERROR] State validation failed:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
                
                # Try to repair
                repaired = self._repair_state(state, validation['issues'])
                if repaired:
                    print("[OK] State repaired successfully")
                    return repaired
        
        print("No valid state found for recovery")
        return None
    
    def _validate_state(self, state: BotState) -> Dict:
        """Validate state integrity"""
        issues = []
        
        # Check positions
        for pos_id, pos in state.positions.items():
            if pos.entry_price <= 0:
                issues.append(f"Invalid entry price for {pos_id}")
            
            if pos.size_coin <= 0:
                issues.append(f"Invalid size for {pos_id}")
            
            if pos.status.value not in ['open', 'closing', 'closed']:
                issues.append(f"Invalid status for {pos_id}")
        
        # Check active levels
        if any(lvl <= 0 for lvl in state.active_levels):
            issues.append("Invalid active levels found")
        
        # Check circuit breaker state
        if state.circuit_breaker_active and not state.circuit_breaker_reason:
            issues.append("Circuit breaker active but no reason")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _repair_state(self, state: BotState, issues: List[str]) -> Optional[BotState]:
        """Attempt to repair corrupted state"""
        try:
            # Remove invalid positions
            valid_positions = {
                k: v for k, v in state.positions.items()
                if v.entry_price > 0 and v.size_coin > 0
            }
            state.positions = valid_positions
            
            # Clean active levels
            state.active_levels = [lvl for lvl in state.active_levels if lvl > 0]
            
            # Reset circuit breaker if reason missing
            if state.circuit_breaker_active and not state.circuit_breaker_reason:
                state.circuit_breaker_active = False
                state.circuit_breaker_activation_time = None
            
            return state
        
        except Exception as e:
            print(f"State repair failed: {e}")
            return None