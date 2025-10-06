#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Grid Trading System V2.0 - Main Trader
Complete production-grade trading bot
"""

import time
import signal
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Optional
import csv
import traceback

import ccxt

from config import TradingConfig, ConfigManager
from models import Position, BotState, PositionStatus
from state_manager import StatePersistence, PositionReconciler, StateRecovery
from health_monitor import HealthMonitor
from risk_manager import StopLossManager, CircuitBreaker, PositionSizer
from signal_engine import SignalEngine
from execution import ExecutionLayer
from utils import DataFetcher, Aggregator5s, GridPlanner, now_ms
from notifications import TelegramNotifier


class UltraGridTrader:
    """
    Ultra Grid Order Flow Trader V2.0
    Complete production system with all safety features
    """
    
    def __init__(self, config: TradingConfig, api_key: str = None, api_secret: str = None):
        self.cfg = config
        
        print("\n" + "="*60)
        print("ULTRA GRID TRADING SYSTEM V2.0")
        print("="*60)
        
        # Initialize exchange
        self.fetcher = DataFetcher(config.symbol, "binance", api_key, api_secret)
        
        # Load grid
        self.grid_planner = GridPlanner(config)
        grid_levels = self._load_grid()
        
        # Core components
        self.engine = SignalEngine(config, grid_levels)
        self.executor = ExecutionLayer(self.fetcher.exchange, config.symbol, config)
        self.aggregator = Aggregator5s(config.bar_interval_ms)
        
        # Risk management
        self.stop_loss_mgr = StopLossManager(config, self.executor)
        self.circuit_breaker = CircuitBreaker(config)
        self.position_sizer = PositionSizer(config)

        # State management
        self.state_manager = StatePersistence(config.state_db)
        self.reconciler = PositionReconciler(
            self.fetcher.exchange, config.symbol, self.state_manager
        )
        self.state_recovery = StateRecovery(self.state_manager)
        
        # Health monitoring
        self.health_monitor = HealthMonitor(config)
        
        # Notifications
        self.notifier = TelegramNotifier(config)
        
        # State
        self.positions: Dict[str, Position] = {}
        self.running = False
        self.last_resync = time.time()
        self.last_health_check = time.time()
        self.last_checkpoint = time.time()
        self.last_reconcile = time.time()  # [OK] FIXED: Use time.time() instead of 0.0
        self._last_mid = 0.0
        
        # Config manager for hot reload
        if os.path.exists('config.yaml'):
            self.config_manager = ConfigManager('config.yaml')
        else:
            self.config_manager = None
        
        # Logging
        self._init_logging()
        
        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        
        # Attempt recovery
        self._attempt_recovery()
        
        print(f"Symbol: {config.symbol}")
        print(f"Mode: {'DRY RUN' if config.dry_run else 'WARNING: LIVE TRADING'}")
        print(f"Budget: ${config.budget_usdt}")
        print(f"Grid Levels: {len(grid_levels)}")
        print(f"State DB: {config.state_db}")
        print("="*60 + "\n")
    
    def _load_grid(self) -> list:
        """Load grid levels"""
        if self.cfg.grid_source == "csv":
            levels = self.grid_planner.load_from_csv(self.cfg.grid_csv)
            if not levels:
                print("WARNING: No grid levels loaded, creating fallback")
                levels = self._create_fallback_grid()
        else:
            levels = self._create_fallback_grid()
        
        return levels
    
    def _create_fallback_grid(self) -> list:
        """Create fallback grid when CSV not available"""
        try:
            ticker = self.fetcher.exchange.fetch_ticker(self.cfg.symbol)
            current_price = ticker['last']
        except:
            current_price = 1.0
        
        levels = []
        for i in range(1, self.cfg.max_grid_levels + 1):
            level = current_price * (1 - 0.015 * i)
            levels.append(level)
        
        return levels
    
    def _attempt_recovery(self):
        """Attempt to recover from crash"""
        recovered_state = self.state_recovery.recover_from_crash()
        
        if recovered_state:
            self.positions = recovered_state.positions
            self.engine.active_levels = set(recovered_state.active_levels)
            self.circuit_breaker.daily_pnl = recovered_state.daily_pnl
            self.circuit_breaker.consecutive_losses = recovered_state.consecutive_losses
            self.circuit_breaker.active = recovered_state.circuit_breaker_active
            self.circuit_breaker.activation_reason = recovered_state.circuit_breaker_reason
            self.circuit_breaker.activation_time = recovered_state.circuit_breaker_activation_time
            
            print(f"[OK] Recovered {len(self.positions)} positions from state")
    
    def _init_logging(self):
        """Initialize CSV logging"""
        os.makedirs(os.path.dirname(self.cfg.log_csv) or '.', exist_ok=True)
        
        if not os.path.exists(self.cfg.log_csv):
            with open(self.cfg.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "bar_ts_ms", "bar_time_utc", "data_quality",
                    "mid_price", "cvd", "cvd_z", "ts_proxy", "ts_z",
                    "rsi", "bb_position", "near_support",
                    "confirm_count", "signal_raw", "signal_confirmed",
                    "grid_candidate", "action", "reason",
                    "bars_total", "active_levels", "open_positions",
                    "volatility", "vpin", "smi", "wyckoff_phase",
                    "spread_pct", "circuit_breaker_active"
                ])
        
        self.csv_file = open(self.cfg.log_csv, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
    
    def _shutdown_handler(self, signum, frame):
        """Handle graceful shutdown"""
        print("\n\n" + "="*60)
        print("SHUTDOWN SIGNAL RECEIVED")
        print("="*60)
        
        self.running = False
        self._save_state()
        
        response = input("Close all positions before exit? (yes/no): ")
        if response.lower() == 'yes':
            self._emergency_exit_all("user_shutdown")
        
        sys.exit(0)
    
    def run(self):
        """Main run loop"""
        print("[INFO] STARTING ULTRA GRID TRADER V2.0\n")
        
        if not self.cfg.dry_run:
            self.executor.prelock_existing_orders(self.engine, self.grid_planner)
        
        self.notifier.send_message(
            f"<b>Bot Started</b>\n"
            f"Symbol: {self.cfg.symbol}\n"
            f"Budget: ${self.cfg.budget_usdt}\n"
            f"Mode: {'DRY RUN' if self.cfg.dry_run else 'LIVE'}\n"
            f"Positions: {len(self.positions)}"
        )
        
        self.running = True
        iteration = 0
        
        try:
            while self.running:
                self._run_cycle()
                time.sleep(0.1)
                
                iteration += 1
                
                if iteration % 600 == 0:
                    self._periodic_tasks()
        
        except KeyboardInterrupt:
            print("\n\n[INFO] STOPPING SYSTEM...")
            self.running = False
        
        except Exception as e:
            print(f"\n\n[ERROR] CRITICAL ERROR: {e}")
            traceback.print_exc()
            self.notifier.send_message(f"<b>CRITICAL ERROR</b>\n{str(e)}")
            self.running = False
        
        finally:
            self._cleanup()
    
    def _run_cycle(self):
        """One iteration of main loop"""
        t0 = now_ms()
        
        # Hot reload config
        if self.config_manager:
            if self.config_manager.check_and_reload():
                self.cfg = self.config_manager.config
        
        # Maybe reload grid
        new_levels = self.grid_planner.maybe_reload()
        if new_levels:
            print(f"\n[INFO] Grid reload detected: {len(new_levels)} new levels")

            # [OK] FIXED: Clear old active levels and reconcile with open orders
            old_levels = set(self.engine.active_levels) if hasattr(self.engine, 'active_levels') else set()

            # Clear active levels
            self.engine.active_levels.clear()
            self.engine.grid_levels_open = sorted(new_levels)

            # Re-lock levels from existing open orders
            if not self.cfg.dry_run:
                try:
                    open_orders = self.fetcher.exchange.fetch_open_orders(self.cfg.symbol)
                    for order in open_orders:
                        if order['side'] == 'sell':  # TP orders
                            tp_price = float(order['price'])
                            # Find corresponding grid level
                            for level in new_levels:
                                expected_tp = level * (1 + self.cfg.fixed_tp_pct)
                                if abs(tp_price - expected_tp) / tp_price < 0.003:  # 0.3% tolerance
                                    self.engine.active_levels.add(level)
                                    break

                    print(f"  [OK] Re-locked {len(self.engine.active_levels)} levels from open orders")
                    print(f"  [INFO] Old active levels: {len(old_levels)} -> New: {len(self.engine.active_levels)}")

                except Exception as e:
                    print(f"  [WARNING] Failed to reconcile active levels: {e}")
            else:
                print(f"  [DRY RUN] Would reconcile active levels")
        
        # Fetch data
        try:
            t_start = time.time()
            ob = self.fetcher.fetch_orderbook(limit=self.cfg.orderbook_depth)
            trades = self.fetcher.fetch_trades(limit=100)
            latency_ms = (time.time() - t_start) * 1000

            # Record successful API call
            self.health_monitor.record_api_call(success=True, latency_ms=latency_ms)
        except Exception as e:
            # Record failed API call
            self.health_monitor.record_api_call(success=False, latency_ms=0)
            self.health_monitor.add_error(f"API fetch failed: {e}")
            return

        # [OK] FIX: Validate orderbook structure first
        if not ob.get("ok") or not ob.get("bids") or not ob.get("asks"):
            return  # Skip this cycle
        
        try:
            # Extract bid/ask
            bb = float(ob["bids"][0][0])
            ba = float(ob["asks"][0][0])
            
            # Validate prices
            if bb <= 0 or ba <= 0 or ba <= bb:
                print(f"[WARNING] Invalid prices: bid={bb}, ask={ba}")
                return
            
            # Calculate mid and spread
            mid = (bb + ba) / 2.0
            sp = (ba - bb) / mid
            
            # Enrich orderbook
            ob["mid"] = mid
            ob["spread_pct"] = sp
            self._last_mid = mid

            # [OK] FIX: Validate spread BEFORE analyzing
            if not (self.cfg.min_spread_pct <= sp <= self.cfg.max_spread_pct):
                return  # Skip if spread out of range
            
            # Safe to analyze now
            ob_an = self.engine.microstructure.analyze_orderbook(ob)
            self.aggregator.add_orderbook_snapshot(ob_an)
            
        except (IndexError, ValueError, TypeError) as e:
            print(f"[WARNING] Orderbook error: {e}")
            return
        
        # Add trades to aggregator
        self.aggregator.add_trades(trades)
        
        # Roll bar
        bar = self.aggregator.roll_bar(t0)
        if not bar:
            return
        
        # Update health monitor
        self.health_monitor.record_data_update(datetime.now(timezone.utc))
        
        # Fetch OHLCV
        ohlcv = self.fetcher.fetch_ohlcv('5m', 100)
        
        # Check circuit breaker
        current_state = self._get_current_state(bar)
        breaker_active, breaker_reason = self.circuit_breaker.check_and_activate(current_state)
        
        if breaker_active:
            if not self.cfg.dry_run:
                self._emergency_exit_all(breaker_reason)
            self.notifier.notify_circuit_breaker(breaker_reason)
            return
        
        # Update signal engine
        decision = self.engine.update(
            bar, ohlcv, trades, ob, t0, self.cfg.max_open_orders
        )
        
        # Log decision
        self._log_decision(bar, decision, breaker_active)
        
        # Print status
        self._print_bar_status(bar, decision)
        
        # Manage existing positions
        self._manage_positions(bar['mid_price'])
        
        # Execute new signal
        if decision.action == 'PLACE_BUY':
            if not self.cfg.dry_run:
                self._execute_buy(decision, bar['mid_price'])
            else:
                # DRY RUN: Simulate buy for testing/dashboard
                self._simulate_buy(decision, bar['mid_price'])
    
    def _periodic_tasks(self):
        """Periodic maintenance tasks"""
        current_time = time.time()

        if not self.cfg.dry_run:
            if current_time - self.last_resync >= self.cfg.resync_open_orders_sec:
                self.executor.resync_open_orders(self.engine)
                self.last_resync = current_time

        if current_time - self.last_health_check >= self.cfg.health_check_interval_sec:
            health = self.health_monitor.check_health()

            # Always print health status
            self.health_monitor.print_health_report(health)

            # Send notification if critical
            if health.status == "critical":
                self.notifier.send_message("[WARNING] <b>Health Check: CRITICAL</b>")

            self.last_health_check = current_time

        if current_time - self.last_checkpoint >= 300:
            self._save_state()
            self.last_checkpoint = current_time

        # [OK] ENHANCED: Timestamp-based reconciliation with active_levels rebuild
        if not self.cfg.dry_run and len(self.positions) > 0:
            if current_time - self.last_reconcile >= self.cfg.reconciliation.interval_sec:
                report = self.reconciler.reconcile(
                    engine=self.signal_engine,
                    grid_levels=self.grid_levels
                )
                if report['discrepancies']:
                    print(f"[WARNING] Reconciliation found {len(report['discrepancies'])} discrepancies")
                if report.get('active_levels_rebuilt'):
                    print(f"[OK] Rebuilt active_levels: {report['fixes_applied'][0]['new_count']} levels")
                self.last_reconcile = current_time

    def reconcile_positions_with_exchange(self):
        """Reconcile local positions with actual exchange state"""
        # [OK] SAFETY: Skip in dry run mode
        if self.cfg.dry_run:
            print("[DRY RUN] Skipping reconciliation")
            return

        try:
            balance = self.fetcher.exchange.fetch_balance()
            base_currency = self.cfg.symbol.split('/')[0]
            coin_total = balance.get(base_currency, {}).get('total', 0)
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            expected_coin = sum(p.size_coin for p in self.positions.values() if p.status.value == 'open')
            
            print("\n" + "="*60)
            print("POSITION RECONCILIATION")
            print("="*60)
            print(f"Exchange {base_currency}: {coin_total:.4f}")
            print(f"Expected {base_currency}: {expected_coin:.4f}")
            print(f"USDT Balance: ${usdt_balance:.2f}")
            
            discrepancy = abs(expected_coin - coin_total)

            if discrepancy > 0.0001:
                print(f"[WARNING] DISCREPANCY: {discrepancy:.6f} {base_currency}")
                
                if coin_total > expected_coin:
                    extra = coin_total - expected_coin
                    print(f"  -> Extra {base_currency}: {extra:.6f}")

                    recon_pos = Position(
                        position_id=f"recon_{int(time.time()*1000)}",
                        symbol=self.cfg.symbol,
                        entry_price=self.fetcher.exchange.fetch_ticker(self.cfg.symbol)['last'],
                        size_coin=extra,
                        size_usd=extra * self._last_mid,
                        tp_price=self._last_mid * 1.015,
                        entry_reason="reconciliation",
                        market_phase="unknown"
                    )

                    self.positions[recon_pos.position_id] = recon_pos
                    self.state_manager.save_position(recon_pos)
                    print(f"  -> Created position: {recon_pos.position_id}")

                else:
                    missing = expected_coin - coin_total
                    print(f"  -> Missing {base_currency}: {missing:.6f}")

                    for pos_id, pos in list(self.positions.items()):
                        if pos.status.value == 'open' and missing > 0:
                            print(f"  -> Closing phantom: {pos_id}")
                            pos.status = PositionStatus.CLOSED
                            pos.exit_reason = "reconciliation_phantom"
                            self.state_manager.close_position(pos_id, "reconciliation_phantom")
                            del self.positions[pos_id]
                            missing -= pos.size_coin
            else:
                print("[OK] Positions synchronized")

            print("="*60 + "\n")

        except Exception as e:
            print(f"[ERROR] Reconciliation failed: {e}")
        
    def _get_current_state(self, bar: Dict) -> Dict:
        """Get current state for circuit breaker"""
        total_position_value = sum(
            p.size_coin * bar['mid_price'] 
            for p in self.positions.values()
        )
        
        try:
            balance = self.fetcher.exchange.fetch_balance()
            current_balance = balance.get('USDT', {}).get('total', self.cfg.budget_usdt)
        except:
            current_balance = self.cfg.budget_usdt
        
        self.health_monitor.update_state(
            len(self.positions),
            total_position_value,
            self.engine.open_orders_count
        )
        
        return {
            'current_price': bar['mid_price'],
            'current_balance': current_balance,
            'current_drawdown': 0.0,
            'total_positions_value': total_position_value,
            'data_quality': bar.get('data_quality', 'ok')
        }
    
    def _manage_positions(self, current_price: float):
        """Check and manage all positions"""
        positions_to_close = []

        for pos_id, pos in list(self.positions.items()):
            pos.update_pnl(current_price)
            self.stop_loss_mgr.update_trailing_stop(pos, current_price, self.engine)

            # Check TP (for dry-run mode)
            if self.cfg.dry_run and current_price >= pos.tp_price:
                positions_to_close.append((pos_id, 'take_profit', pos.tp_price))
            # Check SL
            elif self.stop_loss_mgr.check_stop_triggered(pos, current_price):
                positions_to_close.append((pos_id, 'stop_loss', pos.sl_price))
            # Check timeout
            elif pos.should_timeout(max_hours=self.cfg.safety.max_position_age_hours):
                positions_to_close.append((pos_id, 'timeout', current_price))

        for pos_id, reason, exit_price in positions_to_close:
            self._close_position(pos_id, reason, exit_price)
    
    def _close_position(self, pos_id: str, reason: str, exit_price: float):
        """Close a position"""
        pos = self.positions.get(pos_id)
        if not pos:
            return

        # [OK] ENHANCED: Dry run safety check
        if not self.cfg.dry_run:
            try:
                order = self.fetcher.exchange.create_market_sell_order(
                    self.cfg.symbol,
                    pos.size_coin
                )
                
                filled_qty = float(order.get('filled', pos.size_coin))
                actual_exit = float(order.get('price', exit_price))
                pnl = (actual_exit - pos.entry_price) * filled_qty
                
                pos.exit_time = datetime.now(timezone.utc)
                pos.exit_reason = reason
                pos.status = PositionStatus.CLOSED
                
                self.circuit_breaker.record_trade(pnl)
                self.position_sizer.record_trade(pnl, pos.size_usd)
                self.state_manager.close_position(pos_id, reason)
                
                self.notifier.notify_stop_loss(
                    self.cfg.symbol,
                    pos.entry_price,
                    actual_exit,
                    pnl
                )
                
                print(f"[OK] Position closed: {pos_id}")
                print(f"  Reason: {reason}")
                print(f"  Entry: ${pos.entry_price:.6f} | Exit: ${actual_exit:.6f}")
                print(f"  P&L: ${pnl:.2f}")

            except Exception as e:
                print(f"[ERROR] Failed to close position {pos_id}: {e}")
                return
        else:
            # [OK] DRY RUN: Simulate position close
            print(f"[DRY RUN] Would close position {pos_id}")
            print(f"  Reason: {reason}")
            print(f"  Entry: ${pos.entry_price:.6f} | Exit: ${exit_price:.6f}")
            simulated_pnl = (exit_price - pos.entry_price) * pos.size_coin
            print(f"  Simulated P&L: ${simulated_pnl:.2f}")

        del self.positions[pos_id]
        self.stop_loss_mgr.remove_stop(pos_id)
        self.engine.active_levels.discard(pos.entry_price)
    
    def _simulate_buy(self, decision, current_price: float):
        """Simulate buy order for dry-run mode"""
        level = decision.grid_candidate

        level_info = self.grid_planner.get_level_info(level, self.cfg.grid_tolerance)
        if not level_info:
            return

        # Safety check: TP must be > Entry
        if level_info['tp_price'] <= current_price:
            print(f"[SKIP] TP ${level_info['tp_price']:.4f} <= Entry ${current_price:.4f}")
            return

        available = self.cfg.budget_usdt * (1 - self.cfg.reserve_pct)
        used = sum(p.size_usd for p in self.positions.values())
        size_usd = min(available - used, self.cfg.budget_usdt * 0.2)  # Max 20% per position

        if size_usd < 5:  # Min $5
            return

        coin_size = size_usd / current_price
        pos_id = f"pos_{int(time.time()*1000)}"

        position = Position(
            position_id=pos_id,
            symbol=self.cfg.symbol,
            entry_price=current_price,
            size_coin=coin_size,
            size_usd=size_usd,
            tp_price=level_info['tp_price'],
            tp_order_id=None,
            sl_price=current_price * 0.99,
            sl_type="fixed",
            entry_reason=decision.reason
        )

        self.positions[pos_id] = position
        self.state_manager.save_position(position)

        print(f"[DRY RUN] Simulated buy: {coin_size:.4f} @ ${current_price:.6f} = ${size_usd:.2f}")
        print(f"  TP: ${position.tp_price:.6f}")

    def _execute_buy(self, decision, current_price: float):
        """Execute buy order"""
        level = decision.grid_candidate
        
        level_info = self.grid_planner.get_level_info(
            level, self.cfg.grid_tolerance
        )
        
        if not level_info:
            print(f"[WARNING] No level info for {level:.6f}")
            return
        
        available = self.cfg.budget_usdt * (1 - self.cfg.reserve_pct)
        used = sum(p.size_usd for p in self.positions.values())
        
        size_usd = self.position_sizer.calculate_size(
            available - used,
            confidence=1.0,
            volatility=decision.volatility,
            current_drawdown=0.0
        )
        
        coin_size = size_usd / level
        
        # ========== PLACE BUY ORDER ==========
        order = self.executor.place_buy_order(
            level, coin_size, level_info['tp_price'], self.engine
        )
        
        if not order:
            return  # Order failed
        
        # ========== CREATE POSITION ==========
        tp_order = order.get('tp_order') if isinstance(order, dict) else None
        pos_id = f"pos_{int(time.time()*1000)}"
        
        filled_qty = float(order.get('filled', coin_size))
        cost = float(order.get('cost', level * coin_size))
        avg_price = cost / filled_qty if filled_qty > 0 else level
        
        # ========== CREATE POSITION OBJECT (temporary) ==========
        temp_position = Position(
            position_id=pos_id,
            symbol=self.cfg.symbol,
            entry_price=avg_price,
            size_coin=filled_qty,
            size_usd=cost,
            tp_price=level_info['tp_price'],
            market_phase=decision.wyckoff_phase,
            volatility_at_entry=decision.volatility,
            cvd_z_at_entry=decision.cvd_z,
            entry_reason=decision.reason
        )
        
        # ========== CALCULATE STOP LOSS ==========
        sl_price = self.stop_loss_mgr.create_stop_loss(
            temp_position,
            decision.wyckoff_phase,
            decision.volatility
        )
        
        # ========== CREATE FINAL POSITION ==========
        position = Position(
            position_id=pos_id,
            symbol=self.cfg.symbol,
            entry_price=avg_price,
            size_coin=filled_qty,
            size_usd=cost,
            tp_price=level_info['tp_price'],
            sl_price=sl_price,
            sl_type=self.cfg.safety.stop_loss_type if sl_price else None,
            market_phase=decision.wyckoff_phase,
            volatility_at_entry=decision.volatility,
            cvd_z_at_entry=decision.cvd_z,
            entry_reason=decision.reason
        )
        
        # ========== ADD TP ORDER ID ==========
        if tp_order and isinstance(tp_order, dict):
            tp_id = tp_order.get('id')
            if tp_id:
                position.tp_order_id = tp_id
            
            tp_px = tp_order.get('price')
            if tp_px is not None:
                position.tp_price = tp_px
        
        # [OK] FIX: PLACE STOP LOSS ORDER ON EXCHANGE
        if position.sl_price and self.cfg.safety.enable_stop_loss:
            try:
                sl_order = self.executor.place_stop_loss_order(position, self.engine)
                if sl_order:
                    position.sl_order_id = sl_order.get('id')
                    print(f"  [OK] SL order placed: #{position.sl_order_id} @ ${position.sl_price:.6f}")
                else:
                    print(f"  [WARNING] SL order failed, tracking locally only")
            except Exception as e:
                print(f"  [WARNING] SL order error: {e}, tracking locally only")
        
        # ========== SAVE POSITION ==========
        self.positions[pos_id] = position
        self.state_manager.save_position(position)
        
        # ========== NOTIFY ==========
        self.notifier.notify_order(
            'BUY', self.cfg.symbol, avg_price, filled_qty, decision.reason
        )
        
        # ========== LOG ==========
        print(f"[OK] Position opened: {pos_id}")
        print(f"   Entry: ${avg_price:.6f} x {filled_qty:.4f} = ${cost:.2f}")
        print(f"   TP: ${position.tp_price:.6f} ({((position.tp_price/avg_price-1)*100):.2f}%)")
        if position.sl_price:
            print(f"   SL: ${position.sl_price:.6f} ({((position.sl_price/avg_price-1)*100):.2f}%)")
    
    def _emergency_exit_all(self, reason: str):
        """Emergency exit all positions"""
        print(f"\n{'='*60}")
        print(f"[WARNING] EMERGENCY EXIT: {reason}")
        print(f"{'='*60}")
        
        for pos_id in list(self.positions.keys()):
            self._close_position(pos_id, f"emergency_{reason}", 0)
    
    def _save_state(self):
        """Save current state"""
        state = BotState(
            positions=self.positions.copy(),
            active_levels=list(self.engine.active_levels),
            daily_pnl=self.circuit_breaker.daily_pnl,
            consecutive_losses=self.circuit_breaker.consecutive_losses,
            circuit_breaker_active=self.circuit_breaker.active,
            circuit_breaker_reason=self.circuit_breaker.activation_reason,
            circuit_breaker_activation_time=self.circuit_breaker.activation_time
        )
        
        self.state_manager.save_state_snapshot(state)
    
    def _print_bar_status(self, bar: Dict, decision):
        """Print bar status"""
        ts_str = datetime.fromtimestamp(
            bar['bar_ts']/1000, tz=timezone.utc
        ).strftime("%H:%M:%S")
        
        print(
            f"[{ts_str}] "
            f"mid={decision.mid_price:.6f} "
            f"cvd_z={decision.cvd_z:.2f} "
            f"ts_z={decision.ts_z:.2f} "
            f"rsi={decision.rsi:.1f} "
            f"conf={decision.confirm_count} "
            f"act={decision.action} "
            f"pos={len(self.positions)}"
        )
    
    def _log_decision(self, bar: Dict, decision, breaker_active: bool):
        """Log decision to CSV"""
        bar_dt = datetime.fromtimestamp(bar['bar_ts']/1000, tz=timezone.utc)
        
        self.csv_writer.writerow([
            int(bar['bar_ts']),
            bar_dt.strftime("%Y-%m-%d %H:%M:%S"),
            bar.get('data_quality', 'na'),
            decision.mid_price,
            bar.get('cvd', 0),
            decision.cvd_z,
            bar.get('trade_size_proxy', 0),
            decision.ts_z,
            decision.rsi,
            decision.bb_position,
            decision.near_support,
            decision.confirm_count,
            1 if decision.cvd_z >= self.cfg.cvd_z_threshold else 0,
            1 if decision.action == 'PLACE_BUY' else 0,
            decision.grid_candidate,
            decision.action,
            decision.reason,
            self.engine.bars_total,
            len(self.engine.active_levels),
            len(self.positions),
            decision.volatility,
            decision.vpin,
            decision.smi,
            decision.wyckoff_phase,
            decision.spread_pct,
            breaker_active
        ])
        self.csv_file.flush()
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n" + "="*60)
        print("CLEANING UP...")
        print("="*60)
        
        self._save_state()
        
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
        
        stats = self.engine.get_statistics()
        print(f"\nSession Statistics:")
        print(f"  Total bars: {stats['bars_total']}")
        print(f"  Raw signals: {stats['signals_raw']}")
        print(f"  Confirmed signals: {stats['signals_confirmed']}")
        print(f"  Confirmation rate: {stats['confirmation_rate']:.1%}")
        
        health_stats = self.health_monitor.get_uptime_stats()
        if health_stats:
            print(f"\nHealth Statistics:")
            print(f"  Uptime: {health_stats['uptime_percentage']:.1%}")
            print(f"  Total checks: {health_stats['total_checks']}")
        
        self.notifier.send_message(
            "<b>Bot Stopped</b>\n"
            f"Positions: {len(self.positions)}\n"
            "System shutdown complete"
        )

        print("\n[OK] CLEANUP COMPLETE\n")


if __name__ == "__main__":
    from config import load_config
    import os
    
    cfg = load_config("config.yaml")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    
    bot = UltraGridTrader(cfg, api_key, api_secret)
    bot.run()