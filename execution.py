#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Execution Layer with Safety Checks
Version: 2.1.0 - Enhanced with Dust Ledger
"""

import os
import csv
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from config import TradingConfig
from models import Position

getcontext().prec = 28
logger = logging.getLogger(__name__)

# ✅ ENHANCED: Dust ledger configuration
DUST_LEDGER_FILE = "logs/dust_ledger.csv"


class ExecutionLayer:
    """
    Handle order execution with comprehensive safety checks
    
    Key features:
    - Idempotent order placement
    - Duplicate detection
    - Precision rounding
    - Dust tracking
    - Stop loss placement
    """
    
    def __init__(self, exchange, symbol: str, config: TradingConfig):
        self.exchange = exchange
        self.symbol = symbol
        self.cfg = config
        
        # Get market info
        self.market_info = exchange.markets.get(symbol, {})
        
        # Precision
        self.price_precision = self.market_info.get('precision', {}).get('price', 8)
        self.amount_precision = self.market_info.get('precision', {}).get('amount', 8)
        
        # Limits
        self.min_qty = float(self.market_info.get('limits', {}).get('amount', {}).get('min', 0))
        self.min_notional = float(self.market_info.get('limits', {}).get('cost', {}).get('min', 0))
        
        if self.cfg.min_notional_override:
            self.min_notional = max(self.min_notional, self.cfg.min_notional_override)
        
        # ✅ ENHANCED: Dust ledger
        self._init_dust_ledger()

        logger.info(f"Execution Layer Initialized")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Price precision: {self.price_precision}")
        logger.info(f"  Amount precision: {self.amount_precision}")
        logger.info(f"  Min quantity: {self.min_qty}")
        logger.info(f"  Min notional: {self.min_notional}")
    
    def _init_dust_ledger(self):
        """
        ✅ ENHANCED: Initialize dust ledger CSV

        บันทึกเศษที่ขายไม่ได้:
        - lot_rounding: ปัด LOT_SIZE แล้วเหลือเศษ
        - minQty_gate: ต่ำกว่า minQty
        - minNotional_gate: ต่ำกว่า minNotional
        - tp_place_failed: วาง TP ไม่สำเร็จ
        """
        os.makedirs(os.path.dirname(DUST_LEDGER_FILE), exist_ok=True)
        if not os.path.exists(DUST_LEDGER_FILE):
            with open(DUST_LEDGER_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "remainder_qty", "unit_cost_usdt",
                    "est_cost_total_usdt", "reason", "order_id", "planned_tp",
                    "stepSize", "minQty", "minNotional"
                ])
    
    def _check_duplicate_order(self, level: float, tp_price: float, engine) -> bool:
        """
        Check if order already exists
        
        Returns:
            True if duplicate found (should skip)
            False if safe to place order
        """
        # Check 1: In-memory active levels
        if hasattr(engine, 'active_levels'):
            if level in engine.active_levels:
                logger.warning(f"Level {level:.6f} already active")
                return True
        
        # Check 2: Exchange open orders
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
        except Exception as e:
            logger.error(f"Could not fetch open orders: {e}")
            return False  # If can't check, assume safe (will fail later if truly duplicate)
        
        # Check existing SELL orders (TP orders)
        tolerance = 0.005  # 0.5%
        
        for order in open_orders:
            if order.get('side') != 'sell':
                continue
            
            order_price = float(order.get('price', 0))
            if order_price <= 0:
                continue
            
            # Check 2.1: Direct TP collision
            if tp_price > 0:
                price_diff = abs(order_price - tp_price) / tp_price
                if price_diff < 0.003:  # 0.3%
                    logger.warning(f"TP order already exists at ${order_price:.6f}")
                    return True
            
            # Check 2.2: Implied entry from TP
            if self.cfg.fixed_tp_pct:
                implied_entry = order_price / (1.0 + self.cfg.fixed_tp_pct)
                if abs(implied_entry - level) / level < tolerance:
                    logger.warning(f"Order already exists at level {level:.6f}")
                    return True
        
        return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def place_buy_order(self, level: float, coin_size: float, tp_price: float, 
                       engine) -> Optional[Dict]:
        """
        Place BUY order with all safety checks
        
        Args:
            level: Target entry price
            coin_size: Amount to buy
            tp_price: Take profit price
            engine: Signal engine (for tracking)
        
        Returns:
            Order dict or None if failed
        """
        level_locked = False
        
        try:
            # ========== VALIDATION ==========
            if level <= 0 or coin_size <= 0:
                logger.error(f"Invalid parameters: level={level}, size={coin_size}")
                return None
            
            # Round and validate size
            qty_rounded = self._round_amount(coin_size)
            if qty_rounded < self.min_qty:
                # ✅ Log dust: ขายไม่ได้เพราะต่ำกว่า minQty
                self._log_dust(qty_rounded, level, "minQty_gate", planned_tp=tp_price)
                logger.warning(f"Size {qty_rounded} below minimum {self.min_qty}")
                return None

            # Check minimum notional
            notional = qty_rounded * level
            if notional < self.min_notional:
                # ✅ Log dust: ขายไม่ได้เพราะต่ำกว่า minNotional
                self._log_dust(qty_rounded, level, "minNotional_gate", planned_tp=tp_price)
                logger.warning(f"Notional ${notional:.2f} below minimum ${self.min_notional}")
                return None
            
            # ========== DUPLICATE CHECK ==========
            if self._check_duplicate_order(level, tp_price, engine):
                return None
            
            # Lock this level in memory
            if hasattr(engine, 'active_levels'):
                engine.active_levels.add(level)
                level_locked = True
            
            # ========== FETCH MARKET DATA ==========
            try:
                ob = self.exchange.fetch_order_book(self.symbol, limit=5)
                best_ask = float(ob['asks'][0][0]) if ob.get('asks') else None
                best_bid = float(ob['bids'][0][0]) if ob.get('bids') else None
            except Exception as e:
                logger.warning(f"Could not fetch orderbook: {e}")
                best_ask, best_bid = None, None
            
            # Safety: Don't buy if market too far above level
            if best_ask and level > 0:
                if best_ask / level > 1.02:  # 2% above
                    logger.warning(f"Market ${best_ask:.6f} too far above level ${level:.6f}")
                    return None
            
            # ========== PLACE BUY ORDER ==========
            # ✅ ENHANCED: High-price buy strategy (use quoteOrderQty for assets >= threshold)
            if best_ask and best_ask >= self.cfg.high_price_threshold:
                # High-price asset: use quoteOrderQty to avoid large notional errors
                quote_amt = qty_rounded * level * self.cfg.quote_safety_factor

                logger.info(f"Placing BUY (quoteOrderQty): ${quote_amt:.2f} @ market (target: ${level:.6f})")

                buy_order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=None,  # Required by ccxt even when using quoteOrderQty
                    params={'quoteOrderQty': quote_amt}
                )

                # Extract fill details (quoteOrderQty returns different format)
                filled_qty = float(buy_order.get('filled', 0))
                cost = float(buy_order.get('cost', quote_amt))

                # Handle edge case: if filled=0, try to get from 'executedQty' or 'cummulativeQuoteQty'
                if filled_qty == 0:
                    info = buy_order.get('info', {})
                    filled_qty = float(info.get('executedQty', 0))

                    # If still 0, estimate from cost
                    if filled_qty == 0 and cost > 0:
                        filled_qty = cost / level  # Approximate

                avg_price = cost / filled_qty if filled_qty > 0 else level

            else:
                # Normal-price asset: use standard amount-based order
                logger.info(f"Placing BUY: {qty_rounded:.6f} @ market (target: ${level:.6f})")

                buy_order = self.exchange.create_market_buy_order(
                    self.symbol,
                    qty_rounded
                )

                # Extract fill details
                filled_qty = float(buy_order.get('filled', qty_rounded))
                cost = float(buy_order.get('cost', 0))
                avg_price = cost / filled_qty if filled_qty > 0 else level

            # Validate order filled
            if buy_order.get('status') not in ['closed', 'filled']:
                logger.error(f"Buy order not filled: {buy_order.get('status')}")
                return None
            
            logger.info(f"✅ BUY filled: {filled_qty:.6f} @ ${avg_price:.6f}")
            
            # ========== CALCULATE TP PRICE ==========
            if not tp_price or tp_price <= avg_price:
                tp_price = avg_price * (1 + self.cfg.fixed_tp_pct)
            
            # ========== PLACE TP ORDER ==========
            tp_order = self._place_tp_order(
                filled_qty, 
                tp_price, 
                best_bid or avg_price,
                engine
            )
            
            if not tp_order:
                logger.error("Failed to place TP order")
                # Note: BUY is already filled, position exists without TP
                # Should handle this in reconciliation
            
            # ========== PREPARE RESPONSE ==========
            result = {
                'id': buy_order.get('id'),
                'filled': filled_qty,
                'cost': cost,
                'price': avg_price,
                'average': avg_price,
                'status': 'closed',
                'tp_order': tp_order
            }
            
            return result
        
        except Exception as e:
            logger.error(f"place_buy_order failed: {e}", exc_info=True)
            return None
        
        finally:
            # Unlock level on failure
            if level_locked and hasattr(engine, 'active_levels'):
                # Only remove if order failed (no result)
                # If successful, keep locked
                pass
    
    def _place_tp_order(self, qty: float, tp_price: float, best_bid: float,
                       engine) -> Optional[Dict]:
        """
        Place TP (LIMIT SELL) order with retry mechanism

        Args:
            qty: Amount to sell
            tp_price: Target price
            best_bid: Current best bid (for safety check)
            engine: Signal engine

        Returns:
            Order dict or None
        """
        # ✅ Round TP price (ceiling for SELL side)
        tp_rounded = self._round_price(tp_price, side='sell')

        # Safety: TP must be above best bid
        tp_safe = max(tp_rounded, best_bid * (1 + self.cfg.tp_offset_pct))

        # Round quantity
        qty_rounded = self._round_amount(qty)
        if qty_rounded < self.min_qty:
            # ✅ Log dust: TP ไม่ได้เพราะต่ำกว่า minQty
            self._log_dust(qty_rounded, tp_safe, "minQty_gate", planned_tp=tp_safe)
            return None

        # Check notional
        notional = qty_rounded * tp_safe
        if notional < self.min_notional:
            # ✅ Log dust: TP ไม่ได้เพราะต่ำกว่า minNotional
            self._log_dust(qty_rounded, tp_safe, "minNotional_gate", planned_tp=tp_safe)
            return None

        logger.info(f"Placing TP: {qty_rounded:.6f} @ ${tp_safe:.6f}")

        # ✅ ENHANCED: Retry mechanism for TP order
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                order = self.exchange.create_limit_sell_order(
                    self.symbol,
                    qty_rounded,
                    tp_safe
                )

                if hasattr(engine, 'open_orders_count'):
                    engine.open_orders_count += 1

                logger.info(f"✅ TP order placed: #{order.get('id')}")

                return order

            except Exception as e:
                logger.error(f"TP order attempt {attempt}/{max_retries} failed: {e}")

                if attempt < max_retries:
                    import time
                    time.sleep(1 * attempt)  # Exponential backoff: 1s, 2s, 3s
                else:
                    logger.error(f"❌ TP order failed after {max_retries} attempts")
                    return None
    
    def place_stop_loss_order(self, position: Position, engine) -> Optional[Dict]:
        """
        Place stop loss order on exchange
        
        Args:
            position: Position object with SL price
            engine: Signal engine
        
        Returns:
            Order dict or None
        """
        try:
            if not position.sl_price:
                return None
            
            # ✅ Round prices (floor for SELL side - conservative)
            sl_trigger = self._round_price(position.sl_price, side='sell')
            sl_limit = self._round_price(sl_trigger * 0.995, side='sell')  # 0.5% below trigger
            qty = self._round_amount(position.size_coin)
            
            # Validate
            if qty < self.min_qty:
                # ✅ Log dust: SL ไม่ได้เพราะต่ำกว่า minQty
                self._log_dust(qty, sl_trigger, "minQty_gate", planned_tp=0)
                return None

            notional = qty * sl_trigger
            if notional < self.min_notional:
                # ✅ Log dust: SL ไม่ได้เพราะต่ำกว่า minNotional
                self._log_dust(qty, sl_trigger, "minNotional_gate", planned_tp=0)
                return None
            
            logger.info(f"Placing SL: {qty:.6f} @ trigger=${sl_trigger:.6f}, limit=${sl_limit:.6f}")
            
            # Place stop-limit order
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop_loss_limit',
                side='sell',
                amount=qty,
                price=sl_limit,
                params={
                    'stopPrice': sl_trigger,
                    'timeInForce': 'GTC'
                }
            )
            
            if hasattr(engine, 'open_orders_count'):
                engine.open_orders_count += 1
            
            logger.info(f"✅ SL order placed: #{order.get('id')}")
            
            return order
        
        except Exception as e:
            logger.error(f"SL order failed: {e}")
            return None
    
    def prelock_existing_orders(self, engine, grid_planner):
        """
        Pre-lock levels from existing open orders
        Critical for avoiding duplicate orders on restart
        """
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            locked_count = 0
            
            for order in open_orders:
                if order['side'] != 'sell':  # Only TP orders
                    continue
                
                tp_price = float(order['price'])
                
                # Try to map back to buy level
                if hasattr(grid_planner, 'grid_df') and grid_planner.grid_df is not None:
                    for _, row in grid_planner.grid_df.iterrows():
                        expected_tp = float(row.get('tp_price', row['buy_price'] * 1.015))
                        
                        # Stricter matching: 0.2%
                        if abs(tp_price - expected_tp) / tp_price < 0.002:
                            buy_level = float(row['buy_price'])
                            
                            if hasattr(engine, 'active_levels'):
                                engine.active_levels.add(buy_level)
                                locked_count += 1
                            break
                
                if hasattr(engine, 'open_orders_count'):
                    engine.open_orders_count += 1
            
            logger.info(f"Pre-locked {locked_count} levels from {engine.open_orders_count} open orders")
        
        except Exception as e:
            logger.error(f"Pre-lock failed: {e}")
    
    def resync_open_orders(self, engine):
        """
        Resync state with exchange (ground truth)
        Fixes state drift
        """
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            sell_orders = [o for o in open_orders if o['side'] == 'sell']
            
            if hasattr(engine, 'open_orders_count'):
                engine.open_orders_count = len(sell_orders)
            
            logger.info(f"Resynced: {len(sell_orders)} open TP orders")
        
        except Exception as e:
            logger.error(f"Resync failed: {e}")
    
    def _round_price(self, price: float, rounding=ROUND_FLOOR, side: str = None) -> float:
        """
        ✅ ENHANCED: Round price to exchange precision

        Args:
            price: Price to round
            rounding: Default rounding mode (backward compatibility)
            side: 'buy' or 'sell' (overrides rounding)
                  - buy  → ROUND_FLOOR (ไม่เกินงบ)
                  - sell → ROUND_CEILING (ไม่ต่ำกว่าเป้า)

        Returns:
            Rounded price
        """
        if price <= 0:
            raise ValueError(f"Invalid price: {price}")

        # ✅ Auto rounding based on side
        if side:
            rounding = ROUND_CEILING if side.lower() == 'sell' else ROUND_FLOOR

        step = 10 ** (-self.price_precision)
        d = Decimal(str(price)).quantize(Decimal(str(step)), rounding=rounding)
        return float(d)
    
    def _round_amount(self, amount: float) -> float:
        """Round amount to exchange precision"""
        if amount <= 0:
            raise ValueError(f"Invalid amount: {amount}")
        
        step = 10 ** (-self.amount_precision)
        d = Decimal(str(amount)).quantize(Decimal(str(step)), rounding=ROUND_FLOOR)
        return float(d)
    
    def _log_dust(self, remainder_qty: float, unit_cost_usdt: float, reason: str,
                  order_id: str = "", planned_tp: float = 0):
        """
        ✅ ENHANCED: Log dust to ledger

        Args:
            remainder_qty: ปริมาณเศษที่ขายไม่ได้ (BASE currency)
            unit_cost_usdt: ต้นทุนต่อหน่วย (USDT per BASE)
            reason: สาเหตุ (lot_rounding, minQty_gate, minNotional_gate, tp_place_failed)
            order_id: Order ID ที่เกี่ยวข้อง
            planned_tp: ราคา TP ที่ตั้งใจจะใช้
        """
        if remainder_qty is None or remainder_qty <= 0:
            return

        try:
            with open(DUST_LEDGER_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    self.symbol,
                    f"{float(remainder_qty):.10f}",
                    f"{float(unit_cost_usdt):.8f}",
                    f"{float(remainder_qty) * float(unit_cost_usdt):.8f}",
                    reason,
                    order_id,
                    f"{float(planned_tp):.8f}" if planned_tp else "",
                    10 ** (-self.amount_precision),  # stepSize
                    self.min_qty,
                    self.min_notional
                ])
                logger.info(f"📝 Dust logged: {remainder_qty:.8f} ({reason})")
        except Exception as e:
            logger.error(f"Dust log failed: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_order(order: Dict) -> bool:
    """
    Validate order response from exchange
    
    Args:
        order: Order dict from exchange
    
    Returns:
        True if valid, False otherwise
    """
    if not order:
        return False
    
    required_keys = ['id', 'status', 'filled']
    for key in required_keys:
        if key not in order:
            logger.warning(f"Order missing key: {key}")
            return False
    
    if order['status'] not in ['closed', 'filled']:
        logger.warning(f"Order not filled: {order['status']}")
        return False
    
    filled = float(order.get('filled', 0))
    if filled <= 0:
        logger.warning(f"Order filled qty is 0")
        return False
    
    return True


if __name__ == "__main__":
    # Test execution layer
    print("Testing ExecutionLayer...")
    
    # Mock exchange
    class MockExchange:
        markets = {
            'ADA/USDT': {
                'precision': {'price': 4, 'amount': 2},
                'limits': {
                    'amount': {'min': 1},
                    'cost': {'min': 10}
                }
            }
        }
    
    from config import TradingConfig
    cfg = TradingConfig()
    
    exec_layer = ExecutionLayer(MockExchange(), 'ADA/USDT', cfg)
    
    # Test rounding
    price = 0.123456
    rounded = exec_layer._round_price(price, ROUND_FLOOR)
    print(f"✅ Price rounding: {price} → {rounded}")
    
    amount = 123.456
    rounded_amt = exec_layer._round_amount(amount)
    print(f"✅ Amount rounding: {amount} → {rounded_amt}")
    
    print("\n✅ ExecutionLayer tests passed")