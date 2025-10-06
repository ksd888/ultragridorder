#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Engine with MAD-zscore and Multi-indicator Fusion
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone

from config import TradingConfig
from models import TradeDecision, OrderBookSnapshot
from indicators import (
    calculate_rsi, calculate_bollinger_bands, calculate_atr,
    calculate_vpin, calculate_smi, detect_wyckoff_phase,
    detect_absorption, MicrostructureAnalyzer
)


class MADZscoreTracker:
    """
    Median Absolute Deviation Z-score tracker
    More robust than standard deviation for orderflow
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.values = deque(maxlen=window)
    
    def add(self, value: float):
        """Add new value"""
        self.values.append(value)
    
    def get_zscore(self) -> float:
        """Calculate MAD-based z-score"""
        if len(self.values) < 20:
            return 0.0
        
        values = np.array(self.values)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad < 1e-10:
            return 0.0
        
        # MAD z-score: (x - median) / (1.4826 * MAD)
        # 1.4826 is the constant to make MAD comparable to std dev
        latest = values[-1]
        zscore = (latest - median) / (1.4826 * mad)
        
        return float(zscore)
    
    def get_median(self) -> float:
        """Get current median"""
        if len(self.values) == 0:
            return 0.0
        return float(np.median(self.values))


class SignalEngine:
    """
    Main signal engine combining orderflow and technical analysis
    """
    
    def __init__(self, config: TradingConfig, grid_levels: List[float]):
        self.cfg = config
        self.grid_levels_open = sorted(grid_levels)
        
        # State
        self.active_levels: Set[float] = set()
        self.open_orders_count = 0
        self.last_signal_time = 0
        
        # MAD-zscore trackers
        self.cvd_tracker = MADZscoreTracker(window=100)
        self.ts_tracker = MADZscoreTracker(window=100)
        
        # EMA smoothing
        self.cvd_ema = None
        self.ts_ema = None
        self.imbalance_ema = 0.5
        
        # Confirmation system
        self.confirm_counter = 0
        self.last_signal_raw = False
        
        # OHLCV data for indicators
        self.ohlcv_df = pd.DataFrame()
        
        # Microstructure analysis
        self.microstructure = MicrostructureAnalyzer()
        
        # Statistics
        self.bars_total = 0
        self.signals_raw = 0
        self.signals_confirmed = 0

        print(f"\n[OK] Signal Engine Initialized:")
        print(f"  Grid levels: {len(self.grid_levels_open)}")
        print(f"  CVD threshold: {config.cvd_z_threshold}")
        print(f"  TS threshold: {config.ts_z_threshold}")
        print(f"  Confirm bars: {config.confirm_bars}")
    
    def update(self, bar: Dict, ohlcv: pd.DataFrame, trades: List, 
               orderbook: Dict, now_ms: int, max_orders: int) -> TradeDecision:
        """
        Main update - process new bar and generate decision
        """
        self.bars_total += 1
        
        # Update OHLCV
        if not ohlcv.empty:
            self.ohlcv_df = ohlcv
        
        # Extract bar data
        mid_price = bar['mid_price']
        cvd = bar['cvd']
        ts_proxy = bar['trade_size_proxy']
        order_imb = bar['order_imbalance']
        
        # Update EMA
        self.cvd_ema = self._ema_update(self.cvd_ema, cvd, self.cfg.cvd_ema_span)
        self.ts_ema = self._ema_update(self.ts_ema, ts_proxy, self.cfg.ts_ema_span)
        self.imbalance_ema = self._ema_update(self.imbalance_ema, order_imb, 20)
        
        # Add to MAD trackers
        self.cvd_tracker.add(cvd)
        self.ts_tracker.add(ts_proxy)
        
        # Calculate z-scores
        cvd_z = self.cvd_tracker.get_zscore()
        ts_z = self.ts_tracker.get_zscore()
        
        # Technical indicators
        rsi = self._calculate_rsi()
        bb_position = self._calculate_bb_position(mid_price)
        volatility = self._calculate_volatility()
        
        # Advanced indicators
        vpin = self._calculate_vpin(trades, orderbook)
        smi = self._calculate_smi()
        wyckoff_phase = self._detect_wyckoff()
        
        # Support detection
        near_support = self._check_near_support(mid_price)
        
        # Absorption detection
        absorption = detect_absorption(bar, trades)
        
        # Generate raw signal
        signal_raw = self._check_raw_signal(
            cvd_z, ts_z, order_imb, rsi, bb_position, near_support
        )
        
        # Confirmation system
        confirmed, confirm_count = self._check_confirmation(signal_raw)
        
        # Find grid candidate
        grid_candidate = self._find_nearest_grid(mid_price)
        
        # Apply filters
        action, reason = self._apply_filters(
            confirmed, grid_candidate, mid_price, volatility,
            bar.get('spread_pct', 0), now_ms, max_orders
        )
        
        # Create decision
        decision = TradeDecision(
            timestamp=now_ms,
            action=action,
            reason=reason,
            mid_price=mid_price,
            grid_candidate=grid_candidate if grid_candidate else 0.0,
            cvd_z=cvd_z,
            ts_z=ts_z,
            order_imbalance=order_imb,
            confirm_count=confirm_count,
            rsi=rsi,
            bb_position=bb_position,
            near_support=near_support,
            vpin=vpin,
            smi=smi,
            wyckoff_phase=wyckoff_phase,
            absorption=absorption,
            volatility=volatility,
            spread_pct=bar.get('spread_pct', 0),
            active_positions=len(self.active_levels),
            open_orders=self.open_orders_count
        )
        
        return decision
    
    # Line 180: _check_raw_signal()
    def _check_raw_signal(self, cvd_z, ts_z, order_imb, rsi, bb_position, near_support):
        """
        [OK] OPTIONAL: More lenient - either CVD or TS strong signal
        """
        # Primary orderflow signals
        cvd_strong = cvd_z >= self.cfg.cvd_z_threshold
        ts_strong = ts_z >= self.cfg.ts_z_threshold
        
        # Secondary confirmation
        imb_bullish = order_imb >= self.cfg.imbalance_threshold
        
        # Technical confluence
        rsi_oversold = rsi <= self.cfg.rsi_oversold
        bb_lower = bb_position <= 0.2

        # [OK] ORIGINAL: Both orderflow + imbalance + technical
        # orderflow_signal = (cvd_strong or ts_strong) and imb_bullish
        # technical_signal = rsi_oversold or bb_lower or near_support
        # signal = orderflow_signal and technical_signal

        # Strong orderflow signals
        strong_orderflow = (cvd_strong and ts_strong) and imb_bullish
        moderate_orderflow = (cvd_strong or ts_strong) and imb_bullish

        technical_signal = rsi_oversold or bb_lower or near_support

        # Accept orderflow signal alone (don't require technical confirmation)
        signal = strong_orderflow or moderate_orderflow

        if signal:
            self.signals_raw += 1

        return signal
    
    def _check_confirmation(self, signal_raw: bool) -> tuple:
        """
        Confirmation system - require signal for N consecutive bars
        """
        if signal_raw:
            if self.last_signal_raw:
                self.confirm_counter += 1
            else:
                self.confirm_counter = 1
        else:
            self.confirm_counter = 0
        
        self.last_signal_raw = signal_raw
        
        confirmed = self.confirm_counter >= self.cfg.confirm_bars
        
        if confirmed and self.confirm_counter == self.cfg.confirm_bars:
            self.signals_confirmed += 1
        
        return confirmed, self.confirm_counter
    
    def _apply_filters(self, confirmed: bool, grid_candidate: Optional[float],
                       mid_price: float, volatility: float, spread_pct: float,
                       now_ms: int, max_orders: int) -> tuple:
        """Apply all filters and return action + reason"""
        
        if not confirmed:
            return "HOLD", "signal_not_confirmed"
        
        # Check grid candidate
        if grid_candidate is None:
            return "SKIP", "no_grid_candidate"
        
        # Check if level already active
        if grid_candidate in self.active_levels:
            return "SKIP", "level_already_active"
        
        if not confirmed:
            return "HOLD", "signal_not_confirmed"
        
        # Check max orders
        if self.open_orders_count >= max_orders:
            return "SKIP", "max_orders_reached"
        
        # Check cooldown
        if now_ms - self.last_signal_time < self.cfg.order_cooldown_sec * 1000:
            return "SKIP", "cooldown_active"
        
        # Volatility filter
        if volatility < self.cfg.min_volatility:
            return "SKIP", "volatility_too_low"
        if volatility > self.cfg.max_volatility:
            return "SKIP", "volatility_too_high"
        
        # Spread filter
        if spread_pct > self.cfg.max_spread_pct:
            return "SKIP", "spread_too_wide"
        
        # Price above grid level (safety)
        if mid_price < grid_candidate:
            return "SKIP", "price_below_grid"
        
        # All checks passed
        self.last_signal_time = now_ms
        return "PLACE_BUY", "all_checks_passed"
    
    def _find_nearest_grid(self, current_price: float) -> Optional[float]:
        """Find nearest valid grid level below price"""
        available = [
            lvl for lvl in self.grid_levels_open 
            if lvl not in self.active_levels and lvl <= current_price
        ]
        
        if not available:
            return None
        
        # Find closest below current price
        distances = [(abs(current_price - lvl), lvl) for lvl in available]
        distances.sort()
        
        return distances[0][1] if distances else None
    
    def _check_near_support(self, price: float) -> bool:
        """Check if price near support level"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < 20:
            return False
        
        # Simple support: recent lows
        recent_lows = self.ohlcv_df['low'].tail(20).min()
        distance = abs(price - recent_lows) / price
        
        return distance < 0.005  # Within 0.5%
    
    def _calculate_rsi(self) -> float:
        """Calculate RSI"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < self.cfg.rsi_period + 1:
            return 50.0
        
        rsi = calculate_rsi(self.ohlcv_df['close'], self.cfg.rsi_period)
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _calculate_bb_position(self, current_price: float) -> float:
        """Calculate Bollinger Band position (0=lower, 1=upper)"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < self.cfg.bb_period:
            return 0.5
        
        upper, middle, lower = calculate_bollinger_bands(
            self.ohlcv_df['close'], self.cfg.bb_period, self.cfg.bb_std
        )
        
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        
        if upper_val <= lower_val:
            return 0.5
        
        position = (current_price - lower_val) / (upper_val - lower_val)
        return float(np.clip(position, 0, 1))
    
    def _calculate_volatility(self) -> float:
        """Calculate ATR-based volatility"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < 14:
            return 0.02
        
        atr = calculate_atr(
            self.ohlcv_df['high'],
            self.ohlcv_df['low'],
            self.ohlcv_df['close'],
            period=14
        )
        
        if atr.empty:
            return 0.02
        
        # Normalize by price
        atr_pct = atr.iloc[-1] / self.ohlcv_df['close'].iloc[-1]
        return float(np.clip(atr_pct, 0.001, 0.5))
    
    def _calculate_vpin(self, trades: List, orderbook: Dict) -> float:
        """Calculate Volume-Synchronized Probability of Informed Trading"""
        if not trades or not orderbook.get('ok'):
            return 0.5
        
        return calculate_vpin(trades, orderbook)
    
    def _calculate_smi(self) -> float:
        """Calculate Smart Money Index"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < 20:
            return 0.5
        
        return calculate_smi(self.ohlcv_df)
    
    def _detect_wyckoff(self) -> str:
        """Detect Wyckoff market phase"""
        if self.ohlcv_df.empty or len(self.ohlcv_df) < 50:
            return "unknown"

        phase, confidence = detect_wyckoff_phase(self.ohlcv_df)
        return phase
    
    def _ema_update(self, prev: Optional[float], new_val: float, span: int) -> float:
        """Update EMA"""
        alpha = 2.0 / (span + 1)
        if prev is None:
            return new_val
        return alpha * new_val + (1 - alpha) * prev
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            'bars_total': self.bars_total,
            'signals_raw': self.signals_raw,
            'signals_confirmed': self.signals_confirmed,
            'confirmation_rate': (
                self.signals_confirmed / self.signals_raw 
                if self.signals_raw > 0 else 0
            ),
            'active_levels': len(self.active_levels),
            'open_orders': self.open_orders_count,
            'cvd_median': self.cvd_tracker.get_median(),
            'ts_median': self.ts_tracker.get_median()
        }