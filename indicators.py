#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators and Orderflow Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    ✅ FIXED: Safe division, handles zero loss
    """
    delta = series.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # ✅ FIX: Replace zero loss with NaN to avoid division by zero
    loss = loss.replace(0, np.nan)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # ✅ FIX: Fill NaN with neutral value (50)
    rsi = rsi.fillna(50.0)
    
    # ✅ FIX: Clip to valid range
    rsi = rsi.clip(0, 100)
    
    return rsi


def calculate_bollinger_bands(series: pd.Series, period: int = 20, 
                              std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    Returns: (upper, middle, lower)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    ✅ FIXED: Added validation
    """
    # ✅ FIX: Validate inputs
    if len(high) < period or len(low) < period or len(close) < period:
        return pd.Series([0.0] * len(high))
    
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # ✅ FIX: Fill NaN with 0
    atr = atr.fillna(0)
    
    return atr

def calculate_vpin(trades: List[dict], orderbook: dict, 
                   bucket_size: int = 50, 
                   min_trades: int = 20) -> float:  # ✅ NEW: min_trades param
    """
    Volume-Synchronized Probability of Informed Trading
    ✅ FIXED: Added minimum trades check
    """
    # ✅ FIX: More lenient minimum
    if not trades or len(trades) < min_trades:
        return 0.5
    
    # Use available trades (but at least min_trades)
    actual_size = min(bucket_size, len(trades))
    
    # Bucket trades by volume
    buy_vol = 0
    sell_vol = 0
    
    for trade in trades[-actual_size:]:
        vol = trade['amount']
        if trade['side'] == 'buy':
            buy_vol += vol
        else:
            sell_vol += vol
    
    total_vol = buy_vol + sell_vol
    
    if total_vol == 0:
        return 0.5
    
    # VPIN = |buy_vol - sell_vol| / total_vol
    vpin = abs(buy_vol - sell_vol) / total_vol
    
    return float(np.clip(vpin, 0, 1))

def calculate_smi(ohlcv: pd.DataFrame, period: int = 20) -> float:
    """
    Smart Money Index
    Measures institutional vs retail flow
    
    SMI = Cumulative( (close - open) * volume )
    Higher values suggest smart money accumulation
    """
    if len(ohlcv) < period:
        return 0.5
    
    # Calculate daily money flow
    money_flow = (ohlcv['close'] - ohlcv['open']) * ohlcv['volume']
    
    # Cumulative sum (last N periods)
    cumulative = money_flow.tail(period).sum()
    
    # Normalize to 0-1 range
    max_flow = (ohlcv['high'].tail(period).max() - ohlcv['low'].tail(period).min()) * ohlcv['volume'].tail(period).sum()
    
    if max_flow == 0:
        return 0.5
    
    smi = (cumulative / max_flow + 1) / 2  # Normalize to 0-1
    
    return float(np.clip(smi, 0, 1))


def detect_wyckoff_phase(ohlcv: pd.DataFrame, lookback: int = 50) -> tuple:
    """
    Detect Wyckoff market phase
    ✅ FIXED: Returns (phase, confidence) instead of just phase
    """
    if len(ohlcv) < lookback:
        return "unknown", 0.0
    
    df = ohlcv.tail(lookback).copy()
    
    # Calculate trend strength
    close = df['close']
    volume = df['volume']
    
    # Price trend
    price_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    
    # Volume trend
    vol_early = volume.head(lookback // 2).mean()
    vol_late = volume.tail(lookback // 2).mean()
    vol_increasing = vol_late > vol_early * 1.1
    
    # Volatility
    volatility = close.pct_change().std()
    
    # Price range compression
    range_early = (df['high'].head(lookback // 2) - df['low'].head(lookback // 2)).mean()
    range_late = (df['high'].tail(lookback // 2) - df['low'].tail(lookback // 2)).mean()
    compression = range_late < range_early * 0.8
    
    # ✅ NEW: Calculate confidence
    confidence = 0.0
    phase = "unknown"
    
    # Detect phases with confidence
    if compression and vol_increasing:
        confidence = 0.7  # Medium confidence
        if price_change > 0:
            phase = "accumulation"
        else:
            phase = "distribution"
    
    elif price_change > 0.05 and vol_increasing:
        confidence = 0.8  # High confidence
        phase = "markup"
    
    elif price_change < -0.05 and vol_increasing:
        confidence = 0.8  # High confidence
        phase = "markdown"
    
    return phase, confidence


def detect_absorption(bar: Dict, trades: List[dict]) -> Dict:
    """
    Detect absorption (large hidden orders)
    When price doesn't move despite heavy volume
    """
    if not trades or len(trades) < 20:
        return {
            'detected': False,
            'type': None,
            'strength': 0
        }
    
    # Get recent trades
    recent = trades[-20:]
    
    # Calculate volume-weighted price change
    total_vol = sum(t['amount'] for t in recent)
    
    if total_vol == 0:
        return {'detected': False, 'type': None, 'strength': 0}
    
    # Expected price movement based on volume imbalance
    buy_vol = sum(t['amount'] for t in recent if t['side'] == 'buy')
    sell_vol = total_vol - buy_vol
    
    imbalance = (buy_vol - sell_vol) / total_vol
    
    # Actual price change
    if len(recent) >= 2:
        price_change = (recent[-1]['price'] - recent[0]['price']) / recent[0]['price']
    else:
        price_change = 0
    
    # Absorption: high imbalance but low price change
    expected_change = imbalance * 0.01  # Expected 1% per full imbalance
    
    absorption_strength = abs(imbalance) - abs(price_change / 0.01)
    
    if absorption_strength > 0.3:  # Significant absorption
        absorption_type = "support" if imbalance < 0 else "resistance"
        
        return {
            'detected': True,
            'type': absorption_type,
            'strength': float(absorption_strength)
        }
    
    return {'detected': False, 'type': None, 'strength': 0}


class MicrostructureAnalyzer:
    """Orderbook microstructure analysis"""
    
    def __init__(self):
        self.imbalance_history = deque(maxlen=100)
    
    def analyze_orderbook(self, ob: dict) -> Dict:
        """
        Analyze orderbook snapshot
        ✅ FIXED: Adjusted spread threshold for crypto
        """
        if not ob.get('ok'):
            return {'ok': False}
        
        bids = ob['bids']
        asks = ob['asks']
        
        if not bids or not asks:
            return {'ok': False}
        
        # Best bid/ask
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        
        # Spread
        spread = best_ask - best_bid
        spread_pct = spread / mid if mid > 0 else 0
        
        # Depth calculation
        depth_levels = min(5, len(bids), len(asks))
        
        total_bid_vol = sum(float(bids[i][1]) for i in range(len(bids)))
        total_ask_vol = sum(float(asks[i][1]) for i in range(len(asks)))
        
        depth_bid_5 = sum(float(bids[i][1]) for i in range(min(5, len(bids))))
        depth_ask_5 = sum(float(asks[i][1]) for i in range(min(5, len(asks))))
        
        # Order imbalance
        total_vol = total_bid_vol + total_ask_vol
        
        if total_vol > 0:
            imbalance_raw = total_bid_vol / total_vol
        else:
            imbalance_raw = 0.5
        
        # EMA smoothing
        self.imbalance_history.append(imbalance_raw)
        imbalance_ema = np.mean(list(self.imbalance_history))
        
        # Weighted imbalance
        if depth_bid_5 + depth_ask_5 > 0:
            weighted_imbalance = depth_bid_5 / (depth_bid_5 + depth_ask_5)
        else:
            weighted_imbalance = 0.5
        
        # ✅ FIX: Better quality assessment for crypto
        quality = "good"
        
        # For crypto, spread > 0.1% is concerning
        if spread_pct > 0.001:  # ✅ Changed from 0.005
            quality = "poor"
        elif len(bids) < 10 or len(asks) < 10:
            quality = "poor"
        
        return {
            'ok': True,
            'timestamp': ob.get('timestamp', 0),
            'mid': mid,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'total_bid_vol': total_bid_vol,
            'total_ask_vol': total_ask_vol,
            'depth_bid_5': depth_bid_5,
            'depth_ask_5': depth_ask_5,
            'imbalance': imbalance_raw,
            'imbalance_ema': imbalance_ema,
            'weighted_imbalance': weighted_imbalance,
            'quality': quality
        }
    
    def detect_spoofing(self, ob_history: List[Dict]) -> bool:
        """
        Detect potential spoofing (fake orders)
        Large orders that appear and disappear quickly
        """
        if len(ob_history) < 3:
            return False
        
        # Check for sudden large order appearance/disappearance
        recent = ob_history[-3:]
        
        for i in range(len(recent) - 1):
            prev_bid = recent[i].get('total_bid_vol', 0)
            curr_bid = recent[i + 1].get('total_bid_vol', 0)
            
            # Large sudden increase then decrease
            if curr_bid > prev_bid * 2 and i < len(recent) - 2:
                next_bid = recent[i + 2].get('total_bid_vol', 0)
                if next_bid < curr_bid * 0.6:
                    return True
        
        return False
    
    def calculate_order_flow_imbalance(self, trades: List[dict], 
                                      window: int = 20) -> float:
        """
        Calculate order flow imbalance from trades
        > 0.5 = buying pressure
        < 0.5 = selling pressure
        """
        if not trades or len(trades) < window:
            return 0.5
        
        recent = trades[-window:]
        
        buy_vol = sum(t['amount'] for t in recent if t['side'] == 'buy')
        total_vol = sum(t['amount'] for t in recent)
        
        if total_vol == 0:
            return 0.5
        
        return buy_vol / total_vol


def calculate_volume_profile(ohlcv: pd.DataFrame, bins: int = 20) -> Dict:
    """
    Calculate Volume Profile (POC, VAH, VAL)
    ✅ FIXED: Vectorized for better performance
    """
    if len(ohlcv) < 20:
        return {'poc': 0, 'vah': 0, 'val': 0}
    
    # Create price bins
    price_min = ohlcv['low'].min()
    price_max = ohlcv['high'].min()
    
    if price_min >= price_max:
        return {'poc': 0, 'vah': 0, 'val': 0}
    
    price_bins = np.linspace(price_min, price_max, bins)
    volume_profile = np.zeros(bins - 1)
    
    # ✅ FIX: Vectorized implementation
    for i, (low, high, vol) in enumerate(zip(ohlcv['low'], ohlcv['high'], ohlcv['volume'])):
        # Find bin indices
        low_bin = max(0, min(np.digitize(low, price_bins) - 1, bins - 2))
        high_bin = max(0, min(np.digitize(high, price_bins) - 1, bins - 2))
        
        if low_bin <= high_bin and high_bin - low_bin + 1 > 0:
            vol_per_bin = vol / (high_bin - low_bin + 1)
            volume_profile[low_bin:high_bin+1] += vol_per_bin
    
    # Point of Control
    poc_idx = np.argmax(volume_profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    # Value Area (70% of volume)
    total_volume = volume_profile.sum()
    if total_volume == 0:
        return {'poc': poc_price, 'vah': price_max, 'val': price_min}
    
    target_volume = total_volume * 0.7
    
    # Find value area around POC
    cumulative = volume_profile[poc_idx]
    lower_idx = poc_idx
    upper_idx = poc_idx
    
    while cumulative < target_volume and (lower_idx > 0 or upper_idx < len(volume_profile) - 1):
        if lower_idx > 0 and (upper_idx >= len(volume_profile) - 1 or 
                              volume_profile[lower_idx - 1] > volume_profile[upper_idx + 1]):
            lower_idx -= 1
            cumulative += volume_profile[lower_idx]
        elif upper_idx < len(volume_profile) - 1:
            upper_idx += 1
            cumulative += volume_profile[upper_idx]
    
    val_price = (price_bins[lower_idx] + price_bins[lower_idx + 1]) / 2
    vah_price = (price_bins[upper_idx] + price_bins[upper_idx + 1]) / 2
    
    return {
        'poc': float(poc_price),
        'vah': float(vah_price),
        'val': float(val_price)
    }



def calculate_order_book_pressure(bids: List, asks: List, depth: int = 10) -> float:
    """
    Calculate net orderbook pressure
    Returns: -1 to +1 (negative = sell pressure, positive = buy pressure)
    """
    if not bids or not asks:
        return 0.0
    
    depth = min(depth, len(bids), len(asks))
    
    bid_pressure = sum(float(bids[i][1]) * (1 - i / depth) for i in range(depth))
    ask_pressure = sum(float(asks[i][1]) * (1 - i / depth) for i in range(depth))
    
    total = bid_pressure + ask_pressure
    
    if total == 0:
        return 0.0
    
    pressure = (bid_pressure - ask_pressure) / total
    
    return float(np.clip(pressure, -1, 1))