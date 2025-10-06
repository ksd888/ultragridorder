#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions and Helper Classes
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import deque
import ccxt


def now_ms() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def ema_update(prev: Optional[float], new_val: float, span: int) -> float:
    """Update EMA value"""
    alpha = 2.0 / (span + 1)
    if prev is None:
        return new_val
    return alpha * new_val + (1 - alpha) * prev


class DataFetcher:
    """
    Fetch market data efficiently with rate limiting
    """
    
    def __init__(self, symbol: str, exchange_id: str,
                api_key: str = None, api_secret: str = None,
                rate_limit_ms: int = 200):
        self.symbol = symbol

        kwargs = {'enableRateLimit': True}
        if api_key and api_secret:
            kwargs['apiKey'] = api_key
            kwargs['secret'] = api_secret

        self.exchange = getattr(ccxt, exchange_id)(kwargs)
        self.exchange.load_markets()

        # ✅ ENHANCED: Rate limiting (เพิ่มเป็น 2 วินาที ตามบอทต้นแบบ)
        self._last_trades_fetch = 0
        self._trades_throttle_ms = 2000  # 2 seconds (ประหยัด API quota)

        # ✅ ENHANCED: Duplicate detection (timestamp + id)
        self._last_trade_ts = None
        self._last_trade_id = None

        # API call tracking
        self._api_calls = deque(maxlen=100)
    
    def fetch_orderbook(self, limit: int = 20) -> dict:
        """
        ✅ ENHANCED: Fetch orderbook with metrics calculation

        Returns:
            - mid: Mid price
            - imbalance: Order imbalance (bid/ask ratio)
            - depth_bid_5: Top 5 bid depth
            - depth_ask_5: Top 5 ask depth
            - total_bid_vol: Total bid volume
            - total_ask_vol: Total ask volume
        """
        start = time.time()

        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=limit)

            latency = (time.time() - start) * 1000
            self._api_calls.append(('orderbook', True, latency))

            # Calculate metrics
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])

            if not bids or not asks:
                return {'ok': False, 'error': 'empty_book'}

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0

            # Depth (top 5 levels)
            depth_bid_5 = float(sum([lvl[1] for lvl in bids[:5]]))
            depth_ask_5 = float(sum([lvl[1] for lvl in asks[:5]]))

            # Total volume
            total_bid_vol = float(sum([lvl[1] for lvl in bids]))
            total_ask_vol = float(sum([lvl[1] for lvl in asks]))

            # Imbalance
            denom = total_bid_vol + total_ask_vol
            imbalance = (total_bid_vol - total_ask_vol) / max(denom, 1e-12)
            imbalance = float(np.clip(imbalance, -0.99, 0.99))

            return {
                'ok': True,
                'bids': bids,
                'asks': asks,
                'timestamp': ob.get('timestamp', now_ms()),
                'mid': mid,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'imbalance': imbalance,
                'depth_bid_5': depth_bid_5,
                'depth_ask_5': depth_ask_5,
                'total_bid_vol': total_bid_vol,
                'total_ask_vol': total_ask_vol
            }

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._api_calls.append(('orderbook', False, latency))

            return {'ok': False, 'error': str(e)}
    
    def fetch_trades(self, limit: int = 100) -> List[dict]:
        """
        Fetch recent trades with throttling and duplicate detection

        ✅ ENHANCED:
        - Throttle: 2 วินาที (ประหยัด API)
        - Duplicate detection: timestamp + id (แม่นยำกว่าเดิม)
        """
        now = now_ms()

        # ✅ Throttle check
        if now - self._last_trades_fetch < self._trades_throttle_ms:
            return []

        start = time.time()

        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            self._last_trades_fetch = now

            latency = (time.time() - start) * 1000
            self._api_calls.append(('trades', True, latency))

            # ✅ ENHANCED: Filter duplicates (timestamp + id)
            filtered = []
            for t in trades:
                tid = str(t.get('id', ''))
                ts = int(t.get('timestamp') or now)

                # กรองด้วย timestamp ก่อน
                if self._last_trade_ts is not None:
                    if ts < self._last_trade_ts:
                        continue
                    # ถ้า timestamp เท่ากัน เทียบ id
                    if ts == self._last_trade_ts and self._last_trade_id and tid <= self._last_trade_id:
                        continue

                filtered.append(t)

            # Update last seen
            if filtered:
                self._last_trade_ts = filtered[-1].get('timestamp')
                self._last_trade_id = str(filtered[-1].get('id', ''))

            return filtered

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._api_calls.append(('trades', False, latency))

            return []
    
    def fetch_ohlcv(self, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        start = time.time()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            
            latency = (time.time() - start) * 1000
            self._api_calls.append(('ohlcv', True, latency))
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._api_calls.append(('ohlcv', False, latency))
            
            return pd.DataFrame()
    
    def get_api_stats(self) -> Dict:
        """Get API call statistics"""
        if not self._api_calls:
            return {}
        
        successful = [c for c in self._api_calls if c[1]]
        failed = [c for c in self._api_calls if not c[1]]
        
        latencies = [c[2] for c in successful]
        
        return {
            'total_calls': len(self._api_calls),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self._api_calls),
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'max_latency_ms': np.max(latencies) if latencies else 0
        }


class Aggregator5s:
    """
    ✅ ENHANCED: 5-second bar aggregation from orderbook and trades

    Improvements:
    - เก็บหลาย orderbook snapshots ต่อ bar (ลด noise)
    - คำนวณค่าเฉลี่ยของ depth, imbalance
    - ดีกว่าการใช้ snapshot เดียว
    """

    def __init__(self, bar_ms: int = 5000):
        self.bar_ms = bar_ms
        self.bar_start = 0
        self.ob_snapshots = []  # ✅ เก็บหลาย snapshots
        self.trades = []

    def add_orderbook_snapshot(self, ob: dict):
        """
        ✅ ENHANCED: Add orderbook snapshot

        เก็บทุก snapshot ที่มีใน 5 วินาที
        แล้วคำนวณค่าเฉลี่ยตอน roll_bar()
        """
        if ob.get("ok"):
            self.ob_snapshots.append(ob)
    
    def add_trades(self, trades: List[dict]):
        """Add trades"""
        self.trades.extend(trades)
    
    def roll_bar(self, now_ms: int) -> Optional[Dict]:
        """
        Roll bar every 5 seconds
        Calculate CVD, imbalance, trade size proxy
        """
        if self.bar_start == 0:
            self.bar_start = (now_ms // self.bar_ms) * self.bar_ms
            return None
        
        bar_end = self.bar_start + self.bar_ms
        
        if now_ms < bar_end:
            return None
        
        # Check data availability
        if not self.ob_snapshots:
            self.bar_start = bar_end
            return {
                "ok": False,
                "bar_ts": self.bar_start,
                "mid_price": 0.0,
                "cvd": 0.0,
                "trade_size_proxy": 0.0,
                "order_imbalance": 0.5,
                "total_bid_vol": 0.0,
                "total_ask_vol": 0.0,
                "num_trades": 0,
                "data_quality": "stale"
            }
        
        # ✅ ENHANCED: Calculate bar metrics (ค่าเฉลี่ยจากหลาย snapshots)
        mids = [ob.get("mid", 0) for ob in self.ob_snapshots if ob.get("mid")]
        mid_price = float(np.mean(mids)) if mids else 0.0

        # ✅ Order imbalance (ค่าเฉลี่ย - ดีกว่า last snapshot)
        imbalances = [ob.get("imbalance", 0.5) for ob in self.ob_snapshots if "imbalance" in ob]
        order_imbalance = float(np.mean(imbalances)) if imbalances else 0.5

        # ✅ Total volumes (ค่าเฉลี่ย)
        total_bid_vol = float(np.mean([
            ob.get("total_bid_vol", 0) for ob in self.ob_snapshots
        ]))
        total_ask_vol = float(np.mean([
            ob.get("total_ask_vol", 0) for ob in self.ob_snapshots
        ]))
        
        # CVD calculation
        cvd = 0.0
        for trade in self.trades:
            weight = 1 + order_imbalance if trade['side'] == 'buy' else 1 - order_imbalance
            amount = trade['amount'] * weight
            cvd += amount if trade['side'] == 'buy' else -amount
        
        cvd = float(cvd)
        
        # Trade size proxy (from depth)
        depth_bid_5 = float(np.mean([
            ob.get("depth_bid_5", 0) for ob in self.ob_snapshots
        ]))
        trade_size_proxy = float(depth_bid_5 * (1 + order_imbalance))
        
        # Data quality assessment
        data_quality = "ok"
        if len(self.trades) < 10:
            data_quality = "partial"
        if len(self.ob_snapshots) < 3:
            data_quality = "partial"
        
        bar = {
            "ok": True,
            "bar_ts": self.bar_start,
            "mid_price": mid_price,
            "cvd": cvd,
            "trade_size_proxy": trade_size_proxy,
            "order_imbalance": order_imbalance,
            "total_bid_vol": total_bid_vol,
            "total_ask_vol": total_ask_vol,
            "num_trades": len(self.trades),
            "data_quality": data_quality
        }
        
        # Reset for next bar
        self.bar_start = bar_end
        self.ob_snapshots = []
        self.trades = []
        
        return bar


class GridPlanner:
    """
    Load and manage grid levels from CSV
    """
    
    def __init__(self, config):
        self.cfg = config
        self.grid_df: Optional[pd.DataFrame] = None
        self.last_reload = 0
    
    def load_from_csv(self, csv_path: str) -> List[float]:
        """Load grid levels from CSV"""
        try:
            df = pd.read_csv(csv_path)
            
            required_cols = {'buy_price', 'coin_size'}
            if not required_cols.issubset(df.columns):
                print(f"[WARNING] {csv_path} missing required columns: {required_cols}")
                return []
            
            self.grid_df = df
            levels = sorted(df['buy_price'].dropna().unique().tolist())
            
            print(f"[OK] Loaded {len(levels)} grid levels from {csv_path}")
            
            return levels
        
        except FileNotFoundError:
            print(f"[WARNING] Grid CSV not found: {csv_path}")
            return []
        
        except Exception as e:
            print(f"[ERROR] Error loading grid CSV: {e}")
            return []
    
    def get_level_info(self, level: float, tolerance: float = 0.0015) -> Optional[Dict]:
        """Get coin_size and tp_price for a grid level"""
        if self.grid_df is None:
            return None
        
        matches = self.grid_df[
            np.abs(self.grid_df['buy_price'] - level) / level <= tolerance
        ]
        
        if len(matches) == 0:
            return None
        
        # Take first match
        row = matches.iloc[0]
        
        return {
            'buy_price': float(row['buy_price']),
            'coin_size': float(row['coin_size']),
            'tp_price': float(row.get('tp_price', row['buy_price'] * 1.015)),
            'tp_pct': float(row.get('tp_pct', 0.015))
        }
    
    def maybe_reload(self) -> Optional[List[float]]:
        """Reload grid if reload interval passed"""
        if self.cfg.grid_reload_sec <= 0:
            return None
        
        if time.time() - self.last_reload < self.cfg.grid_reload_sec:
            return None
        
        levels = self.load_from_csv(self.cfg.grid_csv)
        self.last_reload = time.time()
        
        return levels if levels else None


class PerformanceTracker:
    """
    Track trading performance metrics
    """
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        self.trades = []
        self.equity_curve = []
        
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def record_trade(self, entry_price: float, exit_price: float, 
                    size: float, side: str = 'long'):
        """Record completed trade"""
        if side == 'long':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        self.total_pnl += pnl
        self.current_balance += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.trades.append({
            'timestamp': datetime.now(timezone.utc),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl
        })
        
        self.equity_curve.append({
            'timestamp': datetime.now(timezone.utc),
            'balance': self.current_balance
        })
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0
            }
        
        win_rate = self.winning_trades / self.total_trades
        roi = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Drawdown
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Sharpe (simplified)
        if len(self.trades) > 1:
            returns = [t['pnl'] / self.initial_balance for t in self.trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_balance': self.current_balance,
            'roi': roi,
            'drawdown': drawdown,
            'sharpe_ratio': sharpe
        }