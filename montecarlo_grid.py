#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Grid Level Generator
Simplified version for Ultra Grid Trading System

Features:
- Block bootstrap simulation for price projection
- Adaptive grid spacing based on volatility
- Respects exchange min notional and lot size
- Monotone allocation (more budget at lower prices)
"""

import numpy as np
import pandas as pd
import ccxt
from typing import Tuple, List
from datetime import datetime


class MonteCarloGridGenerator:
    """
    Generate grid levels using Monte Carlo simulation

    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair (e.g., 'ADA/USDT')
        budget_usd: Total budget in USDT
        tp_pct: Take profit percentage (e.g., 0.015 for 1.5%)
        lookback_days: Historical data days for simulation
        num_paths: Number of Monte Carlo paths
        horizon_days: Forecast horizon in days
    """

    def __init__(self,
                 exchange,
                 symbol: str,
                 budget_usd: float,
                 tp_pct: float = 0.015,
                 lookback_days: int = 365,
                 num_paths: int = 10000,
                 horizon_days: int = 60):

        self.exchange = exchange
        self.symbol = symbol
        self.budget_usd = budget_usd
        self.tp_pct = tp_pct
        self.lookback_days = lookback_days
        self.num_paths = num_paths
        self.horizon_days = horizon_days

        # Get market info
        self.market_info = exchange.markets.get(symbol, {})
        self.min_notional, self.min_qty, self.tick_size, self.lot_size = self._get_market_meta()

        print(f"Monte Carlo Grid Generator Initialized")
        print(f"  Symbol: {symbol}")
        print(f"  Budget: ${budget_usd:.2f}")
        print(f"  TP: {tp_pct * 100:.2f}%")
        print(f"  Min Notional: ${self.min_notional:.2f}")
        print(f"  Min Qty: {self.min_qty:.8f}")

    def _get_market_meta(self) -> Tuple[float, float, float, float]:
        """Extract market metadata"""
        # Tick size
        tick_size = float(self.market_info.get('precision', {}).get('price', 1e-8))

        # Lot size
        lot_size = float(self.market_info.get('precision', {}).get('amount', 1e-8))

        # Min notional
        min_notional = float(self.market_info.get('limits', {}).get('cost', {}).get('min', 10.0))

        # Min quantity
        min_qty = float(self.market_info.get('limits', {}).get('amount', {}).get('min', 0.0001))

        return min_notional, min_qty, tick_size, lot_size

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        print(f"Fetching {self.lookback_days} days of historical data...")

        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe='1d',
            limit=self.lookback_days
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['returns'] = df['close'].pct_change()

        print(f"[OK] Loaded {len(df)} bars")

        return df

    def run_monte_carlo(self, current_price: float, historical_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Run Monte Carlo simulation to project price range

        Returns:
            (lower_bound, upper_bound) - 10th and 90th percentiles
        """
        print(f"Running Monte Carlo simulation ({self.num_paths} paths, {self.horizon_days} days)...")

        returns = historical_data['returns'].dropna().values

        if len(returns) < 30:
            raise ValueError("Insufficient historical data for simulation")

        # Block bootstrap parameters
        block_len = 24  # ~1 month blocks
        horizon = self.horizon_days

        # Generate paths using block bootstrap
        final_prices = []

        for _ in range(self.num_paths):
            path_return = 1.0

            # Sample blocks
            for _ in range(0, horizon, block_len):
                # Random block start
                start_idx = np.random.randint(0, len(returns) - block_len)
                block = returns[start_idx:start_idx + block_len]

                # Compound returns
                path_return *= (1 + block).prod()

            final_price = current_price * path_return
            final_prices.append(final_price)

        final_prices = np.array(final_prices)

        # Calculate percentiles
        lower_bound = np.percentile(final_prices, 10)
        upper_bound = np.percentile(final_prices, 90)

        # Add stretch to lower bound (more conservative)
        lower_bound *= 0.85  # 15% stretch downward

        print(f"[OK] Price range: ${lower_bound:.6f} - ${upper_bound:.6f}")

        return lower_bound, upper_bound

    def generate_grid_levels(self,
                            lower_price: float,
                            upper_price: float,
                            num_levels: int = None) -> pd.DataFrame:
        """
        Generate grid levels with monotone allocation

        Args:
            lower_price: Lower price bound
            upper_price: Upper price bound
            num_levels: Number of grid levels (auto if None)

        Returns:
            DataFrame with columns: buy_price, coin_size, tp_price, tp_pct, notional
        """
        # Auto-calculate num_levels if not provided
        if num_levels is None:
            # Target 2x minNotional per level on average
            target_per_level = self.min_notional * 2.0
            num_levels = int(self.budget_usd / target_per_level)
            num_levels = max(10, min(num_levels, 200))  # Between 10-200

        print(f"Generating {num_levels} grid levels...")

        # Create linear grid
        raw_levels = np.linspace(lower_price, upper_price, num_levels)

        # Round to tick size
        levels = [self._round_to_tick(p) for p in raw_levels]

        # Calculate monotone allocation (more at lower prices)
        # Using exponential decay
        weights = np.exp(-np.linspace(0, 2, num_levels))
        weights = weights / weights.sum() * self.budget_usd

        grid_data = []

        for buy_price, allocated in zip(levels, weights):
            # Calculate coin size
            coin_size = allocated / buy_price

            # Round to lot size
            coin_size = self._round_to_lot(coin_size)

            # Check minimums
            notional = coin_size * buy_price

            if notional < self.min_notional or coin_size < self.min_qty:
                continue  # Skip levels that don't meet minimums

            # Calculate TP
            tp_price = buy_price * (1 + self.tp_pct)
            tp_price = self._round_to_tick(tp_price)

            grid_data.append({
                'buy_price': buy_price,
                'coin_size': coin_size,
                'tp_price': tp_price,
                'tp_pct': self.tp_pct,
                'notional': notional
            })

        df = pd.DataFrame(grid_data)

        # Adjust allocation to match budget
        total_allocated = df['notional'].sum()

        if total_allocated > 0:
            scale_factor = self.budget_usd / total_allocated
            df['coin_size'] *= scale_factor
            df['coin_size'] = df['coin_size'].apply(self._round_to_lot)
            df['notional'] = df['buy_price'] * df['coin_size']

        print(f"[OK] Generated {len(df)} valid levels")
        print(f"  Total allocation: ${df['notional'].sum():.2f}")
        print(f"  Avg per level: ${df['notional'].mean():.2f}")

        return df

    def _round_to_tick(self, price: float) -> float:
        """Round price to tick size (floor)"""
        if self.tick_size <= 0:
            return price

        return float(np.floor(price / self.tick_size) * self.tick_size)

    def _round_to_lot(self, quantity: float) -> float:
        """Round quantity to lot size (floor)"""
        if self.lot_size <= 0:
            return quantity

        return float(np.floor(quantity / self.lot_size) * self.lot_size)

    def generate_and_save(self, output_csv: str = "grid_plan_mc.csv") -> pd.DataFrame:
        """
        Full workflow: fetch data, simulate, generate grid, save

        Returns:
            Grid DataFrame
        """
        print("\n" + "=" * 60)
        print("MONTE CARLO GRID GENERATION")
        print("=" * 60)

        # Get current price
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = float(ticker['last'])

        print(f"Current price: ${current_price:.6f}")

        # Fetch historical data
        historical_data = self.fetch_historical_data()

        # Run Monte Carlo
        lower_price, upper_price = self.run_monte_carlo(current_price, historical_data)

        # Generate grid
        grid_df = self.generate_grid_levels(lower_price, upper_price)

        # Save to CSV
        grid_df.to_csv(output_csv, index=False)
        print(f"\n[OK] Grid saved to {output_csv}")

        # Summary
        print("\n" + "=" * 60)
        print("GRID SUMMARY")
        print("=" * 60)
        print(f"Levels: {len(grid_df)}")
        print(f"Price range: ${grid_df['buy_price'].min():.6f} - ${grid_df['buy_price'].max():.6f}")
        print(f"Total budget: ${grid_df['notional'].sum():.2f}")
        print(f"Avg allocation: ${grid_df['notional'].mean():.2f}")
        print(f"Min allocation: ${grid_df['notional'].min():.2f}")
        print(f"Max allocation: ${grid_df['notional'].max():.2f}")
        print("=" * 60)

        return grid_df


# ================= Main Entry Point =================

if __name__ == '__main__':
    import sys

    # Configuration
    SYMBOL = "ADA/USDT"
    BUDGET = 300.0
    TP_PCT = 0.015  # 1.5%
    LOOKBACK_DAYS = 365
    NUM_PATHS = 10000
    HORIZON_DAYS = 60

    # Initialize exchange
    print("Initializing exchange...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })

    exchange.load_markets()

    # Create generator
    generator = MonteCarloGridGenerator(
        exchange=exchange,
        symbol=SYMBOL,
        budget_usd=BUDGET,
        tp_pct=TP_PCT,
        lookback_days=LOOKBACK_DAYS,
        num_paths=NUM_PATHS,
        horizon_days=HORIZON_DAYS
    )

    # Generate and save
    grid_df = generator.generate_and_save()

    print("\n[SUCCESS] Monte Carlo grid generation complete!")
