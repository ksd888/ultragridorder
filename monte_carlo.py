import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import ccxt

class MonteCarloOptimizer:
    """Monte Carlo simulation for grid optimization"""
    
    def __init__(self, symbol: str, exchange, lookback_days: int = 30):
        self.symbol = symbol
        self.exchange = exchange
        self.lookback_days = lookback_days
        
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """Fetch historical data for Monte Carlo"""
        limit = self.lookback_days * 24  # Hourly data
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['returns'] = df['close'].pct_change()
        return df
    
    def simulate_paths(self, current_price: float, n_paths: int = 1000, n_steps: int = 24) -> np.ndarray:
        """Simulate future price paths"""
        df = self.fetch_and_prepare_data()
        returns = df['returns'].dropna()
        
        # Calculate parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Initialize paths
        dt = 1  # 1 hour steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = current_price
        
        # Generate paths using Geometric Brownian Motion
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return paths
    
    def optimize_grid_levels(self, budget: float, n_levels: int = 6) -> pd.DataFrame:
        """Optimize grid levels based on Monte Carlo results"""
        # Get current price
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']
        
        # Run simulation
        paths = self.simulate_paths(current_price)
        
        # Find optimal entry points (percentiles of final prices)
        final_prices = paths[:, -1]
        percentiles = np.linspace(10, 40, n_levels)  # 10th to 40th percentile
        
        grid_levels = []
        for i, p in enumerate(percentiles):
            buy_price = np.percentile(final_prices, p)
            
            # Calculate win probability
            win_prob = np.mean(paths[:, -1] > buy_price * 1.015)
            
            # Dynamic TP based on win probability
            if win_prob > 0.8:
                tp_pct = 0.012
            elif win_prob > 0.6:
                tp_pct = 0.015
            else:
                tp_pct = 0.020
            
            allocation = budget / n_levels
            
            grid_levels.append({
                'buy_price': buy_price,
                'coin_size': allocation / buy_price,
                'tp_price': buy_price * (1 + tp_pct),
                'tp_pct': tp_pct,
                'win_probability': win_prob
            })
        
        grid_df = pd.DataFrame(grid_levels)
        
        # Save to CSV
        grid_df.to_csv('grid_plan.csv', index=False)
        print(f"✅ Monte Carlo optimized grid saved to grid_plan.csv")
        
        return grid_df