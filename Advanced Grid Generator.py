#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Grid Generator with Monte Carlo and Dynamic Adjustment
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


class AdvancedGridGenerator:
    """
    Generate optimized grid levels using multiple strategies
    """
    
    def __init__(self, symbol: str, exchange, budget: float):
        self.symbol = symbol
        self.exchange = exchange
        self.budget = budget
        
    def generate_fibonacci_grid(self, current_price: float, 
                               levels: int = 6, 
                               max_drawdown: float = 0.10) -> pd.DataFrame:
        """Generate grid using Fibonacci retracement levels"""
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 0.886][:levels]
        
        grid_data = []
        allocation_per_level = self.budget / levels
        
        for i, ratio in enumerate(fib_ratios):
            buy_price = current_price * (1 - ratio * max_drawdown)
            
            # Dynamic TP based on position in grid
            tp_pct = 0.01 + (0.005 * i)  # 1% to 3.5%
            
            grid_data.append({
                'buy_price': buy_price,
                'coin_size': allocation_per_level / buy_price,
                'tp_price': buy_price * (1 + tp_pct),
                'tp_pct': tp_pct
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_volatility_adjusted_grid(self, 
                                         current_price: float,
                                         historical_data: pd.DataFrame,
                                         levels: int = 6) -> pd.DataFrame:
        """Generate grid based on historical volatility"""
        
        # Calculate volatility metrics
        returns = historical_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # ATR for dynamic spacing
        atr = self._calculate_atr(historical_data)
        atr_pct = atr / current_price
        
        grid_data = []
        allocation_per_level = self.budget / levels
        
        # Use ATR multiples for grid spacing
        for i in range(levels):
            multiplier = 1 + (i * 0.5)  # 0.5, 1, 1.5, 2, 2.5, 3 ATR
            buy_price = current_price - (atr * multiplier)
            
            # Dynamic TP based on volatility
            tp_pct = max(0.01, min(0.05, volatility * 3))  # 1% to 5%
            
            grid_data.append({
                'buy_price': buy_price,
                'coin_size': allocation_per_level / buy_price,
                'tp_price': buy_price * (1 + tp_pct),
                'tp_pct': tp_pct
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_monte_carlo_grid(self,
                                 historical_data: pd.DataFrame,
                                 n_simulations: int = 1000,
                                 levels: int = 6,
                                 confidence: float = 0.95) -> pd.DataFrame:
        """
        Generate grid using Monte Carlo simulation
        Find optimal entry points based on probability
        """
        
        # Prepare historical returns
        returns = historical_data['close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        current_price = historical_data['close'].iloc[-1]
        
        # Run simulations
        n_steps = 100  # Simulate next 100 periods
        simulated_paths = np.zeros((n_simulations, n_steps))
        
        for sim in range(n_simulations):
            price = current_price
            for step in range(n_steps):
                daily_return = np.random.normal(mu, sigma)
                price = price * (1 + daily_return)
                simulated_paths[sim, step] = price
        
        # Find percentile levels for entry points
        percentiles = np.linspace(5, 40, levels)  # 5th to 40th percentile
        entry_levels = []
        
        for p in percentiles:
            level = np.percentile(simulated_paths[:, -1], p)
            entry_levels.append(level)
        
        # Calculate win probability for each level
        grid_data = []
        allocation_per_level = self.budget / levels
        
        for i, buy_price in enumerate(entry_levels):
            # Calculate probability of profit
            win_prob = np.mean(simulated_paths[:, -1] > buy_price * 1.015)
            
            # Adjust TP based on win probability
            if win_prob > 0.8:
                tp_pct = 0.01  # High probability, lower TP
            elif win_prob > 0.6:
                tp_pct = 0.015
            else:
                tp_pct = 0.02  # Lower probability, higher TP
            
            grid_data.append({
                'buy_price': buy_price,
                'coin_size': allocation_per_level / buy_price,
                'tp_price': buy_price * (1 + tp_pct),
                'tp_pct': tp_pct,
                'win_probability': win_prob
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_support_resistance_grid(self,
                                        historical_data: pd.DataFrame,
                                        levels: int = 6) -> pd.DataFrame:
        """Generate grid based on support/resistance levels"""
        
        # Find support levels using pivot points
        highs = historical_data['high'].rolling(5).max()
        lows = historical_data['low'].rolling(5).min()
        
        # Find local minima as support levels
        support_levels = []
        for i in range(10, len(lows) - 10):
            if lows.iloc[i] == lows.iloc[i-10:i+10].min():
                support_levels.append(lows.iloc[i])
        
        support_levels = sorted(list(set(support_levels)))[-levels:]
        
        if len(support_levels) < levels:
            # Fallback to percentage-based
            current_price = historical_data['close'].iloc[-1]
            for i in range(levels - len(support_levels)):
                support_levels.append(current_price * (0.95 - i * 0.02))
        
        # Create grid
        grid_data = []
        allocation_per_level = self.budget / len(support_levels)
        
        for buy_price in support_levels:
            grid_data.append({
                'buy_price': buy_price,
                'coin_size': allocation_per_level / buy_price,
                'tp_price': buy_price * 1.015,
                'tp_pct': 0.015
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_kelly_criterion_grid(self,
                                     historical_data: pd.DataFrame,
                                     win_rate: float = 0.6,
                                     avg_win: float = 0.015,
                                     avg_loss: float = 0.01,
                                     levels: int = 6) -> pd.DataFrame:
        """Generate grid with Kelly Criterion position sizing"""
        
        # Calculate Kelly fraction
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_f = min(0.25, max(0, kelly_f))  # Cap at 25%
        
        current_price = historical_data['close'].iloc[-1]
        
        # Generate levels with Kelly-adjusted sizing
        grid_data = []
        base_allocation = self.budget * kelly_f
        
        for i in range(levels):
            # Geometric spacing
            buy_price = current_price * (0.98 ** (i + 1))
            
            # Adjust allocation based on distance from current price
            distance_factor = 1 + (i * 0.1)  # Allocate more to lower levels
            level_allocation = (base_allocation / levels) * distance_factor
            
            grid_data.append({
                'buy_price': buy_price,
                'coin_size': level_allocation / buy_price,
                'tp_price': buy_price * (1 + avg_win),
                'tp_pct': avg_win,
                'kelly_fraction': kelly_f * distance_factor
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_hybrid_optimized_grid(self,
                                      current_price: float,
                                      historical_data: pd.DataFrame,
                                      levels: int = 6) -> pd.DataFrame:
        """
        Combine multiple strategies for optimal grid
        """
        
        # Generate grids using different methods
        grids = {
            'fibonacci': self.generate_fibonacci_grid(current_price, levels),
            'volatility': self.generate_volatility_adjusted_grid(current_price, historical_data, levels),
            'monte_carlo': self.generate_monte_carlo_grid(historical_data, 500, levels),
            'support': self.generate_support_resistance_grid(historical_data, levels)
        }
        
        # Combine and weight the strategies
        combined_grid = []
        
        for i in range(levels):
            # Average the buy prices with weights
            weights = {'fibonacci': 0.2, 'volatility': 0.3, 'monte_carlo': 0.3, 'support': 0.2}
            
            weighted_price = sum(
                grids[strategy].iloc[min(i, len(grids[strategy])-1)]['buy_price'] * weight
                for strategy, weight in weights.items()
            )
            
            # Use volatility-based TP
            volatility = historical_data['close'].pct_change().std()
            tp_pct = max(0.01, min(0.03, volatility * 2))
            
            allocation = self.budget / levels
            
            combined_grid.append({
                'buy_price': weighted_price,
                'coin_size': allocation / weighted_price,
                'tp_price': weighted_price * (1 + tp_pct),
                'tp_pct': tp_pct,
                'strategy': 'hybrid'
            })
        
        return pd.DataFrame(combined_grid)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr
    
    def save_grid(self, grid_df: pd.DataFrame, filename: str = 'grid_plan_optimized.csv'):
        """Save grid to CSV file"""
        grid_df.to_csv(filename, index=False)
        print(f"✅ Grid saved to {filename}")
        print("\nGrid Summary:")
        print(f"- Levels: {len(grid_df)}")
        print(f"- Total allocation: ${(grid_df['buy_price'] * grid_df['coin_size']).sum():.2f}")
        print(f"- Price range: ${grid_df['buy_price'].min():.4f} - ${grid_df['buy_price'].max():.4f}")
        print(f"- Average TP: {grid_df['tp_pct'].mean()*100:.2f}%")
        
        return grid_df


# Example usage
if __name__ == "__main__":
    import ccxt
    
    # Initialize
    exchange = ccxt.binance()
    symbol = 'ADA/USDT'
    budget = 200
    
    generator = AdvancedGridGenerator(symbol, exchange, budget)
    
    # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    current_price = df['close'].iloc[-1]
    
    print("="*60)
    print("GRID GENERATION STRATEGIES")
    print("="*60)
    
    # 1. Fibonacci Grid
    print("\n1. Fibonacci Grid:")
    fib_grid = generator.generate_fibonacci_grid(current_price)
    print(fib_grid)
    
    # 2. Volatility-Adjusted Grid
    print("\n2. Volatility-Adjusted Grid:")
    vol_grid = generator.generate_volatility_adjusted_grid(current_price, df)
    print(vol_grid)
    
    # 3. Monte Carlo Grid
    print("\n3. Monte Carlo Grid:")
    mc_grid = generator.generate_monte_carlo_grid(df)
    print(mc_grid[['buy_price', 'coin_size', 'tp_pct', 'win_probability']])
    
    # 4. Support/Resistance Grid
    print("\n4. Support/Resistance Grid:")
    sr_grid = generator.generate_support_resistance_grid(df)
    print(sr_grid)
    
    # 5. Kelly Criterion Grid
    print("\n5. Kelly Criterion Grid:")
    kelly_grid = generator.generate_kelly_criterion_grid(df)
    print(kelly_grid[['buy_price', 'coin_size', 'kelly_fraction']])
    
    # 6. Hybrid Optimized Grid
    print("\n6. HYBRID OPTIMIZED GRID (Recommended):")
    hybrid_grid = generator.generate_hybrid_optimized_grid(current_price, df)
    print(hybrid_grid)
    
    # Save the best grid
    generator.save_grid(hybrid_grid, 'grid_plan_optimized.csv')