#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch to add Monte Carlo integration to trader.py
Add this to the __init__ method of UltraGridTrader class
"""

# Add these imports at the top of trader.py
from monte_carlo import MonteCarloOptimizer
import pandas as pd

# Add this method to UltraGridTrader class
def optimize_grid_with_monte_carlo(self):
    """
    Optimize grid using Monte Carlo simulation
    Called during initialization or periodically
    """
    if not hasattr(self.cfg, 'monte_carlo') or not self.cfg.monte_carlo.get('enabled', False):
        print("Monte Carlo optimization disabled")
        return None
    
    print("\n📊 Running Monte Carlo Grid Optimization...")
    
    try:
        # Initialize optimizer
        optimizer = MonteCarloOptimizer(
            self.cfg.symbol, 
            self.fetcher.exchange,
            lookback_days=self.cfg.monte_carlo.get('lookback_days', 30)
        )
        
        # Get current price
        ticker = self.fetcher.exchange.fetch_ticker(self.cfg.symbol)
        current_price = ticker['last']
        
        # Run simulation
        n_simulations = self.cfg.monte_carlo.get('n_simulations', 1000)
        paths = optimizer.simulate_paths(current_price, n_simulations)
        
        # Generate optimized grid
        grid_df = optimizer.optimize_grid_levels(
            self.cfg.budget_usdt,
            self.cfg.max_grid_levels
        )
        
        # Update grid levels in engine
        new_levels = grid_df['buy_price'].tolist()
        self.engine.grid_levels_open = sorted(new_levels)
        
        # Save to CSV
        grid_df.to_csv(self.cfg.grid_csv, index=False)
        
        print(f"✅ Monte Carlo optimization complete:")
        print(f"   - Simulations: {n_simulations}")
        print(f"   - Grid levels: {len(new_levels)}")
        print(f"   - Price range: ${min(new_levels):.4f} - ${max(new_levels):.4f}")
        print(f"   - Avg win probability: {grid_df['win_probability'].mean():.1%}")
        print(f"   - Saved to: {self.cfg.grid_csv}\n")
        
        return grid_df
        
    except Exception as e:
        print(f"❌ Monte Carlo optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Add this to __init__ method after loading grid (around line 85-90)
# Replace this section in __init__:
"""
# Load grid
self.grid_planner = GridPlanner(config)
grid_levels = self._load_grid()
"""

# With this:
"""
# Load grid with Monte Carlo optimization
self.grid_planner = GridPlanner(config)

# Try Monte Carlo optimization first
if config.monte_carlo.get('enabled', False):
    monte_carlo_grid = self.optimize_grid_with_monte_carlo()
    if monte_carlo_grid is not None:
        grid_levels = monte_carlo_grid['buy_price'].tolist()
    else:
        grid_levels = self._load_grid()
else:
    grid_levels = self._load_grid()
"""

# Add periodic Monte Carlo re-optimization (optional)
# Add this to _periodic_tasks() method:
"""
# Re-optimize grid with Monte Carlo (once per day)
if hasattr(self, 'last_monte_carlo') and time.time() - self.last_monte_carlo >= 86400:
    if self.cfg.monte_carlo.get('enabled', False):
        print("🔄 Daily Monte Carlo re-optimization...")
        self.optimize_grid_with_monte_carlo()
        self.last_monte_carlo = time.time()
"""

# Alternative: Dynamic grid adjustment based on market conditions
def should_reoptimize_grid(self, current_volatility: float, avg_win_rate: float) -> bool:
    """
    Determine if grid should be re-optimized based on market conditions
    """
    # Re-optimize if:
    # 1. Volatility changed significantly (>50%)
    # 2. Win rate dropped below threshold (<40%)
    # 3. Time-based (daily/weekly)
    
    if not hasattr(self, 'last_optimization_volatility'):
        self.last_optimization_volatility = current_volatility
        return False
    
    volatility_change = abs(current_volatility - self.last_optimization_volatility) / self.last_optimization_volatility
    
    if volatility_change > 0.5:
        print(f"⚠️ Volatility changed by {volatility_change:.1%}, re-optimizing grid...")
        return True
    
    if avg_win_rate < 0.4:
        print(f"⚠️ Win rate low ({avg_win_rate:.1%}), re-optimizing grid...")
        return True
    
    return False

# Complete integration example
class UltraGridTraderWithMonteCarlo:
    """
    Example of complete integration
    """
    
    def __init__(self, config, api_key=None, api_secret=None):
        # ... existing initialization ...
        
        # Monte Carlo optimizer
        self.monte_carlo_optimizer = None
        if config.monte_carlo.get('enabled', False):
            self.monte_carlo_optimizer = MonteCarloOptimizer(
                config.symbol,
                self.fetcher.exchange,
                config.monte_carlo.get('lookback_days', 30)
            )
        
        # Track optimization metrics
        self.last_optimization_time = 0
        self.optimization_count = 0
        self.optimization_history = []
    
    def adaptive_grid_management(self):
        """
        Dynamically adjust grid based on performance
        """
        # Calculate current metrics
        win_rate = self.calculate_win_rate()
        current_volatility = self.calculate_current_volatility()
        
        # Check if re-optimization needed
        if self.should_reoptimize_grid(current_volatility, win_rate):
            # Run Monte Carlo
            new_grid = self.optimize_grid_with_monte_carlo()
            
            if new_grid is not None:
                # Track optimization
                self.optimization_count += 1
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'volatility': current_volatility,
                    'win_rate': win_rate,
                    'grid_levels': len(new_grid)
                })
                
                # Apply new grid
                self.apply_new_grid(new_grid)
    
    def calculate_win_rate(self) -> float:
        """Calculate recent win rate from positions"""
        if not self.positions:
            return 0.5
        
        recent_positions = list(self.positions.values())[-20:]  # Last 20 trades
        wins = sum(1 for p in recent_positions if p.current_pnl > 0)
        
        return wins / len(recent_positions) if recent_positions else 0.5
    
    def calculate_current_volatility(self) -> float:
        """Calculate current market volatility"""
        ohlcv = self.fetcher.fetch_ohlcv('1h', 100)
        if ohlcv.empty:
            return 0.02
        
        returns = ohlcv['close'].pct_change().dropna()
        return returns.std()