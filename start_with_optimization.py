#!/usr/bin/env python3
"""
Enhanced startup script with Monte Carlo optimization
"""

import sys
import os
from datetime import datetime
import ccxt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monte_carlo import MonteCarloOptimizer
from config import load_config
from trader import UltraGridTrader

def main():
    print("\n" + "="*60)
    print("ULTRA GRID V2.0 - OPTIMIZED START")
    print("="*60)
    
    # Load config
    cfg = load_config('config.yaml')
    
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET')
    
    # Initialize exchange for optimization
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    
    # Check if Monte Carlo is enabled
    if cfg.monte_carlo.get('enabled', False):
        print("\n📊 Running Monte Carlo optimization...")
        optimizer = MonteCarloOptimizer(cfg.symbol, exchange)
        
        # Optimize and save grid
        grid_df = optimizer.optimize_grid_levels(
            cfg.budget_usdt,
            cfg.max_grid_levels
        )
        
        print(f"✅ Grid optimized with {len(grid_df)} levels")
        print(f"   Win probabilities: {grid_df['win_probability'].mean():.2%} average")
        print(f"   Saved to: grid_plan.csv\n")
    else:
        print("⚠️ Monte Carlo disabled, using existing grid_plan.csv")
    
    # Start the bot
    print("🤖 Starting Ultra Grid Trader...")
    bot = UltraGridTrader(cfg, api_key, api_secret)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Graceful shutdown...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()