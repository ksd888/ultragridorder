from monte_carlo import MonteCarloOptimizer
from config import load_config
import ccxt

# Optimize grid before starting
exchange = ccxt.binance()
optimizer = MonteCarloOptimizer('ADA/USDT', exchange)
optimizer.optimize_grid_levels(200, 6)

# Then start bot
from trader import UltraGridTrader
cfg = load_config()
bot = UltraGridTrader(cfg)
bot.run()