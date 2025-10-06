#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Trading Bot with Dry-Run Mode
Safe testing with Monte Carlo generated grid
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trader import UltraGridTrader
from config import load_config
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"[WARNING] Could not load .env: {e}")

def main():
    print("=" * 60)
    print("ULTRA GRID TRADING BOT - DRY RUN TEST")
    print("=" * 60)
    print()
    print("[SAFE MODE] Running in DRY RUN - No real trades will be executed!")
    print()

    # Load test config
    cfg = load_config("config_test.yaml")

    # Skip loading config.yaml if not needed
    if os.path.exists('config.yaml'):
        os.rename('config.yaml', 'config.yaml.bak')

    # Confirm dry-run mode
    if not cfg.dry_run:
        print("[ERROR] Config must be in dry_run mode for testing!")
        print("Set 'dry_run: true' in config_test.yaml")
        return

    print(f"[CONFIG] Symbol: {cfg.symbol}")
    print(f"[CONFIG] Grid CSV: {cfg.grid_csv}")
    print(f"[CONFIG] Budget: ${cfg.budget_usdt}")
    print(f"[CONFIG] Max Orders: {cfg.max_open_orders}")
    print(f"[CONFIG] TP: {cfg.fixed_tp_pct * 100}%")
    print(f"[CONFIG] Dry Run: {cfg.dry_run}")
    print()
    print("=" * 60)
    print("Starting bot... Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Get API credentials (reload .env to be safe)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except:
        pass

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    if not api_key or not api_secret:
        print("[WARNING] Binance API keys not found in environment")
        print("Set environment variables:")
        print("  BINANCE_API_KEY=your_key")
        print("  BINANCE_SECRET=your_secret")
        print()
        print("Continuing without API keys (public data only)...")

    # Create and run bot
    try:
        bot = UltraGridTrader(cfg, api_key, api_secret)
        bot.run()
    except KeyboardInterrupt:
        print("\n\n[STOP] Bot stopped by user")
        print("=" * 60)
    except Exception as e:
        print(f"\n\n[ERROR] Bot crashed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
    finally:
        # Restore config.yaml if backed up
        if os.path.exists('config.yaml.bak'):
            os.rename('config.yaml.bak', 'config.yaml')

if __name__ == "__main__":
    main()
