#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dashboard with Mock Data
Creates sample data for testing the real-time dashboard
"""

import os
import json
import sqlite3
from datetime import datetime, timezone, timedelta

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("Creating mock data for dashboard testing...")

# 1. Create mock database
db_path = "data/bot_state.db"
conn = sqlite3.connect(db_path)

# Create tables
conn.execute("""
CREATE TABLE IF NOT EXISTS state_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    active_positions INTEGER,
    total_exposure REAL,
    active_levels INTEGER,
    circuit_breaker INTEGER
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT,
    entry_price REAL,
    size_coin REAL,
    tp_price REAL,
    sl_price REAL,
    entry_time TEXT,
    status TEXT
)
""")

# Insert mock state
conn.execute("""
INSERT INTO state_snapshots (timestamp, active_positions, total_exposure, active_levels, circuit_breaker)
VALUES (?, 3, 150.50, 12, 0)
""", (datetime.now(timezone.utc).isoformat(),))

# Insert mock positions
positions = [
    ("pos_001", "ADA/USDT", 0.5200, 100.0, 0.5280, 0.5100, (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), "open"),
    ("pos_002", "ADA/USDT", 0.5100, 50.0, 0.5175, 0.5000, (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), "open"),
    ("pos_003", "ADA/USDT", 0.5000, 30.0, 0.5075, 0.4900, datetime.now(timezone.utc).isoformat(), "open"),
]

for pos in positions:
    conn.execute("""
    INSERT OR REPLACE INTO positions
    (position_id, symbol, entry_price, size_coin, tp_price, sl_price, entry_time, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, pos)

conn.commit()
conn.close()

print("[OK] Created mock database: data/bot_state.db")

# 2. Create mock structured log
log_file = "logs/bot.jsonl"

trades = []
base_time = datetime.now(timezone.utc) - timedelta(minutes=30)

for i in range(10):
    trade_time = base_time + timedelta(minutes=i * 3)
    side = "BUY" if i % 2 == 0 else "SELL"
    price = 0.52 + (i * 0.001)

    trades.append({
        "timestamp": trade_time.isoformat(),
        "level": "INFO",
        "component": "execution",
        "event_type": "trade_executed",
        "side": side,
        "symbol": "ADA/USDT",
        "price": price,
        "quantity": 50.0,
        "notional": price * 50.0
    })

with open(log_file, 'w') as f:
    for trade in trades:
        f.write(json.dumps(trade) + '\n')

print(f"[OK] Created mock trades log: {log_file}")

# 3. Create mock decisions log
import pandas as pd
import numpy as np

decisions_data = []
base_time = datetime.now(timezone.utc) - timedelta(minutes=60)

for i in range(60):
    row_time = base_time + timedelta(minutes=i)

    decisions_data.append({
        'bar_time_utc': row_time.strftime('%Y-%m-%d %H:%M:%S'),
        'mid': 0.52 + np.random.uniform(-0.01, 0.01),
        'cvd': np.random.uniform(-100, 100),
        'cvd_z': np.random.uniform(-3, 3),
        'order_imbalance': np.random.uniform(-0.5, 0.5),
        'buy_signal_raw': 0,
        'buy_signal_confirmed': 0
    })

df = pd.DataFrame(decisions_data)
df.to_csv('logs/decisions.csv', index=False)

print("[OK] Created mock decisions log: logs/decisions.csv")

# 4. Create mock dust ledger
dust_data = [
    {
        'timestamp': (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
        'symbol': 'ADA/USDT',
        'remainder_qty': 0.05,
        'unit_cost_usdt': 0.52,
        'est_cost_total_usdt': 0.026,
        'reason': 'minQty_gate',
        'order_id': 'order_001',
        'planned_tp': 0.5278
    },
    {
        'timestamp': (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        'symbol': 'ADA/USDT',
        'remainder_qty': 0.08,
        'unit_cost_usdt': 0.51,
        'est_cost_total_usdt': 0.0408,
        'reason': 'lot_rounding',
        'order_id': 'order_002',
        'planned_tp': 0.5177
    }
]

dust_df = pd.DataFrame(dust_data)
dust_df.to_csv('logs/dust_ledger.csv', index=False)

print("[OK] Created mock dust ledger: logs/dust_ledger.csv")

print("\n" + "=" * 60)
print("Mock data creation complete!")
print("=" * 60)
print("\nNow you can test the dashboard:")
print("  python dashboard_realtime.py")
print("\nThen open: http://localhost:8050")
print("=" * 60)
