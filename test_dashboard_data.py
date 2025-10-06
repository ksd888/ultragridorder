#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd

STATE_DB = os.path.join("data", "bot_state_test.db")

print(f"Checking database: {STATE_DB}")
print(f"File exists: {os.path.exists(STATE_DB)}")

if os.path.exists(STATE_DB):
    conn = sqlite3.connect(STATE_DB)

    # Check positions
    df = pd.read_sql_query(
        "SELECT * FROM positions WHERE status='open' ORDER BY entry_time DESC",
        conn
    )
    print(f"\nPositions found: {len(df)}")
    if not df.empty:
        print("\nPosition details:")
        print(df[['position_id', 'entry_price', 'size_coin', 'tp_price', 'entry_time']].to_string())

    # Check state snapshots
    cursor = conn.execute("SELECT * FROM state_snapshots ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"\nLatest state snapshot:")
        print(f"  Timestamp: {row[1]}")
        print(f"  Active positions: {row[2]}")
        print(f"  Total exposure: ${row[3]:.2f}")
        print(f"  Active levels: {row[4]}")
    else:
        print("\nNo state snapshots found")

    conn.close()
else:
    print("Database not found!")
