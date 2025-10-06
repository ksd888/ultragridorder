#!/usr/bin/env python3
"""
Health check script for monitoring
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

def check_bot_health():
    health = {
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'checks': {}
    }

    # Check if state DB exists
    state_db = Path('state/bot_state.db')
    if state_db.exists():
        health['checks']['state_db'] = 'exists'

        # Check last update with error handling
        try:
            conn = sqlite3.connect(state_db)

            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='state_snapshots'"
            )
            if not cursor.fetchone():
                health['status'] = 'critical'
                health['checks']['state_db'] = 'missing_table'
                conn.close()
                return health

            # Get last update
            cursor = conn.execute(
                "SELECT MAX(timestamp) FROM state_snapshots"
            )
            last_update = cursor.fetchone()[0]
            conn.close()

            if last_update:
                try:
                    last_dt = datetime.fromisoformat(last_update)
                    age_minutes = (datetime.now() - last_dt).total_seconds() / 60

                    if age_minutes < 5:
                        health['status'] = 'healthy'
                    elif age_minutes < 15:
                        health['status'] = 'warning'
                    else:
                        health['status'] = 'critical'

                    health['checks']['last_update'] = f"{age_minutes:.1f} minutes ago"
                except (ValueError, TypeError) as e:
                    health['status'] = 'critical'
                    health['checks']['last_update'] = f'invalid_timestamp: {str(e)}'
            else:
                health['status'] = 'warning'
                health['checks']['last_update'] = 'no_data'

        except sqlite3.Error as e:
            health['status'] = 'critical'
            health['checks']['state_db'] = f'db_error: {str(e)}'
        except Exception as e:
            health['status'] = 'critical'
            health['checks']['state_db'] = f'unexpected_error: {str(e)}'
    else:
        health['status'] = 'critical'
        health['checks']['state_db'] = 'missing'
    
    # Check log file
    log_file = Path('logs/decisions.csv')
    if log_file.exists():
        health['checks']['log_file'] = 'exists'
    else:
        health['checks']['log_file'] = 'missing'
    
    return health

if __name__ == "__main__":
    health = check_bot_health()
    print(json.dumps(health, indent=2))
    
    # Exit code for monitoring systems
    if health['status'] == 'healthy':
        exit(0)
    elif health['status'] == 'warning':
        exit(1)
    else:
        exit(2)