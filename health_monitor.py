#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Health Monitoring
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import deque

from models import HealthStatus


class HealthMonitor:
    """
    Monitor system health and detect issues
    """
    
    def __init__(self, config):
        self.cfg = config
        
        # API monitoring
        self.api_failure_count = 0
        self.last_successful_api_call = time.time()
        self.api_latencies = deque(maxlen=100)
        
        # Data monitoring
        self.last_data_timestamp = datetime.now(timezone.utc)
        self.data_staleness_sec = 0
        
        # State tracking
        self.positions_count = 0
        self.positions_value = 0.0
        self.open_orders_count = 0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.circuit_breaker_active = False
        
        # Warning history
        self.warnings = deque(maxlen=100)
        self.errors = deque(maxlen=100)
        
        # Health check history
        self.health_history = deque(maxlen=1000)
    
    def record_api_call(self, success: bool, latency_ms: float):
        """Record API call result"""
        if success:
            self.api_failure_count = 0
            self.last_successful_api_call = time.time()
            self.api_latencies.append(latency_ms)
        else:
            self.api_failure_count += 1
    
    def record_data_update(self, timestamp: datetime):
        """Record data update"""
        self.last_data_timestamp = timestamp
        self.data_staleness_sec = (
            datetime.now(timezone.utc) - timestamp
        ).total_seconds()
    
    def update_state(self, positions_count: int, positions_value: float,
                    open_orders: int):
        """Update state metrics"""
        self.positions_count = positions_count
        self.positions_value = positions_value
        self.open_orders_count = open_orders
    
    def update_performance(self, daily_pnl: float, drawdown: float,
                          breaker_active: bool):
        """Update performance metrics"""
        self.daily_pnl = daily_pnl
        self.current_drawdown = drawdown
        self.circuit_breaker_active = breaker_active
    
    def add_warning(self, message: str):
        """Add warning"""
        self.warnings.append({
            'timestamp': datetime.now(timezone.utc),
            'message': message
        })
    
    def add_error(self, message: str):
        """Add error"""
        self.errors.append({
            'timestamp': datetime.now(timezone.utc),
            'message': message
        })
    
    def check_health(self) -> HealthStatus:
        """
        Perform comprehensive health check
        """
        now = datetime.now(timezone.utc)
        
        # API connectivity
        time_since_last_call = time.time() - self.last_successful_api_call
        api_connected = (
            self.api_failure_count < self.cfg.max_api_failures and
            time_since_last_call < 300  # 5 minutes
        )
        
        avg_latency = sum(self.api_latencies) / len(self.api_latencies) if self.api_latencies else 0
        
        # Data freshness
        data_fresh = self.data_staleness_sec < self.cfg.max_data_staleness_sec
        
        # Collect warnings and errors
        current_warnings = []
        current_errors = []
        
        # Check API
        if not api_connected:
            current_errors.append(f"API disconnected (failures: {self.api_failure_count})")
        elif self.api_failure_count > 0:
            current_warnings.append(f"API failures: {self.api_failure_count}")
        
        if avg_latency > 1000:
            current_warnings.append(f"High API latency: {avg_latency:.0f}ms")
        
        # Check data
        if not data_fresh:
            current_errors.append(f"Stale data ({self.data_staleness_sec:.0f}s old)")
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            current_warnings.append("Circuit breaker active")
        
        # Check drawdown
        if self.current_drawdown > 0.05:
            current_warnings.append(f"High drawdown: {self.current_drawdown:.1%}")
        
        # Check positions
        if self.open_orders_count > self.cfg.max_open_orders:
            current_warnings.append(f"Too many open orders: {self.open_orders_count}")
        
        # Determine overall status
        if current_errors:
            status = "critical"
        elif current_warnings:
            status = "degraded"
        else:
            status = "healthy"
        
        health = HealthStatus(
            timestamp=now,
            api_connected=api_connected,
            api_latency_ms=avg_latency,
            api_failure_count=self.api_failure_count,
            data_fresh=data_fresh,
            last_data_timestamp=self.last_data_timestamp,
            data_staleness_sec=self.data_staleness_sec,
            positions_count=self.positions_count,
            positions_value_usd=self.positions_value,
            open_orders_count=self.open_orders_count,
            daily_pnl=self.daily_pnl,
            current_drawdown=self.current_drawdown,
            circuit_breaker_active=self.circuit_breaker_active,
            status=status,
            warnings=current_warnings,
            errors=current_errors
        )
        
        self.health_history.append(health)
        
        return health
    
    def print_health_report(self, health: HealthStatus):
        """Print health report"""
        status_emoji = {
            "healthy": "[OK]",
            "degraded": "[WARNING]",
            "critical": "[ERROR]"
        }
        
        print("\n" + "="*60)
        print(f"HEALTH STATUS: {status_emoji[health.status]} {health.status.upper()}")
        print("="*60)
        
        print(f"Time: {health.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"\nConnectivity:")
        print(f"  API: {'Connected' if health.api_connected else 'DISCONNECTED'}")
        print(f"  Latency: {health.api_latency_ms:.0f}ms")
        print(f"  Failures: {health.api_failure_count}")
        
        print(f"\nData Quality:")
        print(f"  Fresh: {'Yes' if health.data_fresh else 'NO'}")
        print(f"  Age: {health.data_staleness_sec:.0f}s")
        
        print(f"\nPositions:")
        print(f"  Count: {health.positions_count}")
        print(f"  Value: ${health.positions_value_usd:.2f}")
        print(f"  Orders: {health.open_orders_count}")
        
        print(f"\nPerformance:")
        print(f"  Daily P&L: ${health.daily_pnl:.2f}")
        print(f"  Drawdown: {health.current_drawdown:.2%}")
        print(f"  Breaker: {'ACTIVE' if health.circuit_breaker_active else 'OK'}")
        
        if health.warnings:
            print(f"\n[WARNING] Warnings:")
            for w in health.warnings:
                print(f"  - {w}")

        if health.errors:
            print(f"\n[ERROR] Errors:")
            for e in health.errors:
                print(f"  - {e}")
        
        print("="*60 + "\n")
    
    def get_uptime_stats(self) -> Dict:
        """Get uptime statistics"""
        if not self.health_history:
            return {}
        
        total_checks = len(self.health_history)
        healthy_checks = sum(1 for h in self.health_history if h.status == "healthy")
        
        uptime_pct = healthy_checks / total_checks if total_checks > 0 else 0
        
        # Calculate MTBF (Mean Time Between Failures)
        failure_times = []
        last_failure = None
        
        for health in self.health_history:
            if health.status == "critical":
                if last_failure:
                    failure_times.append(
                        (health.timestamp - last_failure).total_seconds()
                    )
                last_failure = health.timestamp
        
        mtbf = sum(failure_times) / len(failure_times) if failure_times else 0
        
        return {
            'uptime_percentage': uptime_pct,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'degraded_checks': sum(1 for h in self.health_history if h.status == "degraded"),
            'critical_checks': sum(1 for h in self.health_history if h.status == "critical"),
            'mean_time_between_failures_sec': mtbf,
            'recent_warnings': list(self.warnings)[-10:],
            'recent_errors': list(self.errors)[-10:]
        }