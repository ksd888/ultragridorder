#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Logging Utilities
Provides JSON-formatted logging for better log parsing and analysis
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs

    Usage:
        logger = StructuredLogger("bot_name")
        logger.log_trade("BUY", "ADA/USDT", 0.45, 100, {"reason": "grid_level"})
        logger.log_error("execution_failed", {"symbol": "ADA/USDT", "error": str(e)})
    """

    def __init__(self, component: str, log_file: Optional[str] = None):
        self.component = component
        self.log_file = log_file

        # Setup standard logger
        self.logger = logging.getLogger(f"structured.{component}")
        self.logger.setLevel(logging.INFO)

        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(fh)

    def _build_log(self, level: str, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured log entry"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": self.component,
            "event_type": event_type,
            **data
        }

    def log(self, level: str, event_type: str, data: Dict[str, Any]):
        """Generic structured log"""
        log_entry = self._build_log(level, event_type, data)
        self.logger.info(json.dumps(log_entry))

    def log_trade(self, side: str, symbol: str, price: float, quantity: float,
                  metadata: Optional[Dict] = None):
        """Log trade execution"""
        data = {
            "side": side,
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "notional": price * quantity
        }
        if metadata:
            data.update(metadata)

        self.log("INFO", "trade_executed", data)

    def log_position_open(self, position_id: str, symbol: str, entry_price: float,
                          size: float, tp: float, sl: Optional[float] = None,
                          metadata: Optional[Dict] = None):
        """Log position opened"""
        data = {
            "position_id": position_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "size": size,
            "tp_price": tp,
            "sl_price": sl
        }
        if metadata:
            data.update(metadata)

        self.log("INFO", "position_opened", data)

    def log_position_close(self, position_id: str, exit_price: float, pnl: float,
                           reason: str, metadata: Optional[Dict] = None):
        """Log position closed"""
        data = {
            "position_id": position_id,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason
        }
        if metadata:
            data.update(metadata)

        self.log("INFO", "position_closed", data)

    def log_signal(self, action: str, price: float, confidence: float,
                   indicators: Dict[str, Any]):
        """Log trading signal"""
        data = {
            "action": action,
            "price": price,
            "confidence": confidence,
            "indicators": indicators
        }

        self.log("INFO", "signal_generated", data)

    def log_error(self, error_type: str, details: Dict[str, Any]):
        """Log error"""
        self.log("ERROR", error_type, details)

    def log_warning(self, warning_type: str, details: Dict[str, Any]):
        """Log warning"""
        self.log("WARNING", warning_type, details)

    def log_circuit_breaker(self, reason: str, state: str, metrics: Dict[str, Any]):
        """Log circuit breaker activation"""
        data = {
            "reason": reason,
            "state": state,
            **metrics
        }

        self.log("CRITICAL", "circuit_breaker", data)

    def log_reconciliation(self, discrepancies: int, details: Dict[str, Any]):
        """Log reconciliation results"""
        data = {
            "discrepancies": discrepancies,
            **details
        }

        level = "WARNING" if discrepancies > 0 else "INFO"
        self.log(level, "reconciliation", data)

    def log_health_check(self, status: str, checks: Dict[str, Any]):
        """Log health check"""
        data = {
            "status": status,
            "checks": checks
        }

        level = "CRITICAL" if status == "critical" else "INFO"
        self.log(level, "health_check", data)


class PerformanceLogger:
    """Track and log performance metrics"""

    def __init__(self, log_file: str = "logs/performance.jsonl"):
        self.logger = StructuredLogger("performance", log_file)
        self.metrics = {}

    def record_execution_time(self, operation: str, duration_ms: float):
        """Record operation execution time"""
        self.logger.log("INFO", "execution_time", {
            "operation": operation,
            "duration_ms": duration_ms
        })

    def record_api_call(self, endpoint: str, success: bool, latency_ms: float):
        """Record API call"""
        self.logger.log("INFO", "api_call", {
            "endpoint": endpoint,
            "success": success,
            "latency_ms": latency_ms
        })

    def record_daily_summary(self, trades: int, pnl: float, win_rate: float,
                             positions_opened: int, positions_closed: int):
        """Record daily trading summary"""
        self.logger.log("INFO", "daily_summary", {
            "trades": trades,
            "pnl": pnl,
            "win_rate": win_rate,
            "positions_opened": positions_opened,
            "positions_closed": positions_closed
        })


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_logger(component: str, log_file: Optional[str] = None) -> StructuredLogger:
    """Create a structured logger instance"""
    return StructuredLogger(component, log_file)


def parse_log_file(log_file: str) -> list:
    """
    Parse structured log file and return list of log entries

    Args:
        log_file: Path to log file

    Returns:
        List of parsed log dictionaries
    """
    logs = []

    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return logs


def filter_logs(logs: list, event_type: Optional[str] = None,
                level: Optional[str] = None,
                component: Optional[str] = None) -> list:
    """
    Filter logs by criteria

    Args:
        logs: List of log entries
        event_type: Filter by event type
        level: Filter by log level
        component: Filter by component

    Returns:
        Filtered list of logs
    """
    filtered = logs

    if event_type:
        filtered = [log for log in filtered if log.get('event_type') == event_type]

    if level:
        filtered = [log for log in filtered if log.get('level') == level]

    if component:
        filtered = [log for log in filtered if log.get('component') == component]

    return filtered


if __name__ == "__main__":
    # Test structured logger
    print("Testing StructuredLogger...")

    logger = StructuredLogger("test_bot", "logs/test_structured.jsonl")

    # Test trade log
    logger.log_trade("BUY", "ADA/USDT", 0.45, 100, {"reason": "grid_level_3"})

    # Test position log
    logger.log_position_open("pos_001", "ADA/USDT", 0.45, 100, 0.46, 0.44,
                             {"grid_level": 3, "phase": "accumulation"})

    # Test signal log
    logger.log_signal("BUY", 0.45, 0.85, {
        "cvd_z": 2.1,
        "rsi": 35,
        "bb_position": 0.2
    })

    # Test error log
    logger.log_error("order_failed", {
        "symbol": "ADA/USDT",
        "error": "Insufficient balance",
        "required": 100,
        "available": 50
    })

    print("✅ Structured logging tests complete")
    print("Check logs/test_structured.jsonl for output")
