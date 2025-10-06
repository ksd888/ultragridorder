#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Notifications
"""

import requests
from datetime import datetime, timezone
from typing import Optional
from config import TradingConfig


class TelegramNotifier:
    """Send notifications via Telegram Bot"""
    
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.enabled = config.enable_telegram
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id
        
        if self.enabled:
            if not self.token or not self.chat_id:
                print("[WARNING] Telegram enabled but token/chat_id missing")
                self.enabled = False
            else:
                print("[OK] Telegram notifications enabled")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                print(f"[ERROR] Telegram error: {response.status_code}")
                return False

        except Exception as e:
            print(f"[ERROR] Telegram send failed: {e}")
            return False
    
    def notify_order(self, side: str, symbol: str, price: float,
                    quantity: float, reason: str = "") -> bool:
        """Notify order execution"""
        emoji = "[BUY]" if side == "BUY" else "[SELL]"

        message = (
            f"{emoji} <b>{side} ORDER</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.6f}\n"
            f"Quantity: {quantity:.4f}\n"
            f"Value: ${price * quantity:.2f}\n"
        )
        
        if reason:
            message += f"Reason: {reason}\n"
        
        message += f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return self.send_message(message)
    
    def notify_position_closed(self, symbol: str, entry_price: float,
                              exit_price: float, quantity: float,
                              pnl: float, reason: str = "") -> bool:
        """Notify position closed"""
        emoji = "[PROFIT]" if pnl > 0 else "[LOSS]"
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        message = (
            f"{emoji} <b>POSITION CLOSED</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Entry: ${entry_price:.6f}\n"
            f"Exit: ${exit_price:.6f}\n"
            f"Quantity: {quantity:.4f}\n"
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
        )
        
        if reason:
            message += f"Reason: {reason}\n"
        
        message += f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return self.send_message(message)
    
    def notify_stop_loss(self, symbol: str, entry_price: float,
                        exit_price: float, pnl: float) -> bool:
        """Notify stop loss triggered"""
        message = (
            f"[STOP] <b>STOP LOSS TRIGGERED</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Entry: ${entry_price:.6f}\n"
            f"Exit: ${exit_price:.6f}\n"
            f"Loss: ${pnl:.2f}\n"
            f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        return self.send_message(message)
    
    def notify_circuit_breaker(self, reason: str) -> bool:
        """Notify circuit breaker activation"""
        message = (
            f"[WARNING] <b>CIRCUIT BREAKER ACTIVATED</b>\n\n"
            f"Reason: {reason}\n"
            f"Trading halted for safety\n"
            f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        return self.send_message(message)
    
    def notify_error(self, error_type: str, error_msg: str) -> bool:
        """Notify system error"""
        message = (
            f"[ERROR] <b>SYSTEM ERROR</b>\n\n"
            f"Type: {error_type}\n"
            f"Message: {error_msg}\n"
            f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        return self.send_message(message)
    
    def notify_daily_summary(self, trades: int, pnl: float,
                           win_rate: float, positions: int) -> bool:
        """Send daily summary"""
        emoji = "[UP]" if pnl > 0 else "[DOWN]"

        message = (
            f"{emoji} <b>DAILY SUMMARY</b>\n\n"
            f"Trades: {trades}\n"
            f"P&L: ${pnl:.2f}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Open Positions: {positions}\n"
            f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            print("Telegram is disabled")
            return False

        message = "[BOT] <b>Bot Connection Test</b>\n\nTelegram notifications are working!"

        if self.send_message(message):
            print("[OK] Telegram test successful")
            return True
        else:
            print("[ERROR] Telegram test failed")
            return False