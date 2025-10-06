#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management with Enhanced Validation
Version: 2.0.1 - Fixed and Optimized
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class SafetyConfig:
    """Safety and risk management settings"""
    # Stop Loss
    enable_stop_loss: bool = True
    stop_loss_type: Literal["fixed", "atr", "trailing"] = "atr"
    fixed_sl_pct: float = 0.02
    atr_sl_multiplier: float = 2.5  # ✅ Changed from 2.0
    min_sl_distance_pct: float = 0.01  # ✅ NEW: Minimum 1% SL
    
    # Trailing Stop
    trailing_sl_activation: float = 0.012  # ✅ Changed from 0.01
    trailing_sl_distance: float = 0.007    # ✅ Changed from 0.005
    
    # Circuit Breaker
    enable_circuit_breaker: bool = True
    daily_loss_limit_pct: float = 0.03  # ✅ Changed from 0.05
    max_consecutive_losses: int = 3     # ✅ Changed from 5
    max_drawdown_stop_pct: float = 0.06 # ✅ Changed from 0.08
    circuit_breaker_cooldown_hours: int = 4
    
    # Emergency Exit
    enable_emergency_exit: bool = True
    flash_crash_threshold_pct: float = 0.10
    api_failure_threshold: int = 60  # ✅ NEW
    
    # Position Limits
    max_position_size_pct: float = 0.25  # ✅ Changed from 0.3
    max_total_exposure_pct: float = 0.9
    max_position_age_hours: int = 48  # ✅ NEW
    
    def __post_init__(self):
        """Validate safety config"""
        if self.fixed_sl_pct <= 0 or self.fixed_sl_pct >= 1:
            raise ValueError("fixed_sl_pct must be between 0 and 1")
        
        if self.atr_sl_multiplier <= 0:
            raise ValueError("atr_sl_multiplier must be positive")
        
        if self.min_sl_distance_pct < 0.005:  # Minimum 0.5%
            raise ValueError("min_sl_distance_pct must be >= 0.005 (0.5%)")
        
        if self.trailing_sl_activation <= self.trailing_sl_distance:
            raise ValueError("trailing_sl_activation must be > trailing_sl_distance")
        
        if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct >= 1:
            raise ValueError("daily_loss_limit_pct must be between 0 and 1")
        
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be between 0 and 1")


@dataclass
class MonteCarloConfig:
    """Monte Carlo optimization settings"""
    enabled: bool = False
    n_simulations: int = 1000
    lookback_days: int = 30
    confidence_level: float = 0.95  # ✅ NEW
    
    def __post_init__(self):
        """Validate Monte Carlo config"""
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be >= 100")
        
        if self.lookback_days < 7 or self.lookback_days > 365:
            raise ValueError("lookback_days must be between 7 and 365")
        
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.0")


@dataclass
class ReconciliationConfig:
    """Position reconciliation settings"""
    enabled: bool = True
    interval_sec: int = 300
    auto_fix: bool = True
    balance_threshold_pct: float = 0.05  # ✅ NEW
    
    def __post_init__(self):
        """Validate reconciliation config"""
        if self.interval_sec < 60:
            raise ValueError("interval_sec must be >= 60 seconds")
        
        if self.balance_threshold_pct <= 0 or self.balance_threshold_pct > 0.5:
            raise ValueError("balance_threshold_pct must be between 0 and 0.5")


@dataclass
class AdvancedConfig:
    """Advanced settings"""
    # Performance
    enable_jit_compilation: bool = False
    cache_indicators: bool = True
    
    # Data Quality
    min_trades_per_bar: int = 5
    min_orderbook_levels: int = 10
    
    # Risk
    correlation_threshold: float = 0.8
    max_correlation_exposure: float = 0.5
    
    def __post_init__(self):
        """Validate advanced config"""
        if self.min_trades_per_bar < 0:
            raise ValueError("min_trades_per_bar must be non-negative")
        
        if self.min_orderbook_levels < 5:
            raise ValueError("min_orderbook_levels must be >= 5")


@dataclass
class TradingConfig:
    """Main trading configuration with comprehensive validation"""
    
    # Core Settings
    symbol: str = "ADA/USDT"
    mode: Literal["spot", "futures"] = "spot"
    budget_usdt: float = 200.0
    reserve_pct: float = 0.05
    dry_run: bool = True
    
    # Grid Settings
    grid_source: Literal["csv", "dynamic"] = "csv"
    grid_csv: str = "grid_plan.csv"
    max_grid_levels: int = 6
    min_grid_distance: float = 0.0020
    grid_tolerance: float = 0.0015
    grid_reload_sec: int = 0
    
    # Signal Parameters
    cvd_z_threshold: float = 1.5
    ts_z_threshold: float = 1.5
    imbalance_threshold: float = 0.55
    confirm_bars: int = 1
    
    # Orderflow Settings
    bar_interval_ms: int = 5000
    orderbook_depth: int = 20
    cvd_ema_span: int = 20
    ts_ema_span: int = 20
    
    # Technical Indicators
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0  # ✅ NEW
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Advanced Filters
    min_volatility: float = 0.005  # ✅ Changed from 0.002
    max_volatility: float = 0.05   # ✅ Changed from 0.10
    min_spread_pct: float = 0.0001
    max_spread_pct: float = 0.005  # ✅ Changed from 0.01
    
    # Take Profit Strategy
    tp_strategy: Literal["fixed", "dynamic", "liquidity"] = "fixed"
    fixed_tp_pct: float = 0.015
    tp_offset_pct: float = 0.001
    
    # Position Sizing
    position_sizing: Literal["fixed", "volatility", "kelly"] = "volatility"
    
    # Order Management
    max_open_orders: int = 6
    order_cooldown_sec: int = 30
    resync_open_orders_sec: int = 300
    
    # Execution
    high_price_threshold: float = 1000.0  # ✅ ENHANCED: ถ้าราคา >= ค่านี้ ใช้ quoteOrderQty
    quote_safety_factor: float = 0.999     # ✅ NEW: กันงบเผื่อ (0.1%)
    min_notional_override: Optional[float] = None
    
    # State Management
    state_db: str = "state/bot_state.db"
    
    # Logging
    log_csv: str = "logs/decisions.csv"
    dust_ledger: str = "logs/dust_ledger.csv"
    
    # Health Monitoring
    health_check_interval_sec: int = 60
    max_api_failures: int = 10
    max_data_staleness_sec: int = 30
    
    # Telegram Notifications
    enable_telegram: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""
    
    # Sub-configs
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    
    def __post_init__(self):
        """✅ ENHANCED: Comprehensive validation"""
        
        # ========== Core Settings ==========
        if self.budget_usdt <= 0:
            raise ValueError("budget_usdt must be positive")
        
        if self.budget_usdt < 50:
            raise ValueError("budget_usdt must be >= $50 for meaningful trading")
        
        if not 0 <= self.reserve_pct < 0.2:
            raise ValueError("reserve_pct must be between 0 and 0.2 (20%)")
        
        # ========== Grid Settings ==========
        if self.max_grid_levels <= 0:
            raise ValueError("max_grid_levels must be positive")
        
        if self.max_grid_levels > 20:
            raise ValueError("max_grid_levels too high (max 20)")
        
        if self.min_grid_distance <= 0:
            raise ValueError("min_grid_distance must be positive")
        
        if self.min_grid_distance > 0.1:  # 10%
            raise ValueError("min_grid_distance too large (max 10%)")
        
        if self.grid_tolerance <= 0:
            raise ValueError("grid_tolerance must be positive")
        
        if self.grid_tolerance >= self.min_grid_distance:
            raise ValueError("grid_tolerance must be < min_grid_distance")
        
        # ========== Signal Parameters ==========
        if self.cvd_z_threshold < 0:
            raise ValueError("cvd_z_threshold must be non-negative")
        
        if self.cvd_z_threshold > 5:
            raise ValueError("cvd_z_threshold too high (max 5) - will miss signals")
        
        if self.ts_z_threshold < 0:
            raise ValueError("ts_z_threshold must be non-negative")
        
        if self.ts_z_threshold > 5:
            raise ValueError("ts_z_threshold too high (max 5)")
        
        if not 0.5 <= self.imbalance_threshold <= 1.0:
            raise ValueError("imbalance_threshold must be between 0.5 and 1.0")
        
        if self.confirm_bars < 0:
            raise ValueError("confirm_bars must be non-negative")
        
        if self.confirm_bars > 10:
            raise ValueError("confirm_bars too high (max 10) - will miss opportunities")
        
        # ========== Orderflow Settings ==========
        if self.bar_interval_ms < 1000:
            raise ValueError("bar_interval_ms must be >= 1000 (1 second)")
        
        if self.orderbook_depth < 5:
            raise ValueError("orderbook_depth must be >= 5")
        
        if self.orderbook_depth > 100:
            raise ValueError("orderbook_depth too high (max 100)")
        
        # ========== Technical Indicators ==========
        if self.rsi_period < 5 or self.rsi_period > 50:
            raise ValueError("rsi_period must be between 5 and 50")
        
        if not 0 < self.rsi_oversold < 50:
            raise ValueError("rsi_oversold must be between 0 and 50")
        
        if not 50 < self.rsi_overbought < 100:
            raise ValueError("rsi_overbought must be between 50 and 100")
        
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be < rsi_overbought")
        
        # ========== Volatility & Spread Filters ==========
        if self.min_volatility <= 0:
            raise ValueError("min_volatility must be positive")
        
        if self.min_volatility >= self.max_volatility:
            raise ValueError("min_volatility must be < max_volatility")
        
        if self.max_volatility > 0.5:
            raise ValueError("max_volatility too high (max 50%)")
        
        if self.min_spread_pct <= 0:
            raise ValueError("min_spread_pct must be positive")
        
        if self.min_spread_pct >= self.max_spread_pct:
            raise ValueError("min_spread_pct must be < max_spread_pct")
        
        if self.max_spread_pct > 0.05:  # 5%
            raise ValueError("max_spread_pct too wide (max 5%)")
        
        # ========== Take Profit ==========
        if self.fixed_tp_pct <= 0 or self.fixed_tp_pct > 0.5:
            raise ValueError("fixed_tp_pct must be between 0 and 0.5 (50%)")
        
        if self.tp_offset_pct < 0 or self.tp_offset_pct > 0.01:
            raise ValueError("tp_offset_pct must be between 0 and 0.01 (1%)")
        
        # ========== Order Management ==========
        if self.max_open_orders <= 0:
            raise ValueError("max_open_orders must be positive")
        
        if self.max_open_orders > self.max_grid_levels:
            raise ValueError("max_open_orders cannot exceed max_grid_levels")
        
        if self.order_cooldown_sec < 0:
            raise ValueError("order_cooldown_sec must be non-negative")
        
        if self.resync_open_orders_sec < 60:
            raise ValueError("resync_open_orders_sec must be >= 60 seconds")
        
        # ========== Health Monitoring ==========
        if self.health_check_interval_sec < 30:
            raise ValueError("health_check_interval_sec must be >= 30 seconds")
        
        if self.max_api_failures < 3:
            raise ValueError("max_api_failures must be >= 3")
        
        if self.max_data_staleness_sec < 10:
            raise ValueError("max_data_staleness_sec must be >= 10 seconds")
        
        # ========== File Paths ==========
        if self.grid_source == "csv" and not self.grid_csv:
            raise ValueError("grid_csv must be specified when grid_source='csv'")
        
        # ========== Telegram ==========
        if self.enable_telegram:
            if not self.telegram_token:
                raise ValueError("telegram_token required when enable_telegram=True")
            if not self.telegram_chat_id:
                raise ValueError("telegram_chat_id required when enable_telegram=True")
        
        # ========== Cross-Parameter Validation ==========
        cross_param_errors = []

        # Check: spread vs grid distance
        if self.max_spread_pct > self.min_grid_distance:
            cross_param_errors.append(
                f"max_spread_pct ({self.max_spread_pct*100:.2f}%) should be < "
                f"min_grid_distance ({self.min_grid_distance*100:.2f}%)"
            )

        # Check: volatility vs signal thresholds
        if self.max_volatility < 0.01 and (self.cvd_z_threshold > 2.5 or self.ts_z_threshold > 2.5):
            cross_param_errors.append(
                "Low max_volatility with high signal thresholds may miss all signals"
            )

        # Check: TP vs grid spacing
        if self.fixed_tp_pct < self.min_grid_distance:
            cross_param_errors.append(
                f"fixed_tp_pct ({self.fixed_tp_pct*100:.2f}%) should be >= "
                f"min_grid_distance ({self.min_grid_distance*100:.2f}%)"
            )

        # Check: Stop loss vs TP
        if self.safety.fixed_sl_pct >= self.fixed_tp_pct:
            cross_param_errors.append(
                f"Stop loss ({self.safety.fixed_sl_pct*100:.2f}%) should be < "
                f"TP ({self.fixed_tp_pct*100:.2f}%)"
            )

        # Check: Position sizing vs grid levels
        max_possible_exposure = self.safety.max_position_size_pct * self.max_grid_levels
        if max_possible_exposure > self.safety.max_total_exposure_pct:
            cross_param_errors.append(
                f"Max possible exposure ({max_possible_exposure*100:.0f}%) exceeds "
                f"max_total_exposure ({self.safety.max_total_exposure_pct*100:.0f}%)"
            )

        # Check: Reconciliation interval vs health check
        if self.reconciliation.interval_sec < self.health_check_interval_sec:
            cross_param_errors.append(
                "Reconciliation interval should be >= health_check_interval"
            )

        # Check: Circuit breaker daily loss vs position sizing
        if self.safety.daily_loss_limit_pct < self.safety.max_position_size_pct * 2:
            cross_param_errors.append(
                "daily_loss_limit_pct should be >= 2x max_position_size_pct"
            )

        # Raise errors if any
        if cross_param_errors:
            print("\n[ERROR] Cross-parameter validation failed:")
            for err in cross_param_errors:
                print(f"  - {err}")
            raise ValueError("Cross-parameter validation failed")

        # ========== Summary ==========
        print("[OK] Configuration validation passed")

        # Show warnings
        warnings = []

        if not self.dry_run:
            warnings.append("[WARNING] DRY RUN DISABLED - LIVE TRADING MODE!")

        if self.budget_usdt > 1000:
            warnings.append(f"[WARNING] Large budget: ${self.budget_usdt} - start smaller!")

        if self.confirm_bars == 0:
            warnings.append("[WARNING] No signal confirmation - may get false signals")

        if self.max_spread_pct > 0.01:
            warnings.append(f"[WARNING] Wide spread tolerance: {self.max_spread_pct*100}%")

        # New warnings
        if self.safety.atr_sl_multiplier < 2.0:
            warnings.append(f"[WARNING] Low ATR multiplier: {self.safety.atr_sl_multiplier} - may get stopped out frequently")

        if self.max_open_orders == self.max_grid_levels:
            warnings.append("[WARNING] max_open_orders = max_grid_levels - no reserve capacity")

        if self.position_sizing == "kelly" and len(warnings) > 0:
            warnings.append("[WARNING] Kelly sizing requires stable win rate - use cautiously")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  {w}")
            print()


class ConfigManager:
    """Hot-reload configuration from YAML file"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.last_mtime = 0
        self.config: TradingConfig = self._load_config()
    
    def _load_config(self) -> TradingConfig:
        """Load configuration from YAML with error handling"""
        if not self.config_path.exists():
            print(f"⚠️  Config file not found: {self.config_path}")
            print("Using default configuration")
            return TradingConfig()
        
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse sub-configs
            safety_data = data.pop('safety', {})
            safety_config = SafetyConfig(**safety_data)
            
            monte_carlo_data = data.pop('monte_carlo', {})
            monte_carlo_config = MonteCarloConfig(**monte_carlo_data)
            
            reconciliation_data = data.pop('reconciliation', {})
            reconciliation_config = ReconciliationConfig(**reconciliation_data)
            
            # ✅ NEW: Advanced config
            advanced_data = data.pop('advanced', {})
            advanced_config = AdvancedConfig(**advanced_data)
            
            # Create main config
            config = TradingConfig(
                **data,
                safety=safety_config,
                monte_carlo=monte_carlo_config,
                reconciliation=reconciliation_config,
                advanced=advanced_config
            )
            
            self.last_mtime = self.config_path.stat().st_mtime
            
            print(f"[OK] Configuration loaded from {self.config_path}")
            
            return config
        
        except yaml.YAMLError as e:
            print(f"[ERROR] YAML parsing error: {e}")
            raise

        except TypeError as e:
            print(f"[ERROR] Invalid configuration: {e}")
            raise

        except ValueError as e:
            print(f"[ERROR] Configuration validation failed: {e}")
            raise

        except Exception as e:
            print(f"[ERROR] Unexpected error loading config: {e}")
            raise
    
    def check_and_reload(self) -> bool:
        """Check if config file changed and reload"""
        if not self.config_path.exists():
            return False
        
        try:
            current_mtime = self.config_path.stat().st_mtime
            
            if current_mtime > self.last_mtime:
                print(f"\n🔄 Config file changed, reloading...")
                old_config = self.config
                
                try:
                    self.config = self._load_config()
                    print("[OK] Configuration reloaded successfully")
                    return True
                
                except Exception as e:
                    print(f"[ERROR] Failed to reload config: {e}")
                    print("Keeping previous configuration")
                    self.config = old_config
                    return False

        except Exception as e:
            print(f"[ERROR] Error checking config: {e}")
        
        return False
    
    def save_config(self, config: TradingConfig):
        """Save configuration to YAML"""
        try:
            config_dict = {
                # Core
                'symbol': config.symbol,
                'mode': config.mode,
                'budget_usdt': config.budget_usdt,
                'reserve_pct': config.reserve_pct,
                'dry_run': config.dry_run,
                
                # Grid
                'grid_source': config.grid_source,
                'grid_csv': config.grid_csv,
                'max_grid_levels': config.max_grid_levels,
                'min_grid_distance': config.min_grid_distance,
                'grid_tolerance': config.grid_tolerance,
                'grid_reload_sec': config.grid_reload_sec,
                
                # Signals
                'cvd_z_threshold': config.cvd_z_threshold,
                'ts_z_threshold': config.ts_z_threshold,
                'imbalance_threshold': config.imbalance_threshold,
                'confirm_bars': config.confirm_bars,
                
                # Orderflow
                'bar_interval_ms': config.bar_interval_ms,
                'orderbook_depth': config.orderbook_depth,
                'cvd_ema_span': config.cvd_ema_span,
                'ts_ema_span': config.ts_ema_span,
                
                # Indicators
                'rsi_period': config.rsi_period,
                'rsi_oversold': config.rsi_oversold,
                'rsi_overbought': config.rsi_overbought,
                'bb_period': config.bb_period,
                'bb_std': config.bb_std,
                
                # Filters
                'min_volatility': config.min_volatility,
                'max_volatility': config.max_volatility,
                'min_spread_pct': config.min_spread_pct,
                'max_spread_pct': config.max_spread_pct,
                
                # TP
                'tp_strategy': config.tp_strategy,
                'fixed_tp_pct': config.fixed_tp_pct,
                'tp_offset_pct': config.tp_offset_pct,
                
                # Sizing
                'position_sizing': config.position_sizing,
                
                # Orders
                'max_open_orders': config.max_open_orders,
                'order_cooldown_sec': config.order_cooldown_sec,
                'resync_open_orders_sec': config.resync_open_orders_sec,
                
                # Execution
                'high_price_threshold': config.high_price_threshold,
                'min_notional_override': config.min_notional_override,
                
                # State
                'state_db': config.state_db,
                'log_csv': config.log_csv,
                'dust_ledger': config.dust_ledger,
                
                # Health
                'health_check_interval_sec': config.health_check_interval_sec,
                'max_api_failures': config.max_api_failures,
                'max_data_staleness_sec': config.max_data_staleness_sec,
                
                # Telegram
                'enable_telegram': config.enable_telegram,
                'telegram_token': config.telegram_token,
                'telegram_chat_id': config.telegram_chat_id,
                
                # Sub-configs
                'safety': {
                    'enable_stop_loss': config.safety.enable_stop_loss,
                    'stop_loss_type': config.safety.stop_loss_type,
                    'fixed_sl_pct': config.safety.fixed_sl_pct,
                    'atr_sl_multiplier': config.safety.atr_sl_multiplier,
                    'min_sl_distance_pct': config.safety.min_sl_distance_pct,
                    'trailing_sl_activation': config.safety.trailing_sl_activation,
                    'trailing_sl_distance': config.safety.trailing_sl_distance,
                    'enable_circuit_breaker': config.safety.enable_circuit_breaker,
                    'daily_loss_limit_pct': config.safety.daily_loss_limit_pct,
                    'max_consecutive_losses': config.safety.max_consecutive_losses,
                    'max_drawdown_stop_pct': config.safety.max_drawdown_stop_pct,
                    'circuit_breaker_cooldown_hours': config.safety.circuit_breaker_cooldown_hours,
                    'enable_emergency_exit': config.safety.enable_emergency_exit,
                    'flash_crash_threshold_pct': config.safety.flash_crash_threshold_pct,
                    'api_failure_threshold': config.safety.api_failure_threshold,
                    'max_position_size_pct': config.safety.max_position_size_pct,
                    'max_total_exposure_pct': config.safety.max_total_exposure_pct,
                    'max_position_age_hours': config.safety.max_position_age_hours
                },
                'monte_carlo': {
                    'enabled': config.monte_carlo.enabled,
                    'n_simulations': config.monte_carlo.n_simulations,
                    'lookback_days': config.monte_carlo.lookback_days,
                    'confidence_level': config.monte_carlo.confidence_level
                },
                'reconciliation': {
                    'enabled': config.reconciliation.enabled,
                    'interval_sec': config.reconciliation.interval_sec,
                    'auto_fix': config.reconciliation.auto_fix,
                    'balance_threshold_pct': config.reconciliation.balance_threshold_pct
                },
                'advanced': {
                    'enable_jit_compilation': config.advanced.enable_jit_compilation,
                    'cache_indicators': config.advanced.cache_indicators,
                    'min_trades_per_bar': config.advanced.min_trades_per_bar,
                    'min_orderbook_levels': config.advanced.min_orderbook_levels,
                    'correlation_threshold': config.advanced.correlation_threshold,
                    'max_correlation_exposure': config.advanced.max_correlation_exposure
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            print(f"[OK] Configuration saved to {self.config_path}")
        
        except Exception as e:
            print(f"[ERROR] Error saving config: {e}")
            raise


def load_config(config_path: str = "config.yaml") -> TradingConfig:
    """Convenience function to load config"""
    manager = ConfigManager(config_path)
    return manager.config


# ✅ NEW: Validation utility
def validate_config_file(config_path: str = "config.yaml") -> bool:
    """Validate config file without loading"""
    try:
        config = load_config(config_path)
        print("[OK] Configuration file is valid")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration file is invalid: {e}")
        return False


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration...")
    try:
        cfg = load_config()
        print(f"\n[OK] Config loaded successfully")
        print(f"Symbol: {cfg.symbol}")
        print(f"Budget: ${cfg.budget_usdt}")
        print(f"Dry run: {cfg.dry_run}")
        print(f"Grid levels: {cfg.max_grid_levels}")
    except Exception as e:
        print(f"\n[ERROR] Config test failed: {e}")