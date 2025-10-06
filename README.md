# Ultra Grid Trading System V2.0

Complete production-grade grid trading system with advanced safety features and orderflow analysis.

## 🎯 Features

### Core Trading
- **MAD-zscore Signals**: Robust orderflow analysis using Median Absolute Deviation
- **Multi-indicator Fusion**: RSI, Bollinger Bands, Wyckoff, VPIN, Smart Money Index
- **Dynamic Grid Management**: CSV-based or dynamic grid levels
- **Advanced Take Profit**: Fixed, liquidity-based, or dynamic TP strategies

### Risk Management
- **Stop Loss System**: Fixed, ATR-based, or trailing stops
- **Circuit Breaker**: Daily loss limits, consecutive loss protection
- **Position Sizing**: Kelly criterion, volatility-based, or fixed sizing
- **Emergency Exit**: Flash crash detection, API failure handling

### System Safety
- **State Persistence**: SQLite-based crash recovery
- **Position Reconciliation**: Auto-fix discrepancies with exchange
- **Health Monitoring**: Comprehensive system health checks
- **Graceful Shutdown**: Proper cleanup and state saving

### Monitoring
- **Real-time Dashboard**: Streamlit-based visualization
- **Telegram Notifications**: Trade alerts and system status
- **CSV Logging**: Detailed decision and performance logs
- **Performance Tracking**: Win rate, drawdown, Sharpe ratio

## 📁 Project Structure
ultra-grid-v2/
├── config.py              # Configuration management
├── models.py              # Data models and structures
├── state_manager.py       # State persistence and recovery
├── health_monitor.py      # System health monitoring
├── risk_manager.py        # Risk management components
├── indicators.py          # Technical indicators
├── signal_engine.py       # Signal generation
├── execution.py           # Order execution
├── trader.py              # Main trading system
├── utils.py               # Utility functions
├── notifications.py       # Telegram notifications
├── dashboard.py           # Streamlit dashboard
├── config.yaml            # Configuration file
├── grid_plan.csv          # Grid levels (optional)
└── requirements.txt       # Python dependencies

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ultra-grid-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

2. Configuration
Create config.yaml:

# Core Settings
symbol: "ADA/USDT"
mode: "spot"
budget_usdt: 200.0
dry_run: true  # Set to false for live trading

# Grid Settings
grid_source: "csv"
grid_csv: "grid_plan.csv"
max_grid_levels: 6
min_grid_distance: 0.0020

# Signal Parameters
cvd_z_threshold: 1.5
ts_z_threshold: 1.5
imbalance_threshold: 0.55
confirm_bars: 1

# Risk Management
safety:
  enable_stop_loss: true
  stop_loss_type: "atr"
  enable_circuit_breaker: true
  daily_loss_limit_pct: 0.05
  max_drawdown_stop_pct: 0.08

# Notifications
enable_telegram: false
telegram_token: ""
telegram_chat_id: ""