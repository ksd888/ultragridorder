#!/bin/bash
# Ultra Grid Trading System V2.0 - Setup Script
# Run with: bash setup.sh

set -e  # Exit on error

echo "============================================================"
echo "🚀 Ultra Grid Trading System V2.0 - Setup"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "1️⃣ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found! Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3 found${NC}"
echo ""

# Create virtual environment
echo "2️⃣ Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv already exists, skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi
echo ""

# Activate venv
echo "3️⃣ Activating virtual environment..."
source venv/bin/activate || {
    echo -e "${RED}❌ Failed to activate venv${NC}"
    exit 1
}
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "4️⃣ Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ Pip upgraded${NC}"
echo ""

# Install dependencies
echo "5️⃣ Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ All dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found!${NC}"
    exit 1
fi
echo ""

# Create required folders
echo "6️⃣ Creating required folders..."
mkdir -p logs state backups
echo -e "${GREEN}✅ Folders created: logs/, state/, backups/${NC}"
echo ""

# Check for .env file
echo "7️⃣ Checking for .env file..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file exists${NC}"
    
    # Check if it contains real keys
    if grep -q "9mRYksnd3AKsDY52xPTKL78" .env; then
        echo ""
        echo -e "${RED}🚨 SECURITY WARNING! 🚨${NC}"
        echo -e "${RED}Your .env contains EXPOSED API keys!${NC}"
        echo ""
        echo "IMMEDIATE ACTIONS REQUIRED:"
        echo "1. Go to Binance → API Management"
        echo "2. DELETE the exposed API key"
        echo "3. CREATE new API keys with restrictions:"
        echo "   ✓ Enable: Spot Trading, Reading"
        echo "   ✗ Disable: Withdrawal, Futures"
        echo ""
        
        read -p "Have you revoked the old keys? (yes/no): " revoked
        if [ "$revoked" != "yes" ]; then
            echo -e "${RED}❌ Please revoke old keys before continuing${NC}"
            exit 1
        fi
    fi
    
    echo ""
    read -p "Do you want to create new .env file? (yes/no): " create_new
    if [ "$create_new" = "yes" ]; then
        mv .env .env.backup
        echo -e "${YELLOW}⚠️  Old .env backed up to .env.backup${NC}"
    fi
fi

if [ ! -f ".env" ]; then
    echo ""
    read -p "Enter Binance API Key: " api_key
    read -p "Enter Binance Secret: " api_secret
    read -p "Enter Telegram Token (optional, press Enter to skip): " tg_token
    read -p "Enter Telegram Chat ID (optional, press Enter to skip): " tg_chat
    
    cat > .env << EOF
BINANCE_API_KEY=$api_key
BINANCE_SECRET=$api_secret
TELEGRAM_TOKEN=$tg_token
TELEGRAM_CHAT_ID=$tg_chat
EOF
    
    chmod 600 .env
    echo -e "${GREEN}✅ .env file created and secured${NC}"
fi
echo ""

# Create .gitignore
echo "8️⃣ Creating .gitignore..."
if [ -f ".gitignore" ]; then
    echo -e "${YELLOW}⚠️  .gitignore exists, skipping...${NC}"
else
    cat > .gitignore << 'EOF'
# Environment
.env
*.env
.env.backup

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/

# Project specific
logs/
state/
backups/
*.db
*.csv
!grid_plan.csv

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store
EOF
    echo -e "${GREEN}✅ .gitignore created${NC}"
fi
echo ""

# Test Binance connection
echo "9️⃣ Testing Binance connection..."
python3 << 'PYEOF'
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

try:
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True
    })
    
    # Test connection
    balance = exchange.fetch_balance()
    ticker = exchange.fetch_ticker('ADA/USDT')
    
    print("✅ Binance connection successful!")
    print(f"   USDT Balance: {balance['USDT']['free']:.2f}")
    print(f"   ADA/USDT Price: ${ticker['last']:.4f}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Binance connection working${NC}"
else
    echo -e "${RED}❌ Binance connection failed! Check your API keys${NC}"
    exit 1
fi
echo ""

# Check grid_plan.csv
echo "🔟 Checking grid_plan.csv..."
if [ ! -f "grid_plan.csv" ]; then
    echo -e "${RED}❌ grid_plan.csv not found!${NC}"
    exit 1
fi

line_count=$(wc -l < grid_plan.csv)
if [ "$line_count" -le 1 ]; then
    echo -e "${YELLOW}⚠️  grid_plan.csv only has header (no data)${NC}"
    echo ""
    read -p "Generate grid now using Monte Carlo? (yes/no): " generate
    
    if [ "$generate" = "yes" ]; then
        echo "Generating optimized grid..."
        python start_with_optimization.py || {
            echo -e "${RED}❌ Grid generation failed${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Grid generated successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  You'll need to generate grid before trading${NC}"
        echo "   Run: python start_with_optimization.py"
    fi
else
    echo -e "${GREEN}✅ grid_plan.csv has $line_count levels${NC}"
fi
echo ""

# Final checks
echo "1️⃣1️⃣ Final verification..."
echo ""
echo "Checking files..."

files_to_check=(
    "trader.py:Core bot"
    "execution.py:Order execution"
    "signal_engine.py:Signal generation"
    "models.py:Data structures"
    "config.py:Configuration"
    "config.yaml:Settings"
    "requirements.txt:Dependencies"
)

all_good=true
for file_info in "${files_to_check[@]}"; do
    IFS=':' read -r file desc <<< "$file_info"
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} $file - $desc"
    else
        echo -e "  ${RED}❌${NC} $file - $desc (MISSING!)"
        all_good=false
    fi
done
echo ""

if [ "$all_good" = false ]; then
    echo -e "${RED}❌ Some files are missing! Cannot continue.${NC}"
    exit 1
fi

# Summary
echo "============================================================"
echo "🎉 SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Review configuration:"
echo "   nano config.yaml"
echo ""
echo "2. Test in dry run mode (IMPORTANT!):"
echo "   python trader.py"
echo ""
echo "3. Monitor dashboard (in another terminal):"
echo "   streamlit run dashboard.py"
echo "   Open: http://localhost:8501"
echo ""
echo "4. When ready for live trading:"
echo "   - Edit config.yaml: dry_run: false"
echo "   - Start with small budget (e.g., 50 USDT)"
echo "   - Monitor closely for first 24 hours"
echo ""
echo "⚠️  SAFETY REMINDERS:"
echo "   • Always test in dry_run first"
echo "   • Never share your .env file"
echo "   • Enable Telegram notifications"
echo "   • Backup regularly: ./backup.sh"
echo "   • Monitor circuit breaker status"
echo ""
echo "📚 Documentation:"
echo "   See README.md and checklist in artifacts"
echo ""
echo "✅ Virtual environment is activated"
echo "   To deactivate: deactivate"
echo "   To reactivate: source venv/bin/activate"
echo ""
echo "============================================================"
echo "Happy Trading! 🚀"
echo "============================================================"
