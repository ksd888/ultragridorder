#!/bin/bash
# Ultra Grid V2.0 - Grid Generation Script
# Run this step by step

echo "============================================================"
echo "🚀 ULTRA GRID V2.0 - GRID GENERATION"
echo "============================================================"
echo ""

# Step 1: Check files
echo "Step 1/6: Checking files..."
if [ -f "Advanced Grid Generator.py" ]; then
    echo "✅ Advanced Grid Generator.py found"
else
    echo "❌ Advanced Grid Generator.py NOT found!"
    echo "   Please make sure the file is in current directory"
    exit 1
fi

if [ -f "monte_carlo.py" ]; then
    echo "✅ monte_carlo.py found (required)"
else
    echo "❌ monte_carlo.py NOT found!"
    echo "   This file is required for Advanced Grid Generator"
    exit 1
fi

echo ""

# Step 2: Check virtual environment
echo "Step 2/6: Checking Python environment..."
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    echo "   Activating venv..."
    source venv/bin/activate
else
    echo "⚠️  No venv found, using system Python"
fi

python3 --version
echo ""

# Step 3: Check dependencies
echo "Step 3/6: Checking dependencies..."
required_packages="ccxt pandas numpy"

for package in $required_packages; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package installed"
    else
        echo "❌ $package NOT installed!"
        echo "   Installing..."
        pip install $package
    fi
done

echo ""

# Step 4: Backup existing grid (if exists)
echo "Step 4/6: Backing up existing grid..."
if [ -f "grid_plan.csv" ]; then
    backup_file="grid_plan_backup_$(date +%Y%m%d_%H%M%S).csv"
    cp grid_plan.csv "$backup_file"
    echo "✅ Backup created: $backup_file"
else
    echo "ℹ️  No existing grid to backup"
fi

echo ""

# Step 5: Run Grid Generator
echo "Step 5/6: Running Grid Generator..."
echo "============================================================"
echo "This will take 30-60 seconds..."
echo "============================================================"
echo ""

python3 "Advanced Grid Generator.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Grid Generation Complete!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Grid Generation Failed!"
    echo "============================================================"
    exit 1
fi

echo ""

# Step 6: Verify and use generated grid
echo "Step 6/6: Setting up grid file..."

if [ -f "grid_plan_optimized.csv" ]; then
    # Show preview
    echo "📊 Grid Preview (first 3 levels):"
    head -4 grid_plan_optimized.csv
    echo ""
    
    # Count levels
    levels=$(tail -n +2 grid_plan_optimized.csv | wc -l)
    echo "📈 Total grid levels: $levels"
    echo ""
    
    # Ask user to confirm
    read -p "Use this grid? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        mv grid_plan_optimized.csv grid_plan.csv
        echo "✅ Grid installed as grid_plan.csv"
    else
        echo "ℹ️  Grid saved as grid_plan_optimized.csv"
        echo "   To use later: mv grid_plan_optimized.csv grid_plan.csv"
    fi
else
    echo "❌ grid_plan_optimized.csv not found!"
    echo "   Check the script output for errors"
    exit 1
fi

echo ""
echo "============================================================"
echo "🎉 ALL DONE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Review grid: cat grid_plan.csv"
echo "2. Update config: nano config.yaml"
echo "3. Start bot: python trader.py"
echo "4. Monitor: streamlit run dashboard.py"
echo ""
echo "⚠️  IMPORTANT: Make sure dry_run: true in config.yaml!"
echo "============================================================"
