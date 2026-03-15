#!/bin/bash
# ================================================================
# NIFTY F&O TRADING SYSTEM — Phase 1 Setup
# ================================================================
# Prerequisites:
#   - Python 3.10+
#   - PostgreSQL 14+ running locally
#   - (Optional) TimescaleDB extension installed
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ================================================================

set -e

echo "================================================"
echo "  NIFTY F&O TRADING SYSTEM — Phase 1 Setup"
echo "================================================"

# Step 1: Python virtual environment
echo ""
echo "[1/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists"
fi
source venv/bin/activate
echo "  Activated venv"

# Step 2: Install dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet pytest  # for tests
echo "  Dependencies installed"

# Step 3: Create database
echo ""
echo "[3/5] Setting up PostgreSQL database..."
if createdb trading 2>/dev/null; then
    echo "  Database 'trading' created"
else
    echo "  Database 'trading' already exists"
fi

# Step 4: Run schema
echo ""
echo "[4/5] Running schema.sql..."
python -m db.setup --schema
echo "  Schema applied"

# Step 5: Verify
echo ""
echo "[5/5] Verifying setup..."
python -m db.setup --check

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Load Nifty data:      python -m data.nifty_loader --init"
echo "  2. Label regimes:        python regime_labeler.py --label-all"
echo "  3. Run lookahead test:   python regime_labeler.py --test"
echo "  4. Run unit tests:       python -m pytest tests/ -v"
echo "  5. Check data status:    python -m data.nifty_loader --status"
echo ""
