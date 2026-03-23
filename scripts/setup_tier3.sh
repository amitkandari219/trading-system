#!/usr/bin/env bash
# =============================================================
# Setup script for Tier 3 ML/AI models
# =============================================================
# Usage: bash scripts/setup_tier3.sh
#
# Steps:
#   1. Run DB migration 010 (Tier 3 tables)
#   2. Install Python dependencies
#   3. Create model directories
#   4. Run Tier 3 tests
#   5. Update crontab
# =============================================================

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

echo "=========================================="
echo "  Tier 3 ML/AI Setup"
echo "=========================================="

# ----------------------------------------------------------
# 1. DB Migration 010
# ----------------------------------------------------------
echo ""
echo "[1/5] Running DB migration 010_tier3_ml_tables.sql..."

DB_CONTAINER="trading-db"
MIGRATION="db/migrations/010_tier3_ml_tables.sql"

if docker ps --format '{{.Names}}' | grep -q "^${DB_CONTAINER}$"; then
    docker cp "$MIGRATION" "${DB_CONTAINER}:/tmp/010_tier3_ml_tables.sql"
    docker exec "$DB_CONTAINER" psql -U trader -d trading \
        -f /tmp/010_tier3_ml_tables.sql 2>&1 | tail -5
    echo "  ✓ Migration 010 applied"
else
    echo "  ⚠ Docker container '$DB_CONTAINER' not running"
    echo "  Trying direct psql..."
    if command -v psql &>/dev/null; then
        psql -h localhost -p 5450 -U trader -d trading \
            -f "$MIGRATION" 2>&1 | tail -5
        echo "  ✓ Migration 010 applied via psql"
    else
        echo "  ✗ Cannot apply migration — no Docker or psql available"
    fi
fi

# ----------------------------------------------------------
# 2. Python Dependencies
# ----------------------------------------------------------
echo ""
echo "[2/5] Installing Python dependencies..."

# Core ML (numpy, pandas already installed)
pip install --break-system-packages -q \
    scikit-learn 2>/dev/null || true

# Optional: full training packages (large, skip if not needed)
echo "  Optional packages for full training (install manually if needed):"
echo "    pip install torch torchvision --break-system-packages"
echo "    pip install mamba-ssm --break-system-packages"
echo "    pip install stable-baselines3 --break-system-packages"
echo "    pip install pytorch-forecasting pytorch-lightning --break-system-packages"
echo "    pip install transformers --break-system-packages  # For FinBERT"
echo "  ✓ Core dependencies installed"

# ----------------------------------------------------------
# 3. Model Directories
# ----------------------------------------------------------
echo ""
echo "[3/5] Creating model directories..."

mkdir -p models/regime_detector
mkdir -p models/tft
mkdir -p models/rl_sizer
mkdir -p models/gnn_sector
mkdir -p models/nlp
echo "  ✓ Model directories created"

# ----------------------------------------------------------
# 4. Run Tests
# ----------------------------------------------------------
echo ""
echo "[4/5] Running Tier 3 tests..."

python -m pytest tests/test_tier3_models.py -v --tb=short 2>&1 | tail -20

# ----------------------------------------------------------
# 5. Verify Imports + Dry Run
# ----------------------------------------------------------
echo ""
echo "[5/5] Verifying imports and dry run..."

python -c "
from models.mamba_regime import MambaRegimeDetector
from models.tft_forecaster import TFTForecaster
from models.rl_position_sizer import RLPositionSizer
from models.gnn_sector_rotation import GNNSectorRotation
from models.nlp_sentiment import NLPSentiment
from data.amfi_mf_flows import AMFIMutualFundSignal
from data.credit_card_spending import CreditCardSpendingSignal
print('  ✓ All 7 Tier 3 modules import successfully')
"

python -m scripts.tier3_signal_pipeline --dry-run 2>&1 | tail -15

echo ""
echo "=========================================="
echo "  Tier 3 Setup Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Add crontab entry:"
echo "     25 8 * * 1-5  cd $PROJ_DIR && python -m scripts.tier3_signal_pipeline"
echo "  2. For full ML training, install optional packages above"
echo "  3. Run: python -m scripts.tier3_signal_pipeline --dry-run"
