#!/bin/bash
# ================================================================
# Health Check — alerts if pre-market job didn't run by 9:00 AM
# Runs at 9:00 AM IST Mon-Fri via cron
# ================================================================

TRADING_DIR="/Users/amitkandari/Desktop/trading-system"
VENV_PYTHON="$TRADING_DIR/venv/bin/python3"
HEALTH_DIR="$TRADING_DIR/logs/health"
LOG_DIR="$TRADING_DIR/logs"

TODAY=$(date '+%Y-%m-%d')
ALERT_NEEDED=false
MISSING_JOBS=""

# Check pre-market ran today
if [ -f "$HEALTH_DIR/pre_market.last_run" ]; then
    LAST_RUN=$(cat "$HEALTH_DIR/pre_market.last_run")
    LAST_DATE="${LAST_RUN%% *}"
    if [ "$LAST_DATE" != "$TODAY" ]; then
        ALERT_NEEDED=true
        MISSING_JOBS="pre_market (last: $LAST_RUN)"
    fi
else
    ALERT_NEEDED=true
    MISSING_JOBS="pre_market (never ran)"
fi

if [ "$ALERT_NEEDED" = true ]; then
    echo "[$(date '+%H:%M:%S')] HEALTH CHECK FAILED: $MISSING_JOBS" >> "$LOG_DIR/cron.log"

    cd "$TRADING_DIR"
    "$VENV_PYTHON" -c "
import sys, os
sys.path.insert(0, '$TRADING_DIR')
try:
    from monitoring.telegram_alerter import TelegramAlerter
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if token and chat_id:
        a = TelegramAlerter(token, chat_id)
        a.send('EMERGENCY', 'HEALTH CHECK: Pre-market job did NOT run today!\nMissing: $MISSING_JOBS\nTrading may not execute.', priority='critical')
        import time; time.sleep(2)
except Exception as e:
    print(f'Health alert failed: {e}', file=sys.stderr)
" 2>/dev/null || true
else
    echo "[$(date '+%H:%M:%S')] HEALTH: all OK" >> "$LOG_DIR/cron.log"
fi
