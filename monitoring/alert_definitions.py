"""
Complete alert definitions for the trading system monitoring.
18 alert types across system health, risk, signal health, and execution.
"""

from config import settings

ALERT_DEFINITIONS = {

    # ================================================================
    # SYSTEM HEALTH ALERTS
    # ================================================================

    'DATA_FEED_DISCONNECT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'TrueData WebSocket disconnected',
        'threshold': 'No data received for 60 seconds during market hours',
        'message': 'PRIMARY DATA FEED DOWN. Backup activating.',
        'auto_action': 'Activate Kite ticker backup feed',
        'escalation': 'If backup also fails within 90s → EMERGENCY',
    },

    'BOTH_FEEDS_DOWN': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Both TrueData and Kite ticker failed',
        'threshold': 'Both feeds silent for 90 seconds during market hours',
        'message': 'EMERGENCY: ALL DATA FEEDS DOWN. Positions at risk.',
        'auto_action': 'Close all positions at market after 10 min',
        'note': 'Wait 10 min before emergency close — may be brief outage',
    },

    'KITE_API_TIMEOUT': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Kite API did not respond within 5 seconds',
        'threshold': '1 timeout',
        'message': 'Kite API timeout on order {order_id}',
        'auto_action': 'Retry once. Check if order was placed.',
        'escalation': '3 timeouts in 10 minutes → KITE_API_DOWN (CRITICAL)',
    },

    'KITE_API_DOWN': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Kite API returning errors consistently',
        'threshold': '3 consecutive API failures',
        'message': 'CRITICAL: Kite API down. Trading halted.',
        'auto_action': 'Block all new orders. Keep existing positions.',
        'note': 'Call Zerodha support: 080-33006000',
    },

    'HEARTBEAT_MISSED': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Signal Engine heartbeat not received',
        'threshold': 'No heartbeat for 120 seconds during market hours',
        'message': 'Signal Engine heartbeat missed',
        'auto_action': 'Attempt restart. Alert if restart fails.',
    },

    'POSITION_MISMATCH': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Reconciliation found position discrepancy',
        'threshold': 'Any mismatch between system and Kite positions',
        'message': ('EMERGENCY: Position mismatch detected. '
                    'Manual investigation required.'),
        'auto_action': 'HALT ALL TRADING immediately',
        'note': 'Do NOT auto-close. Unknown positions may be hedges.',
    },

    # ================================================================
    # RISK ALERTS
    # ================================================================

    'DAILY_LOSS_LIMIT_HIT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Daily P&L crosses DAILY_LOSS_LIMIT',
        'threshold': f'net_pnl_today <= -{settings.DAILY_LOSS_LIMIT}',
        'message': 'DAILY LOSS LIMIT HIT: ₹{loss}. No new entries for today.',
        'auto_action': 'Block all new entries. Existing positions held.',
    },

    'DAILY_LOSS_WARNING': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Daily loss approaching limit',
        'threshold': f'net_pnl_today <= -{int(settings.DAILY_LOSS_LIMIT * 0.7)}',
        'message': 'Daily loss at ₹{loss}. Approaching limit.',
        'auto_action': 'Reduce new position sizes to 50%',
    },

    'WEEKLY_LOSS_LIMIT_HIT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Weekly P&L crosses WEEKLY_LOSS_LIMIT',
        'threshold': f'net_pnl_week <= -{settings.WEEKLY_LOSS_LIMIT}',
        'message': 'WEEKLY LOSS LIMIT HIT. Trading suspended this week.',
        'auto_action': ('Block all new entries for rest of week. '
                        'Secondaries suspended 2 weeks.'),
    },

    'MONTHLY_DRAWDOWN_CRITICAL': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Monthly drawdown crosses 15%',
        'threshold': 'monthly_drawdown_pct >= 15',
        'message': ('EMERGENCY: Monthly drawdown {pct}%. '
                    'System entering paper mode.'),
        'auto_action': ('Switch to PAPER_TRADING_MODE. '
                        'All live orders blocked.'),
    },

    'GREEK_LIMIT_BREACH': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Portfolio Greeks approaching limits',
        'threshold': {
            k: v * 0.8 for k, v in settings.GREEK_LIMITS.items()
            if k.startswith('max_portfolio_')
        },
        'message': ('Greek limit approaching: {greek} at {value}. '
                    'New signals filtered.'),
        'auto_action': ('Signal Selector blocks signals '
                        'that worsen breaching Greek'),
    },

    'MARGIN_LOW': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Available margin drops below threshold',
        'threshold': f'available_margin < {int(settings.TOTAL_CAPITAL * 0.30)}',
        'message': ('Low margin: ₹{available} available. '
                    'New entries restricted.'),
        'auto_action': f'Block new entries until margin > {int(settings.TOTAL_CAPITAL * 0.40)}',
    },

    # ================================================================
    # SIGNAL HEALTH ALERTS
    # ================================================================

    'SIGNAL_ROLLING_SHARPE_WATCH': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Signal rolling Sharpe degrading',
        'threshold': 'rolling_sharpe_60d < 0.5',
        'message': ('Signal {signal_id} rolling Sharpe at {sharpe}. '
                    'Status → WATCH.'),
        'auto_action': ('Set signal status to WATCH. '
                        'Reduce size to 50%.'),
    },

    'SIGNAL_DEACTIVATED': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Signal auto-deactivated due to sustained losses',
        'threshold': 'rolling_sharpe_60d < 0.0 for 20 consecutive days',
        'message': ('Signal {signal_id} DEACTIVATED. '
                    'Rolling Sharpe negative 20 days.'),
        'auto_action': 'Set status INACTIVE. Human review required.',
    },

    'MULTIPLE_SIGNALS_DEGRADING': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': '3+ signals enter WATCH in same month',
        'threshold': 'watch_signals_this_month >= 3',
        'message': ('REGIME CHANGE SIGNAL: {n} signals degrading. '
                    'Review all active signals.'),
        'auto_action': ('Reduce ALL signal sizes to 50%. '
                        'Trigger monthly recalibration review.'),
        'note': ('When 3+ signals degrade simultaneously, '
                 'regime has likely changed.'),
    },

    'SIGNAL_ABNORMAL_FREQUENCY': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Signal firing much more or less than expected',
        'threshold': ('actual_fires_30d < 0.4 * expected_fires_30d OR '
                      'actual_fires_30d > 3.0 * expected_fires_30d'),
        'message': ('Signal {signal_id} firing abnormally: '
                    '{actual} fires vs {expected} expected.'),
        'auto_action': 'Log for review. No automatic action.',
    },

    # ================================================================
    # EXECUTION ALERTS
    # ================================================================

    'ORDER_UNFILLED': {
        'level': 'INFO',
        'channel': ['log'],
        'trigger': 'Order cancelled after 90-second fill window',
        'threshold': 'fill_window_expired',
        'message': 'Order {order_id} for {signal_id} unfilled. Cancelled.',
        'auto_action': 'None. Log as UNFILLED trade.',
    },

    'UNUSUAL_SLIPPAGE': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Fill price significantly worse than limit price',
        'threshold': ('slippage > 0.5% of premium for options, '
                      'slippage > 5pts for futures'),
        'message': ('High slippage on {signal_id}: {slippage}pts. '
                    'Market conditions poor.'),
        'auto_action': ('Log. If 3+ high-slippage trades today, '
                        'reduce order sizes by 30%.'),
    },

    'CONSECUTIVE_LOSSES': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Same signal loses N consecutive trades',
        'threshold': 'consecutive_losses_per_signal >= 4',
        'message': 'Signal {signal_id} has {n} consecutive losses.',
        'auto_action': 'Reduce signal to 25% size for next 5 trades.',
    },

    # ================================================================
    # DAILY DIGEST
    # ================================================================

    'DAILY_DIGEST': {
        'level': 'INFO',
        'channel': ['email'],
        'trigger': 'Scheduled — 4:30 PM every trading day',
        'contents': [
            'Today P&L: ₹{daily_pnl}',
            'MTD P&L: ₹{mtd_pnl}',
            'YTD P&L: ₹{ytd_pnl}',
            'Signals fired today: {signals_today}',
            'Signals in WATCH status: {watch_signals}',
            'Signals in INACTIVE status: {inactive_signals}',
            'Portfolio Greeks at close: Δ={delta} V={vega}',
            'Tomorrow regime forecast: {regime_forecast}',
            'Tomorrow calendar events: {events}',
            'Alerts triggered today: {alert_count}',
        ],
    },
}
