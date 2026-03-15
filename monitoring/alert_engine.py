"""
Alert engine — checks thresholds, routes alerts to channels, logs to DB.
"""

import logging
from datetime import datetime

from monitoring.alert_definitions import ALERT_DEFINITIONS


class AlertEngine:
    """
    Central alert processor.
    Checks thresholds, deduplicates, routes to appropriate channels,
    and logs everything to the alerts_log table.
    """

    def __init__(self, db, telegram_alerter, logger=None):
        """
        db:               database connection
        telegram_alerter: TelegramAlerter instance
        logger:           Python Logger
        """
        self.db       = db
        self.alerter  = telegram_alerter
        self.logger   = logger or logging.getLogger(__name__)
        self._recent_alerts = {}  # alert_type -> last_sent_time (dedup)

    def check_and_alert(self, alert_type: str, context: dict = None):
        """
        Trigger an alert by type. Looks up definition, routes to channels.

        alert_type: key from ALERT_DEFINITIONS
        context:    dict with template variables for the message
        """
        defn = ALERT_DEFINITIONS.get(alert_type)
        if not defn:
            self.logger.warning(f"Unknown alert type: {alert_type}")
            return

        context = context or {}
        level = defn['level']
        channels = defn.get('channel', ['log'])

        # Format message with context
        message = defn.get('message', alert_type)
        try:
            message = message.format(**context)
        except (KeyError, IndexError):
            pass  # Use raw message if formatting fails

        # Deduplicate: don't send same alert more than once per 5 minutes
        now = datetime.now()
        last_sent = self._recent_alerts.get(alert_type)
        if last_sent and (now - last_sent).total_seconds() < 300:
            self.logger.debug(f"Suppressing duplicate alert: {alert_type}")
            return

        self._recent_alerts[alert_type] = now

        # Route to channels
        if 'phone' in channels and level in ('CRITICAL', 'EMERGENCY', 'WARNING'):
            try:
                self.alerter.send(
                    level, message,
                    signal_id=context.get('signal_id')
                )
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert: {e}")

        # Always log to database
        self._log_to_db(alert_type, level, message, context)

        # Log locally
        log_method = {
            'INFO': self.logger.info,
            'WARNING': self.logger.warning,
            'CRITICAL': self.logger.critical,
            'EMERGENCY': self.logger.critical,
        }.get(level, self.logger.info)
        log_method(f"[{alert_type}] {message}")

    def _log_to_db(self, alert_type, level, message, context):
        """Log alert to the alerts_log table."""
        try:
            self.db.execute("""
                INSERT INTO alerts_log
                    (alert_type, level, message, signal_id, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (
                alert_type,
                level,
                message,
                context.get('signal_id'),
            ))
        except Exception as e:
            self.logger.error(f"Failed to log alert to DB: {e}")

    def check_daily_loss(self, net_pnl_today: float):
        """Check daily P&L against loss limits."""
        from config import settings

        if net_pnl_today <= -settings.DAILY_LOSS_LIMIT:
            self.check_and_alert('DAILY_LOSS_LIMIT_HIT', {
                'loss': f"{abs(net_pnl_today):,.0f}"
            })
        elif net_pnl_today <= -settings.DAILY_LOSS_LIMIT * 0.7:
            self.check_and_alert('DAILY_LOSS_WARNING', {
                'loss': f"{abs(net_pnl_today):,.0f}"
            })

    def check_weekly_loss(self, net_pnl_week: float):
        """Check weekly P&L against loss limits."""
        from config import settings

        if net_pnl_week <= -settings.WEEKLY_LOSS_LIMIT:
            self.check_and_alert('WEEKLY_LOSS_LIMIT_HIT')

    def check_greek_limits(self, portfolio_greeks):
        """Check portfolio Greeks against warning thresholds."""
        from config import settings

        limits = settings.GREEK_LIMITS
        warn_fraction = 0.8

        if abs(portfolio_greeks.delta) > limits['max_portfolio_delta'] * warn_fraction:
            self.check_and_alert('GREEK_LIMIT_BREACH', {
                'greek': 'Delta',
                'value': f"{portfolio_greeks.delta:.3f}"
            })

        if abs(portfolio_greeks.vega) > limits['max_portfolio_vega'] * warn_fraction:
            self.check_and_alert('GREEK_LIMIT_BREACH', {
                'greek': 'Vega',
                'value': f"{portfolio_greeks.vega:.0f}"
            })

        if abs(portfolio_greeks.gamma) > limits['max_portfolio_gamma'] * warn_fraction:
            self.check_and_alert('GREEK_LIMIT_BREACH', {
                'greek': 'Gamma',
                'value': f"{portfolio_greeks.gamma:.0f}"
            })

        if portfolio_greeks.theta < limits['max_portfolio_theta'] * warn_fraction:
            self.check_and_alert('GREEK_LIMIT_BREACH', {
                'greek': 'Theta',
                'value': f"{portfolio_greeks.theta:.0f}"
            })

    def check_signal_health(self, signal_id: str, rolling_sharpe_60d: float,
                             consecutive_negative_days: int = 0):
        """Check signal health metrics."""
        if rolling_sharpe_60d < 0.0 and consecutive_negative_days >= 20:
            self.check_and_alert('SIGNAL_DEACTIVATED', {
                'signal_id': signal_id,
            })
        elif rolling_sharpe_60d < 0.5:
            self.check_and_alert('SIGNAL_ROLLING_SHARPE_WATCH', {
                'signal_id': signal_id,
                'sharpe': f"{rolling_sharpe_60d:.2f}",
            })
