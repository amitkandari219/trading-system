"""
Centralized exception handling for paper trading operations.

Replaces bare 'except Exception: pass' patterns with proper logging,
traceback capture, and optional Telegram alerting for critical failures.

This module provides:
- @safe_db_operation decorator for database operations
- SignalComputeError custom exception for structured error handling
- log_and_alert() for consistent logging + alerting
- safe_call() for one-off safe function invocations

All exceptions are logged with full tracebacks and optional critical alerts.
"""

import functools
import logging
import os
import traceback
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class SignalComputeError(Exception):
    """Raised when signal computation fails critically.

    This exception wraps lower-level errors with context about which
    signal and operation failed, making debugging easier.

    Attributes:
        signal_id: The signal identifier that was being processed.
        operation: Description of the operation that failed.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(self, signal_id: str, operation: str, original_error: Exception):
        """Initialize a SignalComputeError.

        Args:
            signal_id: The signal identifier being processed.
            operation: Description of the operation (e.g., 'load_positions', 'compute_entry').
            original_error: The underlying exception.
        """
        self.signal_id = signal_id
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"[{signal_id}] {operation} failed: {original_error}")


def log_and_alert(level: str, message: str, signal_id: str = None,
                  alert_telegram: bool = False) -> None:
    """Log a message and optionally send a Telegram alert.

    Provides centralized logging with optional critical alerts via Telegram.
    All messages are logged with appropriate severity. Critical messages can
    trigger Telegram notifications if enabled and credentials are available.

    Args:
        level: Log level string ('WARNING', 'ERROR', 'CRITICAL').
        message: Human-readable message to log.
        signal_id: Optional signal identifier to include in the message.
        alert_telegram: If True, attempt to send alert via Telegram for CRITICAL messages.

    Example:
        log_and_alert('ERROR', 'Failed to compute entry price', signal_id='NIFTY_LONG_1')
        log_and_alert('CRITICAL', 'Database connection lost', alert_telegram=True)
    """
    prefix = f"[{signal_id}] " if signal_id else ""
    full_msg = f"{prefix}{message}"

    if level == 'CRITICAL':
        logger.critical(full_msg)
    elif level == 'ERROR':
        logger.error(full_msg)
    else:
        logger.warning(full_msg)

    if alert_telegram:
        _send_telegram_alert(level, full_msg)


def _send_telegram_alert(level: str, message: str) -> None:
    """Send a Telegram alert (best-effort, never raises).

    Attempts to send an alert via Telegram if bot token and chat ID
    are configured. Failures are logged at debug level and never propagated.

    Args:
        level: Severity level ('WARNING', 'ERROR', 'CRITICAL').
        message: Message text to send.
    """
    try:
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if not (token and chat_id):
            return
        from monitoring.telegram_alerter import TelegramAlerter
        TelegramAlerter(token, chat_id).send(level, message)
    except Exception as e:
        logger.debug(f"Telegram alert failed (non-critical): {e}")


def safe_db_operation(operation_name: str, default_return: Any = None,
                      alert_on_failure: bool = False,
                      signal_id: str = None) -> Callable:
    """Decorator for database operations that should not crash the pipeline.

    Wraps functions to catch and log all exceptions with full tracebacks.
    Replaces bare 'except Exception: pass' patterns with proper error handling.
    Optionally sends Telegram alerts for critical failures.

    The decorated function will:
    - Log full exception + traceback on failure
    - Send Telegram alert if alert_on_failure=True
    - Return default_return value instead of raising

    Args:
        operation_name: Human-readable name for logging (e.g., 'load_positions', 'compute_sizing').
        default_return: Value to return if function fails (default None). Typically [] or {}.
        alert_on_failure: If True, send Telegram alert on exception (for critical operations).
        signal_id: Optional signal ID for context in error messages.

    Returns:
        Decorator function.

    Example:
        @safe_db_operation('load_open_positions', default_return=[], alert_on_failure=False)
        def _load_open_positions(self, trade_type='PAPER'):
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE trade_type = %s", (trade_type,))
            return cursor.fetchall()

        positions = self._load_open_positions()  # Returns [] on error, never raises
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                sid = signal_id or kwargs.get('signal_id', 'UNKNOWN')
                log_and_alert(
                    'ERROR',
                    f"{operation_name} failed: {e}\n{tb}",
                    signal_id=sid,
                    alert_telegram=alert_on_failure,
                )
                return default_return
        return wrapper
    return decorator


def safe_call(func: Callable, operation_name: str, default_return: Any = None,
              signal_id: str = None, alert: bool = False) -> Any:
    """Functional wrapper for one-off safe calls (non-decorator usage).

    Provides exception handling for lambda functions or one-time operations
    without needing to write a decorator. Useful for inline safety wrapping.

    Args:
        func: Callable to execute safely.
        operation_name: Human-readable name for logging.
        default_return: Value to return if func raises (default None).
        signal_id: Optional signal identifier for context.
        alert: If True, send Telegram alert on exception.

    Returns:
        Result of func() if successful, default_return on exception.

    Example:
        rows = safe_call(
            lambda: cur.fetchall(),
            'fetch_sizing_context',
            default_return=[],
            signal_id='NIFTY_LONG_1'
        )
    """
    try:
        return func()
    except Exception as e:
        tb = traceback.format_exc()
        log_and_alert(
            'ERROR',
            f"{operation_name} failed: {e}\n{tb}",
            signal_id=signal_id,
            alert_telegram=alert,
        )
        return default_return
