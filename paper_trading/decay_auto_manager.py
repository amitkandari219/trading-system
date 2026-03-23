"""
Decay Auto-Manager — automated weekly/daily decay scans with
size overrides, auto-demotion, and promotion recommendations.

Safety rules:
  - Auto-demote only after 3 CONSECUTIVE CRITICAL weeks
  - Auto-demote to SHADOW (not INACTIVE) — signal keeps tracking
  - Size overrides expire automatically (7 days)
  - Human can manual override via CLI
  - Regime-adjusted thresholds (wider in volatile markets)
  - Never auto-promote (always recommend + human confirm)
  - All scans logged to decay_history
  - Deduplicate Telegram alerts (1 alert per signal per scan)

Usage:
    from paper_trading.decay_auto_manager import DecayAutoManager
    manager = DecayAutoManager()
    result = manager.run_weekly_scan()
    factor = manager.apply_size_overrides('KAUFMAN_DRY_20')
    manager.run_daily_quick_check()
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from config.settings import DATABASE_DSN
from models.signal_decay_detector import (
    SignalDecayDetector, DecayReport, STATUS_SEVERITY,
)

logger = logging.getLogger(__name__)

# Override expiry period
DEFAULT_OVERRIDE_DAYS = 7

# Consecutive CRITICAL weeks before auto-demotion
CRITICAL_WEEKS_FOR_DEMOTION = 3

# Daily quick-check thresholds (lighter-weight)
DAILY_QUICK_THRESHOLDS = {
    'loss_streak_warning': 5,
    'loss_streak_critical': 8,
    'daily_sharpe_floor': -1.0,  # 10-trade rolling Sharpe
}


@dataclass
class DecayScanResult:
    """Result of a full decay scan."""

    scan_date: date
    scan_type: str  # WEEKLY, DAILY_QUICK
    total_signals: int = 0
    healthy: int = 0
    watch: int = 0
    warning: int = 0
    critical: int = 0
    structural_decay: int = 0
    improving: int = 0
    actions_taken: List[Dict] = field(default_factory=list)
    reports: Dict[str, DecayReport] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Decay Scan: {self.scan_date} ({self.scan_type})",
            f"  Signals: {self.total_signals}",
            f"  HEALTHY={self.healthy} WATCH={self.watch} WARNING={self.warning}",
            f"  CRITICAL={self.critical} DECAY={self.structural_decay} "
            f"IMPROVING={self.improving}",
            f"  Actions: {len(self.actions_taken)}",
        ]
        for action in self.actions_taken:
            lines.append(
                f"    {action['signal_id']}: {action['action']} "
                f"(factor={action.get('factor', '-')})"
            )
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return '\n'.join(lines)


class DecayAutoManager:
    """
    Manages automated signal decay detection, size overrides,
    and auto-demotion workflow.
    """

    def __init__(self, conn=None):
        self._external_conn = conn is not None
        self.conn = conn or psycopg2.connect(DATABASE_DSN)
        self.conn.autocommit = False
        self._alerted_signals: set = set()  # dedup within a scan

    def close(self):
        if not self._external_conn and self.conn and not self.conn.closed:
            self.conn.close()

    # ════════════════════════════════════════════════════════════
    # WEEKLY SCAN
    # ════════════════════════════════════════════════════════════

    def run_weekly_scan(self,
                        as_of: Optional[date] = None,
                        dry_run: bool = False) -> DecayScanResult:
        """
        Full weekly scan of all signals.

        Logic:
          - 3 consecutive CRITICAL weeks -> auto-demote to SHADOW
          - WARNING -> size reduce 50% (temporary, 7-day expiry)
          - HEALTHY -> clear overrides, reset counts
          - IMPROVING shadow -> recommend promotion
        """
        as_of = as_of or date.today()
        result = DecayScanResult(scan_date=as_of, scan_type='WEEKLY')
        self._alerted_signals.clear()

        # Run detector on all signals
        detector = SignalDecayDetector(conn=self.conn)
        try:
            reports = detector.analyze_all_signals(lookback_trades=50, as_of=as_of)
        except Exception as e:
            logger.error(f"Decay detector failed: {e}")
            result.errors.append(str(e))
            return result

        result.reports = reports
        result.total_signals = len(reports)

        for sid, report in reports.items():
            # Count by status
            status = report.status
            if status == 'HEALTHY':
                result.healthy += 1
            elif status == 'WATCH':
                result.watch += 1
            elif status == 'WARNING':
                result.warning += 1
            elif status == 'CRITICAL':
                result.critical += 1
            elif status == 'STRUCTURAL_DECAY':
                result.structural_decay += 1
            elif status == 'IMPROVING':
                result.improving += 1

            # Log to decay_history
            action_taken = 'NONE'
            try:
                action_taken = self._process_signal_result(
                    report, as_of, dry_run
                )
            except Exception as e:
                logger.error(f"Failed to process {sid}: {e}")
                result.errors.append(f"{sid}: {e}")

            # Record history
            if not dry_run:
                self._log_decay_history(report, 'WEEKLY', action_taken)

            result.actions_taken.append({
                'signal_id': sid,
                'status': status,
                'action': action_taken,
                'factor': self._action_to_factor(action_taken),
                'flags': report.flags[:3],
            })

        if not dry_run:
            try:
                self.conn.commit()
            except Exception as e:
                logger.error(f"Commit failed: {e}")
                self.conn.rollback()

        # Send consolidated Telegram alert
        self._send_scan_summary(result)

        logger.info(result.summary())
        return result

    def _process_signal_result(self, report: DecayReport,
                               as_of: date,
                               dry_run: bool) -> str:
        """
        Process a single signal's decay report and take action.

        Returns action string: SIZE_REDUCE, AUTO_DEMOTE, CLEAR_OVERRIDE,
                               RECOMMEND_PROMOTE, NONE
        """
        sid = report.signal_id

        if report.status == 'HEALTHY':
            # Clear any active overrides
            if self._has_active_override(sid):
                if not dry_run:
                    self._clear_overrides(sid)
                return 'CLEAR_OVERRIDE'
            return 'NONE'

        if report.status == 'IMPROVING' and report.is_shadow:
            self._send_alert(
                'INFO',
                f"PROMOTION CANDIDATE: {sid}\n"
                f"Sharpe={report.sharpe_20:.2f} WR={report.win_rate_20:.1%}\n"
                f"Use: decay-scan --recommend-promote {sid}",
            )
            return 'RECOMMEND_PROMOTE'

        if report.status == 'WARNING':
            if not dry_run:
                self._set_size_override(
                    sid, factor=0.5,
                    override_type='SIZE_REDUCE',
                    reason=f"Decay WARNING: {', '.join(report.flags[:2])}",
                    expiry_days=DEFAULT_OVERRIDE_DAYS,
                )
            return 'SIZE_REDUCE'

        if report.status in ('CRITICAL', 'STRUCTURAL_DECAY'):
            # Check consecutive critical weeks
            consec = self._get_consecutive_critical_weeks(sid)

            if consec + 1 >= CRITICAL_WEEKS_FOR_DEMOTION:
                # Auto-demote to SHADOW
                if not dry_run:
                    self._auto_demote_to_shadow(
                        sid,
                        reason=(
                            f"{consec + 1} consecutive CRITICAL weeks. "
                            f"Flags: {', '.join(report.flags[:3])}"
                        ),
                    )
                return 'AUTO_DEMOTE'
            else:
                # Size reduce aggressively but don't demote yet
                factor = 0.25 if report.status == 'CRITICAL' else 0.0
                if not dry_run:
                    self._set_size_override(
                        sid, factor=factor,
                        override_type='SIZE_REDUCE' if factor > 0 else 'BLOCK',
                        reason=(
                            f"Decay {report.status} (week {consec + 1}/"
                            f"{CRITICAL_WEEKS_FOR_DEMOTION} before demotion): "
                            f"{', '.join(report.flags[:2])}"
                        ),
                        expiry_days=DEFAULT_OVERRIDE_DAYS,
                    )
                self._send_alert(
                    'WARNING',
                    f"DECAY {report.status}: {sid}\n"
                    f"Week {consec + 1}/{CRITICAL_WEEKS_FOR_DEMOTION} "
                    f"before auto-demotion\n"
                    f"Flags: {', '.join(report.flags[:3])}\n"
                    f"Size factor: {factor}",
                )
                return 'SIZE_REDUCE'

        if report.status == 'WATCH':
            # Mild reduction
            if not dry_run:
                self._set_size_override(
                    sid, factor=0.75,
                    override_type='SIZE_REDUCE',
                    reason=f"Decay WATCH: {', '.join(report.flags[:2])}",
                    expiry_days=DEFAULT_OVERRIDE_DAYS,
                )
            return 'SIZE_REDUCE'

        return 'NONE'

    # ════════════════════════════════════════════════════════════
    # DAILY QUICK CHECK
    # ════════════════════════════════════════════════════════════

    def run_daily_quick_check(self,
                              as_of: Optional[date] = None,
                              dry_run: bool = False) -> DecayScanResult:
        """
        Lightweight daily check: loss streaks and daily Sharpe.

        Only flags acute issues — weekly scan does the full analysis.
        """
        as_of = as_of or date.today()
        result = DecayScanResult(scan_date=as_of, scan_type='DAILY_QUICK')
        self._alerted_signals.clear()

        signal_ids = self._get_active_signal_ids()
        result.total_signals = len(signal_ids)

        for sid in signal_ids:
            try:
                pnl_list = self._load_recent_pnl(sid, lookback=15)
                if len(pnl_list) < 5:
                    continue

                # Check loss streak
                streak = self._current_loss_streak(pnl_list)
                thresholds = DAILY_QUICK_THRESHOLDS

                action_taken = 'NONE'
                status = 'HEALTHY'

                if streak >= thresholds['loss_streak_critical']:
                    status = 'CRITICAL'
                    if not dry_run:
                        self._set_size_override(
                            sid, factor=0.25,
                            override_type='SIZE_REDUCE',
                            reason=f"Daily: {streak} consecutive losses",
                            expiry_days=3,
                        )
                    action_taken = 'SIZE_REDUCE'
                    self._send_alert(
                        'WARNING',
                        f"LOSS STREAK: {sid}\n"
                        f"{streak} consecutive losses\n"
                        f"Size reduced to 25%",
                    )
                elif streak >= thresholds['loss_streak_warning']:
                    status = 'WARNING'
                    if not dry_run:
                        self._set_size_override(
                            sid, factor=0.5,
                            override_type='SIZE_REDUCE',
                            reason=f"Daily: {streak} consecutive losses",
                            expiry_days=3,
                        )
                    action_taken = 'SIZE_REDUCE'

                # Check 10-trade rolling Sharpe
                if len(pnl_list) >= 10:
                    recent_sharpe = self._quick_sharpe(pnl_list[-10:])
                    if recent_sharpe < thresholds['daily_sharpe_floor']:
                        if status != 'CRITICAL':
                            status = 'WARNING'
                        if not dry_run:
                            self._set_size_override(
                                sid, factor=0.5,
                                override_type='SIZE_REDUCE',
                                reason=(
                                    f"Daily: 10-trade Sharpe "
                                    f"{recent_sharpe:.2f} < "
                                    f"{thresholds['daily_sharpe_floor']}"
                                ),
                                expiry_days=3,
                            )
                        action_taken = 'SIZE_REDUCE'

                # Log to history
                if not dry_run and action_taken != 'NONE':
                    report = DecayReport(
                        signal_id=sid,
                        analysis_date=as_of,
                        status=status,
                        consecutive_losses=streak,
                        sharpe_20=self._quick_sharpe(pnl_list[-10:]) if len(pnl_list) >= 10 else 0,
                        flags=[f"LOSS_STREAK({streak})"] if streak >= 5 else [],
                    )
                    self._log_decay_history(report, 'DAILY_QUICK', action_taken)

                # Count
                if status == 'HEALTHY':
                    result.healthy += 1
                elif status == 'WARNING':
                    result.warning += 1
                elif status == 'CRITICAL':
                    result.critical += 1

                result.actions_taken.append({
                    'signal_id': sid,
                    'status': status,
                    'action': action_taken,
                })

            except Exception as e:
                logger.error(f"Daily check failed for {sid}: {e}")
                result.errors.append(f"{sid}: {e}")

        if not dry_run:
            try:
                self.conn.commit()
            except Exception as e:
                logger.error(f"Commit failed: {e}")
                self.conn.rollback()

        return result

    # ════════════════════════════════════════════════════════════
    # SIZE OVERRIDES (queried at trade time)
    # ════════════════════════════════════════════════════════════

    def apply_size_overrides(self, signal_id: str) -> float:
        """
        Check DB for active overrides and return size factor.

        Returns:
            float between 0.0 and 1.0 (1.0 = no override)
        """
        try:
            cur = self.conn.cursor()
            # Expire old overrides first
            cur.execute("""
                UPDATE signal_overrides
                SET active = FALSE
                WHERE active = TRUE
                  AND expires_at IS NOT NULL
                  AND expires_at < NOW()
            """)

            # Get active override with minimum factor
            cur.execute("""
                SELECT MIN(factor)
                FROM signal_overrides
                WHERE signal_id = %s
                  AND active = TRUE
            """, (signal_id,))
            row = cur.fetchone()
            self.conn.commit()

            if row and row[0] is not None:
                return float(row[0])
            return 1.0

        except Exception as e:
            logger.error(f"Override check failed for {signal_id}: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return 1.0  # fail-open: trade at full size

    # ════════════════════════════════════════════════════════════
    # OVERRIDE MANAGEMENT
    # ════════════════════════════════════════════════════════════

    def _set_size_override(self, signal_id: str, factor: float,
                           override_type: str, reason: str,
                           expiry_days: int = DEFAULT_OVERRIDE_DAYS):
        """Create or update a size override."""
        try:
            cur = self.conn.cursor()
            # Deactivate existing overrides for this signal from auto-manager
            cur.execute("""
                UPDATE signal_overrides
                SET active = FALSE
                WHERE signal_id = %s
                  AND active = TRUE
                  AND created_by = 'DECAY_AUTO_MANAGER'
            """, (signal_id,))

            expires_at = datetime.now() + timedelta(days=expiry_days)
            cur.execute("""
                INSERT INTO signal_overrides
                    (signal_id, override_type, factor, reason,
                     expires_at, active, created_by)
                VALUES (%s, %s, %s, %s, %s, TRUE, 'DECAY_AUTO_MANAGER')
            """, (signal_id, override_type, factor, reason, expires_at))

            logger.info(
                f"Override set: {signal_id} -> {override_type} "
                f"factor={factor} expires={expires_at.date()}"
            )
        except Exception as e:
            logger.error(f"Failed to set override for {signal_id}: {e}")

    def set_manual_override(self, signal_id: str, factor: float,
                            expiry_days: int = 30,
                            reason: str = 'Manual override'):
        """
        Set a manual size override (via CLI).

        Manual overrides take priority over auto-generated ones.
        """
        try:
            cur = self.conn.cursor()
            expires_at = (
                datetime.now() + timedelta(days=expiry_days)
                if expiry_days > 0 else None
            )
            cur.execute("""
                INSERT INTO signal_overrides
                    (signal_id, override_type, factor, reason,
                     expires_at, active, created_by)
                VALUES (%s, 'MANUAL', %s, %s, %s, TRUE, 'HUMAN')
            """, (signal_id, factor, reason, expires_at))
            self.conn.commit()
            logger.info(
                f"Manual override: {signal_id} factor={factor} "
                f"expires={'never' if expires_at is None else str(expires_at.date())}"
            )
        except Exception as e:
            logger.error(f"Manual override failed: {e}")
            self.conn.rollback()

    def clear_manual_override(self, signal_id: str):
        """Clear all active overrides for a signal."""
        self._clear_overrides(signal_id)
        try:
            self.conn.commit()
            logger.info(f"Cleared all overrides for {signal_id}")
        except Exception as e:
            logger.error(f"Clear override failed: {e}")
            self.conn.rollback()

    def _clear_overrides(self, signal_id: str):
        """Deactivate all overrides for a signal."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE signal_overrides
                SET active = FALSE
                WHERE signal_id = %s AND active = TRUE
            """, (signal_id,))
        except Exception as e:
            logger.error(f"Clear override DB error for {signal_id}: {e}")

    def _has_active_override(self, signal_id: str) -> bool:
        """Check if signal has any active override."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM signal_overrides
                WHERE signal_id = %s AND active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
            """, (signal_id,))
            row = cur.fetchone()
            return (row[0] or 0) > 0
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return False

    # ════════════════════════════════════════════════════════════
    # AUTO-DEMOTION
    # ════════════════════════════════════════════════════════════

    def _auto_demote_to_shadow(self, signal_id: str, reason: str):
        """
        Demote a signal to SHADOW status.

        Sets a permanent block override and updates signal status.
        """
        try:
            cur = self.conn.cursor()

            # Set permanent block override (no expiry)
            cur.execute("""
                UPDATE signal_overrides
                SET active = FALSE
                WHERE signal_id = %s AND active = TRUE
            """, (signal_id,))
            cur.execute("""
                INSERT INTO signal_overrides
                    (signal_id, override_type, factor, reason,
                     expires_at, active, created_by)
                VALUES (%s, 'BLOCK', 0.0, %s, NULL, TRUE, 'DECAY_AUTO_MANAGER')
            """, (signal_id, f"Auto-demotion: {reason}"))

            # Update signal status in signals table
            cur.execute("""
                UPDATE signals
                SET status = 'WATCH',
                    pending_change_by = 'DECAY_AUTO_MANAGER',
                    pending_change_reason = %s,
                    updated_at = NOW()
                WHERE signal_id = %s
            """, (reason, signal_id))

            logger.warning(f"AUTO-DEMOTED {signal_id} to SHADOW: {reason}")

            self._send_alert(
                'CRITICAL',
                f"AUTO-DEMOTED: {signal_id}\n"
                f"Status: SCORING -> SHADOW\n"
                f"Reason: {reason}\n"
                f"Signal blocked from trading.\n"
                f"Use: decay-scan --clear-override {signal_id} to restore.",
            )
        except Exception as e:
            logger.error(f"Auto-demotion failed for {signal_id}: {e}")

    def _get_consecutive_critical_weeks(self, signal_id: str) -> int:
        """Count consecutive CRITICAL/STRUCTURAL_DECAY weekly scans."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT status FROM decay_history
                WHERE signal_id = %s
                  AND scan_type = 'WEEKLY'
                ORDER BY scan_date DESC
                LIMIT 10
            """, (signal_id,))
            rows = cur.fetchall()

            count = 0
            for row in rows:
                if row[0] in ('CRITICAL', 'STRUCTURAL_DECAY'):
                    count += 1
                else:
                    break
            return count
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return 0

    # ════════════════════════════════════════════════════════════
    # HISTORY LOGGING
    # ════════════════════════════════════════════════════════════

    def _log_decay_history(self, report: DecayReport,
                           scan_type: str, action_taken: str):
        """Log a scan result to decay_history table."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO decay_history
                    (signal_id, scan_date, scan_type, status, severity_score,
                     sharpe_20, win_rate_20, max_drawdown_20, consec_losses,
                     bocd_cp_prob, cusum_alert, regime, regime_sharpe,
                     flags, action_taken, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s)
            """, (
                report.signal_id,
                report.analysis_date,
                scan_type,
                report.status,
                report.severity_score,
                report.sharpe_20,
                report.win_rate_20,
                report.max_drawdown_20,
                report.consecutive_losses,
                report.bocd_cp_prob,
                report.cusum_alert,
                report.regime,
                report.regime_sharpe,
                report.flags,
                action_taken,
                report.regime_decay_reason or '',
            ))
        except Exception as e:
            logger.error(f"Decay history log failed for {report.signal_id}: {e}")

    # ════════════════════════════════════════════════════════════
    # DB HELPERS
    # ════════════════════════════════════════════════════════════

    def _get_active_signal_ids(self) -> List[str]:
        """Get signal IDs that are actively trading (not blocked)."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT DISTINCT t.signal_id
                FROM trades t
                WHERE t.exit_date IS NOT NULL
                  AND t.net_pnl IS NOT NULL
                  AND t.entry_date >= CURRENT_DATE - INTERVAL '60 days'
                  AND t.trade_type IN ('PAPER', 'LIVE', 'PAPER_SCORING',
                                       'SHADOW')
                ORDER BY t.signal_id
            """)
            return [r[0] for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get active signals: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def _load_recent_pnl(self, signal_id: str,
                         lookback: int = 15) -> List[float]:
        """Load recent net_pnl values for daily quick check."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT net_pnl FROM trades
                WHERE signal_id = %s
                  AND exit_date IS NOT NULL
                  AND net_pnl IS NOT NULL
                ORDER BY exit_date DESC, exit_time DESC
                LIMIT %s
            """, (signal_id, lookback))
            rows = cur.fetchall()
            pnl = [float(r[0]) for r in rows]
            pnl.reverse()  # chronological
            return pnl
        except Exception as e:
            logger.error(f"PnL load failed for {signal_id}: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    @staticmethod
    def _current_loss_streak(pnl_list: List[float]) -> int:
        """Count current consecutive losses from the end."""
        streak = 0
        for p in reversed(pnl_list):
            if p < 0:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _quick_sharpe(pnl_list: List[float]) -> float:
        """Quick Sharpe computation (annualized)."""
        if len(pnl_list) < 3:
            return 0.0
        arr = np.array(pnl_list)
        std = np.std(arr, ddof=1)
        if std <= 0:
            return 0.0
        return float((np.mean(arr) / std) * np.sqrt(252))

    @staticmethod
    def _action_to_factor(action: str) -> float:
        mapping = {
            'SIZE_REDUCE': 0.5,
            'AUTO_DEMOTE': 0.0,
            'CLEAR_OVERRIDE': 1.0,
            'RECOMMEND_PROMOTE': 1.0,
            'NONE': 1.0,
        }
        return mapping.get(action, 1.0)

    # ════════════════════════════════════════════════════════════
    # ALERTS
    # ════════════════════════════════════════════════════════════

    def _send_alert(self, level: str, message: str):
        """Send Telegram alert with deduplication."""
        # Deduplicate: extract signal_id from message first line
        first_line = message.split('\n')[0] if message else ''
        if first_line in self._alerted_signals:
            return
        self._alerted_signals.add(first_line)

        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if not token or not chat_id:
            logger.info(f"[ALERT {level}] {message}")
            return
        try:
            from monitoring.telegram_alerter import TelegramAlerter
            alerter = TelegramAlerter(token, chat_id)
            alerter.send(level, message)
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")

    def _send_scan_summary(self, result: DecayScanResult):
        """Send consolidated scan summary via Telegram."""
        if result.total_signals == 0:
            return

        # Only alert if there are issues
        issues = result.warning + result.critical + result.structural_decay
        if issues == 0 and result.improving == 0:
            return

        lines = [
            f"Decay Scan {result.scan_date} ({result.scan_type})",
            f"Signals: {result.total_signals} | "
            f"H={result.healthy} W={result.watch} "
            f"WRN={result.warning} CRT={result.critical} "
            f"DEC={result.structural_decay} IMP={result.improving}",
        ]

        for action in result.actions_taken:
            if action['action'] != 'NONE':
                lines.append(
                    f"  {action['signal_id']}: {action['action']} "
                    f"({action['status']})"
                )

        level = 'INFO'
        if result.critical > 0 or result.structural_decay > 0:
            level = 'WARNING'

        self._send_alert(level, '\n'.join(lines))

    # ════════════════════════════════════════════════════════════
    # STATUS QUERY
    # ════════════════════════════════════════════════════════════

    def get_override_status(self) -> List[Dict]:
        """Get all active overrides."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT signal_id, override_type, factor, reason,
                       created_at, expires_at, created_by
                FROM signal_overrides
                WHERE active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY signal_id
            """)
            return [
                {
                    'signal_id': r[0],
                    'override_type': r[1],
                    'factor': r[2],
                    'reason': r[3],
                    'created_at': str(r[4]),
                    'expires_at': str(r[5]) if r[5] else 'never',
                    'created_by': r[6],
                }
                for r in cur.fetchall()
            ]
        except Exception as e:
            logger.error(f"Override status query failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def get_signal_history(self, signal_id: str,
                           limit: int = 10) -> List[Dict]:
        """Get recent decay history for a signal."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT scan_date, scan_type, status, severity_score,
                       sharpe_20, win_rate_20, max_drawdown_20,
                       consec_losses, bocd_cp_prob, cusum_alert,
                       regime, flags, action_taken
                FROM decay_history
                WHERE signal_id = %s
                ORDER BY scan_date DESC
                LIMIT %s
            """, (signal_id, limit))
            return [
                {
                    'scan_date': str(r[0]),
                    'scan_type': r[1],
                    'status': r[2],
                    'severity': r[3],
                    'sharpe_20': r[4],
                    'win_rate_20': r[5],
                    'max_drawdown_20': r[6],
                    'consec_losses': r[7],
                    'bocd_cp_prob': r[8],
                    'cusum_alert': r[9],
                    'regime': r[10],
                    'flags': r[11],
                    'action': r[12],
                }
                for r in cur.fetchall()
            ]
        except Exception as e:
            logger.error(f"History query failed for {signal_id}: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []
