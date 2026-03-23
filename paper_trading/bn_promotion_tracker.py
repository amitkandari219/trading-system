"""
BankNifty Signal Promotion Tracker — evaluates SHADOW BN_ signals for promotion.

Lifecycle: SHADOW -> READY_TO_PROMOTE -> (human confirm) -> SCORING -> (demotion check) -> DEMOTED

Promotion criteria (must ALL pass):
  - min 15 trading days of data
  - min 30 trades
  - Sharpe >= 1.0
  - Win rate >= 40%
  - Profit factor >= 1.3
  - Max drawdown <= 15%
  - W/L ratio >= 1.5
  - Max 6 consecutive losses
  - Nifty correlation 0.3-0.7 (diversification sweet spot)

Demotion criteria (any ONE triggers review):
  - Rolling 20-day drawdown > 20%
  - Rolling 20-day Sharpe < 0.5
  - CUSUM changepoint detected (decay)

Safety:
  - No auto-promote — requires human Telegram confirmation
  - BN signals start SHADOW only
  - BN + Nifty share position limits (4 total, 2 same-direction)

Usage:
    from paper_trading.bn_promotion_tracker import BNPromotionTracker
    tracker = BNPromotionTracker()
    report = tracker.evaluate_signal('BN_KAUFMAN_BB_MR')
    reports = tracker.evaluate_all_bn_signals()
    tracker.check_demotion('BN_KAUFMAN_BB_MR')
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from config.settings import DATABASE_DSN, TOTAL_CAPITAL
from models.decay_cusum import CUSUMDetector

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# PROMOTION / DEMOTION CRITERIA
# ═══════════════════════════════════════════════════════════

PROMOTION_CRITERIA = {
    'min_days': 15,
    'min_trades': 30,
    'min_sharpe': 1.0,
    'min_win_rate': 0.40,
    'min_profit_factor': 1.3,
    'max_drawdown_pct': 0.15,
    'min_wl_ratio': 1.5,
    'max_consecutive_losses': 6,
    'min_nifty_corr': 0.3,
    'max_nifty_corr': 0.7,
}

DEMOTION_CRITERIA = {
    'rolling_dd_threshold': 0.20,    # 20-day rolling DD > 20%
    'rolling_sharpe_floor': 0.5,     # 20-day Sharpe < 0.5
    'cusum_decay': True,             # CUSUM changepoint detection
}

# Rolling window for demotion checks
DEMOTION_WINDOW_DAYS = 20

# All BN_ signal IDs (must match BankNiftySignalComputer.SIGNALS keys)
BN_SIGNAL_IDS = [
    'BN_KAUFMAN_BB_MR',
    'BN_GUJRAL_RANGE',
    'BN_ORB_BREAKOUT',
    'BN_VWAP_RECLAIM',
    'BN_VWAP_REJECTION',
    'BN_FIRST_PULLBACK',
    'BN_FAILED_BREAKOUT',
    'BN_GAP_FILL',
    'BN_TREND_BAR',
    'BN_EOD_TREND',
]


@dataclass
class PromotionReport:
    """Result of evaluating a BN signal for promotion or demotion."""

    signal_id: str
    evaluation_date: date
    current_status: str        # SHADOW, SCORING, READY_TO_PROMOTE, DEMOTED

    # Data sufficiency
    trading_days: int = 0
    trade_count: int = 0

    # Performance metrics
    sharpe: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    wl_ratio: float = 0.0
    consecutive_losses_max: int = 0
    nifty_correlation: float = 0.0
    total_pnl: float = 0.0

    # Rolling metrics (for demotion)
    rolling_dd_pct: float = 0.0
    rolling_sharpe: float = 0.0
    cusum_detected: bool = False

    # Verdict
    promotion_eligible: bool = False
    demotion_flagged: bool = False
    fail_reasons: List[str] = field(default_factory=list)
    demotion_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'evaluation_date': str(self.evaluation_date),
            'current_status': self.current_status,
            'trading_days': self.trading_days,
            'trade_count': self.trade_count,
            'sharpe': round(self.sharpe, 2),
            'win_rate': round(self.win_rate, 3),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 3),
            'wl_ratio': round(self.wl_ratio, 2),
            'consecutive_losses_max': self.consecutive_losses_max,
            'nifty_correlation': round(self.nifty_correlation, 3),
            'total_pnl': round(self.total_pnl, 0),
            'rolling_dd_pct': round(self.rolling_dd_pct, 3),
            'rolling_sharpe': round(self.rolling_sharpe, 2),
            'cusum_detected': self.cusum_detected,
            'promotion_eligible': self.promotion_eligible,
            'demotion_flagged': self.demotion_flagged,
            'fail_reasons': self.fail_reasons,
            'demotion_reasons': self.demotion_reasons,
        }

    def summary_line(self) -> str:
        """One-line summary for tables."""
        status_icon = {
            'SHADOW': 'SHD',
            'READY_TO_PROMOTE': 'RDY',
            'SCORING': 'SCR',
            'DEMOTED': 'DEM',
        }.get(self.current_status, '???')

        verdict = ''
        if self.promotion_eligible:
            verdict = 'PROMOTE'
        elif self.demotion_flagged:
            verdict = 'DEMOTE?'
        else:
            verdict = 'HOLD'

        return (
            f"{self.signal_id:<22s} {status_icon:>3s} "
            f"{self.trade_count:>4d}t {self.sharpe:>5.2f}S "
            f"{self.win_rate:>5.1%}WR {self.profit_factor:>5.2f}PF "
            f"{self.max_drawdown_pct:>5.1%}DD {self.wl_ratio:>4.1f}WL "
            f"{self.total_pnl:>+8,.0f} {verdict}"
        )


class BNPromotionTracker:
    """
    Evaluates BankNifty SHADOW signals against promotion/demotion criteria.

    Reads trade data from the `trades` table (trade_type = 'SHADOW',
    instrument = 'BANKNIFTY').
    """

    def __init__(self, conn=None):
        self._external_conn = conn is not None
        self.conn = conn or psycopg2.connect(DATABASE_DSN)

    def close(self):
        if not self._external_conn and self.conn and not self.conn.closed:
            self.conn.close()

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def evaluate_signal(self, signal_id: str,
                        as_of: Optional[date] = None) -> PromotionReport:
        """
        Evaluate a single BN signal for promotion readiness.

        Loads all SHADOW trades for the signal, computes performance
        metrics, and compares against promotion criteria.
        """
        as_of = as_of or date.today()
        status = self._get_signal_status(signal_id)
        report = PromotionReport(
            signal_id=signal_id,
            evaluation_date=as_of,
            current_status=status,
        )

        # Load trades
        trades = self._load_trades(signal_id, as_of)
        if not trades:
            report.fail_reasons.append('NO_TRADES')
            return report

        pnl_list = [t['pnl'] for t in trades if t['pnl'] is not None]
        if not pnl_list:
            report.fail_reasons.append('NO_COMPLETED_TRADES')
            return report

        # Trading days
        trade_dates = sorted(set(t['entry_date'] for t in trades))
        report.trading_days = len(trade_dates)
        report.trade_count = len(pnl_list)

        # Core metrics
        report.sharpe = self._compute_sharpe(pnl_list)
        report.win_rate = self._compute_win_rate(pnl_list)
        report.profit_factor = self._compute_profit_factor(pnl_list)
        report.max_drawdown_pct = self._compute_max_drawdown(pnl_list)
        report.wl_ratio = self._compute_wl_ratio(pnl_list)
        report.consecutive_losses_max = self._compute_max_consecutive_losses(pnl_list)
        report.total_pnl = sum(pnl_list)

        # Nifty correlation
        report.nifty_correlation = self._compute_nifty_correlation(
            trades, as_of
        )

        # Rolling metrics (for demotion)
        recent_pnl = pnl_list[-DEMOTION_WINDOW_DAYS:] if len(pnl_list) >= DEMOTION_WINDOW_DAYS else pnl_list
        report.rolling_dd_pct = self._compute_max_drawdown(recent_pnl)
        report.rolling_sharpe = self._compute_sharpe(recent_pnl)

        # CUSUM decay detection
        report.cusum_detected = self._check_cusum_decay(pnl_list)

        # Evaluate promotion
        report.promotion_eligible = self._check_promotion(report)

        # Evaluate demotion (only for SCORING signals)
        if status == 'SCORING':
            report.demotion_flagged = self._check_demotion_criteria(report)

        return report

    def evaluate_all_bn_signals(self,
                                as_of: Optional[date] = None) -> Dict[str, PromotionReport]:
        """Evaluate all 10 BN signals."""
        as_of = as_of or date.today()
        reports = {}
        for signal_id in BN_SIGNAL_IDS:
            try:
                reports[signal_id] = self.evaluate_signal(signal_id, as_of)
            except Exception as e:
                logger.error(f"Failed to evaluate {signal_id}: {e}")
                reports[signal_id] = PromotionReport(
                    signal_id=signal_id,
                    evaluation_date=as_of,
                    current_status='ERROR',
                    fail_reasons=[str(e)],
                )
        return reports

    def promote_signal(self, signal_id: str) -> bool:
        """
        Promote a BN signal from SHADOW to SCORING.

        Updates the signal_status table and sends a Telegram alert.
        Returns True if successful.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO signal_status (signal_id, status, instrument, updated_at, notes)
                VALUES (%s, 'SCORING', 'BANKNIFTY', NOW(), 'Promoted via BN promotion tracker')
                ON CONFLICT (signal_id)
                DO UPDATE SET status = 'SCORING', updated_at = NOW(),
                             notes = 'Promoted via BN promotion tracker'
            """, (signal_id,))
            self.conn.commit()
            logger.info(f"PROMOTED {signal_id} to SCORING")
            self._send_alert(
                'INFO',
                f"PROMOTED: {signal_id}\n"
                f"Status: SHADOW -> SCORING\n"
                f"Instrument: BANKNIFTY"
            )
            return True
        except Exception as e:
            logger.error(f"Promotion failed for {signal_id}: {e}")
            self.conn.rollback()
            return False

    def demote_signal(self, signal_id: str, reason: str = '') -> bool:
        """Demote a BN signal from SCORING to DEMOTED."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO signal_status (signal_id, status, instrument, updated_at, notes)
                VALUES (%s, 'DEMOTED', 'BANKNIFTY', NOW(), %s)
                ON CONFLICT (signal_id)
                DO UPDATE SET status = 'DEMOTED', updated_at = NOW(), notes = %s
            """, (signal_id, reason, reason))
            self.conn.commit()
            logger.info(f"DEMOTED {signal_id}: {reason}")
            self._send_alert(
                'WARNING',
                f"DEMOTED: {signal_id}\n"
                f"Status: SCORING -> DEMOTED\n"
                f"Reason: {reason}"
            )
            return True
        except Exception as e:
            logger.error(f"Demotion failed for {signal_id}: {e}")
            self.conn.rollback()
            return False

    def check_demotion(self, signal_id: str,
                       as_of: Optional[date] = None) -> PromotionReport:
        """Check a SCORING signal for demotion triggers."""
        report = self.evaluate_signal(signal_id, as_of)
        if report.current_status == 'SCORING' and report.demotion_flagged:
            logger.warning(
                f"DEMOTION FLAG: {signal_id} — "
                f"{', '.join(report.demotion_reasons)}"
            )
            self._send_alert(
                'WARNING',
                f"DEMOTION FLAG: {signal_id}\n"
                f"Reasons: {', '.join(report.demotion_reasons)}\n"
                f"Rolling DD: {report.rolling_dd_pct:.1%}\n"
                f"Rolling Sharpe: {report.rolling_sharpe:.2f}\n"
                f"CUSUM decay: {report.cusum_detected}"
            )
        return report

    def generate_weekly_report(self,
                               as_of: Optional[date] = None) -> str:
        """
        Generate a formatted weekly report for Telegram.

        Returns a markdown-formatted string with a table of all BN signals.
        """
        as_of = as_of or date.today()
        reports = self.evaluate_all_bn_signals(as_of)

        lines = [
            f"*BankNifty Signal Report — {as_of}*",
            "",
            "```",
            f"{'Signal':<22s} {'St':>3s} {'Trd':>4s} {'Shrp':>5s} "
            f"{'WR':>5s} {'PF':>5s} {'DD':>5s} {'W/L':>4s} "
            f"{'P&L':>8s} {'Act'}",
            "-" * 80,
        ]

        promote_candidates = []
        demote_candidates = []

        for signal_id in BN_SIGNAL_IDS:
            report = reports.get(signal_id)
            if report is None:
                continue
            lines.append(report.summary_line())
            if report.promotion_eligible:
                promote_candidates.append(signal_id)
            if report.demotion_flagged:
                demote_candidates.append(signal_id)

        lines.append("```")

        # Total BN shadow P&L
        total_pnl = sum(
            r.total_pnl for r in reports.values() if r.total_pnl
        )
        total_trades = sum(r.trade_count for r in reports.values())
        lines.append(f"\nTotal BN trades: {total_trades} | P&L: {total_pnl:+,.0f}")

        if promote_candidates:
            lines.append(
                f"\nREADY TO PROMOTE: {', '.join(promote_candidates)}"
            )
            lines.append("Use: `--promote SIGNAL_ID` to confirm")

        if demote_candidates:
            lines.append(
                f"\nDEMOTION FLAGS: {', '.join(demote_candidates)}"
            )
            lines.append("Use: `--demote SIGNAL_ID` to confirm")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    # METRIC COMPUTATIONS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_sharpe(pnl_list: List[float]) -> float:
        """Annualized Sharpe ratio from daily P&L."""
        if len(pnl_list) < 5:
            return 0.0
        arr = np.array(pnl_list)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std <= 0:
            return 0.0
        # Annualize: sqrt(252) for daily
        return float((mean / std) * np.sqrt(252))

    @staticmethod
    def _compute_win_rate(pnl_list: List[float]) -> float:
        if not pnl_list:
            return 0.0
        wins = sum(1 for p in pnl_list if p > 0)
        return wins / len(pnl_list)

    @staticmethod
    def _compute_profit_factor(pnl_list: List[float]) -> float:
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        if gross_loss <= 0:
            return 99.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _compute_max_drawdown(pnl_list: List[float]) -> float:
        """Max drawdown as fraction of peak equity."""
        if not pnl_list:
            return 0.0
        equity_curve = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(equity_curve)
        peak = np.where(peak <= 0, 1.0, peak)  # avoid div by zero
        dd = (peak - equity_curve) / np.abs(peak)
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    @staticmethod
    def _compute_wl_ratio(pnl_list: List[float]) -> float:
        """Average win / average loss ratio."""
        wins = [p for p in pnl_list if p > 0]
        losses = [abs(p) for p in pnl_list if p < 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 1.0
        if avg_loss <= 0:
            return 99.0 if avg_win > 0 else 0.0
        return float(avg_win / avg_loss)

    @staticmethod
    def _compute_max_consecutive_losses(pnl_list: List[float]) -> int:
        max_streak = 0
        current_streak = 0
        for p in pnl_list:
            if p < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _compute_nifty_correlation(self, trades: List[Dict],
                                   as_of: date) -> float:
        """
        Compute correlation between BN signal daily P&L and Nifty daily returns.

        Uses Nifty close-to-close returns from nifty_daily table.
        """
        if len(trades) < 10:
            return 0.5  # neutral default if insufficient data

        # Aggregate BN P&L by date
        bn_pnl_by_date = {}
        for t in trades:
            d = t['entry_date']
            bn_pnl_by_date[d] = bn_pnl_by_date.get(d, 0.0) + (t['pnl'] or 0.0)

        dates = sorted(bn_pnl_by_date.keys())
        if len(dates) < 10:
            return 0.5

        # Load Nifty returns for same dates
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT date, close FROM nifty_daily
                WHERE date >= %s AND date <= %s
                ORDER BY date
            """, (dates[0], dates[-1]))
            rows = cur.fetchall()
        except Exception as e:
            logger.warning(f"Nifty data load failed for correlation: {e}")
            self.conn.rollback()
            return 0.5

        if len(rows) < 10:
            return 0.5

        nifty_returns = {}
        for i in range(1, len(rows)):
            d = rows[i][0]
            prev_close = float(rows[i - 1][1])
            curr_close = float(rows[i][1])
            if prev_close > 0:
                nifty_returns[d] = (curr_close - prev_close) / prev_close

        # Align series
        common_dates = sorted(set(bn_pnl_by_date.keys()) & set(nifty_returns.keys()))
        if len(common_dates) < 10:
            return 0.5

        bn_arr = np.array([bn_pnl_by_date[d] for d in common_dates])
        nifty_arr = np.array([nifty_returns[d] for d in common_dates])

        if np.std(bn_arr) == 0 or np.std(nifty_arr) == 0:
            return 0.0

        corr = float(np.corrcoef(bn_arr, nifty_arr)[0, 1])
        return corr

    def _check_cusum_decay(self, pnl_list: List[float]) -> bool:
        """Run CUSUM detector on P&L series to detect mean shift (decay)."""
        if len(pnl_list) < 20:
            return False

        detector = CUSUMDetector()
        detector.calibrate(pnl_list[:20])

        detected = False
        for pnl in pnl_list[20:]:
            cp, _ = detector.update(pnl, expected_mean=np.mean(pnl_list[:20]))
            if cp:
                detected = True

        return detected

    # ══════════════════════════════════════════════════════════
    # CRITERIA EVALUATION
    # ══════════════════════════════════════════════════════════

    def _check_promotion(self, report: PromotionReport) -> bool:
        """Check if report meets all promotion criteria."""
        c = PROMOTION_CRITERIA
        reasons = []

        if report.trading_days < c['min_days']:
            reasons.append(f"days={report.trading_days}<{c['min_days']}")
        if report.trade_count < c['min_trades']:
            reasons.append(f"trades={report.trade_count}<{c['min_trades']}")
        if report.sharpe < c['min_sharpe']:
            reasons.append(f"sharpe={report.sharpe:.2f}<{c['min_sharpe']}")
        if report.win_rate < c['min_win_rate']:
            reasons.append(f"WR={report.win_rate:.1%}<{c['min_win_rate']:.0%}")
        if report.profit_factor < c['min_profit_factor']:
            reasons.append(f"PF={report.profit_factor:.2f}<{c['min_profit_factor']}")
        if report.max_drawdown_pct > c['max_drawdown_pct']:
            reasons.append(f"DD={report.max_drawdown_pct:.1%}>{c['max_drawdown_pct']:.0%}")
        if report.wl_ratio < c['min_wl_ratio']:
            reasons.append(f"W/L={report.wl_ratio:.2f}<{c['min_wl_ratio']}")
        if report.consecutive_losses_max > c['max_consecutive_losses']:
            reasons.append(f"consec_loss={report.consecutive_losses_max}>{c['max_consecutive_losses']}")

        corr = report.nifty_correlation
        if corr < c['min_nifty_corr'] or corr > c['max_nifty_corr']:
            reasons.append(f"corr={corr:.2f} not in [{c['min_nifty_corr']}-{c['max_nifty_corr']}]")

        report.fail_reasons = reasons
        return len(reasons) == 0

    def _check_demotion_criteria(self, report: PromotionReport) -> bool:
        """Check if a SCORING signal should be flagged for demotion."""
        c = DEMOTION_CRITERIA
        reasons = []

        if report.rolling_dd_pct > c['rolling_dd_threshold']:
            reasons.append(f"rolling_DD={report.rolling_dd_pct:.1%}>{c['rolling_dd_threshold']:.0%}")
        if report.rolling_sharpe < c['rolling_sharpe_floor']:
            reasons.append(f"rolling_sharpe={report.rolling_sharpe:.2f}<{c['rolling_sharpe_floor']}")
        if c['cusum_decay'] and report.cusum_detected:
            reasons.append("CUSUM_decay_detected")

        report.demotion_reasons = reasons
        return len(reasons) > 0

    # ══════════════════════════════════════════════════════════
    # DB HELPERS
    # ══════════════════════════════════════════════════════════

    def _load_trades(self, signal_id: str,
                     as_of: date) -> List[Dict]:
        """Load completed SHADOW trades for a BN signal."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT signal_id, direction, entry_date, entry_price,
                       exit_date, exit_price, net_pnl as pnl, exit_reason, instrument
                FROM trades
                WHERE signal_id = %s
                  AND instrument = 'BANKNIFTY'
                  AND trade_type IN ('SHADOW', 'PAPER')
                  AND entry_date <= %s
                  AND exit_date IS NOT NULL
                ORDER BY entry_date, entry_time
            """, (signal_id, as_of))
            rows = cur.fetchall()
            return [
                {
                    'signal_id': r[0],
                    'direction': r[1],
                    'entry_date': r[2],
                    'entry_price': float(r[3]) if r[3] else 0,
                    'exit_date': r[4],
                    'exit_price': float(r[5]) if r[5] else 0,
                    'pnl': float(r[6]) if r[6] is not None else None,
                    'exit_reason': r[7],
                    'instrument': r[8],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Trade load failed for {signal_id}: {e}")
            self.conn.rollback()
            return []

    def _get_signal_status(self, signal_id: str) -> str:
        """Get current signal status from DB. Default: SHADOW."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT status FROM signal_status
                WHERE signal_id = %s
                ORDER BY updated_at DESC LIMIT 1
            """, (signal_id,))
            row = cur.fetchone()
            return row[0] if row else 'SHADOW'
        except Exception:
            self.conn.rollback()
            return 'SHADOW'

    def _send_alert(self, level: str, message: str):
        """Best-effort Telegram alert."""
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
