"""
Signal Decay Detector — unified decay analysis combining BOCD, CUSUM,
rolling metrics, and regime-adjusted thresholds.

Produces a DecayReport for each signal with severity classification:
  HEALTHY / WATCH / WARNING / CRITICAL / STRUCTURAL_DECAY / IMPROVING

Uses existing models:
  - models.decay_bocd.BOCDDetector
  - models.decay_cusum.CUSUMDetector
  - models.decay_regime_adjusted.RegimeAdjustedDecay

Usage:
    from models.signal_decay_detector import SignalDecayDetector
    detector = SignalDecayDetector()
    report = detector.analyze_signal('KAUFMAN_DRY_20', lookback_trades=50)
    all_reports = detector.analyze_all_signals()
    actions = detector.get_recommended_actions()
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from config.settings import DATABASE_DSN
from models.decay_bocd import BOCDDetector
from models.decay_cusum import CUSUMDetector
from models.decay_regime_adjusted import RegimeAdjustedDecay, SIGNAL_TARGET_REGIME

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# REGIME-ADJUSTED THRESHOLDS
# ════════════════════════════════════════════════════════════

REGIME_THRESHOLDS = {
    'CALM': {
        'sharpe_floor': 0.8,
        'dd_ceiling': 0.15,
        'win_rate_floor': 0.40,
        'consec_loss_max': 5,
        'bocd_cp_threshold': 0.25,
    },
    'NORMAL': {
        'sharpe_floor': 0.5,
        'dd_ceiling': 0.20,
        'win_rate_floor': 0.35,
        'consec_loss_max': 6,
        'bocd_cp_threshold': 0.30,
    },
    'HIGH_VOL': {
        'sharpe_floor': 0.3,
        'dd_ceiling': 0.30,
        'win_rate_floor': 0.30,
        'consec_loss_max': 8,
        'bocd_cp_threshold': 0.40,
    },
    'CRISIS': {
        'sharpe_floor': 0.0,
        'dd_ceiling': 0.40,
        'win_rate_floor': 0.25,
        'consec_loss_max': 10,
        'bocd_cp_threshold': 0.50,
    },
}

# VIX-based regime mapping
VIX_REGIME_MAP = [
    (13.0, 'CALM'),
    (18.0, 'NORMAL'),
    (25.0, 'HIGH_VOL'),
    (999.0, 'CRISIS'),
]

# Status severity ordering
STATUS_SEVERITY = {
    'HEALTHY': 0,
    'IMPROVING': 0,
    'WATCH': 1,
    'WARNING': 2,
    'CRITICAL': 3,
    'STRUCTURAL_DECAY': 4,
}


@dataclass
class DecayReport:
    """Result of analyzing a single signal for decay."""

    signal_id: str
    analysis_date: date
    trade_count: int = 0

    # Rolling 20-trade metrics
    sharpe_20: float = 0.0
    win_rate_20: float = 0.0
    profit_factor_20: float = 0.0
    max_drawdown_20: float = 0.0
    consecutive_losses: int = 0

    # BOCD results
    bocd_cp_prob: float = 0.0
    bocd_run_length: int = 0

    # CUSUM results
    cusum_alert: bool = False
    cusum_value: float = 0.0

    # Regime context
    regime: str = 'NORMAL'
    regime_sharpe: Optional[float] = None
    regime_decaying: bool = False
    regime_decay_reason: str = ''

    # Classification
    status: str = 'HEALTHY'
    severity_score: int = 0
    flags: List[str] = field(default_factory=list)

    # For shadow signals
    is_shadow: bool = False
    improving: bool = False

    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'analysis_date': str(self.analysis_date),
            'trade_count': self.trade_count,
            'sharpe_20': round(self.sharpe_20, 2),
            'win_rate_20': round(self.win_rate_20, 3),
            'profit_factor_20': round(self.profit_factor_20, 2),
            'max_drawdown_20': round(self.max_drawdown_20, 3),
            'consecutive_losses': self.consecutive_losses,
            'bocd_cp_prob': round(self.bocd_cp_prob, 3),
            'bocd_run_length': self.bocd_run_length,
            'cusum_alert': self.cusum_alert,
            'cusum_value': round(self.cusum_value, 2),
            'regime': self.regime,
            'regime_sharpe': round(self.regime_sharpe, 2) if self.regime_sharpe is not None else None,
            'regime_decaying': self.regime_decaying,
            'status': self.status,
            'severity_score': self.severity_score,
            'flags': self.flags,
            'is_shadow': self.is_shadow,
            'improving': self.improving,
        }

    def summary_line(self) -> str:
        status_icons = {
            'HEALTHY': 'OK', 'WATCH': 'WCH', 'WARNING': 'WRN',
            'CRITICAL': 'CRT', 'STRUCTURAL_DECAY': 'DEC', 'IMPROVING': 'IMP',
        }
        icon = status_icons.get(self.status, '???')
        shadow = 'SHD' if self.is_shadow else 'SCR'
        return (
            f"{self.signal_id:<28s} {shadow:>3s} {icon:>3s} "
            f"{self.trade_count:>3d}t "
            f"S={self.sharpe_20:>5.2f} WR={self.win_rate_20:>5.1%} "
            f"DD={self.max_drawdown_20:>5.1%} "
            f"CL={self.consecutive_losses:>2d} "
            f"P(cp)={self.bocd_cp_prob:>5.3f} "
            f"CUSUM={'Y' if self.cusum_alert else 'N'} "
            f"flg={self.severity_score}"
        )


class SignalDecayDetector:
    """
    Unified signal decay detection engine.

    Combines BOCD (structural changepoints), CUSUM (persistent mean shifts),
    rolling metrics, and regime-adjusted thresholds to classify signal health.
    """

    def __init__(self, conn=None):
        self._external_conn = conn is not None
        self.conn = conn or psycopg2.connect(DATABASE_DSN)
        self._reports: Dict[str, DecayReport] = {}
        self._current_regime: str = 'NORMAL'
        self._current_vix: float = 15.0
        self._load_current_regime()

    def close(self):
        if not self._external_conn and self.conn and not self.conn.closed:
            self.conn.close()

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def analyze_signal(self, signal_id: str,
                       lookback_trades: int = 50,
                       as_of: Optional[date] = None) -> DecayReport:
        """
        Full decay analysis for a single signal.

        Args:
            signal_id: signal to analyze
            lookback_trades: number of recent trades to analyze
            as_of: analysis date (default today)

        Returns:
            DecayReport with status, metrics, and flags
        """
        as_of = as_of or date.today()
        report = DecayReport(
            signal_id=signal_id,
            analysis_date=as_of,
            regime=self._current_regime,
        )

        # Determine if shadow signal
        report.is_shadow = self._is_shadow_signal(signal_id)

        # Load trades
        trades = self._load_recent_trades(signal_id, lookback_trades)
        if not trades:
            report.flags.append('NO_TRADES')
            self._reports[signal_id] = report
            return report

        pnl_list = [t['net_pnl'] for t in trades if t['net_pnl'] is not None]
        if len(pnl_list) < 10:
            report.trade_count = len(pnl_list)
            report.flags.append('INSUFFICIENT_DATA')
            self._reports[signal_id] = report
            return report

        report.trade_count = len(pnl_list)

        # ── Rolling 20-trade metrics ──
        recent = pnl_list[-20:] if len(pnl_list) >= 20 else pnl_list
        report.sharpe_20 = self._compute_sharpe(recent)
        report.win_rate_20 = self._compute_win_rate(recent)
        report.profit_factor_20 = self._compute_profit_factor(recent)
        report.max_drawdown_20 = self._compute_max_drawdown(recent)
        report.consecutive_losses = self._compute_max_consecutive_losses(pnl_list)

        # ── BOCD analysis ──
        bocd = BOCDDetector(hazard_rate=1 / 50, mu_prior=0.0, sigma_prior=1.0)
        if len(pnl_list) >= 10:
            # Normalize to z-scores
            pnl_arr = np.array(pnl_list)
            mu = np.mean(pnl_arr[:10])
            sigma = max(np.std(pnl_arr[:10]), 1e-6)
            for p in pnl_list:
                z = (p - mu) / sigma
                bocd.update(z)
            report.bocd_cp_prob = bocd.get_recent_max_cp_prob(window=5)
            report.bocd_run_length = bocd.get_most_likely_run_length()

        # ── CUSUM analysis ──
        cusum = CUSUMDetector()
        if len(pnl_list) >= 20:
            cusum.calibrate(pnl_list[:20])
            expected_mean = np.mean(pnl_list[:20])
            cusum_detected = False
            last_cusum_val = 0.0
            for p in pnl_list[20:]:
                detected, val = cusum.update(p, expected_mean=expected_mean)
                if detected:
                    cusum_detected = True
                last_cusum_val = val
            report.cusum_alert = cusum_detected
            report.cusum_value = last_cusum_val

        # ── Regime-adjusted analysis ──
        regime_tracker = RegimeAdjustedDecay(signal_id)
        regime_map = self._load_trade_regimes(signal_id, lookback_trades)
        for i, t in enumerate(trades):
            pnl = t['net_pnl']
            if pnl is None:
                continue
            regime = regime_map.get(t.get('trade_id'), 'NORMAL')
            regime_tracker.update(pnl, regime)
        report.regime_sharpe = regime_tracker.regime_sharpe()
        is_decaying, reason = regime_tracker.is_decaying()
        report.regime_decaying = is_decaying
        report.regime_decay_reason = reason

        # ── Classify severity ──
        self._classify(report)

        # ── Shadow improving detection ──
        if report.is_shadow:
            report.improving = self._check_improving(report, pnl_list)
            if report.improving:
                report.status = 'IMPROVING'
                report.flags.append('SHADOW_IMPROVING')

        self._reports[signal_id] = report
        return report

    def analyze_all_signals(self,
                            lookback_trades: int = 50,
                            as_of: Optional[date] = None) -> Dict[str, DecayReport]:
        """Analyze all scoring + shadow signals."""
        as_of = as_of or date.today()
        signal_ids = self._get_all_signal_ids()
        reports = {}
        for sid in signal_ids:
            try:
                reports[sid] = self.analyze_signal(sid, lookback_trades, as_of)
            except Exception as e:
                logger.error(f"Decay analysis failed for {sid}: {e}")
                reports[sid] = DecayReport(
                    signal_id=sid,
                    analysis_date=as_of,
                    status='HEALTHY',
                    flags=[f'ANALYSIS_ERROR: {e}'],
                )
        self._reports = reports
        return reports

    def get_recommended_actions(self) -> List[Dict]:
        """
        Based on latest analysis, return recommended actions.

        Returns list of dicts:
            {signal_id, status, action, factor, reason}
        """
        actions = []
        for sid, report in self._reports.items():
            action = self._determine_action(report)
            if action:
                actions.append(action)
        # Sort: most severe first
        actions.sort(
            key=lambda a: STATUS_SEVERITY.get(a.get('status', 'HEALTHY'), 0),
            reverse=True,
        )
        return actions

    # ════════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ════════════════════════════════════════════════════════════

    def _classify(self, report: DecayReport):
        """
        Classify signal health based on flag count and detector agreement.

        Severity mapping:
          0 flags → HEALTHY
          1 flag  → WATCH
          2 flags → WARNING
          3+ flags → CRITICAL
          BOCD + CUSUM both fire → STRUCTURAL_DECAY
        """
        thresholds = REGIME_THRESHOLDS.get(report.regime, REGIME_THRESHOLDS['NORMAL'])
        flags = []

        # Sharpe floor
        if report.sharpe_20 < thresholds['sharpe_floor']:
            flags.append(f"LOW_SHARPE({report.sharpe_20:.2f}<{thresholds['sharpe_floor']})")

        # Drawdown ceiling
        if report.max_drawdown_20 > thresholds['dd_ceiling']:
            flags.append(f"HIGH_DD({report.max_drawdown_20:.1%}>{thresholds['dd_ceiling']:.0%})")

        # Win rate floor
        if report.win_rate_20 < thresholds['win_rate_floor']:
            flags.append(f"LOW_WR({report.win_rate_20:.1%}<{thresholds['win_rate_floor']:.0%})")

        # Consecutive losses
        if report.consecutive_losses > thresholds['consec_loss_max']:
            flags.append(f"LOSS_STREAK({report.consecutive_losses}>{thresholds['consec_loss_max']})")

        # BOCD changepoint
        if report.bocd_cp_prob > thresholds['bocd_cp_threshold']:
            flags.append(f"BOCD_CP({report.bocd_cp_prob:.3f}>{thresholds['bocd_cp_threshold']})")

        # CUSUM mean shift
        if report.cusum_alert:
            flags.append('CUSUM_SHIFT')

        # Regime decay
        if report.regime_decaying:
            flags.append(f"REGIME_DECAY({report.regime_decay_reason})")

        report.flags = flags
        report.severity_score = len(flags)

        # Classify status
        if len(flags) == 0:
            report.status = 'HEALTHY'
        elif len(flags) == 1:
            report.status = 'WATCH'
        elif len(flags) == 2:
            report.status = 'WARNING'
        else:
            report.status = 'CRITICAL'

        # Upgrade to STRUCTURAL_DECAY if both BOCD and CUSUM fire
        bocd_fired = report.bocd_cp_prob > thresholds['bocd_cp_threshold']
        if bocd_fired and report.cusum_alert:
            report.status = 'STRUCTURAL_DECAY'

    def _check_improving(self, report: DecayReport,
                         pnl_list: List[float]) -> bool:
        """
        Check if a shadow signal is improving (candidate for promotion).

        Criteria:
          - Positive CUSUM trend (no negative mean shift)
          - Rolling Sharpe > 0.8
          - Win rate > 40%
          - At least 15 trades
        """
        if report.trade_count < 15:
            return False
        if report.cusum_alert:
            return False  # negative shift detected
        if report.sharpe_20 < 0.8:
            return False
        if report.win_rate_20 < 0.40:
            return False

        # Check recent PnL trend is positive
        recent = pnl_list[-10:] if len(pnl_list) >= 10 else pnl_list
        if np.mean(recent) <= 0:
            return False

        return True

    def _determine_action(self, report: DecayReport) -> Optional[Dict]:
        """Determine recommended action for a signal based on its report."""
        if report.status == 'HEALTHY':
            return None  # no action needed (clearing handled by manager)

        action = {
            'signal_id': report.signal_id,
            'status': report.status,
            'action': 'NONE',
            'factor': 1.0,
            'reason': ', '.join(report.flags[:3]),
            'is_shadow': report.is_shadow,
        }

        if report.status == 'IMPROVING' and report.is_shadow:
            action['action'] = 'RECOMMEND_PROMOTE'
            action['reason'] = (
                f"Shadow improving: Sharpe={report.sharpe_20:.2f} "
                f"WR={report.win_rate_20:.1%}"
            )
            return action

        if report.status == 'WARNING':
            action['action'] = 'SIZE_REDUCE'
            action['factor'] = 0.5
        elif report.status == 'CRITICAL':
            action['action'] = 'SIZE_REDUCE'
            action['factor'] = 0.25
        elif report.status == 'STRUCTURAL_DECAY':
            action['action'] = 'CONSIDER_DEMOTE'
            action['factor'] = 0.0
        elif report.status == 'WATCH':
            action['action'] = 'MONITOR'
            action['factor'] = 0.75

        return action

    # ════════════════════════════════════════════════════════════
    # METRIC COMPUTATIONS
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_sharpe(pnl_list: List[float]) -> float:
        if len(pnl_list) < 5:
            return 0.0
        arr = np.array(pnl_list)
        std = np.std(arr, ddof=1)
        if std <= 0:
            return 0.0
        return float((np.mean(arr) / std) * np.sqrt(252))

    @staticmethod
    def _compute_win_rate(pnl_list: List[float]) -> float:
        if not pnl_list:
            return 0.0
        return sum(1 for p in pnl_list if p > 0) / len(pnl_list)

    @staticmethod
    def _compute_profit_factor(pnl_list: List[float]) -> float:
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        if gross_loss <= 0:
            return 99.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _compute_max_drawdown(pnl_list: List[float]) -> float:
        if not pnl_list:
            return 0.0
        equity = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(equity)
        peak = np.where(peak <= 0, 1.0, peak)
        dd = (peak - equity) / np.abs(peak)
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    @staticmethod
    def _compute_max_consecutive_losses(pnl_list: List[float]) -> int:
        max_streak = 0
        current = 0
        for p in pnl_list:
            if p < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    # ════════════════════════════════════════════════════════════
    # DB HELPERS
    # ════════════════════════════════════════════════════════════

    def _load_current_regime(self):
        """Load current VIX to determine market regime."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT india_vix FROM nifty_daily
                WHERE india_vix IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row and row[0]:
                self._current_vix = float(row[0])
                for threshold, regime in VIX_REGIME_MAP:
                    if self._current_vix < threshold:
                        self._current_regime = regime
                        break
        except Exception as e:
            logger.warning(f"Failed to load VIX for regime: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass

    def _load_recent_trades(self, signal_id: str,
                            lookback: int) -> List[Dict]:
        """Load recent completed trades for a signal."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_id, signal_id, direction, entry_date,
                       entry_price, exit_date, exit_price,
                       net_pnl, exit_reason, entry_regime, trade_type
                FROM trades
                WHERE signal_id = %s
                  AND exit_date IS NOT NULL
                  AND net_pnl IS NOT NULL
                ORDER BY exit_date DESC, exit_time DESC
                LIMIT %s
            """, (signal_id, lookback))
            rows = cur.fetchall()
            # Return in chronological order (oldest first)
            trades = [
                {
                    'trade_id': r[0],
                    'signal_id': r[1],
                    'direction': r[2],
                    'entry_date': r[3],
                    'entry_price': float(r[4]) if r[4] else 0,
                    'exit_date': r[5],
                    'exit_price': float(r[6]) if r[6] else 0,
                    'net_pnl': float(r[7]) if r[7] is not None else None,
                    'exit_reason': r[8],
                    'entry_regime': r[9],
                    'trade_type': r[10],
                }
                for r in rows
            ]
            trades.reverse()
            return trades
        except Exception as e:
            logger.error(f"Trade load failed for {signal_id}: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def _load_trade_regimes(self, signal_id: str,
                            lookback: int) -> Dict[int, str]:
        """Map trade_id -> entry_regime for regime-adjusted analysis."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_id, COALESCE(entry_regime, 'NORMAL')
                FROM trades
                WHERE signal_id = %s
                  AND exit_date IS NOT NULL
                ORDER BY exit_date DESC
                LIMIT %s
            """, (signal_id, lookback))
            return {r[0]: r[1] for r in cur.fetchall()}
        except Exception as e:
            logger.debug(f"Regime load failed for {signal_id}: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return {}

    def _is_shadow_signal(self, signal_id: str) -> bool:
        """Check if signal is SHADOW by looking at its trades."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_type FROM trades
                WHERE signal_id = %s
                ORDER BY entry_date DESC
                LIMIT 1
            """, (signal_id,))
            row = cur.fetchone()
            if row:
                return row[0] == 'SHADOW'
            return False
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return False

    def _get_all_signal_ids(self) -> List[str]:
        """Get all signal IDs that have trades."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT DISTINCT signal_id
                FROM trades
                WHERE exit_date IS NOT NULL
                  AND net_pnl IS NOT NULL
                ORDER BY signal_id
            """)
            return [r[0] for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get signal IDs: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []
