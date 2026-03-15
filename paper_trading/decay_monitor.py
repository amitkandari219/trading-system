"""
Signal Decay Monitor: tracks whether each signal is decaying
relative to its backtest expectations. Adjusts position size automatically.
"""

import logging
from typing import List, Dict

import numpy as np
import psycopg2

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

DECAY_THRESHOLDS = {
    'win_rate_drop':    {'YELLOW': 0.15, 'RED': 0.25, 'CRITICAL': 0.35},
    'sharpe_drop':      {'YELLOW': 0.30, 'RED': 0.50, 'CRITICAL': 0.70},
    'pf_drop':          {'YELLOW': 0.25, 'RED': 0.40, 'CRITICAL': 0.55},
    'consecutive_loss': {'YELLOW': 4,    'RED': 6,    'CRITICAL': 8},
}

SIZE_MULTIPLIERS = {
    'GREEN': 1.00,
    'YELLOW': 0.75,
    'RED': 0.40,
    'CRITICAL': 0.00,
}

BACKTEST_METRICS = {
    'KAUFMAN_DRY_20': {'win_rate': 0.47, 'sharpe': 2.37, 'profit_factor': 1.91},
    'KAUFMAN_DRY_16': {'win_rate': 0.49, 'sharpe': 1.65, 'profit_factor': 1.39},
    'KAUFMAN_DRY_12': {'win_rate': 0.39, 'sharpe': 1.78, 'profit_factor': 1.42},
    'SCORING_SYSTEM': {'win_rate': 0.48, 'sharpe': 2.30, 'profit_factor': 1.55},
}


class DecayMonitor:

    def __init__(self, lookback_trades: int = 30, db_conn=None):
        self.lookback = lookback_trades
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)

    def get_signal_health(self, signal_id: str,
                          backtest_metrics: dict = None) -> dict:
        """Compare rolling live metrics to backtest expectations."""
        if backtest_metrics is None:
            backtest_metrics = BACKTEST_METRICS.get(signal_id, {
                'win_rate': 0.45, 'sharpe': 1.5, 'profit_factor': 1.3
            })

        recent = self._get_recent_trades(signal_id)

        if len(recent) < 10:
            return {
                'signal_id': signal_id,
                'status': 'INSUFFICIENT_DATA',
                'size_multiplier': 1.0,
                'trades_in_window': len(recent),
                'alert_message': None,
                'recommendation': 'Continue trading — insufficient data for assessment',
            }

        pnls = [t['net_pnl'] for t in recent]
        live_wr = sum(1 for p in pnls if p > 0) / len(pnls)
        live_pf = self._calc_profit_factor(pnls)
        live_sr = self._calc_sharpe(pnls)
        consec = self._count_consecutive_losses(pnls)

        wr_drop = backtest_metrics['win_rate'] - live_wr
        sr_drop = backtest_metrics['sharpe'] - live_sr
        pf_drop = backtest_metrics['profit_factor'] - live_pf

        status = self._determine_status(wr_drop, sr_drop, pf_drop, consec)

        return {
            'signal_id': signal_id,
            'status': status,
            'size_multiplier': SIZE_MULTIPLIERS[status],
            'live_win_rate': round(live_wr, 3),
            'backtest_win_rate': backtest_metrics['win_rate'],
            'win_rate_drop': round(wr_drop, 3),
            'live_sharpe': round(live_sr, 2),
            'backtest_sharpe': backtest_metrics['sharpe'],
            'sharpe_drop': round(sr_drop, 2),
            'live_pf': round(live_pf, 2),
            'backtest_pf': backtest_metrics['profit_factor'],
            'pf_drop': round(pf_drop, 2),
            'consecutive_losses': consec,
            'trades_in_window': len(recent),
            'alert_message': self._build_alert(status, signal_id, wr_drop, consec),
            'recommendation': self._recommend(status),
        }

    def get_portfolio_health(self) -> dict:
        """Run health check on all confirmed signals."""
        results = {}
        worst_status = 'GREEN'
        status_order = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'CRITICAL': 3, 'INSUFFICIENT_DATA': -1}

        for signal_id in BACKTEST_METRICS:
            health = self.get_signal_health(signal_id)
            results[signal_id] = health
            if status_order.get(health['status'], -1) > status_order.get(worst_status, 0):
                worst_status = health['status']

        portfolio_status = worst_status
        if sum(1 for h in results.values() if h['status'] in ('RED', 'CRITICAL')) >= 2:
            portfolio_status = 'CRITICAL'

        return {
            'portfolio_status': portfolio_status,
            'signals': results,
            'recommendation': self._portfolio_recommendation(results),
        }

    def _get_recent_trades(self, signal_id: str) -> list:
        """Get last N completed trades for a signal."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT net_pnl, gross_pnl, entry_price, exit_price, direction
                FROM trades
                WHERE signal_id = %s
                  AND exit_date IS NOT NULL
                  AND trade_type IN ('PAPER', 'PAPER_CONTROL', 'PAPER_SCORING')
                ORDER BY exit_date DESC
                LIMIT %s
            """, (signal_id, self.lookback))
            return [{'net_pnl': r[0] or 0, 'gross_pnl': r[1] or 0,
                     'entry_price': r[2], 'exit_price': r[3],
                     'direction': r[4]} for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    @staticmethod
    def _calc_profit_factor(pnls):
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        return wins / losses if losses > 0 else 99

    @staticmethod
    def _calc_sharpe(pnls):
        if len(pnls) < 2:
            return 0
        ret = np.array(pnls)
        std = ret.std()
        return (ret.mean() / std * np.sqrt(252)) if std > 0 else 0

    @staticmethod
    def _count_consecutive_losses(pnls):
        streak = 0
        max_streak = 0
        for p in pnls:
            if p <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def _determine_status(wr_drop, sr_drop, pf_drop, consec):
        for level in ['CRITICAL', 'RED', 'YELLOW']:
            if (wr_drop >= DECAY_THRESHOLDS['win_rate_drop'][level] or
                sr_drop >= DECAY_THRESHOLDS['sharpe_drop'][level] or
                pf_drop >= DECAY_THRESHOLDS['pf_drop'][level] or
                consec >= DECAY_THRESHOLDS['consecutive_loss'][level]):
                return level
        return 'GREEN'

    @staticmethod
    def _build_alert(status, signal_id, wr_drop=0, consec=0):
        if status == 'GREEN':
            return None
        emoji = {'YELLOW': '🟡', 'RED': '🔴', 'CRITICAL': '🚨'}.get(status, '⚠️')
        msg = f"{emoji} {signal_id}: {status}"
        if wr_drop > 0.1:
            msg += f" (WR dropped {wr_drop:.0%})"
        if consec >= 4:
            msg += f" ({consec} consecutive losses)"
        return msg

    @staticmethod
    def _recommend(status):
        return {
            'GREEN': 'Continue trading at full size',
            'YELLOW': 'Reduce position size to 75%. Monitor closely.',
            'RED': 'Reduce to 40%. Review signal logic.',
            'CRITICAL': 'SUSPEND trading. Signal may be broken.',
            'INSUFFICIENT_DATA': 'Continue — not enough trades to assess',
        }.get(status, 'Unknown')

    @staticmethod
    def _portfolio_recommendation(results):
        reds = sum(1 for h in results.values() if h['status'] in ('RED', 'CRITICAL'))
        if reds >= 2:
            return 'HALT: Multiple signals degrading. Review regime conditions.'
        if reds == 1:
            return 'CAUTION: One signal degrading. Reduce its allocation.'
        yellows = sum(1 for h in results.values() if h['status'] == 'YELLOW')
        if yellows >= 2:
            return 'WATCH: Multiple signals showing early decay.'
        return 'HEALTHY: All signals performing within expectations.'
