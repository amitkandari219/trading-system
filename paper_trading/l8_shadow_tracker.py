"""
L8 Options Shadow Tracker — track theoretical P&L for options strategies.

Monitors L8 strategies that passed (or failed) walk-forward validation.
SHADOW strategies accumulate theoretical trades; once a signal hits
30 trades + Sharpe >= 1.0 it becomes a promotion candidate for SCORING.

Usage:
    from paper_trading.l8_shadow_tracker import L8ShadowTracker
    tracker = L8ShadowTracker()
    summary = tracker.update_daily(spot, vix, iv_rank, regime, dte_near, as_of)
    candidates = tracker.get_promotion_candidates()
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, RISK_FREE_RATE
from data.options_loader import bs_price
from signals.l8_signals import L8_SIGNALS, OptionsSignalComputer

logger = logging.getLogger(__name__)

# ================================================================
# PROMOTION CRITERIA
# ================================================================

PROMOTION_MIN_TRADES = 30
PROMOTION_MIN_SHARPE = 1.0

# ================================================================
# WALK-FORWARD RESULTS (updated by l8_walk_forward.py)
# ================================================================
# Maps signal_id -> {'verdict': 'PASS'|'FAIL', 'sharpe': float, 'wf_pct': float}
# PASS -> SHADOW status (actively track), FAIL -> MONITOR (log only)

WF_RESULTS: Dict[str, dict] = {
    'L8_SHORT_STRANGLE_EXPIRY':  {'verdict': 'PASS', 'sharpe': 1.35, 'wf_pct': 0.83},
    'L8_IRON_CONDOR_WEEKLY':     {'verdict': 'PASS', 'sharpe': 1.22, 'wf_pct': 0.80},
    'L8_SHORT_STRADDLE_HIGH_IV': {'verdict': 'PASS', 'sharpe': 1.10, 'wf_pct': 0.78},
    'L8_BULL_PUT_SUPPORT':       {'verdict': 'PASS', 'sharpe': 1.18, 'wf_pct': 0.82},
    'L8_BEAR_CALL_RESISTANCE':   {'verdict': 'PASS', 'sharpe': 1.05, 'wf_pct': 0.76},
    'L8_CALENDAR_LOW_VOL':       {'verdict': 'FAIL', 'sharpe': 0.62, 'wf_pct': 0.58},
    'L8_PROTECTIVE_PUT_CRISIS':  {'verdict': 'FAIL', 'sharpe': 0.41, 'wf_pct': 0.50},
    'L8_COVERED_CALL_RANGING':   {'verdict': 'PASS', 'sharpe': 1.08, 'wf_pct': 0.79},
    'L8_RATIO_MEAN_REVERSION':   {'verdict': 'FAIL', 'sharpe': 0.55, 'wf_pct': 0.55},
    'L8_EXPIRY_DAY_STRANGLE':    {'verdict': 'PASS', 'sharpe': 1.42, 'wf_pct': 0.85},
}


# ================================================================
# SHADOW POSITION
# ================================================================

@dataclass
class ShadowPosition:
    """A theoretical options position tracked without real capital."""

    signal_id: str
    strategy_type: str
    entry_date: date
    legs: List[dict] = field(default_factory=list)
    # Each leg dict: {strike, option_type, action, entry_premium, lots}

    entry_credit: float = 0.0     # net credit for credit strategies
    entry_debit: float = 0.0      # net debit for debit strategies
    status: str = 'OPEN'          # OPEN | CLOSED

    exit_date: Optional[date] = None
    exit_pnl: float = 0.0
    exit_reason: str = ''
    hold_days: int = 0

    def net_premium(self) -> float:
        """Positive = credit received, negative = debit paid."""
        return self.entry_credit - self.entry_debit

    def is_credit_strategy(self) -> bool:
        return self.entry_credit > 0 and self.entry_debit == 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['entry_date'] = str(self.entry_date)
        d['exit_date'] = str(self.exit_date) if self.exit_date else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ShadowPosition':
        d = dict(d)
        d['entry_date'] = date.fromisoformat(d['entry_date'])
        if d.get('exit_date'):
            d['exit_date'] = date.fromisoformat(d['exit_date'])
        return cls(**d)


# ================================================================
# L8 SHADOW TRACKER
# ================================================================

class L8ShadowTracker:
    """
    Shadow tracker for L8 options strategies.

    Tracks theoretical entry/exit prices using Black-Scholes,
    computes rolling performance metrics, and identifies signals
    ready for promotion to SCORING.
    """

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.computer = OptionsSignalComputer()

        # Per-signal tracking state
        self.open_positions: Dict[str, ShadowPosition] = {}
        self.closed_trades: Dict[str, List[ShadowPosition]] = {
            sid: [] for sid in L8_SIGNALS
        }
        self.cumulative_pnl: Dict[str, float] = {sid: 0.0 for sid in L8_SIGNALS}

        # Try to load persisted state
        self._try_load_state()

    # ══════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ══════════════════════════════════════════════════════════

    def check_entries(self, spot: float, vix: float, iv_rank: float,
                      regime: str, dte_near: int) -> List[ShadowPosition]:
        """
        Check for new shadow entries using OptionsSignalComputer.

        Returns list of new ShadowPosition objects created.
        """
        fired = self.computer.check_all(
            spot=spot, vix=vix, iv_rank=iv_rank,
            regime=regime, dte_near=dte_near,
        )

        new_entries = []
        for sig in fired:
            signal_id = sig['signal_id']

            # Skip if already tracking an open position for this signal
            if signal_id in self.open_positions:
                continue

            # Skip MONITOR signals (WF FAIL) — just log
            wf = WF_RESULTS.get(signal_id, {})
            if wf.get('verdict') == 'FAIL':
                logger.debug(f"L8 MONITOR skip: {signal_id} (WF FAIL)")
                continue

            rule = L8_SIGNALS[signal_id]
            sigma = vix / 100.0  # approximate IV from VIX
            T = dte_near / 365.0

            # Build legs with theoretical BS prices
            position_legs = []
            total_credit = 0.0
            total_debit = 0.0

            for leg_def in rule.legs:
                strike = self._resolve_strike(
                    spot, leg_def, sigma, T, dte_near
                )
                premium = bs_price(
                    spot, strike, T, RISK_FREE_RATE, sigma,
                    leg_def.option_type,
                )
                lots = leg_def.lots

                leg_entry = {
                    'strike': strike,
                    'option_type': leg_def.option_type,
                    'action': leg_def.action,
                    'entry_premium': round(premium, 2),
                    'lots': lots,
                }
                position_legs.append(leg_entry)

                notional = premium * NIFTY_LOT_SIZE * lots
                if leg_def.action == 'SELL':
                    total_credit += notional
                else:
                    total_debit += notional

            pos = ShadowPosition(
                signal_id=signal_id,
                strategy_type=sig['strategy_type'],
                entry_date=date.today(),
                legs=position_legs,
                entry_credit=round(total_credit, 2),
                entry_debit=round(total_debit, 2),
                status='OPEN',
            )
            self.open_positions[signal_id] = pos
            new_entries.append(pos)
            logger.info(
                f"L8 SHADOW ENTRY: {signal_id} | "
                f"credit={total_credit:.0f} debit={total_debit:.0f} | "
                f"legs={len(position_legs)}"
            )

        return new_entries

    # ══════════════════════════════════════════════════════════
    # EXIT LOGIC
    # ══════════════════════════════════════════════════════════

    def check_exits(self, spot: float, vix: float,
                    dte_remaining: int) -> List[ShadowPosition]:
        """
        Check open shadow positions for exit conditions.

        Exit triggers:
          - Profit target: 50% of credit (credit strategies)
          - Stop loss: 2x credit
          - DTE exit: close at rule.exit_dte days remaining
          - Max hold days exceeded

        Returns list of closed ShadowPosition objects.
        """
        closed = []
        sigma = vix / 100.0
        to_remove = []

        for signal_id, pos in self.open_positions.items():
            rule = L8_SIGNALS.get(signal_id)
            if rule is None:
                continue

            T = max(dte_remaining / 365.0, 1 / 365.0)
            pos.hold_days += 1

            # Compute current value of all legs
            current_value = 0.0
            for leg in pos.legs:
                premium_now = bs_price(
                    spot, leg['strike'], T, RISK_FREE_RATE, sigma,
                    leg['option_type'],
                )
                notional = premium_now * NIFTY_LOT_SIZE * leg['lots']
                if leg['action'] == 'SELL':
                    current_value -= notional  # cost to close sold leg
                else:
                    current_value += notional  # value of bought leg

            # P&L = (credit received - debit paid) + current_value
            # For credit strategies: entry_credit > 0, current_value is
            #   negative (cost to close). Profit = entry_credit + current_value
            pnl = (pos.entry_credit - pos.entry_debit) + current_value
            exit_reason = ''

            # Profit target
            if pos.is_credit_strategy():
                target = pos.entry_credit * rule.profit_target_pct
                if pnl >= target:
                    exit_reason = 'PROFIT_TARGET'
            else:
                target = pos.entry_debit * rule.profit_target_pct
                if pnl >= target:
                    exit_reason = 'PROFIT_TARGET'

            # Stop loss
            if not exit_reason and pos.is_credit_strategy():
                max_loss = pos.entry_credit * rule.stop_loss_multiple
                if pnl <= -max_loss:
                    exit_reason = 'STOP_LOSS'
            elif not exit_reason:
                if pnl <= -pos.entry_debit:
                    exit_reason = 'STOP_LOSS'

            # DTE exit
            if not exit_reason and dte_remaining <= rule.exit_dte:
                exit_reason = 'DTE_EXIT'

            # Max hold days
            if not exit_reason and pos.hold_days >= rule.max_hold_days:
                exit_reason = 'MAX_HOLD'

            if exit_reason:
                pos.status = 'CLOSED'
                pos.exit_date = date.today()
                pos.exit_pnl = round(pnl, 2)
                pos.exit_reason = exit_reason

                self.closed_trades[signal_id].append(pos)
                self.cumulative_pnl[signal_id] += pnl
                to_remove.append(signal_id)
                closed.append(pos)

                logger.info(
                    f"L8 SHADOW EXIT: {signal_id} | "
                    f"pnl={pnl:+,.0f} | reason={exit_reason} | "
                    f"hold={pos.hold_days}d"
                )

        for sid in to_remove:
            del self.open_positions[sid]

        return closed

    # ══════════════════════════════════════════════════════════
    # DAILY UPDATE
    # ══════════════════════════════════════════════════════════

    def update_daily(self, spot: float, vix: float, iv_rank: float,
                     regime: str, dte_near: int,
                     as_of: Optional[date] = None) -> dict:
        """
        Run daily shadow update: check exits, then entries.

        Returns a summary dict with counts and P&L.
        """
        as_of = as_of or date.today()

        exits = self.check_exits(spot, vix, dte_near)
        entries = self.check_entries(spot, vix, iv_rank, regime, dte_near)

        # Persist after each daily run
        self._try_persist_state()

        exit_pnl = sum(p.exit_pnl for p in exits)
        total_open = len(self.open_positions)
        total_closed = sum(len(v) for v in self.closed_trades.values())

        summary = {
            'date': str(as_of),
            'new_entries': len(entries),
            'new_exits': len(exits),
            'exit_pnl_today': round(exit_pnl, 2),
            'open_positions': total_open,
            'total_closed_trades': total_closed,
            'cumulative_pnl': {
                sid: round(pnl, 2)
                for sid, pnl in self.cumulative_pnl.items()
                if pnl != 0.0
            },
            'entries': [e.signal_id for e in entries],
            'exits': [
                {'signal_id': x.signal_id, 'pnl': x.exit_pnl,
                 'reason': x.exit_reason}
                for x in exits
            ],
        }
        return summary

    # ══════════════════════════════════════════════════════════
    # STATS & PROMOTION
    # ══════════════════════════════════════════════════════════

    def get_signal_stats(self, signal_id: str) -> dict:
        """
        Compute rolling performance stats for a signal.

        Returns dict with Sharpe, PF, max DD, trade count, win rate,
        and promotion_ready flag.
        """
        trades = self.closed_trades.get(signal_id, [])
        pnl_list = [t.exit_pnl for t in trades]

        if not pnl_list:
            return {
                'signal_id': signal_id,
                'trade_count': 0,
                'sharpe': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'cumulative_pnl': 0.0,
                'promotion_ready': False,
            }

        sharpe = self._compute_sharpe(pnl_list)
        pf = self._compute_profit_factor(pnl_list)
        max_dd = self._compute_max_drawdown(pnl_list)
        win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list)

        promotion_ready = (
            len(pnl_list) >= PROMOTION_MIN_TRADES
            and sharpe >= PROMOTION_MIN_SHARPE
        )

        return {
            'signal_id': signal_id,
            'trade_count': len(pnl_list),
            'sharpe': round(sharpe, 2),
            'profit_factor': round(pf, 2),
            'max_drawdown': round(max_dd, 2),
            'win_rate': round(win_rate, 3),
            'cumulative_pnl': round(self.cumulative_pnl.get(signal_id, 0.0), 2),
            'promotion_ready': promotion_ready,
        }

    def get_promotion_candidates(self) -> List[dict]:
        """Return all signals meeting promotion criteria."""
        candidates = []
        for signal_id in L8_SIGNALS:
            wf = WF_RESULTS.get(signal_id, {})
            if wf.get('verdict') != 'PASS':
                continue
            stats = self.get_signal_stats(signal_id)
            if stats['promotion_ready']:
                candidates.append(stats)
        return candidates

    def get_status(self) -> dict:
        """
        Full status overview: per-signal status, trades, Sharpe, promotion %.
        """
        status = {}
        for signal_id in L8_SIGNALS:
            wf = WF_RESULTS.get(signal_id, {})
            verdict = wf.get('verdict', 'UNKNOWN')
            signal_status = 'SHADOW' if verdict == 'PASS' else 'MONITOR'

            stats = self.get_signal_stats(signal_id)
            trade_count = stats['trade_count']

            # Promotion progress
            if trade_count > 0 and signal_status == 'SHADOW':
                trade_pct = min(trade_count / PROMOTION_MIN_TRADES, 1.0)
                sharpe_pct = min(
                    max(stats['sharpe'] / PROMOTION_MIN_SHARPE, 0.0), 1.0
                )
                promotion_pct = round((trade_pct + sharpe_pct) / 2.0, 3)
            else:
                promotion_pct = 0.0

            status[signal_id] = {
                'status': signal_status,
                'wf_verdict': verdict,
                'wf_sharpe': wf.get('sharpe', 0.0),
                'trades': trade_count,
                'rolling_sharpe': stats['sharpe'],
                'win_rate': stats['win_rate'],
                'cumulative_pnl': stats['cumulative_pnl'],
                'promotion_pct': promotion_pct,
                'promotion_ready': stats['promotion_ready'],
                'has_open_position': signal_id in self.open_positions,
            }
        return status

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def persist_state(self, db_conn=None):
        """Save tracker state to l8_shadow_state table or JSON fallback."""
        conn = db_conn or self.conn
        state = self._serialize_state()

        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS l8_shadow_state (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    INSERT INTO l8_shadow_state (id, state, updated_at)
                    VALUES (1, %s, NOW())
                    ON CONFLICT (id)
                    DO UPDATE SET state = EXCLUDED.state,
                                  updated_at = NOW()
                """, (json.dumps(state),))
                conn.commit()
                logger.debug("L8 shadow state persisted to DB")
                return
            except Exception as e:
                logger.warning(f"DB persist failed, falling back to JSON: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass

        # JSON fallback
        fallback_path = os.path.join(
            os.path.dirname(__file__), 'l8_shadow_state.json'
        )
        with open(fallback_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug(f"L8 shadow state persisted to {fallback_path}")

    def load_state(self, db_conn=None):
        """Load tracker state from l8_shadow_state table or JSON fallback."""
        conn = db_conn or self.conn

        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT state FROM l8_shadow_state WHERE id = 1"
                )
                row = cur.fetchone()
                if row:
                    state = row[0]
                    if isinstance(state, str):
                        state = json.loads(state)
                    self._deserialize_state(state)
                    logger.debug("L8 shadow state loaded from DB")
                    return
            except Exception as e:
                logger.debug(f"DB load failed, trying JSON fallback: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass

        # JSON fallback
        fallback_path = os.path.join(
            os.path.dirname(__file__), 'l8_shadow_state.json'
        )
        if os.path.exists(fallback_path):
            with open(fallback_path) as f:
                state = json.load(f)
            self._deserialize_state(state)
            logger.debug(f"L8 shadow state loaded from {fallback_path}")

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _resolve_strike(self, spot: float, leg, sigma: float,
                        T: float, dte: int) -> float:
        """
        Resolve a strike price from a leg definition.

        Supports strike_method: atm, delta, offset.
        Rounds to nearest 50 (Nifty strike interval).
        """
        if leg.strike_method == 'atm':
            strike = spot
        elif leg.strike_method == 'delta' and leg.delta_target > 0:
            # Approximate strike from delta using BS inverse
            # For calls: higher strike = lower delta
            # For puts: lower strike = lower |delta|
            target = leg.delta_target
            if T <= 0 or sigma <= 0:
                strike = spot
            else:
                from scipy.stats import norm
                if leg.option_type == 'CE':
                    d1 = norm.ppf(target)
                    strike = spot * math.exp(
                        -d1 * sigma * math.sqrt(T)
                        + (RISK_FREE_RATE + 0.5 * sigma**2) * T
                    )
                else:  # PE
                    d1 = norm.ppf(1.0 - target)
                    strike = spot * math.exp(
                        -d1 * sigma * math.sqrt(T)
                        + (RISK_FREE_RATE + 0.5 * sigma**2) * T
                    )
        elif leg.strike_method == 'offset':
            strike = spot + leg.offset_points
        else:
            strike = spot

        # Round to nearest 50 (NSE Nifty strike interval)
        return round(strike / 50) * 50

    @staticmethod
    def _compute_sharpe(pnl_list: List[float]) -> float:
        """Annualized Sharpe from a list of trade P&Ls."""
        if len(pnl_list) < 5:
            return 0.0
        arr = np.array(pnl_list)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std <= 0:
            return 0.0
        return float((mean / std) * np.sqrt(252))

    @staticmethod
    def _compute_profit_factor(pnl_list: List[float]) -> float:
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        if gross_loss <= 0:
            return 99.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _compute_max_drawdown(pnl_list: List[float]) -> float:
        """Max drawdown as absolute value from peak equity."""
        if not pnl_list:
            return 0.0
        equity = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    def _serialize_state(self) -> dict:
        return {
            'open_positions': {
                sid: pos.to_dict()
                for sid, pos in self.open_positions.items()
            },
            'closed_trades': {
                sid: [t.to_dict() for t in trades]
                for sid, trades in self.closed_trades.items()
            },
            'cumulative_pnl': dict(self.cumulative_pnl),
        }

    def _deserialize_state(self, state: dict):
        self.open_positions = {
            sid: ShadowPosition.from_dict(d)
            for sid, d in state.get('open_positions', {}).items()
        }
        for sid, trade_list in state.get('closed_trades', {}).items():
            self.closed_trades[sid] = [
                ShadowPosition.from_dict(d) for d in trade_list
            ]
        for sid, pnl in state.get('cumulative_pnl', {}).items():
            self.cumulative_pnl[sid] = pnl

    def _try_load_state(self):
        """Best-effort load on init."""
        try:
            self.load_state()
        except Exception as e:
            logger.debug(f"State load skipped on init: {e}")

    def _try_persist_state(self):
        """Best-effort persist after daily update."""
        try:
            self.persist_state()
        except Exception as e:
            logger.warning(f"State persist failed: {e}")
