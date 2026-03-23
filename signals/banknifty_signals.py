"""
Bank Nifty intraday signals — cloned from proven Nifty strategies.

Adapts the two walk-forward-validated signals (KAUFMAN_BB_MR, GUJRAL_RANGE)
plus the 10 L9 Al Brooks signals for Bank Nifty 5-min bars.

Key differences from Nifty:
- Higher ATR (Bank Nifty moves 300-500 pts/day vs Nifty 100-200)
- Lot size: 15 (vs 25 for Nifty)
- Wider thresholds for breakout/range signals (scaled by ATR ratio)
- Same session times (9:15-15:30 IST)

Usage:
    from signals.banknifty_signals import BankNiftySignalComputer
    computer = BankNiftySignalComputer()
    fired = computer.compute_all(bar, prev_bar, session_bars)
"""

import numpy as np
import pandas as pd


# Bank Nifty / Nifty ATR ratio (historical average)
# Bank Nifty is ~2.1x more volatile than Nifty on absolute pts basis
# but similar on % basis (~1.0-1.2x). We scale absolute thresholds.
BN_NIFTY_ATR_RATIO = 2.1


class BankNiftySignalComputer:
    """Computes intraday signals adapted for Bank Nifty 5-min bars."""

    SIGNALS = {
        # BN_KAUFMAN_BB_MR: REMOVED — PF 0.26 on real BankNifty 5-min data (17 configs tested)
        # BN_GUJRAL_RANGE: REMOVED — PF 0.49 on real BankNifty 5-min data.
        # Session-range mean-reversion doesn't work intraday on BN.

        # ── L9 Al Brooks signals adapted for BankNifty ──
        'BN_ORB_BREAKOUT': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.005,   # wider SL for BN volatility
            'max_hold_bars': 35,
            'entry_start': '09:30',
            'entry_end': '14:00',
            'check': '_check_orb_breakout',
        },
        'BN_VWAP_RECLAIM': {
            'direction': 'LONG',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 25,
            'entry_start': '09:45',
            'entry_end': '14:30',
            'check': '_check_vwap_reclaim',
        },
        'BN_VWAP_REJECTION': {
            'direction': 'SHORT',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 25,
            'entry_start': '09:45',
            'entry_end': '14:30',
            'check': '_check_vwap_rejection',
        },
        'BN_FIRST_PULLBACK': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.005,
            'max_hold_bars': 20,
            'entry_start': '09:30',
            'entry_end': '13:00',
            'check': '_check_first_pullback',
        },
        'BN_FAILED_BREAKOUT': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.006,
            'max_hold_bars': 25,
            'entry_start': '09:45',
            'entry_end': '14:00',
            'check': '_check_failed_breakout',
        },
        'BN_GAP_FILL': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 20,
            'entry_start': '09:20',
            'entry_end': '11:00',
            'check': '_check_gap_fill',
        },
        'BN_TREND_BAR': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.005,
            'max_hold_bars': 25,
            'entry_start': '09:30',
            'entry_end': '14:30',
            'check': '_check_trend_bar',
        },
        'BN_EOD_TREND': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 12,
            'entry_start': '14:00',
            'entry_end': '15:10',
            'check': '_check_eod_trend',
        },
    }

    def compute_all(self, bar, prev_bar, session_bars):
        """
        Check all Bank Nifty signals for the current bar.

        Args:
            bar: current 5-min bar (Series with indicators)
            prev_bar: previous 5-min bar
            session_bars: DataFrame of all bars so far today

        Returns:
            list of fired signals [{signal_id, direction, price, reason}, ...]
        """
        fired = []
        bar_time = bar['datetime']

        if bar_time.hour == 15 and bar_time.minute > 10:
            return fired

        for signal_id, config in self.SIGNALS.items():
            entry_start = pd.Timestamp(f"{bar_time.date()} {config['entry_start']}")
            entry_end = pd.Timestamp(f"{bar_time.date()} {config['entry_end']}")
            if bar_time < entry_start or bar_time > entry_end:
                continue

            method = getattr(self, config['check'])
            result = method(signal_id, config, bar, prev_bar, session_bars)
            if result:
                fired.append(result)

        return fired

    # ================================================================
    # SIGNAL: Kaufman BB Mean Reversion (proven on Nifty, adapted)
    # ================================================================
    def _check_kaufman_bb_mr(self, signal_id, config, bar, prev_bar, session_bars):
        """
        Bollinger Band mean reversion when ADX < 25 (range-bound market).
        Long near lower band, short near upper band.
        """
        if pd.isna(bar.get('bb_lower')) or pd.isna(bar.get('adx_14')):
            return None

        close = float(bar['close'])
        bb_lower = float(bar['bb_lower'])
        bb_upper = float(bar['bb_upper'])
        bb_mid = float(bar['bb_mid'])
        adx = float(bar['adx_14'])

        # Only in range-bound conditions
        if adx >= 25:
            return None

        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return None

        # Long: close within 10% of lower band
        lower_zone = bb_lower + bb_width * 0.10
        if close <= lower_zone and close > bb_lower:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN BB MR long: close={close:.0f} near bb_low={bb_lower:.0f} adx={adx:.0f}',
            }

        # Short: close within 10% of upper band
        upper_zone = bb_upper - bb_width * 0.10
        if close >= upper_zone and close < bb_upper:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN BB MR short: close={close:.0f} near bb_up={bb_upper:.0f} adx={adx:.0f}',
            }

        return None

    # ================================================================
    # SIGNAL: Gujral Range Boundary (proven on Nifty, adapted)
    # ================================================================
    def _check_gujral_range(self, signal_id, config, bar, prev_bar, session_bars):
        """
        Mean reversion off session range boundaries.
        Long near session low, short near session high.
        """
        if len(session_bars) < 10:
            return None

        close = float(bar['close'])
        session_high = float(session_bars['high'].max())
        session_low = float(session_bars['low'].min())
        session_range = session_high - session_low

        if session_range <= 0:
            return None

        # Position within range (0 = low, 1 = high)
        position = (close - session_low) / session_range

        # Long: within 15% of session low
        if position <= 0.15 and close > session_low:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN Range long: pos={position:.0%} range={session_range:.0f}',
            }

        # Short: within 15% of session high
        if position >= 0.85 and close < session_high:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN Range short: pos={position:.0%} range={session_range:.0f}',
            }

        return None

    # ================================================================
    # L9 signals — adapted for Bank Nifty volatility
    # ================================================================
    def _check_orb_breakout(self, signal_id, config, bar, prev_bar, session_bars):
        """ORB breakout — volume filter disabled (Kite trade count issue)."""
        if pd.isna(bar.get('opening_range_high')):
            return None

        close = float(bar['close'])
        or_high = float(bar['opening_range_high'])
        or_low = float(bar['opening_range_low'])
        prev_close = float(prev_bar['close']) if prev_bar is not None else close

        if close > or_high and prev_close <= or_high:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN ORB long: close={close:.0f} > OR_high={or_high:.0f}',
            }

        if close < or_low and prev_close >= or_low:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN ORB short: close={close:.0f} < OR_low={or_low:.0f}',
            }

        return None

    def _check_vwap_reclaim(self, signal_id, config, bar, prev_bar, session_bars):
        """Price dips below VWAP then reclaims — bullish."""
        if pd.isna(bar.get('vwap')) or prev_bar is None:
            return None

        close = float(bar['close'])
        vwap = float(bar['vwap'])
        prev_close = float(prev_bar['close'])
        prev_vwap = float(prev_bar['vwap']) if pd.notna(prev_bar.get('vwap')) else vwap

        if prev_close < prev_vwap and close > vwap:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN VWAP reclaim: {close:.0f} > vwap={vwap:.0f}',
            }
        return None

    def _check_vwap_rejection(self, signal_id, config, bar, prev_bar, session_bars):
        """Bar touches VWAP but fails to hold — bearish."""
        if pd.isna(bar.get('vwap')) or prev_bar is None:
            return None

        close = float(bar['close'])
        high = float(bar['high'])
        vwap = float(bar['vwap'])
        prev_close = float(prev_bar['close'])

        if prev_close < vwap and high >= vwap * 0.999 and close < vwap:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN VWAP rejection: high={high:.0f} touch vwap={vwap:.0f} close={close:.0f}',
            }
        return None

    def _check_first_pullback(self, signal_id, config, bar, prev_bar, session_bars):
        """Enter on first pullback after strong trend bar."""
        if prev_bar is None or len(session_bars) < 5:
            return None

        close = float(bar['close'])
        prev_close = float(prev_bar['close'])
        prev_body_pct = float(prev_bar['body_pct']) if pd.notna(prev_bar.get('body_pct')) else 0

        # BN: slightly lower body threshold (0.65 vs 0.70) — BN bars have longer wicks
        if prev_body_pct < 0.65:
            return None

        prev_body = float(prev_bar['close']) - float(prev_bar['open'])

        if prev_body > 0 and close < prev_close and close > float(prev_bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN pullback long: trend_body={prev_body_pct:.0%}',
            }

        if prev_body < 0 and close > prev_close and close < float(prev_bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN pullback short: trend_body={prev_body_pct:.0%}',
            }

        return None

    def _check_failed_breakout(self, signal_id, config, bar, prev_bar, session_bars):
        """Failed breakout reversal — wick beyond range, close inside."""
        if len(session_bars) < 10:
            return None

        close = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])
        open_ = float(bar['open'])

        session_high = float(session_bars['high'].max())
        session_low = float(session_bars['low'].min())

        if high > session_high and close < session_high and close < open_:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN failed BO short: high={high:.0f} > sess_high={session_high:.0f}',
            }

        if low < session_low and close > session_low and close > open_:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN failed BD long: low={low:.0f} < sess_low={session_low:.0f}',
            }

        return None

    def _check_gap_fill(self, signal_id, config, bar, prev_bar, session_bars):
        """Fade overnight gap > 0.4% (wider than Nifty's 0.3% threshold)."""
        if pd.isna(bar.get('overnight_gap_pct')):
            return None

        gap = float(bar['overnight_gap_pct'])
        close = float(bar['close'])
        prev_day_close = float(bar['prev_day_close']) if pd.notna(bar.get('prev_day_close')) else None

        if prev_day_close is None or abs(gap) < 0.004:  # 0.4% for BN (vs 0.3% Nifty)
            return None

        if gap > 0.004 and close > prev_day_close:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN gap fill short: gap={gap:.2%} target={prev_day_close:.0f}',
            }

        if gap < -0.004 and close < prev_day_close:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN gap fill long: gap={gap:.2%} target={prev_day_close:.0f}',
            }

        return None

    def _check_trend_bar(self, signal_id, config, bar, prev_bar, session_bars):
        """Strong trend bar after 3+ dojis — breakout."""
        if len(session_bars) < 5:
            return None

        body_pct = float(bar['body_pct']) if pd.notna(bar.get('body_pct')) else 0
        if body_pct < 0.65:  # 0.65 for BN (vs 0.70 for Nifty)
            return None

        recent = session_bars.tail(4).head(3)
        if len(recent) < 3:
            return None

        small_bars = sum(1 for _, r in recent.iterrows()
                         if pd.notna(r.get('body_pct')) and float(r['body_pct']) < 0.35)
        if small_bars < 3:
            return None

        body = float(bar['close']) - float(bar['open'])
        direction = 'LONG' if body > 0 else 'SHORT'
        return {
            'signal_id': signal_id, 'direction': direction, 'price': float(bar['close']),
            'reason': f'BN trend bar {direction.lower()}: body={body_pct:.0%} after {small_bars} dojis',
        }

    def _check_eod_trend(self, signal_id, config, bar, prev_bar, session_bars):
        """EOD trend resumption — volume filter disabled (Kite trade count issue)."""
        if len(session_bars) < 30:
            return None

        close = float(bar['close'])
        ema_20 = float(bar['ema_20']) if pd.notna(bar.get('ema_20')) else close

        if close > ema_20 * 1.001:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'BN EOD trend long: close > ema20',
            }
        elif close < ema_20 * 0.999:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'BN EOD trend short: close < ema20',
            }

        return None


# ================================================================
# WF VALIDATION RESULTS (populated by backtest/intraday_walk_forward.py)
# Run: venv/bin/python3 -m backtest.intraday_walk_forward --instrument BANKNIFTY
# Criteria: Sharpe >= 0.8, PF >= 1.3, WF >= 60%
# ================================================================

BN_WF_RESULTS = {
    # REAL DATA RUN: venv/bin/python3 -m backtest.intraday_walk_forward (2026-03-22)
    # 495 trading days of real Kite 5-min bars (2024-03-22 to 2026-03-20)
    # Criteria: Sharpe >= 0.8, PF >= 1.3, WF >= 60%
    # RESULT: 0/8 PASS — same collapse as Nifty L9 signals
    'BN_ORB_BREAKOUT':    {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'BN_VWAP_RECLAIM':    {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'BN_VWAP_REJECTION':  {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'BN_FIRST_PULLBACK':  {'verdict': 'FAIL', 'sharpe': -0.97, 'pf': 0.89, 'win_rate': 0.463, 'wf_pass_rate': 0.12, 'trades': 1232},
    'BN_FAILED_BREAKOUT': {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'BN_GAP_FILL':        {'verdict': 'FAIL', 'sharpe': -1.43, 'pf': 0.83, 'win_rate': 0.447, 'wf_pass_rate': 0.06, 'trades': 226},
    'BN_TREND_BAR':       {'verdict': 'FAIL', 'sharpe': -0.52, 'pf': 0.93, 'win_rate': 0.466, 'wf_pass_rate': 0.25, 'trades': 474},
    'BN_EOD_TREND':       {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
}


def get_bn_passing_signals():
    """Return BN signal IDs that passed walk-forward validation."""
    return [sid for sid, r in BN_WF_RESULTS.items() if r.get('verdict') == 'PASS']
