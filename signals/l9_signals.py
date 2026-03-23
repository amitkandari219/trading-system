"""
L9 Al Brooks Price Action Signals for Nifty Intraday.

10 hand-coded signals operating on 5-min bars.
All positions close by 15:20. No overnight holds.

Each signal returns {signal_id, direction, price, reason} or None.
Session context (all bars for current day) is passed to each check.
"""

import numpy as np
import pandas as pd


class IntradaySignalComputer:
    """Computes L9 intraday signals from 5-min bar data."""

    SIGNALS = {
        'L9_ORB_BREAKOUT': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 40,
            'entry_start': '09:30',
            'entry_end': '14:00',
            'check': '_check_orb_breakout',
        },
        'L9_VWAP_RECLAIM': {
            'direction': 'LONG',
            'stop_loss_pct': 0.003,
            'max_hold_bars': 30,
            'entry_start': '09:45',
            'entry_end': '14:30',
            'check': '_check_vwap_reclaim',
        },
        'L9_VWAP_REJECTION': {
            'direction': 'SHORT',
            'stop_loss_pct': 0.003,
            'max_hold_bars': 30,
            'entry_start': '09:45',
            'entry_end': '14:30',
            'check': '_check_vwap_rejection',
        },
        'L9_FIRST_PULLBACK': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 25,
            'entry_start': '09:30',
            'entry_end': '13:00',
            'check': '_check_first_pullback',
        },
        'L9_FAILED_BREAKOUT': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.005,
            'max_hold_bars': 30,
            'entry_start': '09:45',
            'entry_end': '14:00',
            'check': '_check_failed_breakout',
        },
        'L9_DOUBLE_BOTTOM': {
            'direction': 'LONG',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 35,
            'entry_start': '10:00',
            'entry_end': '14:30',
            'check': '_check_double_bottom',
        },
        'L9_DOUBLE_TOP': {
            'direction': 'SHORT',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 35,
            'entry_start': '10:00',
            'entry_end': '14:30',
            'check': '_check_double_top',
        },
        'L9_GAP_FILL': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.003,
            'max_hold_bars': 20,
            'entry_start': '09:20',
            'entry_end': '11:00',
            'check': '_check_gap_fill',
        },
        'L9_TREND_BAR': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.004,
            'max_hold_bars': 30,
            'entry_start': '09:30',
            'entry_end': '14:30',
            'check': '_check_trend_bar',
        },
        'L9_EOD_TREND': {
            'direction': 'BOTH',
            'stop_loss_pct': 0.003,
            'max_hold_bars': 15,
            'entry_start': '14:00',
            'entry_end': '15:10',
            'check': '_check_eod_trend',
        },
    }

    def compute_all(self, bar, prev_bar, session_bars):
        """
        Check all L9 signals for the current bar.

        Args:
            bar: current 5-min bar (Series with indicators)
            prev_bar: previous 5-min bar
            session_bars: DataFrame of all bars so far today

        Returns:
            list of fired signals [{signal_id, direction, price, reason}, ...]
        """
        fired = []
        bar_time = bar['datetime']

        # Don't enter after 15:10 (need time to exit by 15:20)
        if bar_time.hour == 15 and bar_time.minute > 10:
            return fired

        for signal_id, config in self.SIGNALS.items():
            # Time window filter
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
    # SIGNAL 1: Opening Range Breakout (ORB)
    # ================================================================
    def _check_orb_breakout(self, signal_id, config, bar, prev_bar, session_bars):
        """Buy on break above 15-min opening range, short on break below."""
        if pd.isna(bar.get('opening_range_high')):
            return None

        close = float(bar['close'])
        or_high = float(bar['opening_range_high'])
        or_low = float(bar['opening_range_low'])
        prev_close = float(prev_bar['close']) if prev_bar is not None else close

        # Volume filter disabled: Kite returns trade count, not actual volume.
        # vol_ratio on trade counts has different distribution. Re-enable after
        # calibrating thresholds on real Kite trade-count data.

        # Long: close breaks above OR high (and wasn't above before)
        if close > or_high and prev_close <= or_high:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'ORB long: close={close:.0f} > OR_high={or_high:.0f} OR_w={or_high - or_low:.0f}',
            }

        # Short: close breaks below OR low
        if close < or_low and prev_close >= or_low:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'ORB short: close={close:.0f} < OR_low={or_low:.0f} OR_w={or_high - or_low:.0f}',
            }

        return None

    # ================================================================
    # SIGNAL 2: VWAP Reclaim (Long)
    # ================================================================
    def _check_vwap_reclaim(self, signal_id, config, bar, prev_bar, session_bars):
        """Price dips below VWAP then reclaims it — bullish."""
        if pd.isna(bar.get('vwap')) or prev_bar is None:
            return None

        close = float(bar['close'])
        vwap = float(bar['vwap'])
        prev_close = float(prev_bar['close'])
        prev_vwap = float(prev_bar['vwap']) if pd.notna(prev_bar.get('vwap')) else vwap

        # Crosses above VWAP from below
        if prev_close < prev_vwap and close > vwap:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'VWAP reclaim: {close:.0f} > vwap={vwap:.0f}',
            }
        return None

    # ================================================================
    # SIGNAL 3: VWAP Rejection (Short)
    # ================================================================
    def _check_vwap_rejection(self, signal_id, config, bar, prev_bar, session_bars):
        """Price rallies to VWAP but fails to hold — bearish."""
        if pd.isna(bar.get('vwap')) or prev_bar is None:
            return None

        close = float(bar['close'])
        high = float(bar['high'])
        vwap = float(bar['vwap'])
        prev_close = float(prev_bar['close'])

        # Bar touches VWAP (high >= vwap) but closes below it, AND was below before
        if prev_close < vwap and high >= vwap * 0.999 and close < vwap:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'VWAP rejection: high={high:.0f} touched vwap={vwap:.0f}, closed={close:.0f}',
            }
        return None

    # ================================================================
    # SIGNAL 4: First Pullback After Trend Bar
    # ================================================================
    def _check_first_pullback(self, signal_id, config, bar, prev_bar, session_bars):
        """After a strong trend bar, enter on first retracement."""
        if prev_bar is None or len(session_bars) < 5:
            return None

        close = float(bar['close'])
        prev_close = float(prev_bar['close'])
        prev_body_pct = float(prev_bar['body_pct']) if pd.notna(prev_bar.get('body_pct')) else 0

        # Previous bar must be strong trend bar (body > 70% of range)
        if prev_body_pct < 0.7:
            return None

        prev_body = float(prev_bar['close']) - float(prev_bar['open'])

        # Pullback: current bar retraces into previous bar's range
        if prev_body > 0 and close < prev_close and close > float(prev_bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'First pullback long: trend_bar_body={prev_body_pct:.0%}',
            }

        if prev_body < 0 and close > prev_close and close < float(prev_bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'First pullback short: trend_bar_body={prev_body_pct:.0%}',
            }

        return None

    # ================================================================
    # SIGNAL 5: Failed Breakout Reversal
    # ================================================================
    def _check_failed_breakout(self, signal_id, config, bar, prev_bar, session_bars):
        """Breakout above session high fails (wick above, close inside) — reverse short."""
        if len(session_bars) < 10:
            return None

        close = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])
        open_ = float(bar['open'])

        session_high = float(session_bars['high'].max())
        session_low = float(session_bars['low'].min())

        # Failed breakout high: wick above session high but close inside
        if high > session_high and close < session_high and close < open_:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'Failed breakout short: high={high:.0f} > sess_high={session_high:.0f}',
            }

        # Failed breakdown: wick below session low but close inside
        if low < session_low and close > session_low and close > open_:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'Failed breakdown long: low={low:.0f} < sess_low={session_low:.0f}',
            }

        return None

    # ================================================================
    # SIGNAL 6: Double Bottom at VWAP/S1
    # ================================================================
    def _check_double_bottom(self, signal_id, config, bar, prev_bar, session_bars):
        """Two touches at support level (VWAP or session low) — buy."""
        if len(session_bars) < 15 or pd.isna(bar.get('vwap')):
            return None

        close = float(bar['close'])
        low = float(bar['low'])
        vwap = float(bar['vwap'])

        # Find prior lows near current low (within 0.1%)
        recent = session_bars.tail(20)
        prior_lows = recent['low'].values
        tolerance = vwap * 0.001

        # Count how many bars touched this level
        touches = sum(1 for l in prior_lows if abs(l - low) < tolerance)

        if touches >= 2 and close > low + tolerance and close > float(bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'Double bottom: {touches} touches at {low:.0f}',
            }
        return None

    # ================================================================
    # SIGNAL 7: Double Top at VWAP/R1
    # ================================================================
    def _check_double_top(self, signal_id, config, bar, prev_bar, session_bars):
        """Two touches at resistance level — sell."""
        if len(session_bars) < 15 or pd.isna(bar.get('vwap')):
            return None

        close = float(bar['close'])
        high = float(bar['high'])
        vwap = float(bar['vwap'])

        recent = session_bars.tail(20)
        prior_highs = recent['high'].values
        tolerance = vwap * 0.001

        touches = sum(1 for h in prior_highs if abs(h - high) < tolerance)

        if touches >= 2 and close < high - tolerance and close < float(bar['open']):
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'Double top: {touches} touches at {high:.0f}',
            }
        return None

    # ================================================================
    # SIGNAL 8: Gap Fill
    # ================================================================
    def _check_gap_fill(self, signal_id, config, bar, prev_bar, session_bars):
        """Gaps > 0.3% tend to fill by 11:00 AM. Fade the gap."""
        if pd.isna(bar.get('overnight_gap_pct')):
            return None

        gap = float(bar['overnight_gap_pct'])
        close = float(bar['close'])
        prev_day_close = float(bar['prev_day_close']) if pd.notna(bar.get('prev_day_close')) else None

        if prev_day_close is None or abs(gap) < 0.003:
            return None

        # Gap up > 0.3%: short toward previous close
        if gap > 0.003 and close > prev_day_close:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'Gap fill short: gap={gap:.2%} close={close:.0f} target={prev_day_close:.0f}',
            }

        # Gap down > 0.3%: long toward previous close
        if gap < -0.003 and close < prev_day_close:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'Gap fill long: gap={gap:.2%} close={close:.0f} target={prev_day_close:.0f}',
            }

        return None

    # ================================================================
    # SIGNAL 9: Trend Bar After Dojis
    # ================================================================
    def _check_trend_bar(self, signal_id, config, bar, prev_bar, session_bars):
        """Strong trend bar (body > 70%) after 3+ small/doji bars — breakout."""
        if len(session_bars) < 5:
            return None

        body_pct = float(bar['body_pct']) if pd.notna(bar.get('body_pct')) else 0
        if body_pct < 0.7:
            return None

        # Check last 3 bars were small (body_pct < 0.3)
        recent = session_bars.tail(4).head(3)  # 3 bars before current
        if len(recent) < 3:
            return None

        small_bars = sum(1 for _, r in recent.iterrows()
                         if pd.notna(r.get('body_pct')) and float(r['body_pct']) < 0.3)
        if small_bars < 3:
            return None

        body = float(bar['close']) - float(bar['open'])
        if body > 0:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': float(bar['close']),
                'reason': f'Trend bar long: body={body_pct:.0%} after {small_bars} dojis',
            }
        else:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': float(bar['close']),
                'reason': f'Trend bar short: body={body_pct:.0%} after {small_bars} dojis',
            }

    # ================================================================
    # SIGNAL 10: EOD Trend Resumption
    # ================================================================
    def _check_eod_trend(self, signal_id, config, bar, prev_bar, session_bars):
        """Trend resumption in last 90 minutes with volume."""
        if len(session_bars) < 30:
            return None

        close = float(bar['close'])
        ema_20 = float(bar['ema_20']) if pd.notna(bar.get('ema_20')) else close

        # Volume filter disabled: Kite trade count != actual volume.
        # Re-enable after calibrating on real Kite data.

        # Intraday trend direction from EMA
        if close > ema_20 * 1.001:
            return {
                'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                'reason': f'EOD trend long: close={close:.0f} > ema20={ema_20:.0f}',
            }
        elif close < ema_20 * 0.999:
            return {
                'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                'reason': f'EOD trend short: close={close:.0f} < ema20={ema_20:.0f}',
            }

        return None


# ================================================================
# WF VALIDATION RESULTS (populated by backtest/intraday_walk_forward.py)
# Run: venv/bin/python3 -m backtest.intraday_walk_forward
# Criteria: Sharpe >= 0.8, PF >= 1.3, WF >= 60%
# ================================================================

L9_WF_RESULTS = {
    # REAL DATA RUN: venv/bin/python3 -m backtest.intraday_walk_forward (2026-03-22)
    # 495 trading days of real Kite 5-min bars (2024-03-22 to 2026-03-20)
    # Criteria: Sharpe >= 0.8, PF >= 1.3, WF >= 60%
    # RESULT: 0/10 PASS — synthetic bars grossly inflated all signals
    'L9_ORB_BREAKOUT':    {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0,    'notes': 'Zero trades on real data — synthetic opening ranges dont match real price action'},
    'L9_VWAP_RECLAIM':    {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'L9_VWAP_REJECTION':  {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'L9_FIRST_PULLBACK':  {'verdict': 'FAIL', 'sharpe': -0.24, 'pf': 0.97, 'win_rate': 0.463, 'wf_pass_rate': 0.25, 'trades': 1081},
    'L9_FAILED_BREAKOUT': {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
    'L9_DOUBLE_BOTTOM':   {'verdict': 'FAIL', 'sharpe': 0.25,  'pf': 1.04, 'win_rate': 0.452, 'wf_pass_rate': 0.12, 'trades': 748},
    'L9_DOUBLE_TOP':      {'verdict': 'FAIL', 'sharpe': -0.02, 'pf': 1.00, 'win_rate': 0.457, 'wf_pass_rate': 0.25, 'trades': 786},
    'L9_GAP_FILL':        {'verdict': 'FAIL', 'sharpe': -0.73, 'pf': 0.91, 'win_rate': 0.420, 'wf_pass_rate': 0.31, 'trades': 288},
    'L9_TREND_BAR':       {'verdict': 'FAIL', 'sharpe': -0.72, 'pf': 0.89, 'win_rate': 0.412, 'wf_pass_rate': 0.31, 'trades': 245},
    'L9_EOD_TREND':       {'verdict': 'FAIL', 'sharpe': 0.00,  'pf': 0.00, 'win_rate': 0.000, 'wf_pass_rate': 0.00, 'trades': 0},
}


def get_l9_passing_signals():
    """Return L9 signal IDs that passed walk-forward validation."""
    return [sid for sid, r in L9_WF_RESULTS.items() if r.get('verdict') == 'PASS']
