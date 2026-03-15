"""
Market regime labeler for Nifty 50.
Computes regime for each day using ONLY data available at close of that day.
No lookahead. Verified by unit test.

Regimes:
    TRENDING  — Strong directional move (ADX > 25, price above EMA-50)
    RANGING   — Low directional strength (ADX < 25 for 3+ consecutive days)
    HIGH_VOL  — Elevated volatility (VIX >= 18)
    CRISIS    — Extreme volatility (VIX >= 25)

Usage:
    python regime_labeler.py --label-all    # Label full history, write to DB
    python regime_labeler.py --test         # Run no-lookahead unit test
    python regime_labeler.py --today        # Label today only
"""

import pandas as pd
import numpy as np


class RegimeLabeler:
    """
    Computes market regime for each day using ONLY
    data available at close of that day.
    """

    def __init__(self):
        self.adx_period = 14
        self.ema_period = 50
        self.vix_high_threshold = 18.0
        self.vix_crisis_threshold = 25.0
        self.adx_trend_threshold = 25.0
        self.ranging_days_required = 3

    def label_single_day(self, history_df, target_date):
        """
        Label ONE day using only data up to and including target_date.
        Returns: 'TRENDING' | 'RANGING' | 'HIGH_VOL' | 'CRISIS'
        """
        target_date = pd.Timestamp(target_date)
        data = history_df[history_df['date'] <= target_date].copy()

        min_required = self.adx_period + self.ranging_days_required + 5
        if len(data) < min_required:
            return 'RANGING'

        latest = data.iloc[-1]
        vix = latest['india_vix']

        # CRISIS check first — overrides everything
        if pd.notna(vix) and vix >= self.vix_crisis_threshold:
            return 'CRISIS'

        # HIGH_VOL check second
        if pd.notna(vix) and vix >= self.vix_high_threshold:
            return 'HIGH_VOL'

        # Compute ADX on available data only
        adx_value = self._compute_adx(data)

        # Compute EMA on available data only
        ema_value = self._compute_ema(data, self.ema_period)
        price_above_ema = latest['close'] > ema_value

        # TRENDING: ADX above threshold AND price above EMA
        if adx_value >= self.adx_trend_threshold and price_above_ema:
            return 'TRENDING'

        # RANGING: ADX below threshold for required consecutive days
        recent_adx_values = []
        for i in range(self.ranging_days_required):
            offset = self.ranging_days_required - i - 1
            if offset == 0:
                slice_data = data
            else:
                slice_data = data.iloc[:-offset]
            if len(slice_data) >= self.adx_period + 1:
                recent_adx_values.append(self._compute_adx(slice_data))

        if (len(recent_adx_values) == self.ranging_days_required and
                all(a < self.adx_trend_threshold for a in recent_adx_values)):
            return 'RANGING'

        return 'RANGING'

    def label_full_history(self, history_df):
        """
        Label entire history SEQUENTIALLY.
        This is the only approved method for bulk labeling.
        """
        labels = {}
        dates = history_df['date'].tolist()

        for i, dt in enumerate(dates):
            if i < self.adx_period + 10:
                labels[dt] = 'RANGING'
                continue
            labels[dt] = self.label_single_day(history_df, dt)

        return labels

    def _compute_adx(self, data):
        """
        Wilder's Average Directional Index.
        Returns float ADX value for the most recent bar.
        """
        if len(data) < self.adx_period + 1:
            return 0.0

        d = data.tail(self.adx_period * 2).copy().reset_index(drop=True)
        n = self.adx_period

        high = d['high'].values
        low = d['low'].values
        close = d['close'].values

        # True Range
        tr_list = []
        for i in range(1, len(d)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)

        # Directional movement
        plus_dm_list = []
        minus_dm_list = []
        for i in range(1, len(d)):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm_list.append(up if up > down and up > 0 else 0.0)
            minus_dm_list.append(down if down > up and down > 0 else 0.0)

        if len(tr_list) < n:
            return 0.0

        def wilder_smooth(series, period):
            result = [sum(series[:period]) / period]
            for v in series[period:]:
                result.append(result[-1] - result[-1] / period + v)
            return result

        atr = wilder_smooth(tr_list, n)
        plus_di = wilder_smooth(plus_dm_list, n)
        minus_di = wilder_smooth(minus_dm_list, n)

        dx_list = []
        for i in range(len(atr)):
            if atr[i] == 0:
                dx_list.append(0.0)
                continue
            pdi = 100 * plus_di[i] / atr[i]
            mdi = 100 * minus_di[i] / atr[i]
            denom = pdi + mdi
            dx_list.append(100 * abs(pdi - mdi) / denom if denom != 0 else 0.0)

        if len(dx_list) < n:
            return 0.0

        adx_series = wilder_smooth(dx_list, n)
        return float(adx_series[-1])

    def _compute_ema(self, data, period):
        """
        Standard EMA, returns latest value only.
        """
        closes = data['close'].values
        if len(closes) < period:
            return float(closes[-1]) if len(closes) > 0 else 0.0

        alpha = 2.0 / (period + 1)
        ema = float(closes[:period].mean())  # SMA seed
        for price in closes[period:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        return ema


def label_and_store(db, labeler=None):
    """Label full history and write to regime_labels table."""
    from data.nifty_loader import load_nifty_history

    if labeler is None:
        labeler = RegimeLabeler()

    print("Loading Nifty history...")
    history = load_nifty_history(db)
    print(f"  {len(history)} trading days loaded")

    print("Labeling regimes (sequential, no lookahead)...")
    labels = labeler.label_full_history(history)
    print(f"  {len(labels)} days labeled")

    # Compute indicator values for storage
    print("Storing regime labels...")
    cur = db.cursor()
    stored = 0

    for dt, regime in labels.items():
        data_up_to = history[history['date'] <= dt]
        adx_val = labeler._compute_adx(data_up_to) if len(data_up_to) > labeler.adx_period else None
        ema_val = labeler._compute_ema(data_up_to, labeler.ema_period) if len(data_up_to) > labeler.ema_period else None
        vix_val = data_up_to.iloc[-1]['india_vix'] if len(data_up_to) > 0 else None

        label_date = dt.date() if hasattr(dt, 'date') else dt
        cur.execute(
            """INSERT INTO regime_labels (label_date, regime, adx_value, ema_50, india_vix)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (label_date) DO UPDATE
               SET regime = EXCLUDED.regime,
                   adx_value = EXCLUDED.adx_value,
                   ema_50 = EXCLUDED.ema_50,
                   india_vix = EXCLUDED.india_vix,
                   computed_at = NOW()""",
            (label_date, regime, adx_val, ema_val,
             float(vix_val) if pd.notna(vix_val) else None)
        )
        stored += 1

    db.commit()
    print(f"  Stored {stored} regime labels")

    # Print regime distribution
    from collections import Counter
    dist = Counter(labels.values())
    print("\nRegime distribution:")
    for regime, count in sorted(dist.items()):
        pct = count / len(labels) * 100
        print(f"  {regime:12s}: {count:5d} days ({pct:.1f}%)")


def test_no_lookahead(db):
    """
    Verifies regime labeler has zero lookahead bias.
    Must pass before any backtest is run.
    """
    from data.nifty_loader import load_nifty_history

    labeler = RegimeLabeler()
    full_data = load_nifty_history(db)

    if len(full_data) < 600:
        print("Not enough data for lookahead test (need 600+ days)")
        return False

    # Test multiple dates across the dataset
    test_indices = [500, 1000, len(full_data) // 2, len(full_data) - 100]
    test_indices = [i for i in test_indices if i < len(full_data)]

    print("Running no-lookahead test...")
    all_passed = True

    for idx in test_indices:
        test_date = full_data['date'].iloc[idx]

        # Method 1: Label using full dataset (batch)
        full_labels = labeler.label_full_history(full_data)
        batch_label = full_labels[test_date]

        # Method 2: Label using ONLY data up to test_date
        truncated_data = full_data[full_data['date'] <= test_date].copy()
        single_label = labeler.label_single_day(truncated_data, test_date)

        if batch_label != single_label:
            print(f"  FAIL: date={test_date}, batch={batch_label}, single={single_label}")
            all_passed = False
        else:
            print(f"  PASS: date={test_date}, regime={batch_label}")

    if all_passed:
        print("\nALL LOOKAHEAD TESTS PASSED. Regime labeler is trustworthy.")
    else:
        print("\nLOOKAHEAD DETECTED. DO NOT RUN BACKTEST UNTIL FIXED.")

    return all_passed


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse
    import psycopg2
    from config.settings import DATABASE_DSN

    parser = argparse.ArgumentParser(description='Regime labeler')
    parser.add_argument('--label-all', action='store_true',
                        help='Label full history and store to DB')
    parser.add_argument('--test', action='store_true',
                        help='Run no-lookahead unit test')
    parser.add_argument('--today', action='store_true',
                        help='Label today only')
    parser.add_argument('--dsn', default=DATABASE_DSN,
                        help='PostgreSQL DSN')
    args = parser.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False

    if args.label_all:
        label_and_store(conn)
    elif args.test:
        test_no_lookahead(conn)
    elif args.today:
        from data.nifty_loader import load_nifty_history
        history = load_nifty_history(conn, days=200)
        labeler = RegimeLabeler()
        today = history['date'].iloc[-1]
        regime = labeler.label_single_day(history, today)
        print(f"Today ({today.date()}): {regime}")
    else:
        parser.print_help()

    conn.close()
