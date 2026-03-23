"""
FII backfill — historical FII OI data download and pattern validation.

Downloads last N months of NSE participant-wise OI data, computes derived
metrics (fii_net_oi_change, long_short_ratio, fii_options_pcr, dii_net),
runs 6 FII patterns on historical data, and validates pattern frequency.

Expected frequency per pattern: 2-5 per month. If too many or too few,
the pattern thresholds may need recalibration.

CLI:
    venv/bin/python3 -m fii.backfill --months 6
    venv/bin/python3 -m fii.backfill --months 3 --validate-only
    venv/bin/python3 -m fii.backfill --status
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN
from fii.downloader import FIIDataDownloader, DataNotAvailableError
from fii.signal_detector import FIISignalDetector, FIISignalResult

logger = logging.getLogger(__name__)


# ================================================================
# DERIVED METRIC COMPUTATION
# ================================================================

def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived metrics from raw FII daily data.

    Adds columns:
        fii_net_oi_change:  day-over-day change in FII net futures OI
        long_short_ratio:   FII long / (long + short) futures contracts
        fii_options_pcr:    FII put OI / call OI
        dii_net:            DII net futures position (if available)

    Args:
        df: DataFrame with columns from fii_daily_metrics table
            (trade_date, fii_long_contracts, fii_short_contracts, ...)

    Returns:
        DataFrame with additional derived columns
    """
    df = df.sort_values('trade_date').copy()

    # Net futures position
    df['fii_net'] = df['fii_long_contracts'] - df['fii_short_contracts']

    # Day-over-day change in net futures
    df['fii_net_oi_change'] = df['fii_net'].diff()

    # Long-short ratio: long / (long + short)
    total = df['fii_long_contracts'] + df['fii_short_contracts']
    df['long_short_ratio'] = np.where(total > 0, df['fii_long_contracts'] / total, 0.5)

    # FII options PCR (put_oi / call_oi) — from pcr_daily if available
    # Placeholder: filled from joined data if present
    if 'pcr_oi' in df.columns:
        df['fii_options_pcr'] = df['pcr_oi']
    else:
        df['fii_options_pcr'] = np.nan

    # DII net (if dii columns present)
    if 'dii_long_contracts' in df.columns and 'dii_short_contracts' in df.columns:
        df['dii_net'] = (
            df['dii_long_contracts'].fillna(0) - df['dii_short_contracts'].fillna(0)
        )
    else:
        df['dii_net'] = np.nan

    # Rolling stats for z-score and percentile
    df['fii_net_20d_mean'] = df['fii_net'].rolling(20, min_periods=10).mean()
    df['fii_net_20d_std'] = df['fii_net'].rolling(20, min_periods=10).std()
    df['z_score_20d'] = np.where(
        df['fii_net_20d_std'] > 0,
        (df['fii_net'] - df['fii_net_20d_mean']) / df['fii_net_20d_std'],
        0.0,
    )

    # Percentile rank over 252-day window
    df['percentile_252d'] = (
        df['fii_net']
        .rolling(252, min_periods=20)
        .apply(lambda x: (x.values[-1:] <= x.values).mean() * 100, raw=False)
    )

    # 5-day cumulative flow
    df['flow_5d'] = df['fii_net_oi_change'].rolling(5, min_periods=1).sum()

    return df


# ================================================================
# PATTERN RUNNER
# ================================================================

def run_patterns_on_history(
    df: pd.DataFrame,
    conn=None,
) -> List[Dict]:
    """
    Run 6 FII signal patterns on historical data.

    Patterns 1-4 via FIISignalDetector, patterns 5-6 inline.

    Returns list of signal dicts with date, pattern_name, direction, etc.
    """
    detector = FIISignalDetector(conn) if conn else FIISignalDetector(None)
    all_signals: List[Dict] = []

    for i, row in df.iterrows():
        trade_date = row['trade_date']
        fii_net = float(row.get('fii_net', 0))
        z_score = float(row.get('z_score_20d', 0))
        percentile = float(row.get('percentile_252d', 50))
        dii_net = float(row.get('dii_net', 0)) if pd.notna(row.get('dii_net')) else None
        long_short_ratio = float(row.get('long_short_ratio', 0.5))

        # Patterns 1-4: use detector if enough history
        if i >= 20:
            try:
                # Build a mini DataFrame row in the format detector expects
                row_df = pd.DataFrame([{
                    'date': trade_date,
                    'future_long_contracts': float(row.get('fii_long_contracts', 0)),
                    'future_short_contracts': float(row.get('fii_short_contracts', 0)),
                    'put_long_contracts': 0,
                    'put_short_contracts': 0,
                    'call_long_contracts': 0,
                    'call_short_contracts': 0,
                    'fut_net': fii_net,
                    'pcr': float(row.get('fii_options_pcr', 1.0)) if pd.notna(row.get('fii_options_pcr')) else 1.0,
                    'put_ratio': 1.0,
                }])
                signal = detector.detect(row_df, for_date=trade_date)
                if signal.signal_id and signal.direction != 'NEUTRAL':
                    all_signals.append({
                        'date': trade_date,
                        'signal_id': signal.signal_id,
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'pattern_name': signal.pattern_name,
                    })
            except Exception:
                pass

        # Pattern 5: FII-DII divergence
        if dii_net is not None:
            if fii_net < -30_000 and dii_net > 20_000:
                all_signals.append({
                    'date': trade_date,
                    'signal_id': 'NSE_005',
                    'direction': 'NEUTRAL',
                    'confidence': 0.70,
                    'pattern_name': 'FII_DII_DIVERGENCE',
                })
            elif fii_net > 30_000 and dii_net < -20_000:
                all_signals.append({
                    'date': trade_date,
                    'signal_id': 'NSE_005',
                    'direction': 'NEUTRAL',
                    'confidence': 0.65,
                    'pattern_name': 'FII_DII_DIVERGENCE',
                })

        # Pattern 6: Extreme positioning (contrarian)
        if percentile <= 5:
            all_signals.append({
                'date': trade_date,
                'signal_id': 'NSE_006',
                'direction': 'BULLISH',
                'confidence': 0.60,
                'pattern_name': 'FII_EXTREME_POSITIONING_CONTRARIAN',
            })
        elif percentile >= 95:
            all_signals.append({
                'date': trade_date,
                'signal_id': 'NSE_006',
                'direction': 'BEARISH',
                'confidence': 0.60,
                'pattern_name': 'FII_EXTREME_POSITIONING_CONTRARIAN',
            })

    return all_signals


# ================================================================
# PATTERN FREQUENCY VALIDATION
# ================================================================

def validate_pattern_frequency(
    signals: List[Dict],
    months: int,
    min_per_month: float = 2.0,
    max_per_month: float = 5.0,
) -> Dict:
    """
    Validate that each pattern fires at a reasonable frequency.

    Expected: 2-5 signals per pattern per month.
    Flags: TOO_FEW (<2/month) or TOO_MANY (>5/month).

    Returns dict with per-pattern stats and overall health.
    """
    if not signals:
        return {
            'status': 'NO_SIGNALS',
            'patterns': {},
            'total_signals': 0,
            'months': months,
        }

    sig_df = pd.DataFrame(signals)
    pattern_counts = sig_df.groupby('pattern_name').size().to_dict()

    results = {}
    all_ok = True

    for pattern, count in pattern_counts.items():
        rate = count / max(months, 1)
        if rate < min_per_month:
            status = 'TOO_FEW'
            all_ok = False
        elif rate > max_per_month:
            status = 'TOO_MANY'
            all_ok = False
        else:
            status = 'OK'

        results[pattern] = {
            'count': count,
            'rate_per_month': round(rate, 2),
            'status': status,
        }

    return {
        'status': 'HEALTHY' if all_ok else 'NEEDS_REVIEW',
        'patterns': results,
        'total_signals': len(signals),
        'months': months,
    }


# ================================================================
# BACKFILL MAIN
# ================================================================

def run_backfill(
    conn,
    months: int = 6,
    download: bool = True,
    validate_only: bool = False,
) -> Dict:
    """
    Run complete FII backfill pipeline.

    Steps:
        1. Download last N months of NSE participant OI (if download=True)
        2. Load all FII data from DB
        3. Compute derived metrics
        4. Run 6 patterns on history
        5. Validate pattern frequency
        6. Store derived metrics back to DB

    Returns summary dict.
    """
    result = {
        'months': months,
        'download_count': 0,
        'total_rows': 0,
        'signals_found': 0,
        'validation': {},
    }

    # Step 1: Download
    if download and not validate_only:
        logger.info(f"Downloading last {months} months of FII data from NSE...")
        downloader = FIIDataDownloader(db_conn=conn)
        try:
            downloaded = downloader.initial_load(months=months)
            result['download_count'] = len(downloaded)
            logger.info(f"Downloaded {len(downloaded)} days of FII data")
        except Exception as e:
            logger.error(f"Download phase failed: {e}")
            result['download_error'] = str(e)

    # Step 2: Load from DB
    try:
        df = pd.read_sql("""
            SELECT trade_date, fii_long_contracts, fii_short_contracts,
                   dii_long_contracts, dii_short_contracts,
                   z_score_20d, percentile_252d, flow_5d
            FROM fii_daily_metrics
            ORDER BY trade_date
        """, conn)
    except Exception as e:
        logger.error(f"DB load failed: {e}")
        return result

    if df.empty:
        logger.warning("No FII data in DB after download phase")
        return result

    # Join PCR data if available
    try:
        pcr_df = pd.read_sql("""
            SELECT date AS trade_date, pcr_oi
            FROM pcr_daily
        """, conn)
        if not pcr_df.empty:
            df = df.merge(pcr_df, on='trade_date', how='left')
    except Exception:
        pass

    result['total_rows'] = len(df)
    logger.info(f"Loaded {len(df)} rows from fii_daily_metrics")

    # Step 3: Compute derived metrics
    df = compute_derived_metrics(df)

    # Step 4: Run patterns
    signals = run_patterns_on_history(df, conn=conn)
    result['signals_found'] = len(signals)
    logger.info(f"Detected {len(signals)} signals across history")

    # Step 5: Validate frequency
    validation = validate_pattern_frequency(signals, months)
    result['validation'] = validation

    # Step 6: Store derived metrics back to DB
    if not validate_only:
        _store_derived_metrics(conn, df)

    return result


def _store_derived_metrics(conn, df: pd.DataFrame):
    """Update fii_daily_metrics with computed derived columns."""
    cur = conn.cursor()
    updated = 0

    for _, row in df.iterrows():
        trade_date = row['trade_date']
        z_score = row.get('z_score_20d')
        percentile = row.get('percentile_252d')
        flow_5d = row.get('flow_5d')

        if pd.isna(z_score) and pd.isna(percentile) and pd.isna(flow_5d):
            continue

        try:
            cur.execute("""
                UPDATE fii_daily_metrics
                SET z_score_20d = %s,
                    percentile_252d = %s,
                    flow_5d = %s
                WHERE trade_date = %s
            """, (
                float(z_score) if pd.notna(z_score) else None,
                float(percentile) if pd.notna(percentile) else None,
                float(flow_5d) if pd.notna(flow_5d) else None,
                trade_date,
            ))
            updated += 1
        except Exception as e:
            logger.debug(f"Update failed for {trade_date}: {e}")
            conn.rollback()
            continue

    try:
        conn.commit()
    except Exception:
        conn.rollback()

    logger.info(f"Updated derived metrics for {updated} rows")


def show_status(conn):
    """Print FII backfill status."""
    cur = conn.cursor()

    print(f"\n{'='*70}")
    print("  FII BACKFILL STATUS")
    print(f"{'='*70}")

    cur.execute("""
        SELECT count(*), min(trade_date), max(trade_date)
        FROM fii_daily_metrics
    """)
    row = cur.fetchone()
    count, min_date, max_date = row
    print(f"\n  Total rows: {count}")
    print(f"  Date range: {min_date} to {max_date}")

    if count > 0:
        staleness = (date.today() - max_date).days if max_date else 999
        status = "FRESH" if staleness <= 1 else ("STALE" if staleness <= 3 else "CRITICAL")
        print(f"  Freshness: {staleness}d ago [{status}]")

    # Derived metrics coverage
    cur.execute("""
        SELECT count(*) FROM fii_daily_metrics
        WHERE z_score_20d IS NOT NULL
    """)
    derived_count = cur.fetchone()[0]
    print(f"  Derived metrics: {derived_count}/{count} rows")

    # Recent data
    cur.execute("""
        SELECT trade_date, fii_long_contracts, fii_short_contracts,
               net_long_ratio, z_score_20d
        FROM fii_daily_metrics
        ORDER BY trade_date DESC LIMIT 5
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  Recent data:")
        print(f"  {'Date':<12s} {'Long':>12s} {'Short':>12s} {'Ratio':>8s} {'ZScore':>8s}")
        for r in rows:
            z = f"{r[4]:.2f}" if r[4] is not None else "—"
            ratio = f"{r[3]:.3f}" if r[3] is not None else "—"
            print(f"  {r[0]!s:<12s} {r[1]:>12,d} {r[2]:>12,d} {ratio:>8s} {z:>8s}")

    print(f"{'='*70}\n")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FII historical backfill and pattern validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  venv/bin/python3 -m fii.backfill --months 6          # download + backfill
  venv/bin/python3 -m fii.backfill --months 3 --validate-only
  venv/bin/python3 -m fii.backfill --status
  venv/bin/python3 -m fii.backfill --no-download       # compute from existing DB data
        """,
    )
    parser.add_argument(
        '--months', type=int, default=6,
        help='Number of months to backfill (default: 6)',
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Run validation on existing data only (no download)',
    )
    parser.add_argument(
        '--no-download', action='store_true',
        help='Skip download, compute from existing DB data',
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show backfill status',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    conn = None
    try:
        conn = psycopg2.connect(DATABASE_DSN)
        conn.autocommit = False
    except Exception as e:
        print(f"ERROR: DB connection failed: {e}")
        sys.exit(1)

    try:
        if args.status:
            show_status(conn)
            return

        result = run_backfill(
            conn,
            months=args.months,
            download=not args.no_download,
            validate_only=args.validate_only,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"  FII BACKFILL SUMMARY")
        print(f"{'='*60}")
        print(f"  Months: {result['months']}")
        print(f"  Downloaded: {result['download_count']} days")
        print(f"  Total rows: {result['total_rows']}")
        print(f"  Signals found: {result['signals_found']}")

        val = result.get('validation', {})
        print(f"\n  Validation: {val.get('status', 'N/A')}")
        for pattern, info in val.get('patterns', {}).items():
            status_marker = (
                '  ' if info['status'] == 'OK'
                else '!!' if info['status'] == 'TOO_MANY'
                else '??'
            )
            print(
                f"  {status_marker} {pattern:<40s} "
                f"count={info['count']:3d} "
                f"rate={info['rate_per_month']:.1f}/month "
                f"[{info['status']}]"
            )

        if val.get('status') == 'NEEDS_REVIEW':
            print(
                "\n  WARNING: Some patterns fire outside 2-5/month range."
                "\n  Review thresholds in fii/signal_detector.py."
            )

        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logger.critical(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if conn and not conn.closed:
            conn.close()


if __name__ == '__main__':
    main()
