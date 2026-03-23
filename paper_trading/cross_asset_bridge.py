"""
Cross-asset overlay bridge for paper trading.
Fetches global macro data and computes position sizing overlays.
"""
import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache to avoid repeated API calls within same day
_cache: Dict[str, Dict] = {}


def get_cross_asset_multiplier(as_of: date = None) -> Dict:
    """
    Fetch cross-asset data and compute composite position sizing multiplier.

    Returns:
        {
            'composite_multiplier': float (0.2 - 2.0),
            'signals': {signal_name: {'mult': float, 'reason': str}, ...},
            'data_source': 'yfinance' | 'cached' | 'fallback',
            'timestamp': str,
        }
    """
    as_of = as_of or date.today()
    cache_key = str(as_of)

    if cache_key in _cache:
        result = _cache[cache_key].copy()
        result['data_source'] = 'cached'
        return result

    try:
        today_data, prev_data = _fetch_cross_asset_data(as_of)

        from signals.cross_asset_signals import CrossAssetOverlays
        overlays = CrossAssetOverlays()
        overlay_result = overlays.compute(today_data, prev_data)

        # Use pre-computed composite from CrossAssetOverlays
        composite = overlay_result['composite_multiplier']
        signals = overlay_result.get('signals', {})

        result = {
            'composite_multiplier': composite,
            'signals': signals,
            'data_source': 'yfinance',
            'timestamp': datetime.now().isoformat(),
        }
        _cache[cache_key] = result
        logger.info(f"Cross-asset composite multiplier: {composite:.3f}")
        for sig_name, sig_info in signals.items():
            if sig_info.get('mult', 1.0) != 1.0:
                logger.info(f"  {sig_name}: {sig_info['mult']:.2f}x ({sig_info.get('reason', '')})")

        return result

    except Exception as e:
        logger.warning(f"Cross-asset data fetch failed: {e} — trying DB cache fallback")

        # FIX 4: Try DB-cached rolling average instead of silent 1.0x
        db_fallback = _get_db_cached_multiplier(as_of)
        if db_fallback is not None:
            logger.info(f"Using DB-cached cross-asset multiplier: {db_fallback:.3f}")
            return {
                'composite_multiplier': db_fallback,
                'signals': {},
                'data_source': 'db_cache_fallback',
                'timestamp': datetime.now().isoformat(),
                'note': f'yfinance failed, using 7-day DB avg: {e}',
            }

        logger.warning(f"No DB cache available either — using neutral 1.0x")
        return {
            'composite_multiplier': 1.0,
            'signals': {},
            'data_source': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
        }


def _fetch_cross_asset_data(as_of: date) -> tuple:
    """
    Fetch cross-asset prices from yfinance.

    Returns:
        (today_data, prev_data) dicts keyed by instrument name
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — cross-asset overlays unavailable")
        raise

    tickers = {
        'USDINR': 'USDINR=X',
        'CRUDE_WTI': 'CL=F',
        'CRUDE_BRENT': 'BZ=F',
        'GOLD': 'GC=F',
        'SP500': '^GSPC',
        'VIX_US': '^VIX',
        'NIKKEI': '^N225',
        'HANGSENG': '^HSI',
    }

    # Fetch last 10 trading days to compute daily and weekly returns
    start = as_of - timedelta(days=15)
    end = as_of + timedelta(days=1)

    today_data = {}
    prev_data = {}

    for instrument, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start.isoformat(),
                             end=end.isoformat(), progress=False)
            if data.empty or len(data) < 2:
                logger.debug(f"No data for {ticker}")
                continue

            # Handle multi-level columns from yfinance
            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.get_level_values(0)

            latest = data.iloc[-1]
            prev = data.iloc[-2]
            week_ago = data.iloc[-6] if len(data) >= 6 else data.iloc[0]

            close = float(latest['Close'])
            prev_close = float(prev['Close'])
            week_close = float(week_ago['Close'])

            daily_return = (close - prev_close) / prev_close if prev_close else 0
            weekly_return = (close - week_close) / week_close if week_close else 0

            today_data[instrument] = {
                'close': close,
                'daily_return': daily_return,
                'weekly_return': weekly_return,
            }
            prev_data[instrument] = {
                'close': prev_close,
                'daily_return': daily_return,
                'weekly_return': weekly_return,
            }
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            continue

    if not today_data:
        raise RuntimeError("No cross-asset data available from any ticker")

    return today_data, prev_data


def persist_to_db(conn, as_of: date, result: Dict):
    """
    Persist today's cross-asset data to cross_asset_daily table.
    Table schema: (date, instrument, close, daily_return, weekly_return, ...).
    Call this after a successful fetch so future failures can use the DB cache.
    """
    if not conn or result.get('data_source') in ('fallback', 'db_cache_fallback'):
        return  # Don't persist fallback values

    try:
        signals = result.get('signals', {})
        # The cross_asset_daily table uses per-instrument rows, not a wide format.
        # We persist the composite_multiplier as a special "COMPOSITE" instrument row.
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cross_asset_daily (date, instrument, close, daily_return)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (date, instrument) DO UPDATE SET
                close = EXCLUDED.close, daily_return = EXCLUDED.daily_return
        """, (as_of, 'COMPOSITE_MULT', result['composite_multiplier'], 0.0))
        conn.commit()
        logger.debug(f"Persisted cross-asset multiplier {result['composite_multiplier']:.3f} for {as_of}")
    except Exception as e:
        conn.rollback()
        logger.debug(f"Failed to persist cross-asset data: {e}")


def _get_db_cached_multiplier(as_of: date, lookback_days: int = 7) -> Optional[float]:
    """
    Retrieve 7-day rolling average of composite multiplier from cross_asset_daily.
    Returns None if no DB connection or no recent data.
    """
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        conn = psycopg2.connect(DATABASE_DSN)
        cur = conn.cursor()
        cutoff = as_of - timedelta(days=lookback_days)
        cur.execute("""
            SELECT AVG(close)
            FROM cross_asset_daily
            WHERE date >= %s AND date <= %s
              AND instrument = 'COMPOSITE_MULT'
              AND close IS NOT NULL
        """, (cutoff, as_of))
        row = cur.fetchone()
        conn.close()

        if row and row[0] is not None:
            return round(float(row[0]), 3)
        return None
    except Exception as e:
        logger.debug(f"DB cache lookup failed: {e}")
        return None


def clear_cache():
    """Clear the daily cache (call at start of new trading day)."""
    global _cache
    _cache = {}
