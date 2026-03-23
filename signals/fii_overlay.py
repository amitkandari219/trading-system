"""
FII positioning overlay signal for Nifty trading.

Two modes of operation:
1. SIGNAL-BASED (new): loads active signals from fii_signal_results,
   provides per-signal sizing modifiers (concordant/conflicting).
2. RATIO-BASED (legacy): net long ratio thresholds for sizing.

This overlay modifies position sizing — it NEVER generates standalone trades.

Signal-based modifiers:
  Concordant (FII direction matches trade direction):  +1.3x
  Conflicting (FII direction opposes trade direction): 0.5x
  Neutral/no signal:                                   1.0x

Stale data (>1 day old): falls back to 1.0x with WARNING.

Ratio-based modifiers (legacy, used by enhanced_daily_run):
  FII net long ratio > 0.65:       1.2x (bullish positioning)
  FII net long ratio 0.45-0.65:    1.0x (neutral)
  FII net long ratio 0.30-0.45:    0.7x (cautious)
  FII net long ratio < 0.30:       0.4x (extremely bearish)

Data source: NSE daily participant-wise open interest data
URL: https://archives.nseindia.com/content/nsccl/fao_participant_oi_{DDMMYYYY}.csv
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FIIOverlay:
    """FII positioning-based position sizing overlay."""

    # ================================================================
    # RATIO-BASED THRESHOLDS (legacy mode)
    # ================================================================
    BULLISH_THRESHOLD = 0.65
    NEUTRAL_LOW = 0.45
    CAUTIOUS_LOW = 0.30

    MULTIPLIERS = {
        'BULLISH': 1.2,
        'NEUTRAL': 1.0,
        'CAUTIOUS': 0.7,
        'BEARISH': 0.4,
    }

    # ================================================================
    # SIGNAL-BASED MODIFIERS
    # ================================================================
    CONCORDANT_MODIFIER = 1.3     # FII agrees with trade direction
    CONFLICTING_MODIFIER = 0.5    # FII opposes trade direction
    NEUTRAL_MODIFIER = 1.0        # No signal or neutral
    DIVERGENCE_MODIFIER = 0.7     # NSE_005: FII-DII divergence
    STALE_MODIFIER = 1.0          # Stale data — no modification

    # PCR alignment modifiers
    PCR_ALIGNED_BOOST = 1.15          # FII + PCR agree → stronger signal
    PCR_DIVERGENT_DAMPEN = 0.85       # FII + PCR disagree → weaker signal

    # Max data staleness (days) before falling back to neutral
    MAX_STALENESS_DAYS = 1

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self._pcr_validator = None
        self._cache: Dict[str, Dict] = {}
        self._signal_cache: Dict[str, List[Dict]] = {}
        self._acted_signals: set = set()

    # ================================================================
    # SIGNAL-BASED API (new — used by intraday_runner)
    # ================================================================

    def load_active_signals(self, as_of_date: date = None) -> List[Dict]:
        """
        Load active FII signals from fii_signal_results where
        valid_for_date matches as_of_date.

        Returns list of dicts with:
            signal_id, direction, confidence, pattern_name, notes,
            data_date, valid_for_date
        """
        as_of_date = as_of_date or date.today()
        cache_key = str(as_of_date)

        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        if not self.conn:
            return []

        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT signal_id, direction, confidence, pattern_name,
                       notes, data_date, valid_for_date
                FROM fii_signal_results
                WHERE valid_for_date = %s
                  AND signal_id IS NOT NULL
                ORDER BY confidence DESC
            """, (as_of_date,))
            rows = cur.fetchall()

            signals = []
            for r in rows:
                signals.append({
                    'signal_id': r[0],
                    'direction': r[1],
                    'confidence': float(r[2]) if r[2] else 0.0,
                    'pattern_name': r[3],
                    'notes': r[4],
                    'data_date': r[5],
                    'valid_for_date': r[6],
                })

            # Check staleness
            if signals:
                data_date = signals[0].get('data_date')
                if data_date and (as_of_date - data_date).days > self.MAX_STALENESS_DAYS:
                    logger.warning(
                        f"FII signals stale: data from {data_date} "
                        f"(>{self.MAX_STALENESS_DAYS} day old). "
                        f"Using {self.STALE_MODIFIER}x modifier."
                    )
                    # Mark all as stale
                    for sig in signals:
                        sig['stale'] = True

            self._signal_cache[cache_key] = signals
            return signals

        except Exception as e:
            logger.debug(f"load_active_signals failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return []

    def get_sizing_modifier(self, signal_id: str,
                            trade_direction: str,
                            as_of_date: date = None) -> float:
        """
        Get sizing modifier for a specific FII signal relative to
        the trade direction.

        Args:
            signal_id: FII signal ID (e.g. 'NSE_001')
            trade_direction: 'LONG' or 'SHORT'
            as_of_date: date to check signals for

        Returns:
            float modifier: 1.3 (concordant), 0.5 (conflicting),
                           0.7 (divergence), 1.0 (neutral/stale)
        """
        as_of_date = as_of_date or date.today()
        signals = self.load_active_signals(as_of_date)

        if not signals:
            return self.NEUTRAL_MODIFIER

        # Find the requested signal
        target_signal = None
        for sig in signals:
            if sig['signal_id'] == signal_id:
                target_signal = sig
                break

        if target_signal is None:
            # No specific signal found — use strongest directional signal
            for sig in signals:
                if sig['direction'] in ('BULLISH', 'BEARISH'):
                    target_signal = sig
                    break

        if target_signal is None:
            return self.NEUTRAL_MODIFIER

        # Stale data check
        if target_signal.get('stale'):
            logger.warning(
                f"FII signal {signal_id} is stale — "
                f"returning {self.STALE_MODIFIER}x"
            )
            return self.STALE_MODIFIER

        # Divergence signal always returns reduced modifier
        if target_signal['pattern_name'] == 'FII_DII_DIVERGENCE':
            return self.DIVERGENCE_MODIFIER

        # Check concordance
        fii_direction = target_signal['direction']
        trade_dir_upper = trade_direction.upper()

        if fii_direction == 'NEUTRAL':
            return self.NEUTRAL_MODIFIER

        # Map FII direction to trade direction
        fii_is_bullish = fii_direction == 'BULLISH'
        trade_is_long = trade_dir_upper == 'LONG'

        if fii_is_bullish == trade_is_long:
            # Concordant: FII bullish + trade long, or FII bearish + trade short
            return self.CONCORDANT_MODIFIER
        else:
            # Conflicting: FII bearish + trade long, or FII bullish + trade short
            return self.CONFLICTING_MODIFIER

    def get_direction_bias(self, as_of_date: date = None) -> Dict:
        """
        Get the strongest FII directional bias for today.

        Returns:
            {
                'direction': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'confidence': float,
                'signal_id': str or None,
                'pattern_name': str or None,
            }
        """
        as_of_date = as_of_date or date.today()
        signals = self.load_active_signals(as_of_date)

        if not signals:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'signal_id': None,
                'pattern_name': None,
            }

        # Find strongest directional signal (highest confidence)
        for sig in signals:
            if sig['direction'] in ('BULLISH', 'BEARISH'):
                return {
                    'direction': sig['direction'],
                    'confidence': sig['confidence'],
                    'signal_id': sig['signal_id'],
                    'pattern_name': sig['pattern_name'],
                }

        return {
            'direction': 'NEUTRAL',
            'confidence': signals[0]['confidence'] if signals else 0.0,
            'signal_id': signals[0]['signal_id'] if signals else None,
            'pattern_name': signals[0]['pattern_name'] if signals else None,
        }

    def mark_acted(self, signal_id: str, as_of_date: date = None):
        """
        Mark a signal as acted upon to prevent double-application
        within the same session.
        """
        as_of_date = as_of_date or date.today()
        key = f"{signal_id}:{as_of_date}"

        if key in self._acted_signals:
            return  # already marked

        self._acted_signals.add(key)

        # Also update DB
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute("""
                    UPDATE fii_signal_results
                    SET was_executed = TRUE
                    WHERE signal_id = %s
                      AND valid_for_date = %s
                """, (signal_id, as_of_date))
                self.conn.commit()
            except Exception as e:
                logger.debug(f"mark_acted DB update failed: {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def is_acted(self, signal_id: str, as_of_date: date = None) -> bool:
        """Check if a signal has already been acted upon."""
        as_of_date = as_of_date or date.today()
        return f"{signal_id}:{as_of_date}" in self._acted_signals

    # ================================================================
    # PCR-AWARE SIZING (FII + PCR alignment)
    # ================================================================

    def get_sizing_modifier_with_pcr(
        self, signal_id: str, trade_direction: str,
        as_of_date: date = None,
    ) -> Dict:
        """
        Get sizing modifier that combines FII signal with PCR regime.

        When FII and PCR align (both bullish or both bearish), the
        modifier is boosted. When they diverge, it's dampened.

        Returns dict with:
            fii_modifier: float (base FII modifier)
            pcr_modifier: float (PCR alignment modifier)
            combined_modifier: float (product, clamped)
            pcr_regime: str
            aligned: bool
        """
        as_of_date = as_of_date or date.today()

        # Base FII modifier
        fii_mod = self.get_sizing_modifier(signal_id, trade_direction, as_of_date)

        # PCR alignment
        pcr_mod = 1.0
        pcr_regime = 'UNKNOWN'
        aligned = False

        try:
            if self._pcr_validator is None:
                from data.pcr_validator import PCRValidator
                self._pcr_validator = PCRValidator(db_conn=self.conn)

            fii_bias = self.get_direction_bias(as_of_date)
            fii_direction = fii_bias.get('direction', 'NEUTRAL')

            alignment = self._pcr_validator.check_fii_alignment(
                as_of_date, fii_direction
            )
            pcr_mod = alignment.get('modifier', 1.0)
            pcr_regime = alignment.get('pcr_regime', 'UNKNOWN')
            aligned = alignment.get('aligned', False)

        except Exception as e:
            logger.debug(f"PCR alignment check failed: {e}")

        # Combined modifier (clamped to safe range)
        combined = fii_mod * pcr_mod
        combined = max(0.3, min(1.5, combined))

        return {
            'fii_modifier': fii_mod,
            'pcr_modifier': pcr_mod,
            'combined_modifier': combined,
            'pcr_regime': pcr_regime,
            'aligned': aligned,
        }

    # ================================================================
    # RATIO-BASED API (legacy — used by enhanced_daily_run)
    # ================================================================

    def get_multiplier(self, as_of: date = None) -> Dict:
        """
        Get FII positioning multiplier based on net long ratio.
        Legacy API used by enhanced_daily_run.py.

        Returns:
            {
                'multiplier': float (0.4 - 1.2),
                'regime': str ('BULLISH', 'NEUTRAL', 'CAUTIOUS', 'BEARISH'),
                'net_long_ratio': float (0-1),
                'fii_long_contracts': int,
                'fii_short_contracts': int,
                'data_date': str,
                'data_source': str,
            }
        """
        as_of = as_of or date.today()
        cache_key = f"ratio:{as_of}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try DB (fii_daily_metrics)
        fii_data = self._load_from_db(as_of)

        if fii_data is None:
            # Try fetching from NSE
            fii_data = self._fetch_from_nse(as_of)

        if fii_data is None:
            logger.info("FII data unavailable — using neutral multiplier")
            result = {
                'multiplier': 1.0,
                'regime': 'NEUTRAL',
                'net_long_ratio': 0.5,
                'fii_long_contracts': 0,
                'fii_short_contracts': 0,
                'data_date': str(as_of),
                'data_source': 'fallback',
            }
            self._cache[cache_key] = result
            return result

        # Compute net long ratio
        long_c = fii_data.get('fii_long_contracts', 0)
        short_c = fii_data.get('fii_short_contracts', 0)
        total = long_c + short_c

        if total == 0:
            net_long_ratio = 0.5
        else:
            net_long_ratio = long_c / total

        # Determine regime and multiplier
        regime = self._ratio_to_regime(net_long_ratio)
        multiplier = self.MULTIPLIERS[regime]

        result = {
            'multiplier': multiplier,
            'regime': regime,
            'net_long_ratio': round(net_long_ratio, 3),
            'fii_long_contracts': long_c,
            'fii_short_contracts': short_c,
            'data_date': str(fii_data.get('data_date', as_of)),
            'data_source': fii_data.get('source', 'db'),
        }

        logger.info(
            f"FII overlay: {regime} ({multiplier}x) — "
            f"net_long_ratio={net_long_ratio:.3f} "
            f"(L={long_c:,} S={short_c:,})"
        )

        self._cache[cache_key] = result
        return result

    def _load_from_db(self, as_of: date) -> Optional[Dict]:
        """Load FII data from fii_daily_metrics table."""
        if not self.conn:
            return None

        try:
            cur = self.conn.cursor()
            # Try the schema.sql version (date PK, fut_long/fut_short)
            cur.execute("""
                SELECT date, fut_long, fut_short
                FROM fii_daily_metrics
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT 1
            """, (as_of,))
            row = cur.fetchone()

            if row is None:
                return None

            data_date = row[0]
            # Check freshness (max 7 days for weekends/holidays)
            if (as_of - data_date).days > 7:
                logger.warning(
                    f"FII data stale: latest is {data_date} (>7 days)"
                )
                return None

            return {
                'data_date': data_date,
                'fii_long_contracts': int(float(row[1] or 0)),
                'fii_short_contracts': int(float(row[2] or 0)),
                'source': 'db',
            }
        except Exception as e:
            logger.debug(f"FII DB load failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None

    def _fetch_from_nse(self, as_of: date) -> Optional[Dict]:
        """
        Fetch FII participant OI from NSE archives.
        Best-effort — may fail if NSE blocks.
        """
        try:
            import requests
            from io import StringIO
        except ImportError:
            return None

        for offset in range(5):
            check_date = as_of - timedelta(days=offset)
            date_str = check_date.strftime('%d%b%Y')
            url = (
                f"https://archives.nseindia.com/content/nsccl/"
                f"fao_participant_oi_{date_str}.csv"
            )

            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'text/csv',
                    'Referer': 'https://www.nseindia.com/',
                }
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue

                df = pd.read_csv(StringIO(resp.text))

                # Find FII/FPI row
                fii_row = df[
                    df.iloc[:, 0].astype(str).str.strip().str.upper().isin(
                        ['FII', 'FPI']
                    )
                ]

                if fii_row.empty:
                    fii_row = df[
                        df.apply(
                            lambda r: 'FII' in str(r.values).upper()
                            or 'FPI' in str(r.values).upper(),
                            axis=1
                        )
                    ]

                if not fii_row.empty:
                    row = fii_row.iloc[0]
                    long_c = int(float(
                        str(row.iloc[2]).replace(',', '')
                    )) if len(row) > 2 else 0
                    short_c = int(float(
                        str(row.iloc[4]).replace(',', '')
                    )) if len(row) > 4 else 0

                    result = {
                        'data_date': check_date,
                        'fii_long_contracts': abs(long_c),
                        'fii_short_contracts': abs(short_c),
                        'source': 'nse_archive',
                    }

                    self._store_to_db(result)
                    return result

            except Exception as e:
                logger.debug(f"NSE fetch failed for {date_str}: {e}")
                continue

        return None

    def _store_to_db(self, data: Dict) -> None:
        """Store fetched FII data to DB for caching."""
        if not self.conn:
            return
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO fii_daily_metrics
                    (date, fut_long, fut_short, fut_net)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    fut_long = EXCLUDED.fut_long,
                    fut_short = EXCLUDED.fut_short,
                    fut_net = EXCLUDED.fut_net
            """, (
                data['data_date'],
                data['fii_long_contracts'],
                data['fii_short_contracts'],
                data['fii_long_contracts'] - data['fii_short_contracts'],
            ))
            self.conn.commit()
        except Exception as e:
            logger.debug(f"FII DB store failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass

    def get_historical_regime(self, start: date, end: date) -> pd.DataFrame:
        """Get historical FII regime labels for backtesting."""
        if not self.conn:
            return pd.DataFrame()

        try:
            df = pd.read_sql("""
                SELECT date as trade_date, fut_long as fii_long_contracts,
                       fut_short as fii_short_contracts
                FROM fii_daily_metrics
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, self.conn, params=(start, end))

            if df.empty:
                return df

            df['net_long_ratio'] = df['fii_long_contracts'] / (
                df['fii_long_contracts'] + df['fii_short_contracts']
            )
            df['fii_regime'] = df['net_long_ratio'].apply(self._ratio_to_regime)
            df['fii_multiplier'] = df['fii_regime'].map(self.MULTIPLIERS)

            return df
        except Exception as e:
            logger.warning(f"Historical FII data load failed: {e}")
            return pd.DataFrame()

    # ================================================================
    # FII/DII DIVERGENCE REGIME (strangle sizing booster)
    # ================================================================

    def get_divergence_regime(self, as_of: date = None) -> Dict:
        """
        Detect FII/DII divergence for strangle sizing.

        DIVERGENCE (DII buying + FII selling or vice versa) → range-bound
        market → premium selling paradise → boost strangle size 1.3x.

        Returns:
            {regime, dii_5d_net, fii_5d_net, strangle_boost}
        """
        as_of = as_of or date.today()

        try:
            cur = self.db_conn.cursor()
            cur.execute("""
                SELECT trade_date,
                       COALESCE(fii_long_contracts - fii_short_contracts, 0) AS fii_net,
                       COALESCE(dii_fut_long - dii_fut_short, 0) AS dii_net
                FROM fii_daily_metrics
                WHERE trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT 5
            """, (as_of,))
            rows = cur.fetchall()
            cur.close()
        except Exception as e:
            logger.debug(f"FII divergence query failed: {e}")
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return {
                'regime': 'NEUTRAL', 'dii_5d_net': 0, 'fii_5d_net': 0,
                'strangle_boost': 1.0, 'data_days': 0,
            }

        if len(rows) < 3:
            return {
                'regime': 'NEUTRAL', 'dii_5d_net': 0, 'fii_5d_net': 0,
                'strangle_boost': 1.0, 'data_days': len(rows),
            }

        fii_5d = sum(r[1] for r in rows)
        dii_5d = sum(r[2] for r in rows)

        # Divergence: DII buying (>8000) while FII selling (<-3000)
        if dii_5d > 8000 and fii_5d < -3000:
            regime = 'DIVERGENCE'
            boost = 1.3
        elif dii_5d < -8000 and fii_5d > 3000:
            regime = 'DIVERGENCE'
            boost = 1.3
        elif fii_5d > 5000 and dii_5d > 3000:
            regime = 'ALIGNED_BULL'
            boost = 0.8
        elif fii_5d < -5000 and dii_5d < -3000:
            regime = 'ALIGNED_BEAR'
            boost = 0.8
        else:
            regime = 'NEUTRAL'
            boost = 1.0

        return {
            'regime': regime,
            'dii_5d_net': round(dii_5d),
            'fii_5d_net': round(fii_5d),
            'strangle_boost': boost,
            'data_days': len(rows),
        }

    @classmethod
    def _ratio_to_regime(cls, ratio: float) -> str:
        """Convert net long ratio to regime string."""
        if ratio >= cls.BULLISH_THRESHOLD:
            return 'BULLISH'
        if ratio >= cls.NEUTRAL_LOW:
            return 'NEUTRAL'
        if ratio >= cls.CAUTIOUS_LOW:
            return 'CAUTIOUS'
        return 'BEARISH'
