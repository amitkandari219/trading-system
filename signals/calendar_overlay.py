"""
Calendar overlay — event-driven position sizing adjustments.

Modifies trade sizing based on known calendar events (RBI MPC, Budget,
expiry weeks) and monthly seasonality computed from 10-year Nifty data.

Three event overlays:
  a. RBI_MPC_DRIFT:  T-2 reduce 30%, T no entries, T+1 to T+3 drift
  b. EXPIRY_WEEK:    widen SL 1.2x Mon-Wed, no entries Thu
  c. BUDGET_DAY:     T-5 reduce 40%, T no entries, T+1 to T+5 momentum

Monthly seasonality: +-10-15% modifier from 10yr Nifty return distribution.

Safety:
  - Calendar overlays reduce size, never increase beyond 1.2x
  - Event days block new entries, never force-exit existing positions
  - All modifiers are multiplicative and clamped to [0.3, 1.2]

Usage:
    from signals.calendar_overlay import CalendarOverlay
    overlay = CalendarOverlay(db_conn=conn)
    ctx = overlay.get_daily_context(date.today())
    modifier = overlay.get_size_modifier(date.today())
"""

import json
import logging
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to calendar events config
CALENDAR_EVENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'config', 'calendar_events.json',
)


class CalendarOverlay:
    """
    Event-driven and seasonal position sizing overlay.

    Computes a composite size modifier from:
      1. RBI MPC meeting proximity
      2. Expiry week adjustments
      3. Budget day proximity
      4. Monthly seasonality (10-year Nifty data)

    All modifiers are multiplicative and clamped to [0.3, 1.2].
    """

    # ================================================================
    # SIZE MODIFIER BOUNDS
    # ================================================================
    MIN_MODIFIER = 0.3
    MAX_MODIFIER = 1.2

    # ================================================================
    # RBI MPC DRIFT PARAMETERS
    # ================================================================
    MPC_T_MINUS_2_MODIFIER = 0.70     # T-2: reduce 30%
    MPC_T_DAY_MODIFIER = 0.0          # T: no new entries
    MPC_T_PLUS_1_MODIFIER = 0.80      # T+1: cautious resumption
    MPC_T_PLUS_2_MODIFIER = 0.90      # T+2: near-normal
    MPC_T_PLUS_3_MODIFIER = 1.0       # T+3: normal

    # ================================================================
    # EXPIRY WEEK PARAMETERS
    # ================================================================
    EXPIRY_SL_MULTIPLIER = 1.2        # Widen SL by 20% Mon-Wed of expiry week
    EXPIRY_THU_MODIFIER = 0.0         # No new entries on expiry Thursday
    EXPIRY_MON_WED_MODIFIER = 0.85    # Reduce size Mon-Wed of expiry week

    # ================================================================
    # BUDGET DAY PARAMETERS
    # ================================================================
    BUDGET_T_MINUS_5_MODIFIER = 0.60  # T-5 to T-1: reduce 40%
    BUDGET_T_DAY_MODIFIER = 0.0       # T: no new entries
    BUDGET_T_PLUS_1_TO_5_MODIFIER = 0.80  # T+1 to T+5: cautious momentum

    # ================================================================
    # SEASONALITY PARAMETERS
    # ================================================================
    SEASONALITY_MAX_BOOST = 0.15      # Max +-15% monthly adjustment
    SEASONALITY_LOOKBACK_YEARS = 10

    def __init__(self, db_conn=None, events_path: str = None):
        """
        Args:
            db_conn: psycopg2 connection for nifty_daily seasonality
            events_path: path to calendar_events.json (optional override)
        """
        self.conn = db_conn
        self._events_path = events_path or CALENDAR_EVENTS_PATH
        self._events: Optional[Dict] = None
        self._seasonality: Optional[Dict[int, float]] = None
        self._cache: Dict[str, Dict] = {}

    # ================================================================
    # PUBLIC: get_daily_context
    # ================================================================

    def get_daily_context(self, as_of: date) -> Dict:
        """
        Get comprehensive calendar context for a given date.

        Returns dict with:
            date: the date
            is_rbi_mpc: bool (is T-day of RBI MPC)
            rbi_mpc_modifier: float
            rbi_mpc_phase: str ('T-2', 'T', 'T+1', ..., None)
            is_expiry_day: bool
            is_expiry_week: bool
            expiry_modifier: float
            expiry_sl_multiplier: float
            is_budget_day: bool
            budget_modifier: float
            budget_phase: str ('T-5', 'T', 'T+1', ..., None)
            monthly_seasonality: float (-0.15 to +0.15)
            composite_modifier: float (product, clamped to [0.3, 1.2])
            block_new_entries: bool
            events_active: list of str (names of active events)
        """
        cache_key = str(as_of)
        if cache_key in self._cache:
            return self._cache[cache_key]

        events = self._load_events()

        # RBI MPC
        rbi = self._check_rbi_mpc(as_of, events)

        # Expiry
        expiry = self._check_expiry(as_of, events)

        # Budget
        budget = self._check_budget(as_of, events)

        # Seasonality
        seasonality_mod = self._get_monthly_seasonality(as_of.month)

        # Composite modifier (multiplicative)
        composite = rbi['modifier'] * expiry['modifier'] * budget['modifier']
        # Apply seasonality as additive adjustment: (1.0 + seasonal_pct)
        composite *= (1.0 + seasonality_mod)

        # Clamp
        composite = max(self.MIN_MODIFIER, min(self.MAX_MODIFIER, composite))

        # Block entries if any event says so
        block = (
            rbi['modifier'] == 0.0
            or expiry['modifier'] == 0.0
            or budget['modifier'] == 0.0
        )

        # If blocking, set modifier to 0 (caller should check block_new_entries)
        if block:
            composite = 0.0

        # Active events list
        events_active = []
        if rbi['phase']:
            events_active.append(f"RBI_MPC_{rbi['phase']}")
        if expiry['is_expiry_week']:
            tag = 'EXPIRY_DAY' if expiry['is_expiry_day'] else 'EXPIRY_WEEK'
            events_active.append(tag)
        if budget['phase']:
            events_active.append(f"BUDGET_{budget['phase']}")

        ctx = {
            'date': as_of,
            'is_rbi_mpc': rbi['is_mpc_day'],
            'rbi_mpc_modifier': rbi['modifier'],
            'rbi_mpc_phase': rbi['phase'],
            'is_expiry_day': expiry['is_expiry_day'],
            'is_expiry_week': expiry['is_expiry_week'],
            'expiry_modifier': expiry['modifier'],
            'expiry_sl_multiplier': expiry['sl_multiplier'],
            'is_budget_day': budget['is_budget_day'],
            'budget_modifier': budget['modifier'],
            'budget_phase': budget['phase'],
            'monthly_seasonality': seasonality_mod,
            'composite_modifier': composite,
            'block_new_entries': block,
            'events_active': events_active,
        }

        self._cache[cache_key] = ctx
        return ctx

    # ================================================================
    # PUBLIC: get_size_modifier
    # ================================================================

    def get_size_modifier(self, as_of: date) -> float:
        """
        Get the composite size modifier for a given date.

        Returns float in [0.0, 1.2]:
            0.0  = block new entries
            0.3  = minimum allowed sizing
            1.0  = normal
            1.2  = maximum boost (strong seasonal tailwind)
        """
        ctx = self.get_daily_context(as_of)
        return ctx['composite_modifier']

    # ================================================================
    # PUBLIC: get_sl_multiplier
    # ================================================================

    def get_sl_multiplier(self, as_of: date) -> float:
        """
        Get stop-loss multiplier for a date.
        Returns 1.2 during expiry week Mon-Wed, 1.0 otherwise.
        """
        ctx = self.get_daily_context(as_of)
        return ctx['expiry_sl_multiplier']

    # ================================================================
    # PUBLIC: is_entry_blocked
    # ================================================================

    def is_entry_blocked(self, as_of: date) -> bool:
        """Check if new entries should be blocked on this date."""
        ctx = self.get_daily_context(as_of)
        return ctx['block_new_entries']

    # ================================================================
    # PRIVATE: RBI MPC
    # ================================================================

    def _check_rbi_mpc(self, as_of: date, events: Dict) -> Dict:
        """
        Check RBI MPC proximity.

        Returns dict with: is_mpc_day, phase, modifier
        """
        mpc_dates = self._get_mpc_dates(events)
        result = {'is_mpc_day': False, 'phase': None, 'modifier': 1.0}

        for mpc_date in mpc_dates:
            delta = (as_of - mpc_date).days

            if delta == -2:
                result = {'is_mpc_day': False, 'phase': 'T-2', 'modifier': self.MPC_T_MINUS_2_MODIFIER}
                break
            elif delta == -1:
                result = {'is_mpc_day': False, 'phase': 'T-1', 'modifier': self.MPC_T_MINUS_2_MODIFIER}
                break
            elif delta == 0:
                result = {'is_mpc_day': True, 'phase': 'T', 'modifier': self.MPC_T_DAY_MODIFIER}
                break
            elif delta == 1:
                result = {'is_mpc_day': False, 'phase': 'T+1', 'modifier': self.MPC_T_PLUS_1_MODIFIER}
                break
            elif delta == 2:
                result = {'is_mpc_day': False, 'phase': 'T+2', 'modifier': self.MPC_T_PLUS_2_MODIFIER}
                break
            elif delta == 3:
                result = {'is_mpc_day': False, 'phase': 'T+3', 'modifier': self.MPC_T_PLUS_3_MODIFIER}
                break

        return result

    def _get_mpc_dates(self, events: Dict) -> List[date]:
        """Extract all RBI MPC dates from events config."""
        dates = []
        mpc_section = events.get('rbi_mpc_dates', {})
        for year_str, date_strs in mpc_section.items():
            if year_str.startswith('_'):
                continue
            if isinstance(date_strs, list):
                for ds in date_strs:
                    try:
                        dates.append(date.fromisoformat(ds))
                    except (ValueError, TypeError):
                        pass
        return sorted(dates)

    # ================================================================
    # PRIVATE: Expiry
    # ================================================================

    def _check_expiry(self, as_of: date, events: Dict) -> Dict:
        """
        Check if date is an expiry day or in an expiry week.

        Uses ExpiryDayDetector for accurate determination.
        """
        result = {
            'is_expiry_day': False,
            'is_expiry_week': False,
            'modifier': 1.0,
            'sl_multiplier': 1.0,
        }

        try:
            from signals.expiry_day_detector import ExpiryDayDetector
            detector = ExpiryDayDetector()
            info = detector.get_expiry_info(as_of)

            weekday = as_of.weekday()  # 0=Mon, 3=Thu

            if info['is_nifty_expiry'] or info['is_banknifty_expiry']:
                # Expiry day: no new entries
                result['is_expiry_day'] = True
                result['is_expiry_week'] = True
                result['modifier'] = self.EXPIRY_THU_MODIFIER
                result['sl_multiplier'] = 1.0  # Don't widen SL on expiry day itself
            elif info['days_to_nifty_expiry'] <= 3 and weekday < 3:
                # Mon-Wed of expiry week
                result['is_expiry_week'] = True
                result['modifier'] = self.EXPIRY_MON_WED_MODIFIER
                result['sl_multiplier'] = self.EXPIRY_SL_MULTIPLIER
        except Exception as e:
            logger.debug(f"ExpiryDayDetector unavailable: {e}")
            # Fallback: Thursday = possible expiry
            if as_of.weekday() == 3:
                result['is_expiry_day'] = True
                result['modifier'] = self.EXPIRY_THU_MODIFIER

        return result

    # ================================================================
    # PRIVATE: Budget
    # ================================================================

    def _check_budget(self, as_of: date, events: Dict) -> Dict:
        """
        Check Budget Day proximity.

        Budget day: no new entries
        T-5 to T-1: reduce 40%
        T+1 to T+5: cautious momentum follow
        """
        result = {'is_budget_day': False, 'phase': None, 'modifier': 1.0}

        budget_section = events.get('budget_dates', {})
        budget_dates = []
        for year_str, ds in budget_section.items():
            if year_str.startswith('_'):
                continue
            if isinstance(ds, str):
                try:
                    budget_dates.append(date.fromisoformat(ds))
                except (ValueError, TypeError):
                    pass

        for budget_date in sorted(budget_dates):
            delta = (as_of - budget_date).days

            if -5 <= delta <= -1:
                result = {
                    'is_budget_day': False,
                    'phase': f'T{delta}',
                    'modifier': self.BUDGET_T_MINUS_5_MODIFIER,
                }
                break
            elif delta == 0:
                result = {
                    'is_budget_day': True,
                    'phase': 'T',
                    'modifier': self.BUDGET_T_DAY_MODIFIER,
                }
                break
            elif 1 <= delta <= 5:
                result = {
                    'is_budget_day': False,
                    'phase': f'T+{delta}',
                    'modifier': self.BUDGET_T_PLUS_1_TO_5_MODIFIER,
                }
                break

        return result

    # ================================================================
    # PRIVATE: Monthly Seasonality
    # ================================================================

    def _get_monthly_seasonality(self, month: int) -> float:
        """
        Get monthly seasonality modifier from 10-year Nifty data.

        Computes average monthly return for each calendar month,
        then maps to a [-0.15, +0.15] modifier.

        Returns float: positive = seasonal tailwind, negative = headwind.
        """
        if self._seasonality is None:
            self._seasonality = self._compute_seasonality()

        return self._seasonality.get(month, 0.0)

    def _compute_seasonality(self) -> Dict[int, float]:
        """
        Compute monthly seasonality from nifty_daily table.

        Uses 10+ years of data. Returns dict mapping month -> modifier.
        """
        if self.conn is None:
            return self._default_seasonality()

        try:
            df = pd.read_sql("""
                SELECT date, close FROM nifty_daily
                ORDER BY date
            """, self.conn)
        except Exception as e:
            logger.warning(f"Failed to load nifty_daily for seasonality: {e}")
            return self._default_seasonality()

        if len(df) < 252:
            logger.warning("Insufficient data for seasonality computation")
            return self._default_seasonality()

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Monthly returns
        monthly = df['close'].resample('ME').last().pct_change().dropna()
        monthly_df = pd.DataFrame({
            'return': monthly,
            'month': monthly.index.month,
        })

        # Average return by month
        avg_by_month = monthly_df.groupby('month')['return'].mean()

        # Normalize: map to [-SEASONALITY_MAX_BOOST, +SEASONALITY_MAX_BOOST]
        overall_mean = avg_by_month.mean()
        overall_std = avg_by_month.std()

        result = {}
        for month in range(1, 13):
            if month in avg_by_month.index and overall_std > 0:
                z = (avg_by_month[month] - overall_mean) / overall_std
                # Clamp to [-1, 1] then scale
                z_clamped = max(-1.0, min(1.0, z))
                result[month] = round(z_clamped * self.SEASONALITY_MAX_BOOST, 4)
            else:
                result[month] = 0.0

        logger.info(
            f"Seasonality computed from {len(monthly)} months of data: "
            + ", ".join(f"M{m}={v:+.1%}" for m, v in sorted(result.items()))
        )

        return result

    @staticmethod
    def _default_seasonality() -> Dict[int, float]:
        """
        Hardcoded Nifty seasonality from empirical research.

        Based on 20-year Nifty50 analysis:
        - Strong months: Jan (+), Nov (+), Dec (+)
        - Weak months: Jun (-), Sep (-)
        - Neutral: rest
        """
        return {
            1:  0.10,   # January effect
            2:  0.00,   # Budget uncertainty
            3:  0.05,   # Year-end flows
            4:  0.05,   # New fiscal year
            5: -0.05,   # Sell in May
            6: -0.08,   # Monsoon uncertainty
            7:  0.03,   # Q1 results
            8:  0.00,   # Independence day
            9: -0.10,   # Historically weakest
            10: 0.05,   # Pre-festival buying
            11: 0.10,   # Diwali rally
            12: 0.08,   # Year-end window dressing
        }

    # ================================================================
    # PRIVATE: Load events
    # ================================================================

    def _load_events(self) -> Dict:
        """Load calendar events from JSON config."""
        if self._events is not None:
            return self._events

        try:
            with open(self._events_path, 'r') as f:
                self._events = json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Calendar events file not found: {self._events_path}. "
                f"Using empty config."
            )
            self._events = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in calendar events: {e}")
            self._events = {}

        return self._events
