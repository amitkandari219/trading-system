"""
Signal selector — 13-step decision tree run at 8:50 AM every trading day.
Produces an ordered list of signals approved for today.
"""

from datetime import timedelta

from config import settings
from regime_labeler import RegimeLabeler


class SignalSelector:
    """
    Runs at 8:50 AM every trading day.
    Produces: ordered list of signals approved for today.
    """

    # All from config.settings — scales with TOTAL_CAPITAL env var
    TOTAL_CAPITAL = settings.TOTAL_CAPITAL
    CAPITAL_RESERVE = settings.CAPITAL_RESERVE
    AVAILABLE_CAPITAL = settings.AVAILABLE_CAPITAL

    MAX_POSITIONS = 4
    MAX_SAME_DIRECTION = 2

    def __init__(self, db, redis_client, logger):
        """
        db:           database connection (psycopg2 or equivalent)
        redis_client: redis.Redis instance
        logger:       Python Logger or structured logger
        """
        self.db     = db
        self.redis  = redis_client
        self.logger = logger

    def run(self, today, current_portfolio, signal_registry):
        """
        Returns: List of approved signals in execution order.
        """

        # STEP 1: Get today's market regime
        regime = self.get_current_regime(today)

        # If CRISIS: no new trades
        if regime == 'CRISIS':
            return [], "CRISIS_REGIME_NO_TRADES"

        # STEP 2: Get economic calendar events
        events = self.get_todays_events(today)
        # events: ['RBI_DECISION', 'BUDGET', 'NIFTY_EXPIRY', etc.]

        # STEP 3: Get all ACTIVE signals from registry
        all_signals = signal_registry.get_active_signals()

        # STEP 3b: Load any preloaded FII overnight signals from Redis.
        # FII signals are written at 7:30 PM by FIIDailyPipeline.
        # Stale signals are already purged by _purge_stale_redis_signals().
        fii_signals = self._load_fii_overnight_signals(today)
        all_signals = all_signals + fii_signals

        # STEP 4: Filter by regime compatibility
        regime_compatible = [
            s for s in all_signals
            if regime in s.target_regimes
            or 'ANY' in s.target_regimes
        ]

        # STEP 5: Filter by calendar compatibility
        calendar_filtered = self.apply_calendar_filters(
            regime_compatible, events, today
        )

        # STEP 6: Filter by expiry week behavior
        expiry_filtered = self.apply_expiry_filters(
            calendar_filtered, today
        )

        # STEP 7: Filter by current portfolio state
        portfolio_filtered = self.apply_portfolio_filters(
            expiry_filtered, current_portfolio
        )

        # STEP 8: Separate PRIMARY and SECONDARY
        primaries = [s for s in portfolio_filtered
                     if s.classification == 'PRIMARY']
        secondaries = [s for s in portfolio_filtered
                       if s.classification == 'SECONDARY']

        # STEP 9: Rank by capital efficiency score
        # Score = Sharpe × Regime_Multiplier / Margin_Required
        primaries_ranked = self.rank_by_efficiency(primaries, regime)
        secondaries_ranked = self.rank_by_efficiency(secondaries, regime)

        # STEP 10: Allocate capital (primaries first)
        approved = []
        # Start from actual free capital: available minus already-deployed.
        deployed = getattr(current_portfolio, 'deployed_capital', 0) or 0
        remaining_capital = max(0, self.AVAILABLE_CAPITAL - deployed)
        current_positions = len(current_portfolio.open_positions)

        for signal in primaries_ranked:
            if current_positions >= self.MAX_POSITIONS:
                break
            if remaining_capital < signal.required_margin:
                continue  # Skip, not enough capital
            approved.append(signal)
            remaining_capital -= signal.required_margin
            current_positions += 1

        # STEP 11: Fill remaining capacity with secondaries
        for signal in secondaries_ranked:
            if current_positions >= self.MAX_POSITIONS:
                break
            if remaining_capital < signal.required_margin:
                continue
            approved.append(signal)
            remaining_capital -= signal.required_margin
            current_positions += 1

        # STEP 12: Final Greek compatibility check
        approved_with_greeks = self.check_greek_compatibility(
            approved, current_portfolio
        )

        # STEP 13: Assign execution priority
        final = self.assign_execution_priority(approved_with_greeks)

        return final, f"APPROVED_{len(final)}_SIGNALS"

    def apply_calendar_filters(self, signals, events, today):
        """
        Remove signals that conflict with today's events.
        EVENT classification is determined by signal_category == 'EVENT'.
        signal.source ('BOOK'|'FII_OVERNIGHT') is unrelated to calendar gating.
        """
        filtered = []
        for signal in signals:

            # On RBI day: only allow event-specific signals
            if 'RBI_DECISION' in events:
                if signal.signal_category == 'EVENT':
                    filtered.append(signal)
                elif signal.avoid_rbi_day:
                    continue  # Skip this signal today
                else:
                    filtered.append(signal)

            # On expiry day: check signal's expiry behavior
            elif 'NIFTY_EXPIRY' in events:
                if signal.expiry_week_behavior == 'AVOID_EXPIRY_WEEK':
                    continue
                else:
                    filtered.append(signal)

            # Normal day: all signals pass calendar filter
            else:
                filtered.append(signal)

        return filtered

    def apply_expiry_filters(self, signals, today):
        """
        Handle Monday-before-expiry gamma trap.
        """
        days_to_expiry = self.get_days_to_next_expiry(today)
        is_expiry_monday = (
            today.weekday() == 0 and  # Monday
            days_to_expiry == 1       # Tomorrow is expiry Tuesday
        )

        if not is_expiry_monday:
            return signals  # No expiry filter needed

        filtered = []
        for signal in signals:
            behavior = signal.expiry_week_behavior

            if behavior == 'AVOID_MONDAY':
                continue  # Skip today

            elif behavior == 'REDUCE_SIZE_MONDAY':
                # Keep signal but flag for 50% size
                signal_copy = signal.copy()
                signal_copy.size_multiplier = 0.5
                filtered.append(signal_copy)

            else:  # NORMAL or AVOID_EXPIRY_WEEK (already filtered)
                filtered.append(signal)

        return filtered

    def apply_portfolio_filters(self, signals, portfolio):
        """
        Remove signals that would create concentration.
        """
        current_long_count = len([
            p for p in portfolio.open_positions
            if p.direction == 'LONG'
        ])
        current_short_count = len([
            p for p in portfolio.open_positions
            if p.direction == 'SHORT'
        ])

        filtered = []
        for signal in signals:
            # Check directional concentration
            if (signal.direction == 'LONG' and
                    current_long_count >= self.MAX_SAME_DIRECTION):
                continue  # Too many longs already

            if (signal.direction == 'SHORT' and
                    current_short_count >= self.MAX_SAME_DIRECTION):
                continue  # Too many shorts already

            filtered.append(signal)

        return filtered

    def rank_by_efficiency(self, signals, regime):
        """
        Score = Sharpe × Regime_Multiplier × (1 / Margin_Required)
        """
        REGIME_MULTIPLIERS = {
            'TRENDING':  {'TREND': 1.3, 'REVERSION': 0.7,
                          'VOL': 1.0, 'PATTERN': 1.1, 'EVENT': 1.0},
            'RANGING':   {'TREND': 0.7, 'REVERSION': 1.3,
                          'VOL': 1.2, 'PATTERN': 0.9, 'EVENT': 1.0},
            'HIGH_VOL':  {'TREND': 0.8, 'REVERSION': 0.8,
                          'VOL': 1.4, 'PATTERN': 0.7, 'EVENT': 1.2},
        }

        multipliers = REGIME_MULTIPLIERS.get(regime, {})

        for signal in signals:
            regime_mult = multipliers.get(signal.signal_category, 1.0)
            if signal.required_margin <= 0:
                continue
            signal.efficiency_score = (
                signal.sharpe_ratio *
                regime_mult *
                (self.TOTAL_CAPITAL / signal.required_margin)
            )

        return sorted(signals,
                      key=lambda s: s.efficiency_score,
                      reverse=True)

    def get_current_regime(self, today) -> str:
        """
        Fetch today's pre-computed regime label from the DB.
        RegimeLabeler runs at 9:00 PM the prior night (after FII pipeline)
        and writes to the regime_labels table keyed by date.
        At 8:50 AM this record already exists.
        """
        row = self.db.execute(
            "SELECT regime FROM regime_labels WHERE label_date = %s",
            (today,)
        ).fetchone()
        if row:
            return row['regime']
        # Fallback: compute on the fly if regime_labels table not yet populated.
        from data.nifty_loader import load_nifty_history
        history = load_nifty_history(self.db, days=200)
        labeler = RegimeLabeler()
        return labeler.label_single_day(history, today)

    def get_todays_events(self, today) -> list:
        """
        Returns list of special events for today from the economic_calendar table.
        """
        rows = self.db.execute(
            "SELECT event_type FROM economic_calendar WHERE event_date = %s",
            (today,)
        ).fetchall()
        events = [r['event_type'] for r in rows]

        # Always auto-detect expiry day (every Thursday = weekly Nifty expiry)
        if self._is_expiry_day(today):
            events.append('NIFTY_EXPIRY')

        return events

    def get_days_to_next_expiry(self, today) -> int:
        """
        Returns calendar days to the next Nifty expiry Thursday.
        """
        for i in range(1, 8):
            candidate = today + timedelta(days=i)
            if candidate.weekday() == 3:  # Thursday
                return i
        return 7

    def _is_expiry_day(self, today) -> bool:
        """Returns True if today is a Thursday (Nifty expiry)."""
        return today.weekday() == 3

    # ================================================================
    # SIGNAL ID PREFIX TABLE
    # ================================================================
    SIGNAL_ID_PREFIXES = {
        'GUJRAL':      'GUJ',
        'KAUFMAN':     'KAU',
        'NATENBERG':   'NAT',
        'SINCLAIR':    'SIN',
        'GRIMES':      'GRI',
        'LOPEZ':       'LOP',
        'HILPISCH':    'HIL',
        'DOUGLAS':     'DOU',
        'MCMILLAN':    'MCM',
        'AUGEN':       'AUG',
        'HARRIS':      'HAR',
        'NSE_EMPIRICAL': 'NSE',
    }

    @classmethod
    def make_signal_id(cls, book_id: str, sequence: int) -> str:
        """
        Generate a signal_id for a newly extracted signal.
        e.g. make_signal_id('GUJRAL', 7) → 'GUJ_007'
        """
        prefix = cls.SIGNAL_ID_PREFIXES.get(book_id)
        if not prefix:
            raise ValueError(f"Unknown book_id '{book_id}'. "
                             f"Add to SIGNAL_ID_PREFIXES first.")
        return f"{prefix}_{sequence:03d}"

    def assign_execution_priority(self, signals):
        """
        Assigns execution order (1 = execute first).
        """
        for i, signal in enumerate(signals):
            signal.execution_priority = i + 1
        return signals

    def _load_fii_overnight_signals(self, today) -> list:
        """
        Read FII preloaded signals from Redis SIGNAL_QUEUE_PRELOADED.
        Returns list of Signal objects for signals valid today.
        """
        from portfolio.signal_model import Signal

        messages = self.redis.xrange('SIGNAL_QUEUE_PRELOADED', '-', '+')
        signals = []

        for msg_id, fields in messages:
            valid_until = fields.get('valid_until', '')
            if valid_until != str(today):
                continue

            try:
                sig = Signal.from_fii_redis(fields)
                signals.append(sig)
            except Exception as e:
                self.logger.warning(f"Malformed FII signal in Redis: {e}")

        return signals

    def check_greek_compatibility(self, approved_signals, current_portfolio) -> list:
        """
        Pre-screen signals against Greek limits before execution queuing.
        Coarse early filter using existing portfolio Greeks from Redis cache.
        """
        if current_portfolio is None:
            return approved_signals

        try:
            current_delta = current_portfolio.greeks.delta
            current_vega  = current_portfolio.greeks.vega
            current_gamma = current_portfolio.greeks.gamma
            current_theta = current_portfolio.greeks.theta
        except AttributeError:
            return approved_signals

        NEAR_LIMIT_FRACTION = 0.85

        GREEK_LIMITS = {
            'max_portfolio_delta': 0.50,
            'max_portfolio_vega':  3000,
            'max_portfolio_gamma': 30000,
            'max_portfolio_theta': -5000,
        }

        portfolio_too_hot = (
            abs(current_delta) > GREEK_LIMITS['max_portfolio_delta'] * NEAR_LIMIT_FRACTION
            or abs(current_vega) > GREEK_LIMITS['max_portfolio_vega'] * NEAR_LIMIT_FRACTION
            or abs(current_gamma) > GREEK_LIMITS['max_portfolio_gamma'] * NEAR_LIMIT_FRACTION
            or current_theta < GREEK_LIMITS['max_portfolio_theta'] * NEAR_LIMIT_FRACTION
        )

        if portfolio_too_hot:
            safe_signals = [s for s in approved_signals
                            if s.instrument == 'FUTURES']
            self.logger.info(
                f"Portfolio greeks near limit — filtered to {len(safe_signals)} "
                f"futures-only signals from {len(approved_signals)} approved. "
                f"Δ={current_delta:.3f} V={current_vega:.0f} "
                f"Γ={current_gamma:.0f} Θ={current_theta:.0f}"
            )
            return safe_signals

        return approved_signals
