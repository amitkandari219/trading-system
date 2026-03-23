"""
Trading Orchestrator -- daily and intraday session runner.

Connects: data fetch -> regime detection -> overlay evaluation ->
scoring signal evaluation -> conflict resolution -> sizing -> execution/logging.

Usage:
    from core.orchestrator import TradingOrchestrator
    orch = TradingOrchestrator(mode='paper', db_connection=conn)
    orch.run_daily_session()
"""

import importlib
import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, MAX_POSITIONS
except ImportError:
    DATABASE_DSN = 'postgresql://trader:trader123@localhost:5450/trading'
    NIFTY_LOT_SIZE = 25
    MAX_POSITIONS = 4

# ---------------------------------------------------------------------------
# Mapping: signal_name -> (module_path, class_name) for standalone signals
# Daily signals that live as methods on SignalComputer are handled separately
# via DailySignalWrapper.
# ---------------------------------------------------------------------------
SIGNAL_MODULE_MAP: Dict[str, tuple] = {
    # Structural / intraday signals with their own modules
    'MAX_OI_BARRIER':       ('signals.structural.max_oi_barrier', 'MaxOIBarrierSignal'),
    'THURSDAY_PIN_SETUP':   ('signals.structural.thursday_pin_setup', 'ThursdayPinSetupSignal'),
    'EXPIRY_PIN_FADE':      ('signals.intraday.expiry_scalper', 'ExpiryScalperSignal'),
    'ORR_REVERSION':        ('signals.intraday.orb_signal', 'ORBSignal'),
    'ID_GAMMA_BREAKOUT':    ('signals.intraday.microstructure', 'GammaBreakoutSignal'),
    'ID_GAMMA_REVERSAL':    ('signals.intraday.microstructure', 'GammaReversalSignal'),
}

# Daily signals are methods on paper_trading.signal_compute.SignalComputer.
# We wrap them so the orchestrator can call .evaluate() uniformly.
DAILY_SIGNAL_METHODS: Dict[str, str] = {
    'KAUFMAN_DRY_20':          '_check_entry_dry20',
    'KAUFMAN_DRY_16':          '_check_entry_dry16',
    'KAUFMAN_DRY_12':          '_check_entry_dry12',
    'GUJRAL_DRY_8':            '_check_entry_gujral8',
    'GUJRAL_DRY_13':           '_check_entry_gujral13',
    'BULKOWSKI_ADAM_AND_EVE_OR': '_check_entry_adam_eve',
}


# ---------------------------------------------------------------------------
# DailySignalWrapper
# ---------------------------------------------------------------------------
class DailySignalWrapper:
    """
    Wraps a method on SignalComputer so it looks like a standalone signal
    with an evaluate() interface.

    The underlying SignalComputer methods have the signature:
        _check_entry_xxx(signal_id, config, today_row, yesterday_row)
    and return a dict with entry info or None.
    """

    def __init__(self, signal_computer, signal_name: str, method_name: str, signal_config: Dict):
        self.computer = signal_computer
        self.signal_name = signal_name
        self.method_name = method_name
        self.signal_config = signal_config

    def evaluate(self, today_row, yesterday_row) -> Optional[Dict]:
        """
        Call the underlying SignalComputer method and return a
        standardised candidate dict or None.
        """
        method = getattr(self.computer, self.method_name, None)
        if method is None:
            logger.warning('Method %s not found on SignalComputer', self.method_name)
            return None
        try:
            result = method(self.signal_name, self.signal_config, today_row, yesterday_row)
        except Exception as exc:
            logger.error('Signal %s raised %s: %s', self.signal_name, type(exc).__name__, exc)
            return None

        if result is None:
            return None

        # Standardise into candidate dict
        direction_raw = result.get('direction', self.signal_config.get('direction', 'LONG'))
        direction = 'BULLISH' if direction_raw in ('LONG', 'BULLISH') else 'BEARISH'

        return {
            'signal_name': self.signal_name,
            'direction': direction,
            'confidence': result.get('confidence', 0.60),
            'entry_price': result.get('entry_price'),
            'stop_loss': result.get('stop_loss'),
            'sl_points': result.get('sl_points'),
            'take_profit': result.get('take_profit'),
            'metadata': result,
        }


# ---------------------------------------------------------------------------
# TradingOrchestrator
# ---------------------------------------------------------------------------
class TradingOrchestrator:
    """
    Coordinates a full trading session: data -> regime -> overlays ->
    signals -> coordinate -> size -> execute.

    Parameters
    ----------
    mode : str
        'paper' (log only) or 'live' (place real orders via KiteBridge).
    db_connection : optional
        psycopg2 connection. If None, connects using DATABASE_DSN.
    kite_session : optional
        Authenticated KiteConnect instance (required for mode='live').
    """

    def __init__(
        self,
        mode: str = 'paper',
        db_connection=None,
        kite_session=None,
    ):
        self.mode = mode.lower()
        self.kite_session = kite_session

        # ── Database ──────────────────────────────────────────────
        if db_connection is not None:
            self.conn = db_connection
        else:
            try:
                import psycopg2
                self.conn = psycopg2.connect(DATABASE_DSN)
            except Exception as exc:
                logger.warning('DB connection failed (%s), running without DB', exc)
                self.conn = None

        # ── Regime Detector ───────────────────────────────────────
        self.regime_detector = self._init_regime_detector()

        # ── Overlay Pipeline (lazy -- needs DataFrame) ────────────
        self.overlay_pipeline = None

        # ── Signal Coordinator ────────────────────────────────────
        try:
            from core.signal_coordinator import SignalCoordinator
            self.coordinator = SignalCoordinator(max_per_category=2)
        except ImportError:
            logger.warning('SignalCoordinator not available')
            self.coordinator = None

        # ── Unified Sizer ─────────────────────────────────────────
        self.sizer = self._init_sizer()

        # ── KiteBridge (live mode) ────────────────────────────────
        self.kite_bridge = None
        if self.mode == 'live' and self.kite_session is not None:
            self.kite_bridge = self._init_kite_bridge()

        # ── Loaded signal evaluators ──────────────────────────────
        self._signals: Dict[str, Any] = {}
        self._signal_computer = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_regime_detector(self):
        """Load RegimeLabeler (the existing regime detection engine)."""
        try:
            from regime_labeler import RegimeLabeler
            return RegimeLabeler()
        except ImportError:
            logger.warning('RegimeLabeler not available, regime will default to NEUTRAL')
            return None

    def _init_sizer(self):
        """Initialise UnifiedSizer with optional Kelly and Conviction."""
        kelly = None
        conviction = None
        try:
            from execution.adaptive_kelly import AdaptiveKelly
            kelly = AdaptiveKelly(base_fraction=0.75)
        except ImportError:
            logger.info('AdaptiveKelly not available, skipping Kelly step')

        try:
            from execution.conviction_scorer import ConvictionScorer
            conviction = ConvictionScorer()
        except ImportError:
            logger.info('ConvictionScorer not available, skipping conviction step')

        try:
            from core.unified_sizer import UnifiedSizer
            return UnifiedSizer(equity=1_000_000, kelly_engine=kelly, conviction_scorer=conviction)
        except ImportError:
            logger.error('UnifiedSizer import failed')
            return None

    def _init_kite_bridge(self):
        """Initialise KiteBridge for live execution."""
        try:
            from execution.kite_bridge import KiteBridge
            return KiteBridge(
                kite=self.kite_session,
                db=self.conn,
                alerter=None,
            )
        except ImportError:
            logger.warning('KiteBridge not available')
            return None

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def _load_signals(self, signal_list: List[str]) -> Dict[str, Any]:
        """
        Dynamically import signal evaluators.

        For signals in SIGNAL_MODULE_MAP: import the class and instantiate.
        For signals in DAILY_SIGNAL_METHODS: wrap the SignalComputer method.
        Unknown signals are skipped with a warning.
        """
        loaded: Dict[str, Any] = {}

        for name in signal_list:
            # Already loaded?
            if name in self._signals:
                loaded[name] = self._signals[name]
                continue

            # Standalone module signal
            if name in SIGNAL_MODULE_MAP:
                module_path, class_name = SIGNAL_MODULE_MAP[name]
                try:
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, class_name)
                    loaded[name] = cls()
                    logger.debug('Loaded signal %s from %s', name, module_path)
                except Exception as exc:
                    logger.warning('Failed to load signal %s: %s', name, exc)
                continue

            # Daily signal on SignalComputer
            if name in DAILY_SIGNAL_METHODS:
                computer = self._get_signal_computer()
                if computer is not None:
                    # Get signal config from paper_trading.signal_compute
                    config = self._get_signal_config(name)
                    wrapper = DailySignalWrapper(
                        signal_computer=computer,
                        signal_name=name,
                        method_name=DAILY_SIGNAL_METHODS[name],
                        signal_config=config,
                    )
                    loaded[name] = wrapper
                    logger.debug('Loaded daily signal %s via wrapper', name)
                continue

            logger.warning('Signal %s not found in any module map', name)

        self._signals.update(loaded)
        return loaded

    def _get_signal_computer(self):
        """Lazy-load the SignalComputer from paper_trading."""
        if self._signal_computer is not None:
            return self._signal_computer
        try:
            self.conn.rollback()  # Clear any pending transaction before SignalComputer init
            from paper_trading.signal_compute import SignalComputer
            self._signal_computer = SignalComputer(db_conn=self.conn)
            return self._signal_computer
        except Exception as exc:
            logger.error('Failed to load SignalComputer: %s', exc)
            return None

    @staticmethod
    def _get_signal_config(signal_name: str) -> Dict:
        """Fetch signal configuration dict from paper_trading.signal_compute."""
        try:
            from paper_trading.signal_compute import SIGNALS, SHADOW_SIGNALS, OVERLAY_SIGNALS
            return (
                SIGNALS.get(signal_name)
                or SHADOW_SIGNALS.get(signal_name)
                or OVERLAY_SIGNALS.get(signal_name)
                or {}
            )
        except ImportError:
            return {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _fetch_market_data(self, lookback_days: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch nifty_daily data from the database.

        Returns a DataFrame with columns: date, open, high, low, close,
        volume, india_vix, etc.  Returns None on failure.
        """
        if self.conn is None:
            logger.error('No DB connection available for data fetch')
            return None

        query = f"""
            SELECT *
            FROM nifty_daily
            ORDER BY date DESC
            LIMIT {lookback_days}
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.sort_values('date').reset_index(drop=True)
            logger.info('Fetched %d rows from nifty_daily', len(df))
            return df
        except Exception as exc:
            logger.error('Data fetch failed: %s', exc)
            return None

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        try:
            from backtest.indicators import add_all_indicators
            df = add_all_indicators(df)
        except ImportError:
            logger.warning('add_all_indicators not available')
        return df

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime using RegimeLabeler.
        Falls back to 'NEUTRAL' on any failure.
        """
        if self.regime_detector is None:
            return 'NEUTRAL'

        try:
            today = df['date'].iloc[-1]
            regime = self.regime_detector.label_single_day(df, today)
            logger.info('Regime for %s: %s', today, regime)
            return regime
        except Exception as exc:
            logger.warning('Regime detection failed: %s', exc)
            return 'NEUTRAL'

    # ------------------------------------------------------------------
    # Overlay evaluation
    # ------------------------------------------------------------------

    def _evaluate_overlays(self, df: pd.DataFrame, direction: str = 'LONG') -> Dict[str, float]:
        """
        Run the OverlayPipeline and return {overlay_id: modifier}.
        """
        try:
            from execution.overlay_pipeline import OverlayPipeline
            pipeline = OverlayPipeline(df)
            self.overlay_pipeline = pipeline
            trade_date = df['date'].iloc[-1]
            if hasattr(trade_date, 'date'):
                trade_date = trade_date.date()
            modifiers = pipeline.get_modifiers(trade_date, direction=direction)
            logger.info('Overlays evaluated: %d modifiers', len(modifiers))
            return modifiers
        except Exception as exc:
            logger.warning('Overlay evaluation failed: %s', exc)
            return {}

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _evaluate_scoring_signals(
        self, df: pd.DataFrame, signal_names: List[str],
    ) -> List[Dict]:
        """
        Evaluate each scoring signal and collect candidates that fire.
        """
        signals = self._load_signals(signal_names)
        candidates: List[Dict] = []

        if len(df) < 2:
            logger.warning('Not enough data rows for signal evaluation')
            return candidates

        today_row = df.iloc[-1]
        yesterday_row = df.iloc[-2]

        for name, evaluator in signals.items():
            try:
                if isinstance(evaluator, DailySignalWrapper):
                    result = evaluator.evaluate(today_row, yesterday_row)
                elif hasattr(evaluator, 'evaluate'):
                    # Standalone signals have varied signatures; try the
                    # simplest daily form first
                    try:
                        result = evaluator.evaluate(today_row, yesterday_row)
                    except TypeError:
                        result = None
                else:
                    result = None

                if result is not None:
                    candidates.append(result)
                    logger.info('Signal FIRED: %s direction=%s conf=%.2f',
                                result.get('signal_name'),
                                result.get('direction'),
                                result.get('confidence', 0))
            except Exception as exc:
                logger.error('Signal %s evaluation error: %s', name, exc)

        return candidates

    # ------------------------------------------------------------------
    # Daily session
    # ------------------------------------------------------------------

    def run_daily_session(
        self,
        signal_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Full daily session:
        1. Fetch data
        2. Add indicators
        3. Detect regime
        4. Evaluate overlays
        5. Evaluate scoring signals
        6. Filter by regime
        7. Resolve conflicts (SignalCoordinator)
        8. Size each candidate
        9. Log / execute

        Returns list of sized trade dicts.
        """
        logger.info('=== Daily Session Start (mode=%s) ===', self.mode)

        # Default signal list: all scoring signals
        if signal_list is None:
            signal_list = list(DAILY_SIGNAL_METHODS.keys())

        # Step 1-2: Data
        df = self._fetch_market_data()
        if df is None or df.empty:
            logger.error('No market data available, aborting session')
            return []
        df = self._add_indicators(df)

        # Step 3: Regime
        regime = self._detect_regime(df)

        # Step 4: Overlays
        overlay_modifiers = self._evaluate_overlays(df)

        # Step 5: Scoring signals
        candidates = self._evaluate_scoring_signals(df, signal_list)
        if not candidates:
            logger.info('No signals fired today')
            return []

        logger.info('%d raw candidates: %s', len(candidates),
                    [c['signal_name'] for c in candidates])

        # Step 6: Regime filter
        candidates = self._filter_by_regime(candidates, regime)

        # Step 7: Conflict resolution
        if self.coordinator is not None:
            candidates = self.coordinator.process(candidates)
        logger.info('%d candidates after coordination: %s', len(candidates),
                    [c['signal_name'] for c in candidates])

        # Step 8-9: Size and execute/log
        results = []
        spot_price = float(df['close'].iloc[-1])
        vix = float(df['india_vix'].iloc[-1]) if 'india_vix' in df.columns else 15.0
        adx = float(df['adx_14'].iloc[-1]) if 'adx_14' in df.columns else 20.0

        for cand in candidates:
            sized = self._size_and_dispatch(
                candidate=cand,
                spot_price=spot_price,
                overlay_modifiers=overlay_modifiers,
                regime=regime,
                vix=vix,
                adx=adx,
            )
            if sized is not None:
                results.append(sized)

        logger.info('=== Daily Session End: %d trades ===', len(results))
        return results

    # ------------------------------------------------------------------
    # Intraday check
    # ------------------------------------------------------------------

    def run_intraday_check(
        self,
        signal_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Lightweight intraday scan for time-sensitive signals (expiry plays,
        gap fades, OI barriers, etc.).

        Fetches fresh intraday data, evaluates only the given signal_list,
        and returns sized candidates.
        """
        logger.info('=== Intraday Check Start (mode=%s) ===', self.mode)

        if signal_list is None:
            signal_list = [
                'EXPIRY_PIN_FADE', 'ORR_REVERSION',
                'MAX_OI_BARRIER', 'THURSDAY_PIN_SETUP',
            ]

        # Fetch daily data for context (regime, overlays)
        df = self._fetch_market_data(lookback_days=60)
        if df is None or df.empty:
            logger.error('No data for intraday context')
            return []
        df = self._add_indicators(df)

        regime = self._detect_regime(df)
        overlay_modifiers = self._evaluate_overlays(df)

        # Load and evaluate intraday signals
        signals = self._load_signals(signal_list)
        candidates: List[Dict] = []

        spot_price = float(df['close'].iloc[-1])

        for name, evaluator in signals.items():
            try:
                # Intraday signals typically need current price + option chain
                # For now, we call evaluate with what we have
                if hasattr(evaluator, 'evaluate'):
                    result = None
                    if name in SIGNAL_MODULE_MAP:
                        try:
                            result = evaluator.evaluate(
                                trade_date=date.today(),
                                current_time=datetime.now().time(),
                                nifty_price=spot_price,
                                option_chain_snapshot=[],
                            )
                        except TypeError:
                            # Signal may have a different evaluate signature
                            pass

                    if result is not None:
                        # Standardise
                        direction_raw = result.get('direction', 'LONG')
                        direction = 'BULLISH' if direction_raw in ('LONG', 'BULLISH') else 'BEARISH'
                        candidates.append({
                            'signal_name': name,
                            'direction': direction,
                            'confidence': result.get('confidence', 0.60),
                            'entry_price': result.get('entry_price', spot_price),
                            'stop_loss': result.get('stop_loss'),
                            'sl_points': result.get('sl_points'),
                            'metadata': result,
                        })
            except Exception as exc:
                logger.error('Intraday signal %s error: %s', name, exc)

        # Coordinate and size
        if self.coordinator is not None:
            candidates = self.coordinator.process(candidates)

        vix = float(df['india_vix'].iloc[-1]) if 'india_vix' in df.columns else 15.0
        adx = float(df['adx_14'].iloc[-1]) if 'adx_14' in df.columns else 20.0

        results = []
        for cand in candidates:
            sized = self._size_and_dispatch(
                candidate=cand,
                spot_price=spot_price,
                overlay_modifiers=overlay_modifiers,
                regime=regime,
                vix=vix,
                adx=adx,
            )
            if sized is not None:
                results.append(sized)

        logger.info('=== Intraday Check End: %d trades ===', len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_regime(self, candidates: List[Dict], regime: str) -> List[Dict]:
        """
        Drop candidates that are not allowed in the current regime.
        Uses SIGNAL_REGIME_MATRIX from signals.regime_filter if available.
        """
        try:
            from signals.regime_filter import SIGNAL_REGIME_MATRIX
        except ImportError:
            return candidates

        filtered = []
        for cand in candidates:
            name = cand['signal_name']
            policy = SIGNAL_REGIME_MATRIX.get(name)
            if policy is None:
                # No policy defined -> allow
                filtered.append(cand)
                continue
            if policy.get(regime, True):
                filtered.append(cand)
            else:
                logger.info('Regime filter: dropping %s (regime=%s)', name, regime)

        return filtered

    def _size_and_dispatch(
        self,
        candidate: Dict,
        spot_price: float,
        overlay_modifiers: Dict[str, float],
        regime: str,
        vix: float = 15.0,
        adx: float = 20.0,
    ) -> Optional[Dict]:
        """
        Size a candidate and either log (paper) or execute (live).
        Returns a combined trade dict or None on failure.
        """
        signal_name = candidate['signal_name']
        direction = candidate['direction']

        # Determine stop loss in points
        sl_points = candidate.get('sl_points')
        if sl_points is None and candidate.get('stop_loss') and candidate.get('entry_price'):
            sl_points = abs(candidate['entry_price'] - candidate['stop_loss'])
        if sl_points is None or sl_points <= 0:
            # Fallback: use config stop_loss_pct
            config = self._get_signal_config(signal_name)
            sl_pct = config.get('stop_loss_pct', 0.02)
            sl_points = spot_price * sl_pct

        # Size
        if self.sizer is None:
            logger.error('No sizer available, cannot size %s', signal_name)
            return None

        sizing = self.sizer.compute(
            signal_name=signal_name,
            sl_points=sl_points,
            spot_price=spot_price,
            overlay_modifiers=overlay_modifiers,
            regime=regime,
            direction=direction,
            vix=vix,
            adx=adx,
        )

        trade = {
            'signal_name': signal_name,
            'direction': direction,
            'confidence': candidate.get('confidence', 0.0),
            'entry_price': candidate.get('entry_price', spot_price),
            'stop_loss_points': round(sl_points, 2),
            'lots': sizing['lots'],
            'risk_amount': sizing['risk_amount'],
            'sizing': sizing,
            'regime': regime,
            'timestamp': datetime.now().isoformat(),
        }

        # Dispatch
        if self.mode == 'paper':
            self._log_paper_trade(trade)
        elif self.mode == 'live':
            self._execute_live(trade)

        return trade

    def _log_paper_trade(self, trade: Dict):
        """Log trade to DB and console in paper mode."""
        logger.info(
            'PAPER TRADE: %s %s %d lots @ %.0f, SL=%.0f pts, risk=Rs%.0f',
            trade['signal_name'], trade['direction'], trade['lots'],
            trade['entry_price'], trade['stop_loss_points'], trade['risk_amount'],
        )

        if self.conn is not None:
            try:
                cur = self.conn.cursor()
                cur.execute("""
                    INSERT INTO paper_trades
                    (signal_name, direction, lots, entry_price, stop_loss_points,
                     risk_amount, regime, sizing_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    trade['signal_name'],
                    trade['direction'],
                    trade['lots'],
                    trade['entry_price'],
                    trade['stop_loss_points'],
                    trade['risk_amount'],
                    trade['regime'],
                    json.dumps(trade['sizing']),
                ))
                self.conn.commit()
            except Exception as exc:
                logger.warning('Paper trade DB insert failed: %s', exc)
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def _execute_live(self, trade: Dict):
        """Place order via KiteBridge in live mode."""
        if self.kite_bridge is None:
            logger.error('KiteBridge not available for live execution')
            return

        order_dict = {
            'tradingsymbol': 'NIFTY',
            'exchange': 'NFO',
            'transaction_type': 'BUY' if trade['direction'] == 'BULLISH' else 'SELL',
            'quantity': trade['lots'] * NIFTY_LOT_SIZE,
            'product': 'MIS',
            'order_type': 'MARKET',
            'tag': trade['signal_name'][:20],
        }

        try:
            order_id = self.kite_bridge.place_order(order_dict)
            logger.info('LIVE ORDER placed: %s order_id=%s', trade['signal_name'], order_id)
            trade['order_id'] = order_id
        except Exception as exc:
            logger.error('Live order failed for %s: %s', trade['signal_name'], exc)
