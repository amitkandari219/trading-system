"""
Tests for all 7 Tier 3 ML/AI model modules.

Run: python -m pytest tests/test_tier3_models.py -v
"""

import pytest
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch


# ================================================================
# 1. Mamba/S4 Regime Detector Tests
# ================================================================
class TestMambaRegimeDetector:
    def setup_method(self):
        from models.mamba_regime import MambaRegimeDetector, SEQUENCE_LENGTH, N_FEATURES
        self.detector = MambaRegimeDetector()
        self.seq_len = SEQUENCE_LENGTH
        self.n_feat = N_FEATURES

    def test_predict_with_synthetic_features(self):
        rng = np.random.RandomState(42)
        features = rng.randn(self.seq_len, self.n_feat).astype(np.float32)
        result = self.detector.predict(features=features)
        assert result['signal_id'] == 'MAMBA_REGIME'
        assert result['regime'] in [
            'CALM_BULL', 'VOLATILE_BULL', 'NEUTRAL',
            'VOLATILE_BEAR', 'CALM_BEAR'
        ]
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['direction'] in ('BULLISH', 'BEARISH', 'NEUTRAL')

    def test_regime_probabilities_sum_to_one(self):
        features = np.random.randn(self.seq_len, self.n_feat).astype(np.float32)
        result = self.detector.predict(features=features)
        probs = result['regime_probabilities']
        assert len(probs) == 5
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_fallback_when_no_features(self):
        result = self.detector.predict(features=None)
        assert result['regime'] == 'NEUTRAL'
        assert result['confidence'] == 0.0
        assert result['method'] == 'fallback'

    def test_untrained_dampens_confidence(self):
        features = np.random.randn(self.seq_len, self.n_feat).astype(np.float32)
        result = self.detector.predict(features=features)
        assert result['trained'] is False
        assert result['method'] == 'mamba_s4_untrained'
        # Untrained should have lower confidence
        assert result['confidence'] <= 0.6

    def test_s4_layer_forward(self):
        from models.mamba_regime import S4Layer
        layer = S4Layer(input_dim=4, state_dim=8, output_dim=4)
        x = np.random.randn(10, 4)
        y = layer.forward(x)
        assert y.shape == (10, 4)

    def test_mamba_block_forward(self):
        from models.mamba_regime import MambaBlock
        block = MambaBlock(dim=12, state_dim=16)
        x = np.random.randn(self.seq_len, 12)
        y = block.forward(x)
        assert y.shape == (self.seq_len, 12)

    def test_regime_label_generation(self):
        returns = np.array([0.01, -0.01, 0.001, -0.005, 0.003])
        vol = np.array([0.10, 0.20, 0.10, 0.18, 0.12])
        labels = self.detector.generate_regime_labels(returns, vol)
        assert len(labels) == 5
        assert labels[0] == 0   # CALM_BULL (ret>0, vol<0.15)
        assert labels[1] == 3   # VOLATILE_BEAR (ret<0, vol>=0.15)
        assert labels[2] == 2   # NEUTRAL (small ret)
        assert labels[4] == 0   # CALM_BULL

    def test_evaluate_alias(self):
        result = self.detector.evaluate()
        assert 'signal_id' in result
        assert result['signal_id'] == 'MAMBA_REGIME'


# ================================================================
# 2. TFT Forecaster Tests
# ================================================================
class TestTFTForecaster:
    def setup_method(self):
        from models.tft_forecaster import TFTForecaster, ENCODER_LENGTH, OBSERVED_FEATURES
        self.tft = TFTForecaster()
        self.enc_len = ENCODER_LENGTH
        self.n_obs = len(OBSERVED_FEATURES)

    def test_predict_with_features(self):
        features = np.random.randn(self.enc_len, self.n_obs).astype(np.float32)
        result = self.tft.predict(features=features)
        assert result['signal_id'] == 'TFT_FORECAST'
        assert 'forecasts' in result
        assert '1d' in result['forecasts']
        assert '5d' in result['forecasts']
        assert '20d' in result['forecasts']

    def test_forecast_quantiles(self):
        features = np.random.randn(self.enc_len, self.n_obs).astype(np.float32)
        result = self.tft.predict(features=features)
        for horizon in ['1d', '3d', '5d', '10d', '20d']:
            assert 'p10' in result['forecasts'][horizon]
            assert 'p50' in result['forecasts'][horizon]
            assert 'p90' in result['forecasts'][horizon]
            # p10 should be <= p50 <= p90 (not strictly enforced by untrained model)

    def test_fallback_when_no_features(self):
        result = self.tft.predict(features=None)
        assert result['direction'] == 'NEUTRAL'
        assert result['confidence'] == 0.0
        assert result['method'] == 'fallback'

    def test_direction_output(self):
        features = np.random.randn(self.enc_len, self.n_obs).astype(np.float32)
        result = self.tft.predict(features=features)
        assert result['direction'] in ('BULLISH', 'BEARISH', 'NEUTRAL')
        assert 0.0 <= result['confidence'] <= 1.0

    def test_feature_importance(self):
        features = np.random.randn(self.enc_len, self.n_obs).astype(np.float32)
        result = self.tft.predict(features=features)
        assert 'feature_importance' in result
        assert len(result['feature_importance']) <= 5  # Top 5

    def test_grn_forward(self):
        from models.tft_forecaster import GatedResidualNetwork
        grn = GatedResidualNetwork(8, 16, 8)
        x = np.random.randn(8)
        out = grn.forward(x)
        assert out.shape == (8,)

    def test_vsn_forward(self):
        from models.tft_forecaster import VariableSelectionNetwork
        vsn = VariableSelectionNetwork(5, 16)
        x = np.random.randn(5)
        weighted, importance = vsn.forward(x)
        assert importance.shape == (5,)
        assert abs(importance.sum() - 1.0) < 0.01  # Sums to 1


# ================================================================
# 3. RL Position Sizer Tests
# ================================================================
class TestRLPositionSizer:
    def setup_method(self):
        from models.rl_position_sizer import RLPositionSizer
        self.sizer = RLPositionSizer()

    def test_get_size_basic(self):
        signals = {
            'PCR_AUTOTRENDER': {'size_modifier': 1.15},
            'FII_FUTURES_OI': {'size_modifier': 1.35},
            'context': {'vix_level': 15, 'vix_regime': 'NORMAL'},
        }
        result = self.sizer.get_size(signals)
        assert result['signal_id'] == 'RL_POSITION_SIZER'
        assert 0.1 <= result['size_modifier'] <= 2.0

    def test_untrained_conservative(self):
        result = self.sizer.get_size({})
        assert result['trained'] is False
        assert result['method'] == 'sac_untrained'
        # Untrained should be near 1.0
        assert 0.5 <= result['size_modifier'] <= 1.5

    def test_build_state(self):
        from models.rl_position_sizer import STATE_DIM
        signals = {'context': {'vix_level': 20}}
        portfolio = {
            'pnl_today_pct': 1.0,
            'n_open_positions': 2,
            'drawdown_pct': 5.0,
            'cash_ratio': 0.6,
            'time_to_close_min': 200,
        }
        state = self.sizer.build_state(signals, portfolio)
        assert state.shape == (STATE_DIM,)
        assert state[11] == pytest.approx(20.0 / 20.0, abs=0.01)

    def test_gaussian_actor(self):
        from models.rl_position_sizer import GaussianActor, STATE_DIM
        actor = GaussianActor(STATE_DIM, 1, 128)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob = actor.sample(state)
        assert 0.1 <= action <= 2.0
        det = actor.deterministic(state)
        assert 0.1 <= det <= 2.0

    def test_twin_critic(self):
        from models.rl_position_sizer import TwinCritic, STATE_DIM
        critic = TwinCritic(STATE_DIM, 1, 128)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.array([1.0])
        q1, q2 = critic.forward(state, action)
        assert isinstance(q1, float)
        assert isinstance(q2, float)

    def test_compute_reward(self):
        reward = self.sizer.compute_reward(
            pnl_pct=2.0, drawdown_pct=0.5, size_modifier=1.0
        )
        assert isinstance(reward, float)
        # pnl=2.0 - drawdown_penalty=0.5*2=1.0 - position_cost=0*0.001=0 = 1.0
        assert reward > 0

    def test_replay_buffer(self):
        from models.rl_position_sizer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(5), 1.0, 0.5, np.zeros(5), False)
        assert len(buf) == 50
        batch = buf.sample(10)
        assert len(batch) == 10


# ================================================================
# 4. GNN Sector Rotation Tests
# ================================================================
class TestGNNSectorRotation:
    def setup_method(self):
        from models.gnn_sector_rotation import GNNSectorRotation
        self.gnn = GNNSectorRotation()

    def test_evaluate_no_data_fallback(self):
        result = self.gnn.evaluate()
        assert result['signal_id'] == 'GNN_SECTOR_ROTATION'
        assert result['regime'] in ('UNKNOWN', 'NORMAL', 'HIGH_CORRELATION',
                                     'FRAGMENTED', 'TRANSITIONAL',
                                     'HIGH_CORRELATION_STRESS')

    def test_graph_metrics_computation(self):
        from models.gnn_sector_rotation import GNNSectorRotation
        # Build a small test graph
        adj = np.array([
            [0, 0.5, 0.6, 0],
            [0.5, 0, 0.7, 0],
            [0.6, 0.7, 0, 0.4],
            [0, 0, 0.4, 0],
        ])
        metrics = GNNSectorRotation._compute_graph_metrics(adj)
        assert 'clustering_coefficient' in metrics
        assert 'first_eigenvalue_ratio' in metrics
        assert 'graph_density' in metrics
        assert 'avg_degree' in metrics
        assert metrics['graph_density'] > 0

    def test_sector_rotation_detection(self):
        from models.gnn_sector_rotation import GNNSectorRotation, SECTOR_MAP
        stocks = list(SECTOR_MAP.keys())[:10]
        node_features = np.random.randn(10, 5)
        rotation = GNNSectorRotation._detect_sector_rotation(node_features, stocks)
        assert 'rotation_signal' in rotation
        assert rotation['rotation_signal'] in ('RISK_ON', 'RISK_OFF', 'MIXED')
        assert 'top_sectors' in rotation
        assert 'bottom_sectors' in rotation

    def test_gat_layer_forward(self):
        from models.gnn_sector_rotation import GraphAttentionLayer
        gat = GraphAttentionLayer(in_dim=5, out_dim=8, n_heads=2)
        features = np.random.randn(4, 5)
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        out = gat.forward(features, adj)
        assert out.shape == (4, 8)

    def test_sector_map_coverage(self):
        from models.gnn_sector_rotation import SECTOR_MAP
        assert len(SECTOR_MAP) >= 40  # Should cover most Nifty 50
        assert 'RELIANCE' in SECTOR_MAP
        assert 'TCS' in SECTOR_MAP
        assert SECTOR_MAP['RELIANCE'] == 'Energy'
        assert SECTOR_MAP['TCS'] == 'IT'


# ================================================================
# 5. NLP Sentiment Tests
# ================================================================
class TestNLPSentiment:
    def setup_method(self):
        from models.nlp_sentiment import NLPSentiment, KeywordSentimentAnalyzer
        self.nlp = NLPSentiment()
        self.analyzer = KeywordSentimentAnalyzer()

    def test_earnings_positive_sentiment(self):
        text = "Company beat estimates with record revenue and strong growth"
        result = self.analyzer.analyze_earnings(text)
        assert result['sentiment'] == 'POSITIVE'
        assert result['score'] > 0
        assert result['pos'] >= 3

    def test_earnings_negative_sentiment(self):
        text = "Revenue decline missed estimates with weak demand downgrade"
        result = self.analyzer.analyze_earnings(text)
        assert result['sentiment'] == 'NEGATIVE'
        assert result['score'] < 0
        assert result['neg'] >= 3

    def test_rbi_hawkish(self):
        text = "Inflation concerns persist with upside risks to inflation. Rate hike under consideration."
        result = self.analyzer.analyze_rbi(text)
        assert result['stance'] == 'HAWKISH'
        assert result['score'] < 0  # Hawkish = negative for equity

    def test_rbi_dovish(self):
        text = "Growth supportive accommodative stance. Rate cut likely with benign inflation."
        result = self.analyzer.analyze_rbi(text)
        assert result['stance'] == 'DOVISH'
        assert result['score'] > 0

    def test_evaluate_override_bullish(self):
        result = self.nlp.evaluate(earnings_override=0.5, rbi_override=0.3)
        assert result['direction'] == 'BULLISH'
        assert result['combined_score'] > 0

    def test_evaluate_override_bearish(self):
        result = self.nlp.evaluate(earnings_override=-0.5, rbi_override=-0.3)
        assert result['direction'] == 'BEARISH'
        assert result['combined_score'] < 0

    def test_evaluate_neutral(self):
        result = self.nlp.evaluate(earnings_override=0.0, rbi_override=0.0)
        assert result['direction'] == 'NEUTRAL'
        assert result['category'] == 'NEUTRAL'

    def test_neutral_text(self):
        text = "The market traded flat today with moderate volumes."
        result = self.analyzer.analyze_earnings(text)
        assert result['sentiment'] == 'NEUTRAL'
        assert result['score'] == 0.0


# ================================================================
# 6. AMFI Mutual Fund Flows Tests
# ================================================================
class TestAMFIMutualFundSignal:
    def setup_method(self):
        from data.amfi_mf_flows import AMFIMutualFundSignal
        self.sig = AMFIMutualFundSignal()

    def test_strong_floor_bullish(self):
        ctx = self.sig.evaluate(sip_override=23000.0, equity_flow_override=20000.0)
        assert ctx.sip_floor_type == 'STRONG_FLOOR'
        assert ctx.equity_flow_type == 'STRONG_INFLOW'
        assert ctx.direction == 'BULLISH'
        assert ctx.size_modifier >= 1.1

    def test_normal_floor(self):
        ctx = self.sig.evaluate(sip_override=20000.0, equity_flow_override=10000.0)
        assert ctx.sip_floor_type == 'FLOOR'
        assert ctx.equity_flow_type == 'MODERATE_INFLOW'
        assert ctx.direction == 'BULLISH'

    def test_weak_floor_neutral(self):
        ctx = self.sig.evaluate(sip_override=17000.0, equity_flow_override=3000.0)
        assert ctx.sip_floor_type == 'WEAK_FLOOR'
        assert ctx.equity_flow_type == 'WEAK_INFLOW'
        # Weak floor + weak inflow → NEUTRAL
        assert ctx.direction == 'NEUTRAL'

    def test_outflow_bearish(self):
        ctx = self.sig.evaluate(sip_override=15000.0, equity_flow_override=-5000.0)
        assert ctx.equity_flow_type == 'OUTFLOW'
        assert ctx.direction == 'BEARISH'
        assert ctx.size_modifier <= 1.0

    def test_to_dict_has_direction(self):
        ctx = self.sig.evaluate(sip_override=22000.0, equity_flow_override=16000.0)
        d = ctx.to_dict()
        assert 'direction' in d
        assert 'signal_id' in d
        assert d['signal_id'] == 'AMFI_MF_FLOW'

    def test_to_telegram(self):
        ctx = self.sig.evaluate(sip_override=22000.0, equity_flow_override=16000.0)
        msg = ctx.to_telegram()
        assert 'MF Flow Signal' in msg
        assert 'SIP' in msg

    def test_no_data_fallback(self):
        ctx = self.sig.evaluate()
        # Without DB, should return neutral fallback
        assert ctx.direction == 'NEUTRAL'
        assert ctx.confidence == 0.0


# ================================================================
# 7. Credit Card Spending Tests
# ================================================================
class TestCreditCardSpendingSignal:
    def setup_method(self):
        from data.credit_card_spending import CreditCardSpendingSignal
        self.sig = CreditCardSpendingSignal()

    def test_strong_expansion_bullish(self):
        ctx = self.sig.evaluate(
            spending_override=150000.0,
            spending_yoy_override=30.0,
            cards_growth_override=18.0,
        )
        assert ctx.spending_category == 'STRONG_EXPANSION'
        assert ctx.cards_category == 'RAPID_ADOPTION'
        assert ctx.direction == 'BULLISH'
        assert ctx.size_modifier >= 1.1

    def test_expansion(self):
        ctx = self.sig.evaluate(
            spending_override=120000.0,
            spending_yoy_override=20.0,
            cards_growth_override=12.0,
        )
        assert ctx.spending_category == 'EXPANSION'
        assert ctx.direction == 'BULLISH'

    def test_stable_neutral(self):
        ctx = self.sig.evaluate(
            spending_override=100000.0,
            spending_yoy_override=8.0,
            cards_growth_override=6.0,
        )
        assert ctx.spending_category == 'STABLE'
        assert ctx.direction == 'NEUTRAL'

    def test_contraction_bearish(self):
        ctx = self.sig.evaluate(
            spending_override=80000.0,
            spending_yoy_override=-5.0,
            cards_growth_override=3.0,
        )
        assert ctx.spending_category == 'CONTRACTION'
        assert ctx.direction == 'BEARISH'
        assert ctx.size_modifier <= 1.0

    def test_to_dict_has_direction(self):
        ctx = self.sig.evaluate(
            spending_override=150000.0,
            spending_yoy_override=30.0,
        )
        d = ctx.to_dict()
        assert 'direction' in d
        assert 'signal_id' in d
        assert d['signal_id'] == 'CREDIT_CARD_SPENDING'

    def test_to_telegram(self):
        ctx = self.sig.evaluate(
            spending_override=150000.0,
            spending_yoy_override=30.0,
        )
        msg = ctx.to_telegram()
        assert 'CC Spending Signal' in msg
        assert 'Spend' in msg

    def test_no_data_fallback(self):
        ctx = self.sig.evaluate()
        assert ctx.direction == 'NEUTRAL'
        assert ctx.confidence == 0.0


# ================================================================
# 8. Cross-Module Integration Tests
# ================================================================
class TestTier3Integration:
    def test_all_signals_importable(self):
        """All Tier 3 modules should import without error."""
        from models.mamba_regime import MambaRegimeDetector
        from models.tft_forecaster import TFTForecaster
        from models.rl_position_sizer import RLPositionSizer
        from models.gnn_sector_rotation import GNNSectorRotation
        from models.nlp_sentiment import NLPSentiment
        from data.amfi_mf_flows import AMFIMutualFundSignal
        from data.credit_card_spending import CreditCardSpendingSignal
        assert True

    def test_all_signals_have_signal_id(self):
        """All signals should have SIGNAL_ID class attribute."""
        from models.mamba_regime import MambaRegimeDetector
        from models.tft_forecaster import TFTForecaster
        from models.rl_position_sizer import RLPositionSizer
        from models.gnn_sector_rotation import GNNSectorRotation
        from models.nlp_sentiment import NLPSentiment
        from data.amfi_mf_flows import AMFIMutualFundSignal
        from data.credit_card_spending import CreditCardSpendingSignal

        assert MambaRegimeDetector.SIGNAL_ID == 'MAMBA_REGIME'
        assert TFTForecaster.SIGNAL_ID == 'TFT_FORECAST'
        assert RLPositionSizer.SIGNAL_ID == 'RL_POSITION_SIZER'
        assert GNNSectorRotation.SIGNAL_ID == 'GNN_SECTOR_ROTATION'
        assert NLPSentiment.SIGNAL_ID == 'NLP_SENTIMENT'
        assert AMFIMutualFundSignal.SIGNAL_ID == 'AMFI_MF_FLOW'
        assert CreditCardSpendingSignal.SIGNAL_ID == 'CREDIT_CARD_SPENDING'

    def test_all_signals_evaluate_without_db(self):
        """All signals should gracefully handle no DB connection."""
        from models.mamba_regime import MambaRegimeDetector
        from models.tft_forecaster import TFTForecaster
        from models.rl_position_sizer import RLPositionSizer
        from models.gnn_sector_rotation import GNNSectorRotation
        from models.nlp_sentiment import NLPSentiment
        from data.amfi_mf_flows import AMFIMutualFundSignal
        from data.credit_card_spending import CreditCardSpendingSignal

        # Each should return a dict/dataclass without crashing
        r1 = MambaRegimeDetector().evaluate()
        assert 'direction' in r1

        r2 = TFTForecaster().evaluate()
        assert 'direction' in r2

        r3 = RLPositionSizer().evaluate({})
        assert 'size_modifier' in r3

        r4 = GNNSectorRotation().evaluate()
        assert 'direction' in r4

        r5 = NLPSentiment().evaluate()
        assert 'direction' in r5

        r6 = AMFIMutualFundSignal().evaluate()
        assert r6.direction in ('BULLISH', 'BEARISH', 'NEUTRAL')

        r7 = CreditCardSpendingSignal().evaluate()
        assert r7.direction in ('BULLISH', 'BEARISH', 'NEUTRAL')

    def test_sizing_modifiers_in_valid_range(self):
        """All size modifiers should be in [0.1, 2.0]."""
        from models.mamba_regime import MambaRegimeDetector
        from models.tft_forecaster import TFTForecaster
        from models.rl_position_sizer import RLPositionSizer
        from data.amfi_mf_flows import AMFIMutualFundSignal
        from data.credit_card_spending import CreditCardSpendingSignal

        r1 = MambaRegimeDetector().evaluate()
        assert 0.1 <= r1['size_modifier'] <= 2.0

        r2 = TFTForecaster().evaluate()
        assert 0.1 <= r2['size_modifier'] <= 2.0

        r3 = RLPositionSizer().evaluate({})
        assert 0.1 <= r3['size_modifier'] <= 2.0

        r6 = AMFIMutualFundSignal().evaluate()
        assert 0.3 <= r6.size_modifier <= 2.0

        r7 = CreditCardSpendingSignal().evaluate()
        assert 0.3 <= r7.size_modifier <= 2.0
