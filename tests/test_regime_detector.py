"""Tests for GujralRegimeDetector."""

import pytest

from paper_trading.regime_detector import GujralRegimeDetector


# ---------------------------------------------------------------------------
# Helper to build mock trade dicts
# ---------------------------------------------------------------------------

def _trade(gross_pnl, direction="LONG", entry_price=100.0, exit_price=None):
    """Create a mock trade dict."""
    if exit_price is None:
        if direction == "LONG":
            exit_price = entry_price + gross_pnl
        else:
            exit_price = entry_price - gross_pnl
    return {
        "gross_pnl": gross_pnl,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


# ---------------------------------------------------------------------------
# get_streak tests (1-6)
# ---------------------------------------------------------------------------

class TestGetStreak:
    """Tests for regime classification based on shadow trade results."""

    def test_3_wins_favorable(self):
        """1. Three wins out of three -> FAVORABLE."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([_trade(5), _trade(3), _trade(1)])
        result = detector.get_streak()
        assert result["regime"] == "FAVORABLE"
        assert result["streak"] == "3/3"
        assert result["last_3_results"] == [True, True, True]
        assert result["trades_available"] == 3

    def test_2_wins_1_loss_favorable(self):
        """2. Two wins, one loss -> FAVORABLE."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([_trade(5), _trade(-2), _trade(1)])
        result = detector.get_streak()
        assert result["regime"] == "FAVORABLE"
        assert result["streak"] == "2/3"
        assert result["last_3_results"] == [True, False, True]
        assert result["trades_available"] == 3

    def test_1_win_2_losses_standard(self):
        """3. One win, two losses -> STANDARD."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([_trade(-3), _trade(2), _trade(-1)])
        result = detector.get_streak()
        assert result["regime"] == "STANDARD"
        assert result["streak"] == "1/3"
        assert result["last_3_results"] == [False, True, False]
        assert result["trades_available"] == 3

    def test_0_wins_3_losses_standard(self):
        """4. Zero wins, three losses -> STANDARD."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([_trade(-1), _trade(-2), _trade(-3)])
        result = detector.get_streak()
        assert result["regime"] == "STANDARD"
        assert result["streak"] == "0/3"
        assert result["last_3_results"] == [False, False, False]
        assert result["trades_available"] == 3

    def test_only_2_trades_unknown(self):
        """5. Only 2 trades available -> UNKNOWN."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([_trade(5), _trade(3)])
        result = detector.get_streak()
        assert result["regime"] == "UNKNOWN"
        assert result["streak"] == "N/A"
        assert result["trades_available"] == 2

    def test_0_trades_unknown(self):
        """6. Zero trades available -> UNKNOWN."""
        detector = GujralRegimeDetector()
        detector.set_mock_trades([])
        result = detector.get_streak()
        assert result["regime"] == "UNKNOWN"
        assert result["streak"] == "N/A"
        assert result["last_3_results"] == []
        assert result["trades_available"] == 0


# ---------------------------------------------------------------------------
# get_confidence_label tests (7-9)
# ---------------------------------------------------------------------------

class TestGetConfidenceLabel:
    """Tests for confidence label derivation from regime + score."""

    def test_favorable_score_3_high(self):
        """7. FAVORABLE + score 3 -> HIGH."""
        detector = GujralRegimeDetector()
        assert detector.get_confidence_label("FAVORABLE", 3) == "HIGH"

    def test_standard_score_2_medium(self):
        """8. STANDARD + score 2 -> MEDIUM."""
        detector = GujralRegimeDetector()
        assert detector.get_confidence_label("STANDARD", 2) == "MEDIUM"

    def test_unknown_score_3_medium_not_high(self):
        """9. UNKNOWN + score 3 -> MEDIUM (not HIGH)."""
        detector = GujralRegimeDetector()
        label = detector.get_confidence_label("UNKNOWN", 3)
        assert label == "MEDIUM"
        assert label != "HIGH"


# ---------------------------------------------------------------------------
# get_size_multiplier tests (10)
# ---------------------------------------------------------------------------

class TestGetSizeMultiplier:
    """Tests for position size multiplier mapping."""

    def test_size_multipliers(self):
        """10. HIGH -> 1.0, MEDIUM -> 0.5, LOW -> 0.25."""
        detector = GujralRegimeDetector()
        assert detector.get_size_multiplier("HIGH") == 1.0
        assert detector.get_size_multiplier("MEDIUM") == 0.5
        assert detector.get_size_multiplier("LOW") == 0.25
