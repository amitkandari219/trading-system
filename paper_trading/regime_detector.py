"""
Gujral Regime Detector

Uses GUJRAL_DRY_7 shadow trades as a regime indicator.

When GUJRAL_DRY_7 has 2+ winning trades out of last 3,
DRY_20's Sharpe jumps from 0.64 to 3.67 (5.8x improvement).

GUJRAL_DRY_7 rules:
  Entry LONG:  close > pivot AND prev_close <= pivot
  Entry SHORT: close < pivot AND prev_close >= pivot
  Exit LONG:   close < pivot
  Exit SHORT:  close > pivot
  Stop: 2%
"""


class GujralRegimeDetector:
    """
    Uses GUJRAL_DRY_7 shadow trades as a regime indicator.

    When GUJRAL_DRY_7 has 2+ winning trades out of last 3,
    DRY_20's Sharpe jumps from 0.64 to 3.67 (5.8x improvement).

    GUJRAL_DRY_7 rules:
      Entry LONG:  close > pivot AND prev_close <= pivot
      Entry SHORT: close < pivot AND prev_close >= pivot
      Exit LONG:   close < pivot
      Exit SHORT:  close > pivot
      Stop: 2%
    """

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self._mock_trades = None

    def set_mock_trades(self, trades: list):
        """
        Allow injecting mock trade data for testing without a DB connection.

        Each trade should be a dict with keys:
          gross_pnl, direction, entry_price, exit_price
        Ordered most-recent first (same as the SQL query).
        """
        self._mock_trades = trades

    def _fetch_last_3_trades(self) -> list:
        """
        Fetch last 3 completed GUJRAL_DRY_7 shadow trades.

        Returns list of dicts with keys: gross_pnl, direction, entry_price, exit_price.
        Most-recent trade first.
        """
        if self._mock_trades is not None:
            return self._mock_trades[:3]

        if self.conn is None:
            return []

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT gross_pnl, direction, entry_price, exit_price
            FROM trades
            WHERE signal_id = 'GUJRAL_DRY_7'
              AND trade_type = 'SHADOW'
              AND exit_date IS NOT NULL
            ORDER BY exit_date DESC LIMIT 3
            """
        )
        rows = cursor.fetchall()
        columns = ["gross_pnl", "direction", "entry_price", "exit_price"]
        return [dict(zip(columns, row)) for row in rows]

    def _is_win(self, trade: dict) -> bool:
        """Determine if a trade was a win based on gross_pnl."""
        return trade["gross_pnl"] > 0

    def get_streak(self) -> dict:
        """
        Query last 3 completed GUJRAL_DRY_7 shadow trades from DB.

        Returns:
        {
          'regime': 'FAVORABLE' | 'STANDARD' | 'UNKNOWN',
          'streak': '2/3' | '1/3' | '0/3' | 'N/A',
          'last_3_results': [True, False, True],  # win/loss booleans
          'trades_available': int,  # 0-3
        }

        Rules:
        - If fewer than 3 completed trades: regime = 'UNKNOWN'
        - If 2+ of last 3 are wins: regime = 'FAVORABLE'
        - If 0-1 of last 3 are wins: regime = 'STANDARD'

        A win is: positive gross_pnl (exit_price > entry_price for LONG,
                  exit_price < entry_price for SHORT)
        """
        trades = self._fetch_last_3_trades()
        trades_available = len(trades)

        if trades_available < 3:
            results = [self._is_win(t) for t in trades]
            return {
                "regime": "UNKNOWN",
                "streak": "N/A",
                "last_3_results": results,
                "trades_available": trades_available,
            }

        results = [self._is_win(t) for t in trades]
        wins = sum(results)

        if wins >= 2:
            regime = "FAVORABLE"
        else:
            regime = "STANDARD"

        return {
            "regime": regime,
            "streak": f"{wins}/3",
            "last_3_results": results,
            "trades_available": trades_available,
        }

    def get_confidence_label(self, regime: str, score: int) -> str:
        """
        Combine regime + scoring system score into a confidence label.

        FAVORABLE + score >= 2 -> 'HIGH'
        FAVORABLE + score == 1 -> 'MEDIUM'
        STANDARD  + score >= 2 -> 'MEDIUM'
        STANDARD  + score == 1 -> 'LOW'
        UNKNOWN   + any score  -> 'MEDIUM'

        Returns: 'HIGH' | 'MEDIUM' | 'LOW'
        """
        if regime == "UNKNOWN":
            return "MEDIUM"

        if regime == "FAVORABLE":
            if score >= 2:
                return "HIGH"
            else:
                return "MEDIUM"

        if regime == "STANDARD":
            if score >= 2:
                return "MEDIUM"
            else:
                return "LOW"

        # Fallback for unexpected regime values
        return "MEDIUM"

    def get_size_multiplier(self, confidence: str) -> float:
        """
        Map confidence to position size multiplier.

        HIGH   -> 1.0
        MEDIUM -> 0.5
        LOW    -> 0.25
        """
        multipliers = {
            "HIGH": 1.0,
            "MEDIUM": 0.5,
            "LOW": 0.25,
        }
        return multipliers.get(confidence, 0.25)
