"""
AMFI Mutual Fund Flow Data Scraper + Signal.

Monthly mutual fund flow data from amfiindia.com provides structural
floor/ceiling signals for Nifty:
  - SIP flows > ₹21K Cr/month = structural floor (persistent buying)
  - MoM SIP growth > 5% = accelerating inflows = bullish
  - Equity MF outflows = institutional selling pressure = bearish
  - Debt → Equity rotation = bullish regime shift

Data source:
  - AMFI India: https://www.amfiindia.com/research-information/aum-data/aum-data
  - Monthly data, published 10th of following month

Signal logic:
  SIP flows:
    > ₹22K Cr → STRONG_FLOOR (very bullish structural support)
    ₹18-22K Cr → FLOOR (normal structural support)
    < ₹18K Cr → WEAK_FLOOR (potential exhaustion)

  Equity MF net flows (SIP + lump-sum - redemptions):
    > ₹15K Cr → STRONG_INFLOW → Bullish
    ₹5-15K Cr → MODERATE_INFLOW → Mild bullish
    ₹0-5K Cr → WEAK_INFLOW → Neutral
    < ₹0 Cr → OUTFLOW → Bearish

  MoM change:
    Accelerating inflows → trend continuation
    Decelerating inflows → potential reversal warning

Usage:
    from data.amfi_mf_flows import AMFIMutualFundSignal
    sig = AMFIMutualFundSignal(db_conn=conn)
    result = sig.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# THRESHOLDS (₹ Crore)
# ================================================================
SIP_STRONG_FLOOR = 22000
SIP_NORMAL_FLOOR = 18000

EQUITY_STRONG_INFLOW = 15000
EQUITY_MODERATE_INFLOW = 5000
EQUITY_WEAK_INFLOW = 0

# MoM change thresholds
MOM_ACCELERATING = 5.0    # >5% MoM growth
MOM_DECELERATING = -5.0   # <-5% MoM decline

# Size modifiers
SIZE_MAP = {
    'STRONG_BULLISH': 1.20,
    'BULLISH': 1.10,
    'NEUTRAL': 1.00,
    'BEARISH': 0.90,
    'STRONG_BEARISH': 0.80,
}


@dataclass
class MFFlowContext:
    """Evaluation result from mutual fund flow signal."""
    sip_monthly: float            # ₹ Cr
    sip_floor_type: str           # STRONG_FLOOR, FLOOR, WEAK_FLOOR
    equity_net_flow: float        # ₹ Cr (SIP + lump - redemptions)
    equity_flow_type: str         # STRONG_INFLOW, MODERATE, WEAK, OUTFLOW
    mom_change_pct: float         # Month-over-month change %
    mom_trend: str                # ACCELERATING, DECELERATING, STABLE
    debt_equity_ratio: float      # Debt MF AUM / Equity MF AUM
    direction: str
    confidence: float
    size_modifier: float
    data_month: str               # YYYY-MM of latest data
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'AMFI_MF_FLOW',
            'sip_monthly': self.sip_monthly,
            'sip_floor_type': self.sip_floor_type,
            'equity_net_flow': self.equity_net_flow,
            'equity_flow_type': self.equity_flow_type,
            'mom_change_pct': round(self.mom_change_pct, 2),
            'mom_trend': self.mom_trend,
            'debt_equity_ratio': round(self.debt_equity_ratio, 3),
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'data_month': self.data_month,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        return (
            f"{emoji} MF Flow Signal ({self.data_month})\n"
            f"  SIP: ₹{self.sip_monthly:,.0f} Cr ({self.sip_floor_type})\n"
            f"  Equity Net: ₹{self.equity_net_flow:,.0f} Cr ({self.equity_flow_type})\n"
            f"  MoM: {self.mom_change_pct:+.1f}% ({self.mom_trend})\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class AMFIMutualFundSignal:
    """
    AMFI mutual fund flow signal.

    Tracks SIP flows, equity net flows, and debt-equity rotation
    as structural market support/pressure signals.
    """

    SIGNAL_ID = 'AMFI_MF_FLOW'

    def __init__(self, db_conn=None):
        self.conn = db_conn

    def _get_conn(self):
        if self.conn:
            try:
                if not self.conn.closed:
                    return self.conn
            except Exception:
                pass
        try:
            import psycopg2
            from config.settings import DATABASE_DSN
            self.conn = psycopg2.connect(DATABASE_DSN)
            return self.conn
        except Exception as e:
            logger.error("DB connection failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Data retrieval
    # ----------------------------------------------------------
    def _get_mf_data(
        self, trade_date: date, lookback_months: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Fetch monthly MF flow data.

        Returns DataFrame with: month_date, sip_amount, equity_net_flow,
                                 debt_aum, equity_aum
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback_months * 35)

        try:
            df = pd.read_sql(
                """
                SELECT month_date, sip_amount, equity_net_flow,
                       debt_aum, equity_aum
                FROM mf_monthly_flows
                WHERE month_date BETWEEN %s AND %s
                ORDER BY month_date
                """,
                conn, params=(start_date, trade_date)
            )
            return df if len(df) >= 2 else None
        except Exception:
            return None

    def fetch_from_amfi(self) -> Optional[Dict]:
        """
        Fetch latest data from AMFI website.

        Returns dict with sip, equity_net, debt_aum, equity_aum.
        """
        try:
            import requests
            resp = requests.get(
                'https://www.amfiindia.com/research-information/aum-data/aum-data',
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=15
            )
            # Parse HTML for latest month data (simplified)
            # Full implementation would use BeautifulSoup
            logger.info("AMFI page fetched (%d bytes), parsing not implemented yet",
                       len(resp.text))
        except Exception as e:
            logger.debug("AMFI fetch failed: %s", e)
        return None

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_sip(sip: float) -> str:
        if sip >= SIP_STRONG_FLOOR:
            return 'STRONG_FLOOR'
        elif sip >= SIP_NORMAL_FLOOR:
            return 'FLOOR'
        else:
            return 'WEAK_FLOOR'

    @staticmethod
    def _classify_equity_flow(flow: float) -> str:
        if flow >= EQUITY_STRONG_INFLOW:
            return 'STRONG_INFLOW'
        elif flow >= EQUITY_MODERATE_INFLOW:
            return 'MODERATE_INFLOW'
        elif flow >= EQUITY_WEAK_INFLOW:
            return 'WEAK_INFLOW'
        else:
            return 'OUTFLOW'

    @staticmethod
    def _classify_mom(mom_pct: float) -> str:
        if mom_pct > MOM_ACCELERATING:
            return 'ACCELERATING'
        elif mom_pct < MOM_DECELERATING:
            return 'DECELERATING'
        else:
            return 'STABLE'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        sip_override: Optional[float] = None,
        equity_flow_override: Optional[float] = None,
    ) -> MFFlowContext:
        """Evaluate mutual fund flow signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path
        if sip_override is not None:
            sip = sip_override
            equity_flow = equity_flow_override or 10000
            mom_change = 0.0
            debt_equity_ratio = 0.8
            data_month = trade_date.strftime('%Y-%m')
        else:
            df = self._get_mf_data(trade_date)
            if df is None or len(df) < 2:
                return MFFlowContext(
                    sip_monthly=0, sip_floor_type='UNKNOWN',
                    equity_net_flow=0, equity_flow_type='UNKNOWN',
                    mom_change_pct=0, mom_trend='UNKNOWN',
                    debt_equity_ratio=0, direction='NEUTRAL',
                    confidence=0.0, size_modifier=1.0,
                    data_month='N/A',
                    reason='No MF flow data available'
                )

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            sip = float(latest['sip_amount'])
            equity_flow = float(latest['equity_net_flow'])

            # MoM change
            prev_flow = float(prev['equity_net_flow'])
            if abs(prev_flow) > 0:
                mom_change = (equity_flow - prev_flow) / abs(prev_flow) * 100
            else:
                mom_change = 0.0

            # Debt/Equity ratio
            debt_aum = float(latest.get('debt_aum', 0))
            equity_aum = float(latest.get('equity_aum', 1))
            debt_equity_ratio = debt_aum / max(equity_aum, 1)
            data_month = str(latest['month_date'])[:7]

        # Classify
        sip_type = self._classify_sip(sip)
        flow_type = self._classify_equity_flow(equity_flow)
        mom_trend = self._classify_mom(mom_change)

        # Direction
        if flow_type in ('STRONG_INFLOW', 'MODERATE_INFLOW') and sip_type != 'WEAK_FLOOR':
            direction = 'BULLISH'
        elif flow_type == 'OUTFLOW':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        # Strength
        if flow_type == 'STRONG_INFLOW' and mom_trend == 'ACCELERATING':
            strength = 'STRONG_BULLISH'
        elif flow_type == 'OUTFLOW' and mom_trend == 'DECELERATING':
            strength = 'STRONG_BEARISH'
        elif direction == 'BULLISH':
            strength = 'BULLISH'
        elif direction == 'BEARISH':
            strength = 'BEARISH'
        else:
            strength = 'NEUTRAL'

        size_modifier = SIZE_MAP.get(strength, 1.0)

        # Confidence (lower for MF data due to monthly lag)
        confidence = 0.40
        if flow_type.startswith('STRONG'):
            confidence += 0.15
        if mom_trend != 'STABLE':
            confidence += 0.10
        confidence = min(0.75, confidence)

        parts = [
            f"SIP=₹{sip:,.0f}Cr({sip_type})",
            f"NetFlow=₹{equity_flow:,.0f}Cr({flow_type})",
            f"MoM={mom_change:+.1f}%({mom_trend})",
            f"D/E={debt_equity_ratio:.2f}",
            f"Month={data_month}",
        ]

        return MFFlowContext(
            sip_monthly=sip,
            sip_floor_type=sip_type,
            equity_net_flow=equity_flow,
            equity_flow_type=flow_type,
            mom_change_pct=mom_change,
            mom_trend=mom_trend,
            debt_equity_ratio=debt_equity_ratio,
            direction=direction,
            confidence=confidence,
            size_modifier=size_modifier,
            data_month=data_month,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sig = AMFIMutualFundSignal()
    for sip, flow in [(23000, 20000), (20000, 10000), (17000, 3000), (15000, -5000)]:
        ctx = sig.evaluate(sip_override=float(sip), equity_flow_override=float(flow))
        print(f"SIP=₹{sip:,} Flow=₹{flow:,} → {ctx.sip_floor_type:14s} "
              f"{ctx.equity_flow_type:16s} {ctx.direction:8s} size={ctx.size_modifier:.2f}")
