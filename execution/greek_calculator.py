"""
Greek calculator — Python stub that calls the Go risk service.
Computes combined portfolio Greeks for pre-order risk checks.
"""

import requests
from portfolio.portfolio_model import PortfolioGreeks

GREEK_SERVICE_URL = "http://localhost:8080/greeks/combined"
GREEK_SERVICE_TIMEOUT_S = 0.5   # 500ms hard deadline — fail fast


def compute_combined_greeks(current_portfolio, proposed_signal,
                             market_data: dict) -> PortfolioGreeks:
    """
    Compute the portfolio Greeks that would result if proposed_signal
    were added to current_portfolio at current market prices.

    Returns PortfolioGreeks with fields: delta, vega, gamma, theta.
    Raises RuntimeError if the Go service is unreachable or slow —
    caller (greek_pre_check) must treat that as a rejection.

    market_data keys required:
        nifty_spot      float   Current Nifty 50 spot price
        india_vix       float   India VIX value (annualised %)
        risk_free_rate  float   Current repo rate (annualised, e.g. 0.065)
        timestamp       str     ISO8601 — used to compute DTE
    """
    payload = _build_payload(current_portfolio, proposed_signal, market_data)
    try:
        resp = requests.post(
            GREEK_SERVICE_URL,
            json=payload,
            timeout=GREEK_SERVICE_TIMEOUT_S
        )
        resp.raise_for_status()
        data = resp.json()
        return PortfolioGreeks(
            delta = data['portfolio_delta'],
            vega  = data['portfolio_vega'],
            gamma = data['portfolio_gamma'],
            theta = data['portfolio_theta'],
        )
    except requests.Timeout:
        raise RuntimeError("Greek service timeout — rejecting order for safety")
    except requests.RequestException as e:
        raise RuntimeError(f"Greek service error: {e}")


def _build_payload(portfolio, signal, market_data: dict) -> dict:
    """
    Serialise portfolio + proposed signal into the JSON payload
    the Go service expects.
    """
    return {
        "market_data": {
            "nifty_spot":     market_data["nifty_spot"],
            "india_vix":      market_data["india_vix"],
            "risk_free_rate": market_data.get("risk_free_rate", 0.065),
            "timestamp":      market_data["timestamp"],
        },
        # Existing open positions — Go recalculates their Greeks at live prices
        "existing_positions": [
            {
                "instrument_type": "FUTURES" if "FUT" in p.instrument else "OPTIONS",
                "symbol":          p.instrument,
                "direction":       p.direction,
                "lots":            p.lots,
            }
            for p in portfolio.open_positions
        ],
        # Proposed new position
        "proposed_position": {
            "instrument_type": signal.instrument,
            "signal_id":       signal.signal_id,
            "direction":       signal.direction,
            "lots":            getattr(signal, 'lot_size', 1),
            "strike":          getattr(signal, 'strike', None),
            "expiry":          getattr(signal, 'expiry_date', None),
        },
        # Lot size for Nifty F&O: 25 (post-2024 contract size)
        "nifty_lot_size": 25,
    }
