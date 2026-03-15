"""
Greek pre-check — called before every order placement.
Validates that adding a proposed signal won't breach Greek limits.
"""

from execution.greek_calculator import compute_combined_greeks

GREEK_LIMITS = {
    'max_portfolio_delta': 0.50,
    'max_portfolio_vega':  3000,
    'max_portfolio_gamma': 30000,
    'max_portfolio_theta': -5000,
}


def greek_pre_check(proposed_signal, current_portfolio, market_data):
    """
    Called before every order placement.
    Returns: (approved: bool, rejection_reason: str)
    Raises RuntimeError if compute_combined_greeks() times out —
    caller must treat that as a rejection.
    """
    proposed = compute_combined_greeks(
        current_portfolio, proposed_signal, market_data
    )

    if abs(proposed.delta) > GREEK_LIMITS['max_portfolio_delta']:
        return False, f"Delta breach: {proposed.delta:.3f}"

    if abs(proposed.vega) > GREEK_LIMITS['max_portfolio_vega']:
        return False, f"Vega breach: {proposed.vega:.0f}"

    if abs(proposed.gamma) > GREEK_LIMITS['max_portfolio_gamma']:
        return False, f"Gamma breach: {proposed.gamma:.0f}"

    if proposed.theta < GREEK_LIMITS['max_portfolio_theta']:
        return False, f"Theta breach: {proposed.theta:.0f}"

    return True, "APPROVED"
