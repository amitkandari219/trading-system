"""
Signal validation gateway: applies DSR + BH-FDR before signals enter paper trading.

Usage:
    python -m backtest.validate_signals --input combination_results/layer1_wf.json
    python -m backtest.validate_signals --input round2_results/round2_final.json
"""

import json
import argparse

from backtest.fdr_controller import FDRController


def validate_walk_forward_results(results_path: str, n_total_tested: int = None,
                                   alpha: float = 0.05) -> list:
    """
    Apply combined DSR + BH-FDR validation to walk-forward results.

    Args:
        results_path: path to JSON with walk-forward results
        n_total_tested: total signals tested (for DSR trials adjustment)
        alpha: FDR level

    Returns:
        list of validated signals with combined_tier
    """
    with open(results_path) as f:
        data = json.load(f)

    # Handle different result formats
    if isinstance(data, dict):
        # combination_results format
        signals = data.get('tier_a', []) + data.get('tier_b', []) + data.get('dropped', [])
    elif isinstance(data, list):
        signals = data
    else:
        signals = []

    if not signals:
        print("No signals to validate")
        return []

    # Prepare for FDR
    for sig in signals:
        sig['sharpe'] = sig.get('aggregate_sharpe', sig.get('sharpe', sig.get('full_sharpe', 0)))
        sig['trades'] = sig.get('full_trades', sig.get('trades', 100))

    if n_total_tested is None:
        n_total_tested = len(signals)

    fdr = FDRController()
    validated = fdr.combined_acceptance(
        signals,
        dsr_threshold=0.90,
        bh_alpha=alpha,
        n_trials=n_total_tested,
    )

    # Summary
    tier_a = [s for s in validated if s.get('combined_tier') == 'A']
    tier_b = [s for s in validated if s.get('combined_tier') == 'B']
    tier_c = [s for s in validated if s.get('combined_tier') == 'C']
    ghosts = [s for s in validated if s.get('combined_tier') == 'GHOST']

    print(f"\nValidated signal IDs:")
    for s in tier_a + tier_b:
        sid = s.get('signal_id', s.get('sig_a', 'unknown'))
        print(f"  Tier {s['combined_tier']}: {sid} "
              f"(DSR={s.get('dsr', 0):.3f}, p={s.get('p_value', 0):.4f}, "
              f"Sharpe={s.get('sharpe', 0):.2f})")

    return validated


def main():
    parser = argparse.ArgumentParser(description='Validate signals with DSR + BH-FDR')
    parser.add_argument('--input', required=True, help='Path to walk-forward results JSON')
    parser.add_argument('--n-tested', type=int, default=None, help='Total signals tested')
    parser.add_argument('--alpha', type=float, default=0.05, help='FDR alpha level')
    args = parser.parse_args()

    results = validate_walk_forward_results(args.input, args.n_tested, args.alpha)

    # Save validated results
    output = args.input.replace('.json', '_validated.json')
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {output}")


if __name__ == '__main__':
    main()
