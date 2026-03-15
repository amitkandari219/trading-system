"""
Parameter Sensitivity Tester.
Tests whether a signal's edge is robust to parameter variation.
Every numerical parameter tested at 5 values.
Signal fragility scored as ROBUST / MODERATE / FRAGILE.
"""


class ParameterSensitivityTester:
    """
    Tests whether a signal's edge is robust to parameter variation.
    Fragile signals (edge exists only at exact book parameter) are
    flagged and require higher Sharpe threshold to pass.
    """

    SENSITIVITY_MULTIPLIERS = [0.6, 0.8, 1.0, 1.2, 1.4]
    # Test at -40%, -20%, exact, +20%, +40% of book parameter

    ROBUST_THRESHOLD = 0.70
    # Signal is ROBUST if Sharpe at each multiplier >= 70% of peak Sharpe

    FRAGILE_THRESHOLD = 0.50
    # Signal is FRAGILE if Sharpe drops below 50% of peak at +/-20%

    def test_signal_sensitivity(self, signal_id, base_params,
                                backtest_fn, history_df, regime_labels):
        """
        Tests signal at 5 parameter variants.
        Returns sensitivity report.

        base_params: dict of {param_name: book_value}
                     Only numerical params are varied.
        backtest_fn: function(params, data, regimes) -> BacktestResult
        """
        results = {}

        for param_name, base_value in base_params.items():
            if not isinstance(base_value, (int, float)):
                continue

            param_results = []

            for multiplier in self.SENSITIVITY_MULTIPLIERS:
                test_params = base_params.copy()
                test_value = base_value * multiplier

                # Round integers to nearest sensible value
                if isinstance(base_value, int):
                    test_value = max(1, round(test_value))

                test_params[param_name] = test_value
                result = backtest_fn(test_params, history_df, regime_labels)
                param_results.append({
                    'multiplier': multiplier,
                    'param_value': test_value,
                    'sharpe': result.sharpe,
                    'profit_factor': result.profit_factor,
                })

            peak_sharpe = max(r['sharpe'] for r in param_results)
            results[param_name] = {
                'variants': param_results,
                'peak_sharpe': peak_sharpe,
                'fragility': self._compute_fragility(
                    param_results, peak_sharpe
                ),
            }

        return self._generate_report(signal_id, results)

    def _compute_fragility(self, variants, peak_sharpe):
        """
        ROBUST:   All +/-20% variants >= 70% of peak Sharpe
        MODERATE: Worst +/-20% variant between 50-70% of peak
        FRAGILE:  Any +/-20% variant below 50% of peak
        """
        if peak_sharpe <= 0:
            return 'FRAGILE'

        # Check +/-20% variants specifically (multipliers 0.8 and 1.2)
        pm20_variants = [v for v in variants
                         if v['multiplier'] in [0.8, 1.2]]

        verdicts = []
        for v in pm20_variants:
            ratio = v['sharpe'] / peak_sharpe
            if ratio < self.FRAGILE_THRESHOLD:
                verdicts.append('FRAGILE')
            elif ratio < self.ROBUST_THRESHOLD:
                verdicts.append('MODERATE')
            else:
                verdicts.append('ROBUST')

        if 'FRAGILE' in verdicts:
            return 'FRAGILE'
        if 'MODERATE' in verdicts:
            return 'MODERATE'
        return 'ROBUST'

    def _generate_report(self, signal_id, results):
        """
        Returns:
        - overall_fragility: ROBUST | MODERATE | FRAGILE
        - required_sharpe_threshold: adjusted pass bar
        - recommendation: PROCEED | HIGHER_BAR | ARCHIVE
        """
        if not results:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'ROBUST',
                'required_sharpe_threshold': None,
                'recommendation': 'PROCEED',
                'note': 'No numerical parameters to test.',
                'param_details': {},
            }

        fragilities = [r['fragility'] for r in results.values()]

        if 'FRAGILE' in fragilities:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'FRAGILE',
                'required_sharpe_threshold': 2.0,
                'recommendation': 'HIGHER_BAR',
                'note': ('Edge exists only at exact book parameter. '
                         'Requires Sharpe > 2.0 to compensate.'),
                'param_details': results,
            }
        elif 'MODERATE' in fragilities:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'MODERATE',
                'required_sharpe_threshold': 1.6,
                'recommendation': 'PROCEED',
                'note': ('Edge degrades at parameter extremes. '
                         'Use exact book parameter in live system.'),
                'param_details': results,
            }
        else:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'ROBUST',
                'required_sharpe_threshold': None,
                'recommendation': 'PROCEED',
                'note': ('Edge robust across parameter range. '
                         'Book parameter confirmed optimal.'),
                'param_details': results,
            }
