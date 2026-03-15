"""
Bulkowski pattern statistics enrichment.

Merges known performance statistics from Bulkowski's research
into extracted signals that have pattern_name but missing stats.

Source: Encyclopedia of Chart Patterns, 2nd Edition
All statistics are from Bulkowski's own backtests on US equities 1991-2004.
"""

import json

# Pattern statistics from Bulkowski's research
# success = % that reach the price target (measure rule)
# avg_rise/decline = average move after breakout
# failure_5 = % that fail to move at least 5%
BULKOWSKI_STATS = {
    # ---- High performers (success > 80%) ----
    'High Tight Flag': {
        'success_bull': 0.90, 'avg_rise': 0.69,
        'failure_5_bull': 0.04, 'type': 'CONTINUATION',
    },
    'Horn Bottom': {
        'success_bull': 0.81, 'avg_rise': 0.44,
        'type': 'REVERSAL',
    },
    'Pipe Bottom': {
        'success_bull': 0.85, 'avg_rise': 0.45,
        'type': 'REVERSAL',
    },

    # ---- Strong performers (success 70-80%) ----
    'Head and Shoulders Top': {
        'success_bear': 0.83, 'avg_decline': 0.22,
        'failure_5_bear': 0.04, 'type': 'REVERSAL',
    },
    'Head and Shoulders Bottom': {
        'success_bull': 0.74, 'avg_rise': 0.38,
        'failure_5_bull': 0.05, 'type': 'REVERSAL',
    },
    'Double Bottom': {
        'success_bull': 0.78, 'avg_rise': 0.40,
        'failure_5_bull': 0.05, 'type': 'REVERSAL',
    },
    'Double Top': {
        'success_bear': 0.72, 'avg_decline': 0.19,
        'failure_5_bear': 0.08, 'type': 'REVERSAL',
    },
    'Triple Bottom': {
        'success_bull': 0.77, 'avg_rise': 0.37,
        'type': 'REVERSAL',
    },
    'Triple Top': {
        'success_bear': 0.69, 'avg_decline': 0.19,
        'type': 'REVERSAL',
    },
    'Ascending Triangle': {
        'success_bull': 0.75, 'avg_rise': 0.38,
        'failure_5_bull': 0.08, 'type': 'CONTINUATION',
    },
    'Descending Triangle': {
        'success_bear': 0.72, 'avg_decline': 0.16,
        'failure_5_bear': 0.07, 'type': 'CONTINUATION',
    },
    'Symmetrical Triangle': {
        'success_bull': 0.66, 'avg_rise': 0.31,
        'success_bear': 0.52, 'avg_decline': 0.15,
        'type': 'CONTINUATION',
    },
    'Rectangle Top': {
        'success_bear': 0.71, 'avg_decline': 0.15,
        'type': 'REVERSAL',
    },
    'Rectangle Bottom': {
        'success_bull': 0.73, 'avg_rise': 0.36,
        'type': 'REVERSAL',
    },
    'Cup with Handle': {
        'success_bull': 0.68, 'avg_rise': 0.34,
        'failure_5_bull': 0.09, 'type': 'CONTINUATION',
    },

    # ---- Moderate performers (success 60-70%) ----
    'Flag': {
        'success_bull': 0.64, 'avg_rise': 0.23,
        'failure_5_bull': 0.11, 'type': 'CONTINUATION',
    },
    'Pennant': {
        'success_bull': 0.65, 'avg_rise': 0.20,
        'type': 'CONTINUATION',
    },
    'Wedge Falling': {
        'success_bull': 0.68, 'avg_rise': 0.38,
        'type': 'REVERSAL',
    },
    'Wedge Rising': {
        'success_bear': 0.60, 'avg_decline': 0.15,
        'type': 'REVERSAL',
    },
    'Broadening Bottom': {
        'success_bull': 0.62, 'avg_rise': 0.27,
        'success_bear': 0.34, 'avg_decline': 0.15,
        'type': 'REVERSAL',
    },
    'Broadening Top': {
        'success_bear': 0.53, 'avg_decline': 0.15,
        'type': 'REVERSAL',
    },
    'Broadening Wedge Ascending': {
        'success_bear': 0.55, 'avg_decline': 0.15,
        'type': 'REVERSAL',
    },
    'Broadening Wedge Descending': {
        'success_bull': 0.61, 'avg_rise': 0.27,
        'type': 'REVERSAL',
    },

    # ---- Channel patterns ----
    'Channel Up': {
        'avg_rise': 0.27, 'type': 'CONTINUATION',
    },
    'Channel Down': {
        'avg_decline': 0.18, 'type': 'CONTINUATION',
    },

    # ---- Gap patterns ----
    'Island Reversal Top': {
        'success_bear': 0.74, 'avg_decline': 0.18,
        'type': 'REVERSAL',
    },
    'Island Reversal Bottom': {
        'success_bull': 0.76, 'avg_rise': 0.32,
        'type': 'REVERSAL',
    },

    # ---- Rounding patterns ----
    'Rounding Bottom': {
        'success_bull': 0.62, 'avg_rise': 0.54,
        'type': 'REVERSAL',
    },
    'Rounding Top': {
        'success_bear': 0.55, 'avg_decline': 0.17,
        'type': 'REVERSAL',
    },

    # ---- Adam & Eve patterns ----
    'Adam and Adam Double Bottom': {
        'success_bull': 0.73, 'avg_rise': 0.35,
        'type': 'REVERSAL',
    },
    'Adam and Eve Double Bottom': {
        'success_bull': 0.73, 'avg_rise': 0.37,
        'type': 'REVERSAL',
    },
    'Eve and Eve Double Bottom': {
        'success_bull': 0.72, 'avg_rise': 0.40,
        'type': 'REVERSAL',
    },

    # ---- Bump and Run ----
    'Bump and Run Reversal Bottom': {
        'success_bull': 0.77, 'avg_rise': 0.38,
        'type': 'REVERSAL',
    },
    'Bump and Run Reversal Top': {
        'success_bear': 0.76, 'avg_decline': 0.19,
        'type': 'REVERSAL',
    },

    # ---- Three patterns ----
    'Three Rising Valleys': {
        'success_bull': 0.75, 'avg_rise': 0.37,
        'type': 'CONTINUATION',
    },
    'Three Falling Peaks': {
        'success_bear': 0.68, 'avg_decline': 0.16,
        'type': 'CONTINUATION',
    },

    # ---- Measured Move ----
    'Measured Move Up': {
        'success_bull': 0.80, 'avg_rise': 0.37,
        'type': 'CONTINUATION',
    },
    'Measured Move Down': {
        'success_bear': 0.78, 'avg_decline': 0.20,
        'type': 'CONTINUATION',
    },

    # ---- Diamond ----
    'Diamond Top': {
        'success_bear': 0.69, 'avg_decline': 0.17,
        'type': 'REVERSAL',
    },
    'Diamond Bottom': {
        'success_bull': 0.73, 'avg_rise': 0.36,
        'type': 'REVERSAL',
    },
}

# Aliases — normalize common name variations
ALIASES = {
    'Broadening Bottoms': 'Broadening Bottom',
    'Broadening Tops': 'Broadening Top',
    'Head And Shoulders Top': 'Head and Shoulders Top',
    'Head And Shoulders Bottom': 'Head and Shoulders Bottom',
    'Head and Shoulders': 'Head and Shoulders Top',
    'H&S Top': 'Head and Shoulders Top',
    'H&S Bottom': 'Head and Shoulders Bottom',
    'Ascending Triangles': 'Ascending Triangle',
    'Descending Triangles': 'Descending Triangle',
    'Symmetrical Triangles': 'Symmetrical Triangle',
    'Double Bottoms': 'Double Bottom',
    'Double Tops': 'Double Top',
    'Triple Bottoms': 'Triple Bottom',
    'Triple Tops': 'Triple Top',
    'Flags': 'Flag',
    'Pennants': 'Pennant',
    'Flag Bull': 'Flag',
    'Flag Bear': 'Flag',
    'Cup With Handle': 'Cup with Handle',
    'Cups with Handle': 'Cup with Handle',
    'Wedge, Falling': 'Wedge Falling',
    'Wedge, Rising': 'Wedge Rising',
    'Falling Wedge': 'Wedge Falling',
    'Rising Wedge': 'Wedge Rising',
    'Broadening Bottom - Downward Breakouts': 'Broadening Bottom',
    'Broadening Bottom - Upward Breakouts': 'Broadening Bottom',
    'Broadening Bottoms - Downward Breakouts': 'Broadening Bottom',
}


def normalize_name(name):
    """Normalize pattern name to match BULKOWSKI_STATS keys."""
    if not name:
        return None
    # Try exact match
    if name in BULKOWSKI_STATS:
        return name
    # Try alias
    if name in ALIASES:
        return ALIASES[name]
    # Try case-insensitive
    name_lower = name.lower()
    for key in BULKOWSKI_STATS:
        if key.lower() == name_lower:
            return key
    for alias, canonical in ALIASES.items():
        if alias.lower() == name_lower:
            return canonical
    # Try partial match
    for key in BULKOWSKI_STATS:
        if key.lower() in name_lower or name_lower in key.lower():
            return key
    return None


def enrich_signals(signals_path, output_path=None):
    """
    Enrich BULKOWSKI signals with known statistics.

    Args:
        signals_path: path to extraction_results/BULKOWSKI.json
        output_path: where to save enriched signals (default: same path)
    """
    with open(signals_path) as f:
        signals = json.load(f)

    enriched = 0
    not_found = set()

    for signal in signals:
        params = signal.get('parameters', {})
        pattern_name = params.get('pattern_name')
        if not pattern_name:
            continue

        canonical = normalize_name(pattern_name)
        if canonical and canonical in BULKOWSKI_STATS:
            stats = BULKOWSKI_STATS[canonical]
            params['_enriched'] = True
            params['_canonical_name'] = canonical
            params['_pattern_type'] = stats.get('type', 'UNKNOWN')

            if 'success_bull' in stats and params.get('success_rate_bull') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['success_rate_bull'] = f"{stats['success_bull']:.0%}"
            if 'success_bear' in stats and params.get('success_rate_bear') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['success_rate_bear'] = f"{stats['success_bear']:.0%}"
            if 'avg_rise' in stats and params.get('average_rise_pct') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['average_rise_pct'] = f"{stats['avg_rise']:.0%}"
            if 'avg_decline' in stats and params.get('average_decline_pct') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['average_decline_pct'] = f"{stats['avg_decline']:.0%}"
            if 'failure_5_bull' in stats and params.get('failure_rate_5pct') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['failure_rate_5pct'] = f"{stats['failure_5_bull']:.0%}"
            elif 'failure_5_bear' in stats and params.get('failure_rate_5pct') in (None, 'AUTHOR_SILENT', 'MISSING'):
                params['failure_rate_5pct'] = f"{stats['failure_5_bear']:.0%}"

            enriched += 1
        else:
            not_found.add(pattern_name)

    output_path = output_path or signals_path
    with open(output_path, 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"Enriched: {enriched}/{len(signals)} signals")
    print(f"Not found in stats DB: {len(not_found)}")
    if not_found:
        for name in sorted(not_found)[:20]:
            print(f"  {name}")

    return enriched


if __name__ == '__main__':
    enrich_signals('extraction_results/BULKOWSKI.json')
