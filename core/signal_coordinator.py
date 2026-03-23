"""
Signal Coordinator -- deduplication, category limits, and conflict resolution.

Operates on a list of signal candidates (dicts with at minimum:
signal_name, direction, confidence) and returns a filtered list
ready for sizing and execution.

Usage:
    from core.signal_coordinator import SignalCoordinator
    coord = SignalCoordinator()
    filtered = coord.process(candidates)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known high-correlation pairs (backtested correlation > 0.6)
# From each pair, only the highest-confidence signal survives.
# ---------------------------------------------------------------------------
HIGH_CORRELATION_PAIRS = [
    ('KAUFMAN_DRY_16', 'KAUFMAN_DRY_12'),
    ('GUJRAL_DRY_8', 'GUJRAL_DRY_13'),
    ('MAX_OI_BARRIER', 'THURSDAY_PIN_SETUP'),
]

# ---------------------------------------------------------------------------
# Signal categories -- used for per-category position limits
# ---------------------------------------------------------------------------
SIGNAL_CATEGORIES: Dict[str, List[str]] = {
    'KAUFMAN_MOMENTUM': [
        'KAUFMAN_DRY_20', 'KAUFMAN_DRY_16', 'KAUFMAN_DRY_12',
        'KAUFMAN_DRY_7',
    ],
    'GUJRAL_PIVOT': [
        'GUJRAL_DRY_8', 'GUJRAL_DRY_13', 'GUJRAL_DRY_9',
    ],
    'BULKOWSKI_PATTERN': [
        'BULKOWSKI_ADAM_AND_EVE_OR', 'BULKOWSKI_CUP_HANDLE',
        'BULKOWSKI_ROUND_BOTTOM_RDB_PATTERN',
        'BULKOWSKI_EADT_BUSTED_PATTERN',
        'BULKOWSKI_FALLING_VOLUME_TREND_IN',
        'BULKOWSKI_EADB_EARLY_ATTEMPT_TO',
    ],
    'CHAN_QUANT': [
        'CHAN_AT_DRY_4',
    ],
    'INTRADAY_STRUCTURE': [
        'EXPIRY_PIN_FADE', 'ORR_REVERSION',
        'MAX_OI_BARRIER', 'THURSDAY_PIN_SETUP',
        'ID_GAMMA_BREAKOUT', 'ID_GAMMA_REVERSAL',
    ],
    'CANDLESTICK': [
        'CANDLESTICK_DRY_0',
    ],
    'WEEKLY_SEASONAL': [
        'SSRN_WEEKLY_MOM',
    ],
}

# Reverse lookup: signal -> category
_SIGNAL_TO_CATEGORY: Dict[str, str] = {}
for _cat, _sigs in SIGNAL_CATEGORIES.items():
    for _sig in _sigs:
        _SIGNAL_TO_CATEGORY[_sig] = _cat


class SignalCoordinator:
    """
    Filters a list of candidate signals through three stages:

    1. **Deduplication**: From known correlated pairs, keep only the
       candidate with the highest confidence.
    2. **Category limits**: At most ``max_per_category`` signals from
       the same family (e.g. max 2 Kaufman signals at once).
    3. **Conflict resolution**: If both BULLISH and BEARISH candidates
       remain, keep only the higher-confidence direction.
    """

    def __init__(self, max_per_category: int = 2):
        self.max_per_category = max_per_category

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, candidates: List[Dict]) -> List[Dict]:
        """
        Run all three coordination stages on *candidates*.

        Each candidate is a dict with at least:
            signal_name : str
            direction   : str   ('BULLISH' or 'BEARISH')
            confidence  : float (0.0 - 1.0)

        Returns a (possibly smaller) list of candidates.
        """
        if not candidates:
            return []

        result = list(candidates)
        result = self.deduplicate(result)
        result = self.enforce_category_limits(result, self.max_per_category)
        result = self.resolve_conflicts(result)
        return result

    # ------------------------------------------------------------------
    # Stage 1: Deduplicate correlated pairs
    # ------------------------------------------------------------------

    def deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        """
        From each known correlated pair, keep only the candidate with
        the higher confidence score.  If both members of a pair are
        present, the weaker one is dropped.
        """
        names = {c['signal_name'] for c in candidates}
        to_drop = set()

        for sig_a, sig_b in HIGH_CORRELATION_PAIRS:
            if sig_a in names and sig_b in names:
                cand_a = next(c for c in candidates if c['signal_name'] == sig_a)
                cand_b = next(c for c in candidates if c['signal_name'] == sig_b)
                conf_a = cand_a.get('confidence', 0.0)
                conf_b = cand_b.get('confidence', 0.0)
                loser = sig_b if conf_a >= conf_b else sig_a
                to_drop.add(loser)
                logger.info(
                    'Dedup: dropping %s (conf=%.2f) in favour of %s (conf=%.2f)',
                    loser,
                    min(conf_a, conf_b),
                    sig_a if loser == sig_b else sig_b,
                    max(conf_a, conf_b),
                )

        if to_drop:
            candidates = [c for c in candidates if c['signal_name'] not in to_drop]
        return candidates

    # ------------------------------------------------------------------
    # Stage 2: Enforce per-category position limits
    # ------------------------------------------------------------------

    def enforce_category_limits(
        self,
        candidates: List[Dict],
        max_per_category: int = 2,
    ) -> List[Dict]:
        """
        Keep at most *max_per_category* candidates from the same
        signal family.  Within a category, higher-confidence candidates
        are preferred.
        """
        # Group by category
        by_cat: Dict[str, List[Dict]] = {}
        uncategorised: List[Dict] = []

        for c in candidates:
            cat = _SIGNAL_TO_CATEGORY.get(c['signal_name'])
            if cat:
                by_cat.setdefault(cat, []).append(c)
            else:
                uncategorised.append(c)

        kept: List[Dict] = list(uncategorised)
        for cat, members in by_cat.items():
            # Sort descending by confidence, keep top N
            members.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            kept_members = members[:max_per_category]
            if len(members) > max_per_category:
                dropped = [m['signal_name'] for m in members[max_per_category:]]
                logger.info(
                    'Category %s limit (%d): dropped %s',
                    cat, max_per_category, dropped,
                )
            kept.extend(kept_members)

        return kept

    # ------------------------------------------------------------------
    # Stage 3: Resolve directional conflicts
    # ------------------------------------------------------------------

    def resolve_conflicts(self, candidates: List[Dict]) -> List[Dict]:
        """
        If both BULLISH and BEARISH candidates are present, keep only
        the direction with the higher aggregate confidence.
        """
        if len(candidates) <= 1:
            return candidates

        bulls = [c for c in candidates if c.get('direction') == 'BULLISH']
        bears = [c for c in candidates if c.get('direction') == 'BEARISH']
        neutral = [c for c in candidates if c.get('direction') not in ('BULLISH', 'BEARISH')]

        if bulls and bears:
            bull_conf = sum(c.get('confidence', 0.0) for c in bulls)
            bear_conf = sum(c.get('confidence', 0.0) for c in bears)

            if bull_conf >= bear_conf:
                logger.info(
                    'Conflict: BULLISH (%.2f) beats BEARISH (%.2f), dropping %d bear signals',
                    bull_conf, bear_conf, len(bears),
                )
                return bulls + neutral
            else:
                logger.info(
                    'Conflict: BEARISH (%.2f) beats BULLISH (%.2f), dropping %d bull signals',
                    bear_conf, bull_conf, len(bulls),
                )
                return bears + neutral

        return candidates
