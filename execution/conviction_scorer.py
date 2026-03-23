"""
Conviction Score Engine — identifies high-conviction conditions and sizes up to 1.5-2.0x.

Reads overlay modifiers from OverlayPipeline.get_modifiers() and scores how
aligned the market environment is for bulls or bears. High conviction in
the trade direction means larger position sizes; conflicting signals keep
size at baseline or reduce it.

The conviction modifier feeds into LotSizer as an additional multiplier on
top of the hierarchical overlay composite.

Usage:
    from execution.conviction_scorer import ConvictionScorer
    scorer = ConvictionScorer()
    result = scorer.compute(modifiers, vix=14.5, adx=28, direction='LONG',
                            open_positions=1, drawdown_pct=2.0,
                            consecutive_losses=0)
    # result['final_modifier'] -> 1.30  (amplify)
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal thresholds used in conviction scoring
# ---------------------------------------------------------------------------

# Bull conviction scoring rules: (signal_id, threshold, comparator, points)
_BULL_ADDITIVE = [
    ('FII_FUTURES_OI',       1.10, 'gt',  20),
    ('MAMBA_REGIME',         1.05, 'gt',  15),
    ('PCR_AUTOTRENDER',      1.10, 'gt',  15),
    ('DELIVERY_PCT',         1.10, 'gt',  10),
    ('GAMMA_EXPOSURE',       1.05, 'gt',  10),
    ('VOL_TERM_STRUCTURE',   1.05, 'gt',  10),
    ('AMFI_MF_FLOW',         1.05, 'gt',  10),
    ('NLP_SENTIMENT',        1.00, 'gt',   5),
    ('BOND_YIELD_SPREAD',    1.00, 'gte',  5),
    ('GNN_SECTOR_ROTATION',  1.00, 'gt',   5),
    ('CREDIT_CARD_SPENDING', 1.00, 'gt',   5),
]

_BULL_PENALTIES = [
    ('RBI_MACRO_FILTER',     0.70, 'lt', -30),
    ('VOL_TERM_STRUCTURE',   0.90, 'lt', -15),
]

# Bear conviction scoring rules (mirror of bull)
_BEAR_ADDITIVE = [
    ('FII_FUTURES_OI',       0.90, 'lt',  20),
    ('MAMBA_REGIME',         0.95, 'lt',  15),
    ('PCR_AUTOTRENDER',      0.90, 'lt',  15),
    ('DELIVERY_PCT',         0.90, 'lt',  10),
    ('GAMMA_EXPOSURE',       0.95, 'lt',  10),
    ('VOL_TERM_STRUCTURE',   0.95, 'lt',  10),
    ('AMFI_MF_FLOW',         0.95, 'lt',  10),
    ('NLP_SENTIMENT',        1.00, 'lt',   5),
    ('BOND_YIELD_SPREAD',    1.00, 'lt',   5),
    ('GNN_SECTOR_ROTATION',  1.00, 'lt',   5),
    ('CREDIT_CARD_SPENDING', 1.00, 'lt',   5),
]

_BEAR_PENALTIES = [
    ('RBI_MACRO_FILTER',     0.70, 'lt', -30),
    ('VOL_TERM_STRUCTURE',   1.10, 'gt', -15),
]

# Bull VIX penalty
_VIX_HIGH_PENALTY_BULL = 20   # vix threshold
_VIX_HIGH_POINTS_BULL = -20   # deduct points

# Bear VIX bonus (high VIX helps bears)
_VIX_HIGH_BONUS_BEAR = 20
_VIX_HIGH_POINTS_BEAR = 20

# Score-to-modifier mapping (lowered thresholds — typical scores are 30-50
# because 11/22 overlays return 1.0 stubs)
_BULL_MODIFIER_TABLE = [
    (70, 100, 1.40),
    (55,  69, 1.25),
    (40,  54, 1.12),
    (25,  39, 1.05),
    ( 0,  24, 1.00),
]

_BEAR_MODIFIER_TABLE = [
    (70, 100, 0.60),
    (55,  69, 0.75),
    (40,  54, 0.88),
    (25,  39, 0.95),
    ( 0,  24, 1.00),
]

# Safeguard limits
_MAX_POSITIONS_FOR_FULL_AMP = 3
_MAX_POSITIONS_CAP = 1.30
_DRAWDOWN_THRESHOLD_PCT = 8.0
_DRAWDOWN_CAP = 1.00
_CONSEC_LOSS_THRESHOLD = 3
_CONSEC_LOSS_CAP = 1.15
_MODIFIER_FLOOR = 0.30
_MODIFIER_CEILING = 2.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_condition(value: float, threshold: float, comparator: str) -> bool:
    """Evaluate a single threshold comparison."""
    if comparator == 'gt':
        return value > threshold
    elif comparator == 'gte':
        return value >= threshold
    elif comparator == 'lt':
        return value < threshold
    elif comparator == 'lte':
        return value <= threshold
    return False


def _score_rules(
    modifiers: Dict[str, float],
    additive_rules: list,
    penalty_rules: list,
) -> Tuple[int, Dict[str, int]]:
    """
    Score a set of additive and penalty rules against modifiers.
    Returns (raw_score, breakdown_dict).
    """
    score = 0
    breakdown: Dict[str, int] = {}

    for signal_id, threshold, comparator, points in additive_rules:
        val = modifiers.get(signal_id)
        if val is not None and _check_condition(val, threshold, comparator):
            score += points
            breakdown[signal_id] = points

    for signal_id, threshold, comparator, points in penalty_rules:
        val = modifiers.get(signal_id)
        if val is not None and _check_condition(val, threshold, comparator):
            score += points  # points are already negative
            breakdown[f'{signal_id}_penalty'] = points

    return score, breakdown


def _normalize_by_active(raw_score: int, modifiers: Dict[str, float]) -> int:
    """
    Normalize conviction score by fraction of active overlays.

    An overlay is "active" if its modifier != 1.0 (stubs return exactly 1.0).
    If only 5 of 11 overlays are active, the raw score earned from those 5
    is scaled up: score * (total / active).  This prevents stub overlays
    from diluting conviction when most signals are not yet implemented.

    Returns the adjusted integer score.
    """
    if not modifiers:
        return raw_score

    total = len(modifiers)
    active = sum(1 for v in modifiers.values() if v != 1.0)

    if active == 0 or active == total:
        # All stubs or all active — no adjustment needed
        return raw_score

    # Scale score as if all overlays were active
    scale = total / active
    return int(round(raw_score * scale))


def _lookup_modifier(score: int, table: list) -> float:
    """Map a 0-100 score to a modifier via bracket table."""
    for lo, hi, mod in table:
        if lo <= score <= hi:
            return mod
    return 1.0


# ---------------------------------------------------------------------------
# ConvictionScorer
# ---------------------------------------------------------------------------

class ConvictionScorer:
    """
    Scores market conviction (0-100) for bull and bear cases, then maps
    the score to a position-size modifier (0.4 – 1.75) with safeguards.
    """

    def __init__(self):
        self._consecutive_losses = 0
        self._drawdown_pct = 0.0

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def compute_bull_conviction(
        self,
        modifiers: Dict[str, float],
        vix: float,
        adx: float,
    ) -> int:
        """
        Compute bull conviction score (0-100) from overlay alignment.

        Scoring:
          +20 FII_FUTURES_OI > 1.1
          +15 MAMBA_REGIME > 1.05
          +15 PCR_AUTOTRENDER > 1.1 (extreme fear = contrarian bullish)
          +10 DELIVERY_PCT > 1.1
          +10 GAMMA_EXPOSURE > 1.05
          +10 VOL_TERM_STRUCTURE > 1.05 (contango)
          +10 AMFI_MF_FLOW > 1.05
          +5  NLP_SENTIMENT > 1.0
          +5  BOND_YIELD_SPREAD >= 1.0
          +5  GNN_SECTOR_ROTATION > 1.0
          +5  CREDIT_CARD_SPENDING > 1.0
          -30 RBI_MACRO_FILTER < 0.7 (event active)
          -20 vix > 20
          -15 VOL_TERM_STRUCTURE < 0.9 (backwardation)

        Score is normalized by active (non-stub) overlays so that stub
        overlays returning 1.0 don't dilute conviction.
        """
        score, _ = _score_rules(modifiers, _BULL_ADDITIVE, _BULL_PENALTIES)

        # VIX penalty for bulls
        if vix > _VIX_HIGH_PENALTY_BULL:
            score += _VIX_HIGH_POINTS_BULL

        # Normalize by active overlays: scale score as if all overlays
        # were active. An overlay is "active" if its modifier != 1.0.
        score = _normalize_by_active(score, modifiers)

        # Clamp to [0, 100]
        return max(0, min(100, score))

    def compute_bear_conviction(
        self,
        modifiers: Dict[str, float],
        vix: float,
        adx: float,
    ) -> int:
        """
        Compute bear conviction score (0-100) — mirror of bull logic.

        High bear conviction when overlays show institutional selling,
        regime stress, negative gamma, and backwardation.

        Score is normalized by active (non-stub) overlays.
        """
        score, _ = _score_rules(modifiers, _BEAR_ADDITIVE, _BEAR_PENALTIES)

        # VIX bonus for bears (high VIX supports short thesis)
        if vix > _VIX_HIGH_BONUS_BEAR:
            score += _VIX_HIGH_POINTS_BEAR

        # Normalize by active overlays
        score = _normalize_by_active(score, modifiers)

        return max(0, min(100, score))

    def score_to_modifier(
        self,
        bull_score: int,
        bear_score: int,
    ) -> float:
        """
        Map bull/bear scores to a single position-size modifier.

        Bull dominance (higher bull score) amplifies LONG and dampens SHORT.
        Bear dominance does the opposite. When both are low, modifier = 1.0.
        """
        # Determine dominant direction
        if bull_score > bear_score:
            return _lookup_modifier(bull_score, _BULL_MODIFIER_TABLE)
        elif bear_score > bull_score:
            return _lookup_modifier(bear_score, _BEAR_MODIFIER_TABLE)
        else:
            # Tie — neutral
            return 1.0

    def apply_safeguards(
        self,
        modifier: float,
        open_positions: int,
        drawdown_pct: float,
        consecutive_losses: int,
    ) -> Tuple[float, List[str]]:
        """
        Apply risk safeguards that cap the conviction modifier.

        Rules (applied in order):
          - open_positions >= 3  → cap at 1.3
          - drawdown > 8%       → cap at 1.0
          - consecutive_losses >= 3 → cap at 1.15
          - Always clamp to [0.3, 2.0]

        Returns:
            (final_modifier, list_of_safeguards_applied)
        """
        safeguards: List[str] = []

        if open_positions >= _MAX_POSITIONS_FOR_FULL_AMP and modifier > _MAX_POSITIONS_CAP:
            modifier = min(modifier, _MAX_POSITIONS_CAP)
            safeguards.append(
                f'open_positions({open_positions})>=3: capped at {_MAX_POSITIONS_CAP}'
            )

        if drawdown_pct > _DRAWDOWN_THRESHOLD_PCT and modifier > _DRAWDOWN_CAP:
            modifier = min(modifier, _DRAWDOWN_CAP)
            safeguards.append(
                f'drawdown({drawdown_pct:.1f}%)>8%: capped at {_DRAWDOWN_CAP}'
            )

        if consecutive_losses >= _CONSEC_LOSS_THRESHOLD and modifier > _CONSEC_LOSS_CAP:
            modifier = min(modifier, _CONSEC_LOSS_CAP)
            safeguards.append(
                f'consecutive_losses({consecutive_losses})>=3: capped at {_CONSEC_LOSS_CAP}'
            )

        # Hard clamp
        clamped = max(_MODIFIER_FLOOR, min(_MODIFIER_CEILING, modifier))
        if clamped != modifier:
            safeguards.append(
                f'clamped {modifier:.2f} -> [{_MODIFIER_FLOOR}, {_MODIFIER_CEILING}]'
            )
            modifier = clamped

        return modifier, safeguards

    def compute(
        self,
        modifiers: Dict[str, float],
        vix: float,
        adx: float,
        direction: str,
        open_positions: int = 0,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
    ) -> Dict:
        """
        Full conviction pipeline: score → modifier → safeguards.

        Args:
            modifiers: {signal_id: modifier} from OverlayPipeline
            vix: current India VIX
            adx: current ADX(14)
            direction: 'LONG' or 'SHORT'
            open_positions: number of open positions
            drawdown_pct: current drawdown from peak (%)
            consecutive_losses: streak of losing trades

        Returns:
            dict with keys:
              - bull_score (int 0-100)
              - bear_score (int 0-100)
              - raw_modifier (float)
              - final_modifier (float)
              - safeguards_applied (list[str])
              - breakdown (dict)
        """
        bull_score = self.compute_bull_conviction(modifiers, vix, adx)
        bear_score = self.compute_bear_conviction(modifiers, vix, adx)

        # For LONG trades: use bull conviction directly
        # For SHORT trades: swap perspective
        if direction == 'LONG':
            raw_modifier = self.score_to_modifier(bull_score, bear_score)
        else:
            # For shorts: high bear score should amplify (use bull table for bear score)
            raw_modifier = self.score_to_modifier(bear_score, bull_score)

        final_modifier, safeguards = self.apply_safeguards(
            raw_modifier, open_positions, drawdown_pct, consecutive_losses
        )

        # Build detailed breakdown
        _, bull_breakdown = _score_rules(modifiers, _BULL_ADDITIVE, _BULL_PENALTIES)
        _, bear_breakdown = _score_rules(modifiers, _BEAR_ADDITIVE, _BEAR_PENALTIES)

        if vix > _VIX_HIGH_PENALTY_BULL:
            bull_breakdown['vix_penalty'] = _VIX_HIGH_POINTS_BULL
        if vix > _VIX_HIGH_BONUS_BEAR:
            bear_breakdown['vix_bonus'] = _VIX_HIGH_POINTS_BEAR

        breakdown = {
            'bull_components': bull_breakdown,
            'bear_components': bear_breakdown,
            'direction': direction,
            'vix': vix,
            'adx': adx,
        }

        # Update internal state
        self._consecutive_losses = consecutive_losses
        self._drawdown_pct = drawdown_pct

        logger.debug(
            'Conviction: bull=%d bear=%d raw=%.2f final=%.2f safeguards=%s',
            bull_score, bear_score, raw_modifier, final_modifier, safeguards,
        )

        return {
            'bull_score': bull_score,
            'bear_score': bear_score,
            'raw_modifier': round(raw_modifier, 3),
            'final_modifier': round(final_modifier, 3),
            'safeguards_applied': safeguards,
            'breakdown': breakdown,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    scorer = ConvictionScorer()

    print('=' * 70)
    print('ConvictionScorer Self-Test')
    print('=' * 70)

    # ── Test 1: Strong bull environment ──
    bull_mods = {
        'FII_FUTURES_OI': 1.20,
        'MAMBA_REGIME': 1.10,
        'PCR_AUTOTRENDER': 1.25,
        'DELIVERY_PCT': 1.15,
        'GAMMA_EXPOSURE': 1.10,
        'VOL_TERM_STRUCTURE': 1.08,
        'AMFI_MF_FLOW': 1.10,
        'NLP_SENTIMENT': 1.05,
        'BOND_YIELD_SPREAD': 1.02,
        'GNN_SECTOR_ROTATION': 1.05,
        'CREDIT_CARD_SPENDING': 1.03,
        'RBI_MACRO_FILTER': 1.0,
    }
    result = scorer.compute(bull_mods, vix=13.5, adx=28, direction='LONG')
    print('\nTest 1: Strong bull environment (LONG)')
    print(f'  Bull score:      {result["bull_score"]}')
    print(f'  Bear score:      {result["bear_score"]}')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    print(f'  Safeguards:      {result["safeguards_applied"]}')
    assert result['bull_score'] >= 70, f'Expected bull >= 70, got {result["bull_score"]}'
    assert result['final_modifier'] >= 1.3, f'Expected mod >= 1.3, got {result["final_modifier"]}'

    # ── Test 2: High VIX with event filter ──
    stress_mods = {
        'FII_FUTURES_OI': 0.80,
        'MAMBA_REGIME': 0.65,
        'PCR_AUTOTRENDER': 0.80,
        'RBI_MACRO_FILTER': 0.50,
        'VOL_TERM_STRUCTURE': 0.85,
        'GAMMA_EXPOSURE': 0.70,
    }
    result = scorer.compute(stress_mods, vix=26, adx=35, direction='LONG')
    print('\nTest 2: Stress environment (LONG)')
    print(f'  Bull score:      {result["bull_score"]}')
    print(f'  Bear score:      {result["bear_score"]}')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    assert result['bull_score'] <= 10, f'Expected bull <= 10, got {result["bull_score"]}'

    # ── Test 3: Safeguards — drawdown cap ──
    result = scorer.compute(
        bull_mods, vix=13, adx=28, direction='LONG',
        open_positions=1, drawdown_pct=10.0, consecutive_losses=0,
    )
    print('\nTest 3: Drawdown 10% safeguard')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    print(f'  Safeguards:      {result["safeguards_applied"]}')
    assert result['final_modifier'] <= 1.0, f'Expected <= 1.0, got {result["final_modifier"]}'

    # ── Test 4: Safeguards — open positions cap ──
    result = scorer.compute(
        bull_mods, vix=13, adx=28, direction='LONG',
        open_positions=4, drawdown_pct=2.0, consecutive_losses=0,
    )
    print('\nTest 4: 4 open positions safeguard')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    print(f'  Safeguards:      {result["safeguards_applied"]}')
    assert result['final_modifier'] <= 1.3, f'Expected <= 1.3, got {result["final_modifier"]}'

    # ── Test 5: Safeguards — consecutive losses cap ──
    result = scorer.compute(
        bull_mods, vix=13, adx=28, direction='LONG',
        open_positions=1, drawdown_pct=2.0, consecutive_losses=4,
    )
    print('\nTest 5: 4 consecutive losses safeguard')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    print(f'  Safeguards:      {result["safeguards_applied"]}')
    assert result['final_modifier'] <= 1.15, f'Expected <= 1.15, got {result["final_modifier"]}'

    # ── Test 6: SHORT direction with bear conviction ──
    bear_mods = {
        'FII_FUTURES_OI': 0.75,
        'MAMBA_REGIME': 0.80,
        'PCR_AUTOTRENDER': 0.70,
        'DELIVERY_PCT': 0.80,
        'GAMMA_EXPOSURE': 0.85,
        'VOL_TERM_STRUCTURE': 0.80,
        'AMFI_MF_FLOW': 0.90,
        'NLP_SENTIMENT': 0.90,
        'BOND_YIELD_SPREAD': 0.90,
        'GNN_SECTOR_ROTATION': 0.90,
        'CREDIT_CARD_SPENDING': 0.90,
        'RBI_MACRO_FILTER': 1.0,
    }
    result = scorer.compute(bear_mods, vix=24, adx=32, direction='SHORT')
    print('\nTest 6: Bear environment (SHORT)')
    print(f'  Bull score:      {result["bull_score"]}')
    print(f'  Bear score:      {result["bear_score"]}')
    print(f'  Raw modifier:    {result["raw_modifier"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    assert result['bear_score'] >= 50, f'Expected bear >= 50, got {result["bear_score"]}'

    # ── Test 7: Neutral / no alignment ──
    neutral_mods = {k: 1.0 for k in bull_mods}
    result = scorer.compute(neutral_mods, vix=15, adx=20, direction='LONG')
    print('\nTest 7: Neutral environment')
    print(f'  Bull score:      {result["bull_score"]}')
    print(f'  Bear score:      {result["bear_score"]}')
    print(f'  Final modifier:  {result["final_modifier"]}')
    assert result['final_modifier'] == 1.0, f'Expected 1.0, got {result["final_modifier"]}'

    # ── Test 8: score_to_modifier bracket boundaries (lowered thresholds) ──
    print('\nTest 8: Score bracket boundaries')
    for bull, bear, expected in [
        (0, 0, 1.0), (24, 0, 1.0), (25, 0, 1.05), (39, 0, 1.05),
        (40, 0, 1.12), (54, 0, 1.12), (55, 0, 1.25), (69, 0, 1.25),
        (70, 0, 1.40), (100, 0, 1.40),
        (0, 25, 0.95), (0, 40, 0.88), (0, 55, 0.75), (0, 70, 0.60),
    ]:
        got = scorer.score_to_modifier(bull, bear)
        status = 'OK' if got == expected else 'FAIL'
        print(f'  bull={bull:3d} bear={bear:3d} -> {got:.2f} (expected {expected:.2f}) [{status}]')
        assert got == expected, f'bull={bull} bear={bear}: got {got}, expected {expected}'

    print('\n' + '=' * 70)
    print('All ConvictionScorer tests passed.')
    print('=' * 70)
