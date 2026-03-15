# Combination Testing Report
Generated: 2026-03-14 22:10

## Executive Summary
- Signals tested: 15
- Combinations screened: 20 survivors
- Walk-forward tested: 20
- Tier A: 2
- Tier B: 2
- OOS validated: 3

## Tier A/B Combinations

| Sig A | Sig B | Logic | Sharpe | WR | WF% | L4 | Tier |
|-------|-------|-------|--------|-----|-----|-----|------|
| GRIMES_DRY_3_2 | KAUFMAN_DRY_12 | SEQ_5 | 1.92 | 50% | 73% | 4/4 | TIER_A |
| KAUFMAN_DRY_16 | KAUFMAN_DRY_8 | AND | 2.85 | 54% | 85% | 4/4 | TIER_A |
| GRIMES_DRY_6_0 | GUJRAL_DRY_8 | AND | 13.17 | 65% | 50% | 3/4 | TIER_B |
| GRIMES_DRY_3_2 | KAUFMAN_DRY_12 | SEQ_3 | 1.47 | 49% | 62% | 4/4 | TIER_B |

## OOS Validated

- **GRIMES_DRY_3_2 + KAUFMAN_DRY_12** (SEQ_5): OOS Sharpe=5.08, Recent P&L=+0
- **GRIMES_DRY_6_0 + GUJRAL_DRY_8** (AND): OOS Sharpe=6.95, Recent P&L=+0
- **GRIMES_DRY_3_2 + KAUFMAN_DRY_12** (SEQ_3): OOS Sharpe=6.10, Recent P&L=+0

## Baselines
- DRY_20 alone: Sharpe 2.60, WR 47%, MaxDD 24.3%, Recent +60pts
- SCORING system: Sharpe 2.30, MaxDD 12.2%, Recent +340pts
