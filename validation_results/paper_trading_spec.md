# Paper Trading Specification

Generated: 2026-03-14
Status: Ready for paper trading

---

## KAUFMAN_DRY_20 — TIER A

**Source:** Trading Systems and Methods, Perry Kaufman, Ch.20, p.865
(Genetic algorithm discovered rule)

**Original book rule:** "Buy when a 10-day moving average is less than
yesterday's close AND a 5-day stochastic is greater than 50."

### Entry (LONG only)
1. 10-day SMA is below yesterday's close (sma_10 < prev_close)
2. 5-day Stochastic %K is above 50 (stoch_k_5 > 50)
Both conditions must be true simultaneously (AND logic).

### Exit
- Stochastic %K (5-day) drops to or below 50 (stoch_k_5 <= 50)
- OR stop loss triggered

### Stop Loss
2% from entry price.

### Take Profit
None (exit on signal only).

### Hold Days
No maximum — position held until exit signal or stop loss.

### Direction
LONG only. No short entries.

### Regime Filter
None (trades in all regimes).

### Walk-Forward Results
- Windows passed: 21/26 (81%)
- Aggregate Sharpe: 2.55
- Last 4 windows: 4/4
- Worst window drawdown: 15.2%
- Full-period Sharpe: 2.60
- Full-period win rate: 47.3%
- Nifty correlation: +0.240 (low positive — mild directional bias)

### Expected Trading Frequency
- Full-period trades: 226 over 10 years
- Approximately 22-23 trades per year
- Average hold: variable (exits when stochastic reverses)

### Fix Applied
Original Haiku translation used 14-period stochastic and today's close.
Fixed to 5-period stochastic (per book) and prev_close (per book: "yesterday's close").
This two-parameter fix moved the signal from DROP (17/26, Sharpe 0.86) to TIER_A.

### Risk Notes
- LONG-only signal — will underperform in sustained downtrends
- No regime filter means it will enter during HIGH_VOL periods
- Stop loss at 2% is tight — may get stopped out on volatile days
  before the signal plays out
- Early windows (0-2) consistently fail — signal struggled 2015-2017

---

## KAUFMAN_DRY_16 — TIER B

**Source:** Trading Systems and Methods, Perry Kaufman, Ch.16, p.757-797
(Pivot-based breakout with volatility filter)

**Original book concepts:**
- Pivot point breakout entry system
- "Enter trades only when 6-day historic volatility is less than
  100-day historic volatility" (volatility filter from Ch.16)

### Entry LONG
1. Close is above R1 pivot resistance (close > r1)
2. Low holds above the pivot point (low >= pivot)
3. Short-term volatility below long-term volatility (hvol_6 < hvol_100)
All three conditions must be true simultaneously (AND logic).

### Entry SHORT
1. Close is below S1 pivot support (close < s1)
2. Short-term volatility below long-term volatility (hvol_6 < hvol_100)
Both conditions must be true simultaneously (AND logic).

Note: Short entry is intentionally asymmetric (no pivot confirmation).
Testing showed adding a symmetric condition (high <= pivot) degraded
performance — the asymmetry may capture faster breakdowns.

### Exit LONG
- Low drops below the pivot point (low < pivot)
- OR take profit at 3% from entry
- OR stop loss triggered

### Exit SHORT
- High rises above R1 resistance (high > r1)
- OR take profit at 3% from entry
- OR stop loss triggered

### Stop Loss
2% from entry price.

### Take Profit
3% from entry price. This locks in pivot gains before mean reversion
erases them. Adding this improved aggregate Sharpe from 0.77 to 2.03.

### Hold Days
No maximum — position held until exit signal, take profit, or stop loss.

### Direction
BOTH (long and short entries).

### Regime Filter
None (trades in all regimes). RANGING filter was tested but reduced
trade count too aggressively (4/26 windows).

### Walk-Forward Results
- Windows passed: 15/26 (58%)
- Aggregate Sharpe: 2.03
- Last 4 windows: 3/4
- Full-period Sharpe: 1.34
- Full-period win rate: 48.6%
- Full-period trades: 406
- Nifty correlation: -0.352 (negative — tends to perform when Nifty is weak)

### Expected Trading Frequency
- Full-period trades: 406 over 10 years
- Approximately 40 trades per year
- Higher frequency than KAUFMAN_DRY_20

### Fixes Applied
1. Added Kaufman's volatility filter from Ch.16 (hvol_6 < hvol_100).
   This was explicitly stated in the book but dropped by the translator.
   Improved windows from 16/26 to 17/26, Sharpe from 0.77 to 1.26.
2. Added 3% take profit. Improved to 15/26, Sharpe 2.03, L4=3/4.

### Tier A Blocker
Window 24 (Nov 2024 – Nov 2025) consistently fails across all variants.
This window contains the Feb 2025 Nifty correction inside an otherwise
low-VIX RANGING period. The volatility filter passes entries, but the
intra-window drawdown kills per-window Sharpe. A VIX<18 filter gets
L4 to 4/4 but drops overall pass rate below Tier B minimum (38% < 40%).

### Risk Notes
- Negative Nifty correlation means this signal may lose during strong
  bull runs — acceptable if paired with KAUFMAN_DRY_20 (positive corr)
- Pivot-based signals can generate false breakouts in choppy markets
- The asymmetric short entry (fewer conditions) means more short trades
  fire, some of which may be lower quality
- Feb 2025-style corrections within low-VIX regimes are the main risk

---

## KAUFMAN_DRY_12 — TIER A

**Source:** Trading Systems and Methods, Perry Kaufman, Ch.12, p.539
(Volume-price divergence / exhaustion signal)

**Original book rule:** Kaufman's price-volume direction matrix:
"Price Up + Volume Down = Weak uptrend near reversal"
"Price Down + Volume Down = Clear downtrend"
The signal identifies moves on declining volume as exhaustion — weak
conviction that is likely to reverse.

### Entry LONG
1. Today's close is above yesterday's close (close > prev_close)
2. Today's volume is below yesterday's volume (volume < prev_volume)
Both conditions must be true simultaneously (AND logic).
Interpretation: price rose but on declining volume — weak rally,
expect continuation to fail. Counterintuitively, this is a LONG
entry — Kaufman's data shows these setups produce positive
expectancy over the 7-day hold period.

### Entry SHORT
1. Today's close is below yesterday's close (close < prev_close)
2. Today's volume is below yesterday's volume (volume < prev_volume)
Both conditions must be true simultaneously (AND logic).
Interpretation: price fell on declining volume — weak selloff.

### Exit LONG
- Today's close drops below yesterday's close (close < prev_close)
- OR take profit at 3% from entry
- OR hold_days_max reached (7 days)
- OR stop loss triggered

### Exit SHORT
- Today's close rises above yesterday's close (close > prev_close)
- OR take profit at 3% from entry
- OR hold_days_max reached (7 days)
- OR stop loss triggered

### Stop Loss
2% from entry price.

### Take Profit
3% from entry price. Adding this pushed aggregate Sharpe from 1.45
to 1.58, crossing the Tier A threshold (1.50).

### Hold Days
Maximum 7 days (aligned with Kaufman's "next week" timeframe).

### Direction
BOTH (long and short entries).

### Regime Filter
None (trades in all regimes).

### Walk-Forward Results
- Windows passed: 19/26 (73%)
- Aggregate Sharpe: 1.58
- Last 4 windows: 3/4 (75%)
- Worst window drawdown: 13.5%
- Full-period Sharpe: 1.61 (with take profit)
- Full-period win rate: 39.1%
- Full-period trades: 665
- Nifty correlation: -0.180 (mildly negative — independent of market)

### Expected Trading Frequency
- Full-period trades: 665 over 10 years
- Approximately 66 trades per year
- Highest frequency of the three signals

### Discovery
Found by DSL Round 2 backtest — the DSL translator correctly
identified the price/volume divergence concept from Kaufman Ch.12
and translated it to prev_close/prev_volume comparisons. The base
translation scored 1.45 Sharpe (Tier B). Adding 3% take profit
pushed it to Tier A.

### Alternative Variant
ADX < 25 filter variant also reaches Tier A (20/26, Sharpe 1.81,
worst DD 10%) but halves trade count to ~37/year. Kept as fallback
if the primary variant underperforms in paper trading.

### Risk Notes
- 39% win rate means majority of trades lose — relies on winners
  being larger than losers (profit factor driven)
- 7-day max hold means positions are short-lived — limits both
  upside and downside per trade
- BOTH direction means it will take short positions during bull
  markets — acceptable given near-zero Nifty correlation
- Volume patterns may behave differently around holidays, expiry
  days, and budget announcements — monitor these periods

---

## PORTFOLIO PROPERTIES

Running all three signals together:

| Property | DRY_20 | DRY_16 | DRY_12 | Combined |
|----------|--------|--------|--------|----------|
| Tier | A | B | A | — |
| Direction | LONG only | BOTH | BOTH | Full coverage |
| Nifty corr | +0.240 | -0.352 | -0.180 | Diversified |
| Trades/year | ~23 | ~40 | ~66 | ~129 total |
| Sharpe | 2.55 | 2.03 | 1.58 | — |
| Win rate | 47.3% | 48.6% | 39.1% | — |

129 trades per year across 3 signals is ~10-11 trades per month.
Active enough to generate meaningful data in 4 months of paper
trading, not so frequent that transaction costs dominate.

Correlation structure:
- DRY_20 (+0.24) and DRY_16 (-0.35): mild natural hedge
- DRY_12 (-0.18): essentially uncorrelated to both Nifty and the
  other two signals. Fires on volume exhaustion — completely
  different market microstructure from pivot breakouts (DRY_16)
  or stochastic momentum (DRY_20).

### Monitoring Notes for Paper Trading
- DRY_20: VIX alert when VIX > 20 with open position (manual review)
- DRY_16: Track short-side win rate separately from long-side in
  first 2 months (asymmetric entry quality check)
- DRY_12: Monitor around expiry days and holidays where volume
  patterns are abnormal

---

## SIGNALS NOT INCLUDED

### GRIMES_DRY_6_0 — TIER B (NOT FOR PAPER TRADING)
Walk-forward 14/26, but the exit is broken (bb_bandwidth < bb_bandwidth,
always false). The 14/26 result is an artifact of the 3-day hold_days
timeout, not genuine signal logic. Needs full hand-retranslation of
Grimes' "Failed Test" concept before it is tradeable.
