# NIFTY F&O TRADING SYSTEM — RISK MANAGEMENT AUDIT

**Audited:** 2026-03-23 | **Capital:** ₹10,00,000 | **Lot Size:** 25 | **SPAN Margin/Lot:** ₹120,000

---

## EXECUTIVE SUMMARY

The system has **solid architectural foundations** with proper layering of regime detection, Kelly sizing, and conviction scoring. However, there are **critical gaps in live execution safeguards** that must be addressed before going live. The cost model is accurate, but position sizing can create margin conflicts under stress. **Grade: C+** — structurally sound but operationally risky.

---

## 1. POSITION SIZING — UNIFIED_SIZER.PY

### What Works ✓

1. **Proper Kelly pipeline**: Base risk → Kelly fraction → overlay composite → conviction → regime → lots (6-stage pipeline is well-architected)
2. **Conservative baseline**: 2% of capital per trade, divided by 3-4 positions = ₹6,667-₹5,000 risk per trade
3. **Margin guardrails**: Line 211-213 caps position size to available margin (60% of equity × ₹120k SPAN)
4. **Geometric mean compositing**: Overlay composite uses log-space averaging (line 256-257) to avoid extreme multipliers
5. **Reasonable multiplier clamps**: COMPOSITE_FLOOR=0.30, CEILING=2.00 (line 57-58)

### Critical Issues 🔴

#### **1. NO CIRCUIT BREAKER IN SIZER**
- The `compute()` method has **zero protection** against simultaneous signal firings
- If 3 KAUFMAN signals + 2 GUJRAL signals fire on same day → orchestrator's SignalCoordinator filters to 2 (line 160 in orchestrator.py), but **sizer is never told portfolio is already 2 positions**
- With 2-lot positions × 2 concurrent = ₹240k margin. Add third position → ₹360k (60% of equity exactly). But the sizer doesn't know about open positions beyond what's passed in `open_positions` parameter
- **Risk**: Orchestrator has no position tracking mechanism. If live execution happens with stale state, sizer can overallocate

**Fix Required**: Pass current portfolio state to sizer, implement hard stop if (proposed_margin + existing_margin) > 60% of equity

#### **2. RECURSIVE ROUNDING ERROR IN KELLY PIPELINE**
```python
# Line 179: base_lots * kelly_fraction, then round()
adjusted = max(MIN_LOTS, round(base_lots * kelly_fraction))
# Then line 185: adjusted * composite_modifier, then round()
adjusted = max(MIN_LOTS, round(adjusted * composite_modifier))
# Then line 202: adjusted * conviction_modifier, then round()
adjusted = max(MIN_LOTS, round(adjusted * conviction_modifier))
# Then line 207: adjusted * regime_modifier, then round()
adjusted = max(MIN_LOTS, round(adjusted * regime_modifier))
```

Each round() truncates fractional lots. With small base_lots (1-2), you can get:
- Base: 1 lot
- Kelly 0.8 × 1 = 0.8 → round = **1 lot** (no reduction!)
- Composite 0.5 × 1 = 0.5 → round = **1 lot** (no reduction!)
- Conviction 0.6 × 1 = 0.6 → round = **1 lot** (still 1 lot!)

**Expected behavior**: 0.8 × 0.5 × 0.6 = 0.24 → should be ~0 lots or MIN_LOTS=1, but logic makes it hard to de-size.

**Fix**: Use floor() for down-sizing, only round() at the final step. Or use accumulator: `adjusted *= kelly_fraction` without intermediate rounding.

#### **3. DIVISION BY ZERO NOT HANDLED**
```python
# Line 158: risk_per_lot = sl_points * NIFTY_LOT_SIZE
# If risk_per_lot == 0 (e.g., SL=0), line 158 tries math.floor(risk_per_position / risk_per_lot)
# → ZeroDivisionError not caught
```

The code defaults to SL=2% of spot (line 153), but if a signal passes sl_points=0, crash.

**Fix**: Ensure risk_per_lot > 0 before division

---

### WARNINGS ⚠️

#### **1. MARGIN ALLOCATION LOGIC MISMATCH**
```python
# Line 211: available_margin = self.equity * 0.60
# Line 212-213: margin_lots = floor(available_margin / MARGIN_PER_LOT)
```

With ₹10L capital:
- Available: ₹600k (60%)
- Max lots: 5 (₹600k ÷ ₹120k)
- Max notional: 5 × 25 × ₹24k = **₹3M (300% of capital!)**

This is **extreme leverage**. Breaker is only the 60% constraint. If one trade alone uses 5 lots (~₹3M notional), any 1% adverse move = ₹30k loss = 3% of capital. 

**Reality check**: With MAX_CONCURRENT_POSITIONS=3 and base_lots=2, you'd be 6 lots, which is impossible (exceeds margin by 20%). The system will silently cap to 5 lots.

**Better approach**: 
```
max_notional = equity × 2.0  # 200% (standard for leveraged futures)
max_lots = max_notional / (spot_price × 25)
```

---

## 2. RISK CONTROLS — ORCHESTRATOR.PY

### What Works ✓

1. **Regime filtering**: Line 613-636 blocks signals in hostile regimes (CRISIS disables certain trades)
2. **Signal coordination**: Deduplication (high-correlation pairs), category limits (max 2 KAUFMAN), directional conflict resolution
3. **Regime detection**: UnifiedRegimeDetector classifies 8 regimes with confidence scoring
4. **Drawdown awareness**: Sizer receives `drawdown_pct` parameter (line 175), Kelly adjusts for it

### Critical Issues 🔴

#### **1. NO LIVE POSITION TRACKING**
The orchestrator **never queries live positions** from the broker. It has:
```python
# Line 176: _signal_computer = None
# Line 175: self._signals: Dict[str, Any] = {}
```

But **nowhere** does it:
- Fetch open Nifty positions from broker
- Query portfolio delta/notional
- Check if margin is exhausted
- Verify if a new order would breach margin

In paper mode, this is fine (no real orders). But in live mode (line 696), if execution fails or partial fills happen, **sizer and orchestrator have no idea**. You could try to open 3 positions when only 1 margin slot remains.

**Fix Required**: Before `_size_and_dispatch()`, query broker for:
```python
open_positions_notional = kite_bridge.get_positions()
available_margin = kite_bridge.get_margin()['available']
# Pass to sizer
```

#### **2. INTRADAY CHECK RETURNS UNSIZED TRADES IN PAPER MODE**
Lines 517-607 (`run_intraday_check`) have a critical flaw:

```python
# Line 550: spot_price = float(df['close'].iloc[-1])
# But df is 200-day DAILY data, not intraday!
```

For intraday signals (EXPIRY_PIN_FADE, ORR_REVERSION), you need **5-minute data**. The code calls `_fetch_market_data(lookback_days=60)` which gets **daily bars**. Intraday signals then get evaluated on day-old closes, not real-time 5-min ticks.

**Result**: Trades sized/logged from data that's not fresh. In paper trade logs, they'll look like valid trades. In live, they'll blow up when executed.

**Fix**: Intraday check needs to call a separate `_fetch_intraday_data()` method that hits Zerodha's tick data API.

#### **3. NO DAILY LOSS HALT**
The config defines:
```python
# config/unified_config.py line 44:
DAILY_LOSS_FRACTION = 0.05  # 5% of capital
```

But the orchestrator **never checks it**. There's no logic like:
```python
daily_pnl = get_realized_pnl_today()
if daily_pnl < -DAILY_LOSS_FRACTION * equity:
    logger.warning('Daily loss limit breached')
    return []  # halt trading
```

You could lose ₹50k (5% of ₹10L) in a single day and keep trading.

**Fix**: Add daily loss check in `run_daily_session()` before signal evaluation

---

### WARNINGS ⚠️

#### **1. REGIME TRANSITIONS ARE INSTANTANEOUS, NOT SMOOTHED**
The UnifiedRegimeDetector (regime_detector.py) classifies based on **today's VIX and ADX**:

```python
# Line 198-260: detect() is stateless, each call is independent
# If VIX jumps 23 → 26 (NORMAL → EXTREME), regime instantly flips to CRISIS
```

This causes **whipsawing**:
- NORMAL regime: signal fires, gets full 1.0x sizing
- Next bar: VIX spikes, regime → CRISIS, size multiplier → 0.30x
- Existing position gets margin called or widened SL

**Better approach**: Use regime **hysteresis** (require 2 consecutive bars in new regime before switching)

---

## 3. COST MODEL — BACKTEST/TRANSACTION_COSTS.PY

### What Works ✓

1. **Accurate Zerodha fees**: ₹20 brokerage + 18% GST, 0.0125% STT, 0.00495% exchange charges are correct
2. **Realistic slippage**: 0.02% base for Nifty futures, scales with VIX
3. **Round-trip accounting**: Both entry and exit charges included
4. **Itemized breakdown**: Brokerage, STT, exchange, GST, SEBI, stamp duty, slippage all separate

### Test Results ✓

For 1 lot at 24000→24100 (VIX=20):
- **Total cost: ₹1,084.60 (~43.4 Nifty points)**
- Breakdown:
  - Brokerage: ₹47.20
  - STT: ₹75.31
  - Exchange: ₹595.24
  - GST: ₹107.14
  - SEBI: ₹1.20
  - Stamp duty: ₹18.00
  - Slippage: ₹240.50

**Cost as % of notional: 0.180%** ✓ (18 basis points, reasonable for intraday)

### Minor Issues ⚠️

#### **1. SLIPPAGE MODEL IS TOO SIMPLE**
```python
# Line 202-209: slippage_pct = base_slippage_pct (0.02%)
# Only scaled by VIX, not by position size or time of day
```

In reality:
- 1 lot slippage: ~0.01%
- 5 lots slippage: ~0.05% (market impact)
- Opening hours (9:15-10:00): less slippage
- Closing hours (15:00-15:30): more slippage

**Impact**: Small — cost model underestimates slippage for large orders, but your max order is 5 lots, so effect is ~±10 points.

#### **2. NO COMMISSION DISCOUNT FOR VOLUME**
Zerodha charges flat ₹20/order regardless of lots. Code is correct. But if you're trading 5 lots once, you're paying ₹47.20 to enter and ₹47.20 to exit. High costs for small P&L targets.

---

## 4. REGIME DETECTION — CORE/REGIME_DETECTOR.PY

### What Works ✓

1. **8-regime classification**: STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR, HIGH_VOL_BULL, HIGH_VOL_BEAR, CRISIS is comprehensive
2. **Clear trend logic**: SMA 50 vs SMA 200 alignment + ADX > 22 for trend strength
3. **VIX thresholds**: LOW (≤12), NORMAL (12-16), HIGH (16-20), EXTREME (≥20) match NSE structure
4. **Confidence scoring**: Combines ADX, SMA separation, price distance, VIX clarity (line 352-396)

### Findings ✓

- Trend detection logic (line 278-308) is sound: requires both SMA alignment AND ADX > threshold
- CRISIS regime (line 323-326) only triggers on VIX≥25 + downtrend, not just extreme VIX alone ✓

### WARNINGS ⚠️

#### **1. REGIME CONFIDENCE IS NEVER USED FOR POSITION SIZING**
```python
# Line 250: 'confidence': round(confidence, 3)
# This is logged but never consulted by orchestrator or sizer
```

If confidence is 0.3 (weak regime signal), you should reduce size. Currently ignored.

**Fix**: Pass regime + confidence to sizer, apply:
```python
size_mult *= regime_modifier * (0.5 + 0.5 * confidence)  # 50-100% of regime mult
```

#### **2. SMA CALCULATION NOT VALIDATED**
The regime detector assumes SMA 50 and SMA 200 are already in the dataframe:
```python
# Line 224-227: sma_50, sma_200 from market_context
# No check if they're null or stale
```

If indicator calculation fails, regime defaults to NEUTRAL (line 292). Silent failure.

---

## 5. CAPITAL EFFICIENCY & SIZING REALISM

### Math Check

**Available Capital Structure:**
- Initial: ₹10,00,000
- Reserve (20%): ₹2,00,000
- Available: ₹8,00,000... **Wait, config says available = initial - reserve = ₹8L, but sizer uses full ₹10L**

```python
# unified_config.py line 20-24:
INITIAL_CAPITAL = 10_00_000
CAPITAL_RESERVE = 2_00_000
AVAILABLE_CAPITAL = INITIAL_CAPITAL - CAPITAL_RESERVE  # ₹8L

# But unified_sizer.py line 208:
UnifiedSizer(equity=1_000_000)  # Full ₹10L!
```

This is a **data mismatch**. Sizer thinks you have ₹10L to work with, but conceptually ₹2L is reserved (untouchable). If your actual live capital is ₹10L, this is fine. But if you're meant to operate on ₹8L, sizing is overly aggressive.

### Position Feasibility

**Scenario 1: Single position at max sizing**
- Base: 2 lots (₹6,667 risk ÷ ₹3,000 risk/lot)
- Kelly 0.8: still 2 lots (rounding issue!)
- Overlay composite 1.2: 2 lots
- Conviction 1.3: still 2 lots (rounding!)
- Regime 1.1: still 2 lots
- **Result: 2 lots = ₹240k margin (24% of capital)** ✓ Feasible

**Scenario 2: Three concurrent positions**
- Each sized to 2 lots (after Kelly/conviction)
- Total: 6 lots = ₹720k margin (**72% of capital!**)
- But margin cap is 60% = ₹600k = max 5 lots
- **System silently de-risks to 5 lots total** (line 213)
- If 3 signals fire, they'd get 2, 2, and 1 lot = can work, but unfair weighting

**Reality**: With MAX_CONCURRENT_POSITIONS=3, expecting each to get equal sizing doesn't work. FIFO sizing or equal-margin allocation would be cleaner.

---

## 6. TAIL RISK & FLASH CRASHES

### What Protects You 

1. **Regime-based circuit breaker**: CRISIS regime (VIX ≥ 25 + downtrend) blocks most signals
2. **Drawdown scaling**: Kelly reduces to 0.20 at 10%+ drawdown (line 63-68 of adaptive_kelly.py)
3. **Conviction modifiers**: Conviction scorer penalizes sizing in conflicting overlays

### What Doesn't ❌

1. **No gap-up/gap-down protection**
   - If Nifty gaps 2% overnight (₹480 points), your 120-point SL is instantly violated
   - No check: "Is entry price achievable at market open?"
   - **Fix**: Add gap filter — if overnight gap > 0.5%, skip trade

2. **No market hours restriction**
   - Orchestrator can attempt trades at 9:14 or 15:40 (market closed)
   - No check for MARKET_OPEN (line 80 of config says 9:15, but no enforcement)
   - **Fix**: Guard trades with:
   ```python
   now = datetime.now().time()
   if not (MARKET_OPEN <= now <= MARKET_CLOSE):
       return []
   ```

3. **No volatility explosion protection**
   - If VIX goes 16 → 35 in 5 minutes (flash crash), slippage explodes
   - Code scales slippage (line 204-207), but no hard halt
   - You could enter at 24000, market tanks to 23500, slippage eats ₹4k of your ₹6,667 risk budget

4. **No intraday trailing drawdown check**
   - Daily loss halt is defined (₹50k) but never enforced
   - You could accumulate 5 small losses = ₹35k, then take a ₹20k loss = ₹55k total, exceed limit
   - System doesn't know until EOD

---

## 7. ORCHESTRATOR EXECUTION GAPS (LIVE MODE)

### Paper Mode ✓

Paper mode (line 701-733) just logs to DB. Safe.

### Live Mode Issues 🔴

```python
# Line 735-756: _execute_live()
def _execute_live(self, trade: Dict):
    order_dict = {
        'tradingsymbol': 'NIFTY',  # ← Always 'NIFTY', no expiry specified!
        'exchange': 'NFO',
        'transaction_type': 'BUY' if trade['direction'] == 'BULLISH' else 'SELL',
        'quantity': trade['lots'] * NIFTY_LOT_SIZE,  # ← quantity in units, not contracts
        'product': 'MIS',  # ← Intraday margin product
        'order_type': 'MARKET',  # ← Always market, no limit orders
        'tag': trade['signal_name'][:20],
    }
    order_id = self.kite_bridge.place_order(order_dict)
```

**Problems:**

1. **No expiry selected**: Nifty futures expire weekly. If you specify 'NIFTY' without expiry, Zerodha defaults to **far-month**. You want the weekly (nearest expiry). Futures can move 5% month-to-month.

2. **Quantity in units, not lots**: `quantity = trade['lots'] * NIFTY_LOT_SIZE = 2 * 25 = 50 units`. Correct. But no validation that this matches available margin.

3. **Market orders only**: No limit orders. If Nifty is bid-ask 24000-24005 and you market buy, you get 24005 → ₹125 slippage right there. Should use limit orders at mid-price with timeout.

4. **No order status polling**: After `place_order()`, code assumes success. No check for:
   - Order rejected (over limit, invalid symbol, etc.)
   - Partial fill
   - Position mismatch with DB

5. **No stop-loss linkage**: You're placing a BUY market order, but no correlated SELL stop-loss order. Manual stop-loss required!

---

## SUMMARY TABLE

| Risk Area | Status | Issue | Severity |
|-----------|--------|-------|----------|
| **Position Sizing** | ⚠️ Risky | No circuit breaker, recursive rounding errors, division by zero not caught | 🔴 CRITICAL |
| **Risk Controls** | ⚠️ Incomplete | No live position tracking, intraday data stale, no daily loss halt | 🔴 CRITICAL |
| **Cost Model** | ✓ Accurate | Zerodha fees correct, slippage model simplistic but acceptable | ✓ OK |
| **Regime Detection** | ⚠️ Weak | Confidence ignored, transitions not smoothed, instant regime flips cause whipsaw | ⚠️ WARNING |
| **Capital Efficiency** | ⚠️ Risky | Possible to allocate 72% margin (violates 60% cap), rounding causes unfair weighting | ⚠️ WARNING |
| **Tail Risk** | ❌ Unprotected | No gap protection, no market-hours check, no flash-crash halt, no intraday DD tracking | 🔴 CRITICAL |
| **Live Execution** | ❌ Unsafe | No expiry selection, no order status check, no stop-loss linkage, assumes execution success | 🔴 CRITICAL |

---

## RECOMMENDATIONS

### MUST FIX (before live trading)

1. **Implement circuit breaker**
   ```python
   # Before _size_and_dispatch():
   open_notional = sum(p['notional'] for p in open_positions)
   proposed_notional = sizing['lots'] * NIFTY_LOT_SIZE * spot_price
   if (open_notional + proposed_notional) > equity * 2.0:
       logger.warning('Notional limit breached, rejecting trade')
       return None
   ```

2. **Add live position sync**
   ```python
   # In orchestrator.__init__():
   self.live_positions = {}  # {'NIFTY23JAN': {'qty': 50, 'entry': 24000}}
   
   # Before sizing:
   if self.mode == 'live':
       self.live_positions = self.kite_bridge.get_positions()
   ```

3. **Specify Nifty expiry**
   ```python
   # In _execute_live():
   symbol = self._get_nearest_expiry()  # 'NIFTY23JAN'
   order_dict['tradingsymbol'] = symbol
   ```

4. **Add stop-loss order linkage**
   ```python
   # After placing entry order:
   sl_order = {
       'parent_order_id': order_id,
       'trigger_price': entry_price - sl_points,
       'order_type': 'STOPLOSS',
       ...
   }
   kite_bridge.place_order(sl_order)
   ```

5. **Add daily loss halt**
   ```python
   # In run_daily_session():
   daily_pnl = self.conn.query('realized_pnl today')
   if daily_pnl < -DAILY_LOSS_FRACTION * equity:
       logger.warning('Daily loss limit reached, halting')
       return []
   ```

### SHOULD FIX (before taking real money)

1. **Smooth regime transitions** (add hysteresis)
   ```python
   if new_regime != old_regime and bar_count < 2:
       regime = old_regime  # require confirmation
   ```

2. **Use Kelly as floor for conviction**
   ```python
   conviction_mult = max(kelly_fraction, conviction_modifier)
   ```

3. **Add gap filter**
   ```python
   overnight_gap = abs(today_open - yesterday_close) / yesterday_close
   if overnight_gap > 0.005:  # 0.5% gap
       logger.warning('Gap detected, skipping intraday signals')
   ```

4. **Switch to limit orders with timeout**
   ```python
   order_type = 'LIMIT' if signal_confidence > 0.70 else 'MARKET'
   limit_price = spot_price if transaction_type == 'BUY' else spot_price
   ```

5. **Add market-hours check**
   ```python
   if not (MARKET_OPEN <= datetime.now().time() <= MARKET_CLOSE):
       return []
   ```

---

## RISK MANAGEMENT GRADE

| Component | Grade | Reason |
|-----------|-------|--------|
| Sizing | D+ | Lacks guardrails, rounding issues, division by zero |
| Risk Controls | D | No circuit breaker, intraday data stale, no halt logic |
| Costs | A | Accurate, realistic |
| Regime Detection | C | 8 regimes good, but no smoothing and confidence unused |
| Capital Efficiency | C- | Possible to violate margin cap, unfair weighting |
| Tail Risk | D- | Gaps, flash crashes, VIX spikes unprotected |
| Live Execution | F | Expiry not selected, SL not linked, no order polling |

**OVERALL: D+ → C** (if all critical fixes applied)

**Verdict**: Do **NOT** go live in current state. Paper trading for 2-3 months minimum, with live fixes applied.

