# CRITICAL FIXES BEFORE LIVE TRADING

**Status**: 7 CRITICAL issues that MUST be fixed. **DO NOT GO LIVE** until these are resolved.

---

## 1. NO CIRCUIT BREAKER IN POSITION SIZING
**File**: `core/unified_sizer.py`
**Problem**: When multiple signals fire simultaneously, the sizer has no awareness of already-open positions. With 3 concurrent positions × 2 lots each, you could try to allocate ₹360k margin (exceeds 60% cap by 40%).

**Current**:
```python
# Line 211-213: margin cap exists but sizer doesn't know about open positions
available_margin = self.equity * 0.60
margin_lots = math.floor(available_margin / MARGIN_PER_LOT)
adjusted = min(adjusted, margin_lots, MAX_LOTS_CAP)
```

**Fix**: Add position awareness
```python
def compute(self, ..., existing_margin: float = 0.0) -> Dict:
    # NEW: Get total allocated margin
    total_margin_needed = existing_margin + (adjusted * MARGIN_PER_LOT)
    available_margin = self.equity * 0.60

    if total_margin_needed > available_margin:
        logger.warning(f'Margin limit: {total_margin_needed} > {available_margin}')
        # De-size this trade to fit
        adjusted = max(MIN_LOTS, int((available_margin - existing_margin) / MARGIN_PER_LOT))
```

**Test**:
```python
# Before fix: can exceed margin cap
# After fix: last signal to fire gets reduced size
```

---

## 2. RECURSIVE ROUNDING DESTROYS POSITION SIZING INTENT
**File**: `core/unified_sizer.py`, lines 179-207
**Problem**: Multipliers of 0.8, 0.5, 0.6 applied to 1-lot base get rounded to 1 lot at each step. Instead of 0.8×0.5×0.6=0.24 (should be ~0), you get 1 lot. Position sizing doesn't actually compress.

**Current**:
```python
# Line 179
adjusted = round(base_lots * kelly_fraction)  # 1.0 * 0.8 = 0.8 → 1 lot
# Line 185
adjusted = round(adjusted * composite_modifier)  # 1 * 0.5 = 0.5 → 1 lot
# Line 202
adjusted = round(adjusted * conviction_modifier)  # 1 * 0.6 = 0.6 → 1 lot
```

**Fix**: Single round at the end
```python
# Accumulate all multipliers
total_mult = kelly_fraction * composite_modifier * conviction_modifier * regime_modifier
final_lots = round(base_lots * total_mult)  # 1 * (0.8*0.5*0.6*1.1) = 0.264 → 0, clamp to MIN_LOTS=1
```

Or use floor for de-sizing:
```python
if total_mult < 1.0:
    final_lots = max(MIN_LOTS, math.floor(base_lots * total_mult))
else:
    final_lots = math.ceil(base_lots * total_mult)
```

**Test**:
```python
# Verify that Kelly 0.50 actually cuts size in half
# before: 1 → 1 (broken), after: 1 → round(0.5) = 0 clamp to 1 (still 1 but intentional)
```

---

## 3. DIVISION BY ZERO NOT PROTECTED
**File**: `core/unified_sizer.py`, line 158
**Problem**: If a signal passes `sl_points=0`, the line `risk_per_lot = sl_points * NIFTY_LOT_SIZE` gives 0, then `math.floor(risk_per_position / risk_per_lot)` crashes.

**Current**:
```python
# Line 157-158
risk_per_lot = sl_points * NIFTY_LOT_SIZE
base_lots = math.floor(risk_per_position / risk_per_lot) if risk_per_lot > 0 else 1  # ← Check exists!
```

**Actually OK!** The code DOES have `if risk_per_lot > 0`. Verified safe.

---

## 4. NO LIVE POSITION TRACKING IN ORCHESTRATOR
**File**: `core/orchestrator.py`
**Problem**: In live mode, the orchestrator never fetches open positions from the broker. It can try to open a 3rd position when margin is exhausted, or sell a position that's already closed.

**Current**:
```python
# Line 696-756: _execute_live() just places orders
# No query to kite_bridge.get_positions() before checking margin
```

**Fix**: Add position sync before sizing
```python
def run_daily_session(self, ...) -> List[Dict]:
    # NEW: fetch live positions
    if self.mode == 'live':
        try:
            live_pos = self.kite_bridge.get_positions()
            live_margin = self.kite_bridge.get_margin()['available']
            logger.info(f'Live margin available: ₹{live_margin}')
        except Exception as e:
            logger.error(f'Position fetch failed: {e}, aborting')
            return []
    else:
        live_pos = {}
        live_margin = self.equity * 0.60

    # Pass to sizer
    for cand in candidates:
        sizing = self._size_and_dispatch(
            ...,
            existing_margin=sum(p['notional']/50 * 120_000 for p in live_pos.values()),
            existing_positions=live_pos,
        )
```

---

## 5. NO DAILY LOSS HALT ENFORCED
**File**: `core/orchestrator.py`
**Problem**: Config defines `DAILY_LOSS_FRACTION=0.05` (₹50k limit), but orchestrator never checks it. You could lose ₹50k in 3 trades and keep trading.

**Current**:
```python
# config/unified_config.py line 44 defines it
# But orchestrator never uses it
```

**Fix**: Add halt logic
```python
def run_daily_session(self, signal_list=None) -> List[Dict]:
    logger.info('=== Daily Session Start (mode=%s) ===', self.mode)

    # NEW: Check daily loss limit
    if self.conn is not None:
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT COALESCE(SUM(pnl), 0) as daily_pnl
                FROM trades
                WHERE DATE(created_at) = CURRENT_DATE
            """)
            daily_pnl = cur.fetchone()[0]
            loss_limit = self.sizer.equity * DAILY_LOSS_FRACTION
            if daily_pnl < -loss_limit:
                logger.warning(f'Daily loss {daily_pnl} exceeds limit {-loss_limit}, halting')
                return []
        except Exception as e:
            logger.warning(f'Daily loss check failed: {e}')

    # Continue with normal session...
```

---

## 6. NIFTY EXPIRY NOT SPECIFIED IN LIVE ORDERS
**File**: `core/orchestrator.py`, lines 742-748
**Problem**: When placing a live order, the code sets `tradingsymbol: 'NIFTY'` without specifying the expiry. Zerodha defaults to the **far-month contract** (next month), not the **weekly** you intended. Futures can differ 2-5% across expiries.

**Current**:
```python
order_dict = {
    'tradingsymbol': 'NIFTY',  # ← No expiry! Defaults to far-month
    'exchange': 'NFO',
    'quantity': trade['lots'] * NIFTY_LOT_SIZE,
    ...
}
```

**Fix**: Specify the nearest expiry
```python
def _execute_live(self, trade: Dict):
    # NEW: Get nearest Friday (Nifty weekly expiry)
    from datetime import datetime, timedelta
    today = datetime.now().date()
    days_until_friday = (4 - today.weekday()) % 7  # Friday = 4
    if days_until_friday == 0:
        days_until_friday = 7  # Next Friday if today is Friday
    expiry_date = today + timedelta(days=days_until_friday)

    # Format as 'NIFTY23MAY' (Zerodha format)
    symbol = f"NIFTY{expiry_date.strftime('%d%b').upper()}"

    order_dict = {
        'tradingsymbol': symbol,  # ← e.g., 'NIFTY23MAY'
        'exchange': 'NFO',
        ...
    }
    order_id = self.kite_bridge.place_order(order_dict)
    logger.info(f'Placed order for {symbol}, order_id={order_id}')
```

**Test**: Verify that order symbol matches Zerodha's instrument list for Nifty futures

---

## 7. NO STOP-LOSS LINKAGE TO ENTRY ORDER
**File**: `core/orchestrator.py`, lines 735-756
**Problem**: You place a BUY order, but no correlated SELL stop-loss order. If live execution succeeds but you forget to manually set SL, you have unlimited downside.

**Current**:
```python
def _execute_live(self, trade: Dict):
    order_id = self.kite_bridge.place_order(order_dict)
    logger.info('LIVE ORDER placed: %s order_id=%s', trade['signal_name'], order_id)
    # ← No SL order placed!
```

**Fix**: Place OCO (One-Cancels-Other) or bracket order
```python
def _execute_live(self, trade: Dict):
    # Place entry order
    entry_order = {
        'tradingsymbol': symbol,
        'quantity': trade['lots'] * NIFTY_LOT_SIZE,
        'transaction_type': 'BUY' if trade['direction'] == 'BULLISH' else 'SELL',
        'order_type': 'MARKET',
        ...
    }
    entry_id = self.kite_bridge.place_order(entry_order)

    # NEW: Place stop-loss as bracket
    if entry_id:
        # For BUY entry, place SELL stoploss
        sl_price = trade['entry_price'] - trade['stop_loss_points']
        sl_order = {
            'parent_order_id': entry_id,
            'tradingsymbol': symbol,
            'quantity': trade['lots'] * NIFTY_LOT_SIZE,
            'transaction_type': 'SELL',
            'trigger_price': sl_price,
            'order_type': 'STOPLOSS',
            'tag': f"{trade['signal_name']}_SL",
        }
        try:
            sl_id = self.kite_bridge.place_order(sl_order)
            logger.info(f'Stop-loss placed: {sl_id} at {sl_price}')
            trade['stop_loss_order_id'] = sl_id
        except Exception as e:
            logger.error(f'Failed to place SL: {e}, CANCEL ENTRY')
            self.kite_bridge.cancel_order(entry_id)
            return None
```

---

## PRIORITY ORDER

1. **Position tracking + circuit breaker** (fixes margin overflow)
2. **Daily loss halt** (protects capital)
3. **Expiry specification** (fixes wrong contract)
4. **Stop-loss linkage** (protects downside)
5. **Rounding fix** (improves position sizing intent)
6. **Regime hysteresis** (reduces whipsaw)

---

## TESTING CHECKLIST

After fixes, verify:

- [ ] Place 3 simultaneous signals → verify none exceed 60% margin cap
- [ ] Trigger DAILY_LOSS_FRACTION with mock trades → verify trading halts
- [ ] Live order places with correct Nifty expiry (e.g., NIFTY23MAY)
- [ ] Each live order has a linked stop-loss order
- [ ] Kelly 0.5 actually cuts position size by ~50%
- [ ] Regime transition from NORMAL to CRISIS → position size reduces
- [ ] Backtest on 2023-2024 data with live execution simulation

---

## ESTIMATED EFFORT

- Position tracking: 4 hours
- Daily loss halt: 1 hour
- Expiry specification: 2 hours
- Stop-loss linkage: 3 hours
- Rounding fix: 2 hours
- Regime hysteresis: 3 hours
- Testing: 8 hours

**Total: ~23 hours of development** before live trading is safe.

