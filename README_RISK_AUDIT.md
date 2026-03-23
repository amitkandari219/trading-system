# Nifty F&O Trading System — Risk Audit Documentation

**Date**: 2026-03-23  
**Auditor**: Senior Risk Management Quant  
**Capital**: ₹10,00,000  
**Verdict**: **DO NOT DEPLOY LIVE** — 7 critical fixes required

---

## Quick Links

1. **[AUDIT_SUMMARY.txt](AUDIT_SUMMARY.txt)** — Start here. 2-minute executive overview of all findings.
2. **[CRITICAL_FIXES.md](CRITICAL_FIXES.md)** — The 7 must-fix issues with code examples. Estimated 23 hours to implement.
3. **[RISK_AUDIT_REPORT.md](RISK_AUDIT_REPORT.md)** — Full 500-line detailed audit. Read for deep understanding.

---

## Executive Summary

The system has **solid architecture** (6-layer position sizing, 8-regime detection, Kelly framework) but **critical implementation gaps** that make live trading unsafe.

### Grade: D+ → C (if fixed)

| Component | Grade | Status |
|-----------|-------|--------|
| Position Sizing | D+ | Lacks circuit breaker, rounding errors, no bounds checking |
| Risk Controls | D | No position tracking, daily loss halt ignored, intraday data stale |
| Cost Model | A | Accurate Zerodha fees, realistic slippage ✓ |
| Regime Detection | C | Well-designed but instant transitions cause whipsaw |
| Capital Efficiency | C- | Can exceed 60% margin cap via concurrent positions |
| Tail Risk | D- | No gap protection, flash crash undefended |
| Live Execution | F | No expiry selection, no SL linkage, no order polling |

---

## 7 Critical Issues (MUST FIX)

### 1. No Circuit Breaker in Sizer
- **File**: `core/unified_sizer.py` lines 211-213
- **Risk**: 3 concurrent positions × 2 lots each = ₹360k margin (60% cap is ₹600k, so you can exceed by 40%)
- **Fix**: Pass `existing_margin` to sizer, check total before allocation
- **Time**: 4 hours

### 2. Recursive Rounding Destroys Multipliers
- **File**: `core/unified_sizer.py` lines 179-207
- **Risk**: Multipliers 0.8 × 0.5 × 0.6 = 0.24 get rounded to 1 lot at each step, ending at 1 lot instead of ~0
- **Fix**: Single `round()` at final step, use `floor()` for de-sizing
- **Time**: 2 hours

### 3. No Live Position Tracking
- **File**: `core/orchestrator.py`
- **Risk**: Can try to open 3rd position when broker says margin is exhausted
- **Fix**: Query `kite_bridge.get_positions()` before sizing in live mode
- **Time**: 4 hours

### 4. No Daily Loss Halt
- **File**: `core/orchestrator.py`
- **Risk**: Config defines `DAILY_LOSS_FRACTION=5%` (₹50k) but never enforced
- **Fix**: Check daily P&L before `run_daily_session()`, halt if exceeded
- **Time**: 1 hour

### 5. Nifty Expiry Not Specified
- **File**: `core/orchestrator.py` line 742
- **Risk**: Order placed as `'NIFTY'` without expiry → defaults to far-month (1-5% price difference)
- **Fix**: Specify `'NIFTY23MAY'` format with nearest Friday
- **Time**: 2 hours

### 6. No Stop-Loss Linkage
- **File**: `core/orchestrator.py` lines 735-756
- **Risk**: Entry order placed but no correlated SL order → unlimited downside if manual SL forgotten
- **Fix**: Place bracket order with SL trigger after entry confirms
- **Time**: 3 hours

### 7. Intraday Data Stale
- **File**: `core/orchestrator.py` lines 517-607
- **Risk**: Intraday signals evaluated on 200-day daily closes, not 5-min bars
- **Fix**: Separate `_fetch_intraday_data()` method for tick data
- **Time**: 3 hours

**Total Development Time**: ~23 hours (3 days intensive)

---

## 5 Warnings (SHOULD FIX)

1. **Regime transitions instantaneous** → add hysteresis (2 bars confirmation)
2. **Regime confidence ignored** → use it for sizing modifier
3. **Capital reserve mismatch** → ₹2L reserve defined but sizer uses full ₹10L
4. **No margin validation** → check before placing live orders
5. **No market hours check** → can attempt trades at 9:14 or 15:40 (market closed)

---

## What Works Well ✓

1. **Cost Model** (Grade: A)
   - Zerodha fees: ₹20 brokerage + 18% GST accurate
   - STT 0.0125%, exchange 0.00495% match NSE schedule
   - Verified: 1 lot @ 24000→24100, VIX=20 costs ₹1,084 (18 bps) ✓

2. **Regime Detection** (Grade: C+)
   - 8-regime classification comprehensive (STRONG_BULL through CRISIS)
   - SMA 50/200 alignment + ADX > 22 is mathematically sound
   - Confidence scoring solid (ADX, SMA sep, price distance, VIX)

3. **Kelly Framework** (Grade: B)
   - 6-stage pipeline: base risk → Kelly → overlays → conviction → regime → lots
   - Drawdown-aware (cuts to 0.20x at >10% DD)
   - VIX-responsive (penalizes at >20 VIX)

4. **Signal Coordination** (Grade: B)
   - Deduplication of correlated pairs
   - Category limits (max 2 KAUFMAN)
   - Directional conflict resolution

---

## Capital Efficiency Math

With ₹10L capital, ₹120k SPAN margin per lot, ₹24k Nifty price:

**Scenario 1: Single trade**
- Base lots: 2 (from ₹6,667 risk ÷ ₹3,000 risk/lot)
- Margin: ₹240k (24% of capital) ✓ Safe

**Scenario 2: Three concurrent 2-lot positions**
- Total margin needed: 6 × ₹120k = ₹720k
- Cap: 60% of ₹10L = ₹600k
- System silently caps to 5 lots total
- **Issue**: 3rd trade gets 1 lot instead of 2 (unfair weighting)

**Scenario 3: Flash crash (VIX 16→35)**
- Slippage: 0.02% × (1 + (35-20)/10) = 0.04% = ₹2,400
- Position risk: ₹3,000
- **Slippage eats 80% of risk budget**

---

## Deployment Timeline

### Phase 1: Apply Critical Fixes (1 week)
- Position tracking + circuit breaker
- Daily loss halt
- Expiry specification
- Stop-loss linkage
- Rounding fix
- Testing

### Phase 2: Paper Trading (4-6 weeks)
- Verify all fixes in paper mode
- Backtest 2023-2024 with fixes
- Monitor for false alarms

### Phase 3: Live Micro (2-4 weeks)
- Start with 0.5 lots
- Scale to 1 lot after 2 weeks profitability
- Monitor real vs. modeled slippage

### Phase 4: Production (3+ months)
- Only after 3 months at full size with Sharpe > 1.0

---

## Files in This Audit

```
/sessions/upbeat-adoring-cray/mnt/trading-system/
├── AUDIT_SUMMARY.txt          ← Start here (2-minute read)
├── CRITICAL_FIXES.md          ← Implementation guide (23 hours work)
├── RISK_AUDIT_REPORT.md       ← Full detailed audit (500 lines)
├── README_RISK_AUDIT.md       ← This file
├── core/
│   ├── unified_sizer.py       ← Sizing (3 critical issues)
│   ├── orchestrator.py        ← Risk controls (4 critical issues)
│   ├── regime_detector.py     ← Regime logic (C+ grade)
│   └── costs.py               ← Cost model (A grade, accurate)
└── config/
    └── unified_config.py      ← Capital allocation
```

---

## How to Use This Audit

1. **Quick briefing** (5 min): Read AUDIT_SUMMARY.txt
2. **Implementation guide** (4-6 hours): Work through CRITICAL_FIXES.md with your team
3. **Deep dive** (2+ hours): Review RISK_AUDIT_REPORT.md for details and reasoning
4. **Testing**: Verify each fix with unit tests + paper trading simulation
5. **Sign-off**: Only after all 7 critical fixes + 2 weeks paper trading

---

## Key Takeaway

The system **architecture is sound**, but **implementation is incomplete**. The gaps are **not unfixable** — they're textbook risk management oversights that have 23 hours of work + validation.

**Do not go live with real money until:**
1. All 7 critical fixes are implemented
2. Paper trading shows no new issues for 2+ weeks
3. Live micro-trading (0.5 lots) proves execution logic works
4. Sharpe ratio > 1.0 on live returns

---

## Questions?

See RISK_AUDIT_REPORT.md for detailed reasoning on each finding, including code excerpts and test cases.

**Audited by**: Senior Risk Management Quant  
**Severity Level**: REAL MONEY AT STAKE  
**Last Updated**: 2026-03-23
