# Trading System — Design Document

**Version:** 1.0
**Date:** 2026-03-15
**Status:** Paper trading active, extraction pipeline expanding

---

## 1. System Overview

An end-to-end algorithmic trading system that extracts trading rules from books using LLM-powered RAG, translates them into backtestable indicator conditions, validates through walk-forward analysis, and executes via paper trading with A/B testing.

```
┌─────────────────────────────────────────────────────────┐
│                    TRADING SYSTEM                        │
│                                                         │
│  Books → RAG → Extraction → DSL → Backtest → Paper Trade│
│                                                         │
│  27 books   29,782     ~15K      ~200     3 modes       │
│  ingested   chunks    signals   valid    running        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 Pipeline Stages

```
STAGE 1: INGESTION
  PDF books → pdfplumber → text chunks → sentence-transformers → ChromaDB

STAGE 2: EXTRACTION
  ChromaDB chunks → book-specific Claude prompts → SignalCandidate JSON

STAGE 3: TRANSLATION
  SignalCandidate → DSL schema → Claude Haiku → DSLSignalRule → validation

STAGE 4: BACKTESTING
  DSLSignalRule → indicator computation → walk-forward → tier classification

STAGE 5: COMBINATION TESTING
  Pair/triple testing → parameter sweep → OOS validation

STAGE 6: PAPER TRADING
  Daily signal compute → scoring engine → combination engine → Telegram alerts
```

### 2.2 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9 |
| Database | PostgreSQL 15 + TimescaleDB |
| Vector Store | ChromaDB (persistent, disk-backed) |
| Embeddings | all-mpnet-base-v2 (768-dim) |
| LLM — Extraction | Claude Haiku 4.5 (via Anthropic API) |
| LLM — Validation | Claude Sonnet 4 (hallucination checks) |
| Broker API | Zerodha Kite (planned) |
| Alerts | Telegram Bot API |
| Cache/Queue | Redis |
| Deployment | Docker Compose |

### 2.3 Data Flow

```
                    ┌──────────┐
                    │  27 PDFs │
                    └────┬─────┘
                         │ pdfplumber
                         ▼
                    ┌──────────┐
                    │ ChromaDB │ 29,782 chunks
                    └────┬─────┘
                         │ Claude Haiku (book-specific prompts)
                         ▼
                    ┌──────────┐
                    │   JSON   │ ~15,000 raw signals
                    │  files   │ extraction_results/{BOOK}.json
                    └────┬─────┘
                         │ DSL Translator (strict template)
                         ▼
                    ┌──────────┐
                    │ DSL PASS │ ~200 validated signals
                    │  files   │ dsl_results/BEST/{signal}.json
                    └────┬─────┘
                         │ Walk-forward (26 windows)
                         ▼
                    ┌──────────┐
                    │ Tier A/B │ 3 confirmed signals
                    │ signals  │ validation_results/*_fixed.json
                    └────┬─────┘
                         │ Paper trading pipeline
                         ▼
              ┌──────────────────────┐
              │   PostgreSQL trades  │
              │   table (PAPER/      │
              │   SHADOW/SCORING/    │
              │   COMBINATION)       │
              └──────────┬───────────┘
                         │ Telegram
                         ▼
                    ┌──────────┐
                    │  Phone   │
                    └──────────┘
```

---

## 3. Database Schema

### 3.1 Core Tables

| Table | Purpose | Engine |
|-------|---------|--------|
| `nifty_daily` | OHLCV + VIX (2015-2026) | TimescaleDB hypertable |
| `trades` | All trades (LIVE/PAPER/SHADOW/SCORING/COMBINATION) | TimescaleDB hypertable |
| `signals` | Signal definitions + status + live metrics | Standard |
| `portfolio_state` | Daily snapshots (capital, positions, Greeks) | TimescaleDB |
| `regime_labels` | Daily regime classification | Standard |
| `alerts_log` | All alerts triggered | Standard |
| `combination_state` | SEQ_5 engine state persistence | Standard |
| `books` | Ingested books metadata | Standard |
| `market_calendar` | NSE trading days + holidays | Standard |
| `economic_calendar` | RBI decisions, budget, expiry dates | Standard |
| `fii_daily_metrics` | FII derivatives positioning | Standard |

### 3.2 Trade Types

| Type | Purpose |
|------|---------|
| `LIVE` | Real broker execution |
| `PAPER` | General paper trades |
| `PAPER_CONTROL` | DRY_20 alone (A/B baseline) |
| `PAPER_SCORING` | Weighted scoring system |
| `PAPER_COMBINATION` | GRIMES+KAUFMAN SEQ_5 |
| `SHADOW` | Regime detection signals (GUJRAL_DRY_7) |

---

## 4. Signal Pipeline

### 4.1 Extraction

**Book-specific prompts** for different book types:

| Prompt Type | Books | Special Handling |
|-------------|-------|-----------------|
| CONCRETE | Kaufman, Gujral, McMillan, Bulkowski | Direct rule extraction |
| BULKOWSKI_SPECIFIC | Bulkowski, Candlestick | Statistics tables (success rate, avg move) |
| CHAN_SPECIFIC | Chan QT, Chan AT | Python/MATLAB code → natural language |
| VINCE_SPECIFIC | Vince | Position sizing formulas |
| PRINCIPLE | Natenberg, Sinclair, Gatheral | 3 parameter variants per concept |
| METHODOLOGY | Grimes, Lopez, Hilpisch, Chan | Framework-level rules |
| PSYCHOLOGY | Douglas, Kahneman, Taleb, Schwager | Behavioral biases, risk frameworks |

### 4.2 DSL Translation

**Why DSL:** The original free-form Haiku translator produced 52% broken rules (phantom conditions, self-comparisons, unknown indicators). The DSL forces Haiku to fill a strict template.

**DSL Schema:**
```python
@dataclass
class DSLCondition:
    left: str        # must be from ALLOWED_INDICATORS (65 columns)
    operator: str    # must be from ALLOWED_OPERATORS (7 ops)
    right: str       # indicator name or numeric value

@dataclass
class DSLSignalRule:
    entry_long: List[DSLCondition]    # 1-4 conditions
    entry_short: List[DSLCondition]
    exit_long: List[DSLCondition]
    exit_short: List[DSLCondition]
    stop_loss_pct: float              # 0.5 to 10.0
    hold_days_max: int                # 1 to 30
    direction: str                    # LONG, SHORT, BOTH
    target_regime: List[str]
```

**Validation catches:**
- Phantom conditions (atr_14 > 0 — always true)
- Self-comparisons (sma_50 > sma_50 — always false)
- Unknown indicators (close[1] — not in engine)
- Scale confusion (price_pos_20 > 40 — should be 0.40)

**Result:** 95.7% pass rate vs 48.1% with old translator.

### 4.3 Indicators

67 indicators computed by `backtest/indicators.py`:

| Category | Indicators |
|----------|-----------|
| Moving Averages | sma_{5,10,20,40,50,80,100,200}, ema_{5,10,20,50,100,200} |
| RSI | rsi_{7,14,21} |
| MACD | macd, macd_signal, macd_hist |
| ATR | atr_{7,14,20}, true_range |
| Bollinger | bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth |
| Stochastic | stoch_k, stoch_d, stoch_k_5, stoch_d_5 |
| ADX | adx_14 |
| Donchian | dc_upper, dc_lower, dc_middle |
| Pivots | pivot, r1, s1, r2, s2, r3, s3 |
| Volatility | hvol_6, hvol_20, hvol_100, india_vix |
| Volume | vol_ratio_20 |
| Price Position | price_pos_20 |
| Previous Bar | prev_close, prev_high, prev_low, prev_open, prev_volume |
| Returns | returns, log_returns |
| Bar Properties | body, body_pct, upper_wick, lower_wick, range |

---

## 5. Backtesting Framework

### 5.1 Walk-Forward Engine

| Parameter | Value |
|-----------|-------|
| Training window | 36 months |
| Testing window | 12 months |
| Step size | 3 months |
| Purge gap | 21 trading days |
| Embargo | 5 trading days |
| Total windows | 26 (on 10-year data) |

### 5.2 Tier Classification

| Tier | Pass Rate | Agg Sharpe | Last 4 Windows | Max DD |
|------|-----------|-----------|----------------|--------|
| **TIER_A** | ≥ 60% | ≥ 1.5 | ≥ 75% (3/4) | ≤ 20% |
| **TIER_B** | ≥ 40% | ≥ 1.0 | ≥ 50% (2/4) | — |
| DROP | Below Tier B | — | — | — |

### 5.3 Three-Way Data Split

| Period | Date Range | Purpose |
|--------|-----------|---------|
| In-sample | 2015-01 to 2020-12 | Find combinations |
| Validation | 2021-01 to 2023-12 | Tune parameters |
| Out-of-sample | 2024-01 to 2026-03 | Final test (touch once) |

### 5.4 Combination Testing

| Logic | Description |
|-------|------------|
| AND | Both signals fire same day |
| SEQ_3 | Anchor fires, confirmation within 3 days |
| SEQ_5 | Anchor fires, confirmation within 5 days |
| ANCHOR | Signal A entry, Signal B adds exit |
| OR | Either signal fires |

**Funnel:** 4,134 combinations tested → 20 screen survivors → 4 Tier A/B → 3 OOS validated

---

## 6. Confirmed Signals

### 6.1 KAUFMAN_DRY_20 — TIER A (Paper Trading)

| Property | Value |
|----------|-------|
| Source | Kaufman, Trading Systems and Methods, Ch.20, p.865 |
| Concept | Stochastic momentum (genetic algorithm discovered rule) |
| Entry | sma_10 < prev_close AND stoch_k_5 > 50 |
| Exit | stoch_k_5 ≤ 50 |
| Stop | 2% |
| Direction | LONG only |
| WF | 21/26 windows, Sharpe 2.55, L4=4/4 |
| Correlation | +0.240 (mild positive) |
| Trades/year | ~23 |

### 6.2 KAUFMAN_DRY_16 — TIER B (Paper Trading)

| Property | Value |
|----------|-------|
| Source | Kaufman, Ch.16, p.757-797 |
| Concept | Pivot breakout + Kaufman volatility filter |
| Entry Long | close > r1 AND low ≥ pivot AND hvol_6 < hvol_100 |
| Entry Short | close < s1 AND hvol_6 < hvol_100 |
| Exit | Pivot reversal OR 3% take profit |
| Stop | 2% |
| Direction | BOTH |
| WF | 15/26 windows, Sharpe 2.03, L4=3/4 |
| Correlation | -0.352 (negative — hedge) |
| Trades/year | ~40 |

### 6.3 KAUFMAN_DRY_12 — TIER A (Shadow Mode)

| Property | Value |
|----------|-------|
| Source | Kaufman, Ch.12, p.539 |
| Concept | Volume exhaustion (price/volume divergence) |
| Entry Long | close > prev_close AND volume < prev_volume |
| Entry Short | close < prev_close AND volume < prev_volume |
| Exit | Direction reversal OR 3% take profit OR 7-day hold |
| Stop | 2% |
| Direction | BOTH |
| WF | 19/26 windows, Sharpe 1.58, L4=3/4 |
| Correlation | -0.180 (uncorrelated) |
| Status | Shadow — in 1-year drawdown, monitoring recovery |

### 6.4 GUJRAL_DRY_7 — Regime Detector (Shadow)

| Property | Value |
|----------|-------|
| Concept | Pivot crossover — detects orderly market structure |
| Usage | When 2+ of last 3 trades win → FAVORABLE regime |
| Effect | DRY_20 Sharpe jumps from 0.64 to 3.67 (5.8x) |
| Action | Adjusts position size: HIGH=1.0x, MEDIUM=0.5x, LOW=0.25x |

---

## 7. Paper Trading System

### 7.1 Three Parallel Modes

```
MODE 1: PAPER_CONTROL
  Signal:  DRY_20 alone
  Purpose: Clean baseline for comparison
  Trades:  ~23/year

MODE 2: PAPER_SCORING
  Signal:  DRY_20(×2) + DRY_12(±1) + DRY_16(±1)
  Rules:   score ≥ 3 → LONG full, ≥ 2 → LONG half, ≤ -2 → SHORT half
  Purpose: Multi-signal weighted portfolio
  Trades:  ~50/year

MODE 3: PAPER_COMBINATION
  Signal:  GRIMES_DRY_3_2 → KAUFMAN_DRY_12 SEQ_5
  Rules:   Grimes fires, Kaufman confirms within 5 days
  Purpose: High-conviction cross-book entries
  Trades:  ~6-10/year
```

### 7.2 Daily Pipeline

```
3:35 PM IST (cron) → paper_trading/daily_run.py
  │
  ├── signal_compute.py
  │     Compute indicators for DRY_20, DRY_16, DRY_12, GUJRAL_DRY_7
  │     Check entries/exits against open positions
  │     Write PAPER + SHADOW trades to DB
  │
  ├── scoring_engine.py
  │     Compute weighted score from all signals
  │     Write PAPER_SCORING trades
  │
  ├── combination_engine.py
  │     Check GRIMES fire → pending → KAUFMAN confirm
  │     State persisted to combination_state table
  │     Write PAPER_COMBINATION trades
  │
  ├── regime_detector.py
  │     Query GUJRAL_DRY_7 last 3 shadow trades
  │     Return FAVORABLE/STANDARD/UNKNOWN
  │     Adjust position size multiplier
  │
  ├── vix_monitor.py
  │     VIX ≥ 20 with DRY_20 open → WARNING
  │     VIX ≥ 25 with DRY_20 open → CRITICAL
  │
  ├── pnl_tracker.py
  │     Daily/cumulative P&L per trade_type
  │     Separate long/short tracking for DRY_16
  │
  └── Telegram digest
        Score, actions, shadow signals, regime, VIX, P&L
```

### 7.3 Monitoring Alerts

| Alert | Level | Trigger |
|-------|-------|---------|
| VIX spike + DRY_20 open | WARNING/CRITICAL | VIX > 20/25 |
| Daily loss limit | CRITICAL | P&L < -5% of capital |
| Signal rolling Sharpe | WARNING | 60-day Sharpe < 0.5 |
| Signal deactivated | WARNING | 20 consecutive negative days |
| Position mismatch | EMERGENCY | System vs broker reconciliation gap |

---

## 8. Risk Management

### 8.1 Capital Configuration

| Parameter | Value |
|-----------|-------|
| Total capital | ₹10,00,000 |
| Capital reserve | 20% (₹2,00,000 untouchable) |
| Available capital | ₹8,00,000 |
| Max positions | 4 simultaneous |
| Max same direction | 2 (max 2 LONG or 2 SHORT) |

### 8.2 Loss Limits

| Limit | Threshold | Action |
|-------|-----------|--------|
| Daily loss | 5% of capital | Halt trading |
| Weekly loss | 12% of capital | Halt trading |
| Monthly DD critical | 15% | Switch to paper mode |
| Monthly DD halt | 25% | Halt all activity |

### 8.3 Greek Limits (Options)

| Greek | Limit | Scales With |
|-------|-------|-------------|
| Portfolio Delta | ±0.50 | Fixed |
| Portfolio Vega | ±3,000 | Capital |
| Portfolio Gamma | ±30,000 | Capital |
| Portfolio Theta | -5,000 | Capital |

---

## 9. The Funnel (Numbers)

```
27 books ingested
  → 29,782 text chunks
    → ~15,000 signal concepts extracted
      → ~200 DSL-translatable (unique IDs)
        → ~30 pass correlation filter
          → 3 individual Tier A/B signals
          → 4 combination Tier A/B
            → 3 OOS-validated
              → 3 paper trading modes active
```

**Kill rates at each stage:**
- Books → tradeable concepts: 93% filtered (RISK, SIZING, PSYCHOLOGY)
- Concepts → DSL valid: 85% untranslatable
- DSL valid → walk-forward: 94% fail
- Walk-forward → OOS validated: 25% overfit
- OOS → profitable after costs: ~50%

---

## 10. Performance Summary

### 10.1 Raw Signal Performance (10 years, no costs)

| Signal | Trades | Win% | Total Points | Best Year |
|--------|--------|------|-------------|-----------|
| DRY_20 | 226 | 47% | +11,186 | 2022 (+2,804) |
| DRY_16 | 406 | 49% | +8,317 | 2022 (+2,902) |
| DRY_12 | 665 | 39% | +17,075 | 2022 (+7,554) |
| SCORING | 401 | 48% | +9,465 | 2022 (+2,534) |

### 10.2 Consistency (profitable periods out of 9)

| Signal | Score | Status |
|--------|-------|--------|
| DRY_20 | 9/9 | CONSISTENT |
| DRY_16 | 9/9 | CONSISTENT |
| SCORING | 9/9 | CONSISTENT |
| DRY_12 | 7/9 | CONSISTENT (recent drawdown) |

### 10.3 Current State (as of 2026-03-13)

| Signal | 6mo P&L | 1yr P&L | Status |
|--------|---------|---------|--------|
| DRY_20 | +35 | +2,230 | ACTIVE |
| DRY_16 | +462 | +2,299 | ACTIVE |
| SCORING | +52 | +1,528 | ACTIVE |
| DRY_12 | -2,541 | -1,768 | DRAWDOWN |

---

## 11. File Structure

```
trading-system/
├── backtest/
│   ├── generic_backtest.py      # Core backtest engine
│   ├── walk_forward.py          # Walk-forward framework
│   ├── indicators.py            # 67 technical indicators
│   ├── combination_tester.py    # Pairwise/triple combination testing
│   ├── signal_translator.py     # Old free-form translator (replaced)
│   ├── translation_validator.py # Validates translated rules
│   └── types.py                 # BacktestResult dataclass
│
├── extraction/
│   ├── orchestrator.py          # Drives extraction pipeline
│   ├── prompts.py               # Book-specific Claude prompts
│   ├── llm_client.py            # Anthropic/Ollama client
│   ├── dsl_schema.py            # DSL dataclasses + allowed indicators
│   ├── dsl_translator.py        # Strict template translator
│   ├── dsl_validator.py         # Catches phantom/broken conditions
│   ├── dsl_to_backtest.py       # Compiles DSL → backtest format
│   └── bulkowski_enrichment.py  # Pattern statistics enrichment
│
├── paper_trading/
│   ├── daily_run.py             # Daily orchestrator (cron target)
│   ├── signal_compute.py        # Computes all signal conditions
│   ├── scoring_engine.py        # Weighted multi-signal scoring
│   ├── combination_engine.py    # GRIMES+KAUFMAN SEQ_5 engine
│   ├── regime_detector.py       # GUJRAL_DRY_7 regime detection
│   ├── pnl_tracker.py           # P&L tracking per trade_type
│   └── vix_monitor.py           # VIX spike alerter
│
├── ingestion/
│   ├── pdf_ingester.py          # PDF → chunks (book profiles)
│   ├── vector_store.py          # ChromaDB interface
│   └── signal_candidate.py      # SignalCandidate dataclass
│
├── portfolio/
│   ├── signal_selector.py       # 13-step daily signal selection
│   ├── signal_registry.py       # DB query layer for signals
│   ├── signal_model.py          # Runtime Signal dataclass
│   └── portfolio_model.py       # Portfolio state management
│
├── execution/
│   ├── execution_engine.py      # Order placement (Kite API)
│   ├── greek_calculator.py      # Go risk service client
│   └── greek_pre_check.py       # Pre-trade Greek validation
│
├── monitoring/
│   ├── alert_engine.py          # Central alert processor
│   ├── alert_definitions.py     # 18 alert types with thresholds
│   └── telegram_alerter.py      # Telegram Bot sender
│
├── config/
│   └── settings.py              # Capital, limits, DB config
│
├── db/
│   ├── schema.sql               # Full PostgreSQL schema
│   └── setup.py                 # DB initialization
│
├── data/
│   └── nifty_loader.py          # NSE OHLCV data loader
│
├── extraction_results/          # Raw extracted signals (JSON per book)
├── dsl_results/                 # DSL translated signals
├── validation_results/          # Confirmed signal rules + paper trading spec
├── backtest_results/            # Walk-forward and combination results
├── combination_results/         # Systematic combination testing output
├── trade_summary/               # Simple trade logs (CSV per signal)
│
├── tests/
│   ├── test_backtest.py         # Backtest engine tests (23 tests)
│   ├── test_dsl_translator.py   # DSL pipeline tests (16 tests)
│   ├── test_scoring_engine.py   # Scoring system tests (10 tests)
│   ├── test_regime_detector.py  # Regime detection tests (10 tests)
│   ├── test_combination_engine.py # SEQ_5 combination tests (15 tests)
│   ├── test_combination_tester.py # Combination framework tests (14 tests)
│   └── test_rag_pipeline.py     # RAG pipeline tests
│
└── books/                       # PDF source books (gitignored)
```

---

## 12. Testing

**88 tests across 7 test files:**

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_backtest.py | 23 | Backtest engine, walk-forward, sensitivity |
| test_dsl_translator.py | 16 | DSL schema, validation, compilation |
| test_scoring_engine.py | 10 | Weighted scoring, position conflicts |
| test_regime_detector.py | 10 | Streak detection, confidence sizing |
| test_combination_engine.py | 15 | SEQ_5 logic, state persistence, regime filter |
| test_combination_tester.py | 14 | AND/OR/SEQ logic, no-lookahead, WF windows |
| test_rag_pipeline.py | ~20 | PDF ingestion, prompts, signal candidates |

---

## 13. Deployment

### 13.1 Paper Trading (Current)

```bash
# Daily cron job
35 15 * * 1-5 cd /path/to/trading-system && \
  venv/bin/python3 -m paper_trading.daily_run \
  >> logs/paper_trading.log 2>&1
```

### 13.2 Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...     # Claude API
DATABASE_DSN=postgresql://localhost/trading
TELEGRAM_BOT_TOKEN=...           # Telegram alerts
TELEGRAM_CHAT_ID=...             # Your chat ID
TOTAL_CAPITAL=1000000            # ₹10L (scales all limits)
```

### 13.3 Docker

```bash
docker-compose up -d  # PostgreSQL + TimescaleDB + Redis
```

---

## 14. Future Roadmap

### Phase 1 (Current): Paper Trading Validation
- 3 modes running in parallel (4 months)
- Compare CONTROL vs SCORING vs COMBINATION
- Winner informs live trading sizing

### Phase 2: Live Trading
- Connect Kite API with broker credentials
- Start with 1 lot per signal (minimum risk)
- Scale based on paper trading results

### Phase 3: Options Overlay (Tier 2)
- Natenberg/McMillan/Sinclair signals for options
- Black-Scholes option price reconstruction
- Greeks-based position management

### Phase 4: Signal Expansion
- Bulkowski pattern-to-indicator conversion
- SSRN paper signal ingestion
- NSE structural signals (FII positioning)
- Intraday data for time-of-day signals

### Phase 5: Multi-Asset
- BankNifty, FinNifty signals
- Cross-index combinations
- Intermarket signals (bonds, commodities, USD/INR)
