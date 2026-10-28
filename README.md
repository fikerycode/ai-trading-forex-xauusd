# "ANTI-GRAVITY" - AI-Driven Institutional XAUUSD Scalper

[![Architecture](https://img.shields.io/badge/Architecture-Event--Driven-blue.svg)](#)
[![Instrument](https://img.shields.io/badge/Instrument-XAUUSD%20%28Spot%20Gold%29-gold.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](#)
[![Execution](https://img.shields.io/badge/Execution-Sub--Second%20%2F%20%3C50ms-red.svg)](#)

---

## 1. System Overview & Objective
**Anti-Gravity** is a sub-second, event-driven automated quantitative scalping engine engineered for spot Gold (**XAUUSD**). The system exploits:
1. **Micro-liquidity Imbalances & Structural Sweeps:** Asian session High/Low sweeps during the London/NY overlap, Fair Value Gaps (FVG), and institutional order blocks.
2. **Momentum Exhaustion ("Anti-Gravity Vector"):** Extreme Relative Volume (**RVOL > 2.5σ**) coupled with rejection wicks (**wick-to-body ratio > 65%**) and Cumulative Volume Delta (**CVD**) divergence.
3. **Machine Learning & Statistical Conformal Inference:** LightGBM classifier paired with an inductive Conformal Predictor ensuring directional prediction probability $P(\text{Direction}) \ge 0.68$, Shannon entropy filtering, and regime gate suppression.
4. **Institutional Capital Preservation:** Non-negotiable spread gate (max 35 cents), fixed-fractional sizing (0.5%–1.0%), and a hard intraday 3.0% daily drawdown circuit breaker.

---

## 2. State Machine Topology
```
           +-----------------------------------------------+
           |                     IDLE                      |
           +-----------------------------------------------+
                                   | (Initialize & Connect)
                                   v
+--------> +-----------------------------------------------+
|          |                   SCANNING                    | <----+
|          +-----------------------------------------------+      |
|                                  | (Alpha Confluence Fired)      |
|                                  v                              |
|          +-----------------------------------------------+      |
|          |                 SIGNAL_FIRED                  |      |
|          +-----------------------------------------------+      |
|                                  | (Evaluate Sentinels & ML)    |
|                                  v                              |
|          +-----------------------------------------------+      |
|          |                  VALIDATING                   | -----+ (Rejected by Gates)
|          +-----------------------------------------------+
|                                  | (Passed: Sizing & Routing)
|                                  v
|          +-----------------------------------------------+
|          |                   EXECUTING                   |
|          +-----------------------------------------------+
|                                  | (Filled & Protective Brackets Active)
|                                  v
|          +-----------------------------------------------+
|          |                   MANAGING                    |
|          | - 1.0R: Break-Even + Spread Buffer            |
|          | - 1.5R: Scale out 50%                         |
|          | - Trail: 1.2x M1 ATR for remaining 50%        |
|          +-----------------------------------------------+
|                                  | (Position Closed / Exit)
|                                  v
|          +-----------------------------------------------+
+--------- |                     FLAT                      |
           +-----------------------------------------------+
                                   | (Daily Loss >= 3.0% Circuit Breaker)
                                   v
                         TERMINAL LOCKOUT (00:00 UTC)
```

---

## 3. Directory & Module Specifications

| Module File | Component | Core Responsibilities |
| :--- | :--- | :--- |
| `config.py` | Configuration Layer | Instrument specifications (100 oz contract size), spread limits, risk multipliers, thresholds. |
| `feed_manager.py` | Module A: Microstructure Ingestion | Async MT5 connection, Lee-Ready tick delta classifier, M1 candle aggregation, News and Spread Sentinels. |
| `alpha_engine.py` | Module B: Alpha Feature Engine | Asian High/Low sweep detection, FVG & Order Blocks, RVOL (>2.5σ), Rejection wicks (>65%), CVD divergence, Normalized ATR. |
| `ml_inference.py` | Module C: Inference Layer | LightGBM classifier, inductive conformal uncertainty estimation, Shannon entropy threshold ($H < 0.61$), regime filtering. |
| `risk_controller.py` | Module E: Capital Preservation | Fixed-fractional lot sizing, 3.0% daily drawdown circuit breaker, Break-Even router, 50% scale-out, 1.2x ATR trailing stop. |
| `execution_agent.py` | Module D: Micro-Execution Router | IOC market routing, max 10–15 cents slippage tolerance, sub-50ms broker bracket placement, emergency liquidation. |
| `state_machine.py` | Lifecycle Engine | Deterministic state transitions, invalid transition traps, and audit logging. |
| `main.py` | Master Orchestrator | Async event loop, signal handling (`SIGINT`/`SIGTERM`), tick and candle event dispatching. |

---

## 4. XAUUSD Financial Mechanics & Sizing Formula

In spot Gold, 1 Standard Lot represents **100 Troy Ounces**. Quotes are denominated in USD per Troy Oz with 2 decimal places ($0.01 = 1 point). 1 pip is standardly $0.10 (10 points).

### Fixed-Fractional Sizing
$$\text{Lot Size} = \frac{\text{Account Equity} \times \text{Risk \%}}{\text{Stop Loss Distance (USD)} \times 100}$$

*Example:*
- Account Equity: **$50,000**
- Risk: **0.75%** ($\$375.00$ at risk)
- Stop Loss Distance: **$1.50** ($15 pips)
- Loss per Lot: $\$1.50 \times 100 = \$150.00$
- Position Size: $\frac{\$375.00}{\$150.00} = \mathbf{2.50\text{ lots}}$

---

## 5. Volatility & Safety Sentinels

1. **Spread Gate:**
   - Evaluates tick spread in real time.
   - Suppresses entries if:
     $$\text{Spread} > \$0.35 \quad \text{OR} \quad \text{Spread} > 1.5 \times \text{Rolling Median Spread}$$
2. **Economic News Blocker:**
   - Tracks high-impact USD economic events (**NFP, CPI, PPI, FOMC**).
   - Enforces a mandatory hard lock **15 minutes before** and **20 minutes after** release.
3. **Max Daily Drawdown Circuit Breaker:**
   - If cumulative intraday realized loss $\ge 3.0\%$, the system liquidates all positions, cancels all pending orders, and enforces a hard lockout until **00:00 UTC**.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11+
- MetaTrader 5 Terminal (Windows 64-bit) or Headless Mock Engine

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and set your broker credentials:
```ini
MT5_LOGIN=12345678
MT5_PASSWORD=YourPassword
MT5_SERVER=YourBroker-Server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MOCK_MODE=False
```

*(Note: To test without an MT5 terminal or on dry-run environments, leave `MOCK_MODE=True`).*

### 3. Run the Scalper
```bash
python main.py
```

---

## 7. Operational Logging & Telemetry
Log output is formatted with sub-millisecond timestamps and automatically rotated in `antigravity_execution.log`:
```
2026-10-24 10:35:01.120 | INFO     | feed_manager:312 - Logged into MT5 Account #12345678
2026-10-24 10:35:05.450 | INFO     | state_machine:48 - Transition: [IDLE] -> [SCANNING] (Reason: System initialized)
2026-10-24 10:36:00.015 | SUCCESS  | main:148 - [SIGNAL VALIDATED] Conf: 0.780 | Entropy: 0.520 | Regime: COMPRESSED_RANGE
2026-10-24 10:36:00.038 | SUCCESS  | execution_agent:134 - [EXECUTION CONFIRMED] Ticket #901241 filled at $2652.40 | Latency: 23.4ms | Slippage: $0.02
2026-10-24 10:36:42.110 | INFO     | risk_controller:195 - [LIFECYCLE] Trade #901241 achieved 1.0R. Moving SL to BE+Buffer: $2652.50
```
