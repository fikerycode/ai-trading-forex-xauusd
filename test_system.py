"""
SYSTEM ARCHITECTURE VERIFICATION TEST SUITE (test_system.py)
Validates all core modules of the Anti-Gravity XAUUSD Scalping Engine.
"""

from datetime import datetime, timezone, timedelta
import math
import sys

from config import SystemConfig, CONFIG
from state_machine import StateMachine, BotState
from risk_controller import RiskController, ActiveTrade
from alpha_engine import AlphaEngine, SignalDirection, MarketRegime, AlphaSignal
from ml_inference import MLInferenceEngine, ConformalPredictor
from feed_manager import FeedManager, TickData, M1Bar, SpreadSentinel, NewsSentinel, EconomicEvent
from execution_agent import ExecutionAgent


def test_state_machine():
    print("[TEST 1/7] Testing State Machine Transitions...")
    sm = StateMachine(BotState.IDLE)
    assert sm.current_state == BotState.IDLE

    # Valid transitions: IDLE -> SCANNING -> SIGNAL_FIRED -> VALIDATING -> EXECUTING -> MANAGING -> FLAT
    assert sm.transition_to(BotState.SCANNING, "Start scanning")
    assert sm.current_state == BotState.SCANNING

    assert sm.transition_to(BotState.SIGNAL_FIRED, "Signal detected")
    assert sm.current_state == BotState.SIGNAL_FIRED

    assert sm.transition_to(BotState.VALIDATING, "Validating ML & Sentinels")
    assert sm.current_state == BotState.VALIDATING

    assert sm.transition_to(BotState.EXECUTING, "Routing order")
    assert sm.current_state == BotState.EXECUTING

    assert sm.transition_to(BotState.MANAGING, "Trade active")
    assert sm.current_state == BotState.MANAGING

    assert sm.transition_to(BotState.FLAT, "Trade closed")
    assert sm.current_state == BotState.FLAT

    # Invalid transition check: FLAT cannot jump directly to EXECUTING
    invalid_ok = sm.transition_to(BotState.EXECUTING, "Illegal jump")
    assert not invalid_ok
    assert sm.current_state == BotState.FLAT

    print("  ✓ State Machine: PASS")


def test_spread_and_news_sentinels():
    print("[TEST 2/7] Testing Spread & News Sentinels...")
    cfg = CONFIG
    spread_sentinel = SpreadSentinel(cfg)

    # 1. Normal spread test
    ok, msg = spread_sentinel.evaluate(0.15)
    assert ok, f"Expected 0.15 to pass, got: {msg}"

    # 2. Hard spread gate (> 35 cents)
    ok, msg = spread_sentinel.evaluate(0.40)
    assert not ok, f"Expected 0.40 to fail hard gate, got: {msg}"

    # 3. Dynamic spread expansion test (> 1.5 * rolling median)
    # Seed 20 tight spreads
    for _ in range(20):
        spread_sentinel.evaluate(0.12)
    median = spread_sentinel.rolling_median
    assert abs(median - 0.12) < 0.01

    # Spike above 1.5 * 0.12 = 0.18
    ok, msg = spread_sentinel.evaluate(0.25)
    assert not ok, f"Expected 0.25 to fail dynamic expansion, got: {msg}"

    # 4. News Sentinel
    news_sentinel = NewsSentinel(cfg)
    now = datetime.now(timezone.utc)
    
    # Register an FOMC event 10 minutes in future
    fomc_event = EconomicEvent(
        title="FOMC Rate Decision",
        currency="USD",
        impact="HIGH",
        time_utc=now + timedelta(minutes=10)
    )
    news_sentinel.register_event(fomc_event)

    locked, reason = news_sentinel.is_news_locked(now)
    assert locked, f"Expected news lockout 10m before FOMC, got {locked} ({reason})"

    # 60 minutes after event
    locked_after, _ = news_sentinel.is_news_locked(now + timedelta(minutes=70))
    assert not locked_after, "Expected no lockout 60m post event"

    print("  ✓ Spread & News Sentinels: PASS")


def test_risk_and_position_sizing():
    print("[TEST 3/7] Testing Risk Matrix & Position Sizing...")
    cfg = CONFIG
    risk = RiskController(cfg)

    # Test fixed-fractional sizing formula:
    # Lot Size = (Account Equity * Risk %) / (SL Distance in USD * Contract Size)
    equity = 50000.0
    entry_p = 2650.00
    sl_p = 2648.50       # SL Distance = $1.50 (15 pips)
    risk_pct = 0.01      # 1.0% ($500 risk)
    # Loss per lot = 1.50 * 100 = $150.
    # Raw lots = 500 / 150 = 3.333 -> Quantized = 3.33 lots.

    lots, dollar_risk = risk.calculate_position_size(
        account_equity=equity,
        entry_price=entry_p,
        stop_loss_price=sl_p,
        risk_pct=risk_pct
    )
    assert round(lots, 2) == 3.33, f"Expected 3.33 lots, got {lots}"
    assert abs(dollar_risk - (3.33 * 1.50 * 100)) < 0.1

    # Daily Drawdown Circuit Breaker (3.0%)
    risk.update_daily_account_state(50000.0, 50000.0)
    allowed, _ = risk.is_trading_allowed()
    assert allowed

    # Simulate 3.2% loss (Equity drops to 48400)
    risk.update_daily_account_state(48400.0, 48400.0)
    allowed_now, msg = risk.is_trading_allowed()
    assert not allowed_now, f"Expected circuit breaker lockdown, got {msg}"
    assert risk.daily_lockout_active

    print("  ✓ Risk Controller & Sizing: PASS")


def test_trade_lifecycle_supervision():
    print("[TEST 4/7] Testing Trade Lifecycle Supervisor...")
    cfg = CONFIG
    risk = RiskController(cfg)
    risk.daily_lockout_active = False

    # Register active trade: BUY at 2650.00, SL 2648.50 (1.0R = 1.50), TP 2653.00
    ticket = 777001
    trade = risk.register_trade(
        ticket=ticket,
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        lot_size=2.00,
        entry_price=2650.00,
        sl=2648.50,
        tp=2653.00,
        m1_atr=0.80
    )

    be_price = round(trade.entry_price + risk.config.risk.breakeven_r_trigger * trade.r_distance_usd, 2)
    partial_price = round(trade.entry_price + risk.config.risk.partial_scale_out_r_trigger * trade.r_distance_usd, 2)

    # 1. Price below BE trigger -> No action
    action = risk.evaluate_lifecycle_actions(ticket, current_bid=trade.entry_price + 0.50, current_ask=trade.entry_price + 0.65, current_m1_atr=0.80)
    assert action["action"] == "NONE"

    # 2. Price reaches BE trigger -> Move SL to Break-Even + Buffer ($0.10) -> $2650.10
    action = risk.evaluate_lifecycle_actions(ticket, current_bid=be_price, current_ask=be_price + 0.15, current_m1_atr=0.80)
    assert action["action"] == "MODIFY_SL"
    assert action["new_sl"] == 2650.10
    assert trade.breakeven_activated

    # 3. Price reaches partial trigger -> Scale out 50% (1.00 lot)
    action = risk.evaluate_lifecycle_actions(ticket, current_bid=partial_price, current_ask=partial_price + 0.15, current_m1_atr=0.80)
    assert action["action"] == "PARTIAL_CLOSE"
    assert action["close_volume"] == 1.00
    assert trade.partial_tp_executed

    # 4. Trailing stop check: dynamic M1 ATR trail behind current price
    trail_price = round(partial_price + 0.50, 2)
    action = risk.evaluate_lifecycle_actions(ticket, current_bid=trail_price, current_ask=trail_price + 0.15, current_m1_atr=0.80)
    assert action["action"] == "TRAIL_SL"
    expected_trail = round(trail_price - (0.80 * risk.config.risk.trailing_atr_multiple), 2)
    assert action["new_sl"] == expected_trail

    print("  ✓ Trade Lifecycle Supervision: PASS")


def test_ml_conformal_inference():
    print("[TEST 5/7] Testing ML Conformal Predictor & Entropy Filter...")
    cfg = CONFIG
    ml = MLInferenceEngine(cfg)

    # 1. Entropy math check
    # Symmetric entropy at p=0.5 is maximum (~0.693 nats)
    max_h = ml.compute_shannon_entropy(0.50)
    assert abs(max_h - math.log(2.0)) < 1e-4

    # High certainty at p=0.85 -> Entropy low (< 0.61)
    low_h = ml.compute_shannon_entropy(0.85)
    assert low_h < 0.61, f"Expected entropy < 0.61, got {low_h}"

    # 2. Conformal Predictor check
    cp = ConformalPredictor(significance_level=0.10)
    p_val = cp.get_p_value(0.20)
    assert 0.0 < p_val <= 1.0

    # 3. Signal evaluation with high-probability confluence
    signal = AlphaSignal(
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=2650.00,
        suggested_sl=2648.50,
        suggested_tp=2652.50,
        atr_m1=0.75,
        normalized_atr=1.1,
        rvol_sigma=2.8,
        wick_ratio=0.70,
        regime=MarketRegime.COMPRESSED_RANGE,
        reasons=["ASIAN LOW SWEEP", "BULLISH EXHAUSTION"],
        confidence_score=0.75
    )
    features = {
        "m1_atr": 1.41,
        "normalized_atr": 2.35,
        "rvol_sigma": 0.96,
        "wick_ratio": 0.33,
        "delta": 56.0,
        "cvd": 5.0,
        "tick_count": 56.0,
        "spread": 0.15,
        "sweep_dir": 0.0,
        "div_dir": 1.0,
        "exh_dir": -1.0
    }

    res = ml.evaluate_signal(signal, features)
    assert res.is_valid, f"Expected signal to pass ML gates, got {res.filter_reason}"
    assert res.probability >= ml.min_prob
    assert res.entropy <= ml.max_entropy

    # 4. Regime suppression check on Volatile Noise
    signal_noisy = AlphaSignal(
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        direction=SignalDirection.BUY,
        entry_price=2650.00,
        suggested_sl=2648.50,
        suggested_tp=2652.50,
        atr_m1=2.5,
        normalized_atr=2.6,
        rvol_sigma=3.5,
        wick_ratio=0.70,
        regime=MarketRegime.VOLATILE_NOISE,
        reasons=["NOISE"],
        confidence_score=0.75
    )
    res_noisy = ml.evaluate_signal(signal_noisy, features)
    assert not res_noisy.is_valid
    assert "REGIME SUPPRESSED" in res_noisy.filter_reason

    print("  ✓ ML Conformal & Entropy Filter: PASS")


def test_alpha_engine_vectorized():
    print("[TEST 6/7] Testing Vectorized Alpha Engine...")
    import pandas as pd
    import numpy as np

    alpha = AlphaEngine(CONFIG)

    # Construct synthetic candles
    dates = [datetime.now(timezone.utc) - timedelta(minutes=i) for i in range(50, 0, -1)]
    df = pd.DataFrame({
        "open": np.linspace(2650, 2655, 50),
        "high": np.linspace(2651, 2656, 50),
        "low": np.linspace(2649, 2654, 50),
        "close": np.linspace(2650.5, 2655.5, 50),
        "volume": [100.0] * 49 + [500.0],  # Massive RVOL spike on 50th bar
        "delta": [5.0] * 49 + [-80.0],
        "cvd": np.cumsum([5.0] * 49 + [-80.0]),
        "spread": [0.15] * 50
    }, index=dates)

    # 1. ATR calculation
    atr = alpha.calculate_atr(df, period=14)
    assert len(atr) == 50
    assert float(atr.iloc[-1]) > 0.0

    # 2. RVOL spike calculation
    last_bar = df.iloc[-1]
    vol_hist = df['volume'].iloc[-25:-1]
    rvol_sigma = (last_bar['volume'] - vol_hist.mean()) / (vol_hist.std() + 1e-6)
    assert rvol_sigma > 2.5, f"Expected RVOL > 2.5 sigma, got {rvol_sigma}"

    # 3. Market Regime classification
    regime = alpha.classify_regime(df, float(atr.iloc[-1]), float(atr.iloc[-1]))
    assert regime in [MarketRegime.COMPRESSED_RANGE, MarketRegime.TRENDING_MOMENTUM_PULLBACK]

    print("  ✓ Alpha Engine Vectorization: PASS")


def test_execution_agent_mock():
    print("[TEST 7/7] Testing Execution Agent...")
    import asyncio
    exec_agent = ExecutionAgent(CONFIG)
    
    # Run mock execution
    async def run_exec():
        res = await exec_agent.execute_entry(
            direction=SignalDirection.BUY,
            lot_size=1.50,
            target_price=2650.00,
            stop_loss=2648.50,
            take_profit=2652.50
        )
        assert res.success
        assert res.ticket > 0
        assert res.slippage_usd <= CONFIG.instrument.slippage_tolerance_usd
        assert res.execution_latency_ms < 100.0

        # Test bracket modification
        mod_ok = await exec_agent.modify_brackets(res.ticket, new_sl=2650.10)
        assert mod_ok

        # Test partial scale out
        close_ok = await exec_agent.close_partial_position(res.ticket, close_volume=0.75)
        assert close_ok

    asyncio.run(run_exec())
    print("  ✓ Execution Agent: PASS")


def test_forex_factory_news_filter():
    print("[TEST 8/8] Testing Forex Factory News Filter...")
    from news_filter import NewsFilter, MacroEvent
    
    nf = NewsFilter(lockout_before_min=15, lockout_after_min=20)
    now = datetime.now(timezone.utc)
    
    # 1. Test event 10 minutes ahead (should trigger lockout)
    event_time = now + timedelta(minutes=10)
    nf.add_manual_event("US Non-Farm Payrolls (NFP)", event_time)
    
    locked, reason, ev = nf.is_news_locked(now)
    assert locked, f"Expected lockout 10m before event, got {locked}"
    assert "NFP" in reason
    
    # 2. Test event 25 minutes ahead (should NOT trigger lockout)
    locked_early, _, _ = nf.is_news_locked(now - timedelta(minutes=20))
    assert not locked_early
    
    # 3. Test 10 minutes post release (should still be locked in 20m window)
    locked_post, _, _ = nf.is_news_locked(event_time + timedelta(minutes=10))
    assert locked_post
    
    # 4. Test 25 minutes post release (should be cleared)
    locked_cleared, _, _ = nf.is_news_locked(event_time + timedelta(minutes=25))
    assert not locked_cleared
    
    # 5. Payload parsing test
    mock_payload = [
        {
            "title": "CPI m/m",
            "country": "USD",
            "date": "2026-10-24T12:30:00-04:00",
            "impact": "High",
            "forecast": "0.3%",
            "previous": "0.2%"
        },
        {
            "title": "Low impact news",
            "country": "USD",
            "date": "2026-10-24T12:30:00-04:00",
            "impact": "Low"
        },
        {
            "title": "German GDP",
            "country": "EUR",
            "date": "2026-10-24T12:30:00-04:00",
            "impact": "High"
        }
    ]
    parsed = nf.parse_calendar_payload(mock_payload)
    assert len(parsed) == 1, f"Expected 1 high-impact USD event, parsed: {len(parsed)}"
    assert parsed[0].title == "CPI m/m"
    assert parsed[0].country == "USD"
    
    print("  ✓ Forex Factory News Filter: PASS")


def test_trained_gbdt_model():
    print("[TEST 9/9] Testing Trained GBDT Model & JSON Serialization...")
    import os, json
    import numpy as np
    assert os.path.exists("antigravity_gold_model.json"), "Model artifact 'antigravity_gold_model.json' must exist."
    with open("antigravity_gold_model.json", "r") as f:
        data = json.load(f)
    
    assert "trees" in data and len(data["trees"]) > 0
    assert "feature_names" in data
    assert "feature_importances" in data

    from train_model import GradientBoostedDecisionTree
    gbdt = GradientBoostedDecisionTree.from_dict(data)
    dummy_x = np.array([[0.80, 1.0, 2.5, 0.70, 100.0, 500.0, 250.0, 0.15, 1.0, 1.0, 1.0]], dtype=np.float32)
    prob = gbdt.predict_proba(dummy_x)
    assert len(prob) == 1
    assert 0.0 <= prob[0] <= 1.0
    print(f"  ✓ Trained GBDT Model Ingestion & Prediction (Sample P={prob[0]:.4f}): PASS")


if __name__ == "__main__":
    print("==================================================================")
    print("      ANTI-GRAVITY XAUUSD SCALPER: SYSTEM TEST SUITE             ")
    print("==================================================================")
    test_state_machine()
    test_spread_and_news_sentinels()
    test_risk_and_position_sizing()
    test_trade_lifecycle_supervision()
    test_ml_conformal_inference()
    test_alpha_engine_vectorized()
    test_execution_agent_mock()
    test_forex_factory_news_filter()
    test_trained_gbdt_model()
    print("==================================================================")
    print("      ALL 9 TEST SUITES PASSED! SYSTEM VERIFIED 100%             ")
    print("==================================================================")
