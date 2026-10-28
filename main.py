"""
main.py
Asynchronous Event Loop for the Anti-Gravity XAUUSD Scalping Engine.

Orchestrates:
1. Terminal connection and symbol subscription.
2. New candle detection on the M1 timeframe.
3. Feature extraction via AntiGravityAlphaEngine.
4. Risk checks and execution dispatch via ExecutionAgent.
5. Graceful shutdown handling on SIGINT/SIGTERM.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import SystemConfig, CONFIG
from alpha_engine import AntiGravityAlphaEngine, AlphaSignal, SignalDirection
from execution_agent import ExecutionAgent
from risk_controller import RiskController
from state_machine import StateMachine, BotState
from news_filter import NewsFilter, NEWS_FILTER
from ml_inference import MLInferenceEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AntiGravityBot")

SYMBOL = "XAUUSD"
TIMEFRAME = getattr(mt5, "TIMEFRAME_M1", 1) if MT5_AVAILABLE else 1
HISTORY_BARS = 120
POLL_INTERVAL_SEC = 0.5


class ScalperDaemon:
    """
    Asynchronous Scalper Daemon polling M1 rates, evaluating AntiGravity Alpha,
    evaluating ML Conformal Inference, checking Forex Factory news lockouts,
    and routing execution brackets to MetaTrader 5.
    """
    def __init__(
        self,
        symbol: str = SYMBOL,
        config: Optional[SystemConfig] = None,
        news_filter: Optional[NewsFilter] = None
    ):
        self.symbol = symbol
        self.config = config or CONFIG
        self.engine = AntiGravityAlphaEngine(config=self.config)
        self.ml_engine = MLInferenceEngine(config=self.config)
        self.executor = ExecutionAgent(symbol=symbol, config=self.config)
        self.news_filter = news_filter or NewsFilter(config=self.config)
        self.last_bar_time: int = 0
        self.is_running: bool = True
        self._mock_bar_counter: int = 0

    def initialize_terminal(self) -> bool:
        """Connects to MT5 and ensures the symbol is in Market Watch."""
        if getattr(self.config, "mock_mode", False) or not MT5_AVAILABLE:
            logger.warning("[SCALPER DAEMON] Mock/Dry-Run mode active. Synthetic rates enabled.")
            self.executor.risk.sync_daily_balance()
            return True

        init_kwargs = {"timeout": 3000}
        if self.config.mt5.path:
            init_kwargs["path"] = self.config.mt5.path
        if self.config.mt5.portable:
            init_kwargs["portable"] = True
        if self.config.mt5.login:
            init_kwargs["login"] = int(self.config.mt5.login)
        if self.config.mt5.password:
            init_kwargs["password"] = str(self.config.mt5.password)
        if self.config.mt5.server:
            init_kwargs["server"] = str(self.config.mt5.server)

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            logger.error(f"MT5 initialization failed: {err}")
            # If MT5 is already launched, try explicit mt5.login
            if self.config.mt5.login and self.config.mt5.password and self.config.mt5.server:
                if not mt5.login(login=int(self.config.mt5.login), password=str(self.config.mt5.password), server=str(self.config.mt5.server)):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                else:
                    logger.info(f"Successfully authenticated account #{self.config.mt5.login} on {self.config.mt5.server}")
            if getattr(self.config.mt5, "demo_fallback", True):
                logger.warning("Terminal offline. Falling back to synthetic simulation mode.")
                self.executor.risk.sync_daily_balance()
                return True
            return False

        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"Failed to enable {self.symbol} in Market Watch.")
            mt5.shutdown()
            return False

        # Reset daily starting benchmark for drawdown sentinel
        self.executor.risk.sync_daily_balance()

        # Synchronize Forex Factory live economic calendar
        self.news_filter.sync_calendar_sync()

        logger.info(f"Terminal synchronized. Market Watch enabled for {self.symbol}.")
        return True

    def fetch_historical_frame(self) -> pd.DataFrame:
        """Retrieves and standardizes the latest M1 bars from MT5."""
        if getattr(self.config, "mock_mode", False) or not MT5_AVAILABLE or mt5.terminal_info() is None:
            # Generate synthetic M1 candles for offline simulation
            return self._generate_synthetic_candles(HISTORY_BARS)

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, HISTORY_BARS)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('time')

        # Pull current live spread from the broker's real-time tick
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            df['spread'] = tick.ask - tick.bid
        else:
            df['spread'] = df['spread'] * 0.01  # Fallback to points conversion

        return df

    def _generate_synthetic_candles(self, n_bars: int = 120) -> pd.DataFrame:
        """Synthetic M1 bar builder for local testing and simulation"""
        now = datetime.now(timezone.utc)
        timestamps = [now - pd.Timedelta(minutes=i) for i in range(n_bars, 0, -1)]
        base_price = 2650.0
        closes = [base_price + np.sin(i / 5.0) * 2.0 for i in range(n_bars)]
        highs = [c + 0.45 for c in closes]
        lows = [c - 0.45 for c in closes]
        opens = [c - 0.10 for c in closes]
        volumes = [150 + int(np.random.randint(10, 80)) for _ in range(n_bars)]
        
        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes,
            "spread": [0.15] * n_bars
        }, index=timestamps)

    async def run(self) -> None:
        """Main async loop executing bar detection and alpha dispatch."""
        logger.info("Starting Anti-Gravity event loop...")

        while self.is_running:
            try:
                # 1. Fetch live rates
                if MT5_AVAILABLE and not getattr(self.config, "mock_mode", False) and mt5.terminal_info() is not None:
                    rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, 2)
                    if rates is None or len(rates) < 2:
                        await asyncio.sleep(POLL_INTERVAL_SEC)
                        continue
                    completed_bar_time = int(rates[-2]['time'])
                else:
                    # Synthetic tick timing
                    self._mock_bar_counter += 1
                    completed_bar_time = int(datetime.now(timezone.utc).timestamp()) // 60

                # 2. Check if a new M1 bar has closed
                if completed_bar_time > self.last_bar_time:
                    self.last_bar_time = completed_bar_time

                    df_history = self.fetch_historical_frame()
                    if df_history.empty:
                        await asyncio.sleep(POLL_INTERVAL_SEC)
                        continue

                    # Discard the currently forming open bar (index -1)
                    closed_bars = df_history.iloc[:-1] if len(df_history) > 1 else df_history

                    # 3. Evaluate Price Action & Sentinel Filters
                    signal_output = self.engine.evaluate(closed_bars)

                    if signal_output and signal_output.direction != 0:
                        dir_str = "BUY" if signal_output.direction == 1 else "SELL"
                        
                        # 4. Forex Factory Macroeconomic News Filter Gate
                        is_locked, reason, event = self.news_filter.is_news_locked()
                        if is_locked:
                            logger.warning(f"[NEWS SUPPRESSION] Signal aborted: {reason}")
                            await asyncio.sleep(POLL_INTERVAL_SEC)
                            continue

                        # 5. Machine Learning Conformal Inference Gate
                        last_c = closed_bars.iloc[-1]
                        bar_dir = 1.0 if last_c['close'] >= last_c['open'] else -1.0
                        bar_vol = float(last_c.get('tick_volume', 100))
                        features = {
                            "m1_atr": signal_output.atr_val,
                            "normalized_atr": round(signal_output.atr_val / max(0.50, signal_output.atr_val), 3),
                            "rvol_sigma": signal_output.rvol_val,
                            "wick_ratio": signal_output.rejection_ratio,
                            "delta": bar_dir * bar_vol,
                            "cvd": bar_dir * bar_vol * 3.0,
                            "tick_count": bar_vol,
                            "spread": float(signal_output.spread),
                            "sweep_dir": 1.0 if "SWEEP" in signal_output.trigger_type and signal_output.direction == 1 else (-1.0 if "SWEEP" in signal_output.trigger_type and signal_output.direction == -1 else 0.0),
                            "div_dir": float(signal_output.direction),
                            "exh_dir": float(signal_output.direction)
                        }
                        ml_res = self.ml_engine.evaluate_signal(signal_output, features)
                        if not ml_res.is_valid:
                            logger.warning(f"[ML SUPPRESSION] Signal rejected by ML gate: {ml_res.filter_reason}")
                            await asyncio.sleep(POLL_INTERVAL_SEC)
                            continue

                        logger.info(
                            f"Signal Fired & ML Confirmed: {signal_output.trigger_type} | "
                            f"Direction: {dir_str} | "
                            f"Price: {signal_output.entry_price:.2f} | "
                            f"ML Prob: {ml_res.probability:.3f} | "
                            f"ATR: {signal_output.atr_val:.2f} | "
                            f"Spread: {signal_output.spread:.3f}"
                        )

                        # 6. Route through risk gates to MT5 broker server
                        ticket = self.executor.route_alpha_signal(signal_output)
                        if ticket:
                            logger.info(f"Order executed successfully. Ticket #{ticket}")
                        else:
                            logger.warning("Order routing aborted or rejected by risk gates.")

                # 7. Sub-Second Active Position Lifecycle Supervision
                if hasattr(self.executor, "risk") and self.executor.risk.active_trades:
                    # Current Bid/Ask quote
                    if MT5_AVAILABLE and not getattr(self.config, "mock_mode", False) and mt5.terminal_info() is not None:
                        live_tick = mt5.symbol_info_tick(self.symbol)
                        cur_bid = live_tick.bid if live_tick else 2650.0
                        cur_ask = live_tick.ask if live_tick else 2650.15
                    else:
                        cur_bid = getattr(self, "_last_sim_bid", 2650.0) + float((np.random.random() - 0.49) * 0.15)
                        self._last_sim_bid = cur_bid
                        cur_ask = cur_bid + 0.15

                    cur_atr = getattr(self.engine, "latest_atr", 0.80) or 0.80

                    for t_id in list(self.executor.risk.active_trades.keys()):
                        action = self.executor.risk.evaluate_lifecycle_actions(
                            ticket=t_id,
                            current_bid=cur_bid,
                            current_ask=cur_ask,
                            current_m1_atr=cur_atr
                        )
                        act_type = action.get("action")
                        if act_type in ("MODIFY_SL", "TRAIL_SL"):
                            await self.executor.modify_brackets(t_id, new_sl=action["new_sl"])
                        elif act_type == "PARTIAL_CLOSE":
                            await self.executor.close_partial_position(t_id, close_volume=action["close_volume"])

                # Reset circuit breaker check & sync calendar at 00:00 UTC daily
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == 0 and now_utc.minute == 0 and now_utc.second < 2:
                    self.executor.risk.sync_daily_balance()
                    self.news_filter.sync_calendar_sync()

            except Exception as e:
                logger.error(f"Runtime iteration exception: {str(e)}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL_SEC)

    def shutdown(self) -> None:
        """Gracefully halts event loop and closes MT5 IPC bridge."""
        self.is_running = False
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except Exception:
                pass
        logger.info("MT5 connection closed. AntiGravity Engine terminated.")


# Backward compatibility alias for the pipeline
AntiGravityScalper = ScalperDaemon


async def main():
    daemon = ScalperDaemon()

    if not daemon.initialize_terminal():
        return

    loop = asyncio.get_running_loop()

    # Register OS signals for clean exit
    def handle_signal():
        logger.info("Termination signal received. Shutting down daemon...")
        daemon.shutdown()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_signal)
            except NotImplementedError:
                pass
    else:
        signal.signal(signal.SIGINT, lambda s, f: handle_signal())
        signal.signal(signal.SIGTERM, lambda s, f: handle_signal())

    try:
        await daemon.run()
    finally:
        daemon.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
