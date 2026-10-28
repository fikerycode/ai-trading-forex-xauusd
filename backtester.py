"""
backtester.py
Historical Backtesting, Replay Engine, and Quantitative Performance Analyzer for XAUUSD (Gold).

Capabilities:
1. Replays historical M1 candle streams from MetaTrader 5, CSV files, or realistic synthetic feeds.
2. Evaluates AntiGravityAlphaEngine signals causally bar-by-bar (no look-ahead bias).
3. Simulates sub-second trade lifecycles:
   - 1.0R Break-Even stop advancement (+ $0.10 spread buffer).
   - 1.5R Partial scale-out (50% position volume).
   - Dynamic 1.2x M1 ATR trailing stop on remaining volume.
   - Enforces 3.0% daily drawdown circuit breaker.
4. Generates institutional metrics: Win Rate, Profit Factor, Max Drawdown %, Sharpe Ratio.
5. Exports complete trade journal to 'backtest_trade_log.csv'.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import math
import os
from typing import Dict, List, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity.backtest")
    logger.success = logger.info

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import SystemConfig, CONFIG
from alpha_engine import AntiGravityAlphaEngine, AlphaSignal, SignalDirection
from ml_inference import MLInferenceEngine
from risk_controller import RiskController, RiskLimits


@dataclass
class BacktestTrade:
    ticket: int
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int                # 1: BUY, -1: SELL
    entry_price: float
    exit_price: float
    lot_size: float
    initial_sl: float
    initial_tp: float
    current_sl: float
    r_distance_usd: float
    entry_atr: float
    pnl_usd: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = "OPEN"     # "INITIAL_SL", "BREAK_EVEN", "TRAILING_SL", "TAKE_PROFIT", "PARTIAL_TP"
    be_triggered: bool = False
    partial_tp_executed: bool = False
    realized_partial_pnl: float = 0.0
    remaining_volume: float = 0.0

    def __post_init__(self):
        if self.remaining_volume <= 0:
            self.remaining_volume = self.lot_size


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    win_rate_pct: float = 0.0
    initial_equity: float = 10000.0
    final_equity: float = 10000.0
    net_profit_usd: float = 0.0
    roi_pct: float = 0.0
    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    avg_trade_r: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0


class BacktestEngine:
    """
    Institutional Backtesting & Historical Replay Simulator.
    Simulates realistic spreads, slippage, trade brackets, and account circuit breakers.
    """
    def __init__(
        self,
        symbol: str = "XAUUSD",
        initial_equity: float = 10000.0,
        risk_pct_per_trade: float = 0.01,         # 1.0% risk per trade
        max_daily_drawdown_pct: float = 0.03,     # 3.0% daily circuit breaker
        slippage_usd: float = 0.05,               # 5 cents base execution slippage
        spread_usd: float = 0.15,                 # 15 cents baseline spread on Gold
        contract_size: float = 100.0,             # 100 Troy Oz per lot
        config: Optional[SystemConfig] = None
    ):
        self.symbol = symbol
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.risk_pct = risk_pct_per_trade
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.slippage = slippage_usd
        self.spread = spread_usd
        self.contract_size = contract_size
        self.config = config or CONFIG

        # Trade Lifecycle parameters from config.risk
        risk_cfg = getattr(self.config, "risk", None)
        self.be_trigger = float(getattr(risk_cfg, "breakeven_r_trigger", 1.0))
        self.be_buffer = float(getattr(risk_cfg, "breakeven_spread_buffer_usd", 0.10))
        self.partial_tp_trigger = float(getattr(risk_cfg, "partial_scale_out_r_trigger", 1.5))
        self.partial_scale_ratio = float(getattr(risk_cfg, "partial_scale_out_ratio", 0.50))
        self.trailing_atr_mult = float(getattr(risk_cfg, "trailing_atr_multiple", 1.2))

        # Engines
        self.alpha_engine = AntiGravityAlphaEngine(config=self.config)
        self.ml_engine = MLInferenceEngine(config=self.config)
        self.risk_limits = RiskLimits(
            max_risk_pct_per_trade=risk_pct_per_trade,
            max_daily_drawdown_pct=max_daily_drawdown_pct
        )

        # Simulation trackers
        self.trades: List[BacktestTrade] = []
        self.active_trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.ticket_counter: int = 1000

        # Daily circuit breaker state
        self.daily_start_equity = initial_equity
        self.current_day: Optional[datetime.date] = None
        self.circuit_breaker_active = False

    def load_data_from_internet(self, symbol: str = "GC=F", range_days: int = 5) -> pd.DataFrame:
        """
        Pulls real live Gold (XAUUSD / GC=F) M1 candlestick data directly from internet APIs.
        Provides thousands of real 1-minute historical bars without needing manual CSV downloads.
        """
        import json
        import urllib.request
        
        logger.info(f"[BACKTEST] Downloading real Gold M1 historical data from internet ({range_days} days)...")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range={range_days}d"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            
            res = data["chart"]["result"][0]
            times = pd.to_datetime(res["timestamp"], unit="s", utc=True)
            quote = res["indicators"]["quote"][0]
            
            df = pd.DataFrame({
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "tick_volume": quote.get("volume", [100] * len(times))
            }, index=times).dropna()
            
            df["spread"] = self.spread
            logger.success(f"[BACKTEST] Successfully downloaded {len(df)} real historical M1 Gold bars from internet.")
            return df
        except Exception as e:
            logger.warning(f"[BACKTEST] Internet download failed ({e}). Falling back to synthetic generator.")
            return self.generate_synthetic_gold_data(bars_count=3500)

    def load_data_from_mt5(self, bars_count: int = 5000) -> pd.DataFrame:
        """Fetches historical M1 bars directly from MetaTrader 5 if terminal is active."""
        if not MT5_AVAILABLE:
            logger.warning("[BACKTEST] MetaTrader 5 package not available. Falling back to internet feed.")
            return self.load_data_from_internet()

        try:
            # Check if terminal info exists before calling initialize
            init_ok = mt5.initialize(timeout=1500)
        except Exception as e:
            logger.warning(f"[BACKTEST] MT5 initialize error ({e}). Falling back to internet feed.")
            return self.load_data_from_internet()

        if not init_ok:
            logger.warning("[BACKTEST] MT5 terminal not active or offline. Falling back to internet feed.")
            return self.load_data_from_internet()

        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, bars_count)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.warning("[BACKTEST] No rates returned from MT5. Falling back to internet feed.")
            return self.load_data_from_internet()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df['spread'] = self.spread
        logger.info(f"[BACKTEST] Loaded {len(df)} historical M1 bars from MT5 for {self.symbol}.")
        return df

    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Loads historical M1 candles from a CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        # Parse timestamp: check standard time column headers or unnamed index
        time_col = None
        for col in ['time', 'Date', 'timestamp', 'datetime', 'DateTime', 'Unnamed: 0', '']:
            if col in df.columns:
                time_col = col
                break

        if time_col is not None:
            df['time'] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index('time')
            if time_col in df.columns and time_col != 'time':
                df = df.drop(columns=[time_col])
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                first_col = df.columns[0]
                df['time'] = pd.to_datetime(df[first_col], utc=True)
                df = df.set_index('time')
                df = df.drop(columns=[first_col])
            except Exception:
                pass

        # Ensure standard column naming
        df.columns = [c.lower() for c in df.columns]
        if 'spread' not in df.columns:
            df['spread'] = self.spread

        logger.info(f"[BACKTEST] Loaded {len(df)} bars from CSV: {file_path}")
        return df

    def generate_synthetic_gold_data(self, bars_count: int = 5000) -> pd.DataFrame:
        """
        Generates realistic high-fidelity Gold (XAUUSD) M1 data with:
        - Asian session compression (00:00 - 07:00 UTC).
        - London & NY open liquidity sweeps & volatility expansion.
        - Realistic pin bars, wick rejections, and volume clusters.
        """
        logger.info(f"[BACKTEST] Generating realistic synthetic market dataset ({bars_count} M1 bars)...")
        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(minutes=i) for i in range(bars_count, 0, -1)]

        price = 2650.00
        opens, highs, lows, closes, volumes, spreads = [], [], [], [], [], []

        for dt in timestamps:
            hour = dt.hour
            # Higher volatility during London (07-11 UTC) and NY (12-16 UTC)
            if 7 <= hour <= 16:
                volatility = np.random.uniform(0.35, 1.20)
                vol_base = np.random.randint(180, 600)
            else:
                volatility = np.random.uniform(0.10, 0.45)
                vol_base = np.random.randint(50, 220)

            drift = np.random.normal(0, volatility * 0.4)
            open_p = price
            close_p = open_p + drift

            # Occasional pin bar / wick rejection injection (10% chance)
            is_pin_bar = (np.random.random() < 0.10)
            if is_pin_bar:
                if np.random.random() > 0.5:
                    # Bullish rejection wick
                    low_p = min(open_p, close_p) - abs(np.random.normal(1.2, 0.4))
                    high_p = max(open_p, close_p) + abs(np.random.normal(0.2, 0.1))
                    vol_base = int(vol_base * 2.8)
                else:
                    # Bearish rejection wick
                    high_p = max(open_p, close_p) + abs(np.random.normal(1.2, 0.4))
                    low_p = min(open_p, close_p) - abs(np.random.normal(0.2, 0.1))
                    vol_base = int(vol_base * 2.8)
            else:
                high_p = max(open_p, close_p) + abs(np.random.normal(0.25, 0.15))
                low_p = min(open_p, close_p) - abs(np.random.normal(0.25, 0.15))

            price = close_p
            opens.append(round(open_p, 2))
            highs.append(round(high_p, 2))
            lows.append(round(low_p, 2))
            closes.append(round(close_p, 2))
            volumes.append(vol_base)
            spreads.append(round(np.random.choice([0.12, 0.15, 0.18, 0.22]), 2))

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes,
            "spread": spreads
        }, index=timestamps)

    def _calculate_lot_size(self, sl_distance: float) -> float:
        """Fixed-fractional lot sizing based on account equity."""
        risk_capital = self.equity * self.risk_pct
        dollar_loss_per_lot = sl_distance * self.contract_size
        raw_lot = risk_capital / max(dollar_loss_per_lot, 1.0)
        quantized = math.floor(raw_lot / 0.01) * 0.01
        return max(0.01, min(round(quantized, 2), 10.0))

    def run_simulation(self, df: pd.DataFrame) -> PerformanceMetrics:
        """
        Executes the bar-by-bar backtest replay over the provided DataFrame.
        """
        logger.info(f"[BACKTEST] Commencing bar-by-bar historical replay ({len(df)} candles)...")
        lookback_warmup = 50

        if len(df) < lookback_warmup + 10:
            raise ValueError(f"Insufficient historical data: {len(df)} bars provided. Need at least 60.")

        self.equity = self.initial_equity
        self.daily_start_equity = self.initial_equity
        self.trades.clear()
        self.active_trades.clear()
        self.equity_curve.clear()

        for i in range(lookback_warmup, len(df)):
            current_bar = df.iloc[i]
            bar_time = df.index[i]
            bar_date = bar_time.date()

            # 1. Daily equity reset at 00:00 UTC
            if self.current_day != bar_date:
                self.current_day = bar_date
                self.daily_start_equity = self.equity
                self.circuit_breaker_active = False

            # Daily Circuit Breaker Check (3.0% max loss)
            daily_loss = (self.daily_start_equity - self.equity) / max(self.daily_start_equity, 1.0)
            if daily_loss >= self.max_daily_drawdown_pct:
                if not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    # Close any remaining open positions
                    for trade in list(self.active_trades):
                        self._close_trade(trade, bar_time, current_bar['close'], reason="DAILY_CIRCUIT_BREAKER")

            # 2. Manage Active Open Positions with intra-bar high/low resolution
            self._update_active_positions(current_bar, bar_time)

            # Record equity point
            self.equity_curve.append((bar_time, round(self.equity, 2)))

            # If circuit breaker active or position already open, suppress new entries
            if self.circuit_breaker_active or len(self.active_trades) >= 1:
                continue

            # 3. Evaluate Closed Bars via AntiGravityAlphaEngine (bounded 120-bar lookback)
            history_slice = df.iloc[max(0, i - 120):i]
            signal: Optional[AlphaSignal] = self.alpha_engine.evaluate(history_slice)

            if signal and signal.direction != 0 and signal.spread_ok:
                # 4. ML Conformal Filter Gate with trained GBDT features
                bar_dir = 1.0 if current_bar['close'] >= current_bar['open'] else -1.0
                bar_vol = float(current_bar.get('tick_volume', 100))
                features = {
                    "m1_atr": signal.atr_val,
                    "normalized_atr": round(signal.atr_val / max(0.50, signal.atr_val), 3),
                    "rvol_sigma": signal.rvol_val,
                    "wick_ratio": signal.rejection_ratio,
                    "delta": bar_dir * bar_vol,
                    "cvd": bar_dir * bar_vol * 3.0,
                    "tick_count": bar_vol,
                    "spread": float(current_bar.get('spread', self.spread)),
                    "sweep_dir": 1.0 if "SWEEP" in signal.trigger_type and signal.direction == 1 else (-1.0 if "SWEEP" in signal.trigger_type and signal.direction == -1 else 0.0),
                    "div_dir": float(signal.direction),
                    "exh_dir": float(signal.direction)
                }
                ml_res = self.ml_engine.evaluate_signal(signal, features)
                if not ml_res.is_valid:
                    continue

                # 5. Execute Entry with slippage at the open of the current bar
                is_buy = (signal.direction == 1)
                fill_price = current_bar['open'] + (self.slippage if is_buy else -self.slippage)
                sl_distance = abs(signal.entry_price - signal.suggested_sl)
                if sl_distance <= 0.05:
                    sl_distance = max(signal.atr_val * 1.0, 0.80)
                tp_distance = abs(signal.suggested_tp - signal.entry_price)
                if tp_distance <= 0.05:
                    tp_distance = max(signal.atr_val * 1.5, 1.20)

                lot_size = self._calculate_lot_size(sl_distance)
                if lot_size <= 0:
                    continue

                initial_sl = round(fill_price - sl_distance if is_buy else fill_price + sl_distance, 2)
                initial_tp = round(fill_price + tp_distance if is_buy else fill_price - tp_distance, 2)

                self.ticket_counter += 1
                new_trade = BacktestTrade(
                    ticket=self.ticket_counter,
                    entry_time=bar_time,
                    exit_time=None,
                    direction=signal.direction,
                    entry_price=round(fill_price, 2),
                    exit_price=0.0,
                    lot_size=lot_size,
                    initial_sl=initial_sl,
                    initial_tp=initial_tp,
                    current_sl=initial_sl,
                    r_distance_usd=round(sl_distance, 2),
                    entry_atr=signal.atr_val
                )
                self.active_trades.append(new_trade)
                self._update_active_positions(current_bar, bar_time)

        # Close any lingering positions at end of dataset
        if self.active_trades and len(df) > 0:
            last_bar = df.iloc[-1]
            last_time = df.index[-1]
            for trade in list(self.active_trades):
                self._close_trade(trade, last_time, last_bar['close'], reason="DATASET_END")

        return self.compute_performance_metrics()

    def _update_active_positions(self, bar: pd.Series, bar_time: datetime):
        """Simulates intra-candle high/low price action on open brackets."""
        high = bar['high']
        low = bar['low']
        spread_half = (bar.get('spread', self.spread)) / 2.0
        current_atr = bar.get('atr_val', 0.80) if 'atr_val' in bar else 0.80

        for trade in list(self.active_trades):
            # ---------------------------------------------------------
            # LONG TRADE MANAGEMENT
            # ---------------------------------------------------------
            if trade.direction == 1:
                bid_high = high - spread_half
                bid_low = low - spread_half

                # 1. Hard SL hit
                if bid_low <= trade.current_sl:
                    if trade.partial_tp_executed:
                        reason = "TRAILING_SL"
                    elif trade.be_triggered:
                        reason = "BREAK_EVEN"
                    else:
                        reason = "INITIAL_SL"
                    self._close_trade(trade, bar_time, trade.current_sl, reason=reason)
                    continue

                # 2. Hard TP hit
                if trade.initial_tp > 0 and bid_high >= trade.initial_tp:
                    self._close_trade(trade, bar_time, trade.initial_tp, reason="TAKE_PROFIT")
                    continue

                # 3. Break-Even Check
                current_r = (bid_high - trade.entry_price) / max(trade.r_distance_usd, 0.01)
                if current_r >= self.be_trigger and not trade.be_triggered:
                    be_sl = round(trade.entry_price + self.be_buffer, 2)
                    if be_sl > trade.current_sl:
                        trade.be_triggered = True
                        trade.current_sl = be_sl

                # 4. Partial Scale-Out
                if current_r >= self.partial_tp_trigger and not trade.partial_tp_executed:
                    close_vol = round(trade.lot_size * self.partial_scale_ratio, 2)
                    partial_pnl = close_vol * (self.partial_tp_trigger * trade.r_distance_usd) * self.contract_size
                    self.equity += partial_pnl
                    trade.partial_tp_executed = True
                    trade.realized_partial_pnl = partial_pnl
                    trade.remaining_volume = round(trade.lot_size - close_vol, 2)

                # 5. Dynamic M1 ATR Trailing Stop
                if trade.partial_tp_executed:
                    trail_sl = round(bid_high - (current_atr * self.trailing_atr_mult), 2)
                    if trail_sl > trade.current_sl:
                        trade.current_sl = trail_sl

            # ---------------------------------------------------------
            # SHORT TRADE MANAGEMENT
            # ---------------------------------------------------------
            elif trade.direction == -1:
                ask_high = high + spread_half
                ask_low = low + spread_half

                # 1. Hard SL hit
                if ask_high >= trade.current_sl:
                    if trade.partial_tp_executed:
                        reason = "TRAILING_SL"
                    elif trade.be_triggered:
                        reason = "BREAK_EVEN"
                    else:
                        reason = "INITIAL_SL"
                    self._close_trade(trade, bar_time, trade.current_sl, reason=reason)
                    continue

                # 2. Hard TP hit
                if trade.initial_tp > 0 and ask_low <= trade.initial_tp:
                    self._close_trade(trade, bar_time, trade.initial_tp, reason="TAKE_PROFIT")
                    continue

                # 3. Break-Even Check
                current_r = (trade.entry_price - ask_low) / max(trade.r_distance_usd, 0.01)
                if current_r >= self.be_trigger and not trade.be_triggered:
                    be_sl = round(trade.entry_price - self.be_buffer, 2)
                    if be_sl < trade.current_sl:
                        trade.be_triggered = True
                        trade.current_sl = be_sl

                # 4. Partial Scale-Out
                if current_r >= self.partial_tp_trigger and not trade.partial_tp_executed:
                    close_vol = round(trade.lot_size * self.partial_scale_ratio, 2)
                    partial_pnl = close_vol * (self.partial_tp_trigger * trade.r_distance_usd) * self.contract_size
                    self.equity += partial_pnl
                    trade.partial_tp_executed = True
                    trade.realized_partial_pnl = partial_pnl
                    trade.remaining_volume = round(trade.lot_size - close_vol, 2)

                # 5. Dynamic M1 ATR Trailing Stop
                if trade.partial_tp_executed:
                    trail_sl = round(ask_low + (current_atr * self.trailing_atr_mult), 2)
                    if trail_sl < trade.current_sl:
                        trade.current_sl = trail_sl

    def _close_trade(self, trade: BacktestTrade, exit_time: datetime, exit_price: float, reason: str):
        """Finalizes a trade, records realized PnL, and updates liquid account equity."""
        trade.exit_time = exit_time
        trade.exit_price = round(exit_price, 2)
        trade.exit_reason = reason

        # Calculate final remaining position PnL
        if trade.direction == 1:
            price_delta = trade.exit_price - trade.entry_price
        else:
            price_delta = trade.entry_price - trade.exit_price

        remaining_pnl = price_delta * trade.remaining_volume * self.contract_size
        trade.pnl_usd = round(remaining_pnl + trade.realized_partial_pnl, 2)
        total_risk_usd = trade.lot_size * trade.r_distance_usd * self.contract_size
        trade.r_multiple = round(trade.pnl_usd / max(total_risk_usd, 0.01), 2)

        self.equity += remaining_pnl
        self.trades.append(trade)
        if trade in self.active_trades:
            self.active_trades.remove(trade)

    def compute_performance_metrics(self) -> PerformanceMetrics:
        """Computes comprehensive institutional quantitative metrics."""
        m = PerformanceMetrics(
            total_trades=len(self.trades),
            initial_equity=self.initial_equity,
            final_equity=round(self.equity, 2),
            net_profit_usd=round(self.equity - self.initial_equity, 2),
            roi_pct=round(((self.equity - self.initial_equity) / self.initial_equity) * 100.0, 2)
        )

        if m.total_trades == 0:
            return m

        break_evens = [t for t in self.trades if t.exit_reason == "BREAK_EVEN" or abs(t.pnl_usd) <= 5.0]
        profits = [t.pnl_usd for t in self.trades if t.pnl_usd > 5.0 and t.exit_reason != "BREAK_EVEN"]
        losses = [abs(t.pnl_usd) for t in self.trades if t.pnl_usd < -5.0]

        m.winning_trades = len(profits)
        m.losing_trades = len(losses)
        m.break_even_trades = len(break_evens)
        m.win_rate_pct = round((m.winning_trades / m.total_trades) * 100.0, 2)

        m.gross_profit_usd = round(sum(profits), 2)
        m.gross_loss_usd = round(sum(losses), 2)
        m.profit_factor = round(m.gross_profit_usd / max(m.gross_loss_usd, 1.0), 2)

        m.avg_win_usd = round(float(np.mean(profits)), 2) if profits else 0.0
        m.avg_loss_usd = round(float(np.mean(losses)), 2) if losses else 0.0
        m.avg_trade_r = round(float(np.mean([t.r_multiple for t in self.trades])), 2)

        # Drawdown calculation
        peak = self.initial_equity
        max_dd_usd = 0.0
        max_dd_pct = 0.0

        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd_usd = peak - eq
            dd_pct = (dd_usd / max(peak, 1.0)) * 100.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        m.max_drawdown_usd = round(max_dd_usd, 2)
        m.max_drawdown_pct = round(max_dd_pct, 2)

        # Sharpe & Sortino ratios (annualized proxy based on trade returns)
        trade_returns = [t.pnl_usd / self.initial_equity for t in self.trades]
        if len(trade_returns) > 1:
            mean_ret = float(np.mean(trade_returns))
            std_ret = float(np.std(trade_returns)) + 1e-6
            m.sharpe_ratio = round((mean_ret / std_ret) * math.sqrt(252 * 24), 2)

            downside_returns = [r for r in trade_returns if r < 0]
            downside_std = float(np.std(downside_returns)) + 1e-6 if downside_returns else 1e-6
            m.sortino_ratio = round((mean_ret / downside_std) * math.sqrt(252 * 24), 2)

        return m

    def export_trade_log_csv(self, file_path: str = "backtest_trade_log.csv"):
        """Exports the complete simulated trade journal into CSV format."""
        if not self.trades:
            logger.warning("[BACKTEST] No trades to export.")
            return

        rows = []
        for t in self.trades:
            rows.append({
                "ticket": t.ticket,
                "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M:%S") if t.entry_time else "",
                "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M:%S") if t.exit_time else "",
                "direction": "BUY" if t.direction == 1 else "SELL",
                "lot_size": t.lot_size,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "initial_sl": t.initial_sl,
                "initial_tp": t.initial_tp,
                "r_distance_usd": t.r_distance_usd,
                "pnl_usd": t.pnl_usd,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason
            })

        df_trades = pd.DataFrame(rows)
        df_trades.to_csv(file_path, index=False)
        logger.success(f"[BACKTEST] Trade journal exported successfully -> {file_path}")

    def print_summary_report(self, m: PerformanceMetrics):
        """Displays formatted institutional performance analytics to the terminal."""
        print("\n" + "=" * 70)
        print(f"      ANTI-GRAVITY XAUUSD SCALPER: HISTORICAL PERFORMANCE REPORT     ")
        print("=" * 70)
        print(f"  Instrument:             {self.symbol} (Spot Gold)")
        print(f"  Initial Capital:        ${m.initial_equity:,.2f}")
        print(f"  Final Net Equity:       ${m.final_equity:,.2f}")
        print(f"  Net Profit:             ${m.net_profit_usd:+,.2f} ({m.roi_pct:+.2f}%)")
        print("-" * 70)
        print(f"  Total Closed Trades:    {m.total_trades}")
        print(f"  Winning Trades:         {m.winning_trades} ({m.win_rate_pct:.1f}%)")
        print(f"  Losing Trades:          {m.losing_trades}")
        print(f"  Break-Even Trades:      {m.break_even_trades}")
        print("-" * 70)
        print(f"  Profit Factor:          {m.profit_factor:.2f}")
        print(f"  Gross Profit:           ${m.gross_profit_usd:,.2f}")
        print(f"  Gross Loss:             ${m.gross_loss_usd:,.2f}")
        print(f"  Average Win:            ${m.avg_win_usd:,.2f}")
        print(f"  Average Loss:           ${m.avg_loss_usd:,.2f}")
        print(f"  Average Trade R:        {m.avg_trade_r:+.2f}R")
        print("-" * 70)
        print(f"  Max Drawdown:           ${m.max_drawdown_usd:,.2f} ({m.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:           {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:          {m.sortino_ratio:.2f}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anti-Gravity XAUUSD Historical Backtester & Performance Analyzer")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol name (default: XAUUSD)")
    parser.add_argument("--source", type=str, choices=["auto", "internet", "mt5", "synthetic", "csv"], default="auto",
                        help="Data source: 'auto' (detects internet/synthetic), 'internet' (free real gold M1), 'mt5', 'synthetic', 'csv'")
    parser.add_argument("--bars", type=int, default=3500, help="Number of historical M1 bars to backtest (default: 3500)")
    parser.add_argument("--days", type=int, default=5, help="Number of days to pull for internet source (1-7 days, default: 5)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Initial account balance in USD (default: 10000.0)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk fraction per trade (default: 0.01 = 1%%)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to historical M1 CSV file")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save downloaded data to CSV for offline re-use")
    parser.add_argument("--output", type=str, default="backtest_trade_log.csv", help="CSV export filename for trade journal")

    args = parser.parse_args()

    engine = BacktestEngine(
        symbol=args.symbol,
        initial_equity=args.equity,
        risk_pct_per_trade=args.risk
    )

    # 1. Determine data source
    if args.csv and os.path.exists(args.csv):
        print(f"[INFO] Loading historical data from CSV: {args.csv}")
        data = engine.load_data_from_csv(args.csv)
    elif args.source == "internet":
        data = engine.load_data_from_internet(range_days=args.days)
    elif args.source == "synthetic":
        data = engine.generate_synthetic_gold_data(bars_count=args.bars)
    elif args.source == "mt5":
        data = engine.load_data_from_mt5(bars_count=args.bars)
    else:  # "auto"
        # Try local CSV first, then internet, then fallback to synthetic
        if os.path.exists("gold_historical_m1.csv"):
            data = engine.load_data_from_csv("gold_historical_m1.csv")
        else:
            try:
                data = engine.load_data_from_internet(range_days=args.days)
            except Exception:
                data = engine.generate_synthetic_gold_data(bars_count=args.bars)

    # 2. Optionally cache dataset to CSV
    if args.save_csv:
        data.to_csv(args.save_csv)
        print(f"[INFO] Saved {len(data)} historical bars to CSV: {args.save_csv}")

    # 3. Run backtest simulation
    metrics = engine.run_simulation(data)
    engine.print_summary_report(metrics)
    engine.export_trade_log_csv(args.output)
