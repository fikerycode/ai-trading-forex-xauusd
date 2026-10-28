"""
risk_controller.py
Dynamic Lot Sizing, Account Equity Gating, and Daily Circuit Breaker for XAUUSD (Gold).

Provides:
1. RiskLimits: Max risk per trade (0.5% - 1.0%), daily drawdown circuit breaker (3.0%), min/max lot caps.
2. RiskController: Direct MT5 account and symbol sync, lot sizing formula, and trade lifecycle supervision.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, time
import math
from typing import Dict, List, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity")
    logger.success = logger.info

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import SystemConfig, CONFIG
from alpha_engine import SignalDirection


@dataclass
class RiskLimits:
    max_risk_pct_per_trade: float = 0.01  # 1.0% max account risk
    max_daily_drawdown_pct: float = 0.03  # 3.0% max daily account drawdown
    min_lot: float = 0.01
    max_lot: float = 10.0


@dataclass
class ActiveTrade:
    ticket: int
    symbol: str
    direction: SignalDirection
    lot_size: float
    entry_price: float
    initial_sl: float
    initial_tp: float
    current_sl: float
    current_tp: float
    r_distance_usd: float          # 1.0R dollar distance per unit
    m1_atr_at_entry: float
    entry_time_utc: datetime
    breakeven_activated: bool = False
    partial_tp_executed: bool = False
    remaining_ratio: float = 1.0


class RiskController:
    """
    Capital Preservation & Lot Sizing Controller for XAUUSD.
    - Benchmarks daily opening equity at 00:00 UTC.
    - Halts execution if intraday loss >= 3.0% (hard circuit breaker).
    - Sizes positions via: Lot = (Equity * Risk%) / (SL_distance * Contract_Size).
    - Supervises active trade lifecycle (Break-even at 1.0R, partial scale-out at 1.5R, trailing stop).
    """
    def __init__(
        self,
        symbol: Union[str, SystemConfig] = "XAUUSD",
        limits: Optional[RiskLimits] = None,
        config: Optional[SystemConfig] = None
    ):
        # Support SystemConfig instance passed as first positional parameter
        if isinstance(symbol, SystemConfig) or hasattr(symbol, "instrument"):
            self.config = symbol
            self.symbol = self.config.instrument.symbol
            self.limits = limits or RiskLimits(
                max_risk_pct_per_trade=self.config.risk.default_risk_pct,
                max_daily_drawdown_pct=self.config.risk.max_daily_loss_pct,
                min_lot=self.config.instrument.min_lot,
                max_lot=self.config.instrument.max_lot
            )
        else:
            self.config = config or CONFIG
            self.symbol = str(symbol)
            self.limits = limits or RiskLimits()

        self.starting_daily_equity: float = 0.0
        self.circuit_breaker_tripped: bool = False
        self.current_trading_day: Optional[datetime.date] = None
        self.realized_pnl_today: float = 0.0

        # Active trade registry for lifecycle monitoring
        self.active_trades: Dict[int, ActiveTrade] = {}

    @property
    def daily_lockout_active(self) -> bool:
        return self.circuit_breaker_tripped

    @daily_lockout_active.setter
    def daily_lockout_active(self, value: bool):
        self.circuit_breaker_tripped = value

    def sync_daily_balance(self) -> None:
        """Call at 00:00 UTC to benchmark daily starting balance."""
        today = datetime.now(timezone.utc).date()
        self.current_trading_day = today

        if MT5_AVAILABLE and not getattr(self.config, "mock_mode", False):
            acc = mt5.account_info()
            if acc:
                self.starting_daily_equity = float(acc.equity)
                self.circuit_breaker_tripped = False
                logger.info(f"[RISK CONTROLLER] Synced daily balance: ${self.starting_daily_equity:.2f}")
                return

        # Fallback for mock simulation
        if self.starting_daily_equity <= 0:
            self.starting_daily_equity = 10000.0
        self.circuit_breaker_tripped = False
        logger.info(f"[RISK CONTROLLER] Mock daily equity benchmarked: ${self.starting_daily_equity:.2f}")

    def check_circuit_breaker(self) -> bool:
        """Halts execution if intraday loss exceeds max daily threshold (3.0%)."""
        current_equity = 0.0

        if MT5_AVAILABLE and not getattr(self.config, "mock_mode", False):
            acc = mt5.account_info()
            if acc:
                current_equity = float(acc.equity)

        if current_equity <= 0:
            current_equity = self.starting_daily_equity

        if self.starting_daily_equity <= 0:
            self.starting_daily_equity = current_equity

        daily_loss = (self.starting_daily_equity - current_equity) / max(self.starting_daily_equity, 1.0)
        if daily_loss >= self.limits.max_daily_drawdown_pct:
            if not self.circuit_breaker_tripped:
                logger.critical(
                    f"[CIRCUIT BREAKER TRIGGERED] Intraday Loss {daily_loss * 100:.2f}% >= "
                    f"Max {self.limits.max_daily_drawdown_pct * 100:.1f}%. Hard lockout active until 00:00 UTC!"
                )
            self.circuit_breaker_tripped = True

        return self.circuit_breaker_tripped

    def calculate_lot_size(
        self,
        sl_distance_price: float,
        equity: Optional[float] = None
    ) -> float:
        """
        Calculates contract volume for Gold (XAUUSD).
        Formula: Lot = (Equity * Risk%) / (SL_distance * Contract_Size)
        """
        if sl_distance_price <= 0.01 or self.check_circuit_breaker():
            return 0.0

        contract_size = 100.0   # Default 100 oz per standard lot for XAUUSD
        step = 0.01             # Default volume step
        min_vol = self.limits.min_lot
        max_vol = self.limits.max_lot
        account_equity = equity or 0.0

        if MT5_AVAILABLE and not getattr(self.config, "mock_mode", False):
            acc = mt5.account_info()
            sym = mt5.symbol_info(self.symbol)
            if acc:
                account_equity = float(acc.equity) if account_equity <= 0 else account_equity
            if sym:
                contract_size = float(sym.trade_contract_size or 100.0)
                step = float(sym.volume_step or 0.01)
                min_vol = float(sym.volume_min or self.limits.min_lot)
                max_vol = min(float(sym.volume_max or self.limits.max_lot), self.limits.max_lot)

        if account_equity <= 0:
            account_equity = self.starting_daily_equity if self.starting_daily_equity > 0 else 10000.0

        risk_capital = account_equity * self.limits.max_risk_pct_per_trade
        # Dollar risk per 1.0 full lot = sl_distance * contract_size
        raw_lots = risk_capital / (sl_distance_price * contract_size)

        # Snap to broker lot step
        quantized_lots = math.floor(raw_lots / step) * step
        clamped_lots = max(min_vol, min(quantized_lots, max_vol))

        final_lots = round(clamped_lots, 2)
        logger.info(
            f"[LOT SIZING] Equity: ${account_equity:.2f} | Risk%: {self.limits.max_risk_pct_per_trade*100:.2f}% | "
            f"SL Distance: ${sl_distance_price:.2f} | Contract: {contract_size} oz | Lots: {final_lots}"
        )
        return final_lots

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Extended adapter for position sizing calculation.
        Returns: (lot_size, actual_dollar_risk)
        """
        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance <= 0.05:
            sl_distance = 0.80

        if risk_pct is not None:
            old_pct = self.limits.max_risk_pct_per_trade
            self.limits.max_risk_pct_per_trade = risk_pct
            lots = self.calculate_lot_size(sl_distance, equity=account_equity)
            self.limits.max_risk_pct_per_trade = old_pct
        else:
            lots = self.calculate_lot_size(sl_distance, equity=account_equity)

        contract_size = getattr(self.config.instrument, "contract_size", 100.0)
        actual_dollar_risk = lots * sl_distance * contract_size
        return lots, actual_dollar_risk

    def update_daily_account_state(self, current_equity: float, current_balance: float):
        """Monitors intraday drawdown and resets state at 00:00 UTC"""
        today = datetime.now(timezone.utc).date()
        if self.current_trading_day != today:
            self.current_trading_day = today
            self.starting_daily_equity = current_equity
            self.circuit_breaker_tripped = False
            self.realized_pnl_today = 0.0

        if self.starting_daily_equity <= 0:
            self.starting_daily_equity = current_equity

        # Evaluate circuit breaker condition
        drawdown_pct = (self.starting_daily_equity - current_equity) / max(self.starting_daily_equity, 1.0)
        if drawdown_pct >= self.limits.max_daily_drawdown_pct and not self.circuit_breaker_tripped:
            self.circuit_breaker_tripped = True
            logger.critical(
                f"[CIRCUIT BREAKER TRIGGERED] Cumulative Intraday Drawdown {drawdown_pct * 100:.2f}% >= "
                f"Limit {self.limits.max_daily_drawdown_pct * 100:.1f}%. Hard lockout active until 00:00 UTC!"
            )

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Validates all capital and risk constraints before permitting an entry"""
        if self.check_circuit_breaker():
            return False, "CIRCUIT BREAKER ACTIVE: Intraday Loss >= 3.0%. Terminal lockout until 00:00 UTC."

        max_concurrent = getattr(self.config.risk, "max_concurrent_trades", 1)
        if len(self.active_trades) >= max_concurrent:
            return False, f"CONCURRENCY LIMIT: Max open trades ({max_concurrent}) reached."

        return True, "Trading Approved"

    def register_trade(
        self,
        ticket: int,
        symbol: str,
        direction: Union[SignalDirection, int],
        lot_size: float,
        entry_price: float,
        sl: float,
        tp: float,
        m1_atr: float
    ) -> ActiveTrade:
        """Register newly executed trade into the risk supervisor"""
        dir_val = SignalDirection.BUY if (direction == 1 or direction == SignalDirection.BUY) else SignalDirection.SELL
        r_dist = abs(entry_price - sl)
        trade = ActiveTrade(
            ticket=ticket,
            symbol=symbol,
            direction=dir_val,
            lot_size=lot_size,
            entry_price=entry_price,
            initial_sl=sl,
            initial_tp=tp,
            current_sl=sl,
            current_tp=tp,
            r_distance_usd=r_dist,
            m1_atr_at_entry=m1_atr,
            entry_time_utc=datetime.now(timezone.utc)
        )
        self.active_trades[ticket] = trade
        logger.info(f"[RISK CONTROLLER] Registered active trade #{ticket}: {dir_val.name} {lot_size} lots at ${entry_price:.2f}")
        return trade

    def remove_trade(self, ticket: int, realized_pnl: float = 0.0):
        """Unregister closed trade and record realized PnL"""
        if ticket in self.active_trades:
            del self.active_trades[ticket]
            self.realized_pnl_today += realized_pnl
            logger.info(f"[RISK CONTROLLER] Deregistered trade #{ticket}. Realized PnL: ${realized_pnl:+.2f}")

    def evaluate_lifecycle_actions(
        self,
        ticket: int,
        current_bid: float,
        current_ask: float,
        current_m1_atr: float
    ) -> Dict[str, Optional[object]]:
        """
        Supervises trade lifecycle management:
        1. Break-Even: At 1.0R, move SL to Entry + Spread Buffer ($0.10).
        2. Partial Scale Out: At 1.5R, scale out 50% of the position.
        3. Dynamic Trailing Stop: Trail remaining 50% via 1.2x M1 ATR.
        """
        trade = self.active_trades.get(ticket)
        if not trade:
            return {"action": "NONE"}

        be_trigger = getattr(self.config.risk, "breakeven_r_trigger", 1.0)
        partial_trigger = getattr(self.config.risk, "partial_scale_out_r_trigger", 1.5)
        spread_buffer = getattr(self.config.risk, "breakeven_spread_buffer_usd", 0.10)
        trailing_mult = getattr(self.config.risk, "trailing_atr_multiple", 1.2)
        digits = getattr(self.config.instrument, "digits", 2)

        # For BUY trade
        if trade.direction == SignalDirection.BUY:
            current_price = current_bid
            current_r = round((current_price - trade.entry_price) / max(trade.r_distance_usd, 0.01), 4)

            # 1. Break-Even Trigger
            if current_r >= be_trigger and not trade.breakeven_activated:
                be_sl = round(trade.entry_price + spread_buffer, digits)
                if be_sl > trade.current_sl:
                    trade.breakeven_activated = True
                    trade.current_sl = be_sl
                    logger.info(f"[LIFECYCLE] Trade #{ticket} achieved {be_trigger}R. Moving SL to BE+Buffer: ${be_sl:.2f}")
                    return {"action": "MODIFY_SL", "new_sl": be_sl}

            # 2. Partial Scale Out
            if current_r >= partial_trigger and not trade.partial_tp_executed:
                trade.partial_tp_executed = True
                close_volume = round(trade.lot_size * 0.50, 2)
                close_volume = max(self.limits.min_lot, close_volume)
                logger.info(f"[LIFECYCLE] Trade #{ticket} achieved {partial_trigger}R. Executing 50% Partial Close: {close_volume} lots.")
                return {"action": "PARTIAL_CLOSE", "close_volume": close_volume}

            # 3. Dynamic Trailing via M1 ATR for remaining position
            if trade.partial_tp_executed:
                trail_sl = round(current_price - (current_m1_atr * trailing_mult), digits)
                if trail_sl > trade.current_sl:
                    trade.current_sl = trail_sl
                    logger.info(f"[LIFECYCLE] Trailing SL for trade #{ticket} -> ${trail_sl:.2f} ({trailing_mult}x M1 ATR)")
                    return {"action": "TRAIL_SL", "new_sl": trail_sl}

        # For SELL trade
        elif trade.direction == SignalDirection.SELL:
            current_price = current_ask
            current_r = round((trade.entry_price - current_price) / max(trade.r_distance_usd, 0.01), 4)

            # 1. Break-Even Trigger
            if current_r >= be_trigger and not trade.breakeven_activated:
                be_sl = round(trade.entry_price - spread_buffer, digits)
                if be_sl < trade.current_sl:
                    trade.breakeven_activated = True
                    trade.current_sl = be_sl
                    logger.info(f"[LIFECYCLE] Trade #{ticket} achieved {be_trigger}R. Moving SL to BE+Buffer: ${be_sl:.2f}")
                    return {"action": "MODIFY_SL", "new_sl": be_sl}

            # 2. Partial Scale Out
            if current_r >= partial_trigger and not trade.partial_tp_executed:
                trade.partial_tp_executed = True
                close_volume = round(trade.lot_size * 0.50, 2)
                close_volume = max(self.limits.min_lot, close_volume)
                logger.info(f"[LIFECYCLE] Trade #{ticket} achieved {partial_trigger}R. Executing 50% Partial Close: {close_volume} lots.")
                return {"action": "PARTIAL_CLOSE", "close_volume": close_volume}

            # 3. Dynamic Trailing via M1 ATR for remaining position
            if trade.partial_tp_executed:
                trail_sl = round(current_price + (current_m1_atr * trailing_mult), digits)
                if trail_sl < trade.current_sl:
                    trade.current_sl = trail_sl
                    logger.info(f"[LIFECYCLE] Trailing SL for trade #{ticket} -> ${trail_sl:.2f} ({trailing_mult}x M1 ATR)")
                    return {"action": "TRAIL_SL", "new_sl": trail_sl}

        return {"action": "NONE"}
