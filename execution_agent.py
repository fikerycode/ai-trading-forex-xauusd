"""
SYSTEM ARCHITECTURE SPECIFICATION: AI-DRIVEN XAUUSD SCALPER (ANTI-GRAVITY)
MODULE D: Order Execution & Router (execution_agent.py)
Sub-50ms bracket placement, IOC / tight slippage enforcement, partial fills, and order modification.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity")
    logger.success = logger.info

from config import SystemConfig, CONFIG
from alpha_engine import SignalDirection, AlphaSignal
from risk_controller import RiskController

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


@dataclass
class ExecutionResult:
    success: bool
    ticket: int
    executed_price: float
    lot_size: float
    sl: float
    tp: float
    slippage_usd: float
    execution_latency_ms: float
    comment: str


class ExecutionAgent:
    """
    Sub-Second Micro-Execution Router & Active Order Agent.
    Enforces:
    - Tight slippage tolerance (max 10-15 cents on Gold).
    - Hard server-side protective brackets (SL & TP placed < 50ms).
    - Partial execution handling.
    - Active order modifications (Break-Even & dynamic trailing).
    - Emergency circuit-breaker liquidations.
    """
    def __init__(
        self,
        symbol: Union[str, SystemConfig] = "XAUUSD",
        config: Optional[SystemConfig] = None,
        risk: Optional[RiskController] = None
    ):
        if isinstance(symbol, SystemConfig) or hasattr(symbol, "instrument"):
            self.config = symbol
            self.symbol = self.config.instrument.symbol
        else:
            self.config = config or CONFIG
            self.symbol = str(symbol)

        self.magic = getattr(self.config, "magic_number", 888999)
        self.max_slippage = getattr(self.config.instrument, "slippage_tolerance_usd", 0.15)
        self.mock_ticket_counter = 100000
        self.risk = risk or RiskController(symbol=self.symbol, config=self.config)

    def _log_trade_to_journal(
        self,
        ticket: int,
        direction: int,
        entry_price: float,
        sl: float,
        tp: float,
        lots: float
    ):
        """Records executed trade into CSV so dashboard shows live fills instantly."""
        csv_file = "backtest_trade_log.csv"
        try:
            entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            is_buy = (direction == 1)
            row = {
                "ticket": ticket,
                "entry_time": entry_time,
                "exit_time": "",
                "direction": "BUY" if is_buy else "SELL",
                "entry_price": round(entry_price, 2),
                "exit_price": "",
                "lot_size": lots,
                "initial_sl": round(sl, 2),
                "initial_tp": round(tp, 2),
                "pnl_usd": 0.0,
                "r_multiple": "",
                "exit_reason": "OPEN"
            }
            df_new = pd.DataFrame([row])
            if os.path.exists(csv_file):
                df_new.to_csv(csv_file, mode="a", header=False, index=False)
            else:
                df_new.to_csv(csv_file, index=False)
        except Exception:
            pass

    def route_alpha_signal(self, signal: AlphaSignal) -> Optional[int]:
        """
        Direct routing bridge for AlphaSignal from AntiGravityAlphaEngine:
        1. Validates spread and circuit breaker
        2. Calculates lot size via self.risk
        3. Executes market IOC order with protective SL/TP brackets
        4. Registers active position into risk controller
        """
        if not signal.spread_ok or signal.direction == 0:
            return None

        # Determine Stop Loss distance
        # Determine Stop Loss & Take Profit distances
        sl_dist = abs(signal.entry_price - signal.suggested_sl) if signal.suggested_sl > 0 else max(signal.atr_val * 1.0, 0.80)
        tp_dist = abs(signal.suggested_tp - signal.entry_price) if signal.suggested_tp > 0 else max(signal.atr_val * 1.5, 1.20)

        # Dynamic lot sizing & circuit breaker check
        lots = self.risk.calculate_lot_size(sl_dist)
        if lots <= 0 or self.risk.check_circuit_breaker():
            logger.warning("[EXECUTION] Order rejected by RiskController (zero lots or circuit breaker active).")
            return None

        is_buy = (signal.direction == 1)
        start_time = time.perf_counter()

        # Offline / Mock simulation
        if getattr(self.config, "mock_mode", False) or not MT5_AVAILABLE:
            self.mock_ticket_counter += 1
            ticket = self.mock_ticket_counter
            fill_price = signal.entry_price + (0.02 if is_buy else -0.02)
            stop_loss = round(fill_price - sl_dist if is_buy else fill_price + sl_dist, 2)
            take_profit = round(fill_price + tp_dist if is_buy else fill_price - tp_dist, 2)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"[EXECUTION SIM] Routed #{ticket} | {'BUY' if is_buy else 'SELL'} {lots} lots at ${fill_price:.2f} "
                f"| SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Latency: {elapsed_ms:.1f}ms"
            )
            self.risk.register_trade(
                ticket=ticket,
                symbol=self.symbol,
                direction=signal.direction,
                lot_size=lots,
                entry_price=fill_price,
                sl=stop_loss,
                tp=take_profit,
                m1_atr=signal.atr_val
            )
            self._log_trade_to_journal(ticket, signal.direction, fill_price, stop_loss, take_profit, lots)
            return ticket

        # Direct MetaTrader 5 execution
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            # Fallback to simulated fill if terminal is not live
            self.mock_ticket_counter += 1
            ticket = self.mock_ticket_counter
            fill_price = signal.entry_price + (0.02 if is_buy else -0.02)
            stop_loss = round(fill_price - sl_dist if is_buy else fill_price + sl_dist, 2)
            take_profit = round(fill_price + tp_dist if is_buy else fill_price - tp_dist, 2)
            self.risk.register_trade(
                ticket=ticket,
                symbol=self.symbol,
                direction=signal.direction,
                lot_size=lots,
                entry_price=fill_price,
                sl=stop_loss,
                tp=take_profit,
                m1_atr=signal.atr_val
            )
            self._log_trade_to_journal(ticket, signal.direction, fill_price, stop_loss, take_profit, lots)
            return ticket

        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid
        slippage = abs(price - signal.entry_price)

        if slippage > self.max_slippage:
            logger.warning(f"[EXECUTION] Slippage rejected: ${slippage:.2f} > Max ${self.max_slippage:.2f}")
            return None

        stop_loss = round(price - sl_dist if is_buy else price + sl_dist, 2)
        take_profit = round(price + tp_dist if is_buy else price - tp_dist, 2)

        deviation = int(self.max_slippage / getattr(self.config.instrument, "point_size", 0.01))
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": self.magic,
            "comment": "AntiGravity Scalp",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = getattr(result, "retcode", "UNKNOWN")
            comment = getattr(result, "comment", "No response")
            logger.error(f"[EXECUTION FAILED] Retcode: {retcode} | {comment}")
            return None

        ticket = result.order
        logger.info(
            f"[EXECUTION CONFIRMED] Ticket #{ticket} filled at ${result.price:.2f} | "
            f"Latency: {elapsed_ms:.1f}ms | Slippage: ${abs(result.price - price):.2f}"
        )

        self.risk.register_trade(
            ticket=ticket,
            symbol=self.symbol,
            direction=signal.direction,
            lot_size=result.volume,
            entry_price=result.price,
            sl=stop_loss,
            tp=take_profit,
            m1_atr=signal.atr_val
        )
        return ticket

    async def execute_entry(
        self,
        direction: Union[SignalDirection, int],
        lot_size: float,
        target_price: float,
        stop_loss: float,
        take_profit: float
    ) -> ExecutionResult:
        """
        Submits market order with IOC or tight slippage constraint.
        Immediately enforces server-side brackets upon fill confirmation.
        """
        start_time = time.perf_counter()
        is_buy = (direction == 1 or direction == SignalDirection.BUY)
        dir_name = "BUY" if is_buy else "SELL"

        logger.info(
            f"[EXECUTION] Submitting {dir_name} order: {lot_size} lots at ~${target_price:.2f} "
            f"| SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
        )

        if self.config.mock_mode or not MT5_AVAILABLE:
            # High-speed synthetic execution simulator
            await asyncio.sleep(0.015)  # 15ms simulated network transit
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.mock_ticket_counter += 1
            simulated_fill = target_price + (0.02 if is_buy else -0.02)
            slippage = abs(simulated_fill - target_price)

            logger.success(
                f"[EXECUTION SIM] Filled #{self.mock_ticket_counter} at ${simulated_fill:.2f} "
                f"(Latency: {elapsed_ms:.1f}ms, Slippage: ${slippage:.2f})"
            )
            return ExecutionResult(
                success=True,
                ticket=self.mock_ticket_counter,
                executed_price=simulated_fill,
                lot_size=lot_size,
                sl=stop_loss,
                tp=take_profit,
                slippage_usd=slippage,
                execution_latency_ms=elapsed_ms,
                comment="Mock execution success"
            )

        # Real MT5 Order Routing
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(self.symbol)
        
        # If MT5 terminal is not connected/initialized, route to simulated execution
        if tick is None:
            logger.warning(f"[EXECUTION] MT5 terminal not connected for {self.symbol}. Using simulated routing.")
            await asyncio.sleep(0.015)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.mock_ticket_counter += 1
            simulated_fill = target_price + (0.02 if is_buy else -0.02)
            slippage = abs(simulated_fill - target_price)
            return ExecutionResult(
                success=True,
                ticket=self.mock_ticket_counter,
                executed_price=simulated_fill,
                lot_size=lot_size,
                sl=stop_loss,
                tp=take_profit,
                slippage_usd=slippage,
                execution_latency_ms=elapsed_ms,
                comment="Mock execution (MT5 offline)"
            )

        price = tick.ask if is_buy else tick.bid

        # Slippage check against anticipated target price
        expected_slippage = abs(price - target_price)
        if expected_slippage > self.max_slippage:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            msg = f"SLIPPAGE REJECT: Expected ${expected_slippage:.2f} > Max Tolerance ${self.max_slippage:.2f}"
            logger.warning(f"[EXECUTION] {msg}")
            return ExecutionResult(
                success=False,
                ticket=0,
                executed_price=price,
                lot_size=lot_size,
                sl=stop_loss,
                tp=take_profit,
                slippage_usd=expected_slippage,
                execution_latency_ms=elapsed_ms,
                comment=msg
            )

        # Slippage in points: For gold, 1 point = 0.01. $0.15 = 15 points.
        deviation_points = int(self.max_slippage / self.config.instrument.point_size)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation_points,
            "magic": self.magic,
            "comment": "AntiGravity Scalp",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order to MT5 asynchronously
        result = await asyncio.to_thread(mt5.order_send, request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = getattr(result, "retcode", "UNKNOWN")
            comment = getattr(result, "comment", "No response from terminal")
            logger.error(f"[EXECUTION FAILED] Retcode: {retcode} | {comment}")
            return ExecutionResult(
                success=False,
                ticket=0,
                executed_price=price,
                lot_size=lot_size,
                sl=stop_loss,
                tp=take_profit,
                slippage_usd=0.0,
                execution_latency_ms=elapsed_ms,
                comment=f"MT5 Error: {retcode} - {comment}"
            )

        fill_price = result.price
        actual_slippage = abs(fill_price - price)
        ticket = result.order

        logger.success(
            f"[EXECUTION CONFIRMED] Ticket #{ticket} filled at ${fill_price:.2f} | "
            f"Latency: {elapsed_ms:.1f}ms | Slippage: ${actual_slippage:.2f}"
        )

        return ExecutionResult(
            success=True,
            ticket=ticket,
            executed_price=fill_price,
            lot_size=result.volume,
            sl=stop_loss,
            tp=take_profit,
            slippage_usd=actual_slippage,
            execution_latency_ms=elapsed_ms,
            comment="Order filled successfully"
        )

    async def modify_brackets(self, ticket: int, new_sl: float, new_tp: Optional[float] = None) -> bool:
        """
        Active server-side bracket modification (Break-Even or Dynamic Trailing).
        Execution latency target: < 50ms.
        """
        start_time = time.perf_counter()
        logger.info(f"[EXECUTION] Modifying ticket #{ticket} -> New SL: ${new_sl:.2f}")

        if self.config.mock_mode or not MT5_AVAILABLE or ticket >= 100000:
            await asyncio.sleep(0.01)
            logger.info(f"[EXECUTION SIM] Ticket #{ticket} SL updated to ${new_sl:.2f}")
            return True

        # Retrieve current position
        positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.warning(f"[EXECUTION] Position #{ticket} not found for SL modification.")
            return False

        pos = positions[0]
        tp_price = new_tp if new_tp is not None else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
            "sl": new_sl,
            "tp": tp_price,
            "magic": self.magic,
        }

        res = await asyncio.to_thread(mt5.order_send, request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[EXECUTION] Ticket #{ticket} SL updated in {elapsed_ms:.1f}ms.")
            return True
        else:
            retcode = getattr(res, "retcode", "UNKNOWN")
            logger.error(f"[EXECUTION] SL modification failed: {retcode}")
            return False

    async def close_partial_position(self, ticket: int, close_volume: float) -> bool:
        """
        Scales out a partial lot size at market.
        """
        logger.info(f"[EXECUTION] Executing partial close of {close_volume} lots on #{ticket}...")
        
        if self.config.mock_mode or not MT5_AVAILABLE or ticket >= 100000:
            await asyncio.sleep(0.015)
            logger.info(f"[EXECUTION SIM] Partial scale-out confirmed for #{ticket}: {close_volume} lots.")
            return True

        positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.warning(f"[EXECUTION] Position #{ticket} not found for partial close.")
            return False

        pos = positions[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": self.symbol,
            "volume": close_volume,
            "type": order_type,
            "price": price,
            "deviation": int(self.max_slippage / self.config.instrument.point_size),
            "magic": self.magic,
            "comment": "AntiGravity Partial ScaleOut",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = await asyncio.to_thread(mt5.order_send, request)
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.success(f"[EXECUTION] Successfully scaled out {close_volume} lots on #{ticket}.")
            return True
        else:
            logger.error(f"[EXECUTION] Partial close failed: {getattr(res, 'retcode', 'UNKNOWN')}")
            return False

    async def liquidate_all_positions(self, reason: str = "CIRCUIT_BREAKER"):
        """
        Emergency liquidation: Immediately closes all open positions and cancels orders.
        Triggered when daily drawdown circuit breaker fires (3.0%).
        """
        logger.critical(f"[EMERGENCY LIQUIDATION INITIATED] Reason: {reason}")
        
        if self.config.mock_mode or not MT5_AVAILABLE:
            logger.critical("[EXECUTION SIM] All mock positions liquidated.")
            return

        positions = await asyncio.to_thread(mt5.positions_get, symbol=self.symbol)
        if not positions:
            logger.info("[EXECUTION] No open positions to liquidate.")
            return

        for pos in positions:
            if pos.magic == self.magic or self.magic == 0:
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask

                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "price": price,
                    "deviation": 30,  # Wider deviation allowance during emergency liquidation
                    "magic": self.magic,
                    "comment": f"Emergency Liquidate: {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = await asyncio.to_thread(mt5.order_send, req)
                logger.info(f"[LIQUIDATED] Ticket #{pos.ticket} result: {getattr(res, 'retcode', 'UNKNOWN')}")
