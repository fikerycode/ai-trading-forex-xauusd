"""
position_manager.py
Active Open Position Tracking, Break-Even Gating (1.0R), and Dynamic ATR Trailing Stop.

Key Capabilities:
1. Reconciles and tracks live MT5 positions matching symbol and magic number.
2. Moves Stop-Loss to Break-Even + Spread Buffer ($0.10) once price reaches 1.0R.
3. Scales out 50% partial Take-Profit at 1.5R.
4. Dynamically trails the remaining volume behind price via 1.2x M1 ATR.
5. Sends sub-50ms TRADE_ACTION_SLTP modification requests to the broker.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Dict, List, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity.pos_mgr")
    logger.success = logger.info

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import SystemConfig, CONFIG
from alpha_engine import SignalDirection


@dataclass
class PositionMetadata:
    """Institutional tracking metadata for an active position."""
    ticket: int
    symbol: str
    direction: int                # 1: Long, -1: Short
    volume: float
    entry_price: float
    initial_sl: float
    initial_tp: float
    current_sl: float
    r_distance_usd: float         # 1.0R price move in dollars
    entry_atr: float
    open_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    be_triggered: bool = False
    partial_tp_executed: bool = False
    remaining_volume: float = 0.0

    def __post_init__(self):
        if self.remaining_volume <= 0:
            self.remaining_volume = self.volume


class PositionManager:
    """
    Sub-Second Position Lifecycle Supervisor for XAUUSD.
    - Monitors real-time Bid/Ask price vs Entry Price.
    - At +1.0R: Moves SL to Entry + Spread Buffer.
    - At +1.5R: Partially closes 50% position.
    - Trailing: Trails remaining 50% via 1.2x M1 ATR.
    """
    def __init__(
        self,
        symbol: str = "XAUUSD",
        magic_number: int = 888999,
        breakeven_r_trigger: Optional[float] = None,
        breakeven_spread_buffer_usd: Optional[float] = None,
        partial_scale_out_r_trigger: Optional[float] = None,
        partial_scale_out_ratio: Optional[float] = None,
        trailing_atr_multiple: Optional[float] = None,
        min_lot: Optional[float] = None,
        config: Optional[SystemConfig] = None
    ):
        self.config = config or CONFIG
        self.symbol = symbol
        self.magic = magic_number
        self.be_r_trigger = breakeven_r_trigger if breakeven_r_trigger is not None else getattr(self.config.risk, "breakeven_r_trigger", 1.0)
        self.be_spread_buffer = breakeven_spread_buffer_usd if breakeven_spread_buffer_usd is not None else getattr(self.config.risk, "breakeven_spread_buffer_usd", 0.10)
        self.partial_r_trigger = partial_scale_out_r_trigger if partial_scale_out_r_trigger is not None else getattr(self.config.risk, "partial_scale_out_r_trigger", 1.5)
        self.partial_ratio = partial_scale_out_ratio if partial_scale_out_ratio is not None else getattr(self.config.risk, "partial_scale_out_ratio", 0.50)
        self.trailing_atr_mult = trailing_atr_multiple if trailing_atr_multiple is not None else getattr(self.config.risk, "trailing_atr_multiple", 1.2)
        self.min_lot = min_lot if min_lot is not None else getattr(self.config.instrument, "min_lot", 0.01)

        # State storage: {ticket: PositionMetadata}
        self.positions: Dict[int, PositionMetadata] = {}

    def track_position(
        self,
        ticket: int,
        direction: Union[SignalDirection, int],
        volume: float,
        entry_price: float,
        sl: float,
        tp: float,
        atr_val: float
    ) -> PositionMetadata:
        """Registers a newly executed fill into active tracking."""
        dir_val = 1 if (direction == 1 or direction == SignalDirection.BUY) else -1
        r_distance = abs(entry_price - sl)
        if r_distance <= 0.05:
            r_distance = max(atr_val * 1.0, 0.80)  # Default minimum $0.80 R-risk

        meta = PositionMetadata(
            ticket=ticket,
            symbol=self.symbol,
            direction=dir_val,
            volume=round(volume, 2),
            entry_price=round(entry_price, 2),
            initial_sl=round(sl, 2),
            initial_tp=round(tp, 2),
            current_sl=round(sl, 2),
            r_distance_usd=round(r_distance, 2),
            entry_atr=round(atr_val, 2),
            remaining_volume=round(volume, 2)
        )
        self.positions[ticket] = meta
        logger.info(
            f"[POSITION MGR] Tracking ticket #{ticket} | "
            f"{'BUY' if dir_val == 1 else 'SELL'} {volume:.2f} lots at ${entry_price:.2f} "
            f"| 1.0R = ${r_distance:.2f} | Initial SL: ${sl:.2f}"
        )
        return meta

    def untrack_position(self, ticket: int, reason: str = "CLOSED"):
        """Removes a closed position from internal tracking."""
        if ticket in self.positions:
            pos = self.positions.pop(ticket)
            logger.info(f"[POSITION MGR] Untracked ticket #{ticket} ({reason}). Remaining active: {len(self.positions)}")

    def sync_from_terminal(self) -> int:
        """
        Synchronizes open positions directly from MetaTrader 5.
        Adopts existing terminal positions matching symbol and magic number.
        """
        if not MT5_AVAILABLE or getattr(self.config, "mock_mode", False):
            return len(self.positions)

        active_positions = mt5.positions_get(symbol=self.symbol)
        if active_positions is None:
            return 0

        active_tickets = set()
        for pos in active_positions:
            if pos.magic == self.magic or self.magic == 0:
                ticket = pos.ticket
                active_tickets.add(ticket)

                if ticket not in self.positions:
                    dir_val = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                    sl = pos.sl if pos.sl > 0 else (pos.price_open - 1.50 if dir_val == 1 else pos.price_open + 1.50)
                    r_dist = abs(pos.price_open - sl)

                    self.positions[ticket] = PositionMetadata(
                        ticket=ticket,
                        symbol=self.symbol,
                        direction=dir_val,
                        volume=pos.volume,
                        entry_price=pos.price_open,
                        initial_sl=sl,
                        initial_tp=pos.tp,
                        current_sl=pos.sl,
                        r_distance_usd=r_dist,
                        entry_atr=0.80,
                        remaining_volume=pos.volume
                    )
                    logger.info(f"[POSITION MGR] Adopted external MT5 position #{ticket} ({pos.volume} lots)")

        # Untrack any positions closed on terminal side
        closed_tickets = [t for t in self.positions.keys() if t not in active_tickets and t < 100000]
        for t in closed_tickets:
            self.untrack_position(t, reason="Closed on broker terminal")

        return len(self.positions)

    def modify_sl(self, ticket: int, new_sl: float, new_tp: Optional[float] = None) -> bool:
        """
        Transmits MT5 TRADE_ACTION_SLTP request with sub-50ms execution latency.
        """
        start_time = time.perf_counter()
        pos_meta = self.positions.get(ticket)
        new_sl = round(new_sl, 2)

        is_live_mt5 = MT5_AVAILABLE and not getattr(self.config, "mock_mode", False) and mt5.terminal_info() is not None

        # Mock / Dry-run / Offline terminal handling
        if not is_live_mt5 or ticket >= 100000:
            if pos_meta:
                pos_meta.current_sl = new_sl
            logger.info(f"[MOCK SL MODIFY] Ticket #{ticket} SL updated -> ${new_sl:.2f}")
            return True

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"[MODIFY REJECTED] Position #{ticket} not found on broker server.")
            self.untrack_position(ticket, reason="Position not found on server")
            return False

        pos = positions[0]
        tp_val = new_tp if new_tp is not None else pos.tp

        # Safety check: avoid redundant modifications of identical prices
        if abs(pos.sl - new_sl) < 0.01:
            return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
            "sl": new_sl,
            "tp": tp_val,
            "magic": self.magic
        }

        res = mt5.order_send(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            if pos_meta:
                pos_meta.current_sl = new_sl
            logger.info(f"[SL UPDATED] Ticket #{ticket} SL moved to ${new_sl:.2f} ({elapsed_ms:.1f}ms latency)")
            return True
        else:
            retcode = getattr(res, "retcode", "UNKNOWN")
            comment = getattr(res, "comment", "No response")
            logger.error(f"[SL MODIFY FAILED] Ticket #{ticket}: {retcode} | {comment}")
            return False

    def close_partial(self, ticket: int, volume_to_close: float) -> bool:
        """
        Executes a partial close of an active position using an opposite deal.
        """
        start_time = time.perf_counter()
        volume_to_close = round(volume_to_close, 2)
        pos_meta = self.positions.get(ticket)

        is_live_mt5 = MT5_AVAILABLE and not getattr(self.config, "mock_mode", False) and mt5.terminal_info() is not None

        # Mock / Dry-run / Offline terminal handling
        if not is_live_mt5 or ticket >= 100000:
            if pos_meta:
                pos_meta.remaining_volume = max(0.0, pos_meta.remaining_volume - volume_to_close)
                pos_meta.partial_tp_executed = True
            logger.info(f"[MOCK PARTIAL CLOSE] Closed {volume_to_close:.2f} lots on #{ticket}. Remaining: {pos_meta.remaining_volume:.2f}")
            return True

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False

        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": self.symbol,
            "volume": volume_to_close,
            "type": order_type,
            "price": price,
            "deviation": 15,
            "magic": self.magic,
            "comment": "AntiGravity Partial Scale-Out",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        res = mt5.order_send(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            if pos_meta:
                pos_meta.remaining_volume = max(0.0, pos_meta.remaining_volume - volume_to_close)
                pos_meta.partial_tp_executed = True
            logger.success(f"[PARTIAL TP FILLED] Closed {volume_to_close:.2f} lots on #{ticket} in {elapsed_ms:.1f}ms")
            return True
        else:
            logger.error(f"[PARTIAL CLOSE FAILED] Ticket #{ticket}: {getattr(res, 'retcode', 'UNKNOWN')}")
            return False

    def update(
        self,
        current_bid: float,
        current_ask: float,
        current_m1_atr: float
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Master tick/candle evaluation loop:
        1. Break-Even Trigger: Moves SL to Entry + $0.10 buffer once trade touches 1.0R.
        2. Partial Scale Out: Closes 50% lot volume once trade touches 1.5R.
        3. Dynamic ATR Trailing: Trails remaining 50% volume via 1.2x M1 ATR.
        
        Returns: List of executed lifecycle action summaries.
        """
        actions_taken = []
        # Copy items to allow modification during iteration
        for ticket, trade in list(self.positions.items()):
            # -------------------------------------------------------------
            # LONG POSITION EVALUATION
            # -------------------------------------------------------------
            if trade.direction == 1:
                profit_distance = current_bid - trade.entry_price
                current_r = round(profit_distance / max(trade.r_distance_usd, 0.01), 4)

                # 1. Break-Even Trigger at +1.0R
                if current_r >= self.be_r_trigger and not trade.be_triggered:
                    be_sl = round(trade.entry_price + self.be_spread_buffer, 2)
                    if be_sl > trade.current_sl:
                        if self.modify_sl(ticket, new_sl=be_sl):
                            trade.be_triggered = True
                            trade.current_sl = be_sl
                            logger.info(
                                f"[BREAK-EVEN LOCKED] Long #{ticket} achieved +{current_r:.2f}R. "
                                f"SL moved to ${be_sl:.2f} (Entry + ${self.be_spread_buffer:.2f})"
                            )
                            actions_taken.append({"ticket": ticket, "action": "BREAK_EVEN", "new_sl": be_sl})

                # 2. Partial Scale-Out at +1.5R
                if current_r >= self.partial_r_trigger and not trade.partial_tp_executed:
                    close_vol = round(trade.volume * self.partial_ratio, 2)
                    close_vol = max(self.min_lot, close_vol)
                    if self.close_partial(ticket, volume_to_close=close_vol):
                        trade.partial_tp_executed = True
                        logger.info(
                            f"[PARTIAL 50% TP] Long #{ticket} achieved +{current_r:.2f}R. "
                            f"Scaled out {close_vol:.2f} lots at ${current_bid:.2f}"
                        )
                        actions_taken.append({"ticket": ticket, "action": "PARTIAL_TP", "volume": close_vol})

                # 3. Dynamic ATR Trailing Stop (active after partial TP)
                if trade.partial_tp_executed:
                    trail_sl = round(current_bid - (current_m1_atr * self.trailing_atr_mult), 2)
                    if trail_sl > trade.current_sl:
                        if self.modify_sl(ticket, new_sl=trail_sl):
                            trade.current_sl = trail_sl
                            logger.info(
                                f"[TRAILING SL ADVANCED] Long #{ticket} -> New SL: ${trail_sl:.2f} "
                                f"(Current Bid: ${current_bid:.2f}, ATR: ${current_m1_atr:.2f})"
                            )
                            actions_taken.append({"ticket": ticket, "action": "TRAIL_SL", "new_sl": trail_sl})

            # -------------------------------------------------------------
            # SHORT POSITION EVALUATION
            # -------------------------------------------------------------
            elif trade.direction == -1:
                profit_distance = trade.entry_price - current_ask
                current_r = round(profit_distance / max(trade.r_distance_usd, 0.01), 4)

                # 1. Break-Even Trigger at +1.0R
                if current_r >= self.be_r_trigger and not trade.be_triggered:
                    be_sl = round(trade.entry_price - self.be_spread_buffer, 2)
                    if be_sl < trade.current_sl:
                        if self.modify_sl(ticket, new_sl=be_sl):
                            trade.be_triggered = True
                            trade.current_sl = be_sl
                            logger.info(
                                f"[BREAK-EVEN LOCKED] Short #{ticket} achieved +{current_r:.2f}R. "
                                f"SL moved to ${be_sl:.2f} (Entry - ${self.be_spread_buffer:.2f})"
                            )
                            actions_taken.append({"ticket": ticket, "action": "BREAK_EVEN", "new_sl": be_sl})

                # 2. Partial Scale-Out at +1.5R
                if current_r >= self.partial_r_trigger and not trade.partial_tp_executed:
                    close_vol = round(trade.volume * self.partial_ratio, 2)
                    close_vol = max(self.min_lot, close_vol)
                    if self.close_partial(ticket, volume_to_close=close_vol):
                        trade.partial_tp_executed = True
                        logger.info(
                            f"[PARTIAL 50% TP] Short #{ticket} achieved +{current_r:.2f}R. "
                            f"Scaled out {close_vol:.2f} lots at ${current_ask:.2f}"
                        )
                        actions_taken.append({"ticket": ticket, "action": "PARTIAL_TP", "volume": close_vol})

                # 3. Dynamic ATR Trailing Stop (active after partial TP)
                if trade.partial_tp_executed:
                    trail_sl = round(current_ask + (current_m1_atr * self.trailing_atr_mult), 2)
                    if trail_sl < trade.current_sl:
                        if self.modify_sl(ticket, new_sl=trail_sl):
                            trade.current_sl = trail_sl
                            logger.info(
                                f"[TRAILING SL ADVANCED] Short #{ticket} -> New SL: ${trail_sl:.2f} "
                                f"(Current Ask: ${current_ask:.2f}, ATR: ${current_m1_atr:.2f})"
                            )
                            actions_taken.append({"ticket": ticket, "action": "TRAIL_SL", "new_sl": trail_sl})

        return actions_taken

    def close_all(self, reason: str = "EMERGENCY_EXIT"):
        """Liquidates all open tracked positions immediately."""
        logger.critical(f"[POSITION MGR] Liquidating all open positions: {reason}")
        for ticket, meta in list(self.positions.items()):
            self.close_partial(ticket, volume_to_close=meta.remaining_volume)
        self.positions.clear()
