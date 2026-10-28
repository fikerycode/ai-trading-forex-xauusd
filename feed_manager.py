"""
SYSTEM ARCHITECTURE SPECIFICATION: AI-DRIVEN XAUUSD SCALPER (ANTI-GRAVITY)
MODULE A: Data Ingestion & Microstructure Engine (feed_manager.py)
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import random
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity")
    logger.success = logger.info
import numpy as np
import pandas as pd

from config import SystemConfig, CONFIG

# Try importing MetaTrader5; provide graceful fallback for Linux/mock environments
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available or not running on Windows. Mock/ZMQ mode active.")


@dataclass
class TickData:
    """Normalized tick structure with microstructure metadata"""
    symbol: str
    timestamp_utc: datetime
    bid: float
    ask: float
    spread: float
    last: float
    volume: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    delta: float = 0.0         # Net buyer minus seller volume
    flags: int = 0


@dataclass
class M1Bar:
    """Aggregated M1 Bar with Volume Delta & Microstructure Metrics"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    delta: float
    cvd: float               # Cumulative Volume Delta snapshot
    tick_count: int
    median_spread: float
    is_closed: bool = False


@dataclass
class EconomicEvent:
    """Macroeconomic calendar event definition"""
    title: str
    currency: str
    impact: str              # HIGH, MEDIUM, LOW
    time_utc: datetime


class NewsSentinel:
    """
    Monitors economic calendar and enforces lockout intervals:
    - 15 minutes before High-Impact US news (NFP, CPI, PPI, FOMC).
    - 20 minutes after High-Impact US news.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.lockout_before = timedelta(minutes=config.spread_sentinel.news_lockout_before_min)
        self.lockout_after = timedelta(minutes=config.spread_sentinel.news_lockout_after_min)
        self.monitored_keywords = [kw.upper() for kw in config.spread_sentinel.monitored_news_keywords]
        self._scheduled_events: List[EconomicEvent] = []
        self._init_default_calendar()

    def _init_default_calendar(self):
        """Populate initial scheduled events from Forex Factory cache if present"""
        try:
            import json, os
            cache_file = "forex_factory_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    data = json.load(f)
                for item in data.get("events", []):
                    dt = datetime.fromisoformat(item["time_utc"])
                    self._scheduled_events.append(
                        EconomicEvent(
                            title=item["title"],
                            currency=item["country"],
                            impact=item["impact"].upper(),
                            time_utc=dt
                        )
                    )
        except Exception:
            pass

    def register_event(self, event: EconomicEvent):
        """Add an economic release to sentinel tracking"""
        self._scheduled_events.append(event)
        logger.info(f"[NEWS SENTINEL] Registered event: {event.title} at {event.time_utc} UTC (Impact: {event.impact})")

    def is_news_locked(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is prohibited due to impending or recent high-impact release.
        Returns: (is_locked: bool, reason: Optional[str])
        """
        now = current_time or datetime.now(timezone.utc)
        
        for event in self._scheduled_events:
            if event.currency.upper() != "USD" or event.impact.upper() != "HIGH":
                continue

            # Check if event matches high-impact keywords
            is_monitored = any(kw in event.title.upper() for kw in self.monitored_keywords)
            if not is_monitored:
                continue

            lock_start = event.time_utc - self.lockout_before
            lock_end = event.time_utc + self.lockout_after

            if lock_start <= now <= lock_end:
                mins_to_event = (event.time_utc - now).total_seconds() / 60.0
                return True, f"NEWS LOCKOUT: '{event.title}' scheduled at {event.time_utc.strftime('%H:%M:%S UTC')} (Delta: {mins_to_event:+.1f}m)"

        return False, None


class SpreadSentinel:
    """
    Real-time Volatility & Spread Sentinel:
    - Rejects orders if Current Spread > Rolling Median * 1.5
    - Rejects orders if Current Spread > 35 cents on XAUUSD
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.window_size = config.spread_sentinel.rolling_median_window
        self.max_abs_spread = config.spread_sentinel.max_absolute_spread_usd
        self.multiplier = config.spread_sentinel.rolling_median_multiplier
        self._spread_history = deque(maxlen=self.window_size)

    def record_spread(self, spread: float):
        """Record live spread sample"""
        self._spread_history.append(spread)

    @property
    def rolling_median(self) -> float:
        if not self._spread_history:
            return 0.15  # Default tight gold baseline spread (15 cents)
        return float(np.median(self._spread_history))

    def evaluate(self, current_spread: float) -> Tuple[bool, str]:
        """
        Evaluate spread gate conditions.
        Returns: (passes_gate: bool, diagnostic_message: str)
        """
        self.record_spread(current_spread)
        median = self.rolling_median
        dynamic_threshold = median * self.multiplier

        # Condition 1: Hard ceiling (35 cents)
        if current_spread > self.max_abs_spread:
            return False, f"HARD SPREAD GATE EXCEEDED: Spread ${current_spread:.2f} > Max ${self.max_abs_spread:.2f}"

        # Condition 2: Dynamic expansion (> 1.5 * rolling median)
        if len(self._spread_history) >= 10 and current_spread > dynamic_threshold:
            return False, f"DYNAMIC SPREAD SPIKE: Spread ${current_spread:.2f} > 1.5x Median (${dynamic_threshold:.2f}, Median=${median:.2f})"

        return True, f"Spread Normal: ${current_spread:.2f} (Median: ${median:.2f})"


class FeedManager:
    """
    High-Throughput Async Ingestion & Microstructure Engine.
    Handles:
    - MT5 terminal async connection & heartbeat monitoring.
    - Raw Bid/Ask tick ingestion and delta classification (Lee-Ready algorithm).
    - Dynamic M1 bar aggregation with Cumulative Volume Delta (CVD).
    - Spread and News sentinels.
    - Fallback synthetic generator for development and headless backtesting.
    """
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or CONFIG
        self.symbol = self.config.instrument.symbol
        self.news_sentinel = NewsSentinel(self.config)
        self.spread_sentinel = SpreadSentinel(self.config)

        # Microstructure tracking
        self.last_tick: Optional[TickData] = None
        self._cumulative_volume_delta: float = 0.0
        self._current_m1_bar: Optional[M1Bar] = None
        self._current_bar_minute: Optional[int] = None
        self._m1_bar_history: deque[M1Bar] = deque(maxlen=500)
        self._spread_samples_current_bar: List[float] = []

        # Connection state
        self.is_connected: bool = False
        self._running: bool = False
        self._tick_subscribers: List[Callable[[TickData], None]] = []
        self._bar_subscribers: List[Callable[[M1Bar], None]] = []

    def register_tick_listener(self, callback: Callable[[TickData], None]):
        self._tick_subscribers.append(callback)

    def register_bar_listener(self, callback: Callable[[M1Bar], None]):
        self._bar_subscribers.append(callback)

    async def initialize(self) -> bool:
        """Initialize MetaTrader 5 or configure mock data stream"""
        logger.info(f"[FEED MANAGER] Initializing for {self.symbol}...")
        
        if self.config.mock_mode or not MT5_AVAILABLE:
            logger.warning("[FEED MANAGER] Running in SYNTHETIC/MOCK mode.")
            self.is_connected = True
            return True

        # Initialize MT5 terminal
        init_kwargs = {}
        if self.config.mt5.path:
            init_kwargs["path"] = self.config.mt5.path
        if self.config.mt5.portable:
            init_kwargs["portable"] = True

        success = await asyncio.to_thread(mt5.initialize, **init_kwargs)
        if not success:
            err = mt5.last_error()
            logger.error(f"[FEED MANAGER] MT5 Initialize failed: {err}")
            if self.config.mt5.demo_fallback:
                logger.warning("[FEED MANAGER] Falling back to synthetic stream for safety.")
                self.is_connected = True
                return True
            return False

        # Login if credentials provided
        if self.config.mt5.login and self.config.mt5.password:
            login_success = await asyncio.to_thread(
                mt5.login,
                self.config.mt5.login,
                password=self.config.mt5.password,
                server=self.config.mt5.server or ""
            )
            if not login_success:
                logger.error(f"[FEED MANAGER] MT5 Login failed: {mt5.last_error()}")
                return False
            logger.info(f"[FEED MANAGER] Logged into MT5 Account #{self.config.mt5.login}")

        # Ensure symbol is selected
        selected = await asyncio.to_thread(mt5.symbol_select, self.symbol, True)
        if not selected:
            logger.error(f"[FEED MANAGER] Failed to select symbol {self.symbol} in MT5")
            return False

        self.is_connected = True
        logger.info(f"[FEED MANAGER] Successfully connected to MT5 Terminal for {self.symbol}")
        
        # Pre-seed history
        await self._preload_history()
        return True

    async def _preload_history(self):
        """Fetch initial M1 and H1 bars for baseline calculation"""
        if not MT5_AVAILABLE or self.config.mock_mode:
            self._generate_synthetic_history(bars_count=120)
            return

        try:
            rates = await asyncio.to_thread(
                mt5.copy_rates_from_pos,
                self.symbol,
                mt5.TIMEFRAME_M1,
                0,
                150
            )
            if rates is not None and len(rates) > 0:
                for r in rates:
                    bar_dt = datetime.fromtimestamp(r['time'], tz=timezone.utc)
                    spread_val = (r['spread'] * self.config.instrument.point_size) if 'spread' in rates.dtype.names else 0.15
                    m1 = M1Bar(
                        timestamp=bar_dt,
                        open=float(r['open']),
                        high=float(r['high']),
                        low=float(r['low']),
                        close=float(r['close']),
                        volume=float(r['tick_volume']),
                        buy_volume=float(r['tick_volume']) * 0.5,
                        sell_volume=float(r['tick_volume']) * 0.5,
                        delta=0.0,
                        cvd=self._cumulative_volume_delta,
                        tick_count=int(r['tick_volume']),
                        median_spread=spread_val,
                        is_closed=True
                    )
                    self._m1_bar_history.append(m1)
                logger.info(f"[FEED MANAGER] Seeded {len(rates)} historical M1 candles from MT5.")
        except Exception as e:
            logger.error(f"[FEED MANAGER] Failed preloading historical rates: {e}")

    def _generate_synthetic_history(self, bars_count: int = 120):
        """Generate realistic historical gold candles for offline simulation"""
        base_price = 2650.00
        now = datetime.now(timezone.utc)
        current_time = now - timedelta(minutes=bars_count)

        for _ in range(bars_count):
            open_p = base_price
            drift = np.random.normal(0, 0.40)
            close_p = open_p + drift
            high_p = max(open_p, close_p) + abs(np.random.normal(0.2, 0.25))
            low_p = min(open_p, close_p) - abs(np.random.normal(0.2, 0.25))
            vol = random.randint(80, 450)
            buy_vol = vol * random.uniform(0.35, 0.65)
            sell_vol = vol - buy_vol
            delta = buy_vol - sell_vol
            self._cumulative_volume_delta += delta

            bar = M1Bar(
                timestamp=current_time,
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=vol,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                delta=delta,
                cvd=self._cumulative_volume_delta,
                tick_count=vol,
                median_spread=0.15,
                is_closed=True
            )
            self._m1_bar_history.append(bar)
            base_price = close_p
            current_time += timedelta(minutes=1)

    def process_raw_tick(self, bid: float, ask: float, volume: float = 1.0, flags: int = 0) -> TickData:
        """
        Microstructure normalization & Tick-Level Volume Delta Classification.
        Applies Lee-Ready tick algorithm:
        - If price >= previous Ask or ticks upwards: Buyer-initiated (Ask Hit).
        - If price <= previous Bid or ticks downwards: Seller-initiated (Bid Hit).
        """
        now = datetime.now(timezone.utc)
        spread = round(ask - bid, self.config.instrument.digits)
        last_price = (bid + ask) / 2.0

        bid_vol = 0.0
        ask_vol = 0.0
        delta = 0.0

        if self.last_tick is not None:
            if last_price > self.last_tick.last:
                # Uptick -> Buyer aggression
                ask_vol = volume
                delta = volume
            elif last_price < self.last_tick.last:
                # Downtick -> Seller aggression
                bid_vol = volume
                delta = -volume
            else:
                # Flat price: Classify via proximity to current quotes
                dist_to_ask = abs(last_price - self.last_tick.ask)
                dist_to_bid = abs(last_price - self.last_tick.bid)
                if dist_to_ask < dist_to_bid:
                    ask_vol = volume
                    delta = volume
                else:
                    bid_vol = volume
                    delta = -volume
        else:
            delta = 0.0

        self._cumulative_volume_delta += delta

        tick = TickData(
            symbol=self.symbol,
            timestamp_utc=now,
            bid=round(bid, self.config.instrument.digits),
            ask=round(ask, self.config.instrument.digits),
            spread=spread,
            last=round(last_price, self.config.instrument.digits),
            volume=volume,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            delta=delta,
            flags=flags
        )

        self.last_tick = tick
        self.spread_sentinel.record_spread(spread)
        self._aggregate_into_m1(tick)

        # Notify subscribers
        for sub in self._tick_subscribers:
            try:
                sub(tick)
            except Exception as e:
                logger.error(f"[FEED MANAGER] Error in tick subscriber: {e}")

        return tick

    def _aggregate_into_m1(self, tick: TickData):
        """Aggregate real-time ticks into M1 volume-delta candle"""
        minute_key = tick.timestamp_utc.minute

        if self._current_bar_minute != minute_key:
            # Finalize previous bar
            if self._current_m1_bar is not None:
                self._current_m1_bar.is_closed = True
                if self._spread_samples_current_bar:
                    self._current_m1_bar.median_spread = float(np.median(self._spread_samples_current_bar))
                self._m1_bar_history.append(self._current_m1_bar)

                # Dispatch bar closed event
                for sub in self._bar_subscribers:
                    try:
                        sub(self._current_m1_bar)
                    except Exception as e:
                        logger.error(f"[FEED MANAGER] Error in bar subscriber: {e}")

            # Start new bar
            bar_start_dt = tick.timestamp_utc.replace(second=0, microsecond=0)
            self._current_m1_bar = M1Bar(
                timestamp=bar_start_dt,
                open=tick.last,
                high=tick.last,
                low=tick.last,
                close=tick.last,
                volume=tick.volume,
                buy_volume=tick.ask_volume,
                sell_volume=tick.bid_volume,
                delta=tick.delta,
                cvd=self._cumulative_volume_delta,
                tick_count=1,
                median_spread=tick.spread,
                is_closed=False
            )
            self._current_bar_minute = minute_key
            self._spread_samples_current_bar = [tick.spread]
        else:
            # Update active bar
            b = self._current_m1_bar
            b.high = max(b.high, tick.last)
            b.low = min(b.low, tick.last)
            b.close = tick.last
            b.volume += tick.volume
            b.buy_volume += tick.ask_volume
            b.sell_volume += tick.bid_volume
            b.delta += tick.delta
            b.cvd = self._cumulative_volume_delta
            b.tick_count += 1
            self._spread_samples_current_bar.append(tick.spread)

    def get_m1_dataframe(self) -> pd.DataFrame:
        """Export current M1 history as pandas DataFrame"""
        all_bars = list(self._m1_bar_history)
        if self._current_m1_bar:
            all_bars.append(self._current_m1_bar)

        if not all_bars:
            return pd.DataFrame()

        data = [{
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "buy_volume": b.buy_volume,
            "sell_volume": b.sell_volume,
            "delta": b.delta,
            "cvd": b.cvd,
            "tick_count": b.tick_count,
            "spread": b.median_spread
        } for b in all_bars]
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    async def start_streaming(self):
        """Asynchronous tick ingestion loop"""
        self._running = True
        logger.info("[FEED MANAGER] Live streaming started.")

        if not MT5_AVAILABLE or self.config.mock_mode:
            # Run high-fidelity synthetic market generator
            await self._run_synthetic_feed()
            return

        # MT5 live polling loop with uvloop / async sleep
        while self._running:
            try:
                tick_info = await asyncio.to_thread(mt5.symbol_info_tick, self.symbol)
                if tick_info is not None:
                    self.process_raw_tick(
                        bid=tick_info.bid,
                        ask=tick_info.ask,
                        volume=float(getattr(tick_info, 'volume', 1.0) or 1.0),
                        flags=int(getattr(tick_info, 'flags', 0))
                    )
                await asyncio.sleep(0.02)  # 20ms micro-polling interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FEED MANAGER] Ingestion loop error: {e}")
                await asyncio.sleep(1.0)

    async def _run_synthetic_feed(self):
        """Simulate sub-second XAUUSD ticks with liquidity sweeps and realistic microstructure"""
        price = 2650.00
        while self._running:
            try:
                # Stochastic drift + micro jumps
                jump = np.random.choice([0, 0, 0, 0.45, -0.45], p=[0.7, 0.1, 0.1, 0.05, 0.05])
                noise = np.random.normal(0, 0.05)
                price = round(price + jump + noise, 2)
                spread = round(np.random.choice([0.12, 0.15, 0.18, 0.22, 0.38], p=[0.5, 0.3, 0.1, 0.07, 0.03]), 2)
                bid = round(price - spread / 2, 2)
                ask = round(price + spread / 2, 2)
                vol = random.randint(1, 15)

                self.process_raw_tick(bid=bid, ask=ask, volume=vol)
                await asyncio.sleep(0.05)  # 50ms tick rate
            except asyncio.CancelledError:
                break

    async def stop(self):
        """Graceful shutdown of ingestion layer"""
        logger.info("[FEED MANAGER] Stopping stream...")
        self._running = False
        if MT5_AVAILABLE:
            await asyncio.to_thread(mt5.shutdown)
        self.is_connected = False
        logger.info("[FEED MANAGER] Terminated cleanly.")
