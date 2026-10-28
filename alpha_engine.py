"""
alpha_engine.py
Anti-Gravity Alpha Feature Engine for XAUUSD (Gold).

Calculates:
1. Liquidity Sweeps (rolling session highs/lows with failed closes).
2. Wick Rejection Exhaustion (institutional pin-bar / absorption vectors).
3. Dynamic ATR & Rolling Spread Sentinel (volatility gating and execution filters).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SignalDirection(int, Enum):
    BUY = 1
    SELL = -1
    NEUTRAL = 0

    def __str__(self):
        if self.value == 1:
            return "BUY"
        elif self.value == -1:
            return "SELL"
        return "NEUTRAL"


class MarketRegime(str, Enum):
    COMPRESSED_RANGE = "COMPRESSED_RANGE"
    TRENDING_MOMENTUM_PULLBACK = "TRENDING_MOMENTUM_PULLBACK"
    VOLATILE_NOISE = "VOLATILE_NOISE"


@dataclass(frozen=True, init=False)
class AlphaSignal:
    timestamp: pd.Timestamp
    direction: int            # 1: Long, -1: Short, 0: Neutral / Filtered
    trigger_type: str         # "LIQUIDITY_SWEEP", "WICK_EXHAUSTION", "CONFLUENCE", "NONE"
    entry_price: float
    atr_val: float
    spread: float
    spread_ok: bool
    rejection_ratio: float
    
    # Extended bracket & ML fields
    suggested_sl: float
    suggested_tp: float
    rvol_val: float
    regime: MarketRegime
    confidence_score: float
    reasons: Tuple[str, ...]
    symbol: str

    def __init__(
        self,
        timestamp: Optional[pd.Timestamp] = None,
        direction: int = 0,
        trigger_type: str = "NONE",
        entry_price: float = 0.0,
        atr_val: Optional[float] = None,
        spread: float = 0.0,
        spread_ok: bool = True,
        rejection_ratio: Optional[float] = None,
        suggested_sl: float = 0.0,
        suggested_tp: float = 0.0,
        rvol_val: Optional[float] = None,
        regime: MarketRegime = MarketRegime.COMPRESSED_RANGE,
        confidence_score: float = 0.75,
        reasons: Tuple[str, ...] = (),
        symbol: str = "XAUUSD",
        # Aliases for compatibility
        atr_m1: Optional[float] = None,
        wick_ratio: Optional[float] = None,
        rvol_sigma: Optional[float] = None,
        normalized_atr: Optional[float] = None,
        **kwargs
    ):
        object.__setattr__(self, "timestamp", timestamp or pd.Timestamp.now())
        object.__setattr__(self, "direction", int(direction))
        object.__setattr__(self, "trigger_type", trigger_type)
        object.__setattr__(self, "entry_price", float(entry_price))
        
        final_atr = atr_val if atr_val is not None else (atr_m1 if atr_m1 is not None else 0.50)
        object.__setattr__(self, "atr_val", float(final_atr))
        object.__setattr__(self, "spread", float(spread))
        object.__setattr__(self, "spread_ok", bool(spread_ok))
        
        final_rejection = rejection_ratio if rejection_ratio is not None else (wick_ratio if wick_ratio is not None else 0.0)
        object.__setattr__(self, "rejection_ratio", float(final_rejection))
        object.__setattr__(self, "suggested_sl", float(suggested_sl))
        object.__setattr__(self, "suggested_tp", float(suggested_tp))
        
        final_rvol = rvol_val if rvol_val is not None else (rvol_sigma if rvol_sigma is not None else 0.0)
        object.__setattr__(self, "rvol_val", float(final_rvol))
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "confidence_score", float(confidence_score))
        object.__setattr__(self, "reasons", tuple(reasons) if not isinstance(reasons, tuple) else reasons)
        object.__setattr__(self, "symbol", symbol)

    @property
    def atr_m1(self) -> float:
        return self.atr_val

    @property
    def rvol_sigma(self) -> float:
        return self.rvol_val

    @property
    def wick_ratio(self) -> float:
        return self.rejection_ratio


class AntiGravityAlphaEngine:
    """
    Sub-Second Event-Driven Alpha Engine for XAUUSD (Gold).
    Calculates Liquidity Sweeps, Wick Rejection Exhaustion, and Volatility Filters.
    """
    def __init__(
        self,
        atr_period: Union[int, object] = 14,
        max_spread: float = 0.35,          # 35 cents max absolute spread on Gold
        spread_multiplier: float = 1.5,     # Max allowable spread as multiple of rolling median
        swing_lookback: int = 20,           # Bars to evaluate swing liquidity
        wick_ratio_threshold: float = 0.65, # Minimum 65% wick-to-total-range for exhaustion
        rvol_threshold: float = 2.0,        # Relative volume multiple
        config: Optional[object] = None
    ):
        # Support SystemConfig instance passed as first argument or keyword
        target_config = config
        if hasattr(atr_period, "alpha") or hasattr(atr_period, "instrument"):
            target_config = atr_period
            atr_period = 14

        if target_config is not None:
            alpha_cfg = getattr(target_config, "alpha", None)
            sentinel_cfg = getattr(target_config, "spread_sentinel", None)
            
            self.atr_period = int(getattr(alpha_cfg, "atr_period_m1", 14))
            self.max_spread = float(getattr(sentinel_cfg, "max_absolute_spread_usd", 0.35))
            self.spread_multiplier = float(getattr(sentinel_cfg, "rolling_median_multiplier", 1.5))
            self.swing_lookback = 20
            self.wick_ratio_threshold = float(getattr(alpha_cfg, "wick_to_body_ratio_min", 0.65))
            self.rvol_threshold = float(getattr(alpha_cfg, "rvol_sigma_threshold", 2.0))
        else:
            self.atr_period = int(atr_period) if isinstance(atr_period, (int, float)) else 14
            self.max_spread = float(max_spread)
            self.spread_multiplier = float(spread_multiplier)
            self.swing_lookback = int(swing_lookback)
            self.wick_ratio_threshold = float(wick_ratio_threshold)
            self.rvol_threshold = float(rvol_threshold)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vectorized Wilder's Average True Range calculation."""
        if len(df) == 0:
            return pd.Series(dtype=float)

        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Exponential moving average approximation for Wilder's ATR
        min_p = min(period, len(df))
        return tr.ewm(alpha=1.0 / period, min_periods=min_p, adjust=False).mean()

    def apply_spread_sentinel(self, df: pd.DataFrame) -> pd.Series:
        """
        Validates if current tick/candle spread is within institutional execution bounds.
        Rejects wide spreads caused by session roll, low liquidity, or high-impact macro releases.
        """
        if 'spread' not in df.columns:
            # Fallback if broker data feed passes tick ask/bid rather than explicit spread
            if 'ask' in df.columns and 'bid' in df.columns:
                df = df.copy()
                df['spread'] = df['ask'] - df['bid']
            else:
                return pd.Series(True, index=df.index)

        rolling_median_spread = df['spread'].rolling(window=50, min_periods=5).median()
        spread_ok = (df['spread'] <= self.max_spread) & (
            df['spread'] <= (rolling_median_spread * self.spread_multiplier).fillna(self.max_spread)
        )
        return spread_ok

    def compute_wick_exhaustion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects momentum exhaustion via upper/lower wick rejections and volume spikes.
        - Bearish Exhaustion: Extended top wick rejecting higher prices.
        - Bullish Exhaustion: Extended bottom wick absorbing selling pressure.
        """
        candle_range = (df['high'] - df['low']).replace(0, 1e-6)
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']

        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # Relative Volume calculation
        volume_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        if volume_col not in df.columns:
            rolling_vol = pd.Series(1.0, index=df.index)
            rvol = pd.Series(1.0, index=df.index)
        else:
            rolling_vol = df[volume_col].rolling(window=20, min_periods=5).mean()
            rvol = df[volume_col] / rolling_vol.replace(0, 1e-6)

        bearish_exhaustion = (upper_wick_ratio >= self.wick_ratio_threshold) & (rvol >= self.rvol_threshold)
        bullish_exhaustion = (lower_wick_ratio >= self.wick_ratio_threshold) & (rvol >= self.rvol_threshold)

        return pd.DataFrame({
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'rvol': rvol,
            'bear_exhaustion': bearish_exhaustion,
            'bull_exhaustion': bullish_exhaustion
        }, index=df.index)

    def compute_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies stop-hunt liquidity sweeps:
        - Bearish Sweep: Price spikes above recent N-period High but closes strictly back inside.
        - Bullish Sweep: Price pierces below recent N-period Low but closes strictly back inside.
        """
        min_lookback = min(self.swing_lookback, max(2, len(df) - 1))
        rolling_high = df['high'].shift(1).rolling(window=min_lookback, min_periods=2).max()
        rolling_low = df['low'].shift(1).rolling(window=min_lookback, min_periods=2).min()

        # Sweep High: High > prior swing high, but close failed to accept above it
        sweep_high = (df['high'] > rolling_high) & (df['close'] < rolling_high)

        # Sweep Low: Low < prior swing low, but close held above it
        sweep_low = (df['low'] < rolling_low) & (df['close'] > rolling_low)

        return pd.DataFrame({
            'swing_high': rolling_high,
            'swing_low': rolling_low,
            'sweep_high': sweep_high,
            'sweep_low': sweep_low
        }, index=df.index)

    def classify_regime(self, df: pd.DataFrame, m1_atr: float, baseline_atr: float) -> MarketRegime:
        """
        Market Regime Classifier:
        - COMPRESSED_RANGE: low normalized ATR, narrow consolidation.
        - TRENDING_MOMENTUM_PULLBACK: moving average alignment with stable volatility.
        - VOLATILE_NOISE: extreme expansion or uncontained spread.
        """
        if len(df) < 20:
            return MarketRegime.COMPRESSED_RANGE

        normalized_atr = m1_atr / max(baseline_atr, 0.10)
        if normalized_atr > 2.2:
            return MarketRegime.VOLATILE_NOISE

        closes = df['close']
        ema9 = closes.ewm(span=9, adjust=False).mean()
        ema21 = closes.ewm(span=21, adjust=False).mean()

        trend_diff = abs(ema9.iloc[-1] - ema21.iloc[-1])
        if trend_diff < 0.35 and normalized_atr <= 1.25:
            return MarketRegime.COMPRESSED_RANGE
        elif trend_diff >= 0.35:
            return MarketRegime.TRENDING_MOMENTUM_PULLBACK

        return MarketRegime.COMPRESSED_RANGE

    def evaluate(self, df: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Full feature pipeline execution on M1 OHLCV stream.
        Returns the latest bar's structured signal.
        """
        if len(df) < max(self.swing_lookback, self.atr_period, 10):
            return None

        # 1. Volatility & Spread Sentinel
        atr_series = self.calculate_atr(df, period=self.atr_period)
        spread_ok_series = self.apply_spread_sentinel(df)

        # Baseline ATR for regime determination (60-period proxy)
        baseline_atr_series = self.calculate_atr(df, period=min(60, len(df)))

        # 2. Price Action Vectors
        exhaustion_df = self.compute_wick_exhaustion(df)
        sweeps_df = self.compute_liquidity_sweeps(df)

        # Extract latest closed bar attributes
        idx = df.index[-1]
        last_close = float(df['close'].iloc[-1])
        last_atr = float(atr_series.iloc[-1])
        self.latest_atr = last_atr
        last_baseline_atr = float(baseline_atr_series.iloc[-1])
        last_spread = float(df['spread'].iloc[-1]) if 'spread' in df.columns else 0.15
        is_spread_valid = bool(spread_ok_series.iloc[-1])

        # State conditions for Long / Short
        bull_sweep = bool(sweeps_df['sweep_low'].iloc[-1])
        bear_sweep = bool(sweeps_df['sweep_high'].iloc[-1])
        bull_wick = bool(exhaustion_df['bull_exhaustion'].iloc[-1])
        bear_wick = bool(exhaustion_df['bear_exhaustion'].iloc[-1])
        rvol_val = float(exhaustion_df['rvol'].iloc[-1])

        direction = SignalDirection.NEUTRAL.value
        trigger = "NONE"
        rejection_ratio = 0.0
        confidence = 0.0
        reasons_list = []

        if is_spread_valid:
            if bull_sweep and bull_wick:
                direction = SignalDirection.BUY.value
                trigger = "CONFLUENCE"
                rejection_ratio = float(exhaustion_df['lower_wick_ratio'].iloc[-1])
                confidence = 0.85
                reasons_list = ["BULLISH_SWEEP_LOW", "BULLISH_WICK_EXHAUSTION", "CONFLUENCE"]
            elif bear_sweep and bear_wick:
                direction = SignalDirection.SELL.value
                trigger = "CONFLUENCE"
                rejection_ratio = float(exhaustion_df['upper_wick_ratio'].iloc[-1])
                confidence = 0.85
                reasons_list = ["BEARISH_SWEEP_HIGH", "BEARISH_WICK_EXHAUSTION", "CONFLUENCE"]
            elif bull_sweep and (float(exhaustion_df['lower_wick_ratio'].iloc[-1]) >= 0.35 or rvol_val >= 1.5):
                direction = SignalDirection.BUY.value
                trigger = "LIQUIDITY_SWEEP"
                rejection_ratio = float(exhaustion_df['lower_wick_ratio'].iloc[-1])
                confidence = 0.75
                reasons_list = ["BULLISH_SWEEP_LOW"]
            elif bear_sweep and (float(exhaustion_df['upper_wick_ratio'].iloc[-1]) >= 0.35 or rvol_val >= 1.5):
                direction = SignalDirection.SELL.value
                trigger = "LIQUIDITY_SWEEP"
                rejection_ratio = float(exhaustion_df['upper_wick_ratio'].iloc[-1])
                confidence = 0.75
                reasons_list = ["BEARISH_SWEEP_HIGH"]
            elif bull_wick:
                direction = SignalDirection.BUY.value
                trigger = "WICK_EXHAUSTION"
                rejection_ratio = float(exhaustion_df['lower_wick_ratio'].iloc[-1])
                confidence = 0.72
                reasons_list = ["BULLISH_WICK_EXHAUSTION"]
            elif bear_wick:
                direction = SignalDirection.SELL.value
                trigger = "WICK_EXHAUSTION"
                rejection_ratio = float(exhaustion_df['upper_wick_ratio'].iloc[-1])
                confidence = 0.72
                reasons_list = ["BEARISH_WICK_EXHAUSTION"]

        regime = self.classify_regime(df, last_atr, last_baseline_atr)

        # Dynamic brackets: 1.0x ATR for SL (min $0.80), 2.0x ATR for runner TP (min $1.60)
        sl_dist = max(last_atr * 1.0, 0.80)
        tp_dist = max(last_atr * 2.0, 1.60)

        if direction == SignalDirection.BUY.value:
            suggested_sl = round(last_close - sl_dist, 2)
            suggested_tp = round(last_close + tp_dist, 2)
        elif direction == SignalDirection.SELL.value:
            suggested_sl = round(last_close + sl_dist, 2)
            suggested_tp = round(last_close - tp_dist, 2)
        else:
            suggested_sl = 0.0
            suggested_tp = 0.0

        return AlphaSignal(
            timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
            direction=direction,
            trigger_type=trigger,
            entry_price=last_close,
            atr_val=last_atr,
            spread=last_spread,
            spread_ok=is_spread_valid,
            rejection_ratio=rejection_ratio,
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            rvol_val=rvol_val,
            regime=regime,
            confidence_score=confidence,
            reasons=tuple(reasons_list)
        )

    def generate_features_and_signals(self, df: pd.DataFrame) -> Tuple[Optional[AlphaSignal], Dict[str, float]]:
        """
        Orchestration adapter returning the signal and feature vector for downstream ML inference.
        """
        signal = self.evaluate(df)
        if signal is None:
            return None, {}

        last_bar = df.iloc[-1]
        features = {
            "m1_atr": signal.atr_val,
            "normalized_atr": round(signal.atr_val / max(0.50, signal.atr_val), 3),
            "rvol_sigma": signal.rvol_val,
            "wick_ratio": signal.rejection_ratio,
            "delta": float(last_bar.get('delta', 0.0)),
            "cvd": float(last_bar.get('cvd', 0.0)),
            "tick_count": float(last_bar.get('tick_count', 0.0)),
            "spread": signal.spread,
            "sweep_dir": 1.0 if "SWEEP" in signal.trigger_type and signal.direction == 1 else (-1.0 if "SWEEP" in signal.trigger_type and signal.direction == -1 else 0.0),
            "exh_dir": 1.0 if "EXHAUSTION" in signal.trigger_type and signal.direction == 1 else (-1.0 if "EXHAUSTION" in signal.trigger_type and signal.direction == -1 else 0.0),
            "div_dir": 0.0
        }

        return signal, features


AlphaEngine = AntiGravityAlphaEngine
