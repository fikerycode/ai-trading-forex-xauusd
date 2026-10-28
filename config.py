"""
SYSTEM ARCHITECTURE SPECIFICATION: AI-DRIVEN XAUUSD SCALPER (ANTI-GRAVITY)
MODULE: Configuration & Environment Setup
"""

from typing import List, Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = dict
from pydantic import Field


class InstrumentConfig(BaseSettings):
    """XAUUSD Instrument Constants & Specifications"""
    symbol: str = "XAUUSD"
    contract_size: float = 100.0          # 1 Standard Lot = 100 Troy Ounces
    digits: int = 2                       # 2 decimals (0.01 = 1 cent)
    point_size: float = 0.01              # Minimum tick size ($0.01)
    pip_size: float = 0.10                # 1 Pip = $0.10 (10 points / cents)
    min_lot: float = 0.01
    max_lot: float = 50.0
    lot_step: float = 0.01
    slippage_tolerance_usd: float = 0.15  # Max 15 cents slippage tolerance on execution ($0.15)
    bracket_timeout_ms: int = 50          # Protective bracket placement window


class SpreadSentinelConfig(BaseSettings):
    """Volatility and Spread Safety Gates"""
    max_absolute_spread_usd: float = 0.35 # Max 35 cents spread gate ($0.35)
    rolling_median_window: int = 60       # Window in ticks / M1 bars
    rolling_median_multiplier: float = 1.5 # Gate if spread > 1.5 * median
    news_lockout_before_min: int = 15     # Lockout 15 min prior to high-impact events
    news_lockout_after_min: int = 20      # Lockout 20 min post high-impact events
    monitored_news_keywords: List[str] = [
        "CPI", "NFP", "Non-Farm", "FOMC", "Fed Interest Rate", "PPI", "GDP", "Jackson Hole"
    ]


class AlphaEngineConfig(BaseSettings):
    """Alpha Feature & Momentum Exhaustion Constants"""
    rvol_sigma_threshold: float = 2.5     # RVOL spike threshold (> 2.5 sigma)
    wick_to_body_ratio_min: float = 0.65  # Rejection wick-to-body ratio > 65%
    atr_period_m1: int = 14               # 14-period M1 ATR
    atr_period_h1_baseline: int = 14      # H1 baseline ATR period
    fvg_min_gap_usd: float = 0.40         # Minimum dollar gap for Fair Value Gap
    asian_session_start_utc: str = "00:00"
    asian_session_end_utc: str = "07:00"
    london_open_utc: str = "07:00"
    ny_open_utc: str = "12:00"
    ny_close_utc: str = "21:00"


class MLInferenceConfig(BaseSettings):
    """Machine Learning & Statistical Conformal Inference"""
    model_path: Optional[str] = "antigravity_gold_model.json"
    min_direction_probability: float = 0.315 # Calibrated top-tier setup threshold (baseline class prior is ~31.8%)
    max_entropy_threshold: float = 0.68     # Information entropy filter
    horizon_bars: int = 8                    # 8-bar forward target evaluation
    tp_atr_multiple: float = 1.5             # Target TP = +1.5x M1 ATR
    sl_atr_multiple: float = 1.0             # Target SL = -1.0x M1 ATR
    allowed_regimes: List[str] = ["COMPRESSED_RANGE", "TRENDING_MOMENTUM_PULLBACK"]
    suppressed_regimes: List[str] = ["VOLATILE_NOISE"]


class RiskMatrixConfig(BaseSettings):
    """Capital Preservation & Lifecycle Matrix"""
    risk_per_trade_min_pct: float = 0.005    # 0.5% minimum risk
    risk_per_trade_max_pct: float = 0.010    # 1.0% maximum risk
    default_risk_pct: float = 0.0075         # 0.75% baseline risk per trade
    max_daily_loss_pct: float = 0.030        # 3.0% Max Intraday Loss circuit breaker
    max_concurrent_trades: int = 1
    
    # Dynamic Trailing & Scaling
    breakeven_r_trigger: float = 1.0         # Move to BE at 1.0R
    breakeven_spread_buffer_usd: float = 0.10 # BE + spread buffer ($0.10)
    partial_scale_out_r_trigger: float = 1.5 # Scale out 50% at 1.5R
    partial_scale_out_ratio: float = 0.50    # 50% scale out
    trailing_atr_multiple: float = 1.2       # Dynamic trailing 1.2x M1 ATR for remaining 50%


class MT5Credentials(BaseSettings):
    """Broker connection credentials"""
    login: Optional[int] = Field(default=None, alias="MT5_LOGIN")
    password: Optional[str] = Field(default=None, alias="MT5_PASSWORD")
    server: Optional[str] = Field(default=None, alias="MT5_SERVER")
    path: Optional[str] = Field(default=None, alias="MT5_PATH")
    timeout_ms: int = 60000
    portable: bool = False
    demo_fallback: bool = True               # Allows mock streaming if MT5 is offline/unconfigured

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class SystemConfig(BaseSettings):
    """Master Application Configuration"""
    system_name: str = "ANTI-GRAVITY-XAUUSD-SCALPER"
    system_version: str = "1.0.0"
    magic_number: int = 888999
    log_level: str = "INFO"
    mock_mode: bool = False                  # Standalone synthetic market generator for stress testing
    
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    spread_sentinel: SpreadSentinelConfig = Field(default_factory=SpreadSentinelConfig)
    alpha: AlphaEngineConfig = Field(default_factory=AlphaEngineConfig)
    ml: MLInferenceConfig = Field(default_factory=MLInferenceConfig)
    risk: RiskMatrixConfig = Field(default_factory=RiskMatrixConfig)
    mt5: MT5Credentials = Field(default_factory=MT5Credentials)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


    def load_optimized_parameters(self, json_path: str = "optimized_parameters.json") -> bool:
        """Loads tuned hyperparameters from optimizer output if available."""
        import json, os
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, "r") as f:
                params = json.load(f)
            if "wick_ratio" in params:
                self.alpha.wick_to_body_ratio_min = float(params["wick_ratio"])
            if "rvol_sigma" in params:
                self.alpha.rvol_sigma_threshold = float(params["rvol_sigma"])
            if "be_r" in params:
                self.risk.breakeven_r_trigger = float(params["be_r"])
            if "partial_r" in params:
                self.risk.partial_scale_out_r_trigger = float(params["partial_r"])
            if "trailing_atr" in params:
                self.risk.trailing_atr_multiple = float(params["trailing_atr"])
            return True
        except Exception:
            return False


# Global singleton configuration
CONFIG = SystemConfig()
CONFIG.load_optimized_parameters()
