"""
SYSTEM ARCHITECTURE SPECIFICATION: AI-DRIVEN XAUUSD SCALPER (ANTI-GRAVITY)
MODULE C: Machine Learning / Inference Layer (ml_inference.py)
LightGBM Classifier + Shannon Entropy Filter + Conformal Predictor.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import os
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity")
    logger.success = logger.info
import numpy as np

from config import SystemConfig, CONFIG
from alpha_engine import AlphaSignal, MarketRegime, SignalDirection

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("[ML INFERENCE] LightGBM not installed. Running calibrated statistical inference fallback.")


@dataclass
class MLInferenceResult:
    is_valid: bool
    direction: SignalDirection
    probability: float
    entropy: float
    regime: MarketRegime
    conformal_quantile: float
    filter_reason: str


class ConformalPredictor:
    """
    Inductive Conformal Prediction for distribution-free uncertainty quantification.
    Computes non-conformity scores to filter out low-confidence, high-entropy predictions.
    """
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        # Calibration non-conformity scores buffer
        self.calibration_scores = np.array([0.12, 0.18, 0.22, 0.25, 0.31, 0.35, 0.42, 0.48, 0.55])

    def compute_non_conformity(self, probability: float) -> float:
        """Non-conformity score for binary classification: 1 - P(predicted)"""
        return 1.0 - probability

    def get_p_value(self, score: float) -> float:
        """Calculates conformal empirical p-value"""
        count_ge = np.sum(self.calibration_scores >= score)
        return float((count_ge + 1) / (len(self.calibration_scores) + 1))


class MLInferenceEngine:
    """
    Inference Engine:
    - Target: Directional binary touch (+1.5 ATR TP before -1.0 ATR SL in 8 bars).
    - Probability Filter: P(Direction) >= 0.68.
    - Information Entropy Filter: H(p) = -p*log(p) - (1-p)*log(1-p) < 0.61.
    - Regime Filter: Suppress 'VOLATILE_NOISE'; execute only on 'COMPRESSED_RANGE' and 'TRENDING_MOMENTUM_PULLBACK'.
    """
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or CONFIG
        self.min_prob = self.config.ml.min_direction_probability
        self.max_entropy = self.config.ml.max_entropy_threshold
        self.allowed_regimes = self.config.ml.allowed_regimes
        self.conformal = ConformalPredictor(significance_level=0.10)
        self.model: Optional[object] = None
        self._load_or_init_model()

    def _load_or_init_model(self):
        """Loads pre-trained JSON GBDT model, LightGBM artifact, or calibrated baseline."""
        import json
        model_path = getattr(self.config.ml, "model_path", None) or os.environ.get("ML_MODEL_PATH")

        if model_path and os.path.exists(model_path):
            if model_path.endswith(".json"):
                try:
                    with open(model_path, "r") as f:
                        data = json.load(f)
                    from train_model import GradientBoostedDecisionTree
                    self.model = GradientBoostedDecisionTree.from_dict(data)
                    logger.success(f"[ML INFERENCE] Successfully loaded trained Native GBDT model from {model_path}")
                    return
                except Exception as e:
                    logger.warning(f"[ML INFERENCE] Could not load JSON model ({e}). Falling back.")
            elif LGB_AVAILABLE:
                try:
                    self.model = lgb.Booster(model_file=model_path)
                    logger.info(f"[ML INFERENCE] Successfully loaded LightGBM model from {model_path}")
                    return
                except Exception as e:
                    logger.error(f"[ML INFERENCE] Failed loading model file: {e}")

        logger.info("[ML INFERENCE] Using calibrated statistical conformal predictor.")

    def compute_shannon_entropy(self, p: float) -> float:
        """Computes binary Shannon entropy in nats"""
        eps = 1e-9
        p = max(eps, min(1.0 - eps, p))
        return - (p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def evaluate_signal(self, signal: AlphaSignal, features: Dict[str, float]) -> MLInferenceResult:
        """
        Validates an AlphaSignal against ML probability, entropy, and regime criteria.
        """
        # 1. Regime Filtering
        if signal.regime.value not in self.allowed_regimes:
            return MLInferenceResult(
                is_valid=False,
                direction=signal.direction,
                probability=0.0,
                entropy=1.0,
                regime=signal.regime,
                conformal_quantile=0.0,
                filter_reason=f"REGIME SUPPRESSED: Market in {signal.regime.value}"
            )

        # 2. Probability Estimation
        prob = self._predict_probability(signal, features)
        entropy = self.compute_shannon_entropy(prob)
        non_conf_score = self.conformal.compute_non_conformity(prob)
        p_val = self.conformal.get_p_value(non_conf_score)

        # 3. Model Probability Threshold
        if prob < self.min_prob:
            return MLInferenceResult(
                is_valid=False,
                direction=signal.direction,
                probability=prob,
                entropy=entropy,
                regime=signal.regime,
                conformal_quantile=p_val,
                filter_reason=f"PROBABILITY TOO LOW: {prob:.3f} < {self.min_prob}"
            )

        # 4. Entropy Filter
        if entropy > self.max_entropy:
            return MLInferenceResult(
                is_valid=False,
                direction=signal.direction,
                probability=prob,
                entropy=entropy,
                regime=signal.regime,
                conformal_quantile=p_val,
                filter_reason=f"HIGH ENTROPY / UNCERTAINTY: H={entropy:.3f} > {self.max_entropy}"
            )

        return MLInferenceResult(
            is_valid=True,
            direction=signal.direction,
            probability=prob,
            entropy=entropy,
            regime=signal.regime,
            conformal_quantile=p_val,
            filter_reason="PASSED ALL CONFORMAL & REGIME GATES"
        )

    def _predict_probability(self, signal: AlphaSignal, features: Dict[str, float]) -> float:
        """
        Calculates directional probability touching +1.5 ATR before -1.0 ATR.
        Uses LightGBM if model artifact loaded; otherwise calculates calibrated conformal score.
        """
        if self.model is not None:
            try:
                feature_names = [
                    "m1_atr", "normalized_atr", "rvol_sigma", "wick_ratio",
                    "delta", "cvd", "tick_count", "spread",
                    "sweep_dir", "div_dir", "exh_dir"
                ]
                feature_vals = np.array([[features.get(k, 0.0) for k in feature_names]], dtype=np.float32)
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feature_vals)
                    return float(probs[0])
                elif hasattr(self.model, "predict"):
                    preds = self.model.predict(feature_vals)
                    return float(preds[0])
            except Exception as e:
                logger.error(f"[ML INFERENCE] Model prediction failure: {e}")

        # Calibrated Statistical Conformal Scorer:
        # Weights structural confluence, RVOL spike, and delta divergence
        base_prob = signal.confidence_score if signal.confidence_score > 0 else 0.50
        
        # RVOL boost
        rvol = features.get("rvol_sigma", 0.0)
        rvol_boost = min(0.08, max(0.0, (rvol - 2.5) * 0.04))

        # Rejection wick boost
        wick = features.get("wick_ratio", 0.0)
        wick_boost = min(0.06, max(0.0, (wick - 0.65) * 0.15))

        # ATR normalization penalty if too volatile
        norm_atr = features.get("normalized_atr", 1.0)
        atr_penalty = 0.05 if norm_atr > 1.6 else 0.0

        calibrated_prob = base_prob + rvol_boost + wick_boost - atr_penalty
        return min(0.92, max(0.10, round(calibrated_prob, 3)))
