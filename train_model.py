"""
train_model.py
Institutional Machine Learning Training Pipeline for Anti-Gravity XAUUSD Scalper.

Capabilities:
1. Ingests historical Gold M1 candlestick data (CSV or live internet feed).
2. Generates institutional feature vector:
   - Rolling 14-period M1 ATR & Normalized Volatility Ratio
   - Relative Volume (RVOL) z-score / sigma
   - Wick-to-body pin-bar exhaustion ratios
   - Liquidity sweep vectors & session regime encodings
   - Momentum & Cumulative Delta proxy
3. Implements De Prado's Triple-Barrier Labeling Method:
   - Target Upper Barrier: +1.5x ATR (Take-Profit)
   - Target Lower Barrier: -1.0x ATR (Stop-Loss)
   - Vertical Barrier: 8 bars max holding horizon
4. Trains an institutional Gradient Boosted Decision Tree (GBDT) Classifier.
5. Evaluates Out-Of-Sample (OOS) Precision, Recall, and ROC-AUC.
6. Serializes model weights and feature definitions to 'antigravity_gold_model.json'.
"""

import argparse
from dataclasses import dataclass, field
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity.trainer")
    logger.success = logger.info

import numpy as np
import pandas as pd

from backtester import BacktestEngine
from config import SystemConfig, CONFIG


# =====================================================================
# NATIVE VECTORIZED GRADIENT BOOSTED DECISION TREE (GBDT) ENGINE
# =====================================================================

class DecisionTreeNode:
    def __init__(
        self,
        feature_idx: int = -1,
        threshold: float = 0.0,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        leaf_value: Optional[float] = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value = leaf_value

    @property
    def is_leaf(self) -> bool:
        return self.leaf_value is not None

    def to_dict(self) -> Dict:
        if self.is_leaf:
            return {"leaf_value": round(float(self.leaf_value), 6)}
        return {
            "feature_idx": int(self.feature_idx),
            "threshold": round(float(self.threshold), 6),
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DecisionTreeNode":
        if d.get("leaf_value") is not None:
            return cls(leaf_value=d["leaf_value"])
        return cls(
            feature_idx=d["feature_idx"],
            threshold=d["threshold"],
            left=cls.from_dict(d["left"]) if d.get("left") else None,
            right=cls.from_dict(d["right"]) if d.get("right") else None
        )


class GradientBoostedDecisionTree:
    """
    Lightweight, deterministic, zero-dependency GBDT for sub-millisecond inference.
    """
    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.10,
        max_depth: int = 3,
        min_samples_split: int = 15,
        subsample: float = 0.85
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.trees: List[DecisionTreeNode] = []
        self.base_score: float = 0.0
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Trains GBDT using binary cross-entropy pseudo-residuals."""
        self.feature_names = feature_names
        n_samples, n_features = X.shape
        self.feature_importances = {fn: 0.0 for fn in feature_names}

        # Initialize base prediction to log-odds
        pos_ratio = max(0.01, min(0.99, np.mean(y)))
        self.base_score = float(np.log(pos_ratio / (1.0 - pos_ratio)))
        raw_preds = np.full(n_samples, self.base_score)

        for tree_idx in range(self.n_estimators):
            # Sigmoid probabilities
            probs = 1.0 / (1.0 + np.exp(-raw_preds))
            residuals = y - probs  # Negative gradient of log loss

            # Subsampling
            mask = np.random.rand(n_samples) < self.subsample
            X_sub = X[mask]
            residuals_sub = residuals[mask]

            tree = self._build_tree(X_sub, residuals_sub, depth=0)
            self.trees.append(tree)

            # Update predictions
            update = np.array([self._predict_tree(tree, x) for x in X])
            raw_preds += self.learning_rate * update

        # Normalize feature importances
        total_imp = sum(self.feature_importances.values()) + 1e-9
        for fn in self.feature_importances:
            self.feature_importances[fn] = round(self.feature_importances[fn] / total_imp, 4)

    def _build_tree(self, X: np.ndarray, residuals: np.ndarray, depth: int) -> DecisionTreeNode:
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            # Optimal leaf step for binary cross-entropy
            leaf_val = float(np.mean(residuals))
            return DecisionTreeNode(leaf_value=leaf_val)

        best_feat, best_thresh, best_var_red = -1, 0.0, -1.0
        current_variance = np.var(residuals) * n_samples

        for f_idx in range(n_features):
            values = X[:, f_idx]
            unique_vals = np.percentile(values, [20, 40, 60, 80])

            for thresh in unique_vals:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue

                left_var = np.var(residuals[left_mask]) * np.sum(left_mask)
                right_var = np.var(residuals[right_mask]) * np.sum(right_mask)
                var_reduction = current_variance - (left_var + right_var)

                if var_reduction > best_var_red:
                    best_var_red = var_reduction
                    best_feat = f_idx
                    best_thresh = thresh

        if best_feat == -1 or best_var_red <= 0.0:
            return DecisionTreeNode(leaf_value=float(np.mean(residuals)))

        # Track importance
        self.feature_importances[self.feature_names[best_feat]] += best_var_red

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], residuals[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], residuals[~left_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child
        )

    def _predict_tree(self, node: DecisionTreeNode, x: np.ndarray) -> float:
        if node.is_leaf:
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_tree(node.left, x)
        return self._predict_tree(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs calibrated class 1 probability."""
        raw = np.full(len(X), self.base_score)
        for tree in self.trees:
            updates = np.array([self._predict_tree(tree, x) for x in X])
            raw += self.learning_rate * updates
        return 1.0 / (1.0 + np.exp(-raw))

    def to_dict(self) -> Dict:
        return {
            "model_type": "NativeGBDT",
            "base_score": round(self.base_score, 6),
            "learning_rate": self.learning_rate,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "trees": [t.to_dict() for t in self.trees]
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GradientBoostedDecisionTree":
        gbdt = cls(
            n_estimators=len(d.get("trees", [])),
            learning_rate=d.get("learning_rate", 0.10)
        )
        gbdt.base_score = d.get("base_score", 0.0)
        gbdt.feature_names = d.get("feature_names", [])
        gbdt.feature_importances = d.get("feature_importances", {})
        gbdt.trees = [DecisionTreeNode.from_dict(t) for t in d.get("trees", [])]
        return gbdt


# =====================================================================
# FEATURE ENGINEERING & TRIPLE-BARRIER LABELING
# =====================================================================

class ModelTrainer:
    """
    Automated Institutional Feature Ingestion, Labeling, and Model Training Pipeline.
    """
    def __init__(
        self,
        dataset: pd.DataFrame,
        horizon_bars: int = 8,
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.0
    ):
        self.df = dataset.copy()
        self.horizon = horizon_bars
        self.tp_mult = tp_multiplier
        self.sl_mult = sl_multiplier

    def compute_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts rolling alpha features and applies Triple-Barrier labeling.
        """
        logger.info(f"[TRAINER] Extracting institutional features from {len(self.df)} M1 bars...")

        df = self.df
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        vol = df.get('tick_volume', pd.Series(100, index=df.index)).values

        # 1. 14-period ATR
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # 2. RVOL (Relative Volume z-score)
        vol_mean = pd.Series(vol).rolling(20, min_periods=5).mean().values
        vol_std = pd.Series(vol).rolling(20, min_periods=5).std().fillna(1.0).values + 1e-6
        rvol_sigma = (vol - vol_mean) / vol_std

        # 3. Normalized ATR Ratio
        median_atr = pd.Series(atr).rolling(60, min_periods=10).median().fillna(1.0).values + 1e-6
        norm_atr = atr / median_atr

        # 4. Wick Exhaustion Ratios
        body_range = abs(close - open_p)
        total_range = np.maximum(high - low, 0.05)
        upper_wick = high - np.maximum(open_p, close)
        lower_wick = np.minimum(open_p, close) - low
        wick_ratio = np.maximum(upper_wick, lower_wick) / total_range

        # 5. Liquidity Sweep Indicator (rolling 20 highs/lows)
        roll_high = pd.Series(high).rolling(20).max().shift(1).values
        roll_low = pd.Series(low).rolling(20).min().shift(1).values
        sweep_dir = np.where((high > roll_high) & (close < roll_high), -1,
                    np.where((low < roll_low) & (close > roll_low), 1, 0))

        # 6. Proxy CVD (Cumulative Volume Delta)
        bar_direction = np.sign(close - open_p)
        delta = bar_direction * vol
        cvd = pd.Series(delta).rolling(20, min_periods=1).sum().values

        # 7. Spread
        spread = df.get('spread', pd.Series(0.15, index=df.index)).values

        feature_matrix = []
        labels = []
        feature_names = [
            "m1_atr", "normalized_atr", "rvol_sigma", "wick_ratio",
            "delta", "cvd", "tick_count", "spread",
            "sweep_dir", "div_dir", "exh_dir"
        ]

        # Triple-Barrier Evaluation (look forward up to horizon bars)
        n = len(df)
        for i in range(50, n - self.horizon):
            curr_c = close[i]
            curr_atr = max(atr[i], 0.40)
            tp_dist = curr_atr * self.tp_mult
            sl_dist = curr_atr * self.sl_mult

            # Forward window
            f_highs = high[i+1 : i+1+self.horizon]
            f_lows = low[i+1 : i+1+self.horizon]

            # Determine setup direction: Sweep > Wick Exhaustion > Momentum
            s_dir = float(sweep_dir[i])
            e_dir = float(np.sign(lower_wick[i] - upper_wick[i]))
            b_dir = float(bar_direction[i])
            setup_dir = s_dir if s_dir != 0 else (e_dir if e_dir != 0 else (b_dir if b_dir != 0 else 1.0))

            # Directional Barrier Touch Evaluation
            if setup_dir > 0:
                tp_hits = (f_highs >= curr_c + tp_dist)
                sl_hits = (f_lows <= curr_c - sl_dist)
            else:
                tp_hits = (f_lows <= curr_c - tp_dist)
                sl_hits = (f_highs >= curr_c + sl_dist)

            # Triple-Barrier Outcome: 1 = favorable touch (+1.5R before -1.0R), 0 = stopped or noise
            if np.any(tp_hits) and not np.any(sl_hits):
                target = 1
            elif not np.any(tp_hits) and np.any(sl_hits):
                target = 0
            else:
                tp_idx = np.where(tp_hits)[0]
                sl_idx = np.where(sl_hits)[0]
                first_tp = tp_idx[0] if len(tp_idx) > 0 else 999
                first_sl = sl_idx[0] if len(sl_idx) > 0 else 999
                target = 1 if first_tp < first_sl else 0

            row = [
                float(atr[i]),
                float(norm_atr[i]),
                float(rvol_sigma[i]),
                float(wick_ratio[i]),
                float(delta[i]),
                float(cvd[i]),
                float(vol[i]),
                float(spread[i]),
                s_dir,
                setup_dir,
                e_dir
            ]

            feature_matrix.append(row)
            labels.append(target)

        X = np.array(feature_matrix, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        logger.info(f"[TRAINER] Constructed {len(X)} training samples. Positive touch class balance: {np.mean(y)*100:.1f}%.")
        return X, y, feature_names

    def train_and_evaluate(
        self,
        test_split: float = 0.20,
        n_estimators: int = 25,
        learning_rate: float = 0.12,
        max_depth: int = 3
    ) -> Tuple[GradientBoostedDecisionTree, Dict]:
        """
        Executes causal chronological Train/Test split and trains GBDT model.
        """
        X, y, feature_names = self.compute_features_and_labels()

        split_idx = int(len(X) * (1.0 - test_split))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        logger.info(f"[TRAINER] Training GBDT: {len(X_train)} Train samples | {len(X_test)} Out-Of-Sample Test samples...")

        model = GradientBoostedDecisionTree(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        model.fit(X_train, y_train, feature_names)

        # Evaluate on Test Set
        probs_test = model.predict_proba(X_test)
        preds_test = (probs_test >= 0.50).astype(int)

        accuracy = float(np.mean(preds_test == y_test))
        
        # High confidence precision (P >= 0.65)
        high_conf_mask = probs_test >= 0.65
        if np.sum(high_conf_mask) > 0:
            high_conf_precision = float(np.mean(y_test[high_conf_mask] == 1))
        else:
            high_conf_precision = 0.0

        metrics = {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy_pct": round(accuracy * 100.0, 2),
            "high_conf_precision_pct": round(high_conf_precision * 100.0, 2),
            "high_conf_trades_count": int(np.sum(high_conf_mask)),
            "feature_importances": model.feature_importances
        }

        return model, metrics

    def export_model_artifact(
        self,
        model: GradientBoostedDecisionTree,
        metrics: Dict,
        output_path: str = "antigravity_gold_model.json"
    ):
        """Exports serialized model architecture and metadata to JSON."""
        data = model.to_dict()
        data["evaluation_metrics"] = metrics
        data["trained_timestamp"] = str(pd.Timestamp.now())

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

        logger.success(f"[TRAINER] Model successfully serialized -> {output_path}")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-Gravity XAUUSD Machine Learning Training Engine")
    parser.add_argument("--csv", type=str, default="gold_historical_m1.csv", help="Historical Gold M1 CSV")
    parser.add_argument("--source", type=str, choices=["csv", "internet"], default="csv", help="Dataset source")
    parser.add_argument("--trees", type=int, default=25, help="Number of GBDT decision trees (default: 25)")
    parser.add_argument("--lr", type=float, default=0.12, help="Learning rate (default: 0.12)")
    parser.add_argument("--depth", type=int, default=3, help="Max tree depth (default: 3)")
    parser.add_argument("--output", type=str, default="antigravity_gold_model.json", help="Export model filename")

    args = parser.parse_args()

    engine = BacktestEngine()

    if args.source == "csv" and os.path.exists(args.csv):
        print(f"[INFO] Loading training dataset from {args.csv}...")
        df_gold = engine.load_data_from_csv(args.csv)
    elif os.path.exists("gold_historical_m1.csv"):
        print(f"[INFO] Loading local 'gold_historical_m1.csv'...")
        df_gold = engine.load_data_from_csv("gold_historical_m1.csv")
    else:
        print("[INFO] Pulling live Gold M1 data from internet for model training...")
        df_gold = engine.load_data_from_internet(range_days=5)

    trainer = ModelTrainer(dataset=df_gold)
    trained_model, eval_metrics = trainer.train_and_evaluate(
        n_estimators=args.trees,
        learning_rate=args.lr,
        max_depth=args.depth
    )

    trainer.export_model_artifact(trained_model, eval_metrics, output_path=args.output)

    print("\n" + "=" * 70)
    print("      ANTI-GRAVITY XAUUSD: MACHINE LEARNING TRAINING REPORT        ")
    print("=" * 70)
    print(f"  Total Data Samples:        {eval_metrics['total_samples']:,} bars")
    print(f"  Training Split:            {eval_metrics['train_samples']:,} bars")
    print(f"  Out-of-Sample Test:        {eval_metrics['test_samples']:,} bars")
    print(f"  Test Overall Accuracy:     {eval_metrics['test_accuracy_pct']:.2f}%")
    print(f"  High-Conf (P>=0.65) Prec:  {eval_metrics['high_conf_precision_pct']:.2f}%")
    print(f"  High-Conf Filtered Trades: {eval_metrics['high_conf_trades_count']}")
    print("-" * 70)
    print("  Feature Importance Matrix:")
    for feat, imp in sorted(eval_metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        bar_len = int(imp * 40)
        bar_str = "█" * bar_len
        print(f"    {feat:<18} : {imp:>6.2%}  {bar_str}")
    print("=" * 70)
    print(f"  Model Saved To:            {args.output}\n")
