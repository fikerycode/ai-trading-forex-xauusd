"""
optimizer.py
Institutional Strategy Parameter Optimizer and Grid Search for Anti-Gravity XAUUSD Scalper.

Explores parameter spaces across:
1. Alpha Engine: Wick Rejection Ratio, RVOL Sigma Threshold.
2. Machine Learning: Minimum Probability Gate, Maximum Entropy Threshold.
3. Risk & Lifecycle: Break-Even R Trigger, Partial Scale-Out R, Trailing ATR Multiplier.

Ranks candidates by Composite Institutional Objective Function:
    Score = Profit Factor * (1.0 - Max Drawdown %) * Sharpe Ratio Proxy
Exports:
    'optimization_results.csv' & 'optimized_parameters.json'
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity.optimizer")
    logger.success = logger.info

import numpy as np
import pandas as pd

from config import SystemConfig, CONFIG
from backtester import BacktestEngine, PerformanceMetrics


class StrategyOptimizer:
    """
    Automated Parameter Grid & Sensitivity Search Engine.
    """
    def __init__(
        self,
        historical_data: pd.DataFrame,
        base_config: Optional[SystemConfig] = None,
        initial_equity: float = 10000.0,
        risk_pct_per_trade: float = 0.01,
        min_trades_threshold: int = 15
    ):
        self.data = historical_data
        self.base_config = base_config or CONFIG
        self.initial_equity = initial_equity
        self.risk_pct = risk_pct_per_trade
        self.min_trades = min_trades_threshold
        self.results: List[Dict] = []

    def run_grid_search(
        self,
        param_grid: Dict[str, List],
        max_combinations: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Executes grid search over the Cartesian product of parameter lists.
        """
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        total_runs = len(combinations)

        if max_combinations and total_runs > max_combinations:
            logger.info(f"[OPTIMIZER] Subsampling {max_combinations} combinations out of {total_runs} total...")
            np.random.seed(42)
            indices = np.random.choice(total_runs, size=max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
            total_runs = max_combinations

        logger.info(f"[OPTIMIZER] Launching optimization across {total_runs} candidate parameter sets...")

        for idx, values in enumerate(combinations, 1):
            param_dict = dict(zip(keys, values))
            cfg = self._build_candidate_config(param_dict)

            # Instantiate engine with current candidate config
            engine = BacktestEngine(
                symbol="XAUUSD",
                initial_equity=self.initial_equity,
                risk_pct_per_trade=self.risk_pct,
                config=cfg
            )

            try:
                metrics: PerformanceMetrics = engine.run_simulation(self.data)
                score = self._compute_objective_score(metrics)

                record = {
                    "iteration": idx,
                    **param_dict,
                    "total_trades": metrics.total_trades,
                    "win_rate_pct": metrics.win_rate_pct,
                    "profit_factor": metrics.profit_factor,
                    "net_profit_usd": metrics.net_profit_usd,
                    "roi_pct": metrics.roi_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "score": round(score, 4)
                }
                self.results.append(record)

                if idx % 5 == 0 or idx == total_runs:
                    logger.info(
                        f"[{idx}/{total_runs}] PF: {metrics.profit_factor:.2f} | "
                        f"Net: ${metrics.net_profit_usd:+,.1f} | "
                        f"Trades: {metrics.total_trades} | "
                        f"DD: {metrics.max_drawdown_pct:.1f}% | "
                        f"Score: {score:.2f}"
                    )

            except Exception as e:
                logger.debug(f"[OPTIMIZER] Iteration {idx} failed: {e}")
                continue

        df_results = pd.DataFrame(self.results)
        if df_results.empty:
            logger.warning("[OPTIMIZER] No valid results produced.")
            return df_results

        df_results = df_results.sort_values(by="score", ascending=False)
        return df_results

    def _build_candidate_config(self, params: Dict) -> SystemConfig:
        """Deep copies base config and patches tuned parameters."""
        cfg = deepcopy(self.base_config)

        if "wick_ratio" in params:
            cfg.alpha.wick_to_body_ratio_min = float(params["wick_ratio"])
        if "rvol_sigma" in params:
            cfg.alpha.rvol_sigma_threshold = float(params["rvol_sigma"])
        if "min_prob" in params:
            cfg.ml.min_direction_probability = float(params["min_prob"])
        if "be_r" in params:
            cfg.risk.breakeven_r_trigger = float(params["be_r"])
        if "partial_r" in params:
            cfg.risk.partial_scale_out_r_trigger = float(params["partial_r"])
        if "trailing_atr" in params:
            cfg.risk.trailing_atr_multiple = float(params["trailing_atr"])

        return cfg

    def _compute_objective_score(self, m: PerformanceMetrics) -> float:
        """
        Calculates institutional multi-objective fitness score:
        - Rewards Profit Factor > 1.2
        - Penalizes High Max Drawdown (> 5%)
        - Requires statistical significance (min trades penalty)
        """
        if m.total_trades < self.min_trades:
            return -100.0 + m.total_trades

        # Sample size factor
        sample_penalty = min(m.total_trades / 30.0, 1.0)

        # Capital preservation factor
        dd_factor = max(0.01, 1.0 - (m.max_drawdown_pct / 100.0))

        # Profitability factor
        pf_factor = m.profit_factor if m.profit_factor > 0 else 0.1

        # Sharpe factor
        sharpe_bonus = max(0.1, m.sharpe_ratio + 5.0)

        score = pf_factor * dd_factor * sample_penalty * (1.0 + m.roi_pct / 100.0) * sharpe_bonus
        return float(score)

    def print_top_parameters(self, df_results: pd.DataFrame, top_n: int = 5):
        """Prints formatted ranking of top performing configurations."""
        if df_results.empty:
            return

        print("\n" + "=" * 85)
        print("          ANTI-GRAVITY XAUUSD SCALPER: TOP OPTIMIZED PARAMETER SETS         ")
        print("=" * 85)

        top_df = df_results.head(top_n)
        cols_to_show = [
            "iteration", "wick_ratio", "rvol_sigma", "be_r", "partial_r", "trailing_atr",
            "profit_factor", "win_rate_pct", "net_profit_usd", "max_drawdown_pct", "score"
        ]
        available_cols = [c for c in cols_to_show if c in top_df.columns]

        print(top_df[available_cols].to_string(index=False))
        print("=" * 85 + "\n")

    def save_best_config(self, df_results: pd.DataFrame, output_path: str = "optimized_parameters.json"):
        """Exports the winning parameter set to JSON if profitable and better than existing."""
        if df_results.empty:
            return

        profitable = df_results[df_results["net_profit_usd"] > 0]
        if profitable.empty:
            logger.warning("[OPTIMIZER] No profitable parameter sets found in this run. Retaining existing parameters.")
            return

        best = profitable.iloc[0].to_dict()
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    existing = json.load(f)
                if existing.get("net_profit_usd", 0) > best.get("net_profit_usd", 0):
                    logger.info("[OPTIMIZER] Existing parameters have higher net profit. Retaining existing.")
                    return
            except Exception:
                pass

        with open(output_path, "w") as f:
            json.dump(best, f, indent=4)
        logger.success(f"[OPTIMIZER] Best parameter configuration saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-Gravity XAUUSD Strategy Parameter Optimizer")
    parser.add_argument("--csv", type=str, default="gold_historical_m1.csv", help="Historical CSV path")
    parser.add_argument("--source", type=str, choices=["csv", "internet", "synthetic"], default="csv", help="Data source")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting backtest equity ($)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk fraction per trade (default: 0.01 = 1%%)")
    parser.add_argument("--max-evals", type=int, default=36, help="Maximum parameter combinations to evaluate")
    parser.add_argument("--export-csv", type=str, default="optimization_results.csv", help="Export CSV path")
    parser.add_argument("--save-best", type=str, default="optimized_parameters.json", help="Path to save best parameters")

    args = parser.parse_args()

    # 1. Load Dataset
    backtest_bootstrap = BacktestEngine(initial_equity=args.equity, risk_pct_per_trade=args.risk)

    if args.source == "csv" and os.path.exists(args.csv):
        print(f"[INFO] Loading optimization dataset from {args.csv}...")
        df_bars = backtest_bootstrap.load_data_from_csv(args.csv)
    elif args.source == "internet":
        df_bars = backtest_bootstrap.load_data_from_internet(range_days=5)
    else:
        if os.path.exists("gold_historical_m1.csv"):
            print(f"[INFO] Found local 'gold_historical_m1.csv', loading...")
            df_bars = backtest_bootstrap.load_data_from_csv("gold_historical_m1.csv")
        else:
            print(f"[INFO] Pulling live internet Gold M1 data for optimization...")
            df_bars = backtest_bootstrap.load_data_from_internet(range_days=5)

    # 2. Define Parameter Grid Search Space
    parameter_grid = {
        "wick_ratio": [0.60, 0.65, 0.72],           # Wick rejection exhaustion minimum
        "rvol_sigma": [1.8, 2.2, 2.8],              # Relative volume spike threshold
        "be_r": [0.8, 1.0],                          # Break-Even move trigger (R-multiple)
        "partial_r": [1.3, 1.6],                     # 50% Partial Take-Profit trigger (R-multiple)
        "trailing_atr": [1.0, 1.3]                   # Dynamic M1 ATR trailing stop multiplier
    }

    # Total grid = 3 * 3 * 2 * 2 * 2 = 72 configurations (subsampled to --max-evals if specified)
    optimizer = StrategyOptimizer(
        historical_data=df_bars,
        initial_equity=args.equity,
        risk_pct_per_trade=args.risk,
        min_trades_threshold=10
    )

    results_df = optimizer.run_grid_search(parameter_grid, max_combinations=args.max_evals)

    if not results_df.empty:
        optimizer.print_top_parameters(results_df, top_n=5)
        results_df.to_csv(args.export_csv, index=False)
        logger.success(f"[OPTIMIZER] Full optimization records saved -> {args.export_csv}")
        optimizer.save_best_config(results_df, output_path=args.save_best)
