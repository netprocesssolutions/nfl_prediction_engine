"""
Model Evaluation Tools

Comprehensive evaluation of trained models including:
- Per-stat metrics (MAE, RMSE, R2, correlation)
- Fantasy point accuracy
- Position-specific analysis
- Model comparison

Usage:
    python -m phase_4.evaluation.evaluate --season 2024

    python -m phase_4.evaluation.evaluate --season 2024 --compare-baseline
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_4.db import get_connection, load_training_data
from phase_4.models import MultiModelTrainer
from phase_4.config import SAVED_MODELS_DIR, TARGETS, calculate_fantasy_points


def evaluate_models(
    season: int,
    model_path: Optional[Path] = None,
    positions: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate trained models on a specific season.

    Args:
        season: Season to evaluate on
        model_path: Path to trained model (uses latest if None)
        positions: Filter by positions
        verbose: Print results

    Returns:
        Dictionary of evaluation DataFrames
    """
    # Find model
    if model_path is None:
        model_dirs = [d for d in SAVED_MODELS_DIR.iterdir() if d.is_dir()]
        if not model_dirs:
            raise FileNotFoundError("No trained models found")
        model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1]

    if verbose:
        print(f"Evaluating model: {model_path.name}")
        print(f"Test season: {season}")

    # Load model
    trainer = MultiModelTrainer.load(model_path)

    # Load test data
    conn = get_connection()
    df, feature_cols, target_cols = load_training_data(conn, [season], positions, min_games=1)
    conn.close()

    if verbose:
        print(f"Test samples: {len(df):,}")

    X = df[feature_cols]
    y = df[target_cols]

    # Get predictions from all models
    all_preds = trainer.predict(X)

    # Evaluate each model
    results = {}

    model_names = trainer._get_trained_models_list()
    prefixes = {
        "ridge": "pred_ridge_",
        "elasticnet": "pred_elasticnet_",
        "xgboost": "pred_xgb_",
        "lightgbm": "pred_lgb_",
    }

    for model_name in model_names:
        prefix = prefixes.get(model_name, f"pred_{model_name}_")

        model_results = []
        for target in target_cols:
            stat = target.replace("label_", "")
            pred_col = f"{prefix}{stat}"

            if pred_col not in all_preds.columns:
                continue

            y_true = y[target].fillna(0).values
            y_pred = all_preds[pred_col].values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

            model_results.append({
                "stat": stat,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "correlation": corr,
            })

        results[model_name] = pd.DataFrame(model_results)

        if verbose:
            print(f"\n--- {model_name.upper()} ---")
            print(results[model_name].to_string(index=False))

    return results


def compare_models(
    season: int,
    model_path: Optional[Path] = None,
    include_baseline: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare all models side-by-side for each stat.

    Args:
        season: Season to evaluate
        model_path: Path to trained model
        include_baseline: Include Phase 3 baseline predictions
        verbose: Print results

    Returns:
        Comparison DataFrame
    """
    results = evaluate_models(season, model_path, verbose=False)

    # Pivot to comparison format
    comparison_rows = []

    stats = results[list(results.keys())[0]]["stat"].tolist()

    for stat in stats:
        row = {"stat": stat}

        for model_name, model_df in results.items():
            stat_row = model_df[model_df["stat"] == stat]
            if len(stat_row) > 0:
                row[f"{model_name}_mae"] = stat_row["mae"].values[0]
                row[f"{model_name}_corr"] = stat_row["correlation"].values[0]

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    if verbose:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(comparison.to_string(index=False))

        # Find best model per stat
        print("\n--- Best Model Per Stat (lowest MAE) ---")
        mae_cols = [c for c in comparison.columns if c.endswith("_mae")]
        for _, row in comparison.iterrows():
            stat = row["stat"]
            best_mae = float("inf")
            best_model = None
            for col in mae_cols:
                if row[col] < best_mae:
                    best_mae = row[col]
                    best_model = col.replace("_mae", "")
            print(f"{stat:20s} → {best_model} (MAE: {best_mae:.3f})")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NFL prediction models",
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season to evaluate on",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        help="Specific model version to evaluate",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        help="Filter by positions",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show model comparison",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    model_path = None
    if args.model_version:
        model_path = SAVED_MODELS_DIR / args.model_version

    if args.compare:
        compare_models(args.season, model_path, verbose=not args.quiet)
    else:
        evaluate_models(args.season, model_path, args.positions, verbose=not args.quiet)


if __name__ == "__main__":
    main()
