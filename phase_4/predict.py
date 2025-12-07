"""
Prediction Script for Weekly Use

Load trained models and generate predictions for upcoming games.
This is the script you run each week AFTER training is complete.

Usage:
    # Predict for a specific week
    python -m phase_4.predict --season 2024 --week 10

    # Predict and save to database
    python -m phase_4.predict --season 2024 --week 10 --save

    # Use specific model version
    python -m phase_4.predict --season 2024 --week 10 --model-version 20241207_123456

    # Export to CSV
    python -m phase_4.predict --season 2024 --week 10 --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_4.db import get_connection, load_prediction_data
from phase_4.models import MultiModelTrainer
from phase_4.config import SAVED_MODELS_DIR, calculate_fantasy_points


def find_latest_model() -> Path:
    """Find the most recently trained model."""
    model_dirs = [d for d in SAVED_MODELS_DIR.iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError(f"No trained models found in {SAVED_MODELS_DIR}")

    # Sort by modification time
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return model_dirs[0]


def predict_week(
    season: int,
    week: int,
    model_path: Optional[Path] = None,
    positions: Optional[List[str]] = None,
    scoring_type: str = "ppr",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for a specific week.

    Args:
        season: Season to predict
        week: Week to predict
        model_path: Path to model (uses latest if not specified)
        positions: Filter by positions
        scoring_type: Fantasy scoring type

    Returns:
        DataFrame with predictions
    """
    # Load model
    if model_path is None:
        model_path = find_latest_model()

    if verbose:
        print(f"Loading model from: {model_path}")

    trainer = MultiModelTrainer.load(model_path)

    if verbose:
        print(f"Model version: {trainer.version}")
        print(f"Models available: {trainer._get_trained_models_list()}")

    # Load prediction data
    conn = get_connection()
    df, feature_cols = load_prediction_data(conn, season, week, positions)
    conn.close()

    if len(df) == 0:
        raise ValueError(f"No data found for season {season} week {week}")

    if verbose:
        print(f"\nGenerating predictions for {len(df)} players...")

    # Get features
    X = df[feature_cols]

    # Get all model predictions
    all_preds = trainer.predict(X)

    # Get ensemble predictions with fantasy points
    ensemble_preds = trainer.predict_with_ensemble(X)
    fp_preds = trainer.predict_fantasy_points(X, scoring_type=scoring_type)

    # Combine with player info
    result = df[["season", "week", "game_id", "player_id", "player_name", "position", "team", "opponent"]].copy()
    result = pd.concat([result, all_preds, ensemble_preds, fp_preds[[f"pred_fp_{scoring_type}"]]], axis=1)

    # Sort by predicted fantasy points
    result = result.sort_values(f"pred_fp_{scoring_type}", ascending=False)

    if verbose:
        print(f"\nTop 10 Predictions ({scoring_type.upper()}):")
        print("-" * 60)
        top_cols = ["player_name", "position", "team", "opponent", f"pred_fp_{scoring_type}"]
        print(result[top_cols].head(10).to_string(index=False))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate NFL predictions using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict for week 10
    python -m phase_4.predict --season 2024 --week 10

    # Export to CSV
    python -m phase_4.predict --season 2024 --week 10 --output predictions.csv

    # Use specific model
    python -m phase_4.predict --season 2024 --week 10 --model-version nfl_multi_model_20241207

    # Filter by position
    python -m phase_4.predict --season 2024 --week 10 --positions WR RB
        """,
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season to predict",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week to predict",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        help="Specific model version to use (default: latest)",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        help="Filter by positions (e.g., QB RB WR TE)",
    )
    parser.add_argument(
        "--scoring",
        choices=["ppr", "half_ppr", "standard"],
        default="ppr",
        help="Fantasy scoring type (default: ppr)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save predictions to database",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    # Find model path
    model_path = None
    if args.model_version:
        model_path = SAVED_MODELS_DIR / args.model_version
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            print(f"Available models in {SAVED_MODELS_DIR}:")
            for d in SAVED_MODELS_DIR.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
            sys.exit(1)

    # Generate predictions
    predictions = predict_week(
        season=args.season,
        week=args.week,
        model_path=model_path,
        positions=args.positions,
        scoring_type=args.scoring,
        verbose=not args.quiet,
    )

    # Save to CSV if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")

    # Save to database if requested
    if args.save:
        conn = get_connection()
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                week INTEGER,
                game_id TEXT,
                player_id TEXT,
                player_name TEXT,
                position TEXT,
                team TEXT,
                opponent TEXT,
                pred_ridge_targets REAL,
                pred_xgb_targets REAL,
                pred_ensemble_targets REAL,
                pred_fp_ppr REAL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add model version
        predictions["model_version"] = model_path.name if model_path else "latest"

        # Select columns to save
        save_cols = [c for c in predictions.columns if c.startswith("pred_") or c in
                    ["season", "week", "game_id", "player_id", "player_name", "position", "team", "opponent", "model_version"]]

        predictions[save_cols].to_sql("ml_predictions", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()
        print(f"\nPredictions saved to database (ml_predictions table)")

    return predictions


if __name__ == "__main__":
    main()
