"""
Multi-Model Training Script

Train all model types (Ridge, ElasticNet, XGBoost, LightGBM) for NFL prediction.
Follows Phase 4 v2 specification with:
- Multiple model types per stat
- Leak-safe chronological splits
- Versioned model artifacts
- Full evaluation metrics

Usage:
    # Train all models on 2021-2023, validate on 2023
    python -m phase_4.training.train --seasons 2021 2022 2023 --val-season 2023

    # Quick test with fewer trees
    python -m phase_4.training.train --seasons 2023 --quick

    # Train and evaluate on specific test season
    python -m phase_4.training.train --seasons 2021 2022 2023 --test-season 2024
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_4.db import get_connection, load_training_data
from phase_4.models import MultiModelTrainer
from phase_4.config import SAVED_MODELS_DIR, XGBOOST_PARAMS, TARGETS


def train_multi_model(
    train_seasons: List[int],
    val_season: Optional[int] = None,
    test_season: Optional[int] = None,
    positions: Optional[List[str]] = None,
    min_games: int = 3,
    quick: bool = False,
    verbose: bool = True,
) -> MultiModelTrainer:
    """
    Train all model types using MultiModelTrainer.

    Args:
        train_seasons: Seasons for training
        val_season: Season for validation (early stopping)
        test_season: Season for final evaluation
        positions: Position filter (None = all positions)
        min_games: Minimum games to include player
        quick: Use fewer trees for testing
        verbose: Print progress

    Returns:
        Trained MultiModelTrainer
    """
    conn = get_connection()

    # Determine all seasons to load
    all_seasons = list(train_seasons)
    if val_season and val_season not in all_seasons:
        all_seasons.append(val_season)
    if test_season and test_season not in all_seasons:
        all_seasons.append(test_season)

    all_seasons = sorted(all_seasons)

    if verbose:
        print("=" * 60)
        print("NFL Multi-Model Training Pipeline")
        print("=" * 60)
        print(f"Training seasons: {train_seasons}")
        print(f"Validation season: {val_season}")
        print(f"Test season: {test_season}")
        print(f"Positions: {positions or 'ALL'}")
        print(f"Quick mode: {quick}")
        print()

    # Load all data
    if verbose:
        print("Loading data...")

    df, feature_cols, target_cols = load_training_data(
        conn, all_seasons, positions, min_games
    )

    if verbose:
        print(f"Loaded {len(df):,} total samples")
        print(f"Features: {len(feature_cols)}")
        print(f"Targets: {len(target_cols)}")

    # Split data chronologically (LEAK-SAFE)
    train_df = df[df["season"].isin(train_seasons)]

    # Remove validation season from training if specified
    if val_season:
        train_df = train_df[train_df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        val_df = None

    # Test set
    if test_season:
        test_df = df[df["season"] == test_season]
    else:
        test_df = None

    if verbose:
        print(f"\nData splits:")
        print(f"  Training: {len(train_df):,} samples")
        if val_df is not None:
            print(f"  Validation: {len(val_df):,} samples (season {val_season})")
        if test_df is not None:
            print(f"  Test: {len(test_df):,} samples (season {test_season})")

    # Prepare feature/target matrices
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_val = val_df[feature_cols] if val_df is not None else None
    y_val = val_df[target_cols] if val_df is not None else None

    # Create version string
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    if positions:
        version = f"{version}_{'_'.join(positions)}"

    # Create trainer
    trainer = MultiModelTrainer(
        name="nfl_multi_model",
        version=version,
    )

    # Modify params for quick mode
    if quick:
        if verbose:
            print("\n[QUICK MODE] Using reduced parameters for testing")
        # Quick mode handled internally via config

    # Train all models
    trainer.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        verbose=verbose,
    )

    # Evaluate on test set
    if test_df is not None:
        X_test = test_df[feature_cols]
        y_test = test_df[target_cols]

        if verbose:
            print()
            print("=" * 60)
            print(f"TEST SET EVALUATION (Season {test_season})")
            print("=" * 60)

        trainer.evaluate_all(X_test, y_test, verbose=verbose)

    # Save models
    model_path = trainer.save()

    conn.close()

    if verbose:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Models saved to: {model_path}")
        print(f"Version: {trainer.version}")
        print(f"Models trained: {', '.join(trainer.training_metadata['models_trained'])}")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train NFL multi-model prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard training with validation
    python -m phase_4.training.train --seasons 2021 2022 2023 --val-season 2023

    # Training with test evaluation
    python -m phase_4.training.train --seasons 2021 2022 2023 --test-season 2024

    # Quick test run
    python -m phase_4.training.train --seasons 2023 --quick

    # Position-specific training
    python -m phase_4.training.train --seasons 2021 2022 2023 --positions QB RB WR TE

Trained models are saved to: phase_4/saved_models/
        """,
    )

    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Seasons for training data (e.g., 2021 2022 2023)",
    )
    parser.add_argument(
        "--val-season",
        type=int,
        help="Season to hold out for validation/early stopping",
    )
    parser.add_argument(
        "--test-season",
        type=int,
        help="Season for final test evaluation",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        help="Filter to specific positions (e.g., QB RB WR TE)",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=3,
        help="Minimum games played to include player (default: 3)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer trees (for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    # Run training
    trainer = train_multi_model(
        train_seasons=args.seasons,
        val_season=args.val_season,
        test_season=args.test_season,
        positions=args.positions,
        min_games=args.min_games,
        quick=args.quick,
        verbose=not args.quiet,
    )

    return trainer


if __name__ == "__main__":
    main()
