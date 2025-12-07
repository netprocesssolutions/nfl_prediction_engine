#!/usr/bin/env python3
"""
Run Baseline Predictions - Phase 3

This script runs the baseline predictor to generate predictions
for specified seasons/weeks and saves them to the database.

Usage:
    python -m phase_3.pipeline.run_baseline --season 2024 --week 10
    python -m phase_3.pipeline.run_baseline --season 2024 --all-weeks
    python -m phase_3.pipeline.run_baseline --backtest 2023 2024
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_3.predictors.baseline import BaselinePredictor, calculate_fantasy_points
from phase_3.db import read_sql, get_connection


def get_available_weeks(season: int) -> list:
    """Get list of weeks with feature data for a season."""
    query = """
        SELECT DISTINCT week
        FROM player_game_features
        WHERE season = ?
        ORDER BY week
    """
    df = read_sql(query, [season])
    return df['week'].tolist()


def run_predictions(season: int, week: int = None,
                   save: bool = True) -> pd.DataFrame:
    """
    Run baseline predictions for a season/week.

    Args:
        season: Season year
        week: Specific week (None for all weeks)
        save: Whether to save to database

    Returns:
        DataFrame of predictions
    """
    predictor = BaselinePredictor()

    if week is not None:
        print(f"\n{'='*60}")
        print(f"Generating predictions for {season} Week {week}")
        print(f"{'='*60}")
        predictions = predictor.predict_week(season, week)
    else:
        print(f"\n{'='*60}")
        print(f"Generating predictions for {season} (all weeks)")
        print(f"{'='*60}")
        weeks = get_available_weeks(season)
        print(f"Found {len(weeks)} weeks with data")
        predictions = predictor.predict_season(season, min(weeks), max(weeks))

    if predictions.empty:
        print("No predictions generated")
        return predictions

    # Calculate fantasy points
    predictions['pred_fp_ppr'] = predictions.apply(
        lambda r: calculate_fantasy_points(r, 'ppr'), axis=1
    )
    predictions['pred_fp_half'] = predictions.apply(
        lambda r: calculate_fantasy_points(r, 'half_ppr'), axis=1
    )
    predictions['pred_fp_std'] = predictions.apply(
        lambda r: calculate_fantasy_points(r, 'standard'), axis=1
    )

    print(f"\nGenerated {len(predictions)} predictions")

    # Show sample
    print("\nSample predictions (top 10 by PPR points):")
    sample = predictions.nlargest(10, 'pred_fp_ppr')
    cols = ['player_name', 'position', 'team', 'week',
            'targets', 'receptions', 'rec_yards', 'carries',
            'rush_yards', 'pred_fp_ppr']
    cols = [c for c in cols if c in sample.columns]
    print(sample[cols].to_string(index=False))

    if save:
        predictor.save_predictions(predictions)

    return predictions


def run_backtest(seasons: list, save: bool = True) -> pd.DataFrame:
    """
    Run predictions for multiple seasons (backtesting).

    Args:
        seasons: List of seasons to predict
        save: Whether to save to database

    Returns:
        DataFrame of all predictions
    """
    all_predictions = []

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"Backtesting season {season}")
        print(f"{'='*60}")

        preds = run_predictions(season, week=None, save=False)
        if not preds.empty:
            all_predictions.append(preds)

    if not all_predictions:
        return pd.DataFrame()

    combined = pd.concat(all_predictions, ignore_index=True)

    if save:
        predictor = BaselinePredictor()
        predictor.save_predictions(combined)

    return combined


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    """
    Evaluate prediction accuracy against actual results.

    Args:
        predictions: DataFrame with predictions

    Returns:
        Dictionary of evaluation metrics
    """
    # Load actual results
    query = """
        SELECT
            player_id,
            game_id,
            label_targets as actual_targets,
            label_receptions as actual_receptions,
            label_rec_yards as actual_rec_yards,
            label_rec_tds as actual_rec_tds,
            label_carries as actual_carries,
            label_rush_yards as actual_rush_yards,
            label_rush_tds as actual_rush_tds,
            label_pass_attempts as actual_pass_attempts,
            label_pass_completions as actual_completions,
            label_pass_yards as actual_pass_yards,
            label_pass_tds as actual_pass_tds,
            label_interceptions as actual_interceptions
        FROM player_game_features
    """
    actuals = read_sql(query)

    # Merge predictions with actuals
    merged = predictions.merge(
        actuals,
        on=['player_id', 'game_id'],
        how='inner'
    )

    if merged.empty:
        print("No matching actual results found")
        return {}

    print(f"\nEvaluating {len(merged)} predictions with actual results")

    # Calculate metrics
    metrics = {}

    stat_pairs = [
        ('targets', 'actual_targets'),
        ('receptions', 'actual_receptions'),
        ('rec_yards', 'actual_rec_yards'),
        ('carries', 'actual_carries'),
        ('rush_yards', 'actual_rush_yards'),
        ('pass_yards', 'actual_pass_yards'),
    ]

    for pred_col, actual_col in stat_pairs:
        if pred_col not in merged.columns or actual_col not in merged.columns:
            continue

        # Filter to non-null values
        valid = merged[[pred_col, actual_col]].dropna()
        if len(valid) < 10:
            continue

        pred = valid[pred_col]
        actual = valid[actual_col]

        # Mean Absolute Error
        mae = (pred - actual).abs().mean()

        # Root Mean Square Error
        rmse = ((pred - actual) ** 2).mean() ** 0.5

        # Correlation
        corr = pred.corr(actual)

        # Mean Bias
        bias = (pred - actual).mean()

        metrics[pred_col] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'correlation': round(corr, 3),
            'bias': round(bias, 2),
            'n': len(valid)
        }

    # Calculate fantasy points MAE
    # First calculate actual fantasy points
    merged['actual_fp_ppr'] = (
        merged['actual_pass_yards'].fillna(0) * 0.04 +
        merged['actual_pass_tds'].fillna(0) * 4 +
        merged['actual_interceptions'].fillna(0) * -2 +
        merged['actual_rush_yards'].fillna(0) * 0.1 +
        merged['actual_rush_tds'].fillna(0) * 6 +
        merged['actual_rec_yards'].fillna(0) * 0.1 +
        merged['actual_rec_tds'].fillna(0) * 6 +
        merged['actual_receptions'].fillna(0) * 1.0
    )

    if 'pred_fp_ppr' in merged.columns:
        fp_mae = (merged['pred_fp_ppr'] - merged['actual_fp_ppr']).abs().mean()
        fp_corr = merged['pred_fp_ppr'].corr(merged['actual_fp_ppr'])
        metrics['fantasy_points_ppr'] = {
            'mae': round(fp_mae, 2),
            'correlation': round(fp_corr, 3),
            'n': len(merged)
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline predictions for NFL fantasy"
    )
    parser.add_argument('--season', type=int, help='Season to predict')
    parser.add_argument('--week', type=int, help='Specific week (optional)')
    parser.add_argument('--all-weeks', action='store_true',
                       help='Predict all weeks in season')
    parser.add_argument('--backtest', type=int, nargs='+',
                       help='Backtest multiple seasons')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions against actuals')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to database')

    args = parser.parse_args()

    save = not args.no_save

    if args.backtest:
        predictions = run_backtest(args.backtest, save=save)
    elif args.season:
        week = None if args.all_weeks else args.week
        predictions = run_predictions(args.season, week, save=save)
    else:
        # Default: predict latest available season
        query = "SELECT MAX(season) as latest FROM player_game_features"
        latest = read_sql(query)['latest'].iloc[0]
        print(f"No season specified, using latest: {latest}")
        predictions = run_predictions(latest, save=save)

    if args.evaluate and not predictions.empty:
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        metrics = evaluate_predictions(predictions)
        for stat, values in metrics.items():
            print(f"\n{stat}:")
            for metric, value in values.items():
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
