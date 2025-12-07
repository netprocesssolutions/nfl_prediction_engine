"""
Colab Prediction Module

Functions for generating predictions for any week, stat, or player.
"""

from typing import List, Optional, Dict, Union
from pathlib import Path

import pandas as pd
import numpy as np

from . import data
from . import train


# Fantasy scoring rules
SCORING_RULES = {
    "ppr": {
        "receptions": 1.0,
        "rec_yards": 0.1,
        "rec_tds": 6,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2
    },
    "half_ppr": {
        "receptions": 0.5,
        "rec_yards": 0.1,
        "rec_tds": 6,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2
    },
    "standard": {
        "receptions": 0.0,
        "rec_yards": 0.1,
        "rec_tds": 6,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2
    }
}


def predict_week(
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    positions: List[str] = None,
    scoring: str = "ppr",
    top_n: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate predictions for a specific week.

    Args:
        season: Season year
        week: Week number
        model: Trained model (loads latest if None)
        positions: Filter positions (e.g., ['QB', 'RB', 'WR', 'TE'])
        scoring: Fantasy scoring format
        top_n: Return only top N players by fantasy points
        verbose: Print summary

    Returns:
        DataFrame with predictions

    Example:
        preds = predict_week(2024, 14)
        preds = predict_week(2024, 14, positions=['WR', 'RB'])
    """
    # Load model if not provided
    if model is None:
        model = train.load_latest_model()

    # Load data
    df, feature_cols = data.get_prediction_data(season, week, positions)

    if len(df) == 0:
        raise ValueError(f"No data found for {season} week {week}")

    if verbose:
        print(f"Generating predictions for {len(df)} players...")

    # Get predictions
    X = df[feature_cols]
    all_preds = model.predict(X)
    ensemble_preds = model.predict_ensemble(X)
    fp_preds = model.predict_fantasy_points(X, scoring)

    # Combine with player info
    result = df[["season", "week", "game_id", "player_id", "player_name", "position", "team", "opponent"]].copy()
    result = pd.concat([result, all_preds, ensemble_preds, fp_preds], axis=1)

    # Sort by fantasy points
    fp_col = f"pred_fp_{scoring}"
    result = result.sort_values(fp_col, ascending=False)

    if top_n:
        result = result.head(top_n)

    if verbose:
        print(f"\nTop 10 Predictions ({scoring.upper()}):")
        print("-" * 70)
        display_cols = ["player_name", "position", "team", "opponent", fp_col]
        print(result[display_cols].head(10).to_string(index=False))

    return result


def predict_player(
    player_name: str,
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    verbose: bool = True
) -> Dict:
    """
    Generate predictions for a specific player.

    Args:
        player_name: Player name (partial match supported)
        season: Season year
        week: Week number
        model: Trained model (loads latest if None)
        verbose: Print summary

    Returns:
        Dictionary with player predictions

    Example:
        preds = predict_player("Justin Jefferson", 2024, 14)
    """
    # Get all predictions
    all_preds = predict_week(season, week, model, verbose=False)

    # Find player
    mask = all_preds["player_name"].str.contains(player_name, case=False, na=False)
    player_preds = all_preds[mask]

    if len(player_preds) == 0:
        raise ValueError(f"Player '{player_name}' not found in week {week}")

    if len(player_preds) > 1 and verbose:
        print(f"Found {len(player_preds)} matches. Showing first:")

    player = player_preds.iloc[0]

    result = {
        "player_name": player["player_name"],
        "position": player["position"],
        "team": player["team"],
        "opponent": player["opponent"],
        "season": season,
        "week": week,
        "predictions": {}
    }

    # Extract stat predictions
    stats = ["targets", "receptions", "rec_yards", "rec_tds",
             "carries", "rush_yards", "rush_tds",
             "pass_attempts", "pass_completions", "pass_yards", "pass_tds", "interceptions"]

    for stat in stats:
        col = f"pred_ensemble_{stat}"
        if col in player.index and pd.notna(player[col]):
            result["predictions"][stat] = round(player[col], 2)

    # Add fantasy points
    for scoring in ["ppr", "half_ppr", "standard"]:
        fp_col = f"pred_fp_{scoring}"
        if fp_col in player.index:
            result[f"fp_{scoring}"] = round(player[fp_col], 2)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"PREDICTION: {result['player_name']} ({result['position']})")
        print(f"Team: {result['team']} vs {result['opponent']}")
        print(f"{'=' * 50}")
        print("\nPredicted Stats:")
        for stat, val in result["predictions"].items():
            print(f"  {stat}: {val}")
        print(f"\nFantasy Points:")
        print(f"  PPR: {result.get('fp_ppr', 'N/A')}")
        print(f"  Half-PPR: {result.get('fp_half_ppr', 'N/A')}")
        print(f"  Standard: {result.get('fp_standard', 'N/A')}")

    return result


def predict_stat(
    stat: str,
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    positions: List[str] = None,
    top_n: int = 20,
    threshold: float = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get predictions for a specific stat.

    Args:
        stat: Stat to predict (targets, receptions, rec_yards, rec_tds,
              carries, rush_yards, rush_tds, pass_yards, pass_tds, etc.)
        season: Season year
        week: Week number
        model: Trained model
        positions: Filter positions
        top_n: Return top N players
        threshold: Only show players above this threshold
        verbose: Print summary

    Returns:
        DataFrame with stat predictions

    Example:
        rec_yards = predict_stat("rec_yards", 2024, 14, positions=['WR', 'TE'])
        rush_tds = predict_stat("rush_tds", 2024, 14, threshold=0.5)
    """
    # Get all predictions
    all_preds = predict_week(season, week, model, positions, verbose=False)

    # Find the stat column
    stat_col = f"pred_ensemble_{stat}"
    if stat_col not in all_preds.columns:
        available = [c.replace("pred_ensemble_", "") for c in all_preds.columns if c.startswith("pred_ensemble_")]
        raise ValueError(f"Stat '{stat}' not found. Available: {available}")

    # Sort by stat
    result = all_preds.sort_values(stat_col, ascending=False)

    # Apply threshold
    if threshold:
        result = result[result[stat_col] >= threshold]

    # Limit to top N
    if top_n:
        result = result.head(top_n)

    # Select relevant columns
    display_cols = ["player_name", "position", "team", "opponent", stat_col]
    result = result[display_cols].copy()
    result.columns = ["Player", "Pos", "Team", "Opp", f"Pred {stat}"]

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"TOP {len(result)} PREDICTIONS: {stat.upper()}")
        print(f"Season {season} Week {week}")
        print(f"{'=' * 50}")
        print(result.to_string(index=False))

    return result


def predict_position(
    position: str,
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    scoring: str = "ppr",
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get predictions for a specific position.

    Args:
        position: Position (QB, RB, WR, TE)
        season: Season year
        week: Week number
        model: Trained model
        scoring: Fantasy scoring format
        top_n: Return top N players
        verbose: Print summary

    Returns:
        DataFrame with position rankings

    Example:
        qbs = predict_position("QB", 2024, 14)
        rbs = predict_position("RB", 2024, 14, scoring="half_ppr")
    """
    return predict_week(
        season, week, model,
        positions=[position],
        scoring=scoring,
        top_n=top_n,
        verbose=verbose
    )


def compare_players(
    player_names: List[str],
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare predictions for multiple players.

    Args:
        player_names: List of player names to compare
        season: Season year
        week: Week number
        model: Trained model
        verbose: Print comparison

    Returns:
        DataFrame with player comparison

    Example:
        compare_players(["CeeDee Lamb", "Ja'Marr Chase", "Justin Jefferson"], 2024, 14)
    """
    # Get all predictions
    all_preds = predict_week(season, week, model, verbose=False)

    # Find each player
    results = []
    for name in player_names:
        mask = all_preds["player_name"].str.contains(name, case=False, na=False)
        matches = all_preds[mask]
        if len(matches) > 0:
            results.append(matches.iloc[0])
        else:
            print(f"Warning: '{name}' not found")

    if not results:
        raise ValueError("No players found")

    comparison = pd.DataFrame(results)

    # Select relevant columns
    stat_cols = [c for c in comparison.columns if c.startswith("pred_ensemble_")]
    fp_cols = [c for c in comparison.columns if c.startswith("pred_fp_")]
    display_cols = ["player_name", "position", "team", "opponent"] + stat_cols + fp_cols

    comparison = comparison[[c for c in display_cols if c in comparison.columns]]

    if verbose:
        print(f"\n{'=' * 60}")
        print("PLAYER COMPARISON")
        print(f"Season {season} Week {week}")
        print(f"{'=' * 60}")
        print(comparison.T.to_string())

    return comparison


def get_sleepers(
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    scoring: str = "ppr",
    ownership_threshold: float = None,
    min_fp: float = 10.0,
    positions: List[str] = None,
    top_n: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Find sleeper picks (high upside, lower owned players).

    Uses variance in model predictions as a proxy for upside.

    Args:
        season: Season year
        week: Week number
        model: Trained model
        scoring: Fantasy scoring format
        ownership_threshold: Max ownership % (if available)
        min_fp: Minimum predicted fantasy points
        positions: Filter positions
        top_n: Return top N sleepers
        verbose: Print results

    Returns:
        DataFrame with sleeper picks
    """
    # Get all predictions
    all_preds = predict_week(season, week, model, positions, verbose=False)

    fp_col = f"pred_fp_{scoring}"

    # Filter by minimum fantasy points
    sleepers = all_preds[all_preds[fp_col] >= min_fp].copy()

    # Calculate variance across model types as "upside" proxy
    fp_cols = ["pred_ridge_rec_yards", "pred_xgboost_rec_yards", "pred_lightgbm_rec_yards"]
    fp_cols = [c for c in fp_cols if c in sleepers.columns]

    if fp_cols:
        sleepers["upside_score"] = sleepers[fp_cols].std(axis=1)
    else:
        sleepers["upside_score"] = 0

    # Sort by upside score
    sleepers = sleepers.sort_values("upside_score", ascending=False).head(top_n)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"SLEEPER PICKS - Week {week}")
        print(f"(Minimum {min_fp} PPR points, sorted by upside)")
        print(f"{'=' * 50}")
        display_cols = ["player_name", "position", "team", "opponent", fp_col, "upside_score"]
        print(sleepers[display_cols].to_string(index=False))

    return sleepers


def calculate_fantasy_points(
    predictions: Dict[str, float],
    scoring: str = "ppr"
) -> float:
    """
    Calculate fantasy points from stat predictions.

    Args:
        predictions: Dict of stat predictions
        scoring: Scoring format (ppr, half_ppr, standard)

    Returns:
        Fantasy points
    """
    rules = SCORING_RULES.get(scoring, SCORING_RULES["ppr"])
    fp = 0.0

    for stat, multiplier in rules.items():
        if stat in predictions:
            fp += predictions[stat] * multiplier

    return round(fp, 2)
