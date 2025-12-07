"""
Colab Betting Module

Functions for analyzing betting lines, player props, and finding value bets.
"""

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from . import data
from . import train
from . import predict


# Common prop types and their stat mappings
PROP_TYPES = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "interceptions": "interceptions",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    "receptions": "receptions",
    "targets": "targets",
    "completions": "pass_completions",
    "pass_attempts": "pass_attempts",
}


def get_game_lines(
    season: int,
    week: int,
    team: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get betting lines for games.

    Args:
        season: Season year
        week: Week number
        team: Filter by team
        verbose: Print summary

    Returns:
        DataFrame with game betting lines
    """
    lines = data.get_betting_lines(season, week, team)

    if verbose and len(lines) > 0:
        print(f"\n{'=' * 60}")
        print(f"GAME LINES - Week {week}")
        print(f"{'=' * 60}")
        display_cols = [c for c in lines.columns if c in [
            "home_team", "away_team", "spread", "total", "home_ml", "away_ml"
        ]]
        if display_cols:
            print(lines[display_cols].to_string(index=False))
        else:
            print(lines.head().to_string())

    return lines


def analyze_prop(
    player_name: str,
    prop_type: str,
    line: float,
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze a player prop bet.

    Args:
        player_name: Player name
        prop_type: Prop type (passing_yards, rushing_yards, receptions, etc.)
        line: The betting line
        season: Season year
        week: Week number
        model: Trained model
        verbose: Print analysis

    Returns:
        Dictionary with prop analysis

    Example:
        analyze_prop("CeeDee Lamb", "receiving_yards", 85.5, 2024, 14)
        analyze_prop("Josh Allen", "passing_yards", 250.5, 2024, 14)
    """
    # Map prop type to stat
    stat = PROP_TYPES.get(prop_type.lower(), prop_type.lower())

    # Get player prediction
    player_pred = predict.predict_player(player_name, season, week, model, verbose=False)

    if stat not in player_pred["predictions"]:
        raise ValueError(f"Stat '{stat}' not available for {player_name}")

    predicted_value = player_pred["predictions"][stat]
    diff = predicted_value - line
    diff_pct = (diff / line) * 100 if line != 0 else 0

    # Determine recommendation
    if diff > 0:
        recommendation = "OVER"
        edge = diff
    else:
        recommendation = "UNDER"
        edge = -diff

    # Confidence based on edge size
    if abs(diff_pct) > 20:
        confidence = "HIGH"
    elif abs(diff_pct) > 10:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    result = {
        "player": player_pred["player_name"],
        "position": player_pred["position"],
        "team": player_pred["team"],
        "opponent": player_pred["opponent"],
        "prop_type": prop_type,
        "line": line,
        "prediction": predicted_value,
        "difference": round(diff, 2),
        "difference_pct": round(diff_pct, 1),
        "recommendation": recommendation,
        "confidence": confidence,
        "edge": round(edge, 2)
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"PROP ANALYSIS: {result['player']}")
        print(f"{'=' * 50}")
        print(f"Prop: {prop_type.upper()} {line}")
        print(f"Prediction: {predicted_value}")
        print(f"Difference: {diff:+.1f} ({diff_pct:+.1f}%)")
        print(f"\n>>> {recommendation} ({confidence} confidence)")

    return result


def find_value_props(
    season: int,
    week: int,
    prop_lines: pd.DataFrame,
    model: train.NFLMultiModelTrainer = None,
    min_edge_pct: float = 10.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Find value in player prop markets.

    Args:
        season: Season year
        week: Week number
        prop_lines: DataFrame with columns [player_name, prop_type, line]
        model: Trained model
        min_edge_pct: Minimum edge percentage to flag
        verbose: Print results

    Returns:
        DataFrame with value props

    Example:
        props = pd.DataFrame({
            'player_name': ['CeeDee Lamb', 'Josh Allen', 'Saquon Barkley'],
            'prop_type': ['receiving_yards', 'passing_yards', 'rushing_yards'],
            'line': [85.5, 250.5, 75.5]
        })
        values = find_value_props(2024, 14, props)
    """
    results = []

    for _, row in prop_lines.iterrows():
        try:
            analysis = analyze_prop(
                row['player_name'],
                row['prop_type'],
                row['line'],
                season, week, model,
                verbose=False
            )
            if abs(analysis['difference_pct']) >= min_edge_pct:
                results.append(analysis)
        except Exception as e:
            if verbose:
                print(f"Skipping {row['player_name']}: {e}")

    if not results:
        print("No value props found meeting criteria")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("difference_pct", key=abs, ascending=False)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"VALUE PROPS - Week {week} (Min {min_edge_pct}% edge)")
        print(f"{'=' * 60}")
        display_cols = ["player", "prop_type", "line", "prediction", "difference_pct", "recommendation", "confidence"]
        print(df[display_cols].to_string(index=False))

    return df


def analyze_receiving_props(
    season: int,
    week: int,
    yards_lines: Dict[str, float] = None,
    receptions_lines: Dict[str, float] = None,
    model: train.NFLMultiModelTrainer = None,
    positions: List[str] = None,
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze receiving props (yards and receptions).

    Args:
        season: Season year
        week: Week number
        yards_lines: Dict of player_name -> yards line (optional)
        receptions_lines: Dict of player_name -> receptions line (optional)
        model: Trained model
        positions: Filter positions (default: WR, TE, RB)
        top_n: Number of players to show
        verbose: Print analysis

    Returns:
        DataFrame with receiving prop analysis

    Example:
        # Get predictions for top receivers
        analyze_receiving_props(2024, 14)

        # Compare against lines
        analyze_receiving_props(2024, 14,
            yards_lines={"CeeDee Lamb": 85.5, "Ja'Marr Chase": 75.5})
    """
    if positions is None:
        positions = ["WR", "TE", "RB"]

    # Get predictions
    preds = predict.predict_week(season, week, model, positions, verbose=False)

    # Add receiving analysis
    analysis = preds[[
        "player_name", "position", "team", "opponent",
        "pred_ensemble_targets", "pred_ensemble_receptions", "pred_ensemble_rec_yards", "pred_ensemble_rec_tds"
    ]].copy()

    analysis.columns = ["Player", "Pos", "Team", "Opp", "Targets", "Receptions", "Rec Yards", "Rec TDs"]

    # Sort by yards
    analysis = analysis.sort_values("Rec Yards", ascending=False).head(top_n)

    # Add line comparisons if provided
    if yards_lines:
        analysis["Yards Line"] = analysis["Player"].map(yards_lines)
        analysis["Yards Edge"] = analysis["Rec Yards"] - analysis["Yards Line"]

    if receptions_lines:
        analysis["Rec Line"] = analysis["Player"].map(receptions_lines)
        analysis["Rec Edge"] = analysis["Receptions"] - analysis["Rec Line"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RECEIVING PROPS ANALYSIS - Week {week}")
        print(f"{'=' * 70}")
        print(analysis.round(1).to_string(index=False))

    return analysis


def analyze_rushing_props(
    season: int,
    week: int,
    yards_lines: Dict[str, float] = None,
    model: train.NFLMultiModelTrainer = None,
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze rushing props.

    Args:
        season: Season year
        week: Week number
        yards_lines: Dict of player_name -> yards line (optional)
        model: Trained model
        top_n: Number of players to show
        verbose: Print analysis

    Returns:
        DataFrame with rushing prop analysis

    Example:
        analyze_rushing_props(2024, 14)
    """
    # Get predictions for RBs
    preds = predict.predict_week(season, week, model, positions=["RB"], verbose=False)

    # Add rushing analysis
    analysis = preds[[
        "player_name", "position", "team", "opponent",
        "pred_ensemble_carries", "pred_ensemble_rush_yards", "pred_ensemble_rush_tds"
    ]].copy()

    analysis.columns = ["Player", "Pos", "Team", "Opp", "Carries", "Rush Yards", "Rush TDs"]

    # Sort by yards
    analysis = analysis.sort_values("Rush Yards", ascending=False).head(top_n)

    # Add line comparisons if provided
    if yards_lines:
        analysis["Yards Line"] = analysis["Player"].map(yards_lines)
        analysis["Yards Edge"] = analysis["Rush Yards"] - analysis["Yards Line"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RUSHING PROPS ANALYSIS - Week {week}")
        print(f"{'=' * 70}")
        print(analysis.round(1).to_string(index=False))

    return analysis


def analyze_passing_props(
    season: int,
    week: int,
    yards_lines: Dict[str, float] = None,
    tds_lines: Dict[str, float] = None,
    model: train.NFLMultiModelTrainer = None,
    top_n: int = 15,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze passing props.

    Args:
        season: Season year
        week: Week number
        yards_lines: Dict of player_name -> yards line (optional)
        tds_lines: Dict of player_name -> TDs line (optional)
        model: Trained model
        top_n: Number of QBs to show
        verbose: Print analysis

    Returns:
        DataFrame with passing prop analysis

    Example:
        analyze_passing_props(2024, 14)
    """
    # Get predictions for QBs
    preds = predict.predict_week(season, week, model, positions=["QB"], verbose=False)

    # Add passing analysis
    analysis = preds[[
        "player_name", "position", "team", "opponent",
        "pred_ensemble_pass_attempts", "pred_ensemble_pass_completions",
        "pred_ensemble_pass_yards", "pred_ensemble_pass_tds", "pred_ensemble_interceptions"
    ]].copy()

    analysis.columns = ["Player", "Pos", "Team", "Opp", "Att", "Comp", "Pass Yards", "Pass TDs", "INTs"]

    # Sort by yards
    analysis = analysis.sort_values("Pass Yards", ascending=False).head(top_n)

    # Add line comparisons if provided
    if yards_lines:
        analysis["Yards Line"] = analysis["Player"].map(yards_lines)
        analysis["Yards Edge"] = analysis["Pass Yards"] - analysis["Yards Line"]

    if tds_lines:
        analysis["TDs Line"] = analysis["Player"].map(tds_lines)
        analysis["TDs Edge"] = analysis["Pass TDs"] - analysis["TDs Line"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PASSING PROPS ANALYSIS - Week {week}")
        print(f"{'=' * 70}")
        print(analysis.round(1).to_string(index=False))

    return analysis


def create_prop_sheet(
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create a comprehensive prop sheet for the week.

    Args:
        season: Season year
        week: Week number
        model: Trained model
        verbose: Print all sheets

    Returns:
        Dictionary with DataFrames for each prop category

    Example:
        sheets = create_prop_sheet(2024, 14)
        sheets['receiving']  # Top receiving projections
        sheets['rushing']    # Top rushing projections
        sheets['passing']    # Top passing projections
    """
    sheets = {
        "receiving": analyze_receiving_props(season, week, model=model, verbose=verbose),
        "rushing": analyze_rushing_props(season, week, model=model, verbose=verbose),
        "passing": analyze_passing_props(season, week, model=model, verbose=verbose),
    }

    return sheets


def anytime_td_scorers(
    season: int,
    week: int,
    model: train.NFLMultiModelTrainer = None,
    min_td_prob: float = 0.3,
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Find best anytime TD scorer candidates.

    Args:
        season: Season year
        week: Week number
        model: Trained model
        min_td_prob: Minimum combined TD probability
        top_n: Number of players to show
        verbose: Print results

    Returns:
        DataFrame with TD scorer analysis

    Example:
        tds = anytime_td_scorers(2024, 14)
    """
    # Get all predictions
    preds = predict.predict_week(season, week, model, verbose=False)

    # Calculate combined TD expectation
    preds["total_tds"] = (
        preds["pred_ensemble_rec_tds"].fillna(0) +
        preds["pred_ensemble_rush_tds"].fillna(0) +
        preds["pred_ensemble_pass_tds"].fillna(0) * 0.1  # Discount passing TDs for ATTD
    )

    # Filter and sort
    td_scorers = preds[preds["total_tds"] >= min_td_prob].copy()
    td_scorers = td_scorers.sort_values("total_tds", ascending=False).head(top_n)

    # Create analysis DataFrame
    analysis = td_scorers[[
        "player_name", "position", "team", "opponent",
        "pred_ensemble_rec_tds", "pred_ensemble_rush_tds", "total_tds"
    ]].copy()

    analysis.columns = ["Player", "Pos", "Team", "Opp", "Rec TDs", "Rush TDs", "Total TDs"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ANYTIME TD SCORERS - Week {week}")
        print(f"{'=' * 60}")
        print(analysis.round(2).to_string(index=False))

    return analysis
