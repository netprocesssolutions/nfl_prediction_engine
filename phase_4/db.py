"""
Database utilities for Phase 4.
"""

import sqlite3
import pandas as pd
from typing import List, Optional, Tuple
from .config import DB_PATH, FEATURE_PREFIXES, EXCLUDE_COLUMNS, TARGETS


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_feature_columns(conn: sqlite3.Connection) -> List[str]:
    """Get list of feature columns from player_game_features table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(player_game_features)")
    all_columns = [row[1] for row in cursor.fetchall()]

    # Filter to feature columns only
    feature_cols = []
    for col in all_columns:
        # Skip excluded columns
        if col in EXCLUDE_COLUMNS:
            continue
        # Include if matches feature prefix
        if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES):
            feature_cols.append(col)

    return sorted(feature_cols)


def load_training_data(
    conn: sqlite3.Connection,
    seasons: List[int],
    positions: Optional[List[str]] = None,
    min_games: int = 3,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load training data for specified seasons.

    Args:
        conn: Database connection
        seasons: List of seasons to include
        positions: Optional list of positions to filter
        min_games: Minimum games played to include player

    Returns:
        Tuple of (DataFrame, feature_columns, target_columns)
    """
    feature_cols = get_feature_columns(conn)

    # Build query
    season_str = ",".join(str(s) for s in seasons)
    cols_to_select = (
        ["season", "week", "game_id", "player_id", "player_name", "position", "team"]
        + feature_cols
        + TARGETS
    )

    query = f"""
    SELECT {', '.join(cols_to_select)}
    FROM player_game_features
    WHERE season IN ({season_str})
    """

    if positions:
        pos_str = ",".join(f"'{p}'" for p in positions)
        query += f" AND position IN ({pos_str})"

    query += " ORDER BY season, week, player_id"

    df = pd.read_sql_query(query, conn)

    # Filter players with minimum games
    if min_games > 1:
        player_games = df.groupby("player_id").size()
        valid_players = player_games[player_games >= min_games].index
        df = df[df["player_id"].isin(valid_players)]

    return df, feature_cols, TARGETS


def load_prediction_data(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    positions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data for making predictions on a specific week.

    Args:
        conn: Database connection
        season: Season to predict
        week: Week to predict
        positions: Optional list of positions to filter

    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    feature_cols = get_feature_columns(conn)

    cols_to_select = (
        ["season", "week", "game_id", "player_id", "player_name", "position", "team", "opponent"]
        + feature_cols
    )

    query = f"""
    SELECT {', '.join(cols_to_select)}
    FROM player_game_features
    WHERE season = {season} AND week = {week}
    """

    if positions:
        pos_str = ",".join(f"'{p}'" for p in positions)
        query += f" AND position IN ({pos_str})"

    df = pd.read_sql_query(query, conn)

    return df, feature_cols


def save_predictions(
    conn: sqlite3.Connection,
    predictions_df: pd.DataFrame,
    model_version: str,
) -> int:
    """
    Save predictions to the database.

    Args:
        conn: Database connection
        predictions_df: DataFrame with predictions
        model_version: Version string for the model

    Returns:
        Number of rows inserted
    """
    predictions_df = predictions_df.copy()
    predictions_df["model_version"] = model_version

    # Insert into predictions table
    predictions_df.to_sql(
        "ml_predictions",
        conn,
        if_exists="append",
        index=False,
    )

    return len(predictions_df)
