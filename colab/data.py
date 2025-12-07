"""
Colab Data Module

Database utilities and data loading functions.
"""

import sqlite3
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
from . import setup


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(setup.get_db_path())


def query(sql: str, params: List[Any] = None) -> pd.DataFrame:
    """
    Execute SQL query and return DataFrame.

    Args:
        sql: SQL query string
        params: Query parameters

    Returns:
        DataFrame with results
    """
    conn = get_connection()
    if params:
        df = pd.read_sql_query(sql, conn, params=params)
    else:
        df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def get_tables() -> List[str]:
    """Get list of all tables in database."""
    conn = get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    return [t[0] for t in tables]


def get_table_schema(table: str) -> pd.DataFrame:
    """Get schema for a table."""
    return query(f"PRAGMA table_info({table})")


def get_seasons() -> List[int]:
    """Get list of available seasons."""
    df = query("SELECT DISTINCT season FROM player_game_features ORDER BY season")
    return df['season'].tolist()


def get_weeks(season: int) -> List[int]:
    """Get list of available weeks for a season."""
    df = query(
        "SELECT DISTINCT week FROM player_game_features WHERE season = ? ORDER BY week",
        [season]
    )
    return df['week'].tolist()


def get_players(
    season: int = None,
    week: int = None,
    position: str = None,
    team: str = None
) -> pd.DataFrame:
    """
    Get player list with optional filters.

    Args:
        season: Filter by season
        week: Filter by week
        position: Filter by position (QB, RB, WR, TE)
        team: Filter by team

    Returns:
        DataFrame with player info
    """
    sql = """
        SELECT DISTINCT player_id, player_name, position, team
        FROM player_game_features
        WHERE 1=1
    """
    params = []

    if season:
        sql += " AND season = ?"
        params.append(season)
    if week:
        sql += " AND week = ?"
        params.append(week)
    if position:
        sql += " AND position = ?"
        params.append(position)
    if team:
        sql += " AND team = ?"
        params.append(team)

    sql += " ORDER BY player_name"
    return query(sql, params)


def get_player_stats(
    player_id: str = None,
    player_name: str = None,
    season: int = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Get player game stats.

    Args:
        player_id: Player ID
        player_name: Player name (partial match)
        season: Filter by season
        limit: Max rows

    Returns:
        DataFrame with player stats
    """
    sql = """
        SELECT
            p.season, p.week, p.player_name, p.position, p.team, p.opponent,
            s.targets, s.receptions, s.rec_yards, s.rec_tds,
            s.carries, s.rush_yards, s.rush_tds,
            s.pass_attempts, s.pass_completions, s.pass_yards, s.pass_tds, s.interceptions
        FROM player_game_features p
        LEFT JOIN player_game_stats s ON p.player_id = s.player_id AND p.game_id = s.game_id
        WHERE 1=1
    """
    params = []

    if player_id:
        sql += " AND p.player_id = ?"
        params.append(player_id)
    if player_name:
        sql += " AND p.player_name LIKE ?"
        params.append(f"%{player_name}%")
    if season:
        sql += " AND p.season = ?"
        params.append(season)

    sql += " ORDER BY p.season DESC, p.week DESC"

    if limit:
        sql += f" LIMIT {limit}"

    return query(sql, params)


def get_training_data(
    seasons: List[int],
    positions: List[str] = None,
    min_games: int = 3
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load training data for ML models.

    Args:
        seasons: List of seasons to include
        positions: Filter by positions
        min_games: Minimum games for player inclusion

    Returns:
        Tuple of (DataFrame, feature_columns, target_columns)
    """
    conn = get_connection()

    # Get feature columns
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(player_game_features)")
    all_columns = [row[1] for row in cursor.fetchall()]

    # Feature prefixes
    feature_prefixes = ["usage_", "eff_", "oppdef_", "ngs_", "sched_", "pbp_"]

    # Exclude columns
    exclude_cols = [
        "season", "week", "game_id", "player_id", "player_name",
        "position", "team", "opponent", "home_team", "away_team",
        "is_home", "is_playoff"
    ]

    # Target columns
    target_cols = [
        "label_targets", "label_receptions", "label_rec_yards", "label_rec_tds",
        "label_carries", "label_rush_yards", "label_rush_tds",
        "label_pass_attempts", "label_pass_completions", "label_pass_yards",
        "label_pass_tds", "label_interceptions"
    ]

    exclude_cols += target_cols

    # Filter features
    feature_cols = []
    for col in all_columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in feature_prefixes):
            feature_cols.append(col)

    feature_cols = sorted(feature_cols)

    # Build query
    season_str = ",".join(str(s) for s in seasons)
    cols_to_select = (
        ["season", "week", "game_id", "player_id", "player_name", "position", "team"]
        + feature_cols + target_cols
    )

    sql = f"""
        SELECT {', '.join(cols_to_select)}
        FROM player_game_features
        WHERE season IN ({season_str})
    """

    if positions:
        pos_str = ",".join(f"'{p}'" for p in positions)
        sql += f" AND position IN ({pos_str})"

    sql += " ORDER BY season, week, player_id"

    df = pd.read_sql_query(sql, conn)

    # Filter by minimum games
    if min_games > 1:
        player_counts = df.groupby("player_id").size()
        valid_players = player_counts[player_counts >= min_games].index
        df = df[df["player_id"].isin(valid_players)]

    conn.close()

    return df, feature_cols, target_cols


def get_prediction_data(
    season: int,
    week: int,
    positions: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data for making predictions.

    Args:
        season: Season to predict
        week: Week to predict
        positions: Filter by positions

    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    conn = get_connection()

    # Get feature columns
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(player_game_features)")
    all_columns = [row[1] for row in cursor.fetchall()]

    feature_prefixes = ["usage_", "eff_", "oppdef_", "ngs_", "sched_", "pbp_"]
    exclude_cols = [
        "season", "week", "game_id", "player_id", "player_name",
        "position", "team", "opponent", "home_team", "away_team",
        "is_home", "is_playoff"
    ]

    # Also exclude labels
    exclude_cols += [c for c in all_columns if c.startswith("label_")]

    feature_cols = []
    for col in all_columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in feature_prefixes):
            feature_cols.append(col)

    feature_cols = sorted(feature_cols)

    cols_to_select = (
        ["season", "week", "game_id", "player_id", "player_name", "position", "team", "opponent"]
        + feature_cols
    )

    sql = f"""
        SELECT {', '.join(cols_to_select)}
        FROM player_game_features
        WHERE season = {season} AND week = {week}
    """

    if positions:
        pos_str = ",".join(f"'{p}'" for p in positions)
        sql += f" AND position IN ({pos_str})"

    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df, feature_cols


def get_betting_lines(
    season: int = None,
    week: int = None,
    team: str = None
) -> pd.DataFrame:
    """
    Get betting lines data.

    Args:
        season: Filter by season
        week: Filter by week
        team: Filter by team

    Returns:
        DataFrame with betting lines
    """
    sql = "SELECT * FROM betting_lines WHERE 1=1"
    params = []

    if season:
        sql += " AND season = ?"
        params.append(season)
    if week:
        sql += " AND week = ?"
        params.append(week)
    if team:
        sql += " AND (home_team = ? OR away_team = ?)"
        params.extend([team, team])

    sql += " ORDER BY season DESC, week DESC"

    return query(sql, params)


def get_predictions(
    prediction_type: str = "ml",
    season: int = None,
    week: int = None,
    player_name: str = None
) -> pd.DataFrame:
    """
    Get existing predictions.

    Args:
        prediction_type: "ml" or "baseline"
        season: Filter by season
        week: Filter by week
        player_name: Filter by player name

    Returns:
        DataFrame with predictions
    """
    table = "ml_predictions" if prediction_type == "ml" else "baseline_predictions"

    sql = f"SELECT * FROM {table} WHERE 1=1"
    params = []

    if season:
        sql += " AND season = ?"
        params.append(season)
    if week:
        sql += " AND week = ?"
        params.append(week)
    if player_name:
        sql += " AND player_name LIKE ?"
        params.append(f"%{player_name}%")

    sql += " ORDER BY season DESC, week DESC"

    return query(sql, params)
