"""
Play-by-Play Based Features

Extract premium features from play-by-play data:
- EPA (Expected Points Added) metrics
- Air yards and YAC metrics
- CPOE (Completion % Over Expected)
- Success rate
- Situational metrics (red zone, third down)

These features are SIGNIFICANTLY more predictive than box score stats.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "phase_1" / "database" / "nfl_data.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def calculate_pbp_features_for_player(
    conn: sqlite3.Connection,
    player_id: str,
    season: int,
    week: int,
    window: str = "season_to_date",
) -> Dict[str, float]:
    """
    Calculate PBP-based features for a single player.

    Args:
        conn: Database connection
        player_id: Player's GSIS ID
        season: Current season
        week: Current week (features use data BEFORE this week)
        window: 'last1', 'last3', or 'season_to_date'

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Determine week range based on window
    if window == "last1":
        week_filter = f"week = {week - 1}"
    elif window == "last3":
        week_filter = f"week BETWEEN {week - 3} AND {week - 1}"
    else:  # season_to_date
        week_filter = f"week < {week}"

    # === RECEIVING FEATURES ===
    rec_query = f"""
    SELECT
        COUNT(*) as targets,
        SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) as receptions,
        AVG(epa) as epa_per_target,
        SUM(epa) as total_epa,
        AVG(air_yards) as avg_air_yards,
        AVG(CASE WHEN complete_pass = 1 THEN yards_after_catch ELSE NULL END) as avg_yac,
        SUM(air_yards) as total_air_yards,
        AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END) as success_rate,
        SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END) as rec_tds,
        AVG(CASE WHEN cp IS NOT NULL THEN cp ELSE NULL END) as avg_catch_prob,
        SUM(CASE WHEN first_down = 1 THEN 1 ELSE 0 END) as first_downs
    FROM play_by_play
    WHERE receiver_player_id = ?
      AND season = ?
      AND {week_filter}
    """

    try:
        rec_df = pd.read_sql_query(rec_query, conn, params=[player_id, season])
        if len(rec_df) > 0 and rec_df["targets"].iloc[0] > 0:
            row = rec_df.iloc[0]
            features[f"pbp_rec_epa_per_target_{window}"] = row["epa_per_target"]
            features[f"pbp_rec_total_epa_{window}"] = row["total_epa"]
            features[f"pbp_avg_air_yards_{window}"] = row["avg_air_yards"]
            features[f"pbp_avg_yac_{window}"] = row["avg_yac"]
            features[f"pbp_rec_success_rate_{window}"] = row["success_rate"]
            features[f"pbp_avg_catch_prob_{window}"] = row["avg_catch_prob"]
            features[f"pbp_first_down_rate_{window}"] = (
                row["first_downs"] / row["targets"] if row["targets"] > 0 else 0
            )
    except Exception as e:
        pass

    # === RUSHING FEATURES ===
    rush_query = f"""
    SELECT
        COUNT(*) as carries,
        AVG(epa) as epa_per_carry,
        SUM(epa) as total_epa,
        AVG(yards_gained) as avg_yards,
        AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END) as success_rate,
        SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END) as rush_tds,
        SUM(CASE WHEN first_down = 1 THEN 1 ELSE 0 END) as first_downs,
        AVG(CASE WHEN shotgun = 1 THEN 1.0 ELSE 0.0 END) as shotgun_rate
    FROM play_by_play
    WHERE rusher_player_id = ?
      AND season = ?
      AND {week_filter}
      AND play_type = 'run'
    """

    try:
        rush_df = pd.read_sql_query(rush_query, conn, params=[player_id, season])
        if len(rush_df) > 0 and rush_df["carries"].iloc[0] > 0:
            row = rush_df.iloc[0]
            features[f"pbp_rush_epa_per_carry_{window}"] = row["epa_per_carry"]
            features[f"pbp_rush_total_epa_{window}"] = row["total_epa"]
            features[f"pbp_rush_success_rate_{window}"] = row["success_rate"]
            features[f"pbp_rush_first_down_rate_{window}"] = (
                row["first_downs"] / row["carries"] if row["carries"] > 0 else 0
            )
    except Exception as e:
        pass

    # === PASSING FEATURES (QB) ===
    pass_query = f"""
    SELECT
        COUNT(*) as attempts,
        SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) as completions,
        AVG(qb_epa) as qb_epa_per_play,
        SUM(qb_epa) as total_qb_epa,
        AVG(cpoe) as avg_cpoe,
        AVG(air_yards) as avg_intended_air_yards,
        AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END) as success_rate,
        SUM(CASE WHEN interception = 1 THEN 1 ELSE 0 END) as interceptions,
        SUM(CASE WHEN touchdown = 1 AND pass = 1 THEN 1 ELSE 0 END) as pass_tds,
        AVG(CASE WHEN complete_pass = 1 THEN yards_after_catch ELSE NULL END) as avg_yac_generated
    FROM play_by_play
    WHERE passer_player_id = ?
      AND season = ?
      AND {week_filter}
      AND pass = 1
    """

    try:
        pass_df = pd.read_sql_query(pass_query, conn, params=[player_id, season])
        if len(pass_df) > 0 and pass_df["attempts"].iloc[0] > 0:
            row = pass_df.iloc[0]
            features[f"pbp_qb_epa_per_play_{window}"] = row["qb_epa_per_play"]
            features[f"pbp_qb_total_epa_{window}"] = row["total_qb_epa"]
            features[f"pbp_cpoe_{window}"] = row["avg_cpoe"]
            features[f"pbp_avg_intended_air_yards_{window}"] = row["avg_intended_air_yards"]
            features[f"pbp_pass_success_rate_{window}"] = row["success_rate"]
            features[f"pbp_int_rate_{window}"] = (
                row["interceptions"] / row["attempts"] if row["attempts"] > 0 else 0
            )
    except Exception as e:
        pass

    return features


def calculate_all_pbp_features(
    conn: sqlite3.Connection,
    player_id: str,
    season: int,
    week: int,
) -> Dict[str, float]:
    """
    Calculate all PBP features for all windows.

    Returns features for last1, last3, and season_to_date.
    """
    all_features = {}

    for window in ["last1", "last3", "season_to_date"]:
        window_features = calculate_pbp_features_for_player(
            conn, player_id, season, week, window
        )
        all_features.update(window_features)

    return all_features


def build_pbp_features_for_season(
    season: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build PBP features for all players in a season.

    Args:
        season: Season to process
        verbose: Print progress

    Returns:
        DataFrame with PBP features for all player-games
    """
    conn = get_connection()

    # Get all player-games for the season
    query = """
    SELECT DISTINCT
        pgf.season, pgf.week, pgf.player_id, pgf.player_name, pgf.position
    FROM player_game_features pgf
    WHERE pgf.season = ?
    ORDER BY pgf.week, pgf.player_id
    """

    player_games = pd.read_sql_query(query, conn, params=[season])

    if verbose:
        print(f"Building PBP features for {len(player_games)} player-games in {season}")

    # Map Sleeper IDs to GSIS IDs
    id_mapping = pd.read_sql_query(
        "SELECT sleeper_id, gsis_id FROM player_id_mapping WHERE gsis_id IS NOT NULL",
        conn,
    )
    sleeper_to_gsis = dict(zip(id_mapping["sleeper_id"], id_mapping["gsis_id"]))

    results = []
    processed = 0

    for _, row in player_games.iterrows():
        player_id = row["player_id"]
        gsis_id = sleeper_to_gsis.get(str(player_id))

        if gsis_id:
            features = calculate_all_pbp_features(
                conn, gsis_id, row["season"], row["week"]
            )
        else:
            features = {}

        features["season"] = row["season"]
        features["week"] = row["week"]
        features["player_id"] = player_id

        results.append(features)

        processed += 1
        if verbose and processed % 500 == 0:
            print(f"  Processed {processed}/{len(player_games)}")

    conn.close()

    return pd.DataFrame(results)


def get_pbp_feature_columns() -> List[str]:
    """Get list of all PBP feature column names."""
    windows = ["last1", "last3", "season_to_date"]

    rec_features = [
        "pbp_rec_epa_per_target",
        "pbp_rec_total_epa",
        "pbp_avg_air_yards",
        "pbp_avg_yac",
        "pbp_rec_success_rate",
        "pbp_avg_catch_prob",
        "pbp_first_down_rate",
    ]

    rush_features = [
        "pbp_rush_epa_per_carry",
        "pbp_rush_total_epa",
        "pbp_rush_success_rate",
        "pbp_rush_first_down_rate",
    ]

    pass_features = [
        "pbp_qb_epa_per_play",
        "pbp_qb_total_epa",
        "pbp_cpoe",
        "pbp_avg_intended_air_yards",
        "pbp_pass_success_rate",
        "pbp_int_rate",
    ]

    all_features = []
    for base in rec_features + rush_features + pass_features:
        for window in windows:
            all_features.append(f"{base}_{window}")

    return all_features


if __name__ == "__main__":
    # Test
    conn = get_connection()

    # Test for a specific player (Ja'Marr Chase)
    test_id = "00-0036963"  # Chase's GSIS ID
    features = calculate_all_pbp_features(conn, test_id, 2024, 10)

    print("Sample PBP features for Ja'Marr Chase (Week 10):")
    for k, v in sorted(features.items()):
        if v is not None:
            print(f"  {k}: {v:.3f}")

    conn.close()
