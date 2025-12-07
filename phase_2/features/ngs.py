from __future__ import annotations

from typing import List, Tuple

import json
import numpy as np
import pandas as pd

from ..db import read_sql


def _load_player_gsis_mapping() -> pd.DataFrame:
    """
    Load mapping from internal player_id -> gsis_id from the players table.

    Returns a DataFrame with columns:
        - internal_player_id
        - gsis_id
    """
    df = read_sql(
        """
        SELECT
            player_id,
            metadata_json
        FROM players
        """
    )

    if df.empty:
        return pd.DataFrame(columns=["internal_player_id", "gsis_id"])

    def extract_gsis(meta: str) -> str | None:
        if meta is None:
            return None
        try:
            obj = json.loads(meta)
        except Exception:
            return None
        return obj.get("gsis_id")

    df["gsis_id"] = df["metadata_json"].apply(extract_gsis)
    mapping = (
        df[["player_id", "gsis_id"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"player_id": "internal_player_id"})
    )

    return mapping.reset_index(drop=True)


def _load_ngs_raw(season: int, week: int) -> pd.DataFrame:
    """
    Load raw NGS passing + receiving metrics for all weeks up to and including
    (season, week), and join to internal player_ids and player_game_stats so
    we have game_id / team context.

    Returns one row per (player_id, season, week, game_id).
    """
    mapping = _load_player_gsis_mapping()
    if mapping.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "season",
                "week",
                "game_id",
                "team_id",
                "opponent_team_id",
            ]
        )

    # Base player-game shell for join (all weeks up to target week)
    pgs = read_sql(
        """
        SELECT
            player_id,
            season,
            week,
            game_id,
            team_id,
            opponent_team_id
        FROM player_game_stats
        WHERE season = ?
          AND week   <= ?
        """,
        params=[season, week],
    )

    if pgs.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "season",
                "week",
                "game_id",
                "team_id",
                "opponent_team_id",
            ]
        )

    # --- Passing ---

    passing = read_sql(
        """
        SELECT
            player_id,
            season,
            week,
            avg_time_to_throw,
            avg_time_in_pocket,
            avg_completed_air_yards,
            avg_intended_air_yards,
            avg_air_yards_differential,
            avg_air_yards_to_sticks,
            aggressiveness,
            completion_percentage,
            expected_completion_percentage,
            completion_percentage_above_expectation,
            passer_rating
        FROM ngs_passing
        WHERE season = ?
          AND week   <= ?
        """,
        params=[season, week],
    )

    if not passing.empty:
        # player_id here is actually gsis_id
        passing = passing.rename(columns={"player_id": "gsis_id"})
        passing = passing.merge(mapping, on="gsis_id", how="left")
        passing = passing.merge(
            pgs,
            left_on=["internal_player_id", "season", "week"],
            right_on=["player_id", "season", "week"],
            how="inner",
            suffixes=("", "_pgs"),
        )
        # Use internal player_id from pgs
        passing = passing.rename(columns={"player_id": "player_id_internal"})
        passing["player_id"] = passing["internal_player_id"]
        # Keep only relevant columns
        keep_cols = [
            "player_id",
            "season",
            "week",
            "game_id",
            "team_id",
            "opponent_team_id",
            "avg_time_to_throw",
            "avg_time_in_pocket",
            "avg_completed_air_yards",
            "avg_intended_air_yards",
            "avg_air_yards_differential",
            "avg_air_yards_to_sticks",
            "aggressiveness",
            "completion_percentage",
            "expected_completion_percentage",
            "completion_percentage_above_expectation",
            "passer_rating",
        ]
        passing = passing[keep_cols].drop_duplicates()
    else:
        passing = None

    # --- Receiving ---

    receiving = read_sql(
        """
        SELECT
            player_id,
            season,
            week,
            avg_cushion,
            avg_separation,
            avg_intended_air_yards,
            percent_share_of_intended_air_yards,
            catch_percentage,
            avg_yac,
            avg_expected_yac,
            avg_yac_above_expectation
        FROM ngs_receiving
        WHERE season = ?
          AND week   <= ?
        """,
        params=[season, week],
    )

    if not receiving.empty:
        receiving = receiving.rename(columns={"player_id": "gsis_id"})
        receiving = receiving.merge(mapping, on="gsis_id", how="left")
        receiving = receiving.merge(
            pgs,
            left_on=["internal_player_id", "season", "week"],
            right_on=["player_id", "season", "week"],
            how="inner",
            suffixes=("", "_pgs"),
        )
        receiving = receiving.rename(columns={"player_id": "player_id_internal"})
        receiving["player_id"] = receiving["internal_player_id"]
        keep_cols = [
            "player_id",
            "season",
            "week",
            "game_id",
            "team_id",
            "opponent_team_id",
            "avg_cushion",
            "avg_separation",
            "avg_intended_air_yards",
            "percent_share_of_intended_air_yards",
            "catch_percentage",
            "avg_yac",
            "avg_expected_yac",
            "avg_yac_above_expectation",
        ]
        receiving = receiving[keep_cols].drop_duplicates()
    else:
        receiving = None

    # --- Combine passing + receiving ---

    if passing is None and receiving is None:
        return pd.DataFrame(
            columns=[
                "player_id",
                "season",
                "week",
                "game_id",
                "team_id",
                "opponent_team_id",
            ]
        )

    if passing is None:
        df = receiving
    elif receiving is None:
        df = passing
    else:
        df = passing.merge(
            receiving,
            on=["player_id", "season", "week", "game_id", "team_id", "opponent_team_id"],
            how="outer",
            suffixes=("_pass", "_recv"),
        )

    # Normalize numeric dtypes (just to be safe)
    for col in df.columns:
        if col.startswith(("avg_", "percent_", "completion_", "passer_rating")):
            df[col] = df[col].astype("float64")

    return df


def _add_ngs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given per-game NGS rows (with internal player_id + game_id), compute
    leak-safe rolling skill metrics per player-season.

    For each underlying metric, we create:
        ngs_<metric>_last1
        ngs_<metric>_last3
        ngs_<metric>_season_to_date
    """
    if df.empty:
        return df

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    group = df.groupby(["player_id", "season"], group_keys=False)

    # (feature_base_name, source_column_name)
    metric_specs: List[Tuple[str, str]] = [
        # Passing skill
        ("cpoe", "completion_percentage_above_expectation"),
        ("air_yards_to_sticks", "avg_air_yards_to_sticks"),
        ("air_yards_diff", "avg_air_yards_differential"),
        ("intended_air_yards", "avg_intended_air_yards"),
        ("completed_air_yards", "avg_completed_air_yards"),
        ("aggressiveness", "aggressiveness"),
        ("time_to_throw", "avg_time_to_throw"),
        ("time_in_pocket", "avg_time_in_pocket"),
        ("passer_rating", "passer_rating"),
        # Receiving skill
        ("rec_cushion", "avg_cushion"),
        ("rec_separation", "avg_separation"),
        ("rec_intended_air", "avg_intended_air_yards"),
        ("rec_share_intended_air", "percent_share_of_intended_air_yards"),
        ("rec_catch_pct", "catch_percentage"),
        ("rec_yac", "avg_yac"),
        ("rec_yac_oe", "avg_yac_above_expectation"),
    ]

    for base_name, src_col in metric_specs:
        if src_col not in df.columns:
            # Table might be missing this column for some seasons or modules;
            # skip gracefully.
            continue

        raw_col = f"_raw_ngs_{base_name}"
        df[raw_col] = df[src_col].astype("float64")

        # Last game
        df[f"ngs_{base_name}_last1"] = group[raw_col].shift(1)

        # Rolling last 3
        df[f"ngs_{base_name}_last3"] = (
            group[raw_col]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

        # Season-to-date mean
        df[f"ngs_{base_name}_season_to_date"] = (
            group[raw_col]
            .transform(lambda s: s.expanding(min_periods=1).mean())
            .shift(1)
        )

    return df


def build_ngs_features(season: int, week: int) -> pd.DataFrame:
    """
    Build leak-safe NGS skill/talent features for all (player, game) rows
    in the given (season, week).

    Returns:
        DataFrame with:
            - player_id, game_id, season, week
            - ngs_* feature columns
    """
    # Base: all players in this week
    base_query = """
        SELECT
            player_id,
            game_id,
            team_id,
            opponent_team_id,
            season,
            week
        FROM player_game_stats
        WHERE season = ?
          AND week   = ?
    """
    base = read_sql(base_query, params=[season, week])

    if base.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "game_id",
                "season",
                "week",
            ]
        )

    df_all = _load_ngs_raw(season=season, week=week)

    if df_all.empty:
        # No NGS data for this season/week; return only keys so merge is harmless.
        return pd.DataFrame(
            {
                "player_id": base["player_id"],
                "game_id": base["game_id"],
                "season": base["season"],
                "week": base["week"],
            }
        )

    df_all = _add_ngs_features(df_all)

    # Keep only rows for the target week
    df_week = df_all[df_all["week"] == week].copy()

    # Join to base so we only keep players that actually appear in this week
    merged = base.merge(
        df_week,
        on=["player_id", "game_id", "season", "week"],
        how="left",
        suffixes=("", "_ngs"),
    )

    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    feature_cols = [c for c in merged.columns if c.startswith("ngs_")]

    final_cols = [c for c in id_cols + feature_cols if c in merged.columns]

    return merged[final_cols]


if __name__ == "__main__":
    # Use a season/week that actually has NGS + player_game_stats overlap
    TEST_SEASON = 2025
    TEST_WEEK = 12

    df_test = build_ngs_features(TEST_SEASON, TEST_WEEK)
    print("NGS features shape:", df_test.shape)
    print(df_test.head())
