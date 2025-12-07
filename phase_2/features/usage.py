# phase_2/features/usage.py

from __future__ import annotations

from typing import List

import pandas as pd

from ..db import read_sql


def _load_usage_source_data(season: int, week: int) -> pd.DataFrame:
    """
    Load player-level stats and snap data for all weeks up to and including
    the target week (season, week).

    We will later shift the rolling stats so that the *features* for week W
    only use data from weeks < W (no leakage).
    """
    # Core offensive stats (Sleeper)
    stats_query = """
        SELECT
            player_id,
            season,
            week,
            game_id,
            team_id,
            opponent_team_id,
            snaps,
            targets,
            receptions,
            carries,
            rush_yards,
            rec_yards,
            pass_attempts,
            pass_yards
        FROM player_game_stats
        WHERE season = ?
          AND week <= ?
    """

    stats = read_sql(stats_query, params=[season, week])

    # Snap counts table (nflreadpy-derived)
    snaps_query = """
        SELECT
            player_id,
            season,
            week,
            game_id,
            team,
            opponent,
            offense_snaps,
            offense_pct
        FROM snap_counts
        WHERE season = ?
          AND week <= ?
    """

    snaps = read_sql(snaps_query, params=[season, week])

    # Merge them; keep all stat rows and add snap info when available
    df = stats.merge(
        snaps[
            [
                "player_id",
                "season",
                "week",
                "game_id",
                "team",
                "opponent",
                "offense_snaps",
                "offense_pct",
            ]
        ],
        on=["player_id", "season", "week", "game_id"],
        how="left",
        suffixes=("", "_snap"),
    )

    # Fallback: if team/opponent missing from snap_counts, we still at least
    # carry the team_id/opponent_team_id for later joins.
    if "team" not in df.columns:
        df["team"] = pd.NA
    if "opponent" not in df.columns:
        df["opponent"] = pd.NA

    return df


def _add_rolling_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of player-week stats (including the target week),
    compute rolling usage features per player and shift them so that the
    row for week W only uses information from weeks < W.

    This function does NOT filter to a specific week; it just adds columns.
    """

    # Sort and reset index so everything is nicely ordered
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    group = df.groupby(["player_id", "season"], group_keys=False)

    feature_specs = [
        # (source_column, feature_prefix, window)
        ("offense_pct", "usage_snap_pct", 3),
        ("targets", "usage_targets", 3),
        ("carries", "usage_carries", 3),
        ("snaps", "usage_snaps_raw", 3),
    ]

    for source_col, prefix, window in feature_specs:
        if source_col not in df.columns:
            # Column may be missing for some positions; create it so downstream
            # code always has something to operate on.
            df[source_col] = pd.NA

        # Previous game value (week-1)
        df[f"{prefix}_last1"] = group[source_col].shift(1)

        # Rolling mean of last N games, anti-leakage via shift(1)
        df[f"{prefix}_last{window}"] = (
            group[source_col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            .shift(1)
        )

        # Season-to-date mean up to previous week
        df[f"{prefix}_season_to_date"] = (
            group[source_col]
            .transform(lambda s: s.expanding(min_periods=1).mean())
            .shift(1)
        )

        # Number of prior games with non-null data
        df[f"{prefix}_games_played_before"] = (
            group[source_col]
            .transform(lambda s: s.notna().cumsum())
            .shift(1)
        )

    return df


def build_usage_features(season: int, week: int) -> pd.DataFrame:
    """
    Build pre-game usage features for all (player, game) rows in the
    given (season, week).

    Returns a DataFrame with:
        - player_id
        - game_id
        - season
        - week
        - team
        - opponent
        - usage_* feature columns

    All usage_* columns for week W are computed using ONLY weeks < W.
    """

    # First, figure out which player-game rows we actually care about:
    # any offensive player who recorded stats in that week.
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
          AND week = ?
    """
    base = read_sql(base_query, params=[season, week])

    if base.empty:
        # No games (e.g., out-of-range week). Return empty DF with the right columns.
        return pd.DataFrame(
            columns=[
                "player_id",
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
            ]
        )

    # Load source data up through this week (for rolling calculations)
    df_all = _load_usage_source_data(season=season, week=week)

    # Add rolling anti-leakage features
    df_all = _add_rolling_usage_features(df_all)

    # Now keep only the rows for the target week
    df_week = df_all[df_all["week"] == week].copy()

    # Join to base to make sure we keep only players that actually appear
    # in player_game_stats for this week (and to preserve any FK columns)
    merged = base.merge(
        df_week,
        on=["player_id", "game_id", "season", "week"],
        how="left",
        suffixes=("", "_y"),
    )

    # Clean up: keep a tidy set of columns to feed into the master table.
    id_cols: List[str] = [
        "player_id",
        "game_id",
        "season",
        "week",
    ]

    # Prefer the "team" / "opponent" columns from df_week when present;
    # fall back to team_id/opponent_team_id.
    if "team" in merged.columns:
        merged["team"] = merged["team"].fillna(merged.get("team_id"))
    else:
        merged["team"] = merged.get("team_id")

    if "opponent" in merged.columns:
        merged["opponent"] = merged["opponent"].fillna(merged.get("opponent_team_id"))
    else:
        merged["opponent"] = merged.get("opponent_team_id")

    context_cols = ["team", "opponent"]

    # Any column that starts with "usage_" is a feature
    usage_feature_cols = [c for c in merged.columns if c.startswith("usage_")]

    final_cols = id_cols + context_cols + usage_feature_cols

    # Make sure we only return columns that actually exist (defensive programming)
    final_cols = [c for c in final_cols if c in merged.columns]

    return merged[final_cols]


if __name__ == "__main__":
    # Tiny smoke test: adjust season/week to something that exists in your DB.
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_test = build_usage_features(TEST_SEASON, TEST_WEEK)
    print("Usage features shape:", df_test.shape)
    print(df_test.head())
