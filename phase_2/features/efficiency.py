# phase_2/features/efficiency.py

from __future__ import annotations

from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

from ..db import read_sql
from ..player_id_mapping import ensure_mapping_exists, get_gsis_to_sleeper_map


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """
    Safe division that returns NaN when denom <= 0.
    """
    numer = numer.astype("float64")
    denom = denom.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, numer / denom, np.nan)
    return pd.Series(out, index=numer.index)


def _load_efficiency_source_data(season: int, week: int) -> pd.DataFrame:
    """
    Load weekly nflverse stats joined to player_game_stats for all weeks
    up to and including the target week.

    We include:
      - keys: player_id, season, week, game_id
      - position / team context
      - the raw stats needed to build EPA- and share-based efficiency metrics.

    Note: nflverse_weekly_stats uses GSIS IDs, player_game_stats uses Sleeper IDs.
    We use the player_id_mapping table to bridge these formats.
    """
    # Ensure the mapping table exists
    ensure_mapping_exists()

    # First, load nflverse data with GSIS IDs
    nflverse_query = """
        SELECT
            w.player_id as gsis_id,
            w.season,
            w.week,
            w.team,
            w.position,
            w.position_group,

            -- Passing
            w.completions,
            w.attempts,
            w.passing_yards,
            w.passing_tds,
            w.interceptions,
            w.sacks,

            -- Rushing
            w.carries,
            w.rushing_yards,
            w.rushing_tds,

            -- Receiving
            w.targets,
            w.receptions,
            w.receiving_yards,
            w.receiving_air_yards,
            w.receiving_yards_after_catch,

            -- Market share metrics
            w.target_share,
            w.air_yards_share,
            w.wopr,
            w.racr,

            -- EPA metrics
            w.passing_epa,
            w.rushing_epa,
            w.receiving_epa

        FROM nflverse_weekly_stats AS w
        WHERE w.season = ?
          AND w.week   <= ?
    """

    df_nflverse = read_sql(nflverse_query, params=[season, week])

    if df_nflverse.empty:
        return pd.DataFrame()

    # Map GSIS IDs to Sleeper IDs
    gsis_to_sleeper = get_gsis_to_sleeper_map()
    df_nflverse["player_id"] = df_nflverse["gsis_id"].map(gsis_to_sleeper)

    # Drop rows without a valid Sleeper ID mapping
    df_nflverse = df_nflverse[df_nflverse["player_id"].notna()].copy()

    if df_nflverse.empty:
        return pd.DataFrame()

    # Now join with player_game_stats to get game_id and opponent
    pgs_query = """
        SELECT
            player_id,
            game_id,
            season,
            week,
            team_id,
            opponent_team_id
        FROM player_game_stats
        WHERE season = ?
          AND week <= ?
    """
    df_pgs = read_sql(pgs_query, params=[season, week])

    # Merge on Sleeper player_id + season + week
    df = df_nflverse.merge(
        df_pgs,
        on=["player_id", "season", "week"],
        how="inner",
        suffixes=("", "_pgs")
    )

    # Use team from nflverse if available, otherwise from pgs
    df["team"] = df["team"].fillna(df["team_id"])
    df["opponent"] = df["opponent_team_id"]

    # Clean up columns
    df = df.drop(columns=["team_id", "opponent_team_id", "gsis_id"], errors="ignore")

    # Normalize dtypes
    for col in [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_air_yards",
        "receiving_yards_after_catch",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("float64")

    for col in ["target_share", "air_yards_share", "wopr", "racr",
                "passing_epa", "rushing_epa", "receiving_epa"]:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    return df


def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute high-signal efficiency metrics per player-week and then build
    rolling, leak-safe features per player-season.

    We do **not** use any target-week data in the features for that week:
    everything is shifted so that week W only sees weeks < W.
    """

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    group = df.groupby(["player_id", "season"], group_keys=False)

    # --- Step 1: define raw per-opportunity metrics ---

    # Dropbacks for QBs: attempts + sacks
    df["dropbacks"] = df["attempts"] + df["sacks"]

    metric_specs: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        # Passing efficiency
        ("pass_epa_per_db",
         lambda d: _safe_div(d["passing_epa"], d["dropbacks"])),
        ("pass_yards_per_att",
         lambda d: _safe_div(d["passing_yards"], d["attempts"])),
        ("pass_td_rate",
         lambda d: _safe_div(d["passing_tds"], d["attempts"])),
        ("pass_int_rate",
         lambda d: _safe_div(d["interceptions"], d["attempts"])),
        ("pass_comp_pct",
         lambda d: _safe_div(d["completions"], d["attempts"])),

        # Rushing efficiency
        ("rush_epa_per_carry",
         lambda d: _safe_div(d["rushing_epa"], d["carries"])),
        ("rush_yards_per_carry",
         lambda d: _safe_div(d["rushing_yards"], d["carries"])),
        ("rush_td_rate",
         lambda d: _safe_div(d["rushing_tds"], d["carries"])),

        # Receiving efficiency
        ("rec_epa_per_target",
         lambda d: _safe_div(d["receiving_epa"], d["targets"])),
        ("rec_yards_per_target",
         lambda d: _safe_div(d["receiving_yards"], d["targets"])),
        ("rec_yards_per_rec",
         lambda d: _safe_div(d["receiving_yards"], d["receptions"])),
        ("rec_yac_per_rec",
         lambda d: _safe_div(d["receiving_yards_after_catch"], d["receptions"])),

        # Market share / air-yard metrics
        ("target_share",
         lambda d: d["target_share"]),
        ("air_yards_share",
         lambda d: d["air_yards_share"]),
        ("wopr",
         lambda d: d["wopr"]),
        ("racr",
         lambda d: d["racr"]),
    ]

    # Compute raw metric columns (intermediate, not used directly as features)
    raw_cols: List[str] = []
    for base_name, fn in metric_specs:
        raw_col = f"_raw_{base_name}"
        df[raw_col] = fn(df)
        raw_cols.append(raw_col)

    # --- Step 2: build rolling leak-safe features ---

    for base_name, _ in metric_specs:
        raw_col = f"_raw_{base_name}"
        # Last game value
        df[f"eff_{base_name}_last1"] = group[raw_col].shift(1)

        # Rolling last 3 games
        df[f"eff_{base_name}_last3"] = (
            group[raw_col]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

        # Season-to-date mean
        df[f"eff_{base_name}_season_to_date"] = (
            group[raw_col]
            .transform(lambda s: s.expanding(min_periods=1).mean())
            .shift(1)
        )

    # We keep the raw columns for debugging if you ever want to inspect them,
    # but note: they do **not** get merged into the master feature table since
    # they don't start with "eff_".
    return df


def build_efficiency_features(season: int, week: int) -> pd.DataFrame:
    """
    Build pre-game efficiency features for all (player, game) rows in the
    given (season, week).

    Returns a DataFrame with:
        - player_id
        - game_id
        - season
        - week
        - team
        - opponent
        - eff_* feature columns

    All eff_* columns for week W are computed using ONLY weeks < W.
    """

    # Base: which players/games do we need features for in this week?
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
                "team",
                "opponent",
            ]
        )

    # Load all efficiency source data up through this week
    df_all = _load_efficiency_source_data(season=season, week=week)

    # Compute rolling features
    df_all = _add_efficiency_features(df_all)

    # Keep only rows for the target week
    df_week = df_all[df_all["week"] == week].copy()

    # Join back to base to:
    #   - filter to only players we care about
    #   - ensure game_id alignment
    merged = base.merge(
        df_week,
        on=["player_id", "game_id", "season", "week"],
        how="left",
        suffixes=("", "_y"),
    )

    # Team/opponent context
    if "team" in merged.columns:
        merged["team"] = merged["team"].fillna(merged.get("team_id"))
    else:
        merged["team"] = merged.get("team_id")

    if "opponent" in merged.columns:
        merged["opponent"] = merged["opponent"].fillna(merged.get("opponent_team_id"))
    else:
        merged["opponent"] = merged.get("opponent_team_id")

    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    context_cols: List[str] = ["team", "opponent"]

    eff_feature_cols = [c for c in merged.columns if c.startswith("eff_")]

    final_cols = [c for c in id_cols + context_cols + eff_feature_cols if c in merged.columns]

    return merged[final_cols]


if __name__ == "__main__":
    # Smoke test; adjust if needed.
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_test = build_efficiency_features(TEST_SEASON, TEST_WEEK)
    print("Efficiency features shape:", df_test.shape)
    print(df_test.head())
