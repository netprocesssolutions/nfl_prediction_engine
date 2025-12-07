# phase_2/pipeline/build_week.py

from __future__ import annotations

from typing import List

import pandas as pd

import sqlite3

from ..config import FEATURE_TABLE_NAME
from ..db import read_sql, write_dataframe, get_connection
from ..schema import PLAYER_GAME_FEATURES_SCHEMA
from ..features.usage import build_usage_features
from ..features.efficiency import build_efficiency_features
from ..features.team_context import build_team_context_features
from ..features.team_defense import build_team_defense_features
from ..features.ngs import build_ngs_features
from ..features.schedule_context import build_schedule_features
from ..features.weather import build_weather_features
from ..features.pbp_features import calculate_all_pbp_features, get_connection as get_pbp_connection


def build_pbp_features(season: int, week: int) -> pd.DataFrame:
    """
    Build PBP-based features for all players in a week.

    Returns DataFrame with columns: player_id, game_id, season, week, pbp_*
    """
    # Get base player-games for this week
    base_query = """
    SELECT DISTINCT
        pgs.player_id, pgs.game_id, pgs.season, pgs.week
    FROM player_game_stats pgs
    WHERE pgs.season = ? AND pgs.week = ?
    """
    base_df = read_sql(base_query, params=[season, week])

    if base_df.empty:
        return pd.DataFrame()

    # Get ID mapping
    id_query = "SELECT sleeper_id, gsis_id FROM player_id_mapping WHERE gsis_id IS NOT NULL"
    id_df = read_sql(id_query)
    sleeper_to_gsis = dict(zip(id_df["sleeper_id"].astype(str), id_df["gsis_id"]))

    # Calculate PBP features for each player
    conn = get_pbp_connection()
    results = []

    for _, row in base_df.iterrows():
        player_id = str(row["player_id"])
        gsis_id = sleeper_to_gsis.get(player_id)

        if gsis_id:
            features = calculate_all_pbp_features(conn, gsis_id, season, week)
        else:
            features = {}

        features["player_id"] = row["player_id"]
        features["game_id"] = row["game_id"]
        features["season"] = season
        features["week"] = week
        results.append(features)

    conn.close()

    return pd.DataFrame(results)


def _build_base_player_game_frame(season: int, week: int) -> pd.DataFrame:
    """
    Build the base (player, game) frame for a given week with:
    - Identifiers: season, week, game_id, player_id, team, opponent
    - Player info: player_name, position
    - Game context: home_team, away_team, is_home, is_playoff (placeholder)
    - Labels (training targets) from player_game_stats

    This frame intentionally contains NO engineered features: only IDs,
    context, and labels. Feature modules (usage, efficiency, etc.) will
    be merged into this frame.
    """

    query = """
        SELECT
            pgs.player_id,
            pgs.game_id,
            pgs.season,
            pgs.week,
            pgs.team_id,
            pgs.opponent_team_id,
            g.home_team_id,
            g.away_team_id,
            pl.full_name      AS player_name,
            pl.position       AS position,
            -- Labels (targets)
            pgs.targets,
            pgs.receptions,
            pgs.rec_yards,
            pgs.rec_tds,
            pgs.carries,
            pgs.rush_yards,
            pgs.rush_tds,
            pgs.pass_attempts,
            pgs.completions,
            pgs.pass_yards,
            pgs.pass_tds,
            pgs.interceptions,
            pgs.fumbles,
            pgs.two_point_conversions
        FROM player_game_stats AS pgs
        LEFT JOIN players AS pl
            ON pl.player_id = pgs.player_id
        LEFT JOIN games AS g
            ON g.game_id = pgs.game_id
        WHERE pgs.season = ?
          AND pgs.week = ?
    """

    df = read_sql(query, params=[season, week])

    if df.empty:
        # No games that week – return an empty frame with the right columns.
        cols = PLAYER_GAME_FEATURES_SCHEMA.identifiers + PLAYER_GAME_FEATURES_SCHEMA.labels
        return pd.DataFrame(columns=cols)

    # --- Identifier / context columns ---

    df["team"] = df["team_id"]
    df["opponent"] = df["opponent_team_id"]

    df["home_team"] = df["home_team_id"]
    df["away_team"] = df["away_team_id"]

    # is_home: from the player perspective
    df["is_home"] = df["team_id"] == df["home_team_id"]

    # is_playoff: placeholder for now (regular season only = 0).
    # Later we can refine using schedule data or nflreadpy season_type.
    df["is_playoff"] = False

    # --- Labels (rename to schema-consistent names) ---

    label_mapping = {
        "targets": "label_targets",
        "receptions": "label_receptions",
        "rec_yards": "label_rec_yards",
        "rec_tds": "label_rec_tds",
        "carries": "label_carries",
        "rush_yards": "label_rush_yards",
        "rush_tds": "label_rush_tds",
        "pass_attempts": "label_pass_attempts",
        "completions": "label_pass_completions",
        "pass_yards": "label_pass_yards",
        "pass_tds": "label_pass_tds",
        "interceptions": "label_interceptions",
        "fumbles": "label_fumbles",
        "two_point_conversions": "label_two_pt_conversions",
    }

    for src, dst in label_mapping.items():
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = 0

    # --- Final base selection: identifiers + labels only ---

    id_cols: List[str] = PLAYER_GAME_FEATURES_SCHEMA.identifiers
    label_cols: List[str] = PLAYER_GAME_FEATURES_SCHEMA.labels

    # Map schema identifier names to actual columns in df
    # (some are 1:1, some are aliases)
    rename_map = {
        "player_name": "player_name",
        "position": "position",
        "team": "team",
        "opponent": "opponent",
        "home_team": "home_team",
        "away_team": "away_team",
        "is_home": "is_home",
        "is_playoff": "is_playoff",
    }

    # Ensure identifiers exist
    for col in ["season", "week", "game_id", "player_id"]:
        if col not in df.columns:
            raise RuntimeError(f"Expected identifier column `{col}` not found in base frame.")

    base = pd.DataFrame()
    base["season"] = df["season"]
    base["week"] = df["week"]
    base["game_id"] = df["game_id"]
    base["player_id"] = df["player_id"]

    # Add the rest of the identifier columns if present / mappable
    for col in id_cols:
        if col in ["season", "week", "game_id", "player_id"]:
            continue
        src = rename_map.get(col)
        if src is not None and src in df.columns:
            base[col] = df[src]
        else:
            # If an identifier isn't available yet, fill with NA/False.
            if col.startswith("is_"):
                base[col] = False
            else:
                base[col] = pd.NA

    # Attach labels
    for lbl in label_cols:
        if lbl in df.columns:
            base[lbl] = df[lbl]
        else:
            base[lbl] = 0

    return base


def _merge_feature_block(
    base: pd.DataFrame,
    feature_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge a feature block into the base frame on (player_id, game_id, season, week).

    Any overlapping non-key columns from the feature_df that don't start with
    the expected prefix will be ignored to avoid accidental shadowing.
    """
    if feature_df is None or feature_df.empty:
        return base

    key_cols = ["player_id", "game_id", "season", "week"]

    # Ensure we don't clobber ID columns
    for col in key_cols:
        if col not in feature_df.columns:
            raise RuntimeError(f"Feature block `{prefix}` is missing key column `{col}`")

    # Keep only keys + feature columns with the proper prefix
    feature_cols = [c for c in feature_df.columns if c.startswith(prefix)]
    keep_cols = key_cols + feature_cols

    feature_df = feature_df[keep_cols].copy()

    merged = base.merge(feature_df, on=key_cols, how="left")
    return merged


def _persist_week(df: pd.DataFrame, season: int, week: int) -> None:
    """
    Persist the given week of features into the FEATURE_TABLE_NAME.

    Strategy:
        - DELETE any existing rows for (season, week) from the table (if it exists).
        - APPEND the new rows.

    This lets you safely re-run Phase 2 for a given week as often as you want.
    """
    if df.empty:
        return

    with get_connection(readonly=False) as conn:
        # On first run the table may not exist yet; let to_sql create it.
        try:
            conn.execute(
                f"DELETE FROM {FEATURE_TABLE_NAME} WHERE season = ? AND week = ?;",
                (season, week),
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            # Ignore "no such table" (first write); re-raise anything else.
            if "no such table" not in str(e):
                raise

        # Append the new rows; this will create the table on first run if needed.
        df.to_sql(FEATURE_TABLE_NAME, conn, if_exists="append", index=False)

def build_features_for_week(
    season: int,
    week: int,
    persist: bool = False,
) -> pd.DataFrame:
    """
    Build the full player_game_features row set for a given (season, week).

    Steps:
        1. Build the base frame (identifiers + context + labels).
        2. Compute and merge usage features (usage_*).
        3. (Later) compute and merge efficiency, form, team context, defense, etc.
        4. If persist=True, write to the database (deleting any existing rows
           for that (season, week) first).

    Returns:
        DataFrame of features for all players in that week.
    """

    # 1) Base frame
    base = _build_base_player_game_frame(season, week)
    if base.empty:
        return base

    # 2) Usage features
    usage = build_usage_features(season, week)
    base = _merge_feature_block(base, usage, prefix="usage_")

    # 3) Efficiency features (EPA & share based)
    eff = build_efficiency_features(season, week)
    base = _merge_feature_block(base, eff, prefix="eff_")

    # 4) NGS skill/talent features (QB + WR/TE/RB)
    ngs = build_ngs_features(season, week)
    base = _merge_feature_block(base, ngs, prefix="ngs_")

    # 5) Schedule / rest / primetime context
    sched_ctx = build_schedule_features(season, week)
    base = _merge_feature_block(base, sched_ctx, prefix="sched_")

    # 6) Weather / roof / surface context
    weather = build_weather_features(season, week)
    base = _merge_feature_block(base, weather, prefix="weather_")

    # 6) Team context & Vegas pre-game expectation
    team_ctx = build_team_context_features(season, week)
    base = _merge_feature_block(base, team_ctx, prefix="ctx_")

    # 7) Opponent team defense difficulty
    opp_def = build_team_defense_features(season, week)
    base = _merge_feature_block(base, opp_def, prefix="oppdef_")

    # 8) Play-by-play based features (EPA, air yards, CPOE, etc.)
    pbp = build_pbp_features(season, week)
    base = _merge_feature_block(base, pbp, prefix="pbp_")

    # TODO: defender matchup, archetypes, etc.

    # ------------------------------------------------------------------
    # Final column ordering (do NOT drop any columns, just reorder)
    # ------------------------------------------------------------------
    cols = list(base.columns)

    # Use schema-defined identifiers / labels where possible
    id_cols = [c for c in getattr(PLAYER_GAME_FEATURES_SCHEMA, "identifiers", []) if c in cols]
    label_cols = [c for c in getattr(PLAYER_GAME_FEATURES_SCHEMA, "labels", []) if c in cols]

    # All the feature families we currently support
    feature_prefixes = (
        "usage_",
        "eff_",
        "ngs_",
        "sched_",
        "weather_",
        "ctx_",
        "oppdef_",
        "pbp_",
    )
    feature_cols = sorted(
        [c for c in cols if c.startswith(feature_prefixes)]
    )

    # Anything else (debug / helper columns) goes at the end
    used = set(id_cols) | set(label_cols) | set(feature_cols)
    other_cols = [c for c in cols if c not in used]

    ordered_cols = id_cols + label_cols + feature_cols + other_cols
    base = base[ordered_cols]

    if persist:
        _persist_week(base, season, week)

    return base


if __name__ == "__main__":
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_week = build_features_for_week(TEST_SEASON, TEST_WEEK, persist=False)
    print("Week features shape:", df_week.shape)

    for prefix in ["usage_", "eff_", "ngs_", "sched_", "weather_", "ctx_", "oppdef_"]:
        cols = [c for c in df_week.columns if c.startswith(prefix)]
        print(f"{prefix} -> {len(cols)} columns")

    print(df_week.head())
