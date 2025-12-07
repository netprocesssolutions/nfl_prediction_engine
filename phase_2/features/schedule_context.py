from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..db import read_sql


def build_schedule_features(season: int, week: int) -> pd.DataFrame:
    """
    Build schedule / rest context features for all (player, game) rows in a given week.

    Uses the `schedules` table (per game) joined to player_game_stats via game_id.

    Features (by prefix):
        sched_is_home          : 1 if player's team is the home team, else 0
        sched_rest_days        : days of rest for this team before the game
        sched_short_rest_flag  : 1 if rest_days <= 6 (TNF / short week feel)
        sched_long_rest_flag   : 1 if rest_days >= 10 (mini-bye / bye)
        sched_is_primetime     : 1 if game is primetime
        sched_is_divisional    : 1 if game is divisional
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

    # Load schedules for this season/week
    sched_query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_rest_days,
            away_rest_days,
            is_primetime,
            is_divisional
        FROM schedules
        WHERE season = ?
          AND week   = ?
    """
    sched = read_sql(sched_query, params=[season, week])

    if sched.empty:
        # No schedule metadata for this week; return only keys so merge is harmless.
        return pd.DataFrame(
            {
                "player_id": base["player_id"],
                "game_id": base["game_id"],
                "season": base["season"],
                "week": base["week"],
            }
        )

    # Merge schedules onto base via game_id + season + week
    df = base.merge(
        sched,
        on=["game_id", "season", "week"],
        how="left",
        suffixes=("", "_sched"),
    )

    # Determine if player's team is home
    df["sched_is_home"] = (df["team_id"] == df["home_team"]).astype("int8")

    # Rest days for this team
    # NOTE: team_id is an abbreviation like "BUF", same as schedules.home_team/away_team
    df["home_rest_days"] = df["home_rest_days"].astype("float64")
    df["away_rest_days"] = df["away_rest_days"].astype("float64")

    df["sched_rest_days"] = np.where(
        df["sched_is_home"] == 1,
        df["home_rest_days"],
        df["away_rest_days"],
    )

    # Short / long rest flags
    df["sched_short_rest_flag"] = np.where(
        df["sched_rest_days"] <= 6,
        1,
        0,
    ).astype("int8")

    df["sched_long_rest_flag"] = np.where(
        df["sched_rest_days"] >= 10,
        1,
        0,
    ).astype("int8")

    # Primetime / divisional flags (stored as ints 0/1 in schedules)
    df["sched_is_primetime"] = df["is_primetime"].fillna(0).astype("int8")
    df["sched_is_divisional"] = df["is_divisional"].fillna(0).astype("int8")

    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    feature_cols = [c for c in df.columns if c.startswith("sched_")]

    final_cols = [c for c in id_cols + feature_cols if c in df.columns]

    return df[final_cols]


if __name__ == "__main__":
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_test = build_schedule_features(TEST_SEASON, TEST_WEEK)
    print("Schedule features shape:", df_test.shape)
    print(df_test.head())
