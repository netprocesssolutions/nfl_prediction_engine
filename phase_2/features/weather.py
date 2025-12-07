from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..db import read_sql


def build_weather_features(season: int, week: int) -> pd.DataFrame:
    """
    Build game-level weather / stadium context features for all (player, game)
    rows in the given week.

    Uses `game_weather`, joined by (season, week, game_id).

    Features (by prefix):
        weather_temp_f
        weather_feels_like_f
        weather_humidity_pct
        weather_wind_mph
        weather_wind_gust_mph
        weather_precip_in
        weather_visibility_miles
        weather_impact_score

        weather_is_dome
        weather_is_high_wind
        weather_is_heavy_precip

        weather_roof_type  (categorical)
        weather_surface    (categorical)
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

    # Load weather for this season/week
    wx_query = """
        SELECT
            game_id,
            season,
            week,
            roof_type,
            surface,
            temperature_f,
            feels_like_f,
            humidity_pct,
            wind_speed_mph,
            wind_gust_mph,
            precipitation_in,
            visibility_miles,
            weather_impact_score
        FROM game_weather
        WHERE season = ?
          AND week   = ?
    """
    wx = read_sql(wx_query, params=[season, week])

    if wx.empty:
        # No weather records; return only keys so merge is harmless.
        return pd.DataFrame(
            {
                "player_id": base["player_id"],
                "game_id": base["game_id"],
                "season": base["season"],
                "week": base["week"],
            }
        )

    df = base.merge(
        wx,
        on=["game_id", "season", "week"],
        how="left",
        suffixes=("", "_wx"),
    )

    # Numeric weather metrics
    for col in [
        "temperature_f",
        "feels_like_f",
        "humidity_pct",
        "wind_speed_mph",
        "wind_gust_mph",
        "precipitation_in",
        "visibility_miles",
        "weather_impact_score",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    df["weather_temp_f"] = df["temperature_f"]
    df["weather_feels_like_f"] = df["feels_like_f"]
    df["weather_humidity_pct"] = df["humidity_pct"]
    df["weather_wind_mph"] = df["wind_speed_mph"]
    df["weather_wind_gust_mph"] = df["wind_gust_mph"]
    df["weather_precip_in"] = df["precipitation_in"]
    df["weather_visibility_miles"] = df["visibility_miles"]
    df["weather_impact_score"] = df["weather_impact_score"]

    # Roof / surface as categorical features
    df["weather_roof_type"] = df["roof_type"]
    df["weather_surface"] = df["surface"]

    # Flags: dome vs outdoor
    dome_values = {"INDOOR", "DOME", "CLOSED", "RETRACTABLE"}
    df["weather_is_dome"] = df["roof_type"].isin(dome_values).astype("int8")

    # High wind & heavy precip flags
    df["weather_is_high_wind"] = np.where(
        df["wind_speed_mph"] >= 15.0,
        1,
        0,
    ).astype("int8")

    df["weather_is_heavy_precip"] = np.where(
        df["precipitation_in"] >= 0.10,
        1,
        0,
    ).astype("int8")

    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    feature_cols = [c for c in df.columns if c.startswith("weather_")]

    final_cols = [c for c in id_cols + feature_cols if c in df.columns]

    return df[final_cols]


if __name__ == "__main__":
    TEST_SEASON = 2025
    TEST_WEEK = 12

    df_test = build_weather_features(TEST_SEASON, TEST_WEEK)
    print("Weather features shape:", df_test.shape)
    print(df_test.head())
