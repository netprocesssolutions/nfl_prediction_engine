# phase_2/features/team_context.py

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..db import read_sql


def _load_team_tendencies(season: int, week: int) -> pd.DataFrame:
    """
    Load team-level offensive tendencies up to (but not including) the target week.

    We use team_tendencies rows with through_week < target_week and then
    take the latest entry per team_id to represent season-to-date tendencies
    as of that point in time.
    """
    if week <= 1:
        # No prior weeks; return empty so we just get NaNs for week 1.
        return pd.DataFrame(columns=["team_id", "season", "through_week"])

    query = """
        SELECT
            team_id,
            season,
            through_week,
            plays_per_game,
            seconds_per_play,
            pass_rate,
            neutral_pass_rate,
            pass_rate_ahead,
            pass_rate_behind,
            wr_target_share,
            rb_target_share,
            te_target_share,
            rb1_carry_share,
            rb2_carry_share,
            rz_pass_rate,
            rz_plays_per_game,
            points_per_game,
            yards_per_game
        FROM team_tendencies
        WHERE season = ?
          AND through_week < ?
    """
    df = read_sql(query, params=[season, week])

    if df.empty:
        return df

    # For each team, keep the row with the max through_week
    df = df.sort_values(["team_id", "through_week"])
    latest = df.groupby("team_id", as_index=False).tail(1)

    return latest.reset_index(drop=True)


def _attach_team_tendencies(
    base: pd.DataFrame,
    tend: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach team_tendencies for both the player's team and the opponent.

    Produces columns prefixed with:
        - ctx_team_*  (from team perspective)
        - ctx_opp_*   (from opponent perspective)
    """
    if tend is None or tend.empty:
        # Nothing to attach; return base as-is.
        return base

    metrics = [
        "plays_per_game",
        "seconds_per_play",
        "pass_rate",
        "neutral_pass_rate",
        "pass_rate_ahead",
        "pass_rate_behind",
        "wr_target_share",
        "rb_target_share",
        "te_target_share",
        "rb1_carry_share",
        "rb2_carry_share",
        "rz_pass_rate",
        "rz_plays_per_game",
        "points_per_game",
        "yards_per_game",
    ]

    # Team-side tendencies
    team_cols = ["team_id", "season"] + metrics
    team_tend = tend[team_cols].copy()
    team_rename = {
        "team_id": "team_id",
        "season": "season",
    }
    for m in metrics:
        team_rename[m] = f"ctx_team_{m}"
    team_tend = team_tend.rename(columns=team_rename)

    base = base.merge(
        team_tend,
        on=["team_id", "season"],
        how="left",
    )

    # Opponent-side tendencies: same metrics with ctx_opp_ prefix
    opp_cols = ["team_id", "season"] + metrics
    opp_tend = tend[opp_cols].copy()
    opp_rename = {
        "team_id": "opponent_team_id",
        "season": "season",
    }
    for m in metrics:
        opp_rename[m] = f"ctx_opp_{m}"
    opp_tend = opp_tend.rename(columns=opp_rename)

    base = base.merge(
        opp_tend,
        on=["opponent_team_id", "season"],
        how="left",
    )

    return base


def _attach_vegas_context(base: pd.DataFrame) -> pd.DataFrame:
    """
    Attach pre-game Vegas information (spread, total, implied totals, ML odds)
    from vegas_game_context via game_id.

    We derive team- and opponent-specific perspectives:
        ctx_team_spread           (spread from this team's perspective)
        ctx_team_total_line
        ctx_team_implied_total
        ctx_opp_implied_total
        ctx_team_ml_odds
        ctx_opp_ml_odds
    """
    if base.empty:
        return base

    # Load vegas context for the relevant games
    game_ids = base["game_id"].unique().tolist()
    placeholder = ",".join(["?"] * len(game_ids))

    query = f"""
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            spread_line,
            total_line,
            home_ml_odds,
            away_ml_odds,
            home_implied_total,
            away_implied_total
        FROM vegas_game_context
        WHERE game_id IN ({placeholder})
    """
    vega = read_sql(query, params=game_ids)

    if vega.empty:
        # No vegas data; just create NaN columns and return.
        for col in [
            "ctx_team_spread",
            "ctx_team_total_line",
            "ctx_team_implied_total",
            "ctx_opp_implied_total",
            "ctx_team_ml_odds",
            "ctx_opp_ml_odds",
        ]:
            base[col] = np.nan
        return base

    # Also need home/away team IDs so we can determine if the player's team is home
    games_query = f"""
        SELECT
            game_id,
            home_team_id,
            away_team_id
        FROM games
        WHERE game_id IN ({placeholder})
    """
    games = read_sql(games_query, params=game_ids)

    # Merge vegas + games on game_id
    vg = vega.merge(games, on="game_id", how="left")

    # Merge into base on game_id (we already have team_id, opponent_team_id there)
    df = base.merge(vg, on="game_id", how="left", suffixes=("", "_vegas"))

    # Determine if the player's team is the home team
    df["is_home_team"] = df["team_id"] == df["home_team_id"]

    # Spread: stored from home perspective (home spread)
    # For the team's perspective:
    #   team_spread = spread_line if team is home else -spread_line
    spread = df["spread_line"].astype("float64")
    is_home = df["is_home_team"].fillna(False)

    df["ctx_team_spread"] = np.where(is_home, spread, -spread)

    # Total line: same for both sides
    df["ctx_team_total_line"] = df["total_line"].astype("float64")

    # Implied totals and moneyline odds: choose based on home/away
    home_it = df["home_implied_total"].astype("float64")
    away_it = df["away_implied_total"].astype("float64")
    home_ml = df["home_ml_odds"].astype("float64")
    away_ml = df["away_ml_odds"].astype("float64")

    df["ctx_team_implied_total"] = np.where(is_home, home_it, away_it)
    df["ctx_opp_implied_total"] = np.where(is_home, away_it, home_it)

    df["ctx_team_ml_odds"] = np.where(is_home, home_ml, away_ml)
    df["ctx_opp_ml_odds"] = np.where(is_home, away_ml, home_ml)

    # Keep only the columns we actually need to return as a feature block
    feature_cols = [
        "ctx_team_spread",
        "ctx_team_total_line",
        "ctx_team_implied_total",
        "ctx_opp_implied_total",
        "ctx_team_ml_odds",
        "ctx_opp_ml_odds",
    ]

    keep_cols = [
        "player_id",
        "game_id",
        "season",
        "week",
        "team_id",
        "opponent_team_id",
    ] + feature_cols

    return df[keep_cols]


def build_team_context_features(season: int, week: int) -> pd.DataFrame:
    """
    Build team context features for all (player, game) rows in the given week.

    Includes:
        - ctx_team_* metrics from team_tendencies (pace, pass_rate, shares, etc.)
        - ctx_opp_* metrics from opponent tendencies
        - ctx_team_spread, ctx_team_total_line, ctx_team_implied_total,
          ctx_opp_implied_total, ctx_team_ml_odds, ctx_opp_ml_odds

    Returns a DataFrame with:
        - player_id, game_id, season, week
        - team_id, opponent_team_id (for internal merging)
        - ctx_* feature columns
    """

    # Base player-game shell for this week
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

    # Attach team tendencies (season-to-date up to week-1)
    tend = _load_team_tendencies(season, week)
    base = _attach_team_tendencies(base, tend)

    # Attach vegas pre-game information
    base = _attach_vegas_context(base)

    # Convert team_id/opponent_team_id to 'team'/'opponent' context columns
    base["team"] = base["team_id"]
    base["opponent"] = base["opponent_team_id"]

    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    context_cols: List[str] = ["team", "opponent"]

    # Any column starting with "ctx_" is a team-context feature
    ctx_feature_cols = [c for c in base.columns if c.startswith("ctx_")]

    final_cols = [c for c in id_cols + context_cols + ctx_feature_cols if c in base.columns]

    return base[final_cols]


if __name__ == "__main__":
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_test = build_team_context_features(TEST_SEASON, TEST_WEEK)
    print("Team context features shape:", df_test.shape)
    print(df_test.head())
