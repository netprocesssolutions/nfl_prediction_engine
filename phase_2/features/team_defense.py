# phase_2/features/team_defense.py

from __future__ import annotations

from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

from ..db import read_sql


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """
    Safe division that returns NaN when denom <= 0.
    """
    numer = numer.astype("float64")
    denom = denom.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, numer / denom, np.nan)
    return pd.Series(out, index=numer.index)


def _load_team_defense_data(season: int, week: int) -> pd.DataFrame:
    """
    Load per-game team defense stats for all weeks up to and including `week`.
    Each row is one defense performance (team_id) in a game.
    """
    query = """
        SELECT
            team_id,
            season,
            week,
            points_allowed,
            yards_allowed_passing,
            yards_allowed_rushing,
            yards_allowed_total,
            yards_allowed_to_wr,
            yards_allowed_to_te,
            yards_allowed_to_rb,
            targets_allowed_to_wr,
            targets_allowed_to_te,
            targets_allowed_to_rb,
            tds_allowed_to_wr,
            tds_allowed_to_te,
            tds_allowed_to_rb,
            redzone_defense_efficiency,
            epa_allowed,
            success_rate_allowed,
            explosive_plays_allowed,
            sacks,
            interceptions,
            fumbles_recovered,
            defensive_tds
        FROM team_defense_game_stats
        WHERE season = ?
          AND week   <= ?
    """
    df = read_sql(query, params=[season, week])
    return df


def _add_team_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given per-game defense rows, compute opponent-defense feature history:
    - global difficulty metrics (points, yards, EPA, success rate, explosives, sacks, turnovers)
    - position-group metrics (WR/TE/RB yards per target, TDs per target, targets per game)

    Then build leak-safe rolling features per defense team:
        oppdef_<metric>_last1
        oppdef_<metric>_last3
        oppdef_<metric>_season_to_date
    """
    if df.empty:
        return df

    df = df.sort_values(["team_id", "season", "week"]).reset_index(drop=True)
    group = df.groupby(["team_id", "season"], group_keys=False)

    metric_specs: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        # Global difficulty (per game)
        ("points_allowed", lambda d: d["points_allowed"].astype("float64")),
        ("yards_total", lambda d: d["yards_allowed_total"].astype("float64")),
        ("epa_allowed", lambda d: d["epa_allowed"].astype("float64")),
        ("success_rate_allowed", lambda d: d["success_rate_allowed"].astype("float64")),
        ("explosive_plays_allowed", lambda d: d["explosive_plays_allowed"].astype("float64")),
        ("sacks", lambda d: d["sacks"].astype("float64")),
        ("turnovers", lambda d: (d["interceptions"] + d["fumbles_recovered"]).astype("float64")),
        ("defensive_tds", lambda d: d["defensive_tds"].astype("float64")),
    ]

    # Position-specific metrics: WR / TE / RB
    for pos in ["wr", "te", "rb"]:
        y_col = f"yards_allowed_to_{pos}"
        t_col = f"targets_allowed_to_{pos}"
        td_col = f"tds_allowed_to_{pos}"

        metric_specs.extend([
            (
                f"{pos}_yards_per_target",
                lambda d, y=y_col, t=t_col: _safe_div(d[y], d[t]),
            ),
            (
                f"{pos}_tds_per_target",
                lambda d, td=td_col, t=t_col: _safe_div(d[td], d[t]),
            ),
            (
                f"{pos}_targets_per_game",
                lambda d, t=t_col: d[t].astype("float64"),
            ),
        ])

    # Compute raw metric columns
    raw_cols: List[str] = []
    for base_name, fn in metric_specs:
        raw_col = f"_raw_{base_name}"
        df[raw_col] = fn(df)
        raw_cols.append(raw_col)

    # Rolling, leak-safe aggregates
    for base_name, _ in metric_specs:
        raw_col = f"_raw_{base_name}"

        # Last game value
        df[f"oppdef_{base_name}_last1"] = group[raw_col].shift(1)

        # Last 3 games mean
        df[f"oppdef_{base_name}_last3"] = (
            group[raw_col]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

        # Season-to-date mean up to previous week
        df[f"oppdef_{base_name}_season_to_date"] = (
            group[raw_col]
            .transform(lambda s: s.expanding(min_periods=1).mean())
            .shift(1)
        )

    return df


def build_team_defense_features(season: int, week: int) -> pd.DataFrame:
    """
    Build opponent-defense features for every (player, game) in a given week.

    For each player, we:
      - find their opponent_team_id (the defense)
      - look up the defense's history in team_defense_game_stats
      - attach leak-safe rolling metrics prefixed oppdef_*

    Returns a DataFrame with:
        player_id, game_id, season, week, and oppdef_* feature columns.
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
            columns=["player_id", "game_id", "season", "week"]
        )

    # Load all defense data up through this week
    df_all = _load_team_defense_data(season=season, week=week)

    if df_all.empty:
        # No defense stats yet – return just keys so merge is harmless
        return pd.DataFrame(
            {
                "player_id": base["player_id"],
                "game_id": base["game_id"],
                "season": base["season"],
                "week": base["week"],
            }
        )

    df_all = _add_team_defense_features(df_all)

    # Keep only the target week rows for defenses
    df_week = df_all[df_all["week"] == week].copy()

    # Join defenses to players:
    # opponent_team_id (offense's perspective) = team_id (defense)
    merged = base.merge(
        df_week,
        left_on=["opponent_team_id", "season", "week"],
        right_on=["team_id", "season", "week"],
        how="left",
        suffixes=("", "_def"),
    )

    # Build final feature block: keys + all oppdef_* columns
    id_cols: List[str] = ["player_id", "game_id", "season", "week"]
    feature_cols = [c for c in merged.columns if c.startswith("oppdef_")]

    final_cols = [c for c in id_cols + feature_cols if c in merged.columns]

    return merged[final_cols]


if __name__ == "__main__":
    TEST_SEASON = 2023
    TEST_WEEK = 5

    df_test = build_team_defense_features(TEST_SEASON, TEST_WEEK)
    print("Team defense features shape:", df_test.shape)
    print(df_test.head())
