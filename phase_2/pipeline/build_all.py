from __future__ import annotations

import sqlite3

import logging
from typing import List, Tuple, Optional

from ..db import read_sql
from ..schema import PLAYER_GAME_FEATURES_SCHEMA
from .build_week import build_features_for_week

logger = logging.getLogger(__name__)


def _list_weeks(
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Return a sorted list of (season, week) pairs present in player_game_stats,
    optionally filtered by [min_season, max_season].
    """
    where_clauses = []
    params: List[object] = []

    if min_season is not None:
        where_clauses.append("season >= ?")
        params.append(min_season)
    if max_season is not None:
        where_clauses.append("season <= ?")
        params.append(max_season)

    query = """
        SELECT DISTINCT season, week
        FROM player_game_stats
    """
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY season, week"

    df = read_sql(query, params=params)
    if df.empty:
        return []

    return list(df.itertuples(index=False, name=None))  # (season, week)


def _features_exist_for_week(season: int, week: int) -> bool:
    """
    Check whether features for (season, week) already exist in the
    player_game_features table.
    """
    table_name = getattr(PLAYER_GAME_FEATURES_SCHEMA, "table_name", "player_game_features")

    try:
        df = read_sql(
            f"SELECT COUNT(*) AS n_rows FROM {table_name} WHERE season = ? AND week = ?",
            params=[season, week],
        )
    except Exception:
        # Table doesn't exist yet or cannot be queried: treat as "no features yet"
        return False

    if df.empty:
        return False

    return int(df["n_rows"].iloc[0]) > 0


def build_all_features(
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
    overwrite: bool = False,
    persist: bool = True,
) -> None:
    """
    Build player-game feature rows for all (season, week) combinations
    present in player_game_stats, and persist them into the features table.

    Parameters
    ----------
    min_season : int, optional
        If provided, only build weeks with season >= min_season.
    max_season : int, optional
        If provided, only build weeks with season <= max_season.
    overwrite : bool, default False
        If False, weeks that already have rows in the features table
        are skipped. If True, build_all will rebuild those weeks.
    persist : bool, default True
        Passed through to build_features_for_week. Typically True in CLI mode.
    """
    weeks = _list_weeks(min_season=min_season, max_season=max_season)
    if not weeks:
        logger.warning("No (season, week) combos found in player_game_stats.")
        return

    total = len(weeks)
    logger.info("Building features for %d season/week combinations.", total)

    total_rows_written = 0
    next_log_threshold = 10_000  # log every 10k feature rows written

    for idx, (season, week) in enumerate(weeks, start=1):
        if not overwrite and _features_exist_for_week(season, week):
            logger.info(
                "[%d/%d] Skipping season=%s week=%s (features already exist).",
                idx,
                total,
                season,
                week,
            )
            continue

        logger.info(
            "[%d/%d] Building features for season=%s week=%s ...",
            idx,
            total,
            season,
            week,
        )

        try:
            df_week = build_features_for_week(season, week, persist=persist)
        except Exception:
            logger.exception(
                "Failed to build features for season=%s week=%s.",
                season,
                week,
            )
            continue

        n_rows, n_cols = df_week.shape
        logger.info(
            "Built features for season=%s week=%s with shape (%d, %d).",
            season,
            week,
            n_rows,
            n_cols,
        )

        if persist:
            total_rows_written += n_rows

            # Estimate how many actual feature values (cells) we’ve written, just for info
            num_id_cols = len(getattr(PLAYER_GAME_FEATURES_SCHEMA, "identifiers", []))
            num_label_cols = len(getattr(PLAYER_GAME_FEATURES_SCHEMA, "labels", []))
            num_feature_cols = max(0, n_cols - num_id_cols - num_label_cols)
            approx_feature_values = total_rows_written * num_feature_cols

            # Log a progress update every 10,000 rows
            if total_rows_written >= next_log_threshold:
                logger.info(
                    "Cumulative progress: %d player-game feature rows written "
                    "(~%d feature values across all columns).",
                    total_rows_written,
                    approx_feature_values,
                )
                next_log_threshold += 10_000


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build player-game features for all available seasons/weeks."
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Only build weeks with season >= this value.",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Only build weeks with season <= this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild weeks even if rows already exist in the features table.",
    )

    args = parser.parse_args()

    build_all_features(
        min_season=args.min_season,
        max_season=args.max_season,
        overwrite=args.overwrite,
        persist=True,
    )
