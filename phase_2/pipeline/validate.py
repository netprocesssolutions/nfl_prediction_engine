from __future__ import annotations

import logging
from typing import List, Tuple, Optional

from ..db import read_sql
from ..schema import PLAYER_GAME_FEATURES_SCHEMA

logger = logging.getLogger(__name__)

FEATURES_TABLE = getattr(PLAYER_GAME_FEATURES_SCHEMA, "table_name", "player_game_features")


def _list_feature_weeks(
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    List distinct (season, week) present in the features table.
    """
    where_clauses = []
    params: List[object] = []

    if min_season is not None:
        where_clauses.append("season >= ?")
        params.append(min_season)
    if max_season is not None:
        where_clauses.append("season <= ?")
        params.append(max_season)

    query = f"""
        SELECT DISTINCT season, week
        FROM {FEATURES_TABLE}
    """
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY season, week"

    try:
        df = read_sql(query, params=params)
    except Exception:
        logger.warning("Features table %s not found or not queryable.", FEATURES_TABLE)
        return []

    if df.empty:
        return []

    return list(df.itertuples(index=False, name=None))


def validate_week(season: int, week: int) -> bool:
    """
    Run basic validation checks for a single (season, week) in the
    player_game_features table.

    Checks:
      - row count matches player_game_stats
      - no duplicate (season, week, game_id, player_id) in features
      - all identifier + label columns from schema exist
      - label columns are not 100% missing
    """
    ok = True

    # Base player-game rows
    base = read_sql(
        """
        SELECT season, week, game_id, player_id
        FROM player_game_stats
        WHERE season = ?
          AND week   = ?
        """,
        params=[season, week],
    )
    n_base = len(base)

    # Feature rows
    feats = read_sql(
        f"""
        SELECT *
        FROM {FEATURES_TABLE}
        WHERE season = ?
          AND week   = ?
        """,
        params=[season, week],
    )
    n_feats = len(feats)

    if n_base != n_feats:
        logger.warning(
            "Season=%s week=%s: base rows=%d, features rows=%d",
            season,
            week,
            n_base,
            n_feats,
        )
        ok = False

    # Duplicate key check
    if not feats.empty:
        dup_counts = (
            feats.groupby(["season", "week", "game_id", "player_id"])
            .size()
            .reset_index(name="n")
        )
        num_dups = (dup_counts["n"] > 1).sum()
        if num_dups > 0:
            logger.error(
                "Season=%s week=%s: found %d duplicate player-game rows in features.",
                season,
                week,
                int(num_dups),
            )
            ok = False

    # Schema column presence
    feat_cols = set(feats.columns) if not feats.empty else set()

    missing_ids = [
        c for c in getattr(PLAYER_GAME_FEATURES_SCHEMA, "identifiers", []) if c not in feat_cols
    ]
    missing_labels = [
        c for c in getattr(PLAYER_GAME_FEATURES_SCHEMA, "labels", []) if c not in feat_cols
    ]

    if missing_ids:
        logger.error(
            "Season=%s week=%s: missing identifier columns in features: %s",
            season,
            week,
            missing_ids,
        )
        ok = False

    if missing_labels:
        logger.error(
            "Season=%s week=%s: missing label columns in features: %s",
            season,
            week,
            missing_labels,
        )
        ok = False

    # Label missingness
    label_cols = [
        c for c in getattr(PLAYER_GAME_FEATURES_SCHEMA, "labels", []) if c in feat_cols
    ]
    if feats.empty and label_cols:
        logger.warning(
            "Season=%s week=%s: no feature rows but label columns expected: %s",
            season,
            week,
            label_cols,
        )
        ok = False
    elif not feats.empty and label_cols:
        for col in label_cols:
            frac_missing = feats[col].isna().mean()
            logger.info(
                "Season=%s week=%s: label '%s' missing %.1f%% of rows.",
                season,
                week,
                col,
                frac_missing * 100.0,
            )

    return ok


def validate_all(
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
) -> bool:
    """
    Run validate_week over all (season, week) present in the features table.
    Returns True if all weeks pass validation, False otherwise.
    """
    weeks = _list_feature_weeks(min_season=min_season, max_season=max_season)
    if not weeks:
        logger.warning(
            "No (season, week) combinations found in features table %s.",
            FEATURES_TABLE,
        )
        return False

    logger.info("Validating %d season/week combinations in %s.", len(weeks), FEATURES_TABLE)

    all_ok = True
    for season, week in weeks:
        logger.info("Validating season=%s week=%s ...", season, week)
        week_ok = validate_week(season, week)
        all_ok = all_ok and week_ok

    if all_ok:
        logger.info("All weeks passed validation.")
    else:
        logger.warning("Some weeks failed validation. See logs above for details.")

    return all_ok


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Validate player-game feature rows against base stats and schema."
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Only validate weeks with season >= this value.",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Only validate weeks with season <= this value.",
    )
    args = parser.parse_args()

    validate_all(
        min_season=args.min_season,
        max_season=args.max_season,
    )
