#!/usr/bin/env python3
"""
Populate team_defense_game_stats table

This script computes team defense statistics from opponent offensive performance.
For each game, we aggregate how the opposing offense performed against this defense,
which gives us the defensive stats (yards allowed, TDs allowed, etc.).

Data sources:
- nflverse_weekly_stats: Offensive performance by position
- games: Game metadata for team matchups
- player_game_stats: To get opponent context

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Compute database path directly
SCRIPT_DIR = Path(__file__).parent
PHASE1_DIR = SCRIPT_DIR.parent
DATABASE_PATH = PHASE1_DIR / "database" / "nfl_data.db"


def get_connection():
    """Get a database connection."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DATABASE_PATH}")
    return sqlite3.connect(str(DATABASE_PATH))


def populate_team_defense_game_stats():
    """
    Populate team_defense_game_stats by computing defense-allowed metrics
    from opponent offensive performance.

    For each (defense_team, game), we sum up what the opposing offense did:
    - Total yards allowed (passing + rushing)
    - Yards allowed by position (WR, TE, RB)
    - Targets allowed by position
    - TDs allowed by position
    - Points allowed (from game scores if available)
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("Populating team_defense_game_stats...")

    # First, check if we have game data with team matchups
    cursor.execute("SELECT COUNT(*) FROM games")
    game_count = cursor.fetchone()[0]
    print(f"Found {game_count} games in database")

    # Get all unique (season, week, team) combinations from player_game_stats
    # This tells us which teams played in which weeks
    cursor.execute("""
        SELECT DISTINCT
            season,
            week,
            team_id,
            opponent_team_id,
            game_id
        FROM player_game_stats
        WHERE game_id IS NOT NULL
        ORDER BY season, week
    """)
    team_games = cursor.fetchall()
    print(f"Found {len(team_games)} team-game records")

    # For each team's game, compute what the OPPONENT did against them
    # (i.e., what the defense allowed)

    # Aggregate offensive stats by opponent to get defensive allowed stats
    # Note: nflverse uses different team abbreviations, need to handle that

    insert_sql = """
        INSERT OR REPLACE INTO team_defense_game_stats (
            team_id, game_id, season, week, opponent_team_id,
            points_allowed, yards_allowed_passing, yards_allowed_rushing,
            yards_allowed_total, yards_allowed_to_wr, yards_allowed_to_te,
            yards_allowed_to_rb, targets_allowed_to_wr, targets_allowed_to_te,
            targets_allowed_to_rb, tds_allowed_to_wr, tds_allowed_to_te,
            tds_allowed_to_rb, sacks, interceptions,
            raw_json, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Build aggregated defensive stats from opponent offensive performance
    # We need to aggregate nflverse_weekly_stats by (opponent's team, season, week)
    # grouped by position

    # First get the opponent aggregates per game
    agg_query = """
        SELECT
            pgs.opponent_team_id as defense_team,
            pgs.team_id as offense_team,
            pgs.game_id,
            pgs.season,
            pgs.week,
            -- Aggregate passing yards allowed (from QB performance against this defense)
            SUM(CASE WHEN nfl.position = 'QB' THEN COALESCE(nfl.passing_yards, 0) ELSE 0 END) as passing_yards_allowed,
            -- Aggregate rushing yards allowed
            SUM(COALESCE(nfl.rushing_yards, 0)) as rushing_yards_allowed,
            -- WR stats allowed
            SUM(CASE WHEN nfl.position = 'WR' THEN COALESCE(nfl.receiving_yards, 0) ELSE 0 END) as wr_yards_allowed,
            SUM(CASE WHEN nfl.position = 'WR' THEN COALESCE(nfl.targets, 0) ELSE 0 END) as wr_targets_allowed,
            SUM(CASE WHEN nfl.position = 'WR' THEN COALESCE(nfl.receiving_tds, 0) ELSE 0 END) as wr_tds_allowed,
            -- TE stats allowed
            SUM(CASE WHEN nfl.position = 'TE' THEN COALESCE(nfl.receiving_yards, 0) ELSE 0 END) as te_yards_allowed,
            SUM(CASE WHEN nfl.position = 'TE' THEN COALESCE(nfl.targets, 0) ELSE 0 END) as te_targets_allowed,
            SUM(CASE WHEN nfl.position = 'TE' THEN COALESCE(nfl.receiving_tds, 0) ELSE 0 END) as te_tds_allowed,
            -- RB stats allowed (receiving)
            SUM(CASE WHEN nfl.position = 'RB' THEN COALESCE(nfl.receiving_yards, 0) ELSE 0 END) as rb_rec_yards_allowed,
            SUM(CASE WHEN nfl.position = 'RB' THEN COALESCE(nfl.targets, 0) ELSE 0 END) as rb_targets_allowed,
            SUM(CASE WHEN nfl.position = 'RB' THEN COALESCE(nfl.receiving_tds, 0) + COALESCE(nfl.rushing_tds, 0) ELSE 0 END) as rb_tds_allowed,
            -- Sacks (from QB sacks taken)
            SUM(CASE WHEN nfl.position = 'QB' THEN COALESCE(nfl.sacks, 0) ELSE 0 END) as sacks,
            -- Interceptions thrown (defensive INT)
            SUM(CASE WHEN nfl.position = 'QB' THEN COALESCE(nfl.interceptions, 0) ELSE 0 END) as interceptions
        FROM player_game_stats pgs
        LEFT JOIN player_id_mapping pim ON pgs.player_id = pim.sleeper_id
        LEFT JOIN nflverse_weekly_stats nfl
            ON pim.gsis_id = nfl.player_id
            AND pgs.season = nfl.season
            AND pgs.week = nfl.week
        WHERE pgs.game_id IS NOT NULL
        GROUP BY pgs.opponent_team_id, pgs.team_id, pgs.game_id, pgs.season, pgs.week
    """

    print("Aggregating defensive stats from opponent offensive performance...")
    cursor.execute(agg_query)
    agg_results = cursor.fetchall()
    print(f"Generated {len(agg_results)} team-game defensive aggregates")

    # Insert the aggregated data
    inserted = 0
    for row in agg_results:
        (defense_team, offense_team, game_id, season, week,
         pass_yards, rush_yards, wr_yards, wr_targets, wr_tds,
         te_yards, te_targets, te_tds, rb_yards, rb_targets, rb_tds,
         sacks, interceptions) = row

        total_yards = (pass_yards or 0) + (rush_yards or 0)

        cursor.execute(insert_sql, (
            defense_team,  # team_id (the defense)
            game_id,
            season,
            week,
            offense_team,  # opponent_team_id (the offense they faced)
            None,  # points_allowed - not available from this data
            pass_yards,
            rush_yards,
            total_yards,
            wr_yards,
            te_yards,
            rb_yards,
            wr_targets,
            te_targets,
            rb_targets,
            wr_tds,
            te_tds,
            rb_tds,
            sacks,
            interceptions,
            None,  # raw_json
            datetime.utcnow().isoformat()
        ))
        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} team_defense_game_stats records")

    # Verify the data
    cursor.execute("SELECT COUNT(*) FROM team_defense_game_stats")
    final_count = cursor.fetchone()[0]
    print(f"Final team_defense_game_stats row count: {final_count}")

    # Show sample data
    cursor.execute("""
        SELECT team_id, season, week, yards_allowed_total, yards_allowed_to_wr, sacks
        FROM team_defense_game_stats
        ORDER BY season DESC, week DESC
        LIMIT 5
    """)
    print("\nSample defensive stats:")
    for row in cursor.fetchall():
        print(f"  {row[0]} (S{row[1]} W{row[2]}): {row[3]} total yds, {row[4]} WR yds, {row[5]} sacks")

    conn.close()
    return inserted


def main():
    """Main entry point."""
    print("=" * 60)
    print("Team Defense Game Stats Population Script")
    print("=" * 60)

    # First ensure player_id_mapping exists
    print("\nStep 1: Ensuring player ID mapping exists...")
    conn = get_connection()
    cursor = conn.cursor()

    # Check if mapping table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='player_id_mapping'
    """)
    if not cursor.fetchone():
        print("Creating player_id_mapping table...")
        # Create the mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_id_mapping (
                sleeper_id TEXT PRIMARY KEY,
                gsis_id TEXT,
                player_name TEXT,
                position TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_id_mapping_gsis
            ON player_id_mapping(gsis_id)
        """)
        conn.commit()

    # Populate mapping if empty
    cursor.execute("SELECT COUNT(*) FROM player_id_mapping")
    if cursor.fetchone()[0] == 0:
        print("Populating player_id_mapping from players.metadata_json...")
        cursor.execute("""
            INSERT INTO player_id_mapping (sleeper_id, gsis_id, player_name, position)
            SELECT
                player_id,
                TRIM(json_extract(metadata_json, '$.gsis_id')),
                full_name,
                position
            FROM players
            WHERE metadata_json IS NOT NULL
              AND json_extract(metadata_json, '$.gsis_id') IS NOT NULL
        """)
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM player_id_mapping")
        print(f"Created {cursor.fetchone()[0]} player ID mappings")

    conn.close()

    # Now populate defensive stats
    print("\nStep 2: Populating team defense game stats...")
    count = populate_team_defense_game_stats()

    print("\n" + "=" * 60)
    print(f"Complete! Inserted {count} team defense records.")
    print("=" * 60)


if __name__ == "__main__":
    main()
