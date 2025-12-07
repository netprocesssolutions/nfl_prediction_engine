#!/usr/bin/env python3
"""
Link betting_lines to games table

This script matches betting_lines records to games using:
- Team name -> abbreviation mapping from teams table
- Season and week matching
- Home/away team matching

Author: NFL Fantasy Prediction Engine Team
"""

import sqlite3
from pathlib import Path

# Database path
SCRIPT_DIR = Path(__file__).parent
PHASE1_DIR = SCRIPT_DIR.parent
DATABASE_PATH = PHASE1_DIR / "database" / "nfl_data.db"


def get_connection():
    """Get database connection."""
    return sqlite3.connect(str(DATABASE_PATH))


def build_team_name_mapping(cursor) -> dict:
    """Build mapping from full team name to abbreviation."""
    cursor.execute("SELECT team_id, team_name FROM teams")
    name_to_abbr = {}
    for row in cursor.fetchall():
        abbr, full_name = row
        name_to_abbr[full_name] = abbr
    return name_to_abbr


def link_betting_to_games():
    """Link betting_lines records to games by matching teams and week."""
    conn = get_connection()
    cursor = conn.cursor()

    print("Building team name mapping...")
    team_mapping = build_team_name_mapping(cursor)
    print(f"Found {len(team_mapping)} team mappings")

    # Get all games indexed by (season, week, home, away)
    print("\nIndexing games...")
    cursor.execute("""
        SELECT game_id, season, week, home_team_id, away_team_id
        FROM games
    """)
    games_index = {}
    for row in cursor.fetchall():
        game_id, season, week, home, away = row
        # Index by both orderings to handle any inconsistency
        key1 = (season, week, home, away)
        key2 = (season, week, away, home)
        games_index[key1] = game_id
        # Don't add reverse - we want to match home/away correctly
    print(f"Indexed {len(games_index)} games")

    # Update betting_lines with game_id
    print("\nLinking betting lines to games...")
    cursor.execute("""
        SELECT DISTINCT season, week, home_team, away_team
        FROM betting_lines
        WHERE game_id IS NULL
    """)
    distinct_matchups = cursor.fetchall()
    print(f"Found {len(distinct_matchups)} distinct matchups to link")

    linked = 0
    not_found = 0
    for season, week, home_full, away_full in distinct_matchups:
        # Map full names to abbreviations
        home_abbr = team_mapping.get(home_full)
        away_abbr = team_mapping.get(away_full)

        if not home_abbr or not away_abbr:
            # Try partial match
            for name, abbr in team_mapping.items():
                if home_full and home_full in name:
                    home_abbr = abbr
                if away_full and away_full in name:
                    away_abbr = abbr

        if not home_abbr or not away_abbr:
            not_found += 1
            if not_found <= 5:
                print(f"  Could not map: {home_full} vs {away_full}")
            continue

        # Look up game_id
        key = (season, week, home_abbr, away_abbr)
        game_id = games_index.get(key)

        if game_id:
            cursor.execute("""
                UPDATE betting_lines
                SET game_id = ?
                WHERE season = ?
                  AND week = ?
                  AND home_team = ?
                  AND away_team = ?
            """, (game_id, season, week, home_full, away_full))
            linked += cursor.rowcount
        else:
            not_found += 1
            if not_found <= 5:
                print(f"  No game found for: S{season} W{week} {home_abbr} vs {away_abbr}")

    conn.commit()

    # Report results
    print(f"\nLinked {linked} betting_lines records")
    print(f"Could not find games for {not_found} matchups")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM betting_lines WHERE game_id IS NOT NULL")
    linked_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM betting_lines")
    total_count = cursor.fetchone()[0]
    print(f"\nFinal: {linked_count}/{total_count} betting lines have game_id")

    # Show sample linked data
    print("\nSample linked data:")
    cursor.execute("""
        SELECT game_id, season, week, home_team, away_team, bookmaker, market_type
        FROM betting_lines
        WHERE game_id IS NOT NULL
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: S{row[1]} W{row[2]} {row[3]} vs {row[4]} ({row[5]} {row[6]})")

    conn.close()
    return linked_count


if __name__ == "__main__":
    print("=" * 60)
    print("Link Betting Lines to Games")
    print("=" * 60)
    link_betting_to_games()
