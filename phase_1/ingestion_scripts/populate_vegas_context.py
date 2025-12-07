#!/usr/bin/env python3
"""
Populate vegas_game_context from betting_lines

This script aggregates betting_lines data (spreads, totals, h2h) into
the vegas_game_context table format expected by Phase 2 features.

Author: NFL Fantasy Prediction Engine Team
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# Database path
SCRIPT_DIR = Path(__file__).parent
PHASE1_DIR = SCRIPT_DIR.parent
DATABASE_PATH = PHASE1_DIR / "database" / "nfl_data.db"


def get_connection():
    return sqlite3.connect(str(DATABASE_PATH))


def calculate_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds is None:
        return None
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_implied_total(total_line: float, spread: float, is_home: bool) -> float:
    """
    Calculate implied team total from total line and spread.

    Formula: team_implied_total = (total_line / 2) + (spread / 2) for underdog
             team_implied_total = (total_line / 2) - (spread / 2) for favorite

    If team is favored (negative spread from their perspective), they're expected to score more.
    """
    if total_line is None:
        return None

    # Spread is from home perspective (negative = home favored)
    # Home implied = (total + spread) / 2 (since spread is negative for favorite)
    # Wait, let's use simpler formula:
    # home_implied = (total - spread) / 2
    # away_implied = (total + spread) / 2
    if is_home:
        return (total_line - (spread or 0)) / 2
    else:
        return (total_line + (spread or 0)) / 2


def populate_vegas_game_context():
    """Populate vegas_game_context from betting_lines."""
    conn = get_connection()
    cursor = conn.cursor()

    print("Populating vegas_game_context...")

    # Get distinct games with betting data
    cursor.execute("""
        SELECT DISTINCT game_id, season, week, home_team, away_team, bookmaker
        FROM betting_lines
        WHERE game_id IS NOT NULL
        ORDER BY season, week, game_id
    """)
    games = cursor.fetchall()
    print(f"Found {len(games)} game-bookmaker combinations with betting data")

    # For each game, aggregate the betting lines
    insert_sql = """
        INSERT OR REPLACE INTO vegas_game_context (
            game_id, season, week, home_team, away_team,
            spread_line, home_spread_odds, away_spread_odds,
            total_line, over_odds, under_odds,
            home_ml_odds, away_ml_odds,
            home_implied_total, away_implied_total,
            home_implied_win_pct, away_implied_win_pct,
            bookmaker, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    for game_id, season, week, home_team, away_team, bookmaker in games:
        # Get spread (home team line)
        cursor.execute("""
            SELECT line_value, odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'spreads'
              AND outcome_name = ?
        """, (game_id, bookmaker, home_team))
        home_spread_row = cursor.fetchone()

        cursor.execute("""
            SELECT line_value, odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'spreads'
              AND outcome_name = ?
        """, (game_id, bookmaker, away_team))
        away_spread_row = cursor.fetchone()

        spread_line = home_spread_row[0] if home_spread_row else None
        home_spread_odds = home_spread_row[1] if home_spread_row else None
        away_spread_odds = away_spread_row[1] if away_spread_row else None

        # Get total
        cursor.execute("""
            SELECT line_value, odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'totals'
              AND outcome_name = 'Over'
        """, (game_id, bookmaker))
        over_row = cursor.fetchone()

        cursor.execute("""
            SELECT odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'totals'
              AND outcome_name = 'Under'
        """, (game_id, bookmaker))
        under_row = cursor.fetchone()

        total_line = over_row[0] if over_row else None
        over_odds = over_row[1] if over_row else None
        under_odds = under_row[0] if under_row else None

        # Get moneyline
        cursor.execute("""
            SELECT odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'h2h'
              AND outcome_name = ?
        """, (game_id, bookmaker, home_team))
        home_ml_row = cursor.fetchone()

        cursor.execute("""
            SELECT odds_american
            FROM betting_lines
            WHERE game_id = ? AND bookmaker = ? AND market_type = 'h2h'
              AND outcome_name = ?
        """, (game_id, bookmaker, away_team))
        away_ml_row = cursor.fetchone()

        home_ml_odds = home_ml_row[0] if home_ml_row else None
        away_ml_odds = away_ml_row[0] if away_ml_row else None

        # Calculate implied totals and win probabilities
        home_implied_total = calculate_implied_total(total_line, spread_line, True) if total_line else None
        away_implied_total = calculate_implied_total(total_line, spread_line, False) if total_line else None

        home_implied_win_pct = calculate_implied_probability(home_ml_odds)
        away_implied_win_pct = calculate_implied_probability(away_ml_odds)

        # Insert
        cursor.execute(insert_sql, (
            game_id, season, week, home_team, away_team,
            spread_line, home_spread_odds, away_spread_odds,
            total_line, over_odds, under_odds,
            home_ml_odds, away_ml_odds,
            home_implied_total, away_implied_total,
            home_implied_win_pct, away_implied_win_pct,
            bookmaker, datetime.utcnow().isoformat()
        ))
        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} vegas_game_context records")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM vegas_game_context")
    final_count = cursor.fetchone()[0]
    print(f"Final vegas_game_context row count: {final_count}")

    # Sample
    print("\nSample data:")
    cursor.execute("""
        SELECT game_id, spread_line, total_line, home_ml_odds, away_ml_odds,
               home_implied_total, away_implied_total
        FROM vegas_game_context
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: spread={row[1]}, total={row[2]}, "
              f"ML={row[3]}/{row[4]}, implied={row[5]:.1f}/{row[6]:.1f}" if row[5] else f"  {row}")

    conn.close()
    return inserted


if __name__ == "__main__":
    print("=" * 60)
    print("Populate Vegas Game Context")
    print("=" * 60)
    populate_vegas_game_context()
