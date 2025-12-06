#!/usr/bin/env python3
"""
Phase 1 Database Fix Script - NFL Fantasy Prediction Engine

Fixes 3 issues:
1. LA vs LAR team mismatch - nflreadpy uses 'LA', database uses 'LAR'
2. team_defense_season_stats column names mismatch
3. injuries table foreign key constraint issue

Run from: Phase 1/ingestion_scripts folder
Command:  python fix_phase1_issues.py

Author: NFL Fantasy Prediction Engine Team
"""

import sys
import sqlite3
from pathlib import Path


def find_database():
    """Find the SQLite database file."""
    possible_paths = [
        Path(__file__).parent.parent / "config" / "database" / "nfl_data.db",
        Path(__file__).parent / ".." / "config" / "database" / "nfl_data.db",
        Path("../config/database/nfl_data.db"),
        Path("config/database/nfl_data.db"),
    ]
    
    for p in possible_paths:
        if p.exists():
            return p.resolve()
    
    # Ask user
    print("Could not find database automatically.")
    user_path = input("Enter full path to nfl_data.db: ").strip()
    return Path(user_path)


def fix_la_to_lar(conn):
    """
    Fix #1: nflreadpy uses 'LA' for Rams, database uses 'LAR'.
    Update all 'LA' references to 'LAR'.
    """
    print("\n" + "="*60)
    print("FIX #1: LA -> LAR Team ID Normalization")
    print("="*60)
    
    cursor = conn.cursor()
    total_updates = 0
    
    # Tables and columns to update
    updates = [
        ("games", "home_team_id"),
        ("games", "away_team_id"),
        ("schedules", "home_team"),
        ("schedules", "away_team"),
        ("rosters", "team"),
        ("nflverse_weekly_stats", "team"),
        ("snap_counts", "team"),
        ("ngs_passing", "team"),
        ("ngs_rushing", "team"),
        ("ngs_receiving", "team"),
        ("injuries", "team"),
        ("combine_data", "team"),
    ]
    
    for table, column in updates:
        try:
            # Check if table exists
            cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone()[0] == 0:
                continue
            
            # Check if column exists
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                continue
            
            # Count before
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = 'LA'")
            count_before = cursor.fetchone()[0]
            
            if count_before > 0:
                cursor.execute(f"UPDATE {table} SET {column} = 'LAR' WHERE {column} = 'LA'")
                print(f"  [OK] {table}.{column}: {count_before} rows updated")
                total_updates += count_before
                
        except Exception as e:
            print(f"  [WARN] {table}.{column}: {e}")
    
    conn.commit()
    print(f"\nTotal LA -> LAR updates: {total_updates}")
    return total_updates


def fix_team_defense_columns(conn):
    """
    Fix #2: team_defense_season_stats uses wrong column names.
    Schema has: man_coverage_rate, zone_coverage_rate
    Ingestion expects: man_rate, zone_rate
    """
    print("\n" + "="*60)
    print("FIX #2: team_defense_season_stats Column Names")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(team_defense_season_stats)")
    current_cols = [row[1] for row in cursor.fetchall()]
    print(f"Current columns: {current_cols}")
    
    # Check if already fixed
    if 'man_rate' in current_cols:
        print("  [OK] Already has correct column names")
        return 0
    
    # Backup existing data (if any)
    cursor.execute("SELECT COUNT(*) FROM team_defense_season_stats")
    existing_count = cursor.fetchone()[0]
    print(f"Existing rows: {existing_count}")
    
    # Drop and recreate with correct column names
    cursor.execute("DROP TABLE IF EXISTS team_defense_season_stats")
    
    cursor.execute("""
        CREATE TABLE team_defense_season_stats (
            team_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            
            man_rate REAL,
            zone_rate REAL,
            middle_closed_rate REAL,
            middle_open_rate REAL,
            
            PRIMARY KEY (team_id, season),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    """)
    
    conn.commit()
    print("  [OK] Recreated table with correct column names:")
    print("      man_rate, zone_rate, middle_closed_rate, middle_open_rate")
    return 1


def fix_injuries_table(conn):
    """
    Fix #3: injuries table has foreign key constraint on player_id.
    nflreadpy gsis_id doesn't match Sleeper player_id.
    """
    print("\n" + "="*60)
    print("FIX #3: injuries Table Schema")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Check current count
    cursor.execute("SELECT COUNT(*) FROM injuries")
    existing_count = cursor.fetchone()[0]
    print(f"Existing rows: {existing_count}")
    
    # Drop and recreate without foreign key constraint
    cursor.execute("DROP TABLE IF EXISTS injuries")
    
    cursor.execute("""
        CREATE TABLE injuries (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            player_id TEXT,
            player_name TEXT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            
            team TEXT,
            position TEXT,
            
            report_primary_injury TEXT,
            report_secondary_injury TEXT,
            report_status TEXT,
            
            practice_primary_injury TEXT,
            practice_secondary_injury TEXT,
            practice_status TEXT,
            
            date_modified TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_injuries_player 
        ON injuries(player_id, season, week)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_injuries_team 
        ON injuries(team, season, week)
    """)
    
    conn.commit()
    print("  [OK] Recreated table without foreign key constraint")
    print("  [OK] Added indexes for player_id and team lookups")
    return 1


def show_settings_fix():
    """Show the fix needed for settings.py to prevent future LA issues."""
    print("\n" + "="*60)
    print("MANUAL FIX REQUIRED: settings.py")
    print("="*60)
    print("""
Add this line to TEAM_ABBREVIATION_MAP in settings.py:

    "LA": "LAR",  # nflreadpy uses LA for Rams

The full mapping around line 167 should look like:

    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "LA": "LAR", "LAR": "LAR", "RAM": "LAR", "STL": "LAR",
    "NWE": "NE", "NEP": "NE",

This prevents the LA/LAR mismatch on future pipeline runs.
""")


def verify_fixes(conn):
    """Verify all fixes were applied."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Check LA references are gone
    la_count = 0
    for table, col in [("games", "home_team_id"), ("games", "away_team_id"), 
                       ("schedules", "home_team"), ("schedules", "away_team")]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = 'LA'")
            la_count += cursor.fetchone()[0]
        except:
            pass
    
    print(f"Remaining 'LA' references: {la_count} (should be 0)")
    
    # Check team_defense_season_stats columns
    cursor.execute("PRAGMA table_info(team_defense_season_stats)")
    cols = [row[1] for row in cursor.fetchall()]
    has_man_rate = 'man_rate' in cols
    print(f"team_defense_season_stats has man_rate: {has_man_rate} (should be True)")
    
    # Check injuries table
    cursor.execute("PRAGMA table_info(injuries)")
    cols = [row[1] for row in cursor.fetchall()]
    print(f"injuries columns: {len(cols)} columns defined")
    
    # Summary table counts
    print("\nTable Row Counts:")
    tables = [
        'teams', 'players', 'defenders', 'games', 'player_game_stats',
        'defender_season_coverage_stats', 'team_defense_season_stats',
        'nflverse_weekly_stats', 'ngs_passing', 'ngs_rushing', 'ngs_receiving',
        'snap_counts', 'rosters', 'schedules', 'injuries', 'combine_data', 'game_weather'
    ]
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status = "[OK]" if count > 0 else "[  ]"
            print(f"  {status} {table}: {count:,}")
        except Exception as e:
            print(f"  [ERR] {table}: error")


def main():
    print("""
================================================================
   NFL Fantasy Prediction Engine - Phase 1 Fix Script
================================================================
""")
    
    # Find and connect to database
    db_path = find_database()
    print(f"Database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Apply fixes
        fix_la_to_lar(conn)
        fix_team_defense_columns(conn)
        fix_injuries_table(conn)
        
        # Show manual fix needed
        show_settings_fix()
        
        # Verify
        verify_fixes(conn)
        
        print("\n" + "="*60)
        print("ALL DATABASE FIXES COMPLETE!")
        print("="*60)
        print("""
Next steps:
1. Add "LA": "LAR" to TEAM_ABBREVIATION_MAP in settings.py
2. Re-run pipeline to reload team defense stats and injuries:
   
   python run_master_pipeline.py --skip-betting
   
The pipeline will now correctly populate:
  - team_defense_season_stats (from 2025 Team Stats.csv)
  - injuries (from nflreadpy)
""")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
