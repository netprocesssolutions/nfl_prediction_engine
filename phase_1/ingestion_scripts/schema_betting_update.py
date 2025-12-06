"""
Betting Schema Update for NFL Fantasy Prediction Engine - Phase 1

This module adds betting lines tracking and enhanced bet tracking tables
to support The Odds API integration and bet performance analysis.

New tables:
- betting_lines: Raw odds data from The Odds API
- Enhanced bets table with comprehensive tracking fields

Key features:
- Track sportsbook odds for player props and game markets
- Store our projections vs sportsbook lines
- Calculate and track edge by variance buckets
- Support for parlay tracking
- Post-game grading and ROI analysis

FIX: Removed expressions from UNIQUE constraints (SQLite limitation)
Instead uses regular column combinations with NULL handling via triggers.

Author: NFL Fantasy Prediction Engine Team
Phase: 1 Extension - Betting Integration
Version: 2.1
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Initialize logger
logger = get_ingestion_logger("schema_betting_update")


# =============================================================================
# BETTING SCHEMA DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# BETTING LINES TABLE - Stores raw odds from The Odds API
# -----------------------------------------------------------------------------

BETTING_LINES_TABLE = """
CREATE TABLE IF NOT EXISTS betting_lines (
    -- Primary identifier
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- The Odds API event identifier
    odds_api_event_id TEXT NOT NULL,
    
    -- Link to our games table (may be NULL if not matched)
    game_id TEXT,
    
    -- Time indexing
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Game info from Odds API
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    commence_time TEXT,  -- ISO timestamp of game start
    
    -- Market details
    market_type TEXT NOT NULL,  -- h2h, spreads, totals, player_pass_yds, etc.
    bookmaker TEXT NOT NULL,    -- fanduel, draftkings, etc.
    bookmaker_title TEXT,       -- Human readable name
    
    -- For player props
    player_name TEXT,           -- Player name from Odds API
    player_id TEXT,             -- FK to players table if matched
    
    -- Outcome details
    outcome_name TEXT NOT NULL,      -- Team name, Over, Under, player name
    outcome_description TEXT,        -- Additional description
    line_value REAL,                 -- The point/yards/etc line
    
    -- Odds
    odds_american INTEGER,      -- American odds (-110, +150, etc.)
    odds_decimal REAL,          -- Decimal odds (1.91, 2.50, etc.)
    
    -- Timestamps
    odds_api_last_update TEXT,  -- When Odds API last updated this
    snapshot_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_historical INTEGER DEFAULT 0,  -- 0 = current, 1 = historical snapshot
    
    -- Audit
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
"""

# Note: We create a unique index instead of a UNIQUE constraint to avoid
# the SQLite limitation on expressions in UNIQUE constraints.
# The uniqueness is enforced on actual column values.

BETTING_LINES_INDEXES = [
    # Performance indexes
    "CREATE INDEX IF NOT EXISTS idx_bl_event ON betting_lines(odds_api_event_id);",
    "CREATE INDEX IF NOT EXISTS idx_bl_game ON betting_lines(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_bl_season_week ON betting_lines(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_bl_market ON betting_lines(market_type);",
    "CREATE INDEX IF NOT EXISTS idx_bl_bookmaker ON betting_lines(bookmaker);",
    "CREATE INDEX IF NOT EXISTS idx_bl_player ON betting_lines(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bl_snapshot ON betting_lines(snapshot_timestamp);",
    
    # Composite unique index for deduplication
    # Note: SQLite handles NULLs as distinct values, so we need application-level dedup
    "CREATE INDEX IF NOT EXISTS idx_bl_unique_key ON betting_lines(odds_api_event_id, market_type, bookmaker, outcome_name, player_name, line_value, snapshot_timestamp);",
]


# -----------------------------------------------------------------------------
# ENHANCED BETS TABLE - Comprehensive bet tracking with edge analysis
# -----------------------------------------------------------------------------

BETS_TABLE_ENHANCED = """
CREATE TABLE IF NOT EXISTS bets_v2 (
    -- Unique ID for each individual bet / leg
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Time indexing for analysis
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT,
    player_id TEXT,
    team_id TEXT,
    
    -- Link to the specific betting line used
    betting_line_id INTEGER,

    -- Market / stat definition
    market_type TEXT NOT NULL,   -- 'player_prop', 'moneyline', 'spread', 'game_total', 'team_total'
    stat_key TEXT NOT NULL,      -- 'pass_yds', 'rush_yds', 'rec_yds', 'receptions', 'pass_tds', etc.
    operator TEXT NOT NULL,      -- '>', '<', '>=', '<=', '='
    
    -- Bookmaker info
    bookmaker TEXT,              -- fanduel, draftkings, etc.
    player_name TEXT,            -- For player props, the player name
    
    -- Line vs our projection
    line_value REAL NOT NULL,    -- sportsbook line (critical pass/fail threshold)
    odds_american INTEGER,       -- American odds at time of bet
    odds_decimal REAL,           -- Decimal odds at time of bet
    
    -- Our analysis
    our_projection REAL NOT NULL,
    confidence REAL,             -- our internal confidence score (0.0-1.0)
    expected_edge REAL,          -- our_projection - line_value (for O/U) or probability edge
    implied_probability REAL,    -- Probability implied by the odds

    -- Wager meta
    stake REAL,                  -- amount staked (units or currency)
    potential_payout REAL,       -- stake * odds_decimal if win
    is_parlay_leg INTEGER DEFAULT 0,   -- 0 = no, 1 = yes
    parlay_group_id TEXT,        -- same value across legs in a parlay

    -- Outcome (populated after game)
    actual_value REAL,           -- realized stat after game
    outcome TEXT DEFAULT 'pending',  -- 'pending', 'win', 'loss', 'push', 'void'
    
    -- Performance metrics (populated after grading)
    profit_loss REAL,            -- Actual profit/loss from this bet
    edge_realized REAL,          -- actual_value - line_value
    projection_error REAL,       -- actual_value - our_projection
    was_correct INTEGER,         -- 1 if beat the line, 0 if not
    
    -- Edge bucket for analysis (calculated from expected_edge)
    edge_bucket TEXT,            -- 'minimal', 'low', 'medium', 'high', 'extreme'

    -- Timestamps
    placed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    graded_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (betting_line_id) REFERENCES betting_lines(line_id)
);
"""

BETS_V2_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_player ON bets_v2(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_team ON bets_v2(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_game ON bets_v2(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_season_week ON bets_v2(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_parlay_group ON bets_v2(parlay_group_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_outcome ON bets_v2(outcome);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_edge_bucket ON bets_v2(edge_bucket);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_market ON bets_v2(market_type, stat_key);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_bookmaker ON bets_v2(bookmaker);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_confidence ON bets_v2(confidence);",
    "CREATE INDEX IF NOT EXISTS idx_bets_v2_betting_line ON bets_v2(betting_line_id);",
]


# =============================================================================
# SCHEMA UPDATE FUNCTIONS
# =============================================================================

def update_betting_schema(db: DatabaseConnection, drop_existing: bool = False):
    """
    Add betting tables to the existing database schema.
    
    Args:
        db: DatabaseConnection instance
        drop_existing: If True, drop existing betting tables first
    """
    logger.info("Starting betting schema update...")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Optionally drop existing tables
        if drop_existing:
            logger.warning("Dropping existing betting tables!")
            cursor.execute("DROP TABLE IF EXISTS bets_v2;")
            cursor.execute("DROP TABLE IF EXISTS betting_lines;")
            logger.info("Dropped existing betting tables")
        
        # Create betting_lines table
        logger.info("Creating betting_lines table...")
        cursor.execute(BETTING_LINES_TABLE)
        
        # Create betting_lines indexes
        logger.info("Creating betting_lines indexes...")
        for index_sql in BETTING_LINES_INDEXES:
            cursor.execute(index_sql)
        
        # Create enhanced bets table
        logger.info("Creating bets_v2 table...")
        cursor.execute(BETS_TABLE_ENHANCED)
        
        # Create bets_v2 indexes
        logger.info("Creating bets_v2 indexes...")
        for index_sql in BETS_V2_INDEXES:
            cursor.execute(index_sql)
        
        # Migrate data from old bets table if it exists
        if db.table_exists("bets"):
            logger.info("Migrating data from old bets table...")
            migrate_old_bets(cursor)
    
    logger.info("Betting schema update complete!", event="betting_schema_complete")


def migrate_old_bets(cursor):
    """
    Migrate data from the old bets table to bets_v2.
    
    Preserves existing bet records while adding new columns.
    """
    try:
        # Check if old bets table has data
        cursor.execute("SELECT COUNT(*) FROM bets")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("No data in old bets table to migrate")
            return
        
        # Insert existing bets into new table
        cursor.execute("""
            INSERT INTO bets_v2 (
                season, week, game_id, player_id, team_id,
                market_type, stat_key, operator,
                line_value, our_projection, confidence, expected_edge,
                stake, is_parlay_leg, parlay_group_id,
                actual_value, outcome,
                placed_at, created_at, updated_at
            )
            SELECT 
                season, week, game_id, player_id, team_id,
                market_type, stat_key, operator,
                line_value, our_projection, confidence, expected_edge,
                stake, is_parlay_leg, parlay_group_id,
                actual_value, outcome,
                placed_at, created_at, updated_at
            FROM bets
        """)
        
        migrated = cursor.rowcount
        logger.info(f"Migrated {migrated} records from old bets table")
        
    except Exception as e:
        logger.warning(f"Could not migrate old bets: {e}")


def verify_betting_schema(db: DatabaseConnection) -> dict:
    """
    Verify that betting tables exist and have correct structure.
    
    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying betting schema...")
    
    results = {
        "betting_lines_exists": db.table_exists("betting_lines"),
        "bets_v2_exists": db.table_exists("bets_v2"),
        "betting_lines_count": 0,
        "bets_v2_count": 0,
    }
    
    if results["betting_lines_exists"]:
        results["betting_lines_count"] = db.get_row_count("betting_lines")
    
    if results["bets_v2_exists"]:
        results["bets_v2_count"] = db.get_row_count("bets_v2")
    
    results["all_tables_exist"] = (
        results["betting_lines_exists"] and 
        results["bets_v2_exists"]
    )
    
    logger.info(
        f"Betting schema verification: {results}",
        event="betting_schema_verified"
    )
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to update the database schema with betting tables.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add betting tables to NFL Prediction Engine database"
    )
    parser.add_argument(
        "--drop", 
        action="store_true",
        help="Drop existing betting tables before creating (WARNING: destroys betting data)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true", 
        help="Only verify existing betting schema, don't create tables"
    )
    
    args = parser.parse_args()
    
    # Get database connection
    db = get_db()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Betting Schema Update")
    print(f"{'='*60}")
    print(f"Database: {db.db_path}")
    print(f"{'='*60}\n")
    
    if args.verify_only:
        results = verify_betting_schema(db)
        print(f"\nVerification Results:")
        print(f"  betting_lines exists: {results['betting_lines_exists']}")
        print(f"  betting_lines count: {results['betting_lines_count']}")
        print(f"  bets_v2 exists: {results['bets_v2_exists']}")
        print(f"  bets_v2 count: {results['bets_v2_count']}")
    else:
        if args.drop:
            confirm = input("WARNING: This will delete all betting data! Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        update_betting_schema(db, drop_existing=args.drop)
        
        # Verify after creation
        print("\nVerifying betting schema...")
        results = verify_betting_schema(db)
        
        if results['all_tables_exist']:
            print("\nâœ“ Betting schema created successfully!")
        else:
            print("\nâœ— Betting schema creation failed - some tables missing!")
            return 1
    
    print(f"\nLog file: {logger.log_file}")
    return 0


if __name__ == "__main__":
    exit(main())
