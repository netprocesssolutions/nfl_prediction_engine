"""
Database Schema Creation for NFL Fantasy Prediction Engine - Phase 1

This module creates the complete database schema as specified in Phase 1 v2.
All tables follow the exact structure defined in the Comprehensive Operational Plan.

Tables created:
- Entity Reference Tables: players, defenders, teams, seasons, weeks
- Game Metadata Tables: games, game_injuries (optional)
- Player Stats Tables: player_game_stats
- Defense Stats Tables: team_defense_game_stats, defender_game_stats
- Coverage Tables: coverage_events (optional)
- Versioning: data_versions

Key features:
- Foreign key constraints enforced
- Unique constraints for duplicate protection
- Proper indexing for query performance
- Anti-leakage columns (season, week, timestamp)

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Initialize logger
logger = get_ingestion_logger("create_schema")


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# ENTITY REFERENCE TABLES (Section 4.1 of Phase 1 v2)
# -----------------------------------------------------------------------------

TEAMS_TABLE = """
CREATE TABLE IF NOT EXISTS teams (
    -- Primary identifier for the team
    team_id TEXT PRIMARY KEY,
    
    -- Team information
    team_name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE,
    conference TEXT NOT NULL,
    division TEXT NOT NULL,
    
    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

PLAYERS_TABLE = """
CREATE TABLE IF NOT EXISTS players (
    -- Primary identifier from Sleeper (keep their unique string IDs)
    player_id TEXT PRIMARY KEY,
    
    -- Player information
    full_name TEXT NOT NULL,
    position TEXT NOT NULL,  -- QB, RB, WR, TE
    
    -- Team relationship (FK to teams)
    team_id TEXT,
    
    -- Physical attributes (required for archetypes per v2)
    height INTEGER,  -- inches
    weight INTEGER,  -- pounds
    age REAL,
    
    -- Additional info
    college TEXT,
    status TEXT,  -- active, IR, etc.
    
    -- Full raw data dump for extensibility (v2 addition)
    metadata_json TEXT,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""

DEFENDERS_TABLE = """
CREATE TABLE IF NOT EXISTS defenders (
    -- Primary identifier (unique defender ID)
    defender_id TEXT PRIMARY KEY,
    
    -- Defender information
    full_name TEXT NOT NULL,
    
    -- Team relationship
    team_id TEXT,
    
    -- Position grouping (v2 requirement)
    position_group TEXT NOT NULL,  -- CB, S, LB
    
    -- Role assignment (v2 requirement for coverage modeling)
    role TEXT,  -- boundary, slot, deep, box
    
    -- Physical attributes (for archetypes)
    height INTEGER,
    weight INTEGER,
    
    -- Coverage role (v2 requirement)
    coverage_role TEXT,  -- man, zone, hybrid
    
    -- Full raw data
    metadata_json TEXT,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""

SEASONS_TABLE = """
CREATE TABLE IF NOT EXISTS seasons (
    season_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

WEEKS_TABLE = """
CREATE TABLE IF NOT EXISTS weeks (
    week_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    
    -- Unique constraint per season
    UNIQUE(season_id, week_number),
    
    -- Foreign key
    FOREIGN KEY (season_id) REFERENCES seasons(season_id)
);
"""

# -----------------------------------------------------------------------------
# GAME METADATA TABLES (Section 4.2 of Phase 1 v2)
# -----------------------------------------------------------------------------

GAMES_TABLE = """
CREATE TABLE IF NOT EXISTS games (
    -- Primary identifier for the game
    game_id TEXT PRIMARY KEY,
    
    -- Time indexing (critical for anti-leakage)
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Teams involved
    home_team_id TEXT NOT NULL,
    away_team_id TEXT NOT NULL,
    
    -- Game metadata
    datetime TEXT,  -- ISO timestamp
    stadium TEXT,
    
    -- Weather data (v2 addition for environmental features)
    weather_json TEXT,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint (v2 requirement)
    UNIQUE(season, week, home_team_id, away_team_id),
    
    -- Foreign keys
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
"""

GAME_INJURIES_TABLE = """
CREATE TABLE IF NOT EXISTS game_injuries (
    -- Primary identifier
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- References
    player_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    
    -- Injury information
    injury_type TEXT,  -- e.g., hamstring
    status TEXT,  -- active, questionable, doubtful, out
    participation_expectation TEXT,  -- limited, full, no practice
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

# -----------------------------------------------------------------------------
# OFFENSIVE PLAYER STATS TABLE (Section 4.3 of Phase 1 v2)
# -----------------------------------------------------------------------------

PLAYER_GAME_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS player_game_stats (
    -- Composite primary key
    player_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    
    -- Team information
    team_id TEXT NOT NULL,
    opponent_team_id TEXT NOT NULL,
    
    -- Time indexing (critical for anti-leakage)
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Snap and usage stats
    snaps INTEGER DEFAULT 0,
    routes INTEGER DEFAULT 0,
    
    -- Rushing stats
    carries INTEGER DEFAULT 0,
    rush_yards REAL DEFAULT 0,
    rush_tds REAL DEFAULT 0,
    
    -- Receiving stats
    targets INTEGER DEFAULT 0,
    receptions INTEGER DEFAULT 0,
    rec_yards REAL DEFAULT 0,
    rec_tds REAL DEFAULT 0,
    
    -- Passing stats (for QBs)
    completions INTEGER DEFAULT 0,
    pass_attempts INTEGER DEFAULT 0,
    pass_yards REAL DEFAULT 0,
    pass_tds REAL DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    
    -- Other stats
    fumbles REAL DEFAULT 0,
    fumbles_lost REAL DEFAULT 0,
    two_point_conversions INTEGER DEFAULT 0,
    
    -- Fantasy points (from Sleeper)
    fantasy_points_sleeper REAL DEFAULT 0,
    
    -- Full raw stat dump (v2 requirement for reproducibility)
    raw_json TEXT,
    
    -- Anti-leakage timestamp
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key
    PRIMARY KEY (player_id, game_id),
    
    -- Foreign keys
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id)
);
"""

# -----------------------------------------------------------------------------
# TEAM DEFENSE STATS TABLE (Section 4.4 of Phase 1 v2)
# -----------------------------------------------------------------------------

TEAM_DEFENSE_GAME_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS team_defense_game_stats (
    -- Composite primary key
    team_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    
    -- Time indexing
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Opponent
    opponent_team_id TEXT NOT NULL,
    
    -- Points allowed
    points_allowed INTEGER DEFAULT 0,
    
    -- Yards allowed
    yards_allowed_passing REAL DEFAULT 0,
    yards_allowed_rushing REAL DEFAULT 0,
    yards_allowed_total REAL DEFAULT 0,
    
    -- Position-specific yards allowed (v2 requirement)
    yards_allowed_to_wr REAL DEFAULT 0,
    yards_allowed_to_te REAL DEFAULT 0,
    yards_allowed_to_rb REAL DEFAULT 0,
    
    -- Position-specific targets allowed
    targets_allowed_to_wr INTEGER DEFAULT 0,
    targets_allowed_to_te INTEGER DEFAULT 0,
    targets_allowed_to_rb INTEGER DEFAULT 0,
    
    -- Touchdowns allowed by position
    tds_allowed_to_wr REAL DEFAULT 0,
    tds_allowed_to_te REAL DEFAULT 0,
    tds_allowed_to_rb REAL DEFAULT 0,
    
    -- Efficiency metrics (v2 addition)
    redzone_defense_efficiency REAL,
    epa_allowed REAL,
    success_rate_allowed REAL,
    
    -- Explosive plays allowed
    explosive_plays_allowed INTEGER DEFAULT 0,
    
    -- Defensive stats
    sacks REAL DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    fumbles_recovered INTEGER DEFAULT 0,
    defensive_tds INTEGER DEFAULT 0,
    
    -- Raw data
    raw_json TEXT,
    
    -- Timestamps
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key
    PRIMARY KEY (team_id, game_id),
    
    -- Foreign keys
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id)
);
"""

# -----------------------------------------------------------------------------
# DEFENDER STATS TABLE (Section 4.5 of Phase 1 v2 - New in v2)
# -----------------------------------------------------------------------------

DEFENDER_GAME_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS defender_game_stats (
    -- Composite primary key
    defender_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    
    -- Time indexing
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Snap counts
    snaps INTEGER DEFAULT 0,
    coverage_snaps INTEGER DEFAULT 0,
    
    -- Coverage stats (v2 requirement)
    targets_allowed INTEGER DEFAULT 0,
    receptions_allowed INTEGER DEFAULT 0,
    yards_allowed REAL DEFAULT 0,
    yac_allowed REAL DEFAULT 0,
    ypr_allowed REAL DEFAULT 0,  -- yards per reception allowed
    tds_allowed REAL DEFAULT 0,
    
    -- Alignment percentages (v2 requirement for coverage modeling)
    alignment_boundary_pct REAL DEFAULT 0,
    alignment_slot_pct REAL DEFAULT 0,
    alignment_deep_pct REAL DEFAULT 0,
    alignment_box_pct REAL DEFAULT 0,
    
    -- Coverage type percentages (v2 requirement)
    man_coverage_pct REAL DEFAULT 0,
    zone_coverage_pct REAL DEFAULT 0,
    
    -- Other stats
    penalties INTEGER DEFAULT 0,
    pass_breakups INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    
    -- Raw data
    raw_json TEXT,
    
    -- Timestamps
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key
    PRIMARY KEY (defender_id, game_id),
    
    -- Foreign keys
    FOREIGN KEY (defender_id) REFERENCES defenders(defender_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

"""
DEFENDER_SEASON_COVERAGE_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS defender_season_coverage_stats (
    -- Composite key
    defender_id TEXT NOT NULL,
    season INTEGER NOT NULL,

    -- Team info (as of this season)
    team_id TEXT,

    -- Basic season info
    games INTEGER,
    games_started INTEGER,

    -- Coverage stats (season aggregate)
    targets INTEGER,
    completions INTEGER,
    completion_pct REAL,
    interceptions INTEGER,
    yards_allowed REAL,
    yards_per_completion REAL,
    rating_against REAL,
    air_yards_allowed REAL,
    yac_allowed REAL,

    -- Pressure / tackling
    blitzes INTEGER,
    hurries INTEGER,
    qb_hits INTEGER,
    passes_defensed INTEGER,
    sacks REAL,
    total_tackles REAL,
    missed_tackles REAL,
    missed_tackle_pct REAL,

    -- Raw data for audit / reparse
    raw_json TEXT,

    -- Primary key
    PRIMARY KEY (defender_id, season),

    -- Foreign keys
    FOREIGN KEY (defender_id) REFERENCES defenders(defender_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""

TEAM_DEFENSE_SEASON_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS team_defense_season_stats (
    team_id TEXT NOT NULL,
    season INTEGER NOT NULL,

    -- Coverage style
    man_coverage_rate REAL,
    zone_coverage_rate REAL,
    middle_closed_rate REAL,
    middle_open_rate REAL,

    -- Primary key
    PRIMARY KEY (team_id, season),

    -- Foreign key
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""

# -----------------------------------------------------------------------------
# PREDICTIONS TABLE (for model outputs & backtesting)
# -----------------------------------------------------------------------------

PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    -- Unique ID for each prediction
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Model identity
    model_version TEXT NOT NULL,

    -- Time indexing (match stats tables)
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Entity references (optional depending on prediction type)
    game_id TEXT,
    player_id TEXT,
    team_id TEXT,
    position TEXT,

    -- Prediction metadata
    prediction_type TEXT NOT NULL,    -- e.g., 'fantasy_points_ppr', 'prob_top12_rb'
    predicted_value REAL NOT NULL,    -- e.g., projected points or probability
    prediction_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Outcome & evaluation (populated after games complete)
    actual_value REAL,                -- actual fantasy points or outcome metric
    evaluation_metric TEXT,           -- e.g., 'fantasy_points_ppr', 'binary_hit'
    error REAL,                       -- e.g., actual_value - predicted_value

    -- Audit metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

"""

# -----------------------------------------------------------------------------
# COVERAGE EVENTS TABLE (Section 4.6 - Optional but recommended)
# -----------------------------------------------------------------------------

COVERAGE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS coverage_events (
    -- Primary identifier
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Player matchup
    offense_player_id TEXT NOT NULL,
    defender_id TEXT NOT NULL,
    
    -- Game reference
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Coverage details
    coverage_type TEXT,  -- man, zone
    alignment TEXT,  -- slot, boundary, etc.
    separation REAL,
    
    -- Raw data
    raw_json TEXT,
    
    -- Timestamps
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (offense_player_id) REFERENCES players(player_id),
    FOREIGN KEY (defender_id) REFERENCES defenders(defender_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

# -----------------------------------------------------------------------------
# BET TRACKING TABLE (Phase 1 extension: wager/edge tracking)
# -----------------------------------------------------------------------------

BETS_TABLE = """
CREATE TABLE IF NOT EXISTS bets (
    -- Unique ID for each individual bet / leg
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Time indexing for analysis
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    player_id TEXT,
    team_id TEXT,

    -- Market / stat definition
    market_type TEXT NOT NULL,   -- e.g., 'player_prop', 'moneyline', 'spread', 'game_total', 'team_total'
    stat_key TEXT NOT NULL,      -- e.g., 'passing_yards', 'team_points', 'point_diff', 'game_points', 'win_probability'
    operator TEXT NOT NULL,      -- e.g., '>', '<', '>=', '<=', '='

    -- Line vs our projection
    line_value REAL NOT NULL,    -- sportsbook line (critical pass/fail threshold)
    our_projection REAL NOT NULL,
    confidence REAL,             -- our internal confidence score (0-1 or 0-100)
    expected_edge REAL,          -- usually our_projection - line_value (or probability edge for moneyline)

    -- Wager meta
    stake REAL,                  -- amount staked (units or currency)
    is_parlay_leg INTEGER DEFAULT 0,   -- 0 = no, 1 = yes
    parlay_group_id TEXT,        -- same value across legs in a parlay

    -- Outcome
    actual_value REAL,           -- realized stat after game (yards, points, win=1/lose=0, etc.)
    outcome TEXT,                -- 'win', 'loss', 'push', 'void', 'pending'

    -- Timestamps
    placed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""

# -----------------------------------------------------------------------------
# DATA VERSIONING TABLE (Section 8 of Phase 1 v2)
# -----------------------------------------------------------------------------

DATA_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS data_versions (
    -- Version identifier (e.g., "2025_06")
    version_name TEXT PRIMARY KEY,
    
    -- Timestamp when ingestion completed
    timestamp TEXT NOT NULL,
    
    -- Optional notes about this version
    notes TEXT,
    
    -- Row counts for tracking
    offensive_row_count INTEGER DEFAULT 0,
    defensive_row_count INTEGER DEFAULT 0,
    defender_row_count INTEGER DEFAULT 0,
    
    -- Additional metadata
    games_count INTEGER DEFAULT 0,
    players_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# -----------------------------------------------------------------------------
# INDEXES FOR QUERY PERFORMANCE
# -----------------------------------------------------------------------------

INDEXES = [
    # Players indexes
    "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);",
    "CREATE INDEX IF NOT EXISTS idx_players_name ON players(full_name);",
    
    # Defenders indexes
    "CREATE INDEX IF NOT EXISTS idx_defenders_team ON defenders(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_defenders_position ON defenders(position_group);",
    
    # Games indexes
    "CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id);",
    "CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id);",
    
    # Player game stats indexes
    "CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_pgs_game ON player_game_stats(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_pgs_season_week ON player_game_stats(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_pgs_team ON player_game_stats(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_pgs_opponent ON player_game_stats(opponent_team_id);",
    
    # Team defense stats indexes
    "CREATE INDEX IF NOT EXISTS idx_tdgs_team ON team_defense_game_stats(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_tdgs_game ON team_defense_game_stats(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_tdgs_season_week ON team_defense_game_stats(season, week);",
    
    # Defender game stats indexes
    "CREATE INDEX IF NOT EXISTS idx_dgs_defender ON defender_game_stats(defender_id);",
    "CREATE INDEX IF NOT EXISTS idx_dgs_game ON defender_game_stats(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_dgs_season_week ON defender_game_stats(season, week);",
    
    # Coverage events indexes
    "CREATE INDEX IF NOT EXISTS idx_ce_offense ON coverage_events(offense_player_id);",
    "CREATE INDEX IF NOT EXISTS idx_ce_defender ON coverage_events(defender_id);",
    "CREATE INDEX IF NOT EXISTS idx_ce_game ON coverage_events(game_id);",

    # Predictions indexes
    "CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_season_week ON predictions(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);",

    # Bets indexes
    "CREATE INDEX IF NOT EXISTS idx_bets_player ON bets(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_team ON bets(team_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_season_week ON bets(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_bets_parlay_group ON bets(parlay_group_id);",
    
    # Game injuries indexes
    "CREATE INDEX IF NOT EXISTS idx_gi_player ON game_injuries(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_gi_game ON game_injuries(game_id);",
]


# =============================================================================
# SCHEMA CREATION FUNCTIONS
# =============================================================================

def create_all_tables(db: DatabaseConnection, drop_existing: bool = False):
    """
    Create all database tables as specified in Phase 1 v2.
    
    This function is idempotent - safe to run multiple times.
    Uses CREATE TABLE IF NOT EXISTS for all tables.
    
    Args:
        db: DatabaseConnection instance
        drop_existing: If True, drop all existing tables before creating.
                      USE WITH CAUTION - destroys all data!
    """
    logger.info("Starting schema creation...")
    
    tables = [
        ("teams", TEAMS_TABLE),
        ("players", PLAYERS_TABLE),
        ("defenders", DEFENDERS_TABLE),
        ("seasons", SEASONS_TABLE),
        ("weeks", WEEKS_TABLE),
        ("games", GAMES_TABLE),
        ("game_injuries", GAME_INJURIES_TABLE),
        ("player_game_stats", PLAYER_GAME_STATS_TABLE),
        ("team_defense_game_stats", TEAM_DEFENSE_GAME_STATS_TABLE),
        ("defender_game_stats", DEFENDER_GAME_STATS_TABLE),
        ("defender_season_coverage_stats", DEFENDER_SEASON_COVERAGE_STATS_TABLE),
        ("team_defense_season_stats", TEAM_DEFENSE_SEASON_STATS_TABLE),
        ("coverage_events", COVERAGE_EVENTS_TABLE),
        ("bets", BETS_TABLE),
        ("predictions", PREDICTIONS_TABLE),
        ("data_versions", DATA_VERSIONS_TABLE),
    ]
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Optionally drop existing tables
        if drop_existing:
            logger.warning("Dropping all existing tables!")
            # Drop in reverse order to handle foreign key constraints
            for table_name, _ in reversed(tables):
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                logger.info(f"Dropped table: {table_name}")
        
        # Create tables in order (entity tables first due to FK constraints)
        for table_name, create_sql in tables:
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}", event="table_created")
        
        # Create indexes
        logger.info("Creating indexes...")
        for index_sql in INDEXES:
            cursor.execute(index_sql)
        logger.info(f"Created {len(INDEXES)} indexes", event="indexes_created")
    
    logger.info("Schema creation complete!", event="schema_complete")


def verify_schema(db: DatabaseConnection) -> dict:
    """
    Verify that all required tables exist and have correct structure.
    
    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying database schema...")
    
    expected_tables = [
        "teams", "players", "defenders", "seasons", "weeks",
        "games", "game_injuries", "player_game_stats",
        "team_defense_game_stats", "defender_game_stats",
        "coverage_events", "data_versions"
    ]
    
    results = {
        "all_tables_exist": True,
        "tables": {},
        "total_rows": 0
    }
    
    for table_name in expected_tables:
        exists = db.table_exists(table_name)
        row_count = db.get_row_count(table_name) if exists else 0
        
        results["tables"][table_name] = {
            "exists": exists,
            "row_count": row_count
        }
        
        if not exists:
            results["all_tables_exist"] = False
            logger.error(f"Missing table: {table_name}")
        else:
            logger.info(f"Table {table_name}: {row_count} rows")
            results["total_rows"] += row_count
    
    logger.info(
        f"Schema verification complete. All tables exist: {results['all_tables_exist']}",
        event="schema_verified"
    )
    
    return results


def get_table_stats(db: DatabaseConnection) -> dict:
    """
    Get statistics about all tables in the database.
    
    Returns:
        Dictionary with table statistics
    """
    stats = {}
    
    tables = [
        "teams", "players", "defenders", "games",
        "player_game_stats", "team_defense_game_stats",
        "defender_game_stats", "coverage_events", "data_versions"
    ]
    
    for table in tables:
        if db.table_exists(table):
            stats[table] = {
                "row_count": db.get_row_count(table),
                "columns": [col['name'] for col in db.get_table_schema(table)]
            }
    
    return stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to create the database schema.
    
    This script can be run directly to initialize or reset the database.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create NFL Prediction Engine database schema"
    )
    parser.add_argument(
        "--drop", 
        action="store_true",
        help="Drop existing tables before creating (WARNING: destroys all data)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true", 
        help="Only verify existing schema, don't create tables"
    )
    
    args = parser.parse_args()
    
    # Get database connection
    db = get_db()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Phase 1")
    print("Database Schema Creation")
    print(f"{'='*60}")
    print(f"Database: {db.db_path}")
    print(f"{'='*60}\n")
    
    if args.verify_only:
        results = verify_schema(db)
        print(f"\nVerification Results:")
        print(f"  All tables exist: {results['all_tables_exist']}")
        print(f"  Total rows: {results['total_rows']}")
        for table, info in results['tables'].items():
            status = "âœ“" if info['exists'] else "âœ—"
            print(f"    {status} {table}: {info['row_count']} rows")
    else:
        if args.drop:
            confirm = input("WARNING: This will delete all data! Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        create_all_tables(db, drop_existing=args.drop)
        
        # Verify after creation
        print("\nVerifying schema...")
        results = verify_schema(db)
        
        if results['all_tables_exist']:
            print("\nâœ“ Schema created successfully!")
        else:
            print("\nâœ— Schema creation failed - some tables missing!")
            return 1
    
    print(f"\nLog file: {logger.log_file}")
    return 0


if __name__ == "__main__":
    exit(main())
