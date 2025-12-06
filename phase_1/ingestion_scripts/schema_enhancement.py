"""
Master Schema Enhancement for NFL Fantasy Prediction Engine

This script adds ALL new tables required for maximum prediction accuracy:
1. Betting tables (betting_lines, bets_v2)
2. Next Gen Stats (passing, rushing, receiving)
3. NFLverse weekly stats with EPA
4. Weather data
5. Snap count trends
6. Red zone usage
7. Vegas implied game context
8. Player advanced metrics
9. Injuries tracking

Run this AFTER the base create_schema.py has been run.

Author: NFL Fantasy Prediction Engine Team
Version: 2.0 Enhanced
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

logger = get_ingestion_logger("schema_enhancement")


# =============================================================================
# BETTING TABLES (Fixed for SQLite)
# =============================================================================

BETTING_LINES_TABLE = """
CREATE TABLE IF NOT EXISTS betting_lines (
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- External references
    odds_api_event_id TEXT,
    game_id TEXT,
    
    -- Time indexing
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Game info
    home_team TEXT,
    away_team TEXT,
    commence_time TEXT,
    
    -- Market details
    market_type TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    bookmaker_title TEXT,
    
    -- Player info (for player props)
    player_name TEXT,
    player_id TEXT,
    
    -- Line details
    outcome_name TEXT NOT NULL,
    outcome_description TEXT,
    line_value REAL,
    
    -- Odds
    odds_american INTEGER,
    odds_decimal REAL,
    
    -- Metadata
    odds_api_last_update TEXT,
    snapshot_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    is_historical INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys (soft - may not always match)
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
"""

BETS_V2_TABLE = """
CREATE TABLE IF NOT EXISTS bets_v2 (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Time indexing
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- References
    game_id TEXT,
    player_id TEXT,
    team_id TEXT,
    betting_line_id INTEGER,
    
    -- Bet details
    market_type TEXT NOT NULL,
    stat_key TEXT NOT NULL,
    operator TEXT DEFAULT '>',
    
    -- Source info
    bookmaker TEXT,
    player_name TEXT,
    
    -- Line and projection
    line_value REAL NOT NULL,
    odds_american INTEGER,
    odds_decimal REAL,
    
    -- Our analysis
    our_projection REAL NOT NULL,
    confidence REAL,
    expected_edge REAL,
    implied_probability REAL,
    
    -- Stake and payout
    stake REAL,
    potential_payout REAL,
    
    -- Parlay support
    is_parlay_leg INTEGER DEFAULT 0,
    parlay_group_id TEXT,
    
    -- Classification
    edge_bucket TEXT,
    
    -- Results (populated after grading)
    actual_value REAL,
    outcome TEXT DEFAULT 'pending',
    profit_loss REAL,
    edge_realized REAL,
    projection_error REAL,
    was_correct INTEGER,
    graded_at TEXT,
    
    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (betting_line_id) REFERENCES betting_lines(line_id)
);
"""


# =============================================================================
# NEXT GEN STATS TABLES
# =============================================================================

NGS_PASSING_TABLE = """
CREATE TABLE IF NOT EXISTS ngs_passing (
    player_id TEXT NOT NULL,
    player_display_name TEXT,
    player_gsis_id TEXT,
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Time metrics (CRITICAL for predicting efficiency)
    avg_time_to_throw REAL,
    avg_time_in_pocket REAL,
    
    -- Air yards (CRITICAL for pass-first game scripts)
    avg_completed_air_yards REAL,
    avg_intended_air_yards REAL,
    avg_air_yards_differential REAL,
    avg_air_yards_to_sticks REAL,
    
    -- Aggressiveness (predicts deep ball attempts)
    aggressiveness REAL,
    max_completed_air_distance REAL,
    
    -- Expected metrics (BEST predictors)
    completion_percentage REAL,
    expected_completion_percentage REAL,
    completion_percentage_above_expectation REAL,
    
    -- Efficiency
    passer_rating REAL,
    
    -- Volume
    attempts INTEGER,
    completions INTEGER,
    pass_yards INTEGER,
    pass_touchdowns INTEGER,
    interceptions INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""

NGS_RUSHING_TABLE = """
CREATE TABLE IF NOT EXISTS ngs_rushing (
    player_id TEXT NOT NULL,
    player_display_name TEXT,
    player_gsis_id TEXT,
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Efficiency metrics (CRITICAL - isolates RB skill from O-line)
    efficiency REAL,
    percent_attempts_gte_eight_defenders REAL,
    
    -- Expected metrics (BEST predictors for RB performance)
    avg_rush_yards REAL,
    expected_rush_yards REAL,
    rush_yards_over_expected REAL,
    rush_yards_over_expected_per_att REAL,
    rush_pct_over_expected REAL,
    
    -- Speed metrics
    avg_time_to_los REAL,
    
    -- Volume
    rush_attempts INTEGER,
    rush_yards INTEGER,
    rush_touchdowns INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""

NGS_RECEIVING_TABLE = """
CREATE TABLE IF NOT EXISTS ngs_receiving (
    player_id TEXT NOT NULL,
    player_display_name TEXT,
    player_gsis_id TEXT,
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Coverage metrics (CRITICAL - #1 predictor of target share)
    avg_cushion REAL,
    avg_separation REAL,
    
    -- Air yards (predicts big play potential)
    avg_intended_air_yards REAL,
    percent_share_of_intended_air_yards REAL,
    
    -- Catch metrics
    catch_percentage REAL,
    
    -- YAC metrics (separates elite from average)
    avg_yac REAL,
    avg_expected_yac REAL,
    avg_yac_above_expectation REAL,
    
    -- Volume
    targets INTEGER,
    receptions INTEGER,
    receiving_yards INTEGER,
    receiving_touchdowns INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


# =============================================================================
# NFLVERSE WEEKLY STATS WITH EPA
# =============================================================================

NFLVERSE_WEEKLY_TABLE = """
CREATE TABLE IF NOT EXISTS nflverse_weekly_stats (
    player_id TEXT NOT NULL,
    player_name TEXT,
    player_display_name TEXT,
    position TEXT,
    position_group TEXT,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Passing stats
    completions INTEGER DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    passing_yards INTEGER DEFAULT 0,
    passing_tds INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    sacks INTEGER DEFAULT 0,
    sack_yards INTEGER DEFAULT 0,
    sack_fumbles INTEGER DEFAULT 0,
    sack_fumbles_lost INTEGER DEFAULT 0,
    passing_air_yards INTEGER DEFAULT 0,
    passing_yards_after_catch INTEGER DEFAULT 0,
    passing_first_downs INTEGER DEFAULT 0,
    passing_2pt_conversions INTEGER DEFAULT 0,
    
    -- Rushing stats
    carries INTEGER DEFAULT 0,
    rushing_yards INTEGER DEFAULT 0,
    rushing_tds INTEGER DEFAULT 0,
    rushing_fumbles INTEGER DEFAULT 0,
    rushing_fumbles_lost INTEGER DEFAULT 0,
    rushing_first_downs INTEGER DEFAULT 0,
    rushing_2pt_conversions INTEGER DEFAULT 0,
    
    -- Receiving stats
    receptions INTEGER DEFAULT 0,
    targets INTEGER DEFAULT 0,
    receiving_yards INTEGER DEFAULT 0,
    receiving_tds INTEGER DEFAULT 0,
    receiving_fumbles INTEGER DEFAULT 0,
    receiving_fumbles_lost INTEGER DEFAULT 0,
    receiving_air_yards INTEGER DEFAULT 0,
    receiving_yards_after_catch INTEGER DEFAULT 0,
    receiving_first_downs INTEGER DEFAULT 0,
    receiving_2pt_conversions INTEGER DEFAULT 0,
    
    -- Usage shares (CRITICAL for projections)
    target_share REAL,
    air_yards_share REAL,
    wopr REAL,  -- Weighted Opportunity Rating
    racr REAL,  -- Receiver Air Conversion Ratio
    
    -- EPA metrics (THE BEST EFFICIENCY PREDICTORS)
    passing_epa REAL,
    rushing_epa REAL,
    receiving_epa REAL,
    
    -- Fantasy points (for validation)
    fantasy_points REAL,
    fantasy_points_ppr REAL,
    
    -- Special teams
    special_teams_tds INTEGER DEFAULT 0,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


# =============================================================================
# SNAP COUNTS (Critical for usage prediction)
# =============================================================================

SNAP_COUNTS_TABLE = """
CREATE TABLE IF NOT EXISTS snap_counts (
    player_id TEXT NOT NULL,
    game_id TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Team info
    team TEXT,
    opponent TEXT,
    
    -- Snap counts
    offense_snaps INTEGER DEFAULT 0,
    offense_pct REAL,
    defense_snaps INTEGER DEFAULT 0,
    defense_pct REAL,
    special_teams_snaps INTEGER DEFAULT 0,
    special_teams_pct REAL,
    
    -- Derived metrics
    snap_share_rank INTEGER,  -- Rank within team
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


# =============================================================================
# RED ZONE STATS (Critical for TD prediction)
# =============================================================================

REDZONE_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS redzone_stats (
    player_id TEXT NOT NULL,
    game_id TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Inside 20 usage
    rz_targets INTEGER DEFAULT 0,
    rz_receptions INTEGER DEFAULT 0,
    rz_receiving_yards INTEGER DEFAULT 0,
    rz_receiving_tds INTEGER DEFAULT 0,
    rz_carries INTEGER DEFAULT 0,
    rz_rushing_yards INTEGER DEFAULT 0,
    rz_rushing_tds INTEGER DEFAULT 0,
    rz_pass_attempts INTEGER DEFAULT 0,
    rz_completions INTEGER DEFAULT 0,
    rz_passing_yards INTEGER DEFAULT 0,
    rz_passing_tds INTEGER DEFAULT 0,
    
    -- Inside 10 (goal line)
    gl_targets INTEGER DEFAULT 0,
    gl_carries INTEGER DEFAULT 0,
    gl_pass_attempts INTEGER DEFAULT 0,
    gl_tds INTEGER DEFAULT 0,
    
    -- Team context
    team_rz_opportunities INTEGER DEFAULT 0,
    player_rz_share REAL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


# =============================================================================
# WEATHER DATA (Impacts passing games)
# =============================================================================

WEATHER_TABLE = """
CREATE TABLE IF NOT EXISTS game_weather (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Location
    stadium TEXT,
    roof_type TEXT,  -- dome, outdoors, retractable open, retractable closed
    surface TEXT,    -- grass, turf
    
    -- Weather conditions (only for outdoor games)
    temperature_f REAL,
    feels_like_f REAL,
    humidity_pct REAL,
    wind_speed_mph REAL,
    wind_gust_mph REAL,
    wind_direction TEXT,
    precipitation_in REAL,
    precipitation_type TEXT,  -- rain, snow, none
    visibility_miles REAL,
    
    -- Weather impact score (calculated)
    weather_impact_score REAL,  -- 0 = no impact, 100 = severe
    
    -- Source
    data_source TEXT,
    forecast_time TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""


# =============================================================================
# INJURIES (Critical for projections)
# =============================================================================

INJURIES_TABLE = """
CREATE TABLE IF NOT EXISTS injuries (
    injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    player_id TEXT NOT NULL,
    player_name TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Team
    team TEXT,
    position TEXT,
    
    -- Injury details
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status TEXT,  -- Questionable, Doubtful, Out, IR
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status TEXT,  -- Did Not Practice, Limited, Full
    
    -- Dates
    date_modified TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
"""


# =============================================================================
# VEGAS GAME CONTEXT (Use lines as features)
# =============================================================================

VEGAS_GAME_CONTEXT_TABLE = """
CREATE TABLE IF NOT EXISTS vegas_game_context (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Teams
    home_team TEXT,
    away_team TEXT,
    
    -- Spread (positive = home underdog)
    spread_line REAL,
    home_spread_odds INTEGER,
    away_spread_odds INTEGER,
    
    -- Total
    total_line REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    
    -- Moneyline
    home_ml_odds INTEGER,
    away_ml_odds INTEGER,
    
    -- Implied metrics (calculated)
    home_implied_total REAL,
    away_implied_total REAL,
    home_implied_win_pct REAL,
    away_implied_win_pct REAL,
    
    -- Movement (compared to opening line if available)
    spread_open REAL,
    total_open REAL,
    spread_movement REAL,
    total_movement REAL,
    
    -- Snapshot info
    bookmaker TEXT,
    snapshot_time TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""


# =============================================================================
# PLAYER SEASON AVERAGES (Rolling calculations)
# =============================================================================

PLAYER_SEASON_AVERAGES_TABLE = """
CREATE TABLE IF NOT EXISTS player_season_averages (
    player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    through_week INTEGER NOT NULL,  -- Averages through this week
    
    -- Position
    position TEXT,
    team TEXT,
    
    -- Games played
    games_played INTEGER,
    
    -- Per-game averages
    avg_fantasy_points_ppr REAL,
    avg_snaps REAL,
    avg_snap_pct REAL,
    
    -- Passing averages
    avg_pass_attempts REAL,
    avg_completions REAL,
    avg_passing_yards REAL,
    avg_passing_tds REAL,
    avg_interceptions REAL,
    avg_passing_epa REAL,
    
    -- Rushing averages
    avg_carries REAL,
    avg_rushing_yards REAL,
    avg_rushing_tds REAL,
    avg_rushing_epa REAL,
    
    -- Receiving averages
    avg_targets REAL,
    avg_receptions REAL,
    avg_receiving_yards REAL,
    avg_receiving_tds REAL,
    avg_receiving_epa REAL,
    avg_target_share REAL,
    avg_air_yards_share REAL,
    
    -- Red zone averages
    avg_rz_opportunities REAL,
    avg_rz_tds REAL,
    
    -- Trend indicators (last 3 games vs season)
    recent_form_score REAL,  -- Positive = trending up
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, through_week)
);
"""


# =============================================================================
# TEAM PACE AND TENDENCY (Game script prediction)
# =============================================================================

TEAM_TENDENCIES_TABLE = """
CREATE TABLE IF NOT EXISTS team_tendencies (
    team_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    through_week INTEGER NOT NULL,
    
    -- Pace metrics
    plays_per_game REAL,
    seconds_per_play REAL,
    
    -- Run/pass tendency
    pass_rate REAL,
    neutral_pass_rate REAL,  -- When score is close
    pass_rate_ahead REAL,
    pass_rate_behind REAL,
    
    -- Target distribution
    wr_target_share REAL,
    rb_target_share REAL,
    te_target_share REAL,
    
    -- Rush distribution
    rb1_carry_share REAL,
    rb2_carry_share REAL,
    
    -- Red zone tendency
    rz_pass_rate REAL,
    rz_plays_per_game REAL,
    
    -- Efficiency context
    points_per_game REAL,
    yards_per_game REAL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (team_id, season, through_week)
);
"""


# =============================================================================
# COMBINE DATA (For archetypes and physical profiles)
# =============================================================================

COMBINE_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS combine_data (
    player_id TEXT,
    player_name TEXT,
    season INTEGER NOT NULL,  -- Draft year
    
    -- Basic info
    position TEXT,
    school TEXT,
    
    -- Physical measurements
    height TEXT,
    weight INTEGER,
    hand_size REAL,
    arm_length REAL,
    wingspan REAL,
    
    -- Athletic testing
    forty_yard REAL,
    bench_press INTEGER,  -- Reps
    vertical_jump REAL,
    broad_jump INTEGER,
    three_cone REAL,
    shuttle REAL,
    
    -- Draft info
    draft_team TEXT,
    draft_round INTEGER,
    draft_pick INTEGER,
    draft_overall INTEGER,
    
    -- Calculated metrics
    speed_score REAL,  -- (Weight * 200) / (40-time^4)
    bmi REAL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_name, season)
);
"""


# =============================================================================
# ROSTERS (Weekly roster status)
# =============================================================================

ROSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS rosters (
    player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Player info
    player_name TEXT,
    position TEXT,
    team TEXT,
    
    -- Roster status
    status TEXT,  -- ACT, INA, IR, PUP, etc.
    depth_chart_position TEXT,
    jersey_number INTEGER,
    
    -- Contract/experience
    years_exp INTEGER,
    rookie_year INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


# =============================================================================
# SCHEDULES (For rest/travel calculations)
# =============================================================================

SCHEDULES_TABLE = """
CREATE TABLE IF NOT EXISTS schedules (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    game_type TEXT,  -- REG, POST
    week INTEGER NOT NULL,
    
    -- Teams
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    
    -- Game time
    gameday TEXT,
    gametime TEXT,
    weekday TEXT,
    
    -- Location
    location TEXT,
    
    -- Results (filled after game)
    home_score INTEGER,
    away_score INTEGER,
    result TEXT,
    total INTEGER,
    
    -- Spread result
    spread_line REAL,
    home_spread_result REAL,
    
    -- Total result
    total_line REAL,
    over_under_result TEXT,
    
    -- Stadium info
    roof TEXT,
    surface TEXT,
    
    -- Rest days
    home_rest_days INTEGER,
    away_rest_days INTEGER,
    
    -- Primetime
    is_primetime INTEGER DEFAULT 0,
    is_divisional INTEGER DEFAULT 0,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# =============================================================================
# INDEXES
# =============================================================================

ENHANCEMENT_INDEXES = [
    # Betting
    "CREATE INDEX IF NOT EXISTS idx_bl_season_week ON betting_lines(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_bl_game ON betting_lines(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_bl_player ON betting_lines(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bl_market ON betting_lines(market_type);",
    "CREATE INDEX IF NOT EXISTS idx_bl_bookmaker ON betting_lines(bookmaker);",
    "CREATE INDEX IF NOT EXISTS idx_bl_composite ON betting_lines(odds_api_event_id, market_type, bookmaker, outcome_name);",
    
    "CREATE INDEX IF NOT EXISTS idx_bets_season_week ON bets_v2(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_bets_player ON bets_v2(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_outcome ON bets_v2(outcome);",
    "CREATE INDEX IF NOT EXISTS idx_bets_edge_bucket ON bets_v2(edge_bucket);",
    "CREATE INDEX IF NOT EXISTS idx_bets_parlay ON bets_v2(parlay_group_id);",
    
    # NGS
    "CREATE INDEX IF NOT EXISTS idx_ngs_pass_season ON ngs_passing(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_ngs_pass_team ON ngs_passing(team_abbr);",
    "CREATE INDEX IF NOT EXISTS idx_ngs_rush_season ON ngs_rushing(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_ngs_rush_team ON ngs_rushing(team_abbr);",
    "CREATE INDEX IF NOT EXISTS idx_ngs_rec_season ON ngs_receiving(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_ngs_rec_team ON ngs_receiving(team_abbr);",
    
    # NFLverse weekly
    "CREATE INDEX IF NOT EXISTS idx_nflverse_season ON nflverse_weekly_stats(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_nflverse_team ON nflverse_weekly_stats(team);",
    "CREATE INDEX IF NOT EXISTS idx_nflverse_position ON nflverse_weekly_stats(position);",
    
    # Snap counts
    "CREATE INDEX IF NOT EXISTS idx_snaps_season ON snap_counts(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_snaps_team ON snap_counts(team);",
    
    # Red zone
    "CREATE INDEX IF NOT EXISTS idx_rz_season ON redzone_stats(season, week);",
    
    # Weather
    "CREATE INDEX IF NOT EXISTS idx_weather_season ON game_weather(season, week);",
    
    # Injuries
    "CREATE INDEX IF NOT EXISTS idx_injuries_season ON injuries(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team);",
    
    # Vegas context
    "CREATE INDEX IF NOT EXISTS idx_vegas_season ON vegas_game_context(season, week);",
    
    # Season averages
    "CREATE INDEX IF NOT EXISTS idx_avg_season ON player_season_averages(season, through_week);",
    
    # Team tendencies
    "CREATE INDEX IF NOT EXISTS idx_tendencies_season ON team_tendencies(season, through_week);",
    
    # Rosters
    "CREATE INDEX IF NOT EXISTS idx_rosters_season ON rosters(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(team);",
    
    # Schedules
    "CREATE INDEX IF NOT EXISTS idx_schedules_season ON schedules(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_schedules_teams ON schedules(home_team, away_team);",
]


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def create_enhancement_tables(db: DatabaseConnection, drop_existing: bool = False):
    """Create all enhancement tables."""
    
    tables = [
        ("betting_lines", BETTING_LINES_TABLE),
        ("bets_v2", BETS_V2_TABLE),
        ("ngs_passing", NGS_PASSING_TABLE),
        ("ngs_rushing", NGS_RUSHING_TABLE),
        ("ngs_receiving", NGS_RECEIVING_TABLE),
        ("nflverse_weekly_stats", NFLVERSE_WEEKLY_TABLE),
        ("snap_counts", SNAP_COUNTS_TABLE),
        ("redzone_stats", REDZONE_STATS_TABLE),
        ("game_weather", WEATHER_TABLE),
        ("injuries", INJURIES_TABLE),
        ("vegas_game_context", VEGAS_GAME_CONTEXT_TABLE),
        ("player_season_averages", PLAYER_SEASON_AVERAGES_TABLE),
        ("team_tendencies", TEAM_TENDENCIES_TABLE),
        ("combine_data", COMBINE_DATA_TABLE),
        ("rosters", ROSTERS_TABLE),
        ("schedules", SCHEDULES_TABLE),
    ]
    
    logger.info("Creating enhancement tables...")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        if drop_existing:
            logger.warning("Dropping existing enhancement tables!")
            for table_name, _ in reversed(tables):
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                logger.info(f"Dropped table: {table_name}")
        
        for table_name, create_sql in tables:
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")
        
        # Create indexes
        logger.info("Creating indexes...")
        for index_sql in ENHANCEMENT_INDEXES:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")
        
        logger.info(f"Created {len(ENHANCEMENT_INDEXES)} indexes")
    
    logger.info("Enhancement schema creation complete!", event="enhancement_complete")


def verify_enhancement_tables(db: DatabaseConnection) -> dict:
    """Verify all enhancement tables exist."""
    
    expected_tables = [
        "betting_lines", "bets_v2",
        "ngs_passing", "ngs_rushing", "ngs_receiving",
        "nflverse_weekly_stats", "snap_counts", "redzone_stats",
        "game_weather", "injuries", "vegas_game_context",
        "player_season_averages", "team_tendencies",
        "combine_data", "rosters", "schedules"
    ]
    
    results = {"all_exist": True, "tables": {}}
    
    for table_name in expected_tables:
        exists = db.table_exists(table_name)
        row_count = db.get_row_count(table_name) if exists else 0
        
        results["tables"][table_name] = {
            "exists": exists,
            "row_count": row_count
        }
        
        if not exists:
            results["all_exist"] = False
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create enhancement tables for maximum prediction power"
    )
    parser.add_argument("--drop", action="store_true", help="Drop existing tables first")
    parser.add_argument("--verify-only", action="store_true", help="Only verify tables")
    
    args = parser.parse_args()
    
    db = get_db()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Schema Enhancement")
    print(f"{'='*60}")
    print(f"Database: {db.db_path}")
    print(f"{'='*60}\n")
    
    if args.verify_only:
        results = verify_enhancement_tables(db)
        print(f"All tables exist: {results['all_exist']}\n")
        for table, info in results['tables'].items():
            status = "Ã¢Å“â€œ" if info['exists'] else "Ã¢Å“â€”"
            print(f"  {status} {table}: {info['row_count']} rows")
    else:
        if args.drop:
            confirm = input("WARNING: Drop existing tables? Type 'YES': ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        create_enhancement_tables(db, drop_existing=args.drop)
        
        results = verify_enhancement_tables(db)
        print(f"\nVerification:")
        for table, info in results['tables'].items():
            status = "Ã¢Å“â€œ" if info['exists'] else "Ã¢Å“â€”"
            print(f"  {status} {table}")
        
        if results['all_exist']:
            print("\nÃ¢Å“â€œ All enhancement tables created successfully!")
        else:
            print("\nÃ¢Å“â€” Some tables missing!")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
