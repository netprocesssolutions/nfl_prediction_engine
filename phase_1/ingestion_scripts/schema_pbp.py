"""
Play-by-Play Schema for NFL Fantasy Prediction Engine

This module adds a comprehensive play-by-play table to store ALL available
fields from nflreadpy. This is the most granular level of NFL data available
and is essential for advanced analytics and modeling.

Based on: nflreadpy_options.txt field list

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

logger = get_ingestion_logger("schema_pbp")


# =============================================================================
# PLAY-BY-PLAY TABLE - Comprehensive NFL Play Data
# =============================================================================

PLAY_BY_PLAY_TABLE = """
CREATE TABLE IF NOT EXISTS play_by_play (
    -- Primary identifiers
    play_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    old_game_id TEXT,
    
    -- Time indexing (critical for anti-leakage)
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,
    
    -- Teams
    home_team TEXT,
    away_team TEXT,
    posteam TEXT,
    posteam_type TEXT,
    defteam TEXT,
    
    -- Field position
    side_of_field TEXT,
    yardline_100 INTEGER,
    
    -- Game timing
    game_date TEXT,
    quarter_seconds_remaining INTEGER,
    half_seconds_remaining INTEGER,
    game_seconds_remaining INTEGER,
    game_half TEXT,
    quarter_end INTEGER,
    qtr INTEGER,
    
    -- Drive info
    drive INTEGER,
    fixed_drive INTEGER,
    fixed_drive_result TEXT,
    drive_real_start_time TEXT,
    drive_play_count INTEGER,
    drive_time_of_possession TEXT,
    drive_first_downs INTEGER,
    drive_inside20 INTEGER,
    drive_ended_with_score INTEGER,
    drive_quarter_start INTEGER,
    drive_quarter_end INTEGER,
    drive_yards_penalized INTEGER,
    drive_start_transition TEXT,
    drive_end_transition TEXT,
    drive_game_clock_start TEXT,
    drive_game_clock_end TEXT,
    drive_start_yard_line TEXT,
    drive_end_yard_line TEXT,
    drive_play_id_started INTEGER,
    drive_play_id_ended INTEGER,
    
    -- Play situation
    down INTEGER,
    goal_to_go INTEGER,
    time TEXT,
    yrdln TEXT,
    ydstogo INTEGER,
    ydsnet INTEGER,
    
    -- Play description
    desc TEXT,
    play_type TEXT,
    play_type_nfl TEXT,
    special_teams_play INTEGER,
    st_play_type TEXT,
    
    -- Play result
    sp INTEGER,
    yards_gained INTEGER,
    success INTEGER,
    
    -- Formation
    shotgun INTEGER,
    no_huddle INTEGER,
    qb_dropback INTEGER,
    qb_kneel INTEGER,
    qb_spike INTEGER,
    qb_scramble INTEGER,
    
    -- Pass play details
    pass_length TEXT,
    pass_location TEXT,
    air_yards REAL,
    yards_after_catch REAL,
    
    -- Rush play details
    run_location TEXT,
    run_gap TEXT,
    
    -- Scoring plays
    field_goal_result TEXT,
    kick_distance INTEGER,
    extra_point_result TEXT,
    two_point_conv_result TEXT,
    
    -- Timeouts
    home_timeouts_remaining INTEGER,
    away_timeouts_remaining INTEGER,
    timeout INTEGER,
    timeout_team TEXT,
    posteam_timeouts_remaining INTEGER,
    defteam_timeouts_remaining INTEGER,
    
    -- Scoring
    td_team TEXT,
    td_player_name TEXT,
    td_player_id TEXT,
    total_home_score INTEGER,
    total_away_score INTEGER,
    posteam_score INTEGER,
    defteam_score INTEGER,
    score_differential INTEGER,
    posteam_score_post INTEGER,
    defteam_score_post INTEGER,
    score_differential_post INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    
    -- Scoring probabilities (CRITICAL for modeling)
    no_score_prob REAL,
    opp_fg_prob REAL,
    opp_safety_prob REAL,
    opp_td_prob REAL,
    fg_prob REAL,
    safety_prob REAL,
    td_prob REAL,
    extra_point_prob REAL,
    two_point_conversion_prob REAL,
    
    -- Expected Points (CRITICAL for modeling)
    ep REAL,
    epa REAL,
    total_home_epa REAL,
    total_away_epa REAL,
    total_home_rush_epa REAL,
    total_away_rush_epa REAL,
    total_home_pass_epa REAL,
    total_away_pass_epa REAL,
    air_epa REAL,
    yac_epa REAL,
    comp_air_epa REAL,
    comp_yac_epa REAL,
    total_home_comp_air_epa REAL,
    total_away_comp_air_epa REAL,
    total_home_comp_yac_epa REAL,
    total_away_comp_yac_epa REAL,
    total_home_raw_air_epa REAL,
    total_away_raw_air_epa REAL,
    total_home_raw_yac_epa REAL,
    total_away_raw_yac_epa REAL,
    qb_epa REAL,
    
    -- Win Probability (CRITICAL for modeling)
    wp REAL,
    def_wp REAL,
    home_wp REAL,
    away_wp REAL,
    wpa REAL,
    vegas_wpa REAL,
    vegas_home_wpa REAL,
    home_wp_post REAL,
    away_wp_post REAL,
    vegas_wp REAL,
    vegas_home_wp REAL,
    total_home_rush_wpa REAL,
    total_away_rush_wpa REAL,
    total_home_pass_wpa REAL,
    total_away_pass_wpa REAL,
    air_wpa REAL,
    yac_wpa REAL,
    comp_air_wpa REAL,
    comp_yac_wpa REAL,
    total_home_comp_air_wpa REAL,
    total_away_comp_air_wpa REAL,
    total_home_comp_yac_wpa REAL,
    total_away_comp_yac_wpa REAL,
    total_home_raw_air_wpa REAL,
    total_away_raw_air_wpa REAL,
    total_home_raw_yac_wpa REAL,
    total_away_raw_yac_wpa REAL,
    
    -- First downs
    first_down INTEGER,
    first_down_rush INTEGER,
    first_down_pass INTEGER,
    first_down_penalty INTEGER,
    
    -- Third/Fourth down
    third_down_converted INTEGER,
    third_down_failed INTEGER,
    fourth_down_converted INTEGER,
    fourth_down_failed INTEGER,
    
    -- Play outcomes
    punt_blocked INTEGER,
    incomplete_pass INTEGER,
    touchback INTEGER,
    interception INTEGER,
    
    -- Punt details
    punt_inside_twenty INTEGER,
    punt_in_endzone INTEGER,
    punt_out_of_bounds INTEGER,
    punt_downed INTEGER,
    punt_fair_catch INTEGER,
    
    -- Kickoff details
    kickoff_inside_twenty INTEGER,
    kickoff_in_endzone INTEGER,
    kickoff_out_of_bounds INTEGER,
    kickoff_downed INTEGER,
    kickoff_fair_catch INTEGER,
    
    -- Fumbles
    fumble INTEGER,
    fumble_forced INTEGER,
    fumble_not_forced INTEGER,
    fumble_out_of_bounds INTEGER,
    fumble_lost INTEGER,
    own_kickoff_recovery INTEGER,
    own_kickoff_recovery_td INTEGER,
    
    -- Tackles and sacks
    solo_tackle INTEGER,
    assist_tackle INTEGER,
    tackle_with_assist INTEGER,
    tackled_for_loss INTEGER,
    qb_hit INTEGER,
    sack INTEGER,
    
    -- Special teams
    safety INTEGER,
    penalty INTEGER,
    
    -- Play attempt types
    rush_attempt INTEGER,
    pass_attempt INTEGER,
    touchdown INTEGER,
    pass_touchdown INTEGER,
    rush_touchdown INTEGER,
    return_touchdown INTEGER,
    extra_point_attempt INTEGER,
    two_point_attempt INTEGER,
    field_goal_attempt INTEGER,
    kickoff_attempt INTEGER,
    punt_attempt INTEGER,
    complete_pass INTEGER,
    
    -- Lateral plays
    lateral_reception INTEGER,
    lateral_rush INTEGER,
    lateral_return INTEGER,
    lateral_recovery INTEGER,
    
    -- Player IDs and names - Passer
    passer_player_id TEXT,
    passer_player_name TEXT,
    passer TEXT,
    passer_id TEXT,
    passer_jersey_number TEXT,
    passing_yards INTEGER,
    
    -- Player IDs and names - Receiver
    receiver_player_id TEXT,
    receiver_player_name TEXT,
    receiver TEXT,
    receiver_id TEXT,
    receiver_jersey_number TEXT,
    receiving_yards INTEGER,
    
    -- Player IDs and names - Rusher
    rusher_player_id TEXT,
    rusher_player_name TEXT,
    rusher TEXT,
    rusher_id TEXT,
    rusher_jersey_number TEXT,
    rushing_yards INTEGER,
    
    -- Lateral player IDs
    lateral_receiver_player_id TEXT,
    lateral_receiver_player_name TEXT,
    lateral_receiving_yards INTEGER,
    lateral_rusher_player_id TEXT,
    lateral_rusher_player_name TEXT,
    lateral_rushing_yards INTEGER,
    lateral_sack_player_id TEXT,
    lateral_sack_player_name TEXT,
    
    -- Interception players
    interception_player_id TEXT,
    interception_player_name TEXT,
    lateral_interception_player_id TEXT,
    lateral_interception_player_name TEXT,
    
    -- Return players
    punt_returner_player_id TEXT,
    punt_returner_player_name TEXT,
    lateral_punt_returner_player_id TEXT,
    lateral_punt_returner_player_name TEXT,
    kickoff_returner_player_name TEXT,
    kickoff_returner_player_id TEXT,
    lateral_kickoff_returner_player_id TEXT,
    lateral_kickoff_returner_player_name TEXT,
    
    -- Kicker/Punter
    punter_player_id TEXT,
    punter_player_name TEXT,
    kicker_player_name TEXT,
    kicker_player_id TEXT,
    
    -- Recovery
    own_kickoff_recovery_player_id TEXT,
    own_kickoff_recovery_player_name TEXT,
    blocked_player_id TEXT,
    blocked_player_name TEXT,
    
    -- Tackle for loss players
    tackle_for_loss_1_player_id TEXT,
    tackle_for_loss_1_player_name TEXT,
    tackle_for_loss_2_player_id TEXT,
    tackle_for_loss_2_player_name TEXT,
    
    -- QB hit players
    qb_hit_1_player_id TEXT,
    qb_hit_1_player_name TEXT,
    qb_hit_2_player_id TEXT,
    qb_hit_2_player_name TEXT,
    
    -- Forced fumble players
    forced_fumble_player_1_team TEXT,
    forced_fumble_player_1_player_id TEXT,
    forced_fumble_player_1_player_name TEXT,
    forced_fumble_player_2_team TEXT,
    forced_fumble_player_2_player_id TEXT,
    forced_fumble_player_2_player_name TEXT,
    
    -- Solo tackle players
    solo_tackle_1_team TEXT,
    solo_tackle_2_team TEXT,
    solo_tackle_1_player_id TEXT,
    solo_tackle_2_player_id TEXT,
    solo_tackle_1_player_name TEXT,
    solo_tackle_2_player_name TEXT,
    
    -- Assist tackle players
    assist_tackle_1_player_id TEXT,
    assist_tackle_1_player_name TEXT,
    assist_tackle_1_team TEXT,
    assist_tackle_2_player_id TEXT,
    assist_tackle_2_player_name TEXT,
    assist_tackle_2_team TEXT,
    assist_tackle_3_player_id TEXT,
    assist_tackle_3_player_name TEXT,
    assist_tackle_3_team TEXT,
    assist_tackle_4_player_id TEXT,
    assist_tackle_4_player_name TEXT,
    assist_tackle_4_team TEXT,
    
    -- Tackle with assist players
    tackle_with_assist_1_player_id TEXT,
    tackle_with_assist_1_player_name TEXT,
    tackle_with_assist_1_team TEXT,
    tackle_with_assist_2_player_id TEXT,
    tackle_with_assist_2_player_name TEXT,
    tackle_with_assist_2_team TEXT,
    
    -- Pass defense players
    pass_defense_1_player_id TEXT,
    pass_defense_1_player_name TEXT,
    pass_defense_2_player_id TEXT,
    pass_defense_2_player_name TEXT,
    
    -- Fumble players
    fumbled_1_team TEXT,
    fumbled_1_player_id TEXT,
    fumbled_1_player_name TEXT,
    fumbled_2_player_id TEXT,
    fumbled_2_player_name TEXT,
    fumbled_2_team TEXT,
    
    -- Fumble recovery players
    fumble_recovery_1_team TEXT,
    fumble_recovery_1_yards INTEGER,
    fumble_recovery_1_player_id TEXT,
    fumble_recovery_1_player_name TEXT,
    fumble_recovery_2_team TEXT,
    fumble_recovery_2_yards INTEGER,
    fumble_recovery_2_player_id TEXT,
    fumble_recovery_2_player_name TEXT,
    
    -- Sack players
    sack_player_id TEXT,
    sack_player_name TEXT,
    half_sack_1_player_id TEXT,
    half_sack_1_player_name TEXT,
    half_sack_2_player_id TEXT,
    half_sack_2_player_name TEXT,
    
    -- Return info
    return_team TEXT,
    return_yards INTEGER,
    
    -- Penalty info
    penalty_team TEXT,
    penalty_player_id TEXT,
    penalty_player_name TEXT,
    penalty_yards INTEGER,
    penalty_type TEXT,
    
    -- Replay
    replay_or_challenge TEXT,
    replay_or_challenge_result TEXT,
    
    -- Defensive two-point
    defensive_two_point_attempt INTEGER,
    defensive_two_point_conv INTEGER,
    defensive_extra_point_attempt INTEGER,
    defensive_extra_point_conv INTEGER,
    
    -- Safety player
    safety_player_name TEXT,
    safety_player_id TEXT,
    
    -- Completion metrics (CRITICAL for modeling)
    cp REAL,
    cpoe REAL,
    
    -- Series info
    series INTEGER,
    series_success INTEGER,
    series_result TEXT,
    
    -- Play metadata
    order_sequence INTEGER,
    start_time TEXT,
    time_of_day TEXT,
    play_clock TEXT,
    play_deleted INTEGER,
    aborted_play INTEGER,
    end_clock_time TEXT,
    end_yard_line TEXT,
    
    -- Game location info
    stadium TEXT,
    game_stadium TEXT,
    stadium_id TEXT,
    location TEXT,
    roof TEXT,
    surface TEXT,
    weather TEXT,
    
    -- Game context
    result INTEGER,
    total INTEGER,
    spread_line REAL,
    total_line REAL,
    div_game INTEGER,
    
    -- Temperature/wind
    temp REAL,
    wind REAL,
    
    -- Coaches
    home_coach TEXT,
    away_coach TEXT,
    
    -- Generic player references
    pass INTEGER,
    rush INTEGER,
    special INTEGER,
    play INTEGER,
    name TEXT,
    jersey_number TEXT,
    id TEXT,
    
    -- Fantasy references
    fantasy_player_name TEXT,
    fantasy_player_id TEXT,
    fantasy TEXT,
    fantasy_id TEXT,
    
    -- Other
    out_of_bounds INTEGER,
    home_opening_kickoff INTEGER,
    nfl_api_id TEXT,
    
    -- Expected yards metrics (CRITICAL for modeling)
    xyac_epa REAL,
    xyac_mean_yardage REAL,
    xyac_median_yardage REAL,
    xyac_success REAL,
    xyac_fd REAL,
    xpass REAL,
    pass_oe REAL,
    
    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key
    PRIMARY KEY (play_id, game_id)
);
"""


# =============================================================================
# INDEXES FOR PLAY-BY-PLAY QUERIES
# =============================================================================

PBP_INDEXES = [
    # Primary lookup indexes
    "CREATE INDEX IF NOT EXISTS idx_pbp_game ON play_by_play(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_season_week ON play_by_play(season, week);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_season ON play_by_play(season);",
    
    # Team indexes
    "CREATE INDEX IF NOT EXISTS idx_pbp_home_team ON play_by_play(home_team);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_away_team ON play_by_play(away_team);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_posteam ON play_by_play(posteam);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_defteam ON play_by_play(defteam);",
    
    # Player indexes (most queried)
    "CREATE INDEX IF NOT EXISTS idx_pbp_passer ON play_by_play(passer_player_id);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_receiver ON play_by_play(receiver_player_id);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_rusher ON play_by_play(rusher_player_id);",
    
    # Play type indexes
    "CREATE INDEX IF NOT EXISTS idx_pbp_play_type ON play_by_play(play_type);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_pass_attempt ON play_by_play(pass_attempt);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_rush_attempt ON play_by_play(rush_attempt);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_touchdown ON play_by_play(touchdown);",
    
    # Situation indexes
    "CREATE INDEX IF NOT EXISTS idx_pbp_down ON play_by_play(down);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_qtr ON play_by_play(qtr);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_redzone ON play_by_play(yardline_100) WHERE yardline_100 <= 20;",
    
    # EPA/WP indexes for analytics
    "CREATE INDEX IF NOT EXISTS idx_pbp_epa ON play_by_play(epa);",
    "CREATE INDEX IF NOT EXISTS idx_pbp_wpa ON play_by_play(wpa);",
]


# =============================================================================
# SCHEMA CREATION FUNCTIONS
# =============================================================================

def create_pbp_table(db: DatabaseConnection, drop_existing: bool = False):
    """
    Create the play-by-play table.
    
    Args:
        db: DatabaseConnection instance
        drop_existing: If True, drop existing table first
    """
    logger.info("Creating play-by-play table...")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        if drop_existing:
            logger.warning("Dropping existing play_by_play table!")
            cursor.execute("DROP TABLE IF EXISTS play_by_play;")
        
        # Create table
        cursor.execute(PLAY_BY_PLAY_TABLE)
        logger.info("Created play_by_play table")
        
        # Create indexes
        logger.info("Creating play_by_play indexes...")
        for index_sql in PBP_INDEXES:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")
        
        logger.info(f"Created {len(PBP_INDEXES)} indexes")
    
    logger.info("Play-by-play schema creation complete!", event="pbp_schema_complete")


def verify_pbp_table(db: DatabaseConnection) -> dict:
    """Verify play-by-play table exists and has correct structure."""
    
    results = {
        "exists": db.table_exists("play_by_play"),
        "row_count": 0,
    }
    
    if results["exists"]:
        results["row_count"] = db.get_row_count("play_by_play")
        
        # Get column count
        schema = db.get_table_schema("play_by_play")
        results["column_count"] = len(schema)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create play-by-play table for comprehensive NFL data"
    )
    parser.add_argument("--drop", action="store_true", help="Drop existing table first")
    parser.add_argument("--verify-only", action="store_true", help="Only verify table")
    
    args = parser.parse_args()
    
    db = get_db()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Play-by-Play Schema")
    print(f"{'='*60}")
    print(f"Database: {db.db_path}")
    print(f"{'='*60}\n")
    
    if args.verify_only:
        results = verify_pbp_table(db)
        print(f"Table exists: {results['exists']}")
        print(f"Row count: {results['row_count']}")
        if results['exists']:
            print(f"Column count: {results['column_count']}")
    else:
        if args.drop:
            confirm = input("WARNING: Drop existing table? Type 'YES': ")
            if confirm != 'YES':
                print("Aborted.")
                return
        
        create_pbp_table(db, drop_existing=args.drop)
        
        results = verify_pbp_table(db)
        if results['exists']:
            print(f"\n✓ Play-by-play table created with {results['column_count']} columns!")
        else:
            print("\n✗ Table creation failed!")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
