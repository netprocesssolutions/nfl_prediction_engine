"""
NFL Data Py Integration for NFL Fantasy Prediction Engine

This module integrates nfl_data_py to access:
- Play-by-play data with EPA (Expected Points Added)
- Next Gen Stats (separation, cushion, CPOE, rush yards over expected)
- Weekly player stats aggregated
- Roster and position data

nfl_data_py provides access to nflfastR data in Python, which is
THE gold standard for NFL analytics data.

Installation: pip install nfl_data_py

Author: NFL Fantasy Prediction Engine Team
Phase: 1 Extension - Advanced Stats Integration
Version: 1.0
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

# Check if nflreadpy or nfl_data_py is installed (prefer nflreadpy)
try:
    import nflreadpy as nfl
    NFL_DATA_PY_AVAILABLE = True
    NFL_DATA_SOURCE = "nflreadpy"
except ImportError:
    try:
        import nfl_data_py as nfl
        NFL_DATA_PY_AVAILABLE = True
        NFL_DATA_SOURCE = "nfl_data_py"
    except ImportError:
        NFL_DATA_PY_AVAILABLE = False
        NFL_DATA_SOURCE = None
        print("WARNING: Neither nflreadpy nor nfl_data_py installed.")
        print("Run: pip install nflreadpy  or  pip install nfl_data_py")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not installed. Run: pip install pandas")

from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import CURRENT_SEASON

# Initialize logger
logger = get_ingestion_logger("ingest_nfl_data_py")


# =============================================================================
# SCHEMA FOR NGS DATA
# =============================================================================

NGS_PASSING_TABLE = """
CREATE TABLE IF NOT EXISTS ngs_passing (
    player_id TEXT NOT NULL,
    player_display_name TEXT,
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Time metrics
    avg_time_to_throw REAL,
    avg_time_in_pocket REAL,
    
    -- Air yards
    avg_completed_air_yards REAL,
    avg_intended_air_yards REAL,
    avg_air_yards_differential REAL,
    
    -- Efficiency
    aggressiveness REAL,
    max_completed_air_distance REAL,
    
    -- Expected metrics
    completion_percentage REAL,
    expected_completion_percentage REAL,
    completion_percentage_above_expectation REAL,
    
    -- Pressure
    passer_rating REAL,
    
    -- Metadata
    attempts INTEGER,
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
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Efficiency
    efficiency REAL,
    percent_attempts_gte_eight_defenders REAL,
    
    -- Expected metrics
    avg_rush_yards REAL,
    expected_rush_yards REAL,
    rush_yards_over_expected REAL,
    rush_yards_over_expected_per_att REAL,
    rush_pct_over_expected REAL,
    
    -- Speed
    avg_time_to_los REAL,
    
    -- Metadata
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
    team_abbr TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Coverage metrics
    avg_cushion REAL,
    avg_separation REAL,
    
    -- Air yards
    avg_intended_air_yards REAL,
    percent_share_of_intended_air_yards REAL,
    
    -- Catch metrics
    catch_percentage REAL,
    
    -- YAC
    avg_yac REAL,
    avg_expected_yac REAL,
    avg_yac_above_expectation REAL,
    
    -- Metadata
    targets INTEGER,
    receptions INTEGER,
    receiving_yards INTEGER,
    receiving_touchdowns INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""

WEEKLY_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS nflverse_weekly_stats (
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Usage
    completions INTEGER,
    attempts INTEGER,
    passing_yards INTEGER,
    passing_tds INTEGER,
    interceptions INTEGER,
    sacks INTEGER,
    sack_yards INTEGER,
    
    carries INTEGER,
    rushing_yards INTEGER,
    rushing_tds INTEGER,
    rushing_fumbles INTEGER,
    rushing_first_downs INTEGER,
    
    receptions INTEGER,
    targets INTEGER,
    receiving_yards INTEGER,
    receiving_tds INTEGER,
    receiving_fumbles INTEGER,
    receiving_air_yards INTEGER,
    receiving_yards_after_catch INTEGER,
    receiving_first_downs INTEGER,
    
    -- EPA
    passing_epa REAL,
    rushing_epa REAL,
    receiving_epa REAL,
    
    -- Advanced
    target_share REAL,
    air_yards_share REAL,
    wopr REAL,  -- Weighted Opportunity Rating
    racr REAL,  -- Receiver Air Conversion Ratio
    
    -- Fantasy
    fantasy_points REAL,
    fantasy_points_ppr REAL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""

EPA_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS player_epa_summary (
    player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- EPA metrics (aggregated from play-by-play)
    total_epa REAL,
    epa_per_play REAL,
    success_rate REAL,  -- % of plays with positive EPA
    
    -- Passing EPA
    passing_epa REAL,
    cpoe REAL,  -- Completion % Over Expected
    
    -- Rushing EPA
    rushing_epa REAL,
    
    -- Receiving EPA
    receiving_epa REAL,
    yac_epa REAL,
    air_epa REAL,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (player_id, season, week)
);
"""


class NFLDataPyIngestion:
    """
    Ingests NFL data from nfl_data_py package.
    
    Provides access to:
    - Next Gen Stats (passing, rushing, receiving)
    - Weekly player stats with EPA
    - Play-by-play data for custom aggregations
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the ingestion class.
        
        Args:
            db: Optional database connection
        """
        if not NFL_DATA_PY_AVAILABLE:
            raise ImportError(
                "nfl_data_py is not installed. "
                "Install with: pip install nfl_data_py"
            )
        
        self.db = db or get_db()
        
        # Stats tracking
        self.stats = {
            'rows_inserted': 0,
            'rows_updated': 0,
            'errors': [],
        }
    
    def create_ngs_tables(self):
        """Create Next Gen Stats tables if they don't exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(NGS_PASSING_TABLE)
            cursor.execute(NGS_RUSHING_TABLE)
            cursor.execute(NGS_RECEIVING_TABLE)
            cursor.execute(WEEKLY_STATS_TABLE)
            cursor.execute(EPA_SUMMARY_TABLE)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngs_pass_season ON ngs_passing(season, week);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngs_rush_season ON ngs_rushing(season, week);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngs_rec_season ON ngs_receiving(season, week);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_weekly_season ON nflverse_weekly_stats(season, week);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_epa_season ON player_epa_summary(season, week);")
        
        logger.info("Created NGS tables", event="ngs_tables_created")
    
    def ingest_next_gen_passing(
        self,
        seasons: List[int],
    ) -> Dict:
        """
        Ingest Next Gen passing stats.
        
        Args:
            seasons: List of seasons to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting NGS passing for seasons: {seasons}")
        
        try:
            df = nfl.import_ngs_data(stat_type='passing', years=seasons)
            
            if df.empty:
                logger.warning("No NGS passing data returned")
                return {'success': False, 'rows': 0}
            
            rows_inserted = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO ngs_passing (
                                player_id, player_display_name, team_abbr,
                                season, week,
                                avg_time_to_throw, avg_time_in_pocket,
                                avg_completed_air_yards, avg_intended_air_yards,
                                avg_air_yards_differential, aggressiveness,
                                max_completed_air_distance,
                                completion_percentage, expected_completion_percentage,
                                completion_percentage_above_expectation,
                                passer_rating, attempts, pass_yards,
                                pass_touchdowns, interceptions
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('player_gsis_id') or row.get('player_id'),
                            row.get('player_display_name'),
                            row.get('team_abbr'),
                            row.get('season'),
                            row.get('week'),
                            row.get('avg_time_to_throw'),
                            row.get('avg_time_to_pocket') or row.get('avg_time_in_pocket'),
                            row.get('avg_completed_air_yards'),
                            row.get('avg_intended_air_yards'),
                            row.get('avg_air_yards_differential'),
                            row.get('aggressiveness'),
                            row.get('max_completed_air_distance'),
                            row.get('completion_percentage'),
                            row.get('expected_completion_percentage'),
                            row.get('completion_percentage_above_expectation'),
                            row.get('passer_rating'),
                            row.get('attempts'),
                            row.get('pass_yards'),
                            row.get('pass_touchdowns'),
                            row.get('interceptions'),
                        ))
                        rows_inserted += 1
                    except Exception as e:
                        logger.debug(f"Error inserting NGS passing row: {e}")
            
            self.stats['rows_inserted'] += rows_inserted
            logger.info(f"Inserted {rows_inserted} NGS passing rows", event="ngs_passing_complete")
            
            return {'success': True, 'rows': rows_inserted}
            
        except Exception as e:
            logger.error(f"Error ingesting NGS passing: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_next_gen_rushing(
        self,
        seasons: List[int],
    ) -> Dict:
        """
        Ingest Next Gen rushing stats.
        
        Args:
            seasons: List of seasons to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting NGS rushing for seasons: {seasons}")
        
        try:
            df = nfl.import_ngs_data(stat_type='rushing', years=seasons)
            
            if df.empty:
                logger.warning("No NGS rushing data returned")
                return {'success': False, 'rows': 0}
            
            rows_inserted = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO ngs_rushing (
                                player_id, player_display_name, team_abbr,
                                season, week,
                                efficiency, percent_attempts_gte_eight_defenders,
                                avg_rush_yards, expected_rush_yards,
                                rush_yards_over_expected, rush_yards_over_expected_per_att,
                                rush_pct_over_expected, avg_time_to_los,
                                rush_attempts, rush_yards, rush_touchdowns
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('player_gsis_id') or row.get('player_id'),
                            row.get('player_display_name'),
                            row.get('team_abbr'),
                            row.get('season'),
                            row.get('week'),
                            row.get('efficiency'),
                            row.get('percent_attempts_gte_eight_defenders'),
                            row.get('avg_rush_yards'),
                            row.get('expected_rush_yards'),
                            row.get('rush_yards_over_expected'),
                            row.get('rush_yards_over_expected_per_att'),
                            row.get('rush_pct_over_expected'),
                            row.get('avg_time_to_los'),
                            row.get('rush_attempts'),
                            row.get('rush_yards'),
                            row.get('rush_touchdowns'),
                        ))
                        rows_inserted += 1
                    except Exception as e:
                        logger.debug(f"Error inserting NGS rushing row: {e}")
            
            self.stats['rows_inserted'] += rows_inserted
            logger.info(f"Inserted {rows_inserted} NGS rushing rows", event="ngs_rushing_complete")
            
            return {'success': True, 'rows': rows_inserted}
            
        except Exception as e:
            logger.error(f"Error ingesting NGS rushing: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_next_gen_receiving(
        self,
        seasons: List[int],
    ) -> Dict:
        """
        Ingest Next Gen receiving stats.
        
        Args:
            seasons: List of seasons to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting NGS receiving for seasons: {seasons}")
        
        try:
            df = nfl.import_ngs_data(stat_type='receiving', years=seasons)
            
            if df.empty:
                logger.warning("No NGS receiving data returned")
                return {'success': False, 'rows': 0}
            
            rows_inserted = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO ngs_receiving (
                                player_id, player_display_name, team_abbr,
                                season, week,
                                avg_cushion, avg_separation,
                                avg_intended_air_yards, percent_share_of_intended_air_yards,
                                catch_percentage,
                                avg_yac, avg_expected_yac, avg_yac_above_expectation,
                                targets, receptions, receiving_yards, receiving_touchdowns
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('player_gsis_id') or row.get('player_id'),
                            row.get('player_display_name'),
                            row.get('team_abbr'),
                            row.get('season'),
                            row.get('week'),
                            row.get('avg_cushion'),
                            row.get('avg_separation'),
                            row.get('avg_intended_air_yards'),
                            row.get('percent_share_of_intended_air_yards'),
                            row.get('catch_percentage'),
                            row.get('avg_yac'),
                            row.get('avg_expected_yac'),
                            row.get('avg_yac_above_expectation'),
                            row.get('targets'),
                            row.get('receptions'),
                            row.get('yards') or row.get('receiving_yards'),
                            row.get('touchdowns') or row.get('receiving_touchdowns'),
                        ))
                        rows_inserted += 1
                    except Exception as e:
                        logger.debug(f"Error inserting NGS receiving row: {e}")
            
            self.stats['rows_inserted'] += rows_inserted
            logger.info(f"Inserted {rows_inserted} NGS receiving rows", event="ngs_receiving_complete")
            
            return {'success': True, 'rows': rows_inserted}
            
        except Exception as e:
            logger.error(f"Error ingesting NGS receiving: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_weekly_stats(
        self,
        seasons: List[int],
    ) -> Dict:
        """
        Ingest weekly player stats with EPA.
        
        Args:
            seasons: List of seasons to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting weekly stats for seasons: {seasons}")
        
        try:
            df = nfl.import_weekly_data(seasons)
            
            if df.empty:
                logger.warning("No weekly stats data returned")
                return {'success': False, 'rows': 0}
            
            rows_inserted = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO nflverse_weekly_stats (
                                player_id, player_name, position, team,
                                season, week,
                                completions, attempts, passing_yards, passing_tds,
                                interceptions, sacks, sack_yards,
                                carries, rushing_yards, rushing_tds,
                                rushing_fumbles, rushing_first_downs,
                                receptions, targets, receiving_yards, receiving_tds,
                                receiving_fumbles, receiving_air_yards,
                                receiving_yards_after_catch, receiving_first_downs,
                                passing_epa, rushing_epa, receiving_epa,
                                target_share, air_yards_share, wopr, racr,
                                fantasy_points, fantasy_points_ppr
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('player_id'),
                            row.get('player_name') or row.get('player_display_name'),
                            row.get('position'),
                            row.get('recent_team') or row.get('team'),
                            row.get('season'),
                            row.get('week'),
                            row.get('completions'),
                            row.get('attempts'),
                            row.get('passing_yards'),
                            row.get('passing_tds'),
                            row.get('interceptions'),
                            row.get('sacks'),
                            row.get('sack_yards'),
                            row.get('carries'),
                            row.get('rushing_yards'),
                            row.get('rushing_tds'),
                            row.get('rushing_fumbles'),
                            row.get('rushing_first_downs'),
                            row.get('receptions'),
                            row.get('targets'),
                            row.get('receiving_yards'),
                            row.get('receiving_tds'),
                            row.get('receiving_fumbles'),
                            row.get('receiving_air_yards'),
                            row.get('receiving_yards_after_catch'),
                            row.get('receiving_first_downs'),
                            row.get('passing_epa'),
                            row.get('rushing_epa'),
                            row.get('receiving_epa'),
                            row.get('target_share'),
                            row.get('air_yards_share'),
                            row.get('wopr'),
                            row.get('racr'),
                            row.get('fantasy_points'),
                            row.get('fantasy_points_ppr'),
                        ))
                        rows_inserted += 1
                    except Exception as e:
                        logger.debug(f"Error inserting weekly stats row: {e}")
            
            self.stats['rows_inserted'] += rows_inserted
            logger.info(f"Inserted {rows_inserted} weekly stats rows", event="weekly_stats_complete")
            
            return {'success': True, 'rows': rows_inserted}
            
        except Exception as e:
            logger.error(f"Error ingesting weekly stats: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_all(
        self,
        seasons: Optional[List[int]] = None,
    ) -> Dict:
        """
        Ingest all available data for specified seasons.
        
        Args:
            seasons: List of seasons (defaults to current and previous)
            
        Returns:
            Dictionary with combined results
        """
        if seasons is None:
            seasons = [CURRENT_SEASON - 1, CURRENT_SEASON]
        
        logger.info(f"Starting full NFLverse ingestion for seasons: {seasons}")
        
        # Ensure tables exist
        self.create_ngs_tables()
        
        results = {
            'ngs_passing': self.ingest_next_gen_passing(seasons),
            'ngs_rushing': self.ingest_next_gen_rushing(seasons),
            'ngs_receiving': self.ingest_next_gen_receiving(seasons),
            'weekly_stats': self.ingest_weekly_stats(seasons),
            'total_rows': self.stats['rows_inserted'],
            'errors': self.stats['errors'],
        }
        
        logger.info(
            f"NFLverse ingestion complete: {results['total_rows']} total rows",
            event="nflverse_complete"
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get current ingestion statistics."""
        return self.stats.copy()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest NFL data from nfl_data_py package"
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=[CURRENT_SEASON - 1, CURRENT_SEASON],
        help='Seasons to ingest (default: current and previous)'
    )
    parser.add_argument(
        '--ngs-only',
        action='store_true',
        help='Only ingest Next Gen Stats'
    )
    parser.add_argument(
        '--weekly-only',
        action='store_true',
        help='Only ingest weekly stats'
    )
    parser.add_argument(
        '--create-tables',
        action='store_true',
        help='Only create tables without ingesting data'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - NFLverse Data Ingestion")
    print(f"{'='*60}")
    
    if not NFL_DATA_PY_AVAILABLE:
        print("\nERROR: nfl_data_py not installed!")
        print("Install with: pip install nfl_data_py")
        return 1
    
    db = get_db()
    ingestion = NFLDataPyIngestion(db)
    
    if args.create_tables:
        ingestion.create_ngs_tables()
        print("\nÃ¢Å“â€œ Tables created successfully!")
        return 0
    
    print(f"\nIngesting data for seasons: {args.seasons}")
    
    if args.ngs_only:
        ingestion.create_ngs_tables()
        result1 = ingestion.ingest_next_gen_passing(args.seasons)
        result2 = ingestion.ingest_next_gen_rushing(args.seasons)
        result3 = ingestion.ingest_next_gen_receiving(args.seasons)
        print(f"\nNGS Passing: {result1.get('rows', 0)} rows")
        print(f"NGS Rushing: {result2.get('rows', 0)} rows")
        print(f"NGS Receiving: {result3.get('rows', 0)} rows")
        
    elif args.weekly_only:
        ingestion.create_ngs_tables()
        result = ingestion.ingest_weekly_stats(args.seasons)
        print(f"\nWeekly Stats: {result.get('rows', 0)} rows")
        
    else:
        results = ingestion.ingest_all(args.seasons)
        print(f"\nResults:")
        print(f"  NGS Passing: {results['ngs_passing'].get('rows', 0)} rows")
        print(f"  NGS Rushing: {results['ngs_rushing'].get('rows', 0)} rows")
        print(f"  NGS Receiving: {results['ngs_receiving'].get('rows', 0)} rows")
        print(f"  Weekly Stats: {results['weekly_stats'].get('rows', 0)} rows")
        print(f"  Total: {results['total_rows']} rows")
        
        if results['errors']:
            print(f"\nErrors: {results['errors']}")
    
    print(f"\nLog file: {logger.log_file}")
    return 0


if __name__ == "__main__":
    exit(main())
