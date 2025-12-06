"""
================================================================================
COMPLETE FIXED NFL Data Ingestion via nflreadpy
================================================================================

Version: 3.0 - Complete Column Mapping Fix
Author: NFL Fantasy Prediction Engine Team

This module ingests ALL available data from nflreadpy with CORRECT column mappings:
- Weekly player stats with EPA → nflverse_weekly_stats
- Next Gen Stats (passing, rushing, receiving) → ngs_passing/rushing/receiving
- Snap counts → snap_counts
- Rosters → rosters
- Injuries → injuries
- Schedules → schedules
- Combine data → combine_data

CRITICAL FIXES IN THIS VERSION:
1. Polars → Pandas conversion for all nflreadpy calls
2. Correct column mapping for each table
3. player_id field mapping (gsis_id/pfr_player_id → player_id)
4. recent_team → team rename
5. week handling for rosters (set week=0 for season rosters)
6. Special teams snap column rename

Installation: pip install nflreadpy pandas polars
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies - nflreadpy is the PRIMARY source
try:
    import nflreadpy as nfl
    import pandas as pd
    NFL_DATA_AVAILABLE = True
    NFL_DATA_SOURCE = "nflreadpy"
except ImportError:
    try:
        import nfl_data_py as nfl
        import pandas as pd
        NFL_DATA_AVAILABLE = True
        NFL_DATA_SOURCE = "nfl_data_py"
        print("WARNING: Using nfl_data_py (deprecated). Install nflreadpy instead.")
    except ImportError as e:
        NFL_DATA_AVAILABLE = False
        NFL_DATA_SOURCE = None
        print(f"ERROR: Missing dependency - {e}")
        print("Install with: pip install nflreadpy pandas polars")

from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import CURRENT_SEASON

logger = get_ingestion_logger("ingest_nflverse")


# =============================================================================
# POLARS TO PANDAS CONVERSION (CRITICAL FOR NFLREADPY)
# =============================================================================

def polars_to_pandas(df):
    """
    Convert a Polars DataFrame to Pandas DataFrame.
    
    nflreadpy returns Polars DataFrames, but our code expects Pandas.
    This handles the conversion safely.
    
    Args:
        df: Polars DataFrame or Pandas DataFrame
        
    Returns:
        Pandas DataFrame or None if empty/error
    """
    if df is None:
        return None
    
    # Already a Pandas DataFrame?
    if hasattr(df, 'iterrows'):
        if len(df) == 0:
            return None
        return df
    
    # Check if empty Polars DataFrame
    try:
        if hasattr(df, 'height') and df.height == 0:
            return None
        if hasattr(df, '__len__') and len(df) == 0:
            return None
    except:
        pass
    
    # Convert Polars to Pandas
    try:
        pdf = df.to_pandas()
        if len(pdf) == 0:
            return None
        return pdf
    except Exception as e:
        logger.error(f"Error converting Polars to Pandas: {e}")
        return None


# =============================================================================
# MAIN INGESTION CLASS
# =============================================================================

class NFLVerseIngestion:
    """
    Complete ingestion from nflreadpy (nflverse data).
    
    This is THE most important data source for prediction accuracy.
    
    nflreadpy function reference:
    - load_player_stats() → nflverse_weekly_stats
    - load_nextgen_stats() → ngs_passing, ngs_rushing, ngs_receiving
    - load_snap_counts() → snap_counts
    - load_rosters() → rosters
    - load_rosters_weekly() → rosters (with week)
    - load_injuries() → injuries
    - load_schedules() → schedules
    - load_combine() → combine_data
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        if not NFL_DATA_AVAILABLE:
            raise ImportError("nflreadpy and pandas required. Install with: pip install nflreadpy pandas polars")
        
        self.db = db or get_db()
        self.stats = {
            'total_rows': 0,
            'by_table': {},
            'errors': [],
        }
        logger.info(f"NFLVerseIngestion initialized with data source: {NFL_DATA_SOURCE}")
    
    def _safe_value(self, val):
        """Convert pandas NA/NaN to None for SQLite."""
        if pd.isna(val):
            return None
        return val
    
    def _insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        columns: List[str],
        conflict_action: str = "REPLACE"
    ) -> int:
        """
        Insert a DataFrame into a table.
        
        Only inserts columns that exist in BOTH the DataFrame and the column list.
        """
        if df is None or len(df) == 0:
            logger.warning(f"Empty DataFrame for {table_name}")
            return 0
        
        # Filter to columns that exist in the DataFrame
        available_cols = [c for c in columns if c in df.columns]
        
        if not available_cols:
            logger.warning(f"No matching columns for {table_name}")
            logger.debug(f"Expected: {columns[:10]}...")
            logger.debug(f"DataFrame has: {list(df.columns)[:10]}...")
            return 0
        
        placeholders = ", ".join(["?" for _ in available_cols])
        col_names = ", ".join(available_cols)
        
        sql = f"INSERT OR {conflict_action} INTO {table_name} ({col_names}) VALUES ({placeholders})"
        
        rows_inserted = 0
        errors = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, row in df.iterrows():
                try:
                    values = tuple(self._safe_value(row.get(col)) for col in available_cols)
                    cursor.execute(sql, values)
                    rows_inserted += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.debug(f"Row insert error in {table_name}: {e}")
        
        self.stats['total_rows'] += rows_inserted
        self.stats['by_table'][table_name] = self.stats['by_table'].get(table_name, 0) + rows_inserted
        
        if errors > 0:
            logger.warning(f"{table_name}: {errors} errors during insert")
        
        return rows_inserted
    
    # =========================================================================
    # WEEKLY STATS - Most Important Table
    # =========================================================================
    
    def ingest_weekly_stats(self, seasons: List[int]) -> Dict:
        """
        Ingest weekly player stats with EPA into nflverse_weekly_stats.
        
        This is THE MOST IMPORTANT data source for predictions.
        
        nflreadpy.load_player_stats() returns:
        - player_id, player_name, player_display_name
        - position, position_group, recent_team (→ rename to 'team')
        - season, week
        - completions, attempts, passing_yards, passing_tds, interceptions, sacks
        - carries, rushing_yards, rushing_tds
        - receptions, targets, receiving_yards, receiving_tds
        - target_share, air_yards_share, wopr, racr
        - passing_epa, rushing_epa, receiving_epa
        - fantasy_points, fantasy_points_ppr
        """
        logger.info(f"Ingesting weekly stats for seasons {seasons}")
        
        try:
            # Load data - nflreadpy returns Polars DataFrame
            df_raw = nfl.load_player_stats(seasons)
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No weekly stats data returned")
                return {'success': False, 'rows': 0, 'error': 'No data returned'}
            
            logger.info(f"Loaded {len(df)} weekly stats rows")
            logger.debug(f"Columns available: {list(df.columns)[:20]}...")
            
            # CRITICAL FIX: Rename recent_team → team
            if 'recent_team' in df.columns:
                df['team'] = df['recent_team']
                logger.debug("Renamed recent_team → team")
            
            # Define columns to insert (matching schema)
            columns = [
                'player_id', 'player_name', 'player_display_name',
                'position', 'position_group', 'team',
                'season', 'week',
                # Passing
                'completions', 'attempts', 'passing_yards', 'passing_tds',
                'interceptions', 'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost',
                'passing_air_yards', 'passing_yards_after_catch', 'passing_first_downs',
                'passing_2pt_conversions',
                # Rushing
                'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
                'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_2pt_conversions',
                # Receiving
                'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
                'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_2pt_conversions',
                # Usage metrics
                'target_share', 'air_yards_share', 'wopr', 'racr',
                # EPA (CRITICAL for predictions)
                'passing_epa', 'rushing_epa', 'receiving_epa',
                # Fantasy points
                'fantasy_points', 'fantasy_points_ppr',
                # Special teams
                'special_teams_tds'
            ]
            
            rows = self._insert_dataframe(df, 'nflverse_weekly_stats', columns)
            
            logger.info(f"Inserted {rows} weekly stats rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"Weekly stats error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # NEXT GEN STATS
    # =========================================================================
    
    def ingest_ngs_passing(self, seasons: List[int]) -> Dict:
        """
        Ingest Next Gen passing stats (QBs).
        
        nflreadpy.load_nextgen_stats(stat_type='passing') returns:
        - player_gsis_id (→ map to player_id)
        - player_display_name
        - team_abbr, season, week
        - avg_time_to_throw, avg_completed_air_yards, avg_intended_air_yards
        - aggressiveness, completion_percentage, expected_completion_percentage
        - passer_rating, attempts, completions, pass_yards, pass_touchdowns
        """
        logger.info(f"Ingesting NGS passing for seasons {seasons}")
        
        try:
            df_raw = nfl.load_nextgen_stats(seasons, stat_type='passing')
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No NGS passing data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} NGS passing rows")
            
            # CRITICAL FIX: Map player_gsis_id → player_id
            if 'player_gsis_id' in df.columns:
                df['player_id'] = df['player_gsis_id']
                logger.debug("Mapped player_gsis_id → player_id")
            
            columns = [
                'player_id', 'player_display_name', 'player_gsis_id',
                'team_abbr', 'season', 'week',
                'avg_time_to_throw', 'avg_time_in_pocket',
                'avg_completed_air_yards', 'avg_intended_air_yards',
                'avg_air_yards_differential', 'avg_air_yards_to_sticks',
                'aggressiveness', 'max_completed_air_distance',
                'completion_percentage', 'expected_completion_percentage',
                'completion_percentage_above_expectation',
                'passer_rating', 'attempts', 'completions',
                'pass_yards', 'pass_touchdowns', 'interceptions'
            ]
            
            rows = self._insert_dataframe(df, 'ngs_passing', columns)
            
            logger.info(f"Inserted {rows} NGS passing rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"NGS passing error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_ngs_rushing(self, seasons: List[int]) -> Dict:
        """Ingest Next Gen rushing stats."""
        logger.info(f"Ingesting NGS rushing for seasons {seasons}")
        
        try:
            df_raw = nfl.load_nextgen_stats(seasons, stat_type='rushing')
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No NGS rushing data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} NGS rushing rows")
            
            # Map player_gsis_id → player_id
            if 'player_gsis_id' in df.columns:
                df['player_id'] = df['player_gsis_id']
            
            columns = [
                'player_id', 'player_display_name', 'player_gsis_id',
                'team_abbr', 'season', 'week',
                'efficiency', 'percent_attempts_gte_eight_defenders',
                'avg_rush_yards', 'expected_rush_yards',
                'rush_yards_over_expected', 'rush_yards_over_expected_per_att',
                'rush_pct_over_expected', 'avg_time_to_los',
                'rush_attempts', 'rush_yards', 'rush_touchdowns'
            ]
            
            rows = self._insert_dataframe(df, 'ngs_rushing', columns)
            
            logger.info(f"Inserted {rows} NGS rushing rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"NGS rushing error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def ingest_ngs_receiving(self, seasons: List[int]) -> Dict:
        """Ingest Next Gen receiving stats."""
        logger.info(f"Ingesting NGS receiving for seasons {seasons}")
        
        try:
            df_raw = nfl.load_nextgen_stats(seasons, stat_type='receiving')
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No NGS receiving data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} NGS receiving rows")
            
            # Map player_gsis_id → player_id
            if 'player_gsis_id' in df.columns:
                df['player_id'] = df['player_gsis_id']
            
            # Map receiving_touchdowns column if needed
            if 'rec_touchdowns' in df.columns and 'receiving_touchdowns' not in df.columns:
                df['receiving_touchdowns'] = df['rec_touchdowns']
            
            columns = [
                'player_id', 'player_display_name', 'player_gsis_id',
                'team_abbr', 'season', 'week',
                'avg_cushion', 'avg_separation',
                'avg_intended_air_yards', 'percent_share_of_intended_air_yards',
                'catch_percentage',
                'avg_yac', 'avg_expected_yac', 'avg_yac_above_expectation',
                'receptions', 'targets', 'receiving_yards', 'receiving_touchdowns'
            ]
            
            rows = self._insert_dataframe(df, 'ngs_receiving', columns)
            
            logger.info(f"Inserted {rows} NGS receiving rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"NGS receiving error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # SNAP COUNTS
    # =========================================================================
    
    def ingest_snap_counts(self, seasons: List[int]) -> Dict:
        """
        Ingest snap count data.
        
        nflreadpy.load_snap_counts() returns:
        - game_id, pfr_game_id, season, game_type, week
        - player, pfr_player_id (→ use as player_id), position, team
        - offense_snaps, offense_pct
        - defense_snaps, defense_pct
        - st_snaps (→ special_teams_snaps), st_pct (→ special_teams_pct)
        
        Schema expects PRIMARY KEY (player_id, season, week)
        """
        logger.info(f"Ingesting snap counts for seasons {seasons}")
        
        try:
            df_raw = nfl.load_snap_counts(seasons)
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No snap counts data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} snap counts rows")
            logger.debug(f"Snap counts columns: {list(df.columns)}")
            
            # CRITICAL FIX: Use pfr_player_id as player_id
            if 'pfr_player_id' in df.columns:
                df['player_id'] = df['pfr_player_id']
                logger.debug("Mapped pfr_player_id → player_id")
            
            # CRITICAL FIX: Rename st_snaps → special_teams_snaps
            if 'st_snaps' in df.columns:
                df['special_teams_snaps'] = df['st_snaps']
            if 'st_pct' in df.columns:
                df['special_teams_pct'] = df['st_pct']
            
            # Get opponent from game_id if not present
            # game_id format: YYYY_WW_AWAY_HOME
            
            columns = [
                'player_id', 'game_id', 'season', 'week',
                'team', 'opponent',
                'offense_snaps', 'offense_pct',
                'defense_snaps', 'defense_pct',
                'special_teams_snaps', 'special_teams_pct'
            ]
            
            rows = self._insert_dataframe(df, 'snap_counts', columns)
            
            logger.info(f"Inserted {rows} snap count rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"Snap counts error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # ROSTERS
    # =========================================================================
    
    def ingest_rosters(self, seasons: List[int]) -> Dict:
        """
        Ingest roster data.
        
        nflreadpy.load_rosters() returns SEASON-level rosters:
        - season, team, position, depth_chart_position
        - jersey_number, status, full_name, gsis_id
        - years_exp, rookie_year
        
        Schema expects (player_id, season, week) PK - we set week=0 for season data.
        """
        logger.info(f"Ingesting rosters for seasons {seasons}")
        
        try:
            df_raw = nfl.load_rosters(seasons)
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No rosters data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} roster rows")
            
            # CRITICAL FIX: Map gsis_id → player_id
            if 'gsis_id' in df.columns:
                df['player_id'] = df['gsis_id']
                logger.debug("Mapped gsis_id → player_id")
            
            # CRITICAL FIX: Map full_name → player_name
            if 'full_name' in df.columns:
                df['player_name'] = df['full_name']
            
            # CRITICAL FIX: Add week=0 for season-level rosters
            if 'week' not in df.columns:
                df['week'] = 0
                logger.debug("Added week=0 for season-level rosters")
            
            columns = [
                'player_id', 'season', 'week',
                'player_name', 'position', 'team',
                'status', 'depth_chart_position', 'jersey_number',
                'years_exp', 'rookie_year'
            ]
            
            rows = self._insert_dataframe(df, 'rosters', columns)
            
            logger.info(f"Inserted {rows} roster rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"Rosters error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    
    def ingest_injuries(self, seasons: List[int]) -> Dict:
        """
        Ingest injury report data.
        
        Note: Injuries data may not be available for all seasons (404 for future seasons).
        """
        logger.info(f"Ingesting injuries for seasons {seasons}")
        
        total_rows = 0
        
        for season in seasons:
            try:
                df_raw = nfl.load_injuries([season])
                df = polars_to_pandas(df_raw)
                
                if df is None:
                    logger.warning(f"No injuries data for season {season}")
                    continue
                
                logger.info(f"Loaded {len(df)} injury rows for {season}")
                
                # CRITICAL FIX: Map gsis_id → player_id
                if 'gsis_id' in df.columns:
                    df['player_id'] = df['gsis_id']
                
                # CRITICAL FIX: Map full_name → player_name
                if 'full_name' in df.columns:
                    df['player_name'] = df['full_name']
                
                columns = [
                    'player_id', 'player_name', 'season', 'week',
                    'team', 'position',
                    'report_primary_injury', 'report_secondary_injury', 'report_status',
                    'practice_primary_injury', 'practice_secondary_injury', 'practice_status',
                    'date_modified'
                ]
                
                rows = self._insert_dataframe(df, 'injuries', columns)
                total_rows += rows
                
            except Exception as e:
                error_str = str(e)
                if '404' in error_str or 'Not Found' in error_str:
                    logger.warning(f"Injuries not available for season {season} (404)")
                else:
                    logger.error(f"Injuries error for season {season}: {e}")
                    self.stats['errors'].append(f"injuries_{season}: {e}")
        
        logger.info(f"Inserted {total_rows} total injury rows")
        return {'success': total_rows > 0, 'rows': total_rows}
    
    # =========================================================================
    # SCHEDULES
    # =========================================================================
    
    def ingest_schedules(self, seasons: List[int]) -> Dict:
        """
        Ingest schedule data with Vegas lines.
        
        nflreadpy.load_schedules() returns:
        - game_id, season, game_type, week
        - home_team, away_team, gameday, gametime, weekday
        - location, home_score, away_score
        - spread_line, total_line, roof, surface
        - home_rest (→ home_rest_days), away_rest (→ away_rest_days)
        """
        logger.info(f"Ingesting schedules for seasons {seasons}")
        
        try:
            df_raw = nfl.load_schedules(seasons)
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No schedules data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} schedule rows")
            
            # CRITICAL FIX: Rename rest columns
            if 'home_rest' in df.columns:
                df['home_rest_days'] = df['home_rest']
            if 'away_rest' in df.columns:
                df['away_rest_days'] = df['away_rest']
            
            columns = [
                'game_id', 'season', 'game_type', 'week',
                'home_team', 'away_team',
                'gameday', 'gametime', 'weekday',
                'location',
                'home_score', 'away_score', 'result', 'total',
                'spread_line', 'home_spread_result',
                'total_line', 'over_under_result',
                'roof', 'surface',
                'home_rest_days', 'away_rest_days',
                'is_primetime', 'is_divisional'
            ]
            
            rows = self._insert_dataframe(df, 'schedules', columns)
            
            logger.info(f"Inserted {rows} schedule rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"Schedules error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # COMBINE DATA
    # =========================================================================
    
    def ingest_combine(self) -> Dict:
        """
        Ingest NFL combine data.
        
        No seasons parameter - loads all available combine data.
        """
        logger.info("Ingesting combine data")
        
        try:
            df_raw = nfl.load_combine()
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No combine data returned")
                return {'success': False, 'rows': 0}
            
            logger.info(f"Loaded {len(df)} combine rows")
            
            columns = [
                'player_id', 'player_name', 'season',
                'position', 'school',
                'height', 'weight',
                'hand_size', 'arm_length', 'wingspan',
                'forty_yard', 'bench_press',
                'vertical_jump', 'broad_jump',
                'three_cone', 'shuttle',
                'draft_team', 'draft_round', 'draft_pick', 'draft_overall'
            ]
            
            rows = self._insert_dataframe(df, 'combine_data', columns)
            
            logger.info(f"Inserted {rows} combine rows")
            return {'success': True, 'rows': rows}
            
        except Exception as e:
            logger.error(f"Combine error: {e}")
            self.stats['errors'].append(str(e))
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # FULL INGESTION
    # =========================================================================
    
    def run_full_ingestion(self, seasons: Optional[List[int]] = None) -> Dict:
        """
        Run complete NFLverse data ingestion.
        
        Args:
            seasons: List of seasons to ingest. Defaults to recent seasons.
            
        Returns:
            Dictionary with results for each table.
        """
        if seasons is None:
            seasons = [CURRENT_SEASON - 2, CURRENT_SEASON - 1, CURRENT_SEASON]
        
        logger.info(f"Starting full NFLverse ingestion for seasons {seasons}")
        
        results = {}
        
        # 1. Weekly stats (MOST IMPORTANT)
        results['nflverse_weekly_stats'] = self.ingest_weekly_stats(seasons)
        
        # 2. Next Gen Stats
        results['ngs_passing'] = self.ingest_ngs_passing(seasons)
        results['ngs_rushing'] = self.ingest_ngs_rushing(seasons)
        results['ngs_receiving'] = self.ingest_ngs_receiving(seasons)
        
        # 3. Snap counts
        results['snap_counts'] = self.ingest_snap_counts(seasons)
        
        # 4. Rosters
        results['rosters'] = self.ingest_rosters(seasons)
        
        # 5. Schedules
        results['schedules'] = self.ingest_schedules(seasons)
        
        # 6. Injuries (may 404 for current/future seasons)
        results['injuries'] = self.ingest_injuries(seasons)
        
        # 7. Combine (all-time data)
        results['combine_data'] = self.ingest_combine()
        
        # Summary
        total_rows = sum(r.get('rows', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))
        
        logger.info(f"Full ingestion complete: {total_rows} rows, {successful}/{len(results)} tables")
        
        return {
            'total_rows': total_rows,
            'successful_tables': successful,
            'total_tables': len(results),
            'results': results,
            'errors': self.stats['errors']
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def run(seasons: Optional[List[int]] = None, skip_injuries: bool = False) -> Dict:
    """
    Run NFLverse ingestion as standalone function.
    
    Args:
        seasons: List of seasons to ingest
        skip_injuries: Skip injury ingestion (often 404s)
        
    Returns:
        Ingestion results dictionary
    """
    ingestion = NFLVerseIngestion()
    
    if seasons is None:
        seasons = [CURRENT_SEASON - 2, CURRENT_SEASON - 1, CURRENT_SEASON]
    
    return ingestion.run_full_ingestion(seasons)


def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest NFL data from nflreadpy/nflverse"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Seasons to ingest (default: last 3 seasons)"
    )
    parser.add_argument(
        "--skip-injuries",
        action="store_true",
        help="Skip injury data ingestion"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("NFLverse Data Ingestion v3.0 - Complete Column Fix")
    print("="*60)
    print(f"Data Source: {NFL_DATA_SOURCE}")
    print(f"Seasons: {args.seasons or 'Last 3 seasons'}")
    print("="*60 + "\n")
    
    results = run(seasons=args.seasons, skip_injuries=args.skip_injuries)
    
    print("\n" + "="*60)
    print("INGESTION RESULTS")
    print("="*60)
    print(f"Total Rows: {results['total_rows']}")
    print(f"Tables: {results['successful_tables']}/{results['total_tables']} successful")
    print("\nBy Table:")
    for table, result in results['results'].items():
        status = "✓" if result.get('success', False) else "✗"
        rows = result.get('rows', 0)
        print(f"  {status} {table}: {rows} rows")
    
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        for err in results['errors'][:5]:
            print(f"  - {err}")
    
    print("="*60 + "\n")
    
    return 0 if results['successful_tables'] > 0 else 1


if __name__ == "__main__":
    exit(main())
