"""
Play-by-Play Data Ingestion via nflreadpy

This module ingests comprehensive play-by-play data from nflreadpy.
PBP data is LARGE - expect ~50,000 plays per season.

IMPORTANT: 
- nflreadpy uses load_pbp() (NOT import_pbp_data like old nfl_data_py)
- nflreadpy returns POLARS DataFrames, not Pandas!
- Use .is_empty() instead of .empty, and .to_pandas() for conversion.

Installation: pip install nflreadpy pandas polars

Author: NFL Fantasy Prediction Engine Team
Version: 2.2 - Fixed for nflreadpy API (load_pbp function)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nflreadpy as nfl
    import pandas as pd
    NFL_DATA_AVAILABLE = True
except ImportError as e:
    NFL_DATA_AVAILABLE = False
    print(f"ERROR: Missing dependency - {e}")
    print("Install with: pip install nflreadpy pandas polars")

from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import CURRENT_SEASON

logger = get_ingestion_logger("ingest_pbp")


def to_pandas_safe(df) -> pd.DataFrame:
    """
    Safely convert nflreadpy result to pandas DataFrame.
    Handles both Polars and Pandas DataFrames.
    """
    if df is None:
        return pd.DataFrame()
    
    # Check if it's already a pandas DataFrame
    if isinstance(df, pd.DataFrame):
        return df
    
    # Check if it's a Polars DataFrame
    if hasattr(df, 'to_pandas'):
        return df.to_pandas()
    
    # Try to convert via pandas
    try:
        return pd.DataFrame(df)
    except:
        return pd.DataFrame()


def is_empty_safe(df) -> bool:
    """
    Safely check if a DataFrame is empty.
    Handles both Polars and Pandas DataFrames.
    """
    if df is None:
        return True
    
    # Polars DataFrame
    if hasattr(df, 'is_empty'):
        return df.is_empty()
    
    # Pandas DataFrame
    if hasattr(df, 'empty'):
        return df.empty
    
    # Check length as fallback
    try:
        return len(df) == 0
    except:
        return True


class PlayByPlayIngestion:
    """
    Ingest play-by-play data from nflreadpy.
    
    WARNING: PBP data is very large. ~50,000+ plays per season.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        if not NFL_DATA_AVAILABLE:
            raise ImportError("nflreadpy and pandas required")
        
        self.db = db or get_db()
        self.stats = {
            'total_plays': 0,
            'by_season': {},
            'errors': [],
        }
    
    def _safe_value(self, val):
        """Convert pandas NA/NaN to None for SQLite."""
        if pd.isna(val):
            return None
        return val
    
    def _get_table_columns(self) -> List[str]:
        """Get the columns that exist in the play_by_play table."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(play_by_play)")
            columns = [row[1] for row in cursor.fetchall()]
        return columns
    
    def ingest_season(self, season: int) -> Dict:
        """
        Ingest play-by-play data for a single season.
        
        Args:
            season: NFL season year
            
        Returns:
            Dict with results
        """
        logger.info(f"Ingesting PBP for {season}")
        print(f"  Loading play-by-play for {season}...")
        
        try:
            # nflreadpy uses load_pbp (NOT import_pbp_data)
            raw_df = nfl.load_pbp(seasons=[season])
            
            if is_empty_safe(raw_df):
                logger.warning(f"No PBP data returned for {season}")
                return {'success': False, 'plays': 0, 'error': 'No data returned'}
            
            # Convert to pandas
            df = to_pandas_safe(raw_df)
            
            if df.empty:
                return {'success': False, 'plays': 0, 'error': 'Conversion failed'}
            
            print(f"  Downloaded {len(df):,} plays for {season}")
            
            # Get columns that exist in our table
            table_columns = self._get_table_columns()
            
            if not table_columns:
                logger.error("play_by_play table not found - run schema_pbp.py first")
                return {'success': False, 'plays': 0, 'error': 'Table not found'}
            
            # Find matching columns between DataFrame and table
            df_columns = set(df.columns)
            matching_columns = [c for c in table_columns if c in df_columns]
            
            if not matching_columns:
                logger.error("No matching columns between data and table")
                return {'success': False, 'plays': 0, 'error': 'No matching columns'}
            
            logger.info(f"Inserting {len(df)} plays with {len(matching_columns)} columns")
            print(f"  Inserting into database ({len(matching_columns)} columns)...")
            
            # Build insert SQL
            placeholders = ", ".join(["?" for _ in matching_columns])
            col_names = ", ".join(matching_columns)
            sql = f"INSERT OR REPLACE INTO play_by_play ({col_names}) VALUES ({placeholders})"
            
            # Insert in batches for better performance
            batch_size = 1000
            rows_inserted = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    for _, row in batch.iterrows():
                        try:
                            values = tuple(self._safe_value(row.get(col)) for col in matching_columns)
                            cursor.execute(sql, values)
                            rows_inserted += 1
                        except Exception as e:
                            logger.debug(f"Row insert error: {e}")
                    
                    # Progress update
                    if (i + batch_size) % 10000 == 0:
                        print(f"    Progress: {min(i + batch_size, len(df)):,}/{len(df):,} plays")
            
            self.stats['total_plays'] += rows_inserted
            self.stats['by_season'][season] = rows_inserted
            
            logger.info(f"Inserted {rows_inserted} plays for {season}")
            print(f"  ✓ Inserted {rows_inserted:,} plays for {season}")
            
            return {'success': True, 'plays': rows_inserted}
            
        except Exception as e:
            logger.error(f"PBP error for {season}: {e}")
            self.stats['errors'].append(f"{season}: {str(e)}")
            return {'success': False, 'plays': 0, 'error': str(e)}
    
    def ingest_seasons(self, seasons: List[int]) -> Dict:
        """
        Ingest play-by-play data for multiple seasons.
        
        Args:
            seasons: List of seasons to ingest
            
        Returns:
            Dict with results
        """
        logger.info(f"Starting PBP ingestion for {seasons}")
        print(f"\n{'='*60}")
        print("Play-by-Play Data Ingestion")
        print(f"Seasons: {seasons}")
        print(f"{'='*60}\n")
        
        for season in seasons:
            self.ingest_season(season)
        
        results = {
            'total_plays': self.stats['total_plays'],
            'by_season': self.stats['by_season'],
            'errors': self.stats['errors'],
        }
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {results['total_plays']:,} total plays ingested")
        for season, count in results['by_season'].items():
            print(f"  {season}: {count:,} plays")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for err in results['errors']:
                print(f"  - {err}")
        print(f"{'='*60}\n")
        
        return results
    
    def get_play_counts(self) -> Dict[int, int]:
        """Get play counts by season from the database."""
        counts = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT season, COUNT(*) as play_count
                FROM play_by_play
                GROUP BY season
                ORDER BY season
            """)
            for row in cursor.fetchall():
                counts[row[0]] = row[1]
        
        return counts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest NFL play-by-play data from nflreadpy"
    )
    parser.add_argument('--seasons', type=int, nargs='+', help='Seasons to ingest')
    parser.add_argument('--current', action='store_true', help='Current season only')
    parser.add_argument('--last-n', type=int, default=3, help='Last N seasons')
    parser.add_argument('--stats', action='store_true', help='Show current play counts')
    
    args = parser.parse_args()
    
    if not NFL_DATA_AVAILABLE:
        print("ERROR: nflreadpy not installed!")
        print("Run: pip install nflreadpy pandas polars")
        return 1
    
    db = get_db()
    ingestion = PlayByPlayIngestion(db)
    
    if args.stats:
        counts = ingestion.get_play_counts()
        print("\nPlay counts by season:")
        for season, count in counts.items():
            print(f"  {season}: {count:,} plays")
        return 0
    
    if args.seasons:
        seasons = args.seasons
    elif args.current:
        seasons = [CURRENT_SEASON]
    else:
        seasons = list(range(CURRENT_SEASON - args.last_n + 1, CURRENT_SEASON + 1))
    
    results = ingestion.ingest_seasons(seasons)
    
    return 0


if __name__ == "__main__":
    exit(main())
