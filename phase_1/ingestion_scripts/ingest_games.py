"""
================================================================================
FIXED Games Ingestion Script v3.0
================================================================================

This script correctly ingests games using nflreadpy.load_schedules().

CRITICAL FIXES:
1. Uses nflreadpy.load_schedules() as primary data source
2. Polars → Pandas conversion
3. Correct column mapping to games table schema
4. NO home_score/away_score (those go in schedules table, not games)
5. Fallback to Sleeper API if nflreadpy unavailable

Schema for games table (from create_schema.py):
- game_id TEXT PRIMARY KEY
- season INTEGER NOT NULL
- week INTEGER NOT NULL  
- home_team_id TEXT NOT NULL
- away_team_id TEXT NOT NULL
- datetime TEXT
- stadium TEXT
- weather_json TEXT
- created_at TEXT
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try nflreadpy first
try:
    import nflreadpy as nfl
    import pandas as pd
    NFLREADPY_AVAILABLE = True
except ImportError:
    NFLREADPY_AVAILABLE = False

import requests

from config.settings import (
    NFL_TEAMS,
    TEAM_ABBREVIATION_MAP,
    SLEEPER_BASE_URL,
    API_TIMEOUT,
    CURRENT_SEASON,
    ROLLING_WINDOW_SEASONS,
    MAX_REGULAR_SEASON_WEEKS
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

logger = get_ingestion_logger("ingest_games")


def polars_to_pandas(df):
    """Convert Polars DataFrame to Pandas DataFrame."""
    if df is None:
        return None
    if hasattr(df, 'iterrows'):
        return df if len(df) > 0 else None
    try:
        pdf = df.to_pandas()
        return pdf if len(pdf) > 0 else None
    except Exception as e:
        logger.error(f"Polars conversion error: {e}")
        return None


class GamesIngestion:
    """
    Handles ingestion of NFL game schedule data.
    
    Primary source: nflreadpy.load_schedules()
    Fallback: Sleeper API
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_db()
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }
    
    def normalize_team_id(self, team_id: str) -> Optional[str]:
        """Normalize team abbreviation to standard format."""
        if not team_id:
            return None
        
        team_id_upper = str(team_id).upper().strip()
        
        # Direct match in NFL_TEAMS
        if team_id_upper in NFL_TEAMS:
            return team_id_upper
        
        # Check abbreviation map
        return TEAM_ABBREVIATION_MAP.get(team_id_upper)
    
    def generate_game_id(self, season: int, week: int, away_team: str, home_team: str) -> str:
        """Generate unique game_id in nflverse format: YYYY_WW_AWAY_HOME"""
        return f"{season}_{week:02d}_{away_team}_{home_team}"
    
    def _insert_game(self, game: Dict) -> bool:
        """Insert a single game record."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if game exists
                cursor.execute(
                    "SELECT game_id FROM games WHERE game_id = ?",
                    (game["game_id"],)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing game
                    cursor.execute("""
                        UPDATE games 
                        SET datetime = ?, stadium = ?
                        WHERE game_id = ?
                    """, (
                        game.get("datetime"),
                        game.get("stadium"),
                        game["game_id"]
                    ))
                    self.stats["updated"] += 1
                    return True
                
                # Insert new game (CORRECT SCHEMA - no home_score/away_score)
                cursor.execute("""
                    INSERT INTO games 
                    (game_id, season, week, home_team_id, away_team_id, 
                     datetime, stadium, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game["game_id"],
                    game["season"],
                    game["week"],
                    game["home_team_id"],
                    game["away_team_id"],
                    game.get("datetime"),
                    game.get("stadium"),
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                ))
                
                self.stats["inserted"] += 1
                return True
                
        except Exception as e:
            logger.error(f"Error inserting game {game.get('game_id')}: {e}")
            self.stats["errors"] += 1
            return False
    
    def ingest_from_nflreadpy(self, seasons: List[int]) -> int:
        """
        Ingest games using nflreadpy.load_schedules().
        
        This is the PRIMARY and BEST data source.
        """
        if not NFLREADPY_AVAILABLE:
            logger.warning("nflreadpy not available")
            return 0
        
        logger.info(f"Loading schedules from nflreadpy for seasons {seasons}")
        
        try:
            df_raw = nfl.load_schedules(seasons)
            df = polars_to_pandas(df_raw)
            
            if df is None:
                logger.warning("No schedule data returned from nflreadpy")
                return 0
            
            logger.info(f"Loaded {len(df)} games from nflreadpy")
            
            games_inserted = 0
            
            for _, row in df.iterrows():
                # Get game_id from data or generate it
                game_id = row.get('game_id')
                if not game_id:
                    game_id = self.generate_game_id(
                        int(row['season']),
                        int(row['week']),
                        str(row['away_team']),
                        str(row['home_team'])
                    )
                
                # Normalize team IDs
                home_team = self.normalize_team_id(str(row.get('home_team', '')))
                away_team = self.normalize_team_id(str(row.get('away_team', '')))
                
                if not home_team or not away_team:
                    logger.warning(f"Invalid teams for game {game_id}")
                    continue
                
                # Build datetime from gameday + gametime
                datetime_str = None
                if pd.notna(row.get('gameday')):
                    datetime_str = str(row['gameday'])
                    if pd.notna(row.get('gametime')):
                        datetime_str += f"T{row['gametime']}"
                
                # Get stadium/location
                stadium = row.get('location') or row.get('stadium')
                
                game = {
                    "game_id": game_id,
                    "season": int(row['season']),
                    "week": int(row['week']),
                    "home_team_id": home_team,
                    "away_team_id": away_team,
                    "datetime": datetime_str,
                    "stadium": stadium
                }
                
                if self._insert_game(game):
                    games_inserted += 1
            
            logger.info(f"Inserted/updated {games_inserted} games from nflreadpy")
            return games_inserted
            
        except Exception as e:
            logger.error(f"nflreadpy schedule ingestion error: {e}")
            return 0
    
    def ingest_season_games(self, season: int, max_weeks: int = MAX_REGULAR_SEASON_WEEKS) -> int:
        """
        Ingest all games for a season.
        
        Primary: nflreadpy.load_schedules()
        Fallback: Sleeper API
        """
        logger.info(f"Ingesting games for season {season}")
        
        # Try nflreadpy first
        if NFLREADPY_AVAILABLE:
            count = self.ingest_from_nflreadpy([season])
            if count > 0:
                return count
        
        # Fallback: Use Sleeper stats to infer games
        logger.info(f"Using Sleeper stats fallback for season {season}")
        return self._ingest_from_sleeper_stats(season, max_weeks)
    
    def _ingest_from_sleeper_stats(self, season: int, max_weeks: int) -> int:
        """Fallback: Infer games from Sleeper stats data."""
        # This is the old approach - extract unique game combinations from stats
        games_found = set()
        
        for week in range(1, max_weeks + 1):
            try:
                url = f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}/{week}"
                response = requests.get(url, timeout=API_TIMEOUT)
                
                if response.status_code != 200:
                    continue
                
                stats = response.json()
                
                # Extract team matchups from stats
                matchups = {}
                for player_id, player_stats in stats.items():
                    team = player_stats.get('team')
                    opp = player_stats.get('opponent')
                    
                    if team and opp:
                        # Normalize
                        team = self.normalize_team_id(team)
                        opp = self.normalize_team_id(opp)
                        
                        if team and opp:
                            # Create canonical game key
                            game_key = tuple(sorted([team, opp]))
                            if game_key not in matchups:
                                matchups[game_key] = {'teams': [team, opp]}
                
                # Insert games from matchups
                for game_key, info in matchups.items():
                    teams = info['teams']
                    # Determine home/away (approximation)
                    game_id = self.generate_game_id(season, week, teams[0], teams[1])
                    
                    if game_id not in games_found:
                        game = {
                            "game_id": game_id,
                            "season": season,
                            "week": week,
                            "home_team_id": teams[1],
                            "away_team_id": teams[0],
                            "datetime": None,
                            "stadium": None
                        }
                        self._insert_game(game)
                        games_found.add(game_id)
                        
            except Exception as e:
                logger.error(f"Sleeper stats error week {week}: {e}")
        
        return len(games_found)
    
    def run(self, seasons: Optional[List[int]] = None) -> Dict:
        """
        Run games ingestion for multiple seasons.
        
        Args:
            seasons: List of seasons. Defaults to ROLLING_WINDOW_SEASONS.
        """
        if seasons is None:
            seasons = list(range(
                CURRENT_SEASON - ROLLING_WINDOW_SEASONS + 1,
                CURRENT_SEASON + 1
            ))
        
        logger.info(f"Starting games ingestion for seasons {seasons}")
        
        # Reset stats
        self.stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
        
        # Try nflreadpy for all seasons at once (more efficient)
        if NFLREADPY_AVAILABLE:
            self.ingest_from_nflreadpy(seasons)
        else:
            # Fall back to per-season Sleeper approach
            for season in seasons:
                self.ingest_season_games(season)
        
        total = self.stats["inserted"] + self.stats["updated"]
        
        logger.info(f"Games ingestion complete: {total} games")
        logger.info(f"  Inserted: {self.stats['inserted']}")
        logger.info(f"  Updated: {self.stats['updated']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        return {
            'success': total > 0,
            'total': total,
            'inserted': self.stats['inserted'],
            'updated': self.stats['updated'],
            'errors': self.stats['errors']
        }


def run(seasons: Optional[List[int]] = None) -> Dict:
    """Standalone function to run games ingestion."""
    ingestion = GamesIngestion()
    return ingestion.run(seasons)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest NFL games")
    parser.add_argument("--seasons", type=int, nargs="+", help="Seasons to ingest")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Games Ingestion v3.0 - Fixed Schema")
    print("="*60)
    print(f"nflreadpy available: {NFLREADPY_AVAILABLE}")
    print("="*60 + "\n")
    
    results = run(seasons=args.seasons)
    
    print("\nResults:")
    print(f"  Total: {results['total']}")
    print(f"  Inserted: {results['inserted']}")
    print(f"  Updated: {results['updated']}")
    print(f"  Errors: {results['errors']}")
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit(main())
