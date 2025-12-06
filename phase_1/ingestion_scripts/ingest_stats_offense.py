"""
Offensive Stats Ingestion Script for NFL Fantasy Prediction Engine - Phase 1

STEP 5 in the ingestion pipeline as per Phase 1 v2 Section 6.5.

This script populates the most critical table: player_game_stats.

For each (season, week):
- Pull offensive stats JSON from Sleeper
- Parse snaps, routes, carries, targets, completions, yards, TDs
- Store opponent_team_id
- Store raw JSON dump for reproducibility
- Insert one row per (player_id, game_id)

Key requirements from Phase 1 v2:
- Every row must include season, week, game_id, timestamp (anti-leakage)
- No duplicate (player_id, game_id) pairs
- Store raw JSON dump for reproducibility

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import requests

from config.settings import (
    OFFENSIVE_POSITIONS,
    TEAM_ABBREVIATION_MAP,
    SLEEPER_ENDPOINTS,
    API_TIMEOUT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    CURRENT_SEASON,
    ROLLING_WINDOW_SEASONS,
    MAX_REGULAR_SEASON_WEEKS
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from validation.validate_data import validate_stat_row

# Initialize logger
logger = get_ingestion_logger("ingest_stats_offense")


class OffensiveStatsIngestion:
    """
    Handles ingestion of offensive player game statistics from Sleeper API.
    
    This populates the player_game_stats table which is the most important
    raw table for all downstream modeling.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize offensive stats ingestion.
        
        Args:
            db: Optional database connection. Uses default if not provided.
        """
        self.db = db or get_db()
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "missing_player": 0,
            "missing_game": 0
        }
        
        # Cache for player metadata
        self._player_positions: Dict[str, str] = {}
        self._player_teams: Dict[str, str] = {}
        self._team_week_game_cache: Dict[Tuple[int, int, str], Optional[Tuple[str, str]]] = {}
        self._team_week_game_cache = {}
        self._load_player_positions()

    
    def _load_player_positions(self):
        """Load player positions and teams from database for filtering."""
        logger.info("Loading player positions and teams...")
        
        results = self.db.fetch_all(
            "SELECT player_id, position, team_id FROM players"
        )
        
        self._player_positions = {}
        self._player_teams = {}
        
        for row in results:
            player_id = row["player_id"]
            position = (row["position"] or "").upper()
            team_id = row["team_id"]
            
            self._player_positions[player_id] = position
            if team_id:
                self._player_teams[player_id] = team_id
        
        logger.info(
            f"Loaded {len(self._player_positions)} player positions, "
            f"{len(self._player_teams)} player team mappings"
        )

    def get_game_for_team(self, season: int, week: int, team_id: str) -> Optional[Tuple[str, str]]:
        """
        Find this team's game_id and opponent for a given season/week using the games table.

        Returns:
            (game_id, opponent_team_id) or None if no game is found.
        """
        row = self.db.fetch_one(
            """
            SELECT game_id, home_team_id, away_team_id
            FROM games
            WHERE season = ? AND week = ?
              AND (home_team_id = ? OR away_team_id = ?)
            """,
            (season, week, team_id, team_id),
        )
        if not row:
            return None

        game_id = row["game_id"]
        if row["home_team_id"] == team_id:
            opponent = row["away_team_id"]
        else:
            opponent = row["home_team_id"]
        return game_id, opponent


    def _load_player_teams(self):
        """Load player team assignments from the players table."""
        logger.info("Loading player teams...")
        
        results = self.db.fetch_all(
            "SELECT player_id, team_id FROM players"
        )
        
        self._player_teams = {
            row["player_id"]: row["team_id"]
            for row in results
        }
        
        logger.info(f"Loaded {len(self._player_teams)} player teams")
    
    def normalize_team_id(self, team_id: str) -> Optional[str]:
        """
        Normalize team abbreviation to standard format.
        
        Args:
            team_id: Team abbreviation
        
        Returns:
            Normalized team ID or None
        """
        if not team_id:
            return None
        
        team_id = team_id.upper().strip()
        return TEAM_ABBREVIATION_MAP.get(team_id, team_id)
    
    def get_game_for_team(self, season: int, week: int, team_id: str) -> Optional[Tuple[str, str]]:
        """
        Given a season, week, and team, find the corresponding game and opponent
        using the games table.

        Returns:
            (game_id, opponent_team_id) or None if not found.
        """
        if not team_id:
            return None

        key = (season, week, team_id)
        if key in self._team_week_game_cache:
            return self._team_week_game_cache[key]

        row = self.db.fetch_one(
            """
            SELECT game_id, home_team_id, away_team_id
            FROM games
            WHERE season = ? AND week = ?
              AND (home_team_id = ? OR away_team_id = ?)
            """,
            (season, week, team_id, team_id),
        )

        if not row:
            logger.warning(
                f"No game found for team {team_id} in {season} week {week}"
            )
            self._team_week_game_cache[key] = None
            return None

        home = row["home_team_id"]
        away = row["away_team_id"]
        opponent = away if home == team_id else home

        result = (row["game_id"], opponent)
        self._team_week_game_cache[key] = result
        return result
    
    def generate_game_id(self, season: int, week: int, 
                         team: str, opponent: str) -> str:
        """
        Generate game ID matching the format from ingest_games.
        
        The canonical format is: {season}_{week:02d}_{away}_{home}
        Since we don't know home/away, use alphabetical ordering.
        
        Args:
            season: Season year
            week: Week number
            team: Player's team
            opponent: Opponent team
        
        Returns:
            Game ID string
        """
        team1, team2 = sorted([team, opponent])
        return f"{season}_{week:02d}_{team1}_{team2}"
    
    def fetch_week_stats(self, season: int, week: int) -> Optional[Dict]:
        """
        Fetch offensive stats for a specific week from Sleeper.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            Stats dictionary or None
        """
        url = SLEEPER_ENDPOINTS["stats_regular"](season, week)
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=API_TIMEOUT)
                response_time = (time.time() - start_time) * 1000
                
                logger.log_api_call(url, response.status_code, response_time)
                
                if response.status_code == 200:
                    data = response.json()
                    return data if data else None
                elif response.status_code == 404:
                    return None
                    
            except requests.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
        
        return None
    
    def parse_stat_value(self, value: Any, stat_type: str = "real") -> Any:
        """
        Parse a stat value safely.
        
        Args:
            value: Raw value from Sleeper
            stat_type: "int" or "real"
        
        Returns:
            Parsed value or default
        """
        if value is None:
            return 0 if stat_type in ["int", "real"] else None
        
        try:
            if stat_type == "int":
                return int(float(value))
            elif stat_type == "real":
                return float(value)
            return value
        except (ValueError, TypeError):
            return 0 if stat_type in ["int", "real"] else None
    
    def extract_player_stats(self, player_id: str, stats: Dict,
                             season: int, week: int) -> Optional[Dict]:
        """
        Extract and format stats for a single offensive player.

        Args:
            player_id: Sleeper player ID
            stats: Raw stats dictionary from Sleeper
            season: Season year
            week: Week number

        Returns:
            Formatted stats dictionary or None if invalid
        """
        # Must be a stats dict
        if not isinstance(stats, dict):
            return None

        # Filter to offensive positions (QB/RB/WR/TE)
        position = (self._player_positions.get(player_id) or "").upper()
        if position not in OFFENSIVE_POSITIONS:
            return None

        # Determine player's team from players table cache
        team = self._player_teams.get(player_id)
        if not team:
            # We don't know which team this player belongs to
            self.stats["missing_player"] += 1
            return None

        # Use the games table to find this team's game (and opponent) for the week
        game_info = self.get_game_for_team(season, week, team)
        if not game_info:
            # No game found for this team in that week
            self.stats["missing_game"] += 1
            return None

        game_id, opponent = game_info

        # Parse all stats
        return {
            "player_id": player_id,
            "game_id": game_id,
            "team_id": team,
            "opponent_team_id": opponent,
            "season": season,
            "week": week,

            # Snap and usage stats
            "snaps": self.parse_stat_value(stats.get("off_snp"), "int"),
            "routes": self.parse_stat_value(
                stats.get("routes_run") or stats.get("tm_pass_att"),
                "int",
            ),

            # Rushing stats
            "carries": self.parse_stat_value(
                stats.get("rush_att") or stats.get("car"),
                "int",
            ),
            "rush_yards": self.parse_stat_value(
                stats.get("rush_yd") or stats.get("rush_yds"),
                "real",
            ),
            "rush_tds": self.parse_stat_value(stats.get("rush_td"), "real"),

            # Receiving stats
            "targets": self.parse_stat_value(
                stats.get("rec_tgt") or stats.get("tar"),
                "int",
            ),
            "receptions": self.parse_stat_value(
                stats.get("rec") or stats.get("receptions"),
                "int",
            ),
            "rec_yards": self.parse_stat_value(
                stats.get("rec_yd") or stats.get("rec_yds"),
                "real",
            ),
            "rec_tds": self.parse_stat_value(stats.get("rec_td"), "real"),

            # Passing stats (for QBs)
            "completions": self.parse_stat_value(
                stats.get("pass_cmp") or stats.get("cmp"),
                "int",
            ),
            "pass_attempts": self.parse_stat_value(
                stats.get("pass_att") or stats.get("att"),
                "int",
            ),
            "pass_yards": self.parse_stat_value(
                stats.get("pass_yd") or stats.get("pass_yds"),
                "real",
            ),
            "pass_tds": self.parse_stat_value(stats.get("pass_td"), "real"),
            "interceptions": self.parse_stat_value(
                stats.get("pass_int") or stats.get("int"),
                "int",
            ),

            # Other stats
            "fumbles": self.parse_stat_value(stats.get("fum"), "real"),
            "fumbles_lost": self.parse_stat_value(
                stats.get("fum_lost"),
                "real",
            ),
            "two_point_conversions": self.parse_stat_value(
                stats.get("pass_2pt") or stats.get("rush_2pt") or stats.get("rec_2pt"),
                "int",
            ),

            # Fantasy points
            "fantasy_points_sleeper": self.parse_stat_value(
                stats.get("pts_half_ppr") or stats.get("pts_ppr"),
                "real",
            ),

            # Raw JSON for reproducibility
            "raw_json": json.dumps(stats),
        }

    def insert_player_stats(self, stats: Dict, cursor) -> bool:
        """
        Insert or update a player's stats for a game.
        
        Args:
            stats: Formatted stats dictionary
            cursor: Database cursor
        
        Returns:
            True if inserted/updated, False if skipped/error
        """
        try:
            # Check for existing record
            cursor.execute(
                "SELECT player_id FROM player_game_stats WHERE player_id = ? AND game_id = ?",
                (stats["player_id"], stats["game_id"])
            )
            existing = cursor.fetchone()
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE player_game_stats SET
                        team_id = ?,
                        opponent_team_id = ?,
                        snaps = ?,
                        routes = ?,
                        carries = ?,
                        rush_yards = ?,
                        rush_tds = ?,
                        targets = ?,
                        receptions = ?,
                        rec_yards = ?,
                        rec_tds = ?,
                        completions = ?,
                        pass_attempts = ?,
                        pass_yards = ?,
                        pass_tds = ?,
                        interceptions = ?,
                        fumbles = ?,
                        fumbles_lost = ?,
                        two_point_conversions = ?,
                        fantasy_points_sleeper = ?,
                        raw_json = ?,
                        ingested_at = ?
                    WHERE player_id = ? AND game_id = ?
                """, (
                    stats["team_id"],
                    stats["opponent_team_id"],
                    stats["snaps"],
                    stats["routes"],
                    stats["carries"],
                    stats["rush_yards"],
                    stats["rush_tds"],
                    stats["targets"],
                    stats["receptions"],
                    stats["rec_yards"],
                    stats["rec_tds"],
                    stats["completions"],
                    stats["pass_attempts"],
                    stats["pass_yards"],
                    stats["pass_tds"],
                    stats["interceptions"],
                    stats["fumbles"],
                    stats["fumbles_lost"],
                    stats["two_point_conversions"],
                    stats["fantasy_points_sleeper"],
                    stats["raw_json"],
                    timestamp,
                    stats["player_id"],
                    stats["game_id"]
                ))
                self.stats["updated"] += 1
                return True
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO player_game_stats 
                    (player_id, game_id, team_id, opponent_team_id, season, week,
                     snaps, routes, carries, rush_yards, rush_tds,
                     targets, receptions, rec_yards, rec_tds,
                     completions, pass_attempts, pass_yards, pass_tds, interceptions,
                     fumbles, fumbles_lost, two_point_conversions,
                     fantasy_points_sleeper, raw_json, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats["player_id"],
                    stats["game_id"],
                    stats["team_id"],
                    stats["opponent_team_id"],
                    stats["season"],
                    stats["week"],
                    stats["snaps"],
                    stats["routes"],
                    stats["carries"],
                    stats["rush_yards"],
                    stats["rush_tds"],
                    stats["targets"],
                    stats["receptions"],
                    stats["rec_yards"],
                    stats["rec_tds"],
                    stats["completions"],
                    stats["pass_attempts"],
                    stats["pass_yards"],
                    stats["pass_tds"],
                    stats["interceptions"],
                    stats["fumbles"],
                    stats["fumbles_lost"],
                    stats["two_point_conversions"],
                    stats["fantasy_points_sleeper"],
                    stats["raw_json"],
                    timestamp
                ))
                self.stats["inserted"] += 1
                return True
                
        except Exception as e:
            logger.error(f"Error inserting stats for {stats['player_id']}: {e}")
            self.stats["errors"] += 1
            return False
    
    def ingest_week(self, season: int, week: int) -> int:
        """
        Ingest all offensive stats for a specific week.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            Number of records processed
        """
        logger.info(f"Ingesting offensive stats for {season} week {week}...")
        
        stats_data = self.fetch_week_stats(season, week)
        
        if not stats_data:
            logger.info(f"No stats data for {season} week {week}")
            return 0
        
        processed = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for player_id, player_stats in stats_data.items():
                # Check if this is an offensive player
                position = self._player_positions.get(player_id)
                
                if position not in OFFENSIVE_POSITIONS:
                    # Could be a defensive player or unknown
                    self.stats["skipped"] += 1
                    continue
                
                # Extract and format stats
                formatted_stats = self.extract_player_stats(
                    player_id, player_stats, season, week
                )
                
                if not formatted_stats:
                    self.stats["skipped"] += 1
                    continue
                
                # Insert stats
                if self.insert_player_stats(formatted_stats, cursor):
                    processed += 1
        
        logger.info(f"Processed {processed} offensive stat records for {season} week {week}")
        return processed
    
    def ingest_season(self, season: int, 
                      max_weeks: int = MAX_REGULAR_SEASON_WEEKS) -> int:
        """
        Ingest all offensive stats for a season.
        
        Args:
            season: Season year
            max_weeks: Maximum weeks to process
        
        Returns:
            Total records processed
        """
        logger.info(f"Ingesting offensive stats for season {season}...")
        
        total_processed = 0
        consecutive_empty = 0
        
        for week in range(1, max_weeks + 1):
            processed = self.ingest_week(season, week)
            total_processed += processed
            
            if processed == 0:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info(f"End of {season} season detected at week {week}")
                    break
            else:
                consecutive_empty = 0
            
            # Rate limiting
            time.sleep(0.5)
        
        return total_processed
    
    def ingest_rolling_window(self) -> Dict[int, int]:
        """
        Ingest offensive stats for the rolling window of seasons.
        
        Returns:
            Dictionary of season -> records processed
        """
        results = {}
        
        start_season = CURRENT_SEASON - ROLLING_WINDOW_SEASONS + 1
        
        logger.info(f"Ingesting offensive stats from {start_season} to {CURRENT_SEASON}")
        
        for season in range(start_season, CURRENT_SEASON + 1):
            count = self.ingest_season(season)
            results[season] = count
            logger.info(f"Season {season}: {count} records processed")
        
        return results
    
    def validate_stats(self) -> Dict[str, Any]:
        """
        Validate offensive stats ingestion.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating offensive stats...")
        
        results = {
            "total_records": self.db.get_row_count("player_game_stats"),
            "by_season": {},
            "by_position": {},
            "validation_passed": True,
            "issues": []
        }
        
        # Count by season
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT season, COUNT(*) as count
                FROM player_game_stats
                GROUP BY season
                ORDER BY season
            """)
            for row in cursor.fetchall():
                results["by_season"][row["season"]] = row["count"]
        
        # Count by position (join with players)
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.position, COUNT(*) as count
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                GROUP BY p.position
            """)
            for row in cursor.fetchall():
                results["by_position"][row["position"]] = row["count"]
        
        # Check for missing foreign keys
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM player_game_stats pgs
                WHERE pgs.player_id NOT IN (SELECT player_id FROM players)
            """)
            result = cursor.fetchone()
            orphan_count = result["count"] if result else 0
            if orphan_count > 0:
                results["issues"].append(f"{orphan_count} stats with missing player")
                results["validation_passed"] = False
        
        # Check for missing season/week (anti-leakage)
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM player_game_stats
                WHERE season IS NULL OR week IS NULL
            """)
            result = cursor.fetchone()
            missing_time = result["count"] if result else 0
            if missing_time > 0:
                results["issues"].append(f"{missing_time} stats with missing season/week")
                results["validation_passed"] = False
        
        logger.info(f"Validation results: {results}")
        return results
    
    def run(self, seasons: Optional[List[int]] = None, 
            weeks: Optional[List[int]] = None) -> Dict:
        """
        Run the offensive stats ingestion pipeline.
        
        Args:
            seasons: Optional list of specific seasons
            weeks: Optional list of specific weeks (only with single season)
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING OFFENSIVE STATS INGESTION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        if seasons and weeks and len(seasons) == 1:
            # Specific season and weeks
            results_by_season = {}
            total = 0
            for week in weeks:
                count = self.ingest_week(seasons[0], week)
                total += count
            results_by_season[seasons[0]] = total
        elif seasons:
            # Specific seasons, all weeks
            results_by_season = {}
            for season in seasons:
                count = self.ingest_season(season)
                results_by_season[season] = count
        else:
            # Rolling window
            results_by_season = self.ingest_rolling_window()
        
        duration = time.time() - start_time
        
        # Validate
        validation = self.validate_stats()
        
        result = {
            "inserted": self.stats["inserted"],
            "updated": self.stats["updated"],
            "skipped": self.stats["skipped"],
            "errors": self.stats["errors"],
            "by_season": results_by_season,
            "duration_seconds": round(duration, 2),
            "validation": validation
        }
        
        logger.log_ingestion_complete(
            row_count=self.stats["inserted"] + self.stats["updated"],
            duration_seconds=duration
        )
        
        logger.info("=" * 60)
        logger.info("OFFENSIVE STATS INGESTION COMPLETE")
        logger.info("=" * 60)
        
        return result


def main():
    """Main entry point for offensive stats ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest offensive player stats into database"
    )
    parser.add_argument(
        "--season",
        type=int,
        nargs="+",
        help="Specific season(s) to ingest"
    )
    parser.add_argument(
        "--week",
        type=int,
        nargs="+",
        help="Specific week(s) to ingest (requires single --season)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing stats"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Phase 1")
    print("Offensive Stats Ingestion (Step 5)")
    print(f"{'='*60}\n")
    
    ingestion = OffensiveStatsIngestion()
    
    if args.validate_only:
        validation = ingestion.validate_stats()
        print(f"\nValidation Results:")
        print(f"  Total records: {validation['total_records']}")
        print(f"  By season: {validation['by_season']}")
        print(f"  By position: {validation['by_position']}")
        if validation['issues']:
            print(f"  Issues: {validation['issues']}")
        return 0 if validation['validation_passed'] else 1
    
    result = ingestion.run(seasons=args.seasons, weeks=args.week)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Stats inserted: {result['inserted']}")
    print(f"Stats updated: {result['updated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    print(f"Duration: {result['duration_seconds']}s")
    print(f"\nBy season:")
    for season, count in result['by_season'].items():
        print(f"  {season}: {count} records")
    print(f"\nLog file: {logger.log_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
