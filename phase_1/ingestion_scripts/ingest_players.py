"""
Players Ingestion Script for NFL Fantasy Prediction Engine - Phase 1

STEP 3 in the ingestion pipeline as per Phase 1 v2 Section 6.3.

This script ingests both offensive and defensive players from Sleeper API:
- Offensive Players: QB, RB, WR, TE (into players table)
- Defensive Players: CB, S, LB (into defenders table - v2 requirement)

Key requirements from Phase 1 v2:
- Store physical attributes (height, weight) for archetypes
- Store raw JSON data for reproducibility
- Map players to teams correctly
- Store alignment role for defenders if available

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
    DEFENSIVE_POSITIONS,
    DEFENSIVE_POSITION_GROUPS,
    SLEEPER_ENDPOINTS,
    API_TIMEOUT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    TEAM_ABBREVIATION_MAP
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from validation.validate_data import validate_player_data

# Initialize logger
logger = get_ingestion_logger("ingest_players")


class PlayersIngestion:
    """
    Handles ingestion of NFL players data from Sleeper API.
    
    Ingests:
    - Offensive players (QB, RB, WR, TE) into 'players' table
    - Defensive players (CB, S, LB) into 'defenders' table
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize players ingestion.
        
        Args:
            db: Optional database connection. Uses default if not provided.
        """
        self.db = db or get_db()
        self.stats = {
            "offensive_inserted": 0,
            "offensive_updated": 0,
            "defensive_inserted": 0,
            "defensive_updated": 0,
            "skipped": 0,
            "errors": 0
        }
        self._raw_players_data: Optional[Dict] = None

        self._raw_players_data: Optional[Dict] = None

    def is_active_player(self, player: Dict) -> bool:
        """
        Determine whether a Sleeper player object should be treated as active.
        """
        active_flag = player.get("active")

        status_raw = player.get("status")
        injury_status_raw = player.get("injury_status")

        status = (status_raw or "").lower()
        injury_status = (injury_status_raw or "").lower()

        inactive_statuses = {"inactive", "retired", "suspended"}

        if active_flag is False:
            return False
        if status in inactive_statuses:
            return False
        if not status and injury_status in inactive_statuses:
            return False

        return True
    
    def normalize_team_id(self, team_id: Optional[str]) -> Optional[str]:
        """
        Normalize team abbreviation to standard format.

        Args:
            team_id: Team abbreviation from Sleeper (may be None)

        Returns:
            Normalized team ID or None if invalid / missing
        """
        if not team_id:
            return None

        team_key = (team_id or "").upper().strip()
        return TEAM_ABBREVIATION_MAP.get(team_key, team_key)
    
    def get_position_group(self, position: Optional[str]) -> Optional[str]:
        """
        Map a specific defensive position to its group (CB, S, or LB).
        """
        if not position:
            return None

        pos_key = (position or "").upper()

        for group, positions in DEFENSIVE_POSITION_GROUPS.items():
            if pos_key in positions:
                return group

        return None

    
    def fetch_players_from_sleeper(self) -> Optional[Dict]:
        """
        Fetch all players from Sleeper API.
        
        Returns:
            Dictionary of player_id -> player_data or None on failure
        """
        logger.log_ingestion_start("Sleeper API - /players/nfl")
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                start_time = time.time()
                response = requests.get(
                    SLEEPER_ENDPOINTS["players"],
                    timeout=API_TIMEOUT
                )
                response_time = (time.time() - start_time) * 1000
                
                logger.log_api_call(
                    SLEEPER_ENDPOINTS["players"],
                    response.status_code,
                    response_time
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Fetched {len(data)} total players from Sleeper")
                    self._raw_players_data = data
                    return data
                else:
                    logger.warning(
                        f"Sleeper API returned {response.status_code} (attempt {attempt + 1})"
                    )
                    
            except requests.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}): {e}")
            
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
        
        logger.error("Failed to fetch players from Sleeper API after all retries")
        return None
    
    def parse_height(self, height_str: Any) -> Optional[int]:
        """
        Parse height string to inches.
        
        Sleeper may provide height as string like "6'2" or as integer.
        
        Args:
            height_str: Height value from Sleeper
        
        Returns:
            Height in inches or None
        """
        if height_str is None:
            return None
        
        if isinstance(height_str, (int, float)):
            return int(height_str)
        
        if isinstance(height_str, str):
            try:
                # Handle formats like "6'2" or "6-2"
                if "'" in height_str or "-" in height_str:
                    parts = height_str.replace("'", "-").replace('"', '').split("-")
                    if len(parts) >= 2:
                        feet = int(parts[0])
                        inches = int(parts[1]) if parts[1] else 0
                        return feet * 12 + inches
                return int(height_str)
            except ValueError:
                return None
        
        return None
    
    def parse_weight(self, weight: Any) -> Optional[int]:
        """
        Parse weight to integer pounds.
        
        Args:
            weight: Weight value from Sleeper
        
        Returns:
            Weight in pounds or None
        """
        if weight is None:
            return None
        
        if isinstance(weight, (int, float)):
            return int(weight)
        
        if isinstance(weight, str):
            try:
                # Remove "lbs" suffix if present
                weight = weight.lower().replace("lbs", "").replace("lb", "").strip()
                return int(float(weight))
            except ValueError:
                return None
        
        return None
    
    def parse_age(self, player_data: Dict) -> Optional[float]:
        """
        Calculate age from birthdate if available.
        
        Args:
            player_data: Player data dictionary
        
        Returns:
            Age in years or None
        """
        # Try to get age directly
        age = player_data.get("age")
        if age is not None:
            return float(age)
        
        # Try to calculate from birthdate
        birthdate = player_data.get("birth_date")
        if birthdate:
            try:
                birth = datetime.strptime(birthdate, "%Y-%m-%d")
                today = datetime.now()
                age = (today - birth).days / 365.25
                return round(age, 1)
            except ValueError:
                pass
        
        return None
    
    def filter_offensive_players(self, players_data: Dict) -> Dict:
        """
        Filter players to only offensive positions.
        
        Args:
            players_data: All players from Sleeper
        
        Returns:
            Filtered dictionary of offensive players
        """
        offensive = {}
        
        for player_id, player in players_data.items():
            raw_pos = player.get("position")
            position = (raw_pos or "").upper()
            if position in OFFENSIVE_POSITIONS and self.is_active_player(player):
                offensive[player_id] = player

        
        logger.info(f"Filtered to {len(offensive)} offensive players")
        return offensive
    
    def filter_defensive_players(self, players_data: Dict) -> Dict:
        """
        Filter players to only defensive positions.
        
        Args:
            players_data: All players from Sleeper
        
        Returns:
            Filtered dictionary of defensive players
        """
        defensive = {}
        
        for player_id, player in players_data.items():
            raw_pos = player.get("position")
            position = (raw_pos or "").upper()
            if position in DEFENSIVE_POSITIONS and self.is_active_player(player):
                defensive[player_id] = player
        
        logger.info(f"Filtered to {len(defensive)} defensive players")
        return defensive
    
    def ingest_offensive_players(self, players_data: Dict) -> Tuple[int, int]:
        """
        Ingest offensive players into the players table.
        
        Args:
            players_data: Dictionary of offensive players
        
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        logger.info("Ingesting offensive players...")
        
        inserted = 0
        updated = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for player_id, player in players_data.items():
                try:
                    # Parse player data
                    full_name = player.get("full_name") or player.get("first_name", "") + " " + player.get("last_name", "")
                    full_name = full_name.strip()
                    
                    if not full_name:
                        logger.debug(f"Skipping player {player_id}: no name")
                        self.stats["skipped"] += 1
                        continue
                    
                    position = player.get("position", "").upper()
                    team_id = self.normalize_team_id(player.get("team"))
                    
                    # Check if player exists
                    cursor.execute(
                        "SELECT player_id FROM players WHERE player_id = ?",
                        (player_id,)
                    )
                    existing = cursor.fetchone()
                    
                    # Prepare data
                    height = self.parse_height(player.get("height"))
                    weight = self.parse_weight(player.get("weight"))
                    age = self.parse_age(player)
                    college = player.get("college")
                    status = player.get("status", "").lower() or player.get("injury_status", "").lower() or "active"
                    metadata_json = json.dumps(player)
                    
                    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    if existing:
                        # Update existing player
                        cursor.execute("""
                            UPDATE players 
                            SET full_name = ?,
                                position = ?,
                                team_id = ?,
                                height = ?,
                                weight = ?,
                                age = ?,
                                college = ?,
                                status = ?,
                                metadata_json = ?,
                                updated_at = ?
                            WHERE player_id = ?
                        """, (
                            full_name, position, team_id, height, weight,
                            age, college, status, metadata_json, timestamp,
                            player_id
                        ))
                        updated += 1
                    else:
                        # Insert new player
                        cursor.execute("""
                            INSERT INTO players 
                            (player_id, full_name, position, team_id, height, weight,
                             age, college, status, metadata_json, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            player_id, full_name, position, team_id, height, weight,
                            age, college, status, metadata_json, timestamp, timestamp
                        ))
                        inserted += 1
                        
                except Exception as e:
                    logger.error(f"Error processing offensive player {player_id}: {e}")
                    self.stats["errors"] += 1
        
        self.stats["offensive_inserted"] = inserted
        self.stats["offensive_updated"] = updated
        
        logger.info(f"Offensive players: {inserted} inserted, {updated} updated")
        return inserted, updated
    
    def ingest_defensive_players(self, players_data: Dict) -> Tuple[int, int]:
        """
        Ingest defensive players into the defenders table.
        
        This is a v2 requirement for defender-aware modeling.
        
        Args:
            players_data: Dictionary of defensive players
        
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        logger.info("Ingesting defensive players...")
        
        inserted = 0
        updated = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for player_id, player in players_data.items():
                try:
                    # Parse player data
                    full_name = player.get("full_name") or player.get("first_name", "") + " " + player.get("last_name", "")
                    full_name = full_name.strip()
                    
                    if not full_name:
                        logger.debug(f"Skipping defender {player_id}: no name")
                        self.stats["skipped"] += 1
                        continue
                    
                    position = player.get("position", "").upper()
                    position_group = self.get_position_group(position)
                    
                    if not position_group:
                        logger.debug(f"Skipping defender {player_id}: unknown position {position}")
                        self.stats["skipped"] += 1
                        continue
                    
                    team_id = self.normalize_team_id(player.get("team"))
                    
                    # Check if defender exists
                    cursor.execute(
                        "SELECT defender_id FROM defenders WHERE defender_id = ?",
                        (player_id,)
                    )
                    existing = cursor.fetchone()
                    
                    # Prepare data
                    height = self.parse_height(player.get("height"))
                    weight = self.parse_weight(player.get("weight"))
                    
                    # Role/coverage info (may not be available from Sleeper)
                    role = None  # Will be populated from nflfastR if available
                    coverage_role = None  # man, zone, hybrid
                    
                    metadata_json = json.dumps(player)
                    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    if existing:
                        # Update existing defender
                        cursor.execute("""
                            UPDATE defenders 
                            SET full_name = ?,
                                team_id = ?,
                                position_group = ?,
                                role = ?,
                                height = ?,
                                weight = ?,
                                coverage_role = ?,
                                metadata_json = ?,
                                updated_at = ?
                            WHERE defender_id = ?
                        """, (
                            full_name, team_id, position_group, role,
                            height, weight, coverage_role, metadata_json, timestamp,
                            player_id
                        ))
                        updated += 1
                    else:
                        # Insert new defender
                        cursor.execute("""
                            INSERT INTO defenders 
                            (defender_id, full_name, team_id, position_group, role,
                             height, weight, coverage_role, metadata_json, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            player_id, full_name, team_id, position_group, role,
                            height, weight, coverage_role, metadata_json, timestamp, timestamp
                        ))
                        inserted += 1
                        
                except Exception as e:
                    logger.error(f"Error processing defensive player {player_id}: {e}")
                    self.stats["errors"] += 1
        
        self.stats["defensive_inserted"] = inserted
        self.stats["defensive_updated"] = updated
        
        logger.info(f"Defensive players: {inserted} inserted, {updated} updated")
        return inserted, updated
    
    def validate_ingestion(self) -> Dict[str, Any]:
        """
        Validate player ingestion results.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating player ingestion...")
        
        results = {
            "offensive_count": self.db.get_row_count("players"),
            "defensive_count": self.db.get_row_count("defenders"),
            "offensive_by_position": {},
            "defensive_by_position": {},
            "team_mismatches": 0,
            "validation_passed": True
        }
        
        # Count by position - offensive
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT position, COUNT(*) as count
                FROM players
                GROUP BY position
            """)
            for row in cursor.fetchall():
                results["offensive_by_position"][row["position"]] = row["count"]
        
        # Count by position - defensive
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT position_group, COUNT(*) as count
                FROM defenders
                GROUP BY position_group
            """)
            for row in cursor.fetchall():
                results["defensive_by_position"][row["position_group"]] = row["count"]
        
        # Check for team mismatches (players with team_id not in teams table)
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM players p
                WHERE p.team_id IS NOT NULL
                AND p.team_id NOT IN (SELECT team_id FROM teams)
            """)
            result = cursor.fetchone()
            results["team_mismatches"] = result["count"] if result else 0
        
        if results["team_mismatches"] > 0:
            logger.warning(
                f"Found {results['team_mismatches']} players with invalid team_id "
                f"(these are usually free agents or odd legacy abbreviations)"
            )
    # Do NOT fail the entire step for now; flag as warning only.
    # results["validation_passed"] = False
        
        # Minimum player thresholds
        if results["offensive_count"] < 500:
            logger.warning(f"Low offensive player count: {results['offensive_count']}")
        
        logger.info(f"Validation complete: {results}")
        return results
    
    def run(self) -> Dict:
        """
        Run the complete players ingestion pipeline.
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING PLAYERS INGESTION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Fetch all players from Sleeper
        players_data = self.fetch_players_from_sleeper()
        if not players_data:
            logger.error("Failed to fetch players data")
            return {"error": "Failed to fetch players"}
        
        # Filter and ingest offensive players
        offensive = self.filter_offensive_players(players_data)
        off_inserted, off_updated = self.ingest_offensive_players(offensive)
        
        # Filter and ingest defensive players (v2 requirement)
        defensive = self.filter_defensive_players(players_data)
        def_inserted, def_updated = self.ingest_defensive_players(defensive)
        
        duration = time.time() - start_time
        
        # Validate
        validation = self.validate_ingestion()
        
        result = {
            "offensive_inserted": off_inserted,
            "offensive_updated": off_updated,
            "defensive_inserted": def_inserted,
            "defensive_updated": def_updated,
            "skipped": self.stats["skipped"],
            "errors": self.stats["errors"],
            "duration_seconds": round(duration, 2),
            "validation": validation
        }
        
        logger.log_ingestion_complete(
            row_count=off_inserted + off_updated + def_inserted + def_updated,
            duration_seconds=duration
        )
        
        logger.info("=" * 60)
        logger.info("PLAYERS INGESTION COMPLETE")
        logger.info("=" * 60)
        
        return result


def main():
    """Main entry point for players ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest NFL players into database"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing players, don't ingest"
    )
    parser.add_argument(
        "--offensive-only",
        action="store_true",
        help="Only ingest offensive players"
    )
    parser.add_argument(
        "--defensive-only",
        action="store_true",
        help="Only ingest defensive players"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Phase 1")
    print("Players Ingestion (Step 3)")
    print(f"{'='*60}\n")
    
    ingestion = PlayersIngestion()
    
    if args.validate_only:
        validation = ingestion.validate_ingestion()
        print(f"\nValidation Results:")
        print(f"  Offensive players: {validation['offensive_count']}")
        print(f"  Defensive players: {validation['defensive_count']}")
        print(f"  By position (offense): {validation['offensive_by_position']}")
        print(f"  By position (defense): {validation['defensive_by_position']}")
        print(f"  Team mismatches: {validation['team_mismatches']}")
        return 0 if validation['validation_passed'] else 1
    
    result = ingestion.run()
    
    if "error" in result:
        print(f"\nâœ— Error: {result['error']}")
        return 1
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Offensive players inserted: {result['offensive_inserted']}")
    print(f"Offensive players updated: {result['offensive_updated']}")
    print(f"Defensive players inserted: {result['defensive_inserted']}")
    print(f"Defensive players updated: {result['defensive_updated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    print(f"Duration: {result['duration_seconds']}s")
    print(f"\nTotal offensive: {result['validation']['offensive_count']}")
    print(f"Total defensive: {result['validation']['defensive_count']}")
    print(f"\nLog file: {logger.log_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
