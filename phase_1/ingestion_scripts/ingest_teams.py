"""
Teams Ingestion Script for NFL Fantasy Prediction Engine - Phase 1

STEP 2 in the ingestion pipeline as per Phase 1 v2 Section 6.2.

This script ingests all 32 NFL teams from either:
1. Static configuration (preferred for consistency)
2. Sleeper API (as backup/validation)

The teams table rarely changes but must remain correct as it's referenced
by foreign keys in players, defenders, games, and all stat tables.

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import requests

from config.settings import (
    NFL_TEAMS, 
    TEAM_ABBREVIATION_MAP,
    SLEEPER_ENDPOINTS,
    API_TIMEOUT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Initialize logger
logger = get_ingestion_logger("ingest_teams")


class TeamsIngestion:
    """
    Handles ingestion of NFL teams data.
    
    Teams are ingested from static configuration to ensure consistency.
    Sleeper API can be used for validation or additional data.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize teams ingestion.
        
        Args:
            db: Optional database connection. Uses default if not provided.
        """
        self.db = db or get_db()
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }
    
    def normalize_team_id(self, team_id: str) -> str:
        """
        Normalize team abbreviation to standard format.
        
        Handles variations like JAC -> JAX, WSH -> WAS, etc.
        
        Args:
            team_id: Team abbreviation
        
        Returns:
            Normalized team ID
        """
        if not team_id:
            return team_id
        
        team_id = team_id.upper().strip()
        return TEAM_ABBREVIATION_MAP.get(team_id, team_id)
    
    def ingest_from_static(self) -> Tuple[int, int]:
        """
        Ingest teams from static configuration.
        
        This is the primary method for team ingestion as it ensures
        consistency and doesn't depend on external APIs.
        
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        logger.log_ingestion_start("static configuration")
        start_time = time.time()
        
        inserted = 0
        updated = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for team_id, team_info in NFL_TEAMS.items():
                try:
                    # Check if team exists
                    cursor.execute(
                        "SELECT team_id FROM teams WHERE team_id = ?",
                        (team_id,)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing team
                        cursor.execute("""
                            UPDATE teams 
                            SET team_name = ?,
                                abbreviation = ?,
                                conference = ?,
                                division = ?,
                                updated_at = ?
                            WHERE team_id = ?
                        """, (
                            team_info["name"],
                            team_id,
                            team_info["conference"],
                            team_info["division"],
                            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            team_id
                        ))
                        updated += 1
                        logger.debug(f"Updated team: {team_id}", team_id=team_id)
                    else:
                        # Insert new team
                        cursor.execute("""
                            INSERT INTO teams 
                            (team_id, team_name, abbreviation, conference, division, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            team_id,
                            team_info["name"],
                            team_id,
                            team_info["conference"],
                            team_info["division"],
                            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                        ))
                        inserted += 1
                        logger.debug(f"Inserted team: {team_id}", team_id=team_id)
                        
                except Exception as e:
                    logger.error(f"Error processing team {team_id}: {e}", team_id=team_id)
                    self.stats["errors"] += 1
        
        duration = time.time() - start_time
        self.stats["inserted"] = inserted
        self.stats["updated"] = updated
        
        logger.log_ingestion_complete(
            row_count=inserted + updated,
            expected_count=len(NFL_TEAMS),
            duration_seconds=duration
        )
        
        return inserted, updated
    
    def fetch_teams_from_sleeper(self) -> Optional[Dict]:
        """
        Fetch teams data from Sleeper API.
        
        Note: Sleeper doesn't have a dedicated teams endpoint, but we can
        extract team information from the players endpoint.
        
        Returns:
            Dictionary of team data or None on failure
        """
        logger.log_ingestion_start("Sleeper API")
        
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
                    return response.json()
                else:
                    logger.warning(
                        f"Sleeper API returned {response.status_code} (attempt {attempt + 1})"
                    )
                    
            except requests.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}): {e}")
            
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
        
        logger.error("Failed to fetch from Sleeper API after all retries")
        return None
    
    def extract_teams_from_players(self, players_data: Dict) -> Dict[str, Dict]:
        """
        Extract unique teams from Sleeper players data.
        
        Args:
            players_data: Raw players data from Sleeper API
        
        Returns:
            Dictionary of team_id -> team_info
        """
        teams = {}
        
        for player_id, player in players_data.items():
            team = player.get("team")
            if team and team not in teams:
                # Normalize team ID
                normalized_team = self.normalize_team_id(team)
                
                # Get static info if available, otherwise create basic entry
                if normalized_team in NFL_TEAMS:
                    teams[normalized_team] = NFL_TEAMS[normalized_team].copy()
                else:
                    teams[normalized_team] = {
                        "name": f"Unknown Team ({normalized_team})",
                        "conference": "Unknown",
                        "division": "Unknown"
                    }
        
        return teams
    
    def validate_teams(self) -> bool:
        """
        Validate that all 32 NFL teams are present.
        
        Returns:
            True if validation passes
        """
        logger.info("Validating teams...")
        
        row_count = self.db.get_row_count("teams")
        expected_count = len(NFL_TEAMS)
        
        if row_count != expected_count:
            logger.warning(
                f"Team count mismatch: found {row_count}, expected {expected_count}"
            )
            return False
        
        # Check each team exists
        missing_teams = []
        for team_id in NFL_TEAMS.keys():
            result = self.db.fetch_one(
                "SELECT team_id FROM teams WHERE team_id = ?",
                (team_id,)
            )
            if not result:
                missing_teams.append(team_id)
        
        if missing_teams:
            logger.error(f"Missing teams: {missing_teams}")
            return False
        
        logger.info(f"Validation passed: {row_count} teams present")
        return True
    
    def run(self, use_api: bool = False) -> Dict:
        """
        Run the teams ingestion pipeline.
        
        Args:
            use_api: If True, use Sleeper API. If False, use static config.
        
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING TEAMS INGESTION")
        logger.info("=" * 60)
        
        if use_api:
            # Fetch from API and merge with static data
            api_data = self.fetch_teams_from_sleeper()
            if api_data:
                api_teams = self.extract_teams_from_players(api_data)
                logger.info(f"Found {len(api_teams)} teams from API")
        
        # Always use static configuration for consistency
        inserted, updated = self.ingest_from_static()
        
        # Validate
        validation_passed = self.validate_teams()
        
        result = {
            "inserted": inserted,
            "updated": updated,
            "errors": self.stats["errors"],
            "validation_passed": validation_passed,
            "total_teams": self.db.get_row_count("teams")
        }
        
        logger.info("=" * 60)
        logger.info("TEAMS INGESTION COMPLETE")
        logger.info(f"Results: {result}")
        logger.info("=" * 60)
        
        return result


def main():
    """Main entry point for teams ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest NFL teams into database"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Also fetch data from Sleeper API"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing teams, don't ingest"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Phase 1")
    print("Teams Ingestion (Step 2)")
    print(f"{'='*60}\n")
    
    ingestion = TeamsIngestion()
    
    if args.validate_only:
        if ingestion.validate_teams():
            print("\nâœ“ Validation passed!")
            return 0
        else:
            print("\nâœ— Validation failed!")
            return 1
    
    result = ingestion.run(use_api=args.use_api)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Teams inserted: {result['inserted']}")
    print(f"Teams updated: {result['updated']}")
    print(f"Errors: {result['errors']}")
    print(f"Total teams: {result['total_teams']}")
    print(f"Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
    print(f"\nLog file: {logger.log_file}")
    
    return 0 if result['validation_passed'] else 1


if __name__ == "__main__":
    exit(main())
