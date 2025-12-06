"""
Phase 1 Pipeline Orchestrator for NFL Fantasy Prediction Engine

This script orchestrates the complete Phase 1 data ingestion pipeline as specified
in the Comprehensive Operational Plan v2 and Phase 1 v2 documentation.

Pipeline Steps (executed in strict order per Section 6):
1. Create/Initialize Database Schema
2. Ingest Teams
3. Ingest Players (Offensive and Defensive)
4. Ingest Game Schedules
5. Ingest Offensive Player Game Stats
6a. Ingest Team Defense Stats (aggregated)
6b. Ingest Individual Defender Stats (placeholder/from CSV)
7. Run Validations
8. Create Data Version Record

Key Features:
- Deterministic, reproducible execution
- Fail-fast on validation errors
- Comprehensive logging
- Data versioning support
- Idempotent (safe to re-run)

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    CURRENT_SEASON,
    ROLLING_WINDOW_SEASONS,
    DATABASE_PATH,
    generate_data_version
)
from utils.database import get_db, get_version_manager, DatabaseConnection
from utils.logger import get_ingestion_logger, setup_system_logger

# Import ingestion modules
from ingestion_scripts.create_schema import create_all_tables, verify_schema
from ingestion_scripts.ingest_teams import TeamsIngestion
from ingestion_scripts.ingest_players import PlayersIngestion
from ingestion_scripts.ingest_games import GamesIngestion
from ingestion_scripts.ingest_stats_offense import OffensiveStatsIngestion
from ingestion_scripts.ingest_stats_defense import TeamDefenseStatsIngestion
from ingestion_scripts.ingest_stats_defenders import DefenderStatsIngestion
from validation.validate_data import DataValidator, ValidationError

# Initialize logger
logger = get_ingestion_logger("pipeline")


class Phase1Pipeline:
    """
    Orchestrates the complete Phase 1 data ingestion pipeline.
    
    Executes all steps in the correct order with proper validation
    and error handling.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the pipeline.
        
        Args:
            db: Optional database connection.
        """
        self.db = db or get_db()
        self.version_manager = get_version_manager()
        
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "steps": {},
            "data_version": None,
            "success": False,
            "errors": []
        }
    
    def _record_step(self, step_name: str, result: Dict, success: bool):
        """Record the result of a pipeline step."""
        self.results["steps"][step_name] = {
            "success": success,
            "result": result,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        if not success:
            self.results["errors"].append(f"Step '{step_name}' failed")
    
    def step_1_create_schema(self, drop_existing: bool = False) -> bool:
        """
        Step 1: Create/Initialize Database Schema
        
        Args:
            drop_existing: If True, drop existing tables (WARNING: destroys data)
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 1: CREATE DATABASE SCHEMA")
        logger.info("=" * 60)
        
        try:
            create_all_tables(self.db, drop_existing=drop_existing)
            
            # Verify schema
            verification = verify_schema(self.db)
            
            if verification["all_tables_exist"]:
                logger.info("Schema creation successful")
                self._record_step("create_schema", verification, True)
                return True
            else:
                logger.error("Schema verification failed")
                self._record_step("create_schema", verification, False)
                return False
                
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            self._record_step("create_schema", {"error": str(e)}, False)
            return False
    
    def step_2_ingest_teams(self) -> bool:
        """
        Step 2: Ingest Teams
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 2: INGEST TEAMS")
        logger.info("=" * 60)
        
        try:
            ingestion = TeamsIngestion(self.db)
            result = ingestion.run()
            
            success = result.get("validation_passed", False)
            self._record_step("ingest_teams", result, success)
            
            if success:
                logger.info(f"Teams ingestion successful: {result['total_teams']} teams")
            else:
                logger.error("Teams ingestion failed validation")
            
            return success
            
        except Exception as e:
            logger.error(f"Teams ingestion failed: {e}")
            self._record_step("ingest_teams", {"error": str(e)}, False)
            return False
    
    def step_3_ingest_players(self) -> bool:
        """
        Step 3: Ingest Players (Offensive and Defensive)
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 3: INGEST PLAYERS")
        logger.info("=" * 60)
        
        try:
            ingestion = PlayersIngestion(self.db)
            result = ingestion.run()
            
            if "error" in result:
                self._record_step("ingest_players", result, False)
                return False
            
            success = result["validation"]["validation_passed"]
            self._record_step("ingest_players", result, success)
            
            if success:
                logger.info(
                    f"Players ingestion successful: "
                    f"{result['validation']['offensive_count']} offensive, "
                    f"{result['validation']['defensive_count']} defensive"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Players ingestion failed: {e}")
            self._record_step("ingest_players", {"error": str(e)}, False)
            return False
    
    def step_4_ingest_games(self, seasons: Optional[List[int]] = None) -> bool:
        """
        Step 4: Ingest Game Schedules
        
        Args:
            seasons: Optional list of seasons to ingest
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 4: INGEST GAMES")
        logger.info("=" * 60)
        
        try:
            ingestion = GamesIngestion(self.db)
            result = ingestion.run(seasons=seasons)
            
            success = result["validation"]["validation_passed"]
            self._record_step("ingest_games", result, success)
            
            if success:
                logger.info(f"Games ingestion successful: {result['inserted']} games inserted")
            
            return success
            
        except Exception as e:
            logger.error(f"Games ingestion failed: {e}")
            self._record_step("ingest_games", {"error": str(e)}, False)
            return False
    
    def step_5_ingest_offensive_stats(self, seasons: Optional[List[int]] = None) -> bool:
        """
        Step 5: Ingest Offensive Player Game Stats
        
        Args:
            seasons: Optional list of seasons to ingest
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 5: INGEST OFFENSIVE STATS")
        logger.info("=" * 60)
        
        try:
            ingestion = OffensiveStatsIngestion(self.db)
            result = ingestion.run(seasons=seasons)
            
            success = result["validation"]["validation_passed"]
            self._record_step("ingest_stats_offense", result, success)
            
            if success:
                logger.info(
                    f"Offensive stats ingestion successful: "
                    f"{result['inserted']} inserted, {result['updated']} updated"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Offensive stats ingestion failed: {e}")
            self._record_step("ingest_stats_offense", {"error": str(e)}, False)
            return False
    
    def step_6a_ingest_defense_stats(self, seasons: Optional[List[int]] = None) -> bool:
        """
        Step 6a: Ingest Team Defense Stats
        
        Args:
            seasons: Optional list of seasons to ingest
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 6a: INGEST TEAM DEFENSE STATS")
        logger.info("=" * 60)
        
        try:
            ingestion = TeamDefenseStatsIngestion(self.db)
            result = ingestion.run(seasons=seasons)
            
            success = result["validation"]["validation_passed"]
            self._record_step("ingest_stats_defense", result, success)
            
            if success:
                logger.info(
                    f"Team defense stats ingestion successful: "
                    f"{result['inserted']} inserted, {result['updated']} updated"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Team defense stats ingestion failed: {e}")
            self._record_step("ingest_stats_defense", {"error": str(e)}, False)
            return False
    
    def step_6b_ingest_defender_stats(self,
                                      seasons: Optional[List[int]] = None,
                                      create_placeholders: bool = True,
                                      csv_file: Optional[Path] = None) -> bool:
        """
        Step 6b: Ingest Individual Defender Stats

        Args:
            seasons: Optional list of seasons to ingest
            create_placeholders: If True, create placeholder records
            csv_file: Optional CSV file with defender data

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("STEP 6b: INGEST DEFENDER STATS")
        logger.info("=" * 60)

        try:
            ingestion = DefenderStatsIngestion(self.db)
            result = ingestion.run(
                seasons=seasons,
                csv_file=csv_file,
                create_placeholders=create_placeholders,
            )

            success = result["validation"]["validation_passed"]
            self._record_step("ingest_stats_defenders", result, success)

            logger.info(
                f"Defender stats ingestion complete: "
                f"{result['inserted']} inserted, {result['updated']} updated"
            )

            return success

        except Exception as e:
            logger.error(f"Defender stats ingestion failed: {e}")
            self._record_step("ingest_stats_defenders", {"error": str(e)}, False)
            return False
    
    def step_7_run_validations(self, fail_fast: bool = True) -> bool:
        """
        Step 7: Run All Validations
        
        Args:
            fail_fast: If True, stop on first critical error
        
        Returns:
            True if all validations pass
        """
        logger.info("=" * 60)
        logger.info("STEP 7: RUN VALIDATIONS")
        logger.info("=" * 60)
        
        try:
            validator = DataValidator(self.db)
            all_passed, results = validator.run_all_validations(fail_fast=fail_fast)
            summary = validator.get_summary()
            
            self._record_step("validation", summary, all_passed)
            
            if all_passed:
                logger.info(f"All validations passed: {summary['passed']}/{summary['total_checks']}")
            else:
                logger.error(f"Validation failures: {summary['failed']}/{summary['total_checks']}")
            
            return all_passed
            
        except ValidationError as e:
            logger.error(f"Critical validation failure: {e}")
            self._record_step("validation", {"error": str(e)}, False)
            return False
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self._record_step("validation", {"error": str(e)}, False)
            return False
    
    def step_8_create_data_version(self, season: int, week: int,
                                    notes: str = "") -> Optional[str]:
        """
        Step 8: Create Data Version Record
        
        Args:
            season: Season year
            week: Week number
            notes: Optional notes
        
        Returns:
            Version name or None if failed
        """
        logger.info("=" * 60)
        logger.info("STEP 8: CREATE DATA VERSION")
        logger.info("=" * 60)
        
        try:
            # Get row counts
            offensive_count = self.db.get_row_count("player_game_stats")
            defensive_count = self.db.get_row_count("team_defense_game_stats")
            defender_count = self.db.get_row_count("defender_game_stats")
            
            version_name = self.version_manager.create_version(
                season=season,
                week=week,
                offensive_row_count=offensive_count,
                defensive_row_count=defensive_count,
                defender_row_count=defender_count,
                notes=notes
            )
            
            logger.log_data_version_created(version_name, notes)
            
            self._record_step("create_version", {
                "version": version_name,
                "offensive_count": offensive_count,
                "defensive_count": defensive_count,
                "defender_count": defender_count
            }, True)
            
            return version_name
            
        except Exception as e:
            logger.error(f"Failed to create data version: {e}")
            self._record_step("create_version", {"error": str(e)}, False)
            return None
    
    def run_full_pipeline(self,
                          drop_existing: bool = False,
                          seasons: Optional[List[int]] = None,
                          current_week: int = 1,
                          skip_defender_placeholders: bool = False,
                          fail_fast: bool = True) -> Dict:
        """
        Run the complete Phase 1 pipeline.
        
        Args:
            drop_existing: If True, drop existing tables
            seasons: Seasons to ingest (None = rolling window)
            current_week: Current week for data version
            skip_defender_placeholders: Skip creating placeholder defender records
            fail_fast: Stop on first critical validation error
        
        Returns:
            Pipeline results dictionary
        """
        logger.info("=" * 70)
        logger.info("PHASE 1 PIPELINE - FULL EXECUTION")
        logger.info("=" * 70)
        logger.info(f"Database: {DATABASE_PATH}")
        logger.info(f"Seasons: {seasons or 'rolling window'}")
        logger.info(f"Drop existing: {drop_existing}")
        logger.info("=" * 70)
        
        self.results["start_time"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        start = time.time()
        
        # Execute pipeline steps
        steps = [
            ("Step 1: Schema", lambda: self.step_1_create_schema(drop_existing)),
            ("Step 2: Teams", lambda: self.step_2_ingest_teams()),
            ("Step 3: Players", lambda: self.step_3_ingest_players()),
            ("Step 4: Games", lambda: self.step_4_ingest_games(seasons)),
            ("Step 5: Offensive Stats", lambda: self.step_5_ingest_offensive_stats(seasons)),
            ("Step 6a: Defense Stats", lambda: self.step_6a_ingest_defense_stats(seasons)),
            ("Step 6b: Defender Stats", lambda: self.step_6b_ingest_defender_stats(
                seasons,
                create_placeholders=not skip_defender_placeholders,
            )),
            ("Step 7: Validations", lambda: self.step_7_run_validations(fail_fast)),
        ]
        
        all_success = True
        
        for step_name, step_func in steps:
            logger.info(f"\n>>> Executing {step_name}...")
            
            try:
                success = step_func()
                if not success:
                    all_success = False
                    if fail_fast:
                        logger.error(f"Pipeline stopped at {step_name}")
                        break
            except Exception as e:
                logger.error(f"Exception in {step_name}: {e}")
                all_success = False
                if fail_fast:
                    break
        
        # Create data version if pipeline succeeded
        if all_success:
            current_season = seasons[-1] if seasons else CURRENT_SEASON
            version = self.step_8_create_data_version(
                current_season, current_week,
                notes=f"Full pipeline execution"
            )
            self.results["data_version"] = version
        
        # Record final results
        self.results["end_time"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.results["duration_seconds"] = round(time.time() - start, 2)
        self.results["success"] = all_success
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1 PIPELINE - EXECUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Success: {all_success}")
        logger.info(f"Duration: {self.results['duration_seconds']}s")
        logger.info(f"Data Version: {self.results.get('data_version', 'N/A')}")
        
        if self.results["errors"]:
            logger.info(f"Errors: {self.results['errors']}")
        
        logger.info("=" * 70)
        
        return self.results
    
    def run_incremental_update(self, season: int, week: int) -> Dict:
        """
        Run an incremental update for a specific week.
        
        This is used for weekly updates after the initial full ingestion.
        
        Args:
            season: Season year
            week: Week number to update
        
        Returns:
            Update results
        """
        logger.info("=" * 70)
        logger.info(f"PHASE 1 PIPELINE - INCREMENTAL UPDATE ({season} Week {week})")
        logger.info("=" * 70)
        
        self.results["start_time"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        start = time.time()
        
        # For incremental updates, we only update stats for the specified week
        # Schema and teams should already exist
        
        try:
            # Step 5: Update offensive stats for this week
            logger.info("Updating offensive stats...")
            offense = OffensiveStatsIngestion(self.db)
            offense_result = offense.run(seasons=[season], weeks=[week])
            self._record_step("ingest_stats_offense", offense_result, 
                            offense_result["validation"]["validation_passed"])
            
            # Step 6a: Update team defense stats
            logger.info("Updating team defense stats...")
            defense = TeamDefenseStatsIngestion(self.db)
            defense.ingest_week(season, week)
            
            # Step 7: Validate
            logger.info("Running validations...")
            validator = DataValidator(self.db)
            validator.validate_weekly_completeness(season, week)
            
            # Step 8: Create version
            version = self.step_8_create_data_version(
                season, week,
                notes=f"Incremental update for {season} week {week}"
            )
            self.results["data_version"] = version
            self.results["success"] = True
            
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            self.results["success"] = False
            self.results["errors"].append(str(e))
        
        self.results["end_time"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.results["duration_seconds"] = round(time.time() - start, 2)
        
        return self.results


def main():
    """Main entry point for the Phase 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings (rolling window)
  python run_pipeline.py

  # Full pipeline for specific seasons
  python run_pipeline.py --season 2023 2024

  # Reset database and run fresh
  python run_pipeline.py --drop-existing --confirm-drop

  # Incremental update for a specific week
  python run_pipeline.py --incremental --season 2024 --week 10

  # Validate only
  python run_pipeline.py --validate-only
        """
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
        default=1,
        help="Current week number (for versioning)"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating"
    )
    parser.add_argument(
        "--confirm-drop",
        action="store_true",
        help="Confirm dropping existing tables"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental update for specified season/week"
    )
    parser.add_argument(
        "--skip-defender-placeholders",
        action="store_true",
        help="Skip creating placeholder defender records"
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue on validation errors"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validations"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("NFL Fantasy Prediction Engine")
    print("Phase 1: Data Ingestion & Database Setup")
    print(f"{'='*70}\n")
    
    pipeline = Phase1Pipeline()
    
    # Validate only mode
    if args.validate_only:
        print("Running validations only...")
        success = pipeline.step_7_run_validations(fail_fast=not args.no_fail_fast)
        return 0 if success else 1
    
    # Check for drop confirmation
    if args.drop_existing and not args.confirm_drop:
        print("ERROR: --drop-existing requires --confirm-drop")
        print("This will DELETE ALL DATA in the database!")
        return 1
    
    # Incremental update mode
    if args.incremental:
        if not args.season or len(args.season) != 1:
            print("ERROR: --incremental requires exactly one --season")
            return 1
        
        results = pipeline.run_incremental_update(args.season[0], args.week)
        
    # Full pipeline mode
    else:
        results = pipeline.run_full_pipeline(
            drop_existing=args.drop_existing,
            seasons=args.season,
            current_week=args.week,
            skip_defender_placeholders=args.skip_defender_placeholders,
            fail_fast=not args.no_fail_fast
        )
    
    # Print final summary
    print(f"\n{'='*70}")
    print("PIPELINE RESULTS")
    print(f"{'='*70}")
    print(f"Success: {results['success']}")
    print(f"Duration: {results['duration_seconds']}s")
    print(f"Data Version: {results.get('data_version', 'N/A')}")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nLog file: {logger.log_file}")
    print(f"{'='*70}\n")
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit(main())
