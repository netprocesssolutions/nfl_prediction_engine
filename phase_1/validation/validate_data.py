"""
Validation Framework for NFL Fantasy Prediction Engine - Phase 1

This module implements all validation rules specified in Phase 1 v2 Section 7.
All validations are STRICT - if any validation fails, ingestion must STOP immediately.

Validation Categories:
- Entity Integrity (7.1): Foreign key validation
- Duplicate Protection (7.2): Unique constraint validation
- Null and Range Validation (7.3): Value bounds checking
- Coverage Probability Validation (7.4): man% + zone% Ã¢â€°Ë† 1
- Weekly Completeness (7.5): All games have all required data
- Anti-Leakage Validation (7.6): Required time indexing fields

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import VALIDATION_CONFIG

# Initialize logger
logger = get_ingestion_logger("validation")


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        status = "Ã¢Å“â€œ PASS" if self.passed else "Ã¢Å“â€” FAIL"
        return f"[{self.severity.value}] {status}: {self.check_name} - {self.message}"


class ValidationError(Exception):
    """
    Exception raised when a critical validation fails.
    
    As per Phase 1 v2: "If any validation fails, ingestion must STOP immediately."
    """
    
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(str(result))


class DataValidator:
    """
    Comprehensive data validator implementing all Phase 1 v2 validation rules.
    
    Usage:
        validator = DataValidator(db)
        
        # Run all validations
        results = validator.run_all_validations()
        
        # Or run specific validations
        result = validator.validate_entity_integrity()
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the validator.
        
        Args:
            db: Database connection. If None, uses default connection.
        """
        self.db = db or get_db()
        self.results: List[ValidationResult] = []
    
    def _add_result(self, result: ValidationResult):
        """Add a validation result and log it."""
        self.results.append(result)
        
        if result.passed:
            logger.info(str(result), event="validation_pass")
        elif result.severity == ValidationSeverity.WARNING:
            logger.warning(str(result), event="validation_warning")
        else:
            logger.error(str(result), event="validation_fail")
    
    def _check_foreign_key(self, 
                           source_table: str, 
                           source_column: str,
                           target_table: str,
                           target_column: str) -> ValidationResult:
        """
        Check that all values in source column exist in target table.
        
        Args:
            source_table: Table containing foreign key
            source_column: Column with FK values
            target_table: Referenced table
            target_column: Referenced column (usually PK)
        
        Returns:
            ValidationResult
        """
        query = f"""
            SELECT COUNT(*) as orphan_count
            FROM {source_table} s
            WHERE s.{source_column} IS NOT NULL
            AND s.{source_column} NOT IN (
                SELECT {target_column} FROM {target_table}
            )
        """
        
        result = self.db.fetch_one(query)
        orphan_count = result['orphan_count'] if result else 0
        
        return ValidationResult(
            check_name=f"FK: {source_table}.{source_column} -> {target_table}.{target_column}",
            passed=orphan_count == 0,
            severity=ValidationSeverity.ERROR if orphan_count > 0 else ValidationSeverity.INFO,
            message=f"{orphan_count} orphan records found" if orphan_count > 0 else "All FKs valid",
            details={"orphan_count": orphan_count}
        )
    
    # =========================================================================
    # 7.1 Entity Integrity Validations
    # =========================================================================
    
    def validate_entity_integrity(self) -> List[ValidationResult]:
        """
        Validate all foreign key relationships as per Section 7.1.
        
        Checks:
        - Every player_id in player_game_stats must exist in players
        - Every defender_id in defender_game_stats must exist in defenders
        - Every team_id must exist in teams
        - Every game_id must exist in games
        """
        logger.info("Validating entity integrity (Section 7.1)...")
        results = []
        
        fk_checks = [
            # player_game_stats foreign keys
            ("player_game_stats", "player_id", "players", "player_id"),
            ("player_game_stats", "team_id", "teams", "team_id"),
            ("player_game_stats", "opponent_team_id", "teams", "team_id"),
            ("player_game_stats", "game_id", "games", "game_id"),
            
            # defender_game_stats foreign keys
            ("defender_game_stats", "defender_id", "defenders", "defender_id"),
            ("defender_game_stats", "game_id", "games", "game_id"),
            
            # team_defense_game_stats foreign keys
            ("team_defense_game_stats", "team_id", "teams", "team_id"),
            ("team_defense_game_stats", "opponent_team_id", "teams", "team_id"),
            ("team_defense_game_stats", "game_id", "games", "game_id"),
            
            # games foreign keys
            ("games", "home_team_id", "teams", "team_id"),
            ("games", "away_team_id", "teams", "team_id"),
            
            # players foreign keys
            ("players", "team_id", "teams", "team_id"),
            
            # defenders foreign keys
            ("defenders", "team_id", "teams", "team_id"),
        ]
        
        for source_table, source_col, target_table, target_col in fk_checks:
            # Only check if both tables exist and have data
            if not self.db.table_exists(source_table):
                continue
            if self.db.get_row_count(source_table) == 0:
                continue
            if not self.db.table_exists(target_table):
                continue
                
            result = self._check_foreign_key(source_table, source_col, target_table, target_col)
            results.append(result)
            self._add_result(result)
        
        return results
    
    # =========================================================================
    # 7.2 Duplicate Protection Validations
    # =========================================================================
    
    def validate_no_duplicates(self) -> List[ValidationResult]:
        """
        Validate no duplicate records exist as per Section 7.2.
        
        Checks:
        - (player_id, game_id) must be unique in player_game_stats
        - (defender_id, game_id) must be unique in defender_game_stats
        - (team_id, game_id) must be unique in team_defense_game_stats
        """
        logger.info("Validating duplicate protection (Section 7.2)...")
        results = []
        
        duplicate_checks = [
            ("player_game_stats", ["player_id", "game_id"]),
            ("defender_game_stats", ["defender_id", "game_id"]),
            ("team_defense_game_stats", ["team_id", "game_id"]),
        ]
        
        for table, key_cols in duplicate_checks:
            if not self.db.table_exists(table):
                continue
            if self.db.get_row_count(table) == 0:
                continue
            
            cols = ", ".join(key_cols)
            query = f"""
                SELECT {cols}, COUNT(*) as count
                FROM {table}
                GROUP BY {cols}
                HAVING COUNT(*) > 1
            """
            
            duplicates = self.db.fetch_all(query)
            dup_count = len(duplicates)
            
            result = ValidationResult(
                check_name=f"Duplicates: {table} ({cols})",
                passed=dup_count == 0,
                severity=ValidationSeverity.ERROR if dup_count > 0 else ValidationSeverity.INFO,
                message=f"{dup_count} duplicate key combinations found" if dup_count > 0 else "No duplicates",
                details={"duplicate_count": dup_count}
            )
            results.append(result)
            self._add_result(result)
        
        return results
    
    # =========================================================================
    # 7.3 Null and Range Validations
    # =========================================================================
    
    def validate_ranges_and_nulls(self) -> List[ValidationResult]:
        """
        Validate value ranges and null constraints as per Section 7.3.
        
        Checks:
        - Snap counts must be >= 0
        - Routes >= 0
        - Targets allowed >= 0
        - Alignment percentages Ã¢Ë†Ë† [0,1]
        """
        logger.info("Validating ranges and nulls (Section 7.3)...")
        results = []
        
        # Range validations
        range_checks = [
            ("player_game_stats", "snaps", 0, None),
            ("player_game_stats", "routes", 0, None),
            ("player_game_stats", "carries", 0, None),
            ("player_game_stats", "targets", 0, None),
            ("player_game_stats", "receptions", 0, None),
            ("defender_game_stats", "snaps", 0, None),
            ("defender_game_stats", "coverage_snaps", 0, None),
            ("defender_game_stats", "targets_allowed", 0, None),
            ("defender_game_stats", "alignment_boundary_pct", 0, 1),
            ("defender_game_stats", "alignment_slot_pct", 0, 1),
            ("defender_game_stats", "alignment_deep_pct", 0, 1),
            ("defender_game_stats", "alignment_box_pct", 0, 1),
            ("defender_game_stats", "man_coverage_pct", 0, 1),
            ("defender_game_stats", "zone_coverage_pct", 0, 1),
        ]
        
        for table, column, min_val, max_val in range_checks:
            if not self.db.table_exists(table):
                continue
            if self.db.get_row_count(table) == 0:
                continue
            
            # Check for out-of-range values
            conditions = []
            if min_val is not None:
                conditions.append(f"{column} < {min_val}")
            if max_val is not None:
                conditions.append(f"{column} > {max_val}")
            
            if conditions:
                where_clause = " OR ".join(conditions)
                query = f"""
                    SELECT COUNT(*) as violation_count
                    FROM {table}
                    WHERE {column} IS NOT NULL AND ({where_clause})
                """
                result_row = self.db.fetch_one(query)
                violation_count = result_row['violation_count'] if result_row else 0
                
                range_str = f"[{min_val}, {max_val}]" if max_val else f">= {min_val}"
                result = ValidationResult(
                    check_name=f"Range: {table}.{column} {range_str}",
                    passed=violation_count == 0,
                    severity=ValidationSeverity.ERROR if violation_count > 0 else ValidationSeverity.INFO,
                    message=f"{violation_count} out-of-range values" if violation_count > 0 else "All values in range",
                    details={"violation_count": violation_count}
                )
                results.append(result)
                self._add_result(result)
        
        return results
    
    # =========================================================================
    # 7.4 Coverage Probability Validation
    # =========================================================================
    
    def validate_coverage_probabilities(self) -> List[ValidationResult]:
        """
        Validate coverage probability consistency as per Section 7.4.
        
        For each defender, man_coverage_pct + zone_coverage_pct should Ã¢â€°Ë† 1.
        """
        logger.info("Validating coverage probabilities (Section 7.4)...")
        results = []
        
        if not self.db.table_exists("defender_game_stats"):
            return results
        if self.db.get_row_count("defender_game_stats") == 0:
            return results
        
        tolerance = VALIDATION_CONFIG["coverage_sum_tolerance"]
        
        query = f"""
            SELECT COUNT(*) as violation_count
            FROM defender_game_stats
            WHERE man_coverage_pct IS NOT NULL 
            AND zone_coverage_pct IS NOT NULL
            AND ABS((man_coverage_pct + zone_coverage_pct) - 1.0) > {tolerance}
        """
        
        result_row = self.db.fetch_one(query)
        violation_count = result_row['violation_count'] if result_row else 0
        
        result = ValidationResult(
            check_name="Coverage probability sum Ã¢â€°Ë† 1",
            passed=violation_count == 0,
            severity=ValidationSeverity.WARNING if violation_count > 0 else ValidationSeverity.INFO,
            message=f"{violation_count} defenders with invalid coverage sum (tolerance: {tolerance})" if violation_count > 0 else "All coverage probabilities valid",
            details={"violation_count": violation_count, "tolerance": tolerance}
        )
        results.append(result)
        self._add_result(result)
        
        return results
    
    # =========================================================================
    # 7.5 Weekly Completeness Validation
    # =========================================================================
    
    def validate_weekly_completeness(self, season: int, week: int) -> List[ValidationResult]:
        """
        Validate that a specific week has complete data as per Section 7.5.
        
        Every scheduled game must contain:
        - offensive stats
        - team defense stats
        - defender stats (where available)
        
        Args:
            season: Season year
            week: Week number
        """
        logger.info(f"Validating weekly completeness for {season} week {week} (Section 7.5)...")
        results = []
        
        # Count games for this week
        games_query = """
            SELECT COUNT(*) as game_count
            FROM games
            WHERE season = ? AND week = ?
        """
        games_result = self.db.fetch_one(games_query, (season, week))
        game_count = games_result['game_count'] if games_result else 0
        
        # Check offensive stats completeness
        offensive_query = """
            SELECT COUNT(DISTINCT game_id) as games_with_stats
            FROM player_game_stats
            WHERE season = ? AND week = ?
        """
        off_result = self.db.fetch_one(offensive_query, (season, week))
        games_with_offensive = off_result['games_with_stats'] if off_result else 0
        
        result = ValidationResult(
            check_name=f"Offensive stats completeness ({season} week {week})",
            passed=games_with_offensive >= game_count,
            severity=ValidationSeverity.ERROR if games_with_offensive < game_count else ValidationSeverity.INFO,
            message=f"{games_with_offensive}/{game_count} games have offensive stats",
            details={"expected": game_count, "actual": games_with_offensive}
        )
        results.append(result)
        self._add_result(result)
        
        # Check team defense stats completeness
        defense_query = """
            SELECT COUNT(DISTINCT game_id) as games_with_stats
            FROM team_defense_game_stats
            WHERE season = ? AND week = ?
        """
        def_result = self.db.fetch_one(defense_query, (season, week))
        games_with_defense = def_result['games_with_stats'] if def_result else 0
        
        result = ValidationResult(
            check_name=f"Defense stats completeness ({season} week {week})",
            passed=games_with_defense >= game_count,
            severity=ValidationSeverity.WARNING if games_with_defense < game_count else ValidationSeverity.INFO,
            message=f"{games_with_defense}/{game_count} games have defense stats",
            details={"expected": game_count, "actual": games_with_defense}
        )
        results.append(result)
        self._add_result(result)
        
        return results
    
    # =========================================================================
    # 7.6 Anti-Leakage Validation
    # =========================================================================
    
    def validate_anti_leakage(self) -> List[ValidationResult]:
        """
        Validate anti-leakage fields are present as per Section 7.6.
        
        Every row must include:
        - season
        - week
        - game_id
        - insertion timestamp
        
        Rows missing indexing information must be rejected.
        """
        logger.info("Validating anti-leakage fields (Section 7.6)...")
        results = []
        
        tables_to_check = [
            ("player_game_stats", ["season", "week", "game_id", "ingested_at"]),
            ("team_defense_game_stats", ["season", "week", "game_id", "ingested_at"]),
            ("defender_game_stats", ["season", "week", "game_id", "ingested_at"]),
        ]
        
        for table, required_cols in tables_to_check:
            if not self.db.table_exists(table):
                continue
            if self.db.get_row_count(table) == 0:
                continue
            
            for col in required_cols:
                query = f"""
                    SELECT COUNT(*) as null_count
                    FROM {table}
                    WHERE {col} IS NULL
                """
                result_row = self.db.fetch_one(query)
                null_count = result_row['null_count'] if result_row else 0
                
                result = ValidationResult(
                    check_name=f"Anti-leakage: {table}.{col} not null",
                    passed=null_count == 0,
                    severity=ValidationSeverity.ERROR if null_count > 0 else ValidationSeverity.INFO,
                    message=f"{null_count} rows missing {col}" if null_count > 0 else f"All {col} values present",
                    details={"null_count": null_count}
                )
                results.append(result)
                self._add_result(result)
        
        return results
    
    # =========================================================================
    # Run All Validations
    # =========================================================================
    
    def run_all_validations(self, fail_fast: bool = True) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Args:
            fail_fast: If True, raise exception on first critical failure.
                      If False, continue and collect all results.
        
        Returns:
            Tuple of (all_passed, results_list)
        
        Raises:
            ValidationError: If fail_fast=True and a critical validation fails.
        """
        logger.info("Running all validations...")
        self.results = []
        all_passed = True
        
        validation_methods = [
            self.validate_entity_integrity,
            self.validate_no_duplicates,
            self.validate_ranges_and_nulls,
            self.validate_coverage_probabilities,
            self.validate_anti_leakage,
        ]
        
        for method in validation_methods:
            try:
                results = method()
                for result in results:
                    if not result.passed:
                        all_passed = False
                        if fail_fast and result.severity == ValidationSeverity.ERROR:
                            raise ValidationError(result)
            except ValidationError:
                raise
            except Exception as e:
                logger.error(f"Validation error in {method.__name__}: {e}")
                all_passed = False
        
        summary = "All validations passed" if all_passed else "Some validations failed"
        logger.info(f"Validation complete: {summary}", event="validation_complete")
        
        return all_passed, self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count
        
        by_severity = {}
        for result in self.results:
            if not result.passed:
                severity = result.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_checks": len(self.results),
            "passed": passed_count,
            "failed": failed_count,
            "failures_by_severity": by_severity,
            "all_passed": failed_count == 0
        }


# =============================================================================
# Convenience Functions for Pre-Insertion Validation
# =============================================================================

def validate_player_data(player_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate player data before insertion.
    
    Args:
        player_data: Dictionary with player fields
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required = ["player_id", "full_name", "position"]
    for field in required:
        if not player_data.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Position validation
    valid_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
    if player_data.get("position") and player_data["position"] not in valid_positions:
        errors.append(f"Invalid position: {player_data['position']}")
    
    return len(errors) == 0, errors


def validate_stat_row(stat_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a stats row before insertion.
    
    Args:
        stat_data: Dictionary with stat fields
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required anti-leakage fields
    required = ["player_id", "game_id", "season", "week"]
    for field in required:
        if stat_data.get(field) is None:
            errors.append(f"Missing anti-leakage field: {field}")
    
    # Non-negative validations
    non_negative = ["snaps", "routes", "carries", "targets", "receptions"]
    for field in non_negative:
        val = stat_data.get(field)
        if val is not None and val < 0:
            errors.append(f"Negative value for {field}: {val}")
    
    return len(errors) == 0, errors


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run validations from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate NFL database")
    parser.add_argument("--season", type=int, help="Season to validate")
    parser.add_argument("--week", type=int, help="Week to validate")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Data Validation")
    print(f"{'='*60}\n")
    
    db = get_db()
    validator = DataValidator(db)
    
    try:
        all_passed, results = validator.run_all_validations(fail_fast=args.fail_fast)
        
        # Run weekly completeness if season/week specified
        if args.season and args.week:
            validator.validate_weekly_completeness(args.season, args.week)
        
        summary = validator.get_summary()
        
        print(f"\n{'='*60}")
        print("Validation Summary")
        print(f"{'='*60}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failures_by_severity']:
            print("\nFailures by severity:")
            for severity, count in summary['failures_by_severity'].items():
                print(f"  {severity}: {count}")
        
        print(f"\nLog file: {logger.log_file}")
        
        return 0 if all_passed else 1
        
    except ValidationError as e:
        print(f"\nÃ¢Å“â€” CRITICAL VALIDATION FAILURE: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
