"""
Load Season Defense Stats from CSV Files

This script loads defender and team defense statistics from CSV files.
Designed to be run from the Phase 1 directory or called as a module.

Author: NFL Fantasy Prediction Engine Team
Version: 2.0
"""

from pathlib import Path
import sys

# Figure out where we are and add project root to sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from utils.database import get_db

# Try to import ingestion classes - they might be in ingestion_scripts/ or current dir
try:
    from ingestion_scripts.ingest_stats_defenders import DefenderStatsIngestion
    from ingestion_scripts.ingest_stats_defense import TeamDefenseStatsIngestion
except ImportError:
    # Fallback: try importing from current directory (if in ingestion_scripts/)
    from ingest_stats_defenders import DefenderStatsIngestion
    from ingest_stats_defense import TeamDefenseStatsIngestion


# <<< PASTE / EDIT YOUR CSV PATHS RIGHT HERE >>>
DEFENDER_SEASON_FILES = {
    2023: BASE_DIR / "2023 Player Stats.csv",
    2024: BASE_DIR / "2024 Player Stats.csv",
    2025: BASE_DIR / "2025 Player Stats.csv",
}

TEAM_SEASON_FILES = {
    2025: BASE_DIR / "2025 Team Stats.csv",
}
# <<< END OF "PASTE PATHS" AREA >>>


def load_all_defense_stats(db=None):
    """
    Load all defense stats from CSV files.
    Returns total row count.
    
    This function is called by run_master_pipeline.py.
    """
    if db is None:
        db = get_db()
    
    total_count = 0
    
    # Defender season coverage stats
    defender_ingestion = DefenderStatsIngestion(db)
    for season, csv_path in DEFENDER_SEASON_FILES.items():
        if csv_path.exists():
            print(f"Loading defender season stats for {season} from {csv_path}...")
            try:
                result = defender_ingestion.load_season_coverage_from_csv(csv_path, season)
                if isinstance(result, dict):
                    total_count += result.get('rows_inserted', 0)
                elif isinstance(result, int):
                    total_count += result
            except Exception as e:
                print(f"Error loading defender stats for {season}: {e}")
        else:
            print(f"Warning: CSV not found: {csv_path}")
    
    # Team defense season coverage style
    team_ingestion = TeamDefenseStatsIngestion(db)
    for season, csv_path in TEAM_SEASON_FILES.items():
        if csv_path.exists():
            print(f"Loading team defense season stats for {season} from {csv_path}...")
            try:
                result = team_ingestion.load_team_season_coverage_from_csv(csv_path, season)
                if isinstance(result, dict):
                    total_count += result.get('rows_inserted', 0)
                elif isinstance(result, int):
                    total_count += result
            except Exception as e:
                print(f"Error loading team defense stats for {season}: {e}")
        else:
            print(f"Warning: CSV not found: {csv_path}")
    
    return total_count


def main():
    """Main entry point when run as a script."""
    db = get_db()
    count = load_all_defense_stats(db)
    print(f"\nTotal defense stats loaded: {count}")


if __name__ == "__main__":
    main()
