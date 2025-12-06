#!/usr/bin/env python3
"""
Master Data Ingestion Pipeline for NFL Fantasy Prediction Engine

This script orchestrates ALL data ingestion in the correct order:
1. Schema creation (base + enhancement + pbp)
2. Teams and Players (from Sleeper API)
3. Games (from nflreadpy schedules)
4. Offensive stats (from Sleeper API)
5. Defensive stats (from CSVs)
6. NFLverse data (weekly stats, NGS, snaps, injuries, etc.)
7. Play-by-play (from nflreadpy)
8. Betting lines (from The Odds API)
9. Weather data (from Open-Meteo)

Run this weekly to keep your database up to date.

Author: NFL Fantasy Prediction Engine Team
Version: 3.0 - Fixed imports for correct directory structure
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# =============================================================================
# PATH SETUP - CRITICAL FOR IMPORTS
# =============================================================================
# Determine where this script lives
_script_dir = Path(__file__).parent.resolve()

# Add Phase 1 directory to path (this script should be in Phase 1/)
sys.path.insert(0, str(_script_dir))

# Add parent directory (FootballPredictor/) for requirements.txt lookup
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

# Add ingestion_scripts directory if it exists (for ingest_*.py imports)
_ingestion_dir = _script_dir / "ingestion_scripts"
if _ingestion_dir.exists():
    sys.path.insert(0, str(_ingestion_dir))

# Also check if utils/config directories exist (for package imports)
_utils_dir = _script_dir / "utils"
_config_dir = _script_dir / "config"

# Debug: Print paths if needed
# print(f"Script dir: {_script_dir}")
# print(f"Ingestion dir: {_ingestion_dir} (exists: {_ingestion_dir.exists()})")
# print(f"sys.path: {sys.path[:5]}")

# =============================================================================
# IMPORTS - Try multiple import strategies
# =============================================================================

# Try to import database utilities
try:
    from utils.database import get_db
    from utils.logger import get_ingestion_logger
except ImportError:
    try:
        from database import get_db
        from logger import get_ingestion_logger
    except ImportError as e:
        print(f"ERROR: Cannot import database utilities: {e}")
        print(f"Looking in: {_script_dir}")
        print("Make sure database.py and logger.py are in utils/ or Phase 1/ directory")
        sys.exit(1)

# Try to import settings
try:
    from config.settings import CURRENT_SEASON, ODDS_API_KEY
except ImportError:
    try:
        from settings import CURRENT_SEASON, ODDS_API_KEY
    except ImportError as e:
        print(f"ERROR: Cannot import settings: {e}")
        print("Make sure settings.py is in config/ or Phase 1/ directory")
        sys.exit(1)

logger = get_ingestion_logger("master_pipeline")


def check_and_install_requirements():
    """Check if requirements are installed, install if not."""
    # Check multiple possible locations for requirements.txt
    possible_paths = [
        _script_dir / "requirements.txt",
        _project_root / "requirements.txt",
        _script_dir.parent / "requirements.txt",
    ]
    
    requirements_file = None
    for path in possible_paths:
        if path.exists():
            requirements_file = path
            break
    
    if requirements_file:
        print(f"Checking requirements from {requirements_file}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("Requirements OK")
        except subprocess.CalledProcessError:
            print("Warning: Could not install requirements automatically")
    else:
        print("No requirements.txt found (checked multiple locations)")


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(num: str, text: str):
    """Print step header."""
    print(f"\n[Step {num}] {text}")
    print("-" * 50)


def import_module(module_name: str, class_name: str = None):
    """
    Try to import a module from multiple locations.
    Returns the module or class, or None if not found.
    """
    # Try different import strategies
    import_attempts = [
        module_name,  # Direct import (if in sys.path)
        f"ingestion_scripts.{module_name}",  # Package import
    ]
    
    for attempt in import_attempts:
        try:
            module = __import__(attempt, fromlist=[class_name] if class_name else [])
            if class_name:
                return getattr(module, class_name)
            return module
        except (ImportError, AttributeError):
            continue
    
    return None


def run_full_pipeline(
    seasons: list = None,
    skip_base: bool = False,
    skip_nflverse: bool = False,
    skip_betting: bool = False,
    skip_weather: bool = False,
    skip_pbp: bool = False,
    pbp_only: bool = False,
    current_week: int = None,
    auto_install: bool = True,
):
    """
    Run the complete data ingestion pipeline.
    """
    if seasons is None:
        seasons = [CURRENT_SEASON - 2, CURRENT_SEASON - 1, CURRENT_SEASON]
    
    # Auto-install requirements
    if auto_install:
        check_and_install_requirements()
    
    print_header("NFL Fantasy Prediction Engine - Master Pipeline")
    print(f"Seasons: {seasons}")
    print(f"Current Season: {CURRENT_SEASON}")
    print(f"Odds API Key: {'Set' if ODDS_API_KEY else 'NOT SET'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    db = get_db()
    results = {}
    
    # Handle PBP-only mode
    if pbp_only:
        print_step("PBP", "Play-by-Play Only Mode")
        PlayByPlayIngestion = import_module("ingest_pbp", "PlayByPlayIngestion")
        if PlayByPlayIngestion:
            try:
                pbp = PlayByPlayIngestion(db)
                pbp_results = pbp.ingest_seasons(seasons)
                results['pbp'] = pbp_results.get('total_plays', 0)
                print(f"PBP: {results['pbp']} plays")
            except Exception as e:
                print(f"PBP error: {e}")
                results['pbp'] = 0
        else:
            print("PBP import error: Could not find ingest_pbp module")
            results['pbp'] = 0
        
        print_header("Pipeline Complete!")
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {db.db_path}")
        return results
    
    # -------------------------------------------------------------------------
    # Step 1: Base Schema
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(1, "Creating/Verifying Base Schema")
        try:
            # Try both import strategies
            try:
                from create_schema import create_all_tables, verify_schema
            except ImportError:
                from ingestion_scripts.create_schema import create_all_tables, verify_schema
            
            create_all_tables(db)
            verify = verify_schema(db)
            results['base_schema'] = verify['all_tables_exist']
            print(f"Base schema: {'OK' if verify['all_tables_exist'] else 'ISSUES'}")
        except Exception as e:
            print(f"Base schema error: {e}")
            results['base_schema'] = False
    
    # -------------------------------------------------------------------------
    # Step 2: Enhancement Schema
    # -------------------------------------------------------------------------
    print_step(2, "Creating Enhancement Tables")
    try:
        try:
            from schema_enhancement import create_enhancement_tables, verify_enhancement_tables
        except ImportError:
            from ingestion_scripts.schema_enhancement import create_enhancement_tables, verify_enhancement_tables
        
        create_enhancement_tables(db)
        verify = verify_enhancement_tables(db)
        results['enhancement_schema'] = verify['all_exist']
        print(f"Enhancement schema: {'OK' if verify['all_exist'] else 'ISSUES'}")
    except Exception as e:
        print(f"Enhancement schema error: {e}")
        results['enhancement_schema'] = False
    
    # -------------------------------------------------------------------------
    # Step 2b: Play-by-Play Schema
    # -------------------------------------------------------------------------
    print_step("2b", "Creating Play-by-Play Table")
    try:
        try:
            from schema_pbp import create_pbp_table
        except ImportError:
            from ingestion_scripts.schema_pbp import create_pbp_table
        
        create_pbp_table(db)
        results['pbp_schema'] = True
        print("Play-by-play schema: OK")
    except Exception as e:
        print(f"PBP schema error: {e}")
        results['pbp_schema'] = False
    
    # -------------------------------------------------------------------------
    # Step 3: Teams (Sleeper API)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(3, "Ingesting Teams from Sleeper API")
        TeamsIngestion = import_module("ingest_teams", "TeamsIngestion")
        if TeamsIngestion:
            try:
                ingestion = TeamsIngestion(db)
                result = ingestion.run()
                count = result.get('total_teams', 0)
                results['teams'] = count
                print(f"Teams: {count} ingested")
            except Exception as e:
                print(f"Teams error: {e}")
                results['teams'] = 0
        else:
            print("Teams error: Could not import TeamsIngestion")
            results['teams'] = 0
    
    # -------------------------------------------------------------------------
    # Step 4: Players (Sleeper API)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(4, "Ingesting Players from Sleeper API")
        PlayersIngestion = import_module("ingest_players", "PlayersIngestion")
        if PlayersIngestion:
            try:
                ingestion = PlayersIngestion(db)
                result = ingestion.run()
                off_count = result.get('validation', {}).get('offensive_count', 0)
                def_count = result.get('validation', {}).get('defensive_count', 0)
                count = off_count + def_count
                results['players'] = count
                print(f"Players: {count} ingested ({off_count} offensive, {def_count} defensive)")
            except Exception as e:
                print(f"Players error: {e}")
                results['players'] = 0
        else:
            print("Players error: Could not import PlayersIngestion")
            results['players'] = 0
    
    # -------------------------------------------------------------------------
    # Step 5: Games (nflreadpy schedules)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(5, "Ingesting Games from nflreadpy schedules")
        GamesIngestion = import_module("ingest_games", "GamesIngestion")
        if GamesIngestion:
            try:
                ingestion = GamesIngestion(db)
                result = ingestion.run(seasons=seasons)
                count = result.get('inserted', 0) + result.get('updated', 0)
                results['games'] = count
                print(f"Games: {count} ingested")
            except Exception as e:
                print(f"Games error: {e}")
                results['games'] = 0
        else:
            print("Games error: Could not import GamesIngestion")
            results['games'] = 0
    
    # -------------------------------------------------------------------------
    # Step 6: Offensive Stats (Sleeper API)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(6, "Ingesting Offensive Stats from Sleeper API")
        OffensiveStatsIngestion = import_module("ingest_stats_offense", "OffensiveStatsIngestion")
        if OffensiveStatsIngestion:
            try:
                ingestion = OffensiveStatsIngestion(db)
                result = ingestion.run(seasons=seasons)
                count = result.get('total_stats', 0)
                results['offensive_stats'] = count
                print(f"Offensive stats: {count} ingested")
            except Exception as e:
                print(f"Offensive stats error: {e}")
                results['offensive_stats'] = 0
        else:
            print("Offensive stats error: Could not import OffensiveStatsIngestion")
            results['offensive_stats'] = 0
    
    # -------------------------------------------------------------------------
    # Step 7: Defensive Stats (CSVs)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(7, "Loading Defensive Stats from CSVs")
        load_func = import_module("load_season_defense_stats", "load_all_defense_stats")
        if load_func:
            try:
                count = load_func(db)
                results['defensive_stats'] = count
                print(f"Defensive stats: {count} loaded")
            except Exception as e:
                print(f"Defensive stats error: {e}")
                results['defensive_stats'] = 0
        else:
            print("Defensive stats error: Could not import load_all_defense_stats")
            results['defensive_stats'] = 0
    
    # -------------------------------------------------------------------------
    # Step 8: NFLverse Data (nflreadpy)
    # -------------------------------------------------------------------------
    if not skip_nflverse:
        print_step(8, "Ingesting NFLverse Data (nflreadpy - Weekly Stats, NGS, etc.)")
        NFLVerseIngestion = import_module("ingest_nflverse", "NFLVerseIngestion")
        if NFLVerseIngestion:
            try:
                nfl_ingestion = NFLVerseIngestion(db)
                nfl_results = nfl_ingestion.run_full_ingestion(seasons)
                results['nflverse'] = nfl_results.get('total_rows', 0)
                print(f"NFLverse: {results['nflverse']} rows")
            except Exception as e:
                print(f"NFLverse error: {e}")
                import traceback
                traceback.print_exc()
                results['nflverse'] = 0
        else:
            print("NFLverse import error: Could not import NFLVerseIngestion")
            print("Install with: pip install nflreadpy pandas")
            results['nflverse'] = 0
    
    # -------------------------------------------------------------------------
    # Step 8b: Play-by-Play Data (nflreadpy) - OPTIONAL
    # -------------------------------------------------------------------------
    if not skip_pbp:
        print_step("8b", "Ingesting Play-by-Play Data (nflreadpy)")
        PlayByPlayIngestion = import_module("ingest_pbp", "PlayByPlayIngestion")
        if PlayByPlayIngestion:
            try:
                pbp = PlayByPlayIngestion(db)
                pbp_results = pbp.ingest_seasons(seasons)
                results['pbp'] = pbp_results.get('total_plays', 0)
                print(f"PBP: {results['pbp']} plays")
            except Exception as e:
                print(f"PBP error: {e}")
                results['pbp'] = 0
        else:
            print("PBP import error: Could not import PlayByPlayIngestion")
            results['pbp'] = 0
    
    # -------------------------------------------------------------------------
    # Step 9: Betting Lines (The Odds API)
    # -------------------------------------------------------------------------
    if not skip_betting:
        print_step(9, "Fetching Betting Lines from The Odds API")
        if ODDS_API_KEY:
            BettingLinesIngestion = import_module("ingest_betting", "BettingLinesIngestion")
            if BettingLinesIngestion:
                try:
                    betting = BettingLinesIngestion(db)
                    bet_results = betting.weekly_pull(include_props=True)
                    total_lines = (
                        bet_results.get('featured', {}).get('lines', 0) +
                        bet_results.get('props', {}).get('lines', 0)
                    )
                    results['betting'] = total_lines
                    print(f"Betting lines: {total_lines}")
                except Exception as e:
                    print(f"Betting error: {e}")
                    results['betting'] = 0
            else:
                print("Betting error: Could not import BettingLinesIngestion")
                results['betting'] = 0
        else:
            print("ODDS_API_KEY not set - skipping betting lines")
            print("Set with: set ODDS_API_KEY=your_key_here")
            results['betting'] = 0
    
    # -------------------------------------------------------------------------
    # Step 10: Weather Data (Open-Meteo)
    # -------------------------------------------------------------------------
    if not skip_weather:
        print_step(10, "Fetching Weather Data from Open-Meteo")
        WeatherIngestion = import_module("ingest_weather", "WeatherIngestion")
        if WeatherIngestion:
            try:
                weather = WeatherIngestion(db)
                if current_week:
                    weather_results = weather.ingest_for_week(CURRENT_SEASON, current_week)
                else:
                    weather_results = weather.ingest_current_week()
                results['weather'] = weather_results.get('games_processed', 0)
                print(f"Weather: {results['weather']} games")
            except Exception as e:
                print(f"Weather error: {e}")
                results['weather'] = 0
        else:
            print("Weather error: Could not import WeatherIngestion")
            results['weather'] = 0
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("Pipeline Complete!")
    
    print("\nResults Summary:")
    for key, value in results.items():
        if isinstance(value, bool):
            status = "OK" if value else "FAILED"
        elif isinstance(value, int):
            status = "OK" if value > 0 else "FAILED"
        else:
            status = "OK" if value else "FAILED"
        print(f"  {status:8} {key}: {value}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {db.db_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="NFL Fantasy Prediction Engine - Master Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_master_pipeline.py                    # Full pipeline (skips PBP by default)
  python run_master_pipeline.py --seasons 2024 2025  # Specific seasons
  python run_master_pipeline.py --skip-base        # Skip Sleeper ingestion
  python run_master_pipeline.py --include-pbp      # Include play-by-play (large!)
  python run_master_pipeline.py --pbp-only         # Only play-by-play data
  python run_master_pipeline.py --nflverse-only    # Only NFLverse data
  python run_master_pipeline.py --betting-only     # Only betting lines
  python run_master_pipeline.py --skip-betting     # Skip betting lines
  python run_master_pipeline.py --no-auto-install  # Skip requirements check
        """
    )
    
    parser.add_argument('--seasons', type=int, nargs='+', help='Seasons to ingest')
    parser.add_argument('--week', type=int, help='Current week for weather')
    parser.add_argument('--skip-base', action='store_true', help='Skip base Sleeper ingestion')
    parser.add_argument('--skip-nflverse', action='store_true', help='Skip NFLverse data')
    parser.add_argument('--skip-betting', action='store_true', help='Skip betting lines')
    parser.add_argument('--skip-weather', action='store_true', help='Skip weather data')
    parser.add_argument('--skip-pbp', action='store_true', default=True, help='Skip play-by-play data (default: True)')
    parser.add_argument('--include-pbp', action='store_true', help='Include play-by-play data')
    parser.add_argument('--pbp-only', action='store_true', help='Only ingest play-by-play')
    parser.add_argument('--nflverse-only', action='store_true', help='Only NFLverse data')
    parser.add_argument('--betting-only', action='store_true', help='Only betting lines')
    parser.add_argument('--no-auto-install', action='store_true', help='Skip auto-installing requirements')
    
    args = parser.parse_args()
    
    # Handle convenience flags
    skip_base = args.skip_base
    skip_nflverse = args.skip_nflverse
    skip_betting = args.skip_betting
    skip_weather = args.skip_weather
    skip_pbp = not args.include_pbp  # PBP skipped by default unless --include-pbp
    pbp_only = args.pbp_only
    
    if args.nflverse_only:
        skip_base = True
        skip_betting = True
        skip_weather = True
        skip_pbp = True
    
    if args.betting_only:
        skip_base = True
        skip_nflverse = True
        skip_weather = True
        skip_pbp = True
    
    results = run_full_pipeline(
        seasons=args.seasons,
        skip_base=skip_base,
        skip_nflverse=skip_nflverse,
        skip_betting=skip_betting,
        skip_weather=skip_weather,
        skip_pbp=skip_pbp,
        pbp_only=pbp_only,
        current_week=args.week,
        auto_install=not args.no_auto_install,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
