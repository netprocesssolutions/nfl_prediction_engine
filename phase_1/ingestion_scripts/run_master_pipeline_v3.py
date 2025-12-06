#!/usr/bin/env python3
"""
Master Data Ingestion Pipeline for NFL Fantasy Prediction Engine

This script orchestrates ALL data ingestion in the correct order:
1. Schema creation (base + enhancement + PBP)
2. Teams and Players (from Sleeper API)
3. Games (from nflreadpy schedules)
4. Offensive stats (from Sleeper API)
5. Defensive stats (from CSVs)
6. NFLverse data (weekly stats, NGS, snaps, injuries, etc.)
7. Betting lines (from The Odds API)
8. Weather data (from Open-Meteo)
9. Play-by-Play data (from nflreadpy) [OPTIONAL - large dataset]

Run this weekly to keep your database up to date.

Author: NFL Fantasy Prediction Engine Team
Version: 3.0 - Added Play-by-Play support
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add Phase 1 directory to path (parent of ingestion_scripts)
_script_dir = Path(__file__).parent
_phase1_dir = _script_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_script_dir))  # Also add ingestion_scripts for local imports

# Import from correct locations based on directory structure
from utils.database import get_db
from utils.logger import get_ingestion_logger
from config.settings import CURRENT_SEASON, ODDS_API_KEY

logger = get_ingestion_logger("master_pipeline")


def check_and_install_requirements():
    """Check if requirements are installed, install if not."""
    # Check both possible locations for requirements.txt
    requirements_file = _phase1_dir / "requirements.txt"
    if not requirements_file.exists():
        requirements_file = _phase1_dir.parent / "requirements.txt"  # Check FootballPredictor root
    
    if requirements_file.exists():
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
        print("No requirements.txt found")


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(num: int, text: str):
    """Print step header."""
    print(f"\n[Step {num}] {text}")
    print("-" * 50)


def run_full_pipeline(
    seasons: list = None,
    skip_base: bool = False,
    skip_nflverse: bool = False,
    skip_betting: bool = False,
    skip_weather: bool = False,
    skip_pbp: bool = False,
    current_week: int = None,
    auto_install: bool = True,
):
    """
    Run the complete data ingestion pipeline.
    
    Args:
        seasons: List of seasons to ingest (default: last 3)
        skip_base: Skip base schema and Sleeper ingestion
        skip_nflverse: Skip NFLverse data
        skip_betting: Skip betting lines
        skip_weather: Skip weather data
        skip_pbp: Skip play-by-play (very large dataset)
        current_week: Current NFL week (for weather)
        auto_install: Automatically install requirements
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
    print(f"Play-by-Play: {'ENABLED' if not skip_pbp else 'SKIPPED'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    db = get_db()
    results = {}
    
    # -------------------------------------------------------------------------
    # Step 1: Base Schema
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(1, "Creating/Verifying Base Schema")
        try:
            from create_schema import create_all_tables, verify_schema
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
        from schema_enhancement import create_enhancement_tables, verify_enhancement_tables
        create_enhancement_tables(db)
        verify = verify_enhancement_tables(db)
        results['enhancement_schema'] = verify['all_exist']
        print(f"Enhancement schema: {'OK' if verify['all_exist'] else 'ISSUES'}")
    except Exception as e:
        print(f"Enhancement schema error: {e}")
        results['enhancement_schema'] = False
    
    # -------------------------------------------------------------------------
    # Step 3: Play-by-Play Schema (only if PBP enabled)
    # -------------------------------------------------------------------------
    if not skip_pbp:
        print_step(3, "Creating Play-by-Play Schema")
        try:
            from schema_pbp import create_pbp_table, verify_pbp_table
            create_pbp_table(db)
            verify = verify_pbp_table(db)
            results['pbp_schema'] = verify['exists']
            col_count = verify.get('column_count', 0)
            print(f"PBP schema: {'OK' if verify['exists'] else 'ISSUES'} ({col_count} columns)")
        except Exception as e:
            print(f"PBP schema error: {e}")
            results['pbp_schema'] = False
    
    # -------------------------------------------------------------------------
    # Step 4: Teams (Sleeper API) - Using TeamsIngestion CLASS
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(4, "Ingesting Teams from Sleeper API")
        try:
            from ingest_teams import TeamsIngestion
            ingestion = TeamsIngestion(db)
            result = ingestion.run()
            count = result.get('total_teams', 0)
            results['teams'] = count
            print(f"Teams: {count} ingested")
        except Exception as e:
            print(f"Teams error: {e}")
            results['teams'] = 0
    
    # -------------------------------------------------------------------------
    # Step 5: Players (Sleeper API) - Using PlayersIngestion CLASS
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(5, "Ingesting Players from Sleeper API")
        try:
            from ingest_players import PlayersIngestion
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
    
    # -------------------------------------------------------------------------
    # Step 6: Games (nflreadpy schedules) - Using GamesIngestion CLASS
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(6, "Ingesting Games from nflreadpy schedules")
        try:
            from ingest_games import GamesIngestion
            ingestion = GamesIngestion(db)
            result = ingestion.run(seasons=seasons)
            count = result.get('inserted', 0) + result.get('updated', 0)
            results['games'] = count
            print(f"Games: {count} ingested")
        except Exception as e:
            print(f"Games error: {e}")
            results['games'] = 0
    
    # -------------------------------------------------------------------------
    # Step 7: Offensive Stats (Sleeper API) - Using OffensiveStatsIngestion CLASS
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(7, "Ingesting Offensive Stats from Sleeper API")
        try:
            from ingest_stats_offense import OffensiveStatsIngestion
            ingestion = OffensiveStatsIngestion(db)
            # FIXED: Use seasons= (plural, list) not season= (singular)
            result = ingestion.run(seasons=seasons)
            count = result.get('total_stats', 0)
            results['offensive_stats'] = count
            print(f"Offensive stats: {count} ingested")
        except Exception as e:
            print(f"Offensive stats error: {e}")
            results['offensive_stats'] = 0
    
    # -------------------------------------------------------------------------
    # Step 8: Defensive Stats (CSVs)
    # -------------------------------------------------------------------------
    if not skip_base:
        print_step(8, "Loading Defensive Stats from CSVs")
        try:
            from load_season_defense_stats import load_all_defense_stats
            count = load_all_defense_stats(db)
            results['defensive_stats'] = count
            print(f"Defensive stats: {count} loaded")
        except Exception as e:
            print(f"Defensive stats error: {e}")
            results['defensive_stats'] = 0
    
    # -------------------------------------------------------------------------
    # Step 9: NFLverse Data (nflreadpy)
    # -------------------------------------------------------------------------
    if not skip_nflverse:
        print_step(9, "Ingesting NFLverse Data (Weekly Stats, NGS, etc.)")
        try:
            from ingest_nflverse import NFLVerseIngestion
            nfl_ingestion = NFLVerseIngestion(db)
            # FIXED: Method is run_full_ingestion(), not ingest_all()
            nfl_results = nfl_ingestion.run_full_ingestion(seasons)
            results['nflverse'] = nfl_results.get('total_rows', 0)
            print(f"NFLverse: {results['nflverse']} rows")
        except ImportError as e:
            print(f"NFLverse import error: {e}")
            print("Install with: pip install nflreadpy pandas polars")
            results['nflverse'] = 0
        except Exception as e:
            print(f"NFLverse error: {e}")
            results['nflverse'] = 0
    
    # -------------------------------------------------------------------------
    # Step 10: Betting Lines (The Odds API)
    # -------------------------------------------------------------------------
    if not skip_betting:
        print_step(10, "Fetching Betting Lines from The Odds API")
        if ODDS_API_KEY:
            try:
                from ingest_betting import BettingLinesIngestion
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
            print("ODDS_API_KEY not set - skipping betting lines")
            print("Set with: set ODDS_API_KEY=your_key_here")
            results['betting'] = 0
    
    # -------------------------------------------------------------------------
    # Step 11: Weather Data (Open-Meteo)
    # -------------------------------------------------------------------------
    if not skip_weather:
        print_step(11, "Fetching Weather Data from Open-Meteo")
        try:
            from ingest_weather import WeatherIngestion
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
    
    # -------------------------------------------------------------------------
    # Step 12: Play-by-Play Data (LARGE - ~50K plays/season)
    # -------------------------------------------------------------------------
    if not skip_pbp:
        print_step(12, "Ingesting Play-by-Play Data")
        print("WARNING: This is a large dataset (~50,000 plays per season)")
        print("Expected time: 5-15 minutes for 3 seasons...")
        try:
            from ingest_pbp import PlayByPlayIngestion
            pbp = PlayByPlayIngestion(db)
            pbp_results = pbp.ingest_seasons(seasons)
            results['play_by_play'] = pbp_results.get('total_plays', 0)
            print(f"Play-by-Play: {results['play_by_play']:,} plays ingested")
        except ImportError as e:
            print(f"PBP import error: {e}")
            print("Install with: pip install nflreadpy pandas polars pyarrow")
            results['play_by_play'] = 0
        except Exception as e:
            print(f"PBP error: {e}")
            results['play_by_play'] = 0
    else:
        print_step(12, "Skipping Play-by-Play Data")
        print("Use without --skip-pbp to include (adds 5-15 min)")
        results['play_by_play'] = 'SKIPPED'
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("Pipeline Complete!")
    
    print("\nResults Summary:")
    for key, value in results.items():
        if isinstance(value, bool):
            status = "OK" if value else "FAILED"
        elif isinstance(value, int):
            status = "OK" if value > 0 else "SKIPPED"
        elif value == 'SKIPPED':
            status = "SKIP"
        else:
            status = "OK" if value else "SKIPPED"
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
  python run_master_pipeline.py                    # Full pipeline (includes PBP)
  python run_master_pipeline.py --skip-pbp         # Skip play-by-play (faster)
  python run_master_pipeline.py --pbp-only         # Only ingest PBP data
  python run_master_pipeline.py --seasons 2024 2025  # Specific seasons
  python run_master_pipeline.py --skip-base        # Skip Sleeper ingestion
  python run_master_pipeline.py --nflverse-only    # Only NFLverse data
  python run_master_pipeline.py --betting-only     # Only betting lines
  python run_master_pipeline.py --no-auto-install  # Skip requirements check
        """
    )
    
    parser.add_argument('--seasons', type=int, nargs='+', help='Seasons to ingest')
    parser.add_argument('--week', type=int, help='Current week for weather')
    parser.add_argument('--skip-base', action='store_true', help='Skip base Sleeper ingestion')
    parser.add_argument('--skip-nflverse', action='store_true', help='Skip NFLverse data')
    parser.add_argument('--skip-betting', action='store_true', help='Skip betting lines')
    parser.add_argument('--skip-weather', action='store_true', help='Skip weather data')
    parser.add_argument('--skip-pbp', action='store_true', help='Skip play-by-play data (faster)')
    parser.add_argument('--pbp-only', action='store_true', help='Only ingest play-by-play data')
    parser.add_argument('--nflverse-only', action='store_true', help='Only NFLverse data')
    parser.add_argument('--betting-only', action='store_true', help='Only betting lines')
    parser.add_argument('--no-auto-install', action='store_true', help='Skip auto-installing requirements')
    
    args = parser.parse_args()
    
    # Handle convenience flags
    skip_base = args.skip_base
    skip_nflverse = args.skip_nflverse
    skip_betting = args.skip_betting
    skip_weather = args.skip_weather
    skip_pbp = args.skip_pbp
    
    if args.pbp_only:
        skip_base = True
        skip_nflverse = True
        skip_betting = True
        skip_weather = True
        skip_pbp = False  # Make sure PBP is NOT skipped
    
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
        current_week=args.week,
        auto_install=not args.no_auto_install,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
