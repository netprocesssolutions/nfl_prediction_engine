"""
Ingestion Scripts package for NFL Fantasy Prediction Engine - Phase 1.

This package contains all data ingestion modules as specified in Phase 1 v2:

Pipeline Steps (executed in order):
1. create_schema - Database schema creation
2. ingest_teams - NFL teams ingestion
3. ingest_players - Offensive and defensive players
4. ingest_games - Game schedules
5. ingest_stats_offense - Offensive player game stats
6. ingest_stats_defense - Team defense stats
7. ingest_stats_defenders - Individual defender stats

Orchestration:
- run_pipeline - Full pipeline orchestrator
- run_master_pipeline - Master orchestrator with all data sources
"""

# Lazy imports to avoid circular dependencies
# Users should import specific modules as needed:
#   from ingestion_scripts.ingest_teams import TeamsIngestion
#   from ingestion_scripts.run_master_pipeline import run_full_pipeline

__all__ = [
    "create_schema",
    "ingest_teams",
    "ingest_players", 
    "ingest_games",
    "ingest_stats_offense",
    "ingest_stats_defense",
    "ingest_stats_defenders",
    "ingest_nflverse",
    "ingest_betting",
    "ingest_weather",
    "run_pipeline",
    "run_master_pipeline",
    "load_season_defense_stats",
]
