# NFL Fantasy Prediction Engine - Phase 1

## Data Ingestion & Database Setup

This is the complete implementation of Phase 1 as specified in the Comprehensive Operational Plan v2 and Phase 1 v2 documentation.

## Overview

Phase 1 establishes the **entire data foundation** for the fantasy prediction engine and game simulation system. This is the "ground truth layer" that all subsequent phases depend on.

## Project Structure

```
nfl_prediction_engine/
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration
├── database/
│   └── nfl_data.db              # SQLite database
├── ingestion_scripts/
│   ├── __init__.py
│   ├── create_schema.py         # Step 1: Database schema
│   ├── ingest_teams.py          # Step 2: NFL teams
│   ├── ingest_players.py        # Step 3: Players & defenders
│   ├── ingest_games.py          # Step 4: Game schedules
│   ├── ingest_stats_offense.py  # Step 5: Offensive stats
│   ├── ingest_stats_defense.py  # Step 6a: Team defense stats
│   ├── ingest_stats_defenders.py# Step 6b: Individual defender stats
│   ├── run_pipeline.py          # Full pipeline orchestrator
│   └── load_sample_data.py      # Sample data for testing
├── validation/
│   ├── __init__.py
│   └── validate_data.py         # All validation rules
├── utils/
│   ├── __init__.py
│   ├── database.py              # Database connection utilities
│   └── logger.py                # Structured JSON logging
└── logs/
    └── data_ingestion/          # Ingestion logs (JSONL format)
```

## Database Schema

### Entity Reference Tables
- **teams** - 32 NFL teams with conference/division
- **players** - Offensive players (QB, RB, WR, TE) with physical attributes
- **defenders** - Defensive players (CB, S, LB) with position groups and roles
- **seasons** / **weeks** - Time reference tables

### Game Metadata Tables
- **games** - Game schedules with home/away teams, datetime, weather
- **game_injuries** - Player injury tracking (optional)

### Stats Tables
- **player_game_stats** - The most critical table: offensive player stats per game
- **team_defense_game_stats** - Team-level defensive performance
- **defender_game_stats** - Individual defender stats (v2 requirement)
- **coverage_events** - Play-level coverage data (optional)

### Versioning
- **data_versions** - Tracks each ingestion cycle (e.g., "2024_06")

## Pipeline Execution Order

As per Phase 1 v2 Section 6, steps must be executed in strict order:

1. **Create Schema** - Initialize all tables with proper foreign keys
2. **Ingest Teams** - Load 32 NFL teams
3. **Ingest Players** - Load offensive & defensive players from Sleeper API
4. **Ingest Games** - Load game schedules for rolling window
5. **Ingest Offensive Stats** - Populate player_game_stats
6. **Ingest Defense Stats** - Aggregate team defense stats
7. **Ingest Defender Stats** - Individual defender data (from nflfastR/CSV)
8. **Run Validations** - Enforce all Phase 1 v2 validation rules
9. **Create Data Version** - Record version for reproducibility

## Quick Start

## Installation 
 
1. Create a virtual environment: 
 
   ```bash 
   python -m venv .venv 
   source .venv/bin/activate  # Windows: .venv\Scripts\activate 
2.  Install dependencies: 
3.  pip install -r requirements.txt 
4.  Run the Phase 1 pipeline: 
5.  python -m ingestion_scripts.run_pipeline --mode ful

### 1. Initialize the Database
```bash
python "Phase 1/ingestion_scripts/create_schema.py"
```

### 2. Run Full Pipeline
```bash
python "Phase 1/ingestion_scripts/run_pipeline.py"
```

### 3. Run with Sample Data (for testing)
```bash
python "Phase 1/ingestion_scripts/load_sample_data.py" --weeks 4
python "Phase 1/ingestion_scripts/ingest_stats_defense.py" --season 2024
python "Phase 1/validation/validate_data.py"
```
### 4. Incremental Weekly Update
```bash
python "Phase 1/ingestion_scripts/run_pipeline.py" --incremental --season 2024 --week 10
```

## Configuration

Key settings in `config/settings.py`:

```python
# Database
DATABASE_PATH = "database/nfl_data.db"

# API (Sleeper)
SLEEPER_BASE_URL = "https://api.sleeper.app/v1"

# Data Window
ROLLING_WINDOW_SEASONS = 3  # Last 3 seasons

# Positions
OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE"]
DEFENSIVE_POSITIONS = ["CB", "S", "LB", ...]
```

## Validation Rules (Phase 1 v2 Section 7)

All validations are **STRICT** - failures stop ingestion immediately.

### 7.1 Entity Integrity
- Every player_id in player_game_stats must exist in players
- Every defender_id in defender_game_stats must exist in defenders
- All team_ids and game_ids must exist in their respective tables

### 7.2 Duplicate Protection
- (player_id, game_id) must be unique in player_game_stats
- (defender_id, game_id) must be unique in defender_game_stats

### 7.3 Null and Range Validation
- Snap counts ≥ 0
- Routes ≥ 0
- Alignment percentages ∈ [0, 1]

### 7.4 Coverage Probability Validation
- man_coverage_pct + zone_coverage_pct ≈ 1.0 (within tolerance)

### 7.5 Weekly Completeness
- Every scheduled game must have offensive, defensive, and defender stats

### 7.6 Anti-Leakage Validation
- Every row must include: season, week, game_id, ingested_at timestamp

## Logging

All logs use structured JSON lines format (as per Plan v2 Section 10.3):

```json
{
  "timestamp": "2025-09-14T03:14:07Z",
  "level": "INFO",
  "source": "ingest_players",
  "event": "ingestion_complete",
  "row_count": 1500,
  "detail": "Ingested 1500 rows in 5.2s"
}
```

## Data Sources

### Primary: Sleeper API
- `/players/nfl` - Player metadata
- `/stats/nfl/regular/{season}/{week}` - Weekly stats
- `/state/nfl` - Current NFL state

### Optional: nflfastR
- Play-by-play data with defender coverage stats
- Alignment metrics, EPA, coverage types
- Load via CSV/Parquet files

## Anti-Leakage Guarantees

Every row contains time-indexing fields to prevent future data from leaking into historical features:
- season
- week
- game_id
- ingested_at (timestamp)

Phase 2 feature engineering will strictly enforce:
- Only use data from weeks < target_week when building features
- No mixing of week N stats for predicting week N

## Key Dependencies

- Python 3.10+
- sqlite3 (built-in)
- requests (for API calls)
- pandas (optional, for CSV/data processing)

## CLI Reference

### create_schema.py
```bash
python create_schema.py [--drop] [--verify-only]
```

### ingest_teams.py
```bash
python ingest_teams.py [--use-api] [--validate-only]
```

### ingest_players.py
```bash
python ingest_players.py [--validate-only] [--offensive-only] [--defensive-only]
```

### ingest_games.py
```bash
python ingest_games.py [--season 2023 2024] [--validate-only]
```

### ingest_stats_offense.py
```bash
python ingest_stats_offense.py [--season 2024] [--week 1 2 3] [--validate-only]
```

### run_pipeline.py
```bash
python run_pipeline.py [--season 2023 2024] [--week 10] [--drop-existing --confirm-drop]
                       [--incremental] [--validate-only] [--no-fail-fast]
```

### validate_data.py
```bash
python validate_data.py [--season 2024] [--week 10] [--fail-fast]
```

## Next Steps: Phase 2

Phase 2 (Feature Engineering) will consume this data layer to compute:
- Player usage features (snap share, target share, carry share)
- Player efficiency features (yards per route, yards per carry)
- Player form deltas (recent performance vs season average)
- Team context features (pass rate, game script)
- Opponent defense features (yards allowed, targets allowed by position)
- Defender-specific features (effective defender metrics)
- Archetype assignments (offensive and defensive)

## Author

NFL Fantasy Prediction Engine Team
Phase 1 - Data Ingestion & Database Setup
Version 2.0
