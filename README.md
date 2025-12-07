# NFL Fantasy Prediction Engine

A machine learning pipeline for NFL fantasy football predictions, built with a modular, phased architecture designed for weekly production use.

---

## Table of Contents

1. [Quick Start: Weekly Usage](#quick-start-weekly-usage)
2. [How Training Works](#how-training-works)
3. [Project Architecture](#project-architecture)
4. [Installation & Setup](#installation--setup)
5. [Data Sources](#data-sources)
6. [Phase Descriptions](#phase-descriptions)
7. [Database Schema](#database-schema)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start: Weekly Usage

Once the system is fully trained, here's your weekly workflow during the NFL season:

### Every Tuesday (After Games Complete)

```bash
# Step 1: Ingest new game results and player stats
python -m phase_1.ingestion_scripts.populate_player_game_stats --season 2024 --week <LAST_WEEK>

# Step 2: Rebuild features with new data
python -m phase_2.build_features --season 2024

# Step 3: (Optional) Re-evaluate model performance
python -m phase_3.pipeline.run_baseline --season 2024 --evaluate
```

### Every Wednesday-Saturday (Before Games)

```bash
# Generate predictions for upcoming week
python -m phase_4.predict --season 2024 --week <UPCOMING_WEEK>

# Export predictions for your fantasy lineup
python -m phase_4.export --format csv --output predictions.csv
```

### Typical Weekly Timeline

| Day | Task | Command |
|-----|------|---------|
| Tuesday | Ingest results | `populate_player_game_stats` |
| Tuesday | Rebuild features | `build_features` |
| Wednesday | Generate predictions | `predict` |
| Thursday | Set lineups | Use exported CSV |
| Sunday/Monday | Games play | Watch and enjoy |

---

## How Training Works

### Understanding the Training Pipeline

This system uses a **multi-phase approach** where each phase builds upon the previous:

```
Raw Data → Features → Baseline Model → ML Models → Ensemble
 (Phase 1)  (Phase 2)    (Phase 3)      (Phase 4)   (Phase 5)
```

### Phase 3: Baseline Predictor (Rule-Based, No Training)

The baseline predictor in Phase 3 is **NOT machine learning**. It uses fixed mathematical formulas:

```python
# Simplified example of baseline prediction
prediction = (
    0.60 * player_season_average +     # Long-term form
    0.25 * player_recent_3_game_avg +  # Short-term momentum
    0.15 * opponent_defense_adjustment # Matchup factor
)
```

**Key characteristics:**
- No training required - formulas are deterministic
- Provides stable, interpretable predictions
- Serves as a "floor" that ML models must beat
- Acts as a fallback when ML models are uncertain

To run baseline predictions:
```bash
python -m phase_3.pipeline.run_baseline --season 2024 --week 10 --evaluate
```

### Phase 4: Machine Learning Training (Coming Soon)

Phase 4 will introduce actual ML training:

```bash
# Train gradient boosting model on historical data
python -m phase_4.train --model xgboost --seasons 2021 2022 2023

# Train neural network model
python -m phase_4.train --model neural_net --seasons 2021 2022 2023

# Validate on holdout season
python -m phase_4.validate --season 2024
```

**Training requirements:**
- Historical data: 3-4 seasons recommended (2021-2024)
- Compute: ~10-30 minutes on standard laptop
- GPU: Optional, provides 5-10x speedup for neural networks

### Phase 5: Ensemble Meta-Model

The final ensemble combines all models:

```
Final Prediction = weighted_average(
    baseline_prediction,      # Stable foundation
    xgboost_prediction,       # Tree-based patterns
    neural_net_prediction,    # Complex interactions
    # ... other models
)
```

Weights are learned automatically to minimize prediction error.

### When to Retrain

| Scenario | Recommended Action |
|----------|-------------------|
| New season starts | Retrain with previous season added |
| Mid-season | Usually not needed; features update automatically |
| Major roster changes | Features auto-adapt via rolling windows |
| Model performance drops | Evaluate and retrain if MAE increases >15% |

---

## Project Architecture

```
nfl_prediction_engine/
├── phase_1/                    # Data Ingestion
│   ├── database/
│   │   └── nfl_data.db        # SQLite database (all data)
│   ├── ingestion_scripts/
│   │   ├── populate_games.py
│   │   ├── populate_players.py
│   │   ├── populate_player_game_stats.py
│   │   ├── populate_team_stats.py
│   │   ├── populate_nflverse_stats.py
│   │   ├── populate_defender_stats.py
│   │   └── backfill_sleeper_stats.py
│   └── schema/
│       └── schema.sql
│
├── phase_2/                    # Feature Engineering
│   ├── build_features.py      # Main feature builder
│   ├── player_id_mapping.py   # Sleeper↔GSIS ID mapping
│   └── analyze_features.py    # Feature analysis tools
│
├── phase_3/                    # Baseline Predictor
│   ├── predictors/
│   │   └── baseline.py        # Rule-based predictor
│   ├── pipeline/
│   │   └── run_baseline.py    # Prediction runner
│   └── db.py                  # Database utilities
│
├── phase_4/                    # ML Models (planned)
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── phase_5/                    # Ensemble (planned)
│   ├── meta_model/
│   └── production/
│
└── docs/                       # Documentation
    └── planning/
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- 2GB disk space (for database)
- Internet connection (for data fetching)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nfl_prediction_engine.git
cd nfl_prediction_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
pandas>=1.5.0
numpy>=1.23.0
sqlite3 (standard library)
requests>=2.28.0
nfl_data_py>=0.3.0
scikit-learn>=1.1.0  # Phase 4+
xgboost>=1.7.0       # Phase 4+
torch>=2.0.0         # Phase 4+ (optional)
```

### Initial Data Population

```bash
# Populate all historical data (takes 15-30 minutes)
python -m phase_1.ingestion_scripts.populate_all --seasons 2021 2022 2023 2024

# Build initial features
python -m phase_2.build_features --seasons 2021 2022 2023 2024
```

---

## Data Sources

### Primary Sources

| Source | Data Type | Coverage | Update Frequency |
|--------|-----------|----------|------------------|
| **Sleeper API** | Player stats, rosters | 2021-present | Real-time |
| **nflverse/nflfastR** | Play-by-play, advanced stats | 1999-present | Weekly |
| **nfl_data_py** | Player ID mapping | Current | As needed |

### Data Flow

```
Sleeper API ─────────────┐
                         ├──→ SQLite DB ──→ Features ──→ Predictions
nflverse (play-by-play) ─┘
```

### Player ID Mapping

The system maps between different ID systems:
- **Sleeper ID**: Used by Sleeper fantasy platform
- **GSIS ID**: Official NFL Game Statistics ID
- **nflverse ID**: Used in nflfastR ecosystem

Mapping is handled automatically via `nfl_data_py.import_ids()`.

---

## Phase Descriptions

### Phase 1: Data Ingestion

**Purpose**: Collect and store raw NFL data

**Tables populated**:
- `games` - Game schedules and results
- `players` - Player biographical info
- `player_game_stats` - Per-game statistics
- `team_stats` - Team-level aggregates
- `defender_stats` - Defensive player statistics

**Key scripts**:
```bash
python -m phase_1.ingestion_scripts.populate_games --season 2024
python -m phase_1.ingestion_scripts.populate_player_game_stats --season 2024 --week 10
```

### Phase 2: Feature Engineering

**Purpose**: Transform raw stats into predictive features

**Anti-leakage design**: All features are calculated using ONLY data available BEFORE the game being predicted.

**Feature categories**:
- `usage_*` - Volume metrics (targets, carries, snaps)
- `efficiency_*` - Rate stats (yards/carry, catch rate)
- `opp_def_*` - Opponent defensive adjustments
- `vegas_*` - Betting market signals (planned)
- `weather_*` - Game conditions (planned)

**Key script**:
```bash
python -m phase_2.build_features --season 2024 --week 10
```

### Phase 3: Baseline Predictor

**Purpose**: Provide stable, interpretable predictions

**Method**: Weighted combination of:
- Season-to-date averages (60%)
- Recent form (last 3 games) (25%)
- Opponent adjustment (15%)

**Output**: Predictions saved to `baseline_predictions` table

**Key script**:
```bash
python -m phase_3.pipeline.run_baseline --season 2024 --evaluate
```

### Phase 4: ML Models (Planned)

**Purpose**: Learn complex patterns from historical data

**Models planned**:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Neural networks (if GPU available)

### Phase 5: Ensemble (Planned)

**Purpose**: Combine all models for best predictions

**Method**: Stacking meta-learner that weights individual models based on their strengths.

---

## Database Schema

### Core Tables

| Table | Rows | Description |
|-------|------|-------------|
| `games` | ~1,100 | NFL game schedules |
| `players` | ~12,000 | Player info |
| `player_game_stats` | ~85,000 | Per-game stats |
| `player_game_features` | ~29,000 | Engineered features |
| `baseline_predictions` | ~29,000 | Phase 3 predictions |
| `defender_stats` | ~52,000 | Defensive stats |

### Key Relationships

```
players.player_id ──┬──→ player_game_stats.player_id
                    └──→ player_game_features.player_id

games.game_id ──────┬──→ player_game_stats.game_id
                    └──→ player_game_features.game_id
```

---

## Troubleshooting

### Common Issues

**"No features found for season X"**
```bash
# Rebuild features for that season
python -m phase_2.build_features --season X
```

**"Player ID not found"**
```bash
# Refresh ID mapping
python -m phase_2.player_id_mapping --refresh
```

**"Database locked"**
- Close any other connections to the database
- Check for zombie Python processes

**Low prediction accuracy**
- Ensure features are up-to-date
- Check that historical data is complete
- Verify player ID mappings are current

### Getting Help

- Check the `docs/` folder for detailed documentation
- Open an issue on GitHub for bugs
- Review `phase_X/__init__.py` files for phase-specific notes

---

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data ingestion working |
| Phase 2 | ✅ Complete | Features built for 2021-2025 |
| Phase 3 | ✅ Complete | Baseline predictor operational |
| Phase 4 | 🔜 Planned | ML model training |
| Phase 5 | 🔜 Planned | Ensemble integration |

### Performance Metrics (Baseline)

| Metric | Targets | Fantasy Points (PPR) |
|--------|---------|---------------------|
| MAE | 1.13 | 4.89 |
| Correlation | 0.735 | 0.503 |

---

## License

[Your License Here]

## Contributing

[Contribution Guidelines]

---

*Last updated: December 2024*
