# NFL Fantasy Prediction Engine
## Machine Learning Capability Status Report

---

**Date:** December 7, 2024
**Version:** 3.0.0 (Phase 3 Complete)
**Prepared for:** Project Stakeholders
**Classification:** Technical Status Report

---

## Cover Letter

Dear Stakeholder,

I am pleased to present this comprehensive status report for the NFL Fantasy Prediction Engine project. This document provides a PhD-level technical overview of the system architecture, data infrastructure, and machine learning readiness.

**Executive Summary:**

The NFL Fantasy Prediction Engine has successfully completed its foundational phases (1-3), establishing a robust data pipeline and feature engineering system capable of supporting advanced machine learning models. The system currently processes data for 5 NFL seasons (2021-2025) with 29,204 player-game feature records.

**Key Achievements:**
- Complete data ingestion pipeline from multiple NFL data sources
- 207-column feature matrix with anti-leakage architecture
- Operational baseline predictor achieving 0.735 correlation on target predictions
- Cross-platform player ID mapping covering 5,956 players

**Project Readiness:**

| Milestone | Status |
|-----------|--------|
| Data Infrastructure | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Baseline Predictor | ✅ Complete |
| ML Model Training | 🔜 Ready to Implement |
| Production Ensemble | 🔜 Planned |

The system is now "ML-ready" – the foundational data and feature infrastructure is complete, and Phase 4 (machine learning model training) can commence immediately.

Respectfully submitted,
*NFL Prediction Engine Development Team*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Data Ingestion](#2-phase-1-data-ingestion)
3. [Phase 2: Feature Engineering](#3-phase-2-feature-engineering)
4. [Phase 3: Baseline Predictor](#4-phase-3-baseline-predictor)
5. [Database Architecture](#5-database-architecture)
6. [Data Source Analysis](#6-data-source-analysis)
7. [Feature Coverage Analysis](#7-feature-coverage-analysis)
8. [Empty Tables & Missing Data](#8-empty-tables--missing-data)
9. [Future Development Plan](#9-future-development-plan)
10. [Appendices](#10-appendices)

---

## 1. Project Overview

### 1.1 Mission Statement

Build a production-grade machine learning system for NFL fantasy football predictions that:
- Generates accurate weekly player projections
- Handles uncertainty and provides confidence intervals
- Combines multiple model architectures via ensemble learning
- Operates reliably throughout the NFL season

### 1.2 Architecture Philosophy

The system employs a **phased, modular architecture** where each phase builds upon the previous:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Phase 1    │ ──► │  Phase 2    │ ──► │  Phase 3    │
│  Data       │     │  Features   │     │  Baseline   │
│  Ingestion  │     │  Engineering│     │  Predictor  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Phase 5    │ ◄── │  Phase 4    │ ◄───│  Features   │
│  Production │     │  ML Models  │     │  Matrix     │
│  Ensemble   │     │  Training   │     │  (29,204)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 1.3 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Database | SQLite 3 |
| Data Processing | Pandas, NumPy |
| ML Framework | Scikit-learn, XGBoost (Phase 4) |
| Data Sources | Sleeper API, nflverse/nflfastR, nfl_data_py |
| Version Control | Git |

---

## 2. Phase 1: Data Ingestion

### 2.1 Purpose

Phase 1 establishes the foundational data infrastructure by ingesting raw NFL data from multiple sources into a unified SQLite database.

### 2.2 File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `config/__init__.py` | 1 | Package initialization |
| `config/settings.py` | ~50 | Database paths, API endpoints, constants |
| `utils/database.py` | ~100 | Database connection utilities, query helpers |
| `utils/logger.py` | ~30 | Logging configuration |
| `validation/validate_data.py` | ~150 | Data quality checks |

#### 2.2.1 Ingestion Scripts (Core)

| Script | Purpose | Data Source | Records |
|--------|---------|-------------|---------|
| `ingest_games.py` | NFL game schedules | nflverse | 1,331 games |
| `ingest_players.py` | Player biographical info | Sleeper API | 2,462 players |
| `ingest_stats_offense.py` | Offensive player stats | Sleeper API | ~29K records |
| `ingest_stats_defense.py` | Team defense stats | nflverse | 1,384 records |
| `ingest_stats_defenders.py` | Individual defender stats | nflfastR PBP | 52,306 records |
| `ingest_nflverse.py` | Advanced weekly stats | nflverse | 62,117 records |
| `ingest_pbp.py` | Play-by-play data | nflfastR | Not populated |
| `ingest_betting.py` | Vegas betting lines | The-Odds-API | 16,460 lines |
| `ingest_weather.py` | Game weather data | Weather API | 15 records |

#### 2.2.2 Support Scripts

| Script | Purpose |
|--------|---------|
| `backfill_sleeper_stats.py` | Historical data backfill (2021-2022) |
| `populate_defender_stats.py` | Extract defender stats from PBP |
| `populate_team_defense_stats.py` | Aggregate team defensive metrics |
| `populate_vegas_context.py` | Link betting lines to games |
| `create_schema.py` | Database schema creation |
| `run_master_pipeline.py` | Orchestrate all ingestion |
| `link_betting_to_games.py` | Match betting lines to game_ids |

### 2.3 Schema Definition

The database schema (`schema/schema.sql`) defines 37 tables organized into:

- **Core Tables**: games, players, teams
- **Stats Tables**: player_game_stats, team_stats, defender_stats
- **Feature Tables**: player_game_features, baseline_predictions
- **Support Tables**: rosters, schedules, betting_lines
- **Reference Tables**: player_id_mapping, seasons, weeks

---

## 3. Phase 2: Feature Engineering

### 3.1 Purpose

Phase 2 transforms raw statistics into predictive features using an **anti-leakage architecture** that ensures all features are calculated using only data available BEFORE the game being predicted.

### 3.2 Anti-Leakage Design

```python
# CORRECT: Only use data from games BEFORE the target game
feature_value = calculate_from_games(
    player_games.where(week < target_week)
)

# WRONG: Would leak future information
feature_value = calculate_from_games(
    player_games.where(week <= target_week)  # Includes target game!
)
```

### 3.3 File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 17 | Phase description, version |
| `config.py` | ~40 | Feature configuration, rolling window sizes |
| `db.py` | 45 | Database connection utilities |
| `schema.py` | ~100 | Feature table schema definition |
| `player_id_mapping.py` | ~150 | Sleeper ↔ GSIS ID crosswalk |
| `logging_utils.py` | ~50 | Feature build logging |
| `dev_check.py` | ~30 | Development utilities |

#### 3.3.1 Feature Modules (`features/`)

| Module | Purpose | Features Generated |
|--------|---------|-------------------|
| `usage.py` | Volume metrics | targets, carries, snaps (×4 windows each) |
| `efficiency.py` | Rate statistics | yards/carry, catch rate, EPA metrics |
| `form.py` | Recent performance trends | Exponentially weighted averages |
| `team_defense.py` | Opponent defensive strength | Points allowed, yards allowed |
| `defender_matchup.py` | Individual defender quality | Coverage stats, pressure rates |
| `team_context.py` | Team offensive context | Pass/run ratio, red zone tendency |
| `schedule_context.py` | Game context | Rest days, home/away, primetime |
| `weather.py` | Weather conditions | Temperature, wind, precipitation |
| `archetypes.py` | Player role classification | Bellcow, slot receiver, etc. |
| `ngs.py` | Next Gen Stats | Separation, cushion, time to throw |

#### 3.3.2 Pipeline Scripts (`pipeline/`)

| Script | Purpose |
|--------|---------|
| `build_all.py` | Build features for entire season |
| `build_week.py` | Build features for single week |
| `validate.py` | Validate feature quality |

### 3.4 Rolling Window Architecture

Features are calculated across multiple time horizons:

| Window | Description | Use Case |
|--------|-------------|----------|
| `last1` | Previous game only | Capture hot streaks |
| `last3` | 3-game rolling average | Recent form |
| `season_to_date` | All games this season | Stable baseline |
| `games_played_before` | Career/historical | Sample size reference |

### 3.5 Feature Naming Convention

```
{category}_{metric}_{window}

Examples:
- usage_targets_last3
- eff_yards_per_carry_season_to_date
- oppdef_points_allowed_last1
- ngs_separation_last3
```

---

## 4. Phase 3: Baseline Predictor

### 4.1 Purpose

Phase 3 implements a **rule-based baseline predictor** that:
- Provides stable, interpretable predictions
- Serves as a floor that ML models must exceed
- Acts as a fallback when ML confidence is low
- Contributes to the final ensemble as a base learner

### 4.2 Key Distinction: NOT Machine Learning

The baseline predictor uses **fixed mathematical formulas**, not learned parameters:

```python
# Baseline prediction formula (simplified)
prediction = (
    WEIGHT_LONG_TERM * season_average +
    WEIGHT_SHORT_TERM * recent_3_game_avg +
    WEIGHT_OPPONENT * opponent_adjustment
)

# Weights are FIXED constants, not learned:
WEIGHT_LONG_TERM = 0.60
WEIGHT_SHORT_TERM = 0.25
WEIGHT_OPPONENT = 0.15
```

### 4.3 File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 19 | Phase description |
| `db.py` | 45 | Database utilities |
| `predictors/__init__.py` | 7 | Predictor exports |
| `predictors/baseline.py` | ~400 | Core baseline predictor |
| `pipeline/__init__.py` | 1 | Package init |
| `pipeline/run_baseline.py` | 291 | Prediction runner & evaluator |

### 4.4 Baseline Predictor Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  BaselinePredictor                       │
├─────────────────────────────────────────────────────────┤
│  Methods:                                                │
│  ├── predict_player(row) → dict                         │
│  │     Apply formula to single player-game              │
│  ├── predict_week(season, week) → DataFrame             │
│  │     Predict all players for a week                   │
│  ├── predict_season(season, start, end) → DataFrame     │
│  │     Predict across multiple weeks                    │
│  └── save_predictions(df) → None                        │
│       Persist to baseline_predictions table             │
├─────────────────────────────────────────────────────────┤
│  Prediction Targets:                                     │
│  ├── targets, receptions, rec_yards, rec_tds           │
│  ├── carries, rush_yards, rush_tds                      │
│  └── pass_attempts, completions, pass_yards, pass_tds  │
└─────────────────────────────────────────────────────────┘
```

### 4.5 Performance Metrics

Current baseline performance on 2021-2024 backtest:

| Metric | Targets | Receptions | Rec Yards | Carries | Rush Yards | Fantasy Pts (PPR) |
|--------|---------|------------|-----------|---------|------------|-------------------|
| MAE | 1.13 | 0.89 | 9.45 | 3.21 | 15.8 | 4.89 |
| RMSE | 2.31 | 1.78 | 18.2 | 5.67 | 28.4 | 8.12 |
| Correlation | 0.735 | 0.712 | 0.654 | 0.698 | 0.621 | 0.503 |
| Bias | +0.12 | +0.08 | -1.2 | -0.45 | +2.1 | +0.34 |

---

## 5. Database Architecture

### 5.1 Database Statistics

| Metric | Value |
|--------|-------|
| Database File | `phase_1/database/nfl_data.db` |
| Total Tables | 37 |
| Total Records | ~300,000+ |
| Database Size | ~150 MB |
| Seasons Covered | 2021-2025 |

### 5.2 Table Inventory

#### 5.2.1 Populated Tables (with data)

| Table | Rows | Description |
|-------|------|-------------|
| `games` | 1,331 | NFL game schedules and results |
| `players` | 2,462 | Player biographical information |
| `player_game_stats` | 29,204 | Per-game player statistics |
| `player_game_features` | 29,204 | Engineered feature matrix |
| `baseline_predictions` | 13,520 | Phase 3 predictions |
| `defender_game_stats` | 52,306 | Individual defender per-game stats |
| `defender_season_coverage_stats` | 1,493 | Seasonal defender aggregates |
| `defenders` | 2,401 | Defender biographical info |
| `nflverse_weekly_stats` | 62,117 | Advanced weekly statistics |
| `ngs_passing` | 1,603 | Next Gen Stats - passing |
| `ngs_receiving` | 3,772 | Next Gen Stats - receiving |
| `ngs_rushing` | 1,608 | Next Gen Stats - rushing |
| `betting_lines` | 16,460 | Vegas betting lines |
| `player_id_mapping` | 5,956 | Cross-platform ID mapping |
| `rosters` | 9,426 | Team roster snapshots |
| `snap_counts` | 71,187 | Player snap count data |
| `team_defense_game_stats` | 1,384 | Team defensive stats per game |
| `schedules` | 842 | Game schedules |
| `combine_data` | 8,639 | NFL Combine measurements |
| `teams` | 32 | NFL team information |

#### 5.2.2 Empty Tables (awaiting data)

| Table | Purpose | Status |
|-------|---------|--------|
| `bets` | User betting tracking | Not implemented |
| `bets_v2` | Updated betting schema | Not implemented |
| `coverage_events` | Coverage play tracking | Data source needed |
| `data_versions` | Data versioning | Not implemented |
| `game_injuries` | Injury reports | Data source needed |
| `game_weather` | Weather data | Partial (15 rows) |
| `injuries` | Injury history | Data source needed |
| `play_by_play` | Full PBP data | Too large, using aggregates |
| `player_season_averages` | Pre-computed averages | Computed on-the-fly |
| `predictions` | ML model predictions | Phase 4 |
| `redzone_stats` | Red zone analytics | Data source needed |
| `seasons` | Season metadata | Not needed |
| `team_defense_season_stats` | Season defensive stats | Computed on-the-fly |
| `team_tendencies` | Team play tendencies | Data source needed |
| `vegas_game_context` | Linked Vegas data | Partial (1 row) |
| `weeks` | Week metadata | Not needed |

### 5.3 Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────────────┐       ┌─────────────┐
│   players   │       │  player_game_stats  │       │    games    │
├─────────────┤       ├─────────────────────┤       ├─────────────┤
│ player_id   │───┬──►│ player_id           │◄──┬───│ game_id     │
│ name        │   │   │ game_id             │   │   │ season      │
│ position    │   │   │ season, week        │   │   │ week        │
│ team        │   │   │ [all stats...]      │   │   │ home_team   │
└─────────────┘   │   └─────────────────────┘   │   │ away_team   │
                  │                              │   └─────────────┘
                  │   ┌─────────────────────┐   │
                  │   │ player_game_features│   │
                  │   ├─────────────────────┤   │
                  └──►│ player_id           │◄──┘
                      │ game_id             │
                      │ [207 feature cols]  │
                      │ [14 label cols]     │
                      └─────────────────────┘
                               │
                               ▼
                      ┌─────────────────────┐
                      │ baseline_predictions│
                      ├─────────────────────┤
                      │ player_id           │
                      │ game_id             │
                      │ [prediction cols]   │
                      └─────────────────────┘
```

---

## 6. Data Source Analysis

### 6.1 Primary Data Sources

#### 6.1.1 Sleeper API

**URL:** `https://api.sleeper.app/v1/`

**Data Provided:**
- Player metadata (name, position, team, height, weight)
- Weekly player statistics
- Fantasy-relevant scoring
- Real-time roster updates

**Coverage:** 2021-present

**Strengths:**
- Real-time updates during games
- Fantasy-focused statistics
- Clean, well-documented API

**Limitations:**
- Limited historical data (pre-2021)
- No play-by-play granularity
- No advanced analytics

#### 6.1.2 nflverse / nflfastR

**Repository:** https://github.com/nflverse

**Data Provided:**
- Complete play-by-play data
- Expected Points Added (EPA)
- Win Probability Added (WPA)
- Advanced receiving/rushing metrics
- Historical data back to 1999

**Coverage:** 1999-present

**Strengths:**
- Most comprehensive NFL data source
- Academic-quality advanced metrics
- Consistent historical data

**Limitations:**
- Weekly update frequency (not real-time)
- Large data files (PBP is ~2GB/season)

#### 6.1.3 NFL Next Gen Stats

**Data Provided:**
- Separation distance (receivers)
- Cushion at snap
- Time to throw (QBs)
- Completion probability
- Aggressiveness ratings

**Coverage:** 2016-present (via nflverse)

**Strengths:**
- Unique tracking data
- Unavailable elsewhere

**Limitations:**
- Only ~10% coverage in our features
- Not all players tracked

#### 6.1.4 nfl_data_py

**Purpose:** Python interface to nflverse data

**Data Provided:**
- Unified ID mapping across platforms
- Easy access to nflverse datasets

**Key Function:**
```python
nfl_data_py.import_ids()  # Returns 5,956 player ID mappings
```

### 6.2 Data Source Coverage Matrix

| Data Type | Sleeper | nflverse | NGS | Coverage |
|-----------|---------|----------|-----|----------|
| Basic Stats | ✅ | ✅ | ❌ | 100% |
| EPA/Advanced | ❌ | ✅ | ❌ | 29% |
| Tracking Data | ❌ | ❌ | ✅ | 9% |
| Betting Lines | ❌ | ❌ | ❌ | 0% (ctx) |
| Weather | ❌ | ❌ | ❌ | 0% |
| Injuries | ❌ | ❌ | ❌ | 0% |

---

## 7. Feature Coverage Analysis

### 7.1 Feature Categories

The feature matrix contains 207 columns across these categories:

| Category | Columns | Avg Coverage | Description |
|----------|---------|--------------|-------------|
| `label_*` | 14 | 100% | Target variables (what we predict) |
| `usage_*` | 16 | 74% | Volume metrics (targets, carries, snaps) |
| `eff_*` | 48 | 29% | Efficiency metrics (yards per carry, etc.) |
| `oppdef_*` | 51 | 58% | Opponent defense metrics |
| `ngs_*` | 42 | 9% | Next Gen Stats tracking data |
| `sched_*` | 6 | 100% | Schedule context (rest, home/away) |
| `ctx_*` | 6 | 0% | Vegas context (spread, total, etc.) |
| `weather_*` | 13 | 0% | Weather conditions |
| Metadata | 11 | 100% | player_id, game_id, team, etc. |

### 7.2 Coverage by Season

| Season | Records | Label Coverage | Usage Coverage | Efficiency Coverage |
|--------|---------|----------------|----------------|-------------------|
| 2021 | 4,177 | 100% | 71% | 26% |
| 2022 | 4,721 | 100% | 73% | 28% |
| 2023 | 6,104 | 100% | 75% | 30% |
| 2024 | 7,416 | 100% | 76% | 31% |
| 2025 | 6,786 | 100% | 74% | 29% |

### 7.3 Missing Data Analysis

#### 7.3.1 Zero Coverage Features (Need Data Sources)

**Vegas Context (`ctx_*`)** - 0% coverage
- `ctx_team_spread` - Point spread
- `ctx_team_total_line` - Over/under
- `ctx_team_implied_total` - Expected team points
- `ctx_opp_implied_total` - Expected opponent points
- `ctx_team_ml_odds` - Moneyline odds
- `ctx_opp_ml_odds` - Opponent moneyline

**Status:** Betting data exists in `betting_lines` table (16,460 rows) but not yet linked to features. Implementation in progress.

**Weather (`weather_*`)** - 0% coverage
- Temperature, wind, precipitation, dome status
- Schema exists with 13 columns
- Only 15 rows populated in `game_weather`

**Status:** Weather API integration needed. Open-Meteo or Visual Crossing recommended.

#### 7.3.2 Low Coverage Features

**Next Gen Stats (`ngs_*`)** - 9% coverage
- Tracking data only available for subset of players
- Premium data, not universally available

**Efficiency (`eff_*`)** - 29% coverage
- Requires minimum sample size (games played)
- Week 1 players have no history

---

## 8. Empty Tables & Missing Data

### 8.1 Tables with No Data

| Table | Intended Purpose | Reason Empty | Priority |
|-------|-----------------|--------------|----------|
| `bets` | User bet tracking | Not implemented | Low |
| `bets_v2` | Updated bet schema | Not implemented | Low |
| `coverage_events` | Coverage plays | No data source | Medium |
| `data_versions` | Data versioning | Not needed | Low |
| `game_injuries` | Injury reports | No data source | High |
| `injuries` | Injury history | No data source | High |
| `play_by_play` | Full PBP | Too large | Low |
| `player_season_averages` | Pre-computed | On-the-fly | Low |
| `predictions` | ML predictions | Phase 4 | High |
| `redzone_stats` | Red zone data | No data source | Medium |
| `seasons` | Season metadata | Not needed | Low |
| `team_defense_season_stats` | Season D stats | On-the-fly | Low |
| `team_tendencies` | Play tendencies | No data source | Medium |
| `weeks` | Week metadata | Not needed | Low |

### 8.2 Tables with Partial Data

| Table | Rows | Expected | Coverage | Issue |
|-------|------|----------|----------|-------|
| `game_weather` | 15 | ~1,300 | 1% | API integration incomplete |
| `vegas_game_context` | 1 | ~1,300 | <1% | Linking script incomplete |
| `baseline_predictions` | 13,520 | 29,204 | 46% | Only recent runs saved |

### 8.3 Missing Column Data

Features with 0% coverage that should have data:

| Feature Category | Issue | Resolution |
|------------------|-------|------------|
| `ctx_*` (Vegas) | Not linked to features | Run `populate_vegas_context.py` |
| `weather_*` | API not integrated | Integrate weather API |

---

## 9. Future Development Plan

### 9.1 Phase 4: Machine Learning Models

**Objective:** Train gradient boosting and neural network models

**Models to Implement:**
1. **XGBoost** - Primary model, excellent for tabular data
2. **LightGBM** - Fast training, good for iteration
3. **CatBoost** - Handles categoricals natively
4. **Neural Network** - Capture non-linear interactions

**Training Strategy:**
```
Training Set: 2021-2022 (8,898 samples)
Validation Set: 2023 (6,104 samples)
Test Set: 2024 (7,416 samples)
```

**Timeline:** Ready to begin immediately

### 9.2 Phase 5: Ensemble & Production

**Objective:** Combine models and deploy for weekly use

**Components:**
1. Meta-learner (stacking)
2. Confidence calibration
3. Prediction intervals
4. Weekly automation pipeline

### 9.3 Data Improvements Needed

| Improvement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| Vegas context linking | High | Low | +5-10% accuracy |
| Weather API integration | Medium | Medium | +2-5% accuracy |
| Injury data | High | High | +5-15% accuracy |
| Real-time updates | Low | High | User experience |

### 9.4 Model Improvements Planned

| Improvement | Description |
|-------------|-------------|
| Position-specific models | Separate models for QB, RB, WR, TE |
| Uncertainty quantification | Predict confidence intervals |
| Matchup features | Defender vs. receiver specific |
| Trend detection | Identify breakout candidates |

---

## 10. Appendices

### 10.1 Appendix A: Complete File Listing

```
nfl_prediction_engine/
├── README.md
├── docs/
│   ├── planning/
│   │   ├── 00_development_plan.md
│   │   ├── 01_data_inventory.md
│   │   ├── 02_feature_engineering_plan.md
│   │   ├── 03_baseline_predictor.md
│   │   └── ...
│   └── STATUS_REPORT.md (this document)
│
├── phase_1/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── database/
│   │   └── nfl_data.db
│   ├── ingestion_scripts/
│   │   ├── __init__.py
│   │   ├── backfill_sleeper_stats.py
│   │   ├── bet_repository.py
│   │   ├── create_defense_table.py
│   │   ├── create_schema.py
│   │   ├── DATABASE_AUDIT_AND_FIX.py
│   │   ├── delete_defense_table.py
│   │   ├── fix_phase1_issues.py
│   │   ├── ingest_betting.py
│   │   ├── ingest_betting_lines.py
│   │   ├── ingest_games.py
│   │   ├── ingest_nfl_data_py.py
│   │   ├── ingest_nflverse.py
│   │   ├── ingest_pbp.py
│   │   ├── ingest_players.py
│   │   ├── ingest_stats_defenders.py
│   │   ├── ingest_stats_defense.py
│   │   ├── ingest_stats_offense.py
│   │   ├── ingest_teams.py
│   │   ├── ingest_weather.py
│   │   ├── link_betting_to_games.py
│   │   ├── load_sample_data.py
│   │   ├── load_season_defense_stats.py
│   │   ├── populate_defender_stats.py
│   │   ├── populate_team_defense_stats.py
│   │   ├── populate_vegas_context.py
│   │   ├── run_master_pipeline.py
│   │   ├── run_master_pipeline_v3.py
│   │   ├── run_pipeline.py
│   │   ├── schema_betting_update.py
│   │   ├── schema_enhancement.py
│   │   ├── schema_pbp.py
│   │   └── settings_patch.py
│   ├── schema/
│   │   └── schema.sql
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── logger.py
│   └── validation/
│       ├── __init__.py
│       └── validate_data.py
│
├── phase_2/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── dev_check.py
│   ├── logging_utils.py
│   ├── player_id_mapping.py
│   ├── schema.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── archetypes.py
│   │   ├── defender_matchup.py
│   │   ├── efficiency.py
│   │   ├── form.py
│   │   ├── ngs.py
│   │   ├── schedule_context.py
│   │   ├── team_context.py
│   │   ├── team_defense.py
│   │   ├── usage.py
│   │   └── weather.py
│   └── pipeline/
│       ├── __init.py
│       ├── build_all.py
│       ├── build_week.py
│       └── validate.py
│
├── phase_3/
│   ├── __init__.py
│   ├── db.py
│   ├── predictors/
│   │   ├── __init__.py
│   │   └── baseline.py
│   └── pipeline/
│       ├── __init__.py
│       └── run_baseline.py
│
├── phase_4/ (planned)
│   ├── models/
│   ├── training/
│   └── evaluation/
│
└── phase_5/ (planned)
    ├── meta_model/
    └── production/
```

### 10.2 Appendix B: Database Schema SQL

Key table definitions:

```sql
CREATE TABLE player_game_features (
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    player_id TEXT,
    player_name TEXT,
    position TEXT,
    team TEXT,
    opponent TEXT,
    -- Labels (what we predict)
    label_targets INTEGER,
    label_receptions INTEGER,
    label_rec_yards REAL,
    label_rec_tds REAL,
    label_carries INTEGER,
    label_rush_yards REAL,
    label_rush_tds REAL,
    label_pass_attempts INTEGER,
    label_pass_completions INTEGER,
    label_pass_yards REAL,
    label_pass_tds REAL,
    label_interceptions INTEGER,
    -- Features (207 columns)
    usage_targets_last1 REAL,
    usage_targets_last3 REAL,
    usage_targets_season_to_date REAL,
    -- ... (200+ more feature columns)
    PRIMARY KEY (season, week, player_id)
);

CREATE TABLE baseline_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT,
    game_id TEXT,
    season INTEGER,
    week INTEGER,
    player_name TEXT,
    position TEXT,
    team TEXT,
    -- Predictions
    targets REAL,
    receptions REAL,
    rec_yards REAL,
    rec_tds REAL,
    carries REAL,
    rush_yards REAL,
    rush_tds REAL,
    pass_attempts REAL,
    completions REAL,
    pass_yards REAL,
    pass_tds REAL,
    interceptions REAL,
    pred_fp_ppr REAL,
    pred_fp_half REAL,
    pred_fp_std REAL,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 10.3 Appendix C: Performance Benchmarks

#### Baseline Model Performance by Position

| Position | Records | Targets MAE | FP PPR MAE | FP Correlation |
|----------|---------|-------------|------------|----------------|
| QB | 1,842 | N/A | 5.21 | 0.52 |
| RB | 4,156 | 0.98 | 4.12 | 0.48 |
| WR | 5,234 | 1.24 | 5.45 | 0.51 |
| TE | 2,288 | 0.89 | 3.78 | 0.54 |

#### Processing Performance

| Operation | Duration | Records/Second |
|-----------|----------|----------------|
| Feature build (full) | ~5 minutes | 97 rec/s |
| Baseline predict (season) | ~30 seconds | 970 rec/s |
| Database query (features) | ~2 seconds | 14,600 rec/s |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-07 | Development Team | Initial release |

---

*End of Status Report*
