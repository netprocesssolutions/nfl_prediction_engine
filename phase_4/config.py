"""
Phase 4 Configuration

Contains all settings for model training, features, and targets.
"""

import os
from pathlib import Path

# Paths
PHASE_4_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_4_DIR.parent
DB_PATH = PROJECT_ROOT / "phase_1" / "database" / "nfl_data.db"
SAVED_MODELS_DIR = PHASE_4_DIR / "saved_models"

# Ensure saved_models directory exists
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# Target columns (what we predict)
TARGETS = [
    "label_targets",
    "label_receptions",
    "label_rec_yards",
    "label_rec_tds",
    "label_carries",
    "label_rush_yards",
    "label_rush_tds",
    "label_pass_attempts",
    "label_pass_completions",
    "label_pass_yards",
    "label_pass_tds",
    "label_interceptions",
]

# Feature column prefixes to use
FEATURE_PREFIXES = [
    "usage_",
    "eff_",
    "oppdef_",
    "ngs_",
    "sched_",
    "pbp_",  # Play-by-play based features (EPA, air yards, CPOE, success rate)
    # "ctx_",  # Vegas context - not populated yet
    # "weather_",  # Weather - not populated yet
]

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "home_team",
    "away_team",
    "is_home",
    "is_playoff",
] + TARGETS  # Don't use labels as features!

# Position groups for position-specific models
POSITION_GROUPS = {
    "QB": ["QB"],
    "RB": ["RB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "FLEX": ["RB", "WR", "TE"],  # Combined skill positions
}

# Default XGBoost parameters (tuned for NFL data)
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,  # Use all CPU cores
}

# Training configuration
TRAIN_CONFIG = {
    "test_size": 0.2,
    "validation_seasons": [2023],  # Hold out for validation
    "test_seasons": [2024],  # Hold out for final testing
    "min_games_played": 3,  # Minimum games to include player
    "early_stopping_rounds": 20,
}

# Fantasy point scoring (for calculating fantasy points from predictions)
SCORING = {
    "ppr": {
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "receptions": 1.0,
        "rec_yards": 0.1,
        "rec_tds": 6,
    },
    "half_ppr": {
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "receptions": 0.5,
        "rec_yards": 0.1,
        "rec_tds": 6,
    },
    "standard": {
        "pass_yards": 0.04,
        "pass_tds": 4,
        "interceptions": -2,
        "rush_yards": 0.1,
        "rush_tds": 6,
        "receptions": 0.0,
        "rec_yards": 0.1,
        "rec_tds": 6,
    },
}


def calculate_fantasy_points(predictions: dict, scoring_type: str = "ppr") -> float:
    """Calculate fantasy points from predicted stats."""
    scoring = SCORING[scoring_type]
    fp = 0.0

    # Passing
    fp += predictions.get("pass_yards", 0) * scoring["pass_yards"]
    fp += predictions.get("pass_tds", 0) * scoring["pass_tds"]
    fp += predictions.get("interceptions", 0) * scoring["interceptions"]

    # Rushing
    fp += predictions.get("rush_yards", 0) * scoring["rush_yards"]
    fp += predictions.get("rush_tds", 0) * scoring["rush_tds"]

    # Receiving
    fp += predictions.get("receptions", 0) * scoring["receptions"]
    fp += predictions.get("rec_yards", 0) * scoring["rec_yards"]
    fp += predictions.get("rec_tds", 0) * scoring["rec_tds"]

    return fp
