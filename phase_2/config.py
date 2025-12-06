# phase_2/config.py

from pathlib import Path
from typing import List

# Repo root = .../NFL_PREDICTION_ENGINE
BASE_DIR = Path(__file__).resolve().parents[1]

# Phase 1 database (canonical location)
# We prefer the phase_1/database copy since that's the "user-facing" path.
DB_PATH = BASE_DIR / "phase_1" / "database" / "nfl_data.db"

# If you ever want to swap DBs (e.g., test vs prod), you can override via env var.
# For now we keep it simple and direct.

# Default seasons to build features for
# You can adjust this as needed or pass explicit seasons into the pipeline.
DEFAULT_SEASONS: List[int] = [2021, 2022, 2023, 2024]

# Table name for the engineered features
FEATURE_TABLE_NAME: str = "player_game_features"

# Chunk size to process games (helps if you ever need to scale up)
GAME_CHUNK_SIZE: int = 256

# Logging defaults
LOG_LEVEL = "INFO"
LOG_NAME = "phase_2_feature_engineering"
