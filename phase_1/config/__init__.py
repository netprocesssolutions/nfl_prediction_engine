"""
Configuration package for NFL Fantasy Prediction Engine.

This package contains all configuration settings.
"""

# Import commonly used items from settings
# Using lazy imports to avoid circular dependency issues
from .settings import (
    # Paths
    PROJECT_ROOT,
    DATABASE_DIR,
    DATABASE_PATH,
    LOGS_DIR,
    VALIDATION_DIR,
    
    # Sleeper API
    SLEEPER_BASE_URL,
    SLEEPER_ENDPOINTS,
    API_TIMEOUT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    
    # Season config
    CURRENT_SEASON,
    CURRENT_WEEK,
    MAX_REGULAR_SEASON_WEEKS,
    ROLLING_WINDOW_SEASONS,
    
    # SQLite config
    SQLITE_PRAGMAS,
    
    # Positions
    OFFENSIVE_POSITIONS,
    DEFENSIVE_POSITIONS,
    DEFENSIVE_POSITION_GROUPS,
    
    # Teams
    NFL_TEAMS,
    TEAM_ABBREVIATION_MAP,
    
    # Odds API
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    ODDS_API_PREFERRED_BOOKMAKER,
)

__all__ = [
    "PROJECT_ROOT",
    "DATABASE_DIR", 
    "DATABASE_PATH",
    "LOGS_DIR",
    "VALIDATION_DIR",
    "SLEEPER_BASE_URL",
    "SLEEPER_ENDPOINTS",
    "API_TIMEOUT",
    "API_RETRY_ATTEMPTS",
    "API_RETRY_DELAY",
    "CURRENT_SEASON",
    "CURRENT_WEEK",
    "MAX_REGULAR_SEASON_WEEKS",
    "ROLLING_WINDOW_SEASONS",
    "SQLITE_PRAGMAS",
    "OFFENSIVE_POSITIONS",
    "DEFENSIVE_POSITIONS",
    "DEFENSIVE_POSITION_GROUPS",
    "NFL_TEAMS",
    "TEAM_ABBREVIATION_MAP",
    "ODDS_API_KEY",
    "ODDS_API_BASE_URL",
    "ODDS_API_PREFERRED_BOOKMAKER",
]
