"""
Configuration settings for NFL Fantasy Prediction Engine - Enhanced

This module centralizes ALL configuration settings including:
- Project paths and directories
- Sleeper API configuration
- The Odds API configuration (NEW)
- Season and team settings
- Odds conversion utilities

SECURITY: API keys should NEVER be hardcoded. Use environment variables.

Author: NFL Fantasy Prediction Engine Team
Version: 2.0 Enhanced
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# =============================================================================
# PROJECT PATHS (from original settings.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATABASE_DIR = PROJECT_ROOT / "database"
LOGS_DIR = PROJECT_ROOT / "logs"
VALIDATION_DIR = PROJECT_ROOT / "validation"

# Database file path
DATABASE_PATH = DATABASE_DIR / "nfl_data.db"

# Log subdirectories
LOG_INGESTION_DIR = LOGS_DIR / "data_ingestion"
LOG_FEATURE_DIR = LOGS_DIR / "feature_engineering"
LOG_TRAINING_DIR = LOGS_DIR / "model_training"
LOG_INFERENCE_DIR = LOGS_DIR / "inference"
LOG_SYSTEM_DIR = LOGS_DIR / "system"

# Log formatting
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

# =============================================================================
# SLEEPER API CONFIGURATION
# =============================================================================
SLEEPER_BASE_URL = "https://api.sleeper.app/v1"

SLEEPER_ENDPOINTS = {
    "players": f"{SLEEPER_BASE_URL}/players/nfl",
    "stats_regular": lambda season, week: f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}/{week}",
    "stats_projections": lambda season, week: f"{SLEEPER_BASE_URL}/projections/nfl/regular/{season}/{week}",
    "nfl_state": f"{SLEEPER_BASE_URL}/state/nfl",
    "user": lambda username: f"{SLEEPER_BASE_URL}/user/{username}",
}

API_TIMEOUT = 30
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2

# =============================================================================
# DATA WINDOW CONFIGURATION
# =============================================================================
ROLLING_WINDOW_SEASONS = 3

# =============================================================================
# SQLITE PRAGMA CONFIGURATION (Phase 1 v2 Section 3.2)
# =============================================================================
SQLITE_PRAGMAS = {
    "foreign_keys": "ON",
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
}

# =============================================================================
# VALIDATION CONFIGURATION (Phase 1 v2 Section 7)
# =============================================================================
VALIDATION_CONFIG = {
    "coverage_sum_tolerance": 0.05,  # Allow 5% tolerance for man + zone = 100%
    "min_snap_count": 0,
    "max_snap_count": 100,
    "min_route_count": 0,
    "alignment_range": (0.0, 1.0),
    "require_season_week": True,
    "require_game_id": True,
}

# =============================================================================
# SEASON CONFIGURATION
# =============================================================================

CURRENT_SEASON = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
CURRENT_WEEK = 13  # Update this weekly or detect automatically

MAX_REGULAR_SEASON_WEEKS = 18
MAX_PLAYOFF_WEEKS = 4

# =============================================================================
# OFFENSIVE POSITIONS
# =============================================================================
OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE"]

# =============================================================================
# DEFENSIVE POSITIONS
# =============================================================================
DEFENSIVE_POSITIONS = ["CB", "S", "LB", "DB", "SS", "FS", "ILB", "OLB", "MLB"]
DEFENSIVE_POSITION_GROUPS = {
    "CB": ["CB", "DB"],
    "S": ["S", "SS", "FS", "DB"],
    "LB": ["LB", "ILB", "OLB", "MLB"],
}

# =============================================================================
# NFL TEAMS STATIC DATA
# =============================================================================
NFL_TEAMS = {
    "ARI": {"name": "Arizona Cardinals", "conference": "NFC", "division": "NFC West"},
    "ATL": {"name": "Atlanta Falcons", "conference": "NFC", "division": "NFC South"},
    "BAL": {"name": "Baltimore Ravens", "conference": "AFC", "division": "AFC North"},
    "BUF": {"name": "Buffalo Bills", "conference": "AFC", "division": "AFC East"},
    "CAR": {"name": "Carolina Panthers", "conference": "NFC", "division": "NFC South"},
    "CHI": {"name": "Chicago Bears", "conference": "NFC", "division": "NFC North"},
    "CIN": {"name": "Cincinnati Bengals", "conference": "AFC", "division": "AFC North"},
    "CLE": {"name": "Cleveland Browns", "conference": "AFC", "division": "AFC North"},
    "DAL": {"name": "Dallas Cowboys", "conference": "NFC", "division": "NFC East"},
    "DEN": {"name": "Denver Broncos", "conference": "AFC", "division": "AFC West"},
    "DET": {"name": "Detroit Lions", "conference": "NFC", "division": "NFC North"},
    "GB": {"name": "Green Bay Packers", "conference": "NFC", "division": "NFC North"},
    "HOU": {"name": "Houston Texans", "conference": "AFC", "division": "AFC South"},
    "IND": {"name": "Indianapolis Colts", "conference": "AFC", "division": "AFC South"},
    "JAX": {"name": "Jacksonville Jaguars", "conference": "AFC", "division": "AFC South"},
    "KC": {"name": "Kansas City Chiefs", "conference": "AFC", "division": "AFC West"},
    "LAC": {"name": "Los Angeles Chargers", "conference": "AFC", "division": "AFC West"},
    "LAR": {"name": "Los Angeles Rams", "conference": "NFC", "division": "NFC West"},
    "LV": {"name": "Las Vegas Raiders", "conference": "AFC", "division": "AFC West"},
    "MIA": {"name": "Miami Dolphins", "conference": "AFC", "division": "AFC East"},
    "MIN": {"name": "Minnesota Vikings", "conference": "NFC", "division": "NFC North"},
    "NE": {"name": "New England Patriots", "conference": "AFC", "division": "AFC East"},
    "NO": {"name": "New Orleans Saints", "conference": "NFC", "division": "NFC South"},
    "NYG": {"name": "New York Giants", "conference": "NFC", "division": "NFC East"},
    "NYJ": {"name": "New York Jets", "conference": "AFC", "division": "AFC East"},
    "PHI": {"name": "Philadelphia Eagles", "conference": "NFC", "division": "NFC East"},
    "PIT": {"name": "Pittsburgh Steelers", "conference": "AFC", "division": "AFC North"},
    "SF": {"name": "San Francisco 49ers", "conference": "NFC", "division": "NFC West"},
    "SEA": {"name": "Seattle Seahawks", "conference": "NFC", "division": "NFC West"},
    "TB": {"name": "Tampa Bay Buccaneers", "conference": "NFC", "division": "NFC South"},
    "TEN": {"name": "Tennessee Titans", "conference": "AFC", "division": "AFC South"},
    "WAS": {"name": "Washington Commanders", "conference": "NFC", "division": "NFC East"},
}

# Team abbreviation mapping for different data sources
TEAM_ABBREVIATION_MAP = {
    # Standard abbreviations (no change needed)
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC",
    "LAC": "LAC", "LAR": "LAR", "LV": "LV", "MIA": "MIA",
    "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SF": "SF",
    "SEA": "SEA", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
    # Alternative abbreviations from various sources
    "GNB": "GB", "GBP": "GB",
    "KAN": "KC", "KCC": "KC",
    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "LA": "LAR", "LAR": "LAR", "RAM": "LAR", "STL": "LAR",
    "NWE": "NE", "NEP": "NE",
    "NOR": "NO", "NOS": "NO",
    "SFO": "SF", "SF49": "SF",
    "TAM": "TB", "TBB": "TB",
    "WAS": "WAS", "WSH": "WAS", "WFT": "WAS",
    "JAC": "JAX", "JAG": "JAX",
    "SDG": "LAC", "SD": "LAC",
}


# =============================================================================
# THE ODDS API CONFIGURATION
# =============================================================================

# Load API key from environment variable (SECURE)
# Set with: export ODDS_API_KEY="your_key_here"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

if not ODDS_API_KEY:
    import warnings
    warnings.warn(
        "ODDS_API_KEY not set. Set it with: export ODDS_API_KEY='your_key_here'\n"
        "On Windows: set ODDS_API_KEY=your_key_here"
    )

# API endpoints
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "americanfootball_nfl"
ODDS_API_REGIONS = "us"
ODDS_API_TIMEOUT = 30
ODDS_API_RETRY_ATTEMPTS = 3
ODDS_API_RETRY_DELAY = 2


def get_odds_api_endpoints() -> Dict[str, str]:
    """Get The Odds API endpoint templates."""
    return {
        "odds": f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds",
        "events": f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/events",
        "event_odds": lambda event_id: f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/events/{event_id}/odds",
        "sports": f"{ODDS_API_BASE_URL}/sports",
    }


# =============================================================================
# BOOKMAKER CONFIGURATION
# =============================================================================

# Primary bookmaker for consistency
ODDS_API_PREFERRED_BOOKMAKER = "fanduel"

# All US bookmakers
ODDS_API_BOOKMAKERS = [
    "fanduel",
    "draftkings", 
    "betmgm",
    "caesars",
    "pointsbetus",
    "betrivers",
    "unibet_us",
    "wynnbet",
    "bovada",
    "betonlineag",
]

# Display names
ODDS_API_BOOKMAKER_NAMES = {
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "pointsbetus": "PointsBet",
    "betrivers": "BetRivers",
    "bovada": "Bovada",
}


# =============================================================================
# MARKET CONFIGURATION
# =============================================================================

# Featured markets (low cost)
ODDS_API_FEATURED_MARKETS = ["h2h", "spreads", "totals"]

# Player prop markets
ODDS_API_PLAYER_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]

# Market to stat key mapping
ODDS_API_MARKET_TO_STAT_KEY = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "game_total",
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
}


# =============================================================================
# TEAM MAPPINGS
# =============================================================================

ODDS_API_TEAM_NAME_TO_ABBREV = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

ABBREV_TO_ODDS_API_TEAM_NAME = {v: k for k, v in ODDS_API_TEAM_NAME_TO_ABBREV.items()}


# =============================================================================
# EDGE BUCKET CONFIGURATION
# =============================================================================

EDGE_BUCKET_THRESHOLDS = {
    "minimal": 1.0,
    "low": 3.0,
    "medium": 5.0,
    "high": 10.0,
    "extreme": float('inf'),
}


def calculate_edge_bucket(expected_edge: float) -> str:
    """Calculate edge bucket from expected edge value."""
    abs_edge = abs(expected_edge) if expected_edge is not None else 0
    
    if abs_edge < EDGE_BUCKET_THRESHOLDS["minimal"]:
        return "minimal"
    elif abs_edge < EDGE_BUCKET_THRESHOLDS["low"]:
        return "low"
    elif abs_edge < EDGE_BUCKET_THRESHOLDS["medium"]:
        return "medium"
    elif abs_edge < EDGE_BUCKET_THRESHOLDS["high"]:
        return "high"
    else:
        return "extreme"


# =============================================================================
# ODDS CONVERSION UTILITIES
# =============================================================================

def american_odds_to_decimal(american_odds: int) -> Optional[float]:
    """Convert American odds to decimal odds."""
    if american_odds is None:
        return None
    if american_odds >= 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_odds_to_implied_probability(american_odds: int) -> Optional[float]:
    """Convert American odds to implied probability."""
    if american_odds is None:
        return None
    if american_odds >= 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def decimal_odds_to_american(decimal_odds: float) -> Optional[int]:
    """Convert decimal odds to American odds."""
    if decimal_odds is None:
        return None
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def calculate_ev(win_probability: float, decimal_odds: float) -> Optional[float]:
    """Calculate expected value as percentage of stake."""
    if win_probability is None or decimal_odds is None:
        return None
    ev = (win_probability * (decimal_odds - 1)) - (1 - win_probability)
    return ev * 100


def calculate_kelly_fraction(win_probability: float, decimal_odds: float) -> Optional[float]:
    """Calculate Kelly Criterion fraction for optimal bet sizing."""
    if win_probability is None or decimal_odds is None:
        return None
    b = decimal_odds - 1
    p = win_probability
    q = 1 - p
    if b <= 0:
        return 0
    return (b * p - q) / b


# =============================================================================
# WEEK ESTIMATION
# =============================================================================

def estimate_nfl_week_from_date(date: datetime) -> tuple:
    """Estimate NFL season and week from a date."""
    year = date.year
    month = date.month
    
    if month >= 9:
        season = year
    elif month <= 2:
        season = year - 1
    else:
        season = year
    
    if month == 9:
        week = max(1, (date.day // 7) + 1)
    elif month == 10:
        week = 4 + (date.day // 7) + 1
    elif month == 11:
        week = 8 + (date.day // 7) + 1
    elif month == 12:
        week = 13 + (date.day // 7) + 1
    elif month == 1:
        week = 17 + (date.day // 7) + 1
    elif month == 2:
        week = 18
    else:
        week = 0
    
    week = max(0, min(week, 22))
    return season, week


# =============================================================================
# DATABASE CONFIGURATION (Additional settings)
# =============================================================================

# DATABASE_PATH is defined at the top of the file as DATABASE_DIR / "nfl_data.db"
DATABASE_WAL_MODE = True


# =============================================================================
# WEATHER API (Free - Open-Meteo)
# =============================================================================

WEATHER_API_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Stadium coordinates (for weather lookup)
STADIUM_COORDINATES = {
    "ARI": (33.5276, -112.2626),  # State Farm Stadium
    "ATL": (33.7553, -84.4006),   # Mercedes-Benz Stadium (dome)
    "BAL": (39.2780, -76.6227),   # M&T Bank Stadium
    "BUF": (42.7738, -78.7870),   # Highmark Stadium
    "CAR": (35.2258, -80.8528),   # Bank of America Stadium
    "CHI": (41.8623, -87.6167),   # Soldier Field
    "CIN": (39.0955, -84.5161),   # Paycor Stadium
    "CLE": (41.5061, -81.6995),   # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945),   # AT&T Stadium (dome)
    "DEN": (39.7439, -105.0201),  # Empower Field
    "DET": (42.3400, -83.0456),   # Ford Field (dome)
    "GB": (44.5013, -88.0622),    # Lambeau Field
    "HOU": (29.6847, -95.4107),   # NRG Stadium (dome)
    "IND": (39.7601, -86.1639),   # Lucas Oil Stadium (dome)
    "JAX": (30.3239, -81.6373),   # EverBank Stadium
    "KC": (39.0489, -94.4839),    # Arrowhead Stadium
    "LAC": (33.9535, -118.3390),  # SoFi Stadium (dome)
    "LAR": (33.9535, -118.3390),  # SoFi Stadium (dome)
    "LV": (36.0909, -115.1833),   # Allegiant Stadium (dome)
    "MIA": (25.9580, -80.2389),   # Hard Rock Stadium
    "MIN": (44.9736, -93.2575),   # U.S. Bank Stadium (dome)
    "NE": (42.0909, -71.2643),    # Gillette Stadium
    "NO": (29.9511, -90.0812),    # Caesars Superdome (dome)
    "NYG": (40.8128, -74.0742),   # MetLife Stadium
    "NYJ": (40.8128, -74.0742),   # MetLife Stadium
    "PHI": (39.9008, -75.1675),   # Lincoln Financial Field
    "PIT": (40.4468, -80.0158),   # Acrisure Stadium
    "SF": (37.4032, -121.9698),   # Levi's Stadium
    "SEA": (47.5952, -122.3316),  # Lumen Field
    "TB": (27.9759, -82.5033),    # Raymond James Stadium
    "TEN": (36.1665, -86.7713),   # Nissan Stadium
    "WAS": (38.9076, -76.8645),   # Northwest Stadium
}

# Dome stadiums (no weather impact)
DOME_STADIUMS = ["ATL", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "LV", "MIN", "NO"]


# =============================================================================
# DATA VERSION UTILITIES
# =============================================================================

def generate_data_version(season: int, week: int) -> str:
    """
    Generate a data version string in the format season_week.
    
    Per Phase 1 v2 Section 8:
    Each ingestion cycle creates a version label: data_version = season_week
    Examples: 2025_01, 2025_06, 2026_14
    
    Args:
        season: NFL season year
        week: NFL week number
    
    Returns:
        Version string in format "YYYY_WW"
    """
    return f"{season}_{week:02d}"


def parse_data_version(version_str: str) -> tuple:
    """
    Parse a data version string back to season and week.
    
    Args:
        version_str: Version string in format "YYYY_WW"
    
    Returns:
        Tuple of (season, week)
    """
    parts = version_str.split("_")
    return int(parts[0]), int(parts[1])


if __name__ == "__main__":
    print("NFL Fantasy Prediction Engine Settings")
    print(f"Current Season: {CURRENT_SEASON}")
    print(f"Odds API Key Set: {'Yes' if ODDS_API_KEY else 'NO - SET ENVIRONMENT VARIABLE!'}")
    print(f"Preferred Bookmaker: {ODDS_API_PREFERRED_BOOKMAKER}")
