"""
PATCH FOR settings.py - Add LA → LAR mapping

Find line 166-167 in settings.py (the LVR/LAR section):

BEFORE:
    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "LAR": "LAR", "RAM": "LAR", "STL": "LAR",

AFTER:
    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "LA": "LAR", "LAR": "LAR", "RAM": "LAR", "STL": "LAR",

This single addition ("LA": "LAR") fixes the nflreadpy team code mismatch.
"""

# Here's the corrected TEAM_ABBREVIATION_MAP section to copy:

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
    "LA": "LAR", "LAR": "LAR", "RAM": "LAR", "STL": "LAR",  # <-- ADDED "LA": "LAR"
    "NWE": "NE", "NEP": "NE",
    "NOR": "NO", "NOS": "NO",
    "SFO": "SF", "SF49": "SF",
    "TAM": "TB", "TBB": "TB",
    "WAS": "WAS", "WSH": "WAS", "WFT": "WAS",
    "JAC": "JAX", "JAG": "JAX",
    "SDG": "LAC", "SD": "LAC",
}
