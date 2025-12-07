# phase_2/features/__init__.py

"""
Feature modules for Phase 2.

Each module should expose a function of the form:

    build_<name>_features(season: int, week: int) -> pd.DataFrame

which returns a DataFrame keyed by (player_id, game_id) containing
only features that are valid *before* that game is played
(no data leakage from the target week).
"""

from .usage import build_usage_features  # convenience re-export
