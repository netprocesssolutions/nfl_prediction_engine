# phase_2/pipeline/__init__.py

"""
Pipeline orchestration for Phase 2 feature engineering.

Key entrypoint:
    build_features_for_week(season: int, week: int, persist: bool = False)

Later we'll add:
    - run_phase_2() to loop over all weeks/seasons
    - validation helpers
"""

from .build_week import build_features_for_week
