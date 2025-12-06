# phase_2/__init__.py

"""
Phase 2 - Feature Engineering

This package builds the player_game_features table from the Phase 1 database.
Each row corresponds to a single (player, game) pair and contains:
- Identifiers and metadata (season, week, team, opponent, etc.)
- Engineered features (usage, efficiency, form, matchup, archetypes, etc.)
- Label columns (e.g., fantasy-relevant stats) for supervised learning.

Usage:
    from phase_2.pipeline.build_all import run_phase_2

Design goals:
- No data leakage (features for week W only use data from weeks < W).
- Structured by feature family (usage, efficiency, team context, etc.).
- Re-runnable as new weeks are ingested in Phase 1.
"""
