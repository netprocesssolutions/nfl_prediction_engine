"""
Phase 4: Machine Learning Models

This phase trains gradient boosting models on the engineered features
from Phase 2 to predict NFL player statistics.

Key components:
- models/: Model definitions (XGBoost, LightGBM)
- training/: Training pipelines
- evaluation/: Model evaluation and metrics
- saved_models/: Persisted trained models

Usage:
    # Train models
    python -m phase_4.training.train --seasons 2021 2022 2023

    # Make predictions
    python -m phase_4.predict --season 2024 --week 10

    # Evaluate model
    python -m phase_4.evaluation.evaluate --season 2024
"""

__version__ = "1.0.0"
