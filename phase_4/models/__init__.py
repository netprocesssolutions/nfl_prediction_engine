"""
Phase 4 Models

Available models:
- LinearPredictor: Ridge and ElasticNet regression
- GBMPredictor: XGBoost and LightGBM gradient boosting
- MultiModelTrainer: Orchestrates all model types

For ensemble use in Phase 6, use MultiModelTrainer which trains
all model types and outputs predictions from each.
"""

from .linear_model import LinearPredictor
from .gbm_model import GBMPredictor
from .multi_model import MultiModelTrainer

# Keep old NFLPredictor for backwards compatibility
from .xgboost_model import NFLPredictor

__all__ = [
    "LinearPredictor",
    "GBMPredictor",
    "MultiModelTrainer",
    "NFLPredictor",  # Deprecated, use MultiModelTrainer
]
