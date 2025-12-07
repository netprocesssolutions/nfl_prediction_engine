# phase_3/__init__.py
"""
Phase 3 - Baseline Predictor

This phase creates a deterministic, rule-based baseline predictor that:
1. Produces stable, interpretable predictions
2. Acts as a fallback when ML models are unstable
3. Serves as a base learner input to the ensemble meta-model

The baseline predictor uses features from Phase 2 and applies:
- Long-term player averages
- Short-term form adjustments (exponentially weighted)
- Opponent team-defense adjustments
- Usage share projections
- Stability constraints
"""

__version__ = "3.0.0"
