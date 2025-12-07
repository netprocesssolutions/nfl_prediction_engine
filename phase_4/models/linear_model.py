"""
Linear Models for NFL Prediction

Implements Ridge and ElasticNet regression models.
Linear models are:
- Simple and interpretable
- Robust to outliers
- Good baseline for ensemble
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import SAVED_MODELS_DIR


class LinearPredictor:
    """
    Linear regression predictor using Ridge and ElasticNet.

    Trains both model types and can output predictions from either.
    """

    def __init__(
        self,
        name: str = "linear_predictor",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
    ):
        """
        Initialize linear predictor.

        Args:
            name: Model name for saving
            alpha: Regularization strength
            l1_ratio: ElasticNet mixing parameter (0=Ridge, 1=Lasso)
        """
        self.name = name
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # Models for each target
        self.ridge_models: Dict[str, Ridge] = {}
        self.elasticnet_models: Dict[str, ElasticNet] = {}

        # Scalers for feature standardization
        self.scalers: Dict[str, StandardScaler] = {}

        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = True,
    ) -> "LinearPredictor":
        """
        Train Ridge and ElasticNet models for all targets.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            verbose: Print progress

        Returns:
            self
        """
        self.feature_columns = list(X.columns)
        self.target_columns = list(y.columns)

        # Handle missing values
        X_clean = X.fillna(0)

        if verbose:
            print(f"Training Linear Models ({len(self.target_columns)} targets)...")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Samples: {len(X_clean)}")

        for i, target in enumerate(self.target_columns):
            if verbose:
                print(f"  [{i+1}/{len(self.target_columns)}] {target}")

            y_target = y[target].fillna(0).values

            # Scale features for this target
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            self.scalers[target] = scaler

            # Train Ridge
            ridge = Ridge(alpha=self.alpha, random_state=42)
            ridge.fit(X_scaled, y_target)
            self.ridge_models[target] = ridge

            # Train ElasticNet
            elasticnet = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=42,
                max_iter=5000,
            )
            elasticnet.fit(X_scaled, y_target)
            self.elasticnet_models[target] = elasticnet

            # Store feature importance (coefficients)
            importance = pd.DataFrame({
                "feature": self.feature_columns,
                "ridge_coef": np.abs(ridge.coef_),
                "elasticnet_coef": np.abs(elasticnet.coef_),
            })
            importance["avg_importance"] = (
                importance["ridge_coef"] + importance["elasticnet_coef"]
            ) / 2
            importance = importance.sort_values("avg_importance", ascending=False)
            self.feature_importance[target] = importance

        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_clean),
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        }

        if verbose:
            print("Linear model training complete!")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        model_type: str = "ridge",
    ) -> pd.DataFrame:
        """
        Make predictions using specified model type.

        Args:
            X: Feature DataFrame
            model_type: 'ridge' or 'elasticnet'

        Returns:
            DataFrame with predictions
        """
        models = self.ridge_models if model_type == "ridge" else self.elasticnet_models

        if not models:
            raise ValueError("Model not trained. Call fit() first.")

        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        X_clean = X_aligned.fillna(0)

        predictions = {}
        for target in self.target_columns:
            scaler = self.scalers[target]
            X_scaled = scaler.transform(X_clean)

            model = models[target]
            pred = model.predict(X_scaled)

            col_name = target.replace("label_", f"pred_{model_type}_")
            predictions[col_name] = pred

        return pd.DataFrame(predictions, index=X.index)

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from both Ridge and ElasticNet.

        Returns DataFrame with both model predictions.
        """
        ridge_preds = self.predict(X, model_type="ridge")
        elasticnet_preds = self.predict(X, model_type="elasticnet")

        return pd.concat([ridge_preds, elasticnet_preds], axis=1)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        model_type: str = "ridge",
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance.
        """
        predictions = self.predict(X, model_type=model_type)

        results = {}
        for target in self.target_columns:
            pred_col = target.replace("label_", f"pred_{model_type}_")
            y_true = y[target].fillna(0)
            y_pred = predictions[pred_col]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

            results[target] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "correlation": corr,
            }

            if verbose:
                clean_name = target.replace("label_", "")
                print(f"{clean_name:20s} MAE: {mae:6.2f}  R2: {r2:5.2f}  Corr: {corr:5.3f}")

        return results

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SAVED_MODELS_DIR / f"{self.name}_{timestamp}.pkl"

        save_data = {
            "name": self.name,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "ridge_models": self.ridge_models,
            "elasticnet_models": self.elasticnet_models,
            "scalers": self.scalers,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "feature_importance": self.feature_importance,
            "training_metadata": self.training_metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Linear model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "LinearPredictor":
        """Load model from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        predictor = cls(
            name=save_data["name"],
            alpha=save_data["alpha"],
            l1_ratio=save_data["l1_ratio"],
        )
        predictor.ridge_models = save_data["ridge_models"]
        predictor.elasticnet_models = save_data["elasticnet_models"]
        predictor.scalers = save_data["scalers"]
        predictor.feature_columns = save_data["feature_columns"]
        predictor.target_columns = save_data["target_columns"]
        predictor.feature_importance = save_data["feature_importance"]
        predictor.training_metadata = save_data["training_metadata"]

        return predictor
