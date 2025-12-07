"""
XGBoost-based NFL Predictor

This module provides a multi-target prediction model using XGBoost
for NFL player statistics prediction.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import XGBOOST_PARAMS, SAVED_MODELS_DIR, calculate_fantasy_points


class NFLPredictor:
    """
    Multi-target NFL statistics predictor using XGBoost.

    Trains separate models for each target statistic (targets, yards, TDs, etc.)
    and combines them for fantasy point predictions.
    """

    def __init__(
        self,
        name: str = "nfl_predictor",
        params: Optional[Dict] = None,
        positions: Optional[List[str]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            name: Model name (used for saving/loading)
            params: XGBoost parameters (uses defaults if not provided)
            positions: List of positions this model handles
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.name = name
        self.params = params or XGBOOST_PARAMS.copy()
        self.positions = positions
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        eval_set: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ) -> "NFLPredictor":
        """
        Train models for all target columns.

        Args:
            X: Feature DataFrame
            y: Target DataFrame (multiple columns)
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress

        Returns:
            self
        """
        self.feature_columns = list(X.columns)
        self.target_columns = list(y.columns)

        # Handle missing values
        X_clean = X.fillna(0)

        if verbose:
            print(f"Training {len(self.target_columns)} models...")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Samples: {len(X_clean)}")

        for i, target in enumerate(self.target_columns):
            if verbose:
                print(f"\n[{i+1}/{len(self.target_columns)}] Training: {target}")

            # Get target values
            y_target = y[target].fillna(0)

            # Create model
            model = xgb.XGBRegressor(**self.params)

            # Prepare eval set if provided
            fit_params = {}
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val_clean = X_val.fillna(0)
                y_val_target = y_val[target].fillna(0)
                fit_params["eval_set"] = [(X_val_clean, y_val_target)]
                fit_params["verbose"] = False

            # Train
            model.fit(X_clean, y_target, **fit_params)

            # Store model
            self.models[target] = model

            # Store feature importance
            importance = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            self.feature_importance[target] = importance

            if verbose:
                # Show top features
                top_features = importance.head(5)
                print(f"  Top features: {', '.join(top_features['feature'].tolist())}")

        # Store training metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_clean),
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
            "positions": self.positions,
            "params": self.params,
        }

        if verbose:
            print(f"\nTraining complete!")

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all targets.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with predictions for each target
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")

        # Ensure columns match training
        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        X_clean = X_aligned.fillna(0)

        predictions = {}
        for target, model in self.models.items():
            pred = model.predict(X_clean)
            # Clean up prediction column name (remove 'label_' prefix)
            col_name = target.replace("label_", "pred_")
            predictions[col_name] = pred

        return pd.DataFrame(predictions, index=X.index)

    def predict_with_fantasy_points(
        self,
        X: pd.DataFrame,
        scoring_type: str = "ppr",
    ) -> pd.DataFrame:
        """
        Make predictions and calculate fantasy points.

        Args:
            X: Feature DataFrame
            scoring_type: Fantasy scoring type ('ppr', 'half_ppr', 'standard')

        Returns:
            DataFrame with predictions and fantasy points
        """
        predictions = self.predict(X)

        # Calculate fantasy points for each row
        fp_values = []
        for idx in predictions.index:
            row = predictions.loc[idx]
            stats = {
                "pass_yards": row.get("pred_pass_yards", 0),
                "pass_tds": row.get("pred_pass_tds", 0),
                "interceptions": row.get("pred_interceptions", 0),
                "rush_yards": row.get("pred_rush_yards", 0),
                "rush_tds": row.get("pred_rush_tds", 0),
                "receptions": row.get("pred_receptions", 0),
                "rec_yards": row.get("pred_rec_yards", 0),
                "rec_tds": row.get("pred_rec_tds", 0),
            }
            fp_values.append(calculate_fantasy_points(stats, scoring_type))

        predictions[f"pred_fp_{scoring_type}"] = fp_values

        return predictions

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature DataFrame
            y: True target DataFrame
            verbose: Print results

        Returns:
            Dictionary of metrics for each target
        """
        predictions = self.predict(X)

        results = {}
        for target in self.target_columns:
            pred_col = target.replace("label_", "pred_")
            y_true = y[target].fillna(0)
            y_pred = predictions[pred_col]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            # Correlation
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

            results[target] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "correlation": corr,
            }

            if verbose:
                clean_name = target.replace("label_", "")
                print(f"{clean_name:20s} MAE: {mae:6.2f}  RMSE: {rmse:6.2f}  R2: {r2:5.2f}  Corr: {corr:5.3f}")

        return results

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.

        Args:
            path: Optional path (uses default if not provided)

        Returns:
            Path where model was saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SAVED_MODELS_DIR / f"{self.name}_{timestamp}.pkl"

        # Save everything
        save_data = {
            "name": self.name,
            "params": self.params,
            "positions": self.positions,
            "models": self.models,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "feature_importance": self.feature_importance,
            "training_metadata": self.training_metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        # Also save metadata as JSON for easy inspection
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump({
                "name": self.name,
                "positions": self.positions,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "training_metadata": self.training_metadata,
            }, f, indent=2, default=str)

        print(f"Model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "NFLPredictor":
        """
        Load a trained model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded NFLPredictor
        """
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        predictor = cls(
            name=save_data["name"],
            params=save_data["params"],
            positions=save_data["positions"],
        )
        predictor.models = save_data["models"]
        predictor.feature_columns = save_data["feature_columns"]
        predictor.target_columns = save_data["target_columns"]
        predictor.feature_importance = save_data["feature_importance"]
        predictor.training_metadata = save_data["training_metadata"]

        return predictor

    def get_top_features(self, target: str, n: int = 20) -> pd.DataFrame:
        """Get top N most important features for a target."""
        if target not in self.feature_importance:
            raise ValueError(f"Unknown target: {target}")
        return self.feature_importance[target].head(n)
