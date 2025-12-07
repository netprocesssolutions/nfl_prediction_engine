"""
Gradient Boosting Models for NFL Prediction

Implements XGBoost and LightGBM models.
Tree-based models:
- Handle nonlinear relationships
- Capture feature interactions
- Handle missing values internally
- Strong out-of-box performance
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from ..config import SAVED_MODELS_DIR, XGBOOST_PARAMS


# LightGBM default parameters
LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


class GBMPredictor:
    """
    Gradient Boosting predictor using XGBoost and LightGBM.

    Trains both model types (when available) for ensemble use.
    """

    def __init__(
        self,
        name: str = "gbm_predictor",
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
    ):
        """
        Initialize GBM predictor.

        Args:
            name: Model name for saving
            xgb_params: XGBoost parameters
            lgb_params: LightGBM parameters
        """
        self.name = name
        self.xgb_params = xgb_params or XGBOOST_PARAMS.copy()
        self.lgb_params = lgb_params or LIGHTGBM_PARAMS.copy()

        # Models for each target
        self.xgb_models: Dict[str, Any] = {}
        self.lgb_models: Dict[str, Any] = {}

        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.training_metadata: Dict[str, Any] = {}

        # Check availability
        self.has_xgboost = HAS_XGBOOST
        self.has_lightgbm = HAS_LIGHTGBM

        if not self.has_xgboost and not self.has_lightgbm:
            raise ImportError(
                "Neither XGBoost nor LightGBM installed. "
                "Install with: pip install xgboost lightgbm"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        eval_set: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        verbose: bool = True,
    ) -> "GBMPredictor":
        """
        Train XGBoost and LightGBM models for all targets.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            eval_set: Optional (X_val, y_val) for early stopping
            verbose: Print progress

        Returns:
            self
        """
        self.feature_columns = list(X.columns)
        self.target_columns = list(y.columns)

        X_clean = X.fillna(0)

        models_to_train = []
        if self.has_xgboost:
            models_to_train.append("xgboost")
        if self.has_lightgbm:
            models_to_train.append("lightgbm")

        if verbose:
            print(f"Training GBM Models: {models_to_train}")
            print(f"Targets: {len(self.target_columns)}")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Samples: {len(X_clean)}")

        for i, target in enumerate(self.target_columns):
            if verbose:
                print(f"\n[{i+1}/{len(self.target_columns)}] Training: {target}")

            y_target = y[target].fillna(0)

            # Prepare validation set
            fit_params_xgb = {}
            fit_params_lgb = {}
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val_clean = X_val.fillna(0)
                y_val_target = y_val[target].fillna(0)
                fit_params_xgb["eval_set"] = [(X_val_clean, y_val_target)]
                fit_params_xgb["verbose"] = False
                fit_params_lgb["eval_set"] = [(X_val_clean, y_val_target)]

            # Train XGBoost
            if self.has_xgboost:
                xgb_model = xgb.XGBRegressor(**self.xgb_params)
                xgb_model.fit(X_clean, y_target, **fit_params_xgb)
                self.xgb_models[target] = xgb_model

                if verbose:
                    print(f"    XGBoost trained")

            # Train LightGBM
            if self.has_lightgbm:
                lgb_model = lgb.LGBMRegressor(**self.lgb_params)
                # Suppress LightGBM warnings during fit
                lgb_model.fit(
                    X_clean, y_target,
                    eval_set=fit_params_lgb.get("eval_set"),
                )
                self.lgb_models[target] = lgb_model

                if verbose:
                    print(f"    LightGBM trained")

            # Store feature importance (average of available models)
            importance_data = {"feature": self.feature_columns}

            if self.has_xgboost and target in self.xgb_models:
                importance_data["xgb_importance"] = self.xgb_models[target].feature_importances_

            if self.has_lightgbm and target in self.lgb_models:
                importance_data["lgb_importance"] = self.lgb_models[target].feature_importances_

            importance = pd.DataFrame(importance_data)

            # Calculate average importance
            imp_cols = [c for c in importance.columns if c.endswith("_importance")]
            if imp_cols:
                importance["avg_importance"] = importance[imp_cols].mean(axis=1)
                importance = importance.sort_values("avg_importance", ascending=False)

            self.feature_importance[target] = importance

        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_clean),
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
            "models_trained": models_to_train,
            "xgb_params": self.xgb_params if self.has_xgboost else None,
            "lgb_params": self.lgb_params if self.has_lightgbm else None,
        }

        if verbose:
            print("\nGBM model training complete!")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        model_type: str = "xgboost",
    ) -> pd.DataFrame:
        """
        Make predictions using specified model type.

        Args:
            X: Feature DataFrame
            model_type: 'xgboost' or 'lightgbm'

        Returns:
            DataFrame with predictions
        """
        if model_type == "xgboost":
            if not self.has_xgboost:
                raise ValueError("XGBoost not available")
            models = self.xgb_models
            prefix = "xgb"
        else:
            if not self.has_lightgbm:
                raise ValueError("LightGBM not available")
            models = self.lgb_models
            prefix = "lgb"

        if not models:
            raise ValueError(f"No {model_type} models trained")

        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        X_clean = X_aligned.fillna(0)

        predictions = {}
        for target in self.target_columns:
            if target not in models:
                continue

            model = models[target]
            pred = model.predict(X_clean)

            col_name = target.replace("label_", f"pred_{prefix}_")
            predictions[col_name] = pred

        return pd.DataFrame(predictions, index=X.index)

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all available GBM models.

        Returns DataFrame with all model predictions.
        """
        dfs = []

        if self.has_xgboost and self.xgb_models:
            dfs.append(self.predict(X, model_type="xgboost"))

        if self.has_lightgbm and self.lgb_models:
            dfs.append(self.predict(X, model_type="lightgbm"))

        if not dfs:
            raise ValueError("No models trained")

        return pd.concat(dfs, axis=1)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        model_type: str = "xgboost",
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance."""
        prefix = "xgb" if model_type == "xgboost" else "lgb"
        predictions = self.predict(X, model_type=model_type)

        results = {}
        for target in self.target_columns:
            pred_col = target.replace("label_", f"pred_{prefix}_")
            if pred_col not in predictions.columns:
                continue

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
            "xgb_params": self.xgb_params,
            "lgb_params": self.lgb_params,
            "xgb_models": self.xgb_models,
            "lgb_models": self.lgb_models,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "feature_importance": self.feature_importance,
            "training_metadata": self.training_metadata,
            "has_xgboost": self.has_xgboost,
            "has_lightgbm": self.has_lightgbm,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"GBM model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "GBMPredictor":
        """Load model from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        predictor = cls(
            name=save_data["name"],
            xgb_params=save_data["xgb_params"],
            lgb_params=save_data["lgb_params"],
        )
        predictor.xgb_models = save_data["xgb_models"]
        predictor.lgb_models = save_data["lgb_models"]
        predictor.feature_columns = save_data["feature_columns"]
        predictor.target_columns = save_data["target_columns"]
        predictor.feature_importance = save_data["feature_importance"]
        predictor.training_metadata = save_data["training_metadata"]
        predictor.has_xgboost = save_data.get("has_xgboost", HAS_XGBOOST)
        predictor.has_lightgbm = save_data.get("has_lightgbm", HAS_LIGHTGBM)

        return predictor

    def get_top_features(self, target: str, n: int = 20) -> pd.DataFrame:
        """Get top N most important features for a target."""
        if target not in self.feature_importance:
            raise ValueError(f"Unknown target: {target}")
        return self.feature_importance[target].head(n)
