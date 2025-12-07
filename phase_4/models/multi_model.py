"""
Multi-Model Trainer for NFL Prediction

Orchestrates training of multiple model types:
- Linear Models (Ridge, ElasticNet)
- Gradient Boosting (XGBoost, LightGBM)

All predictions are saved for ensemble use in Phase 6.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

from .linear_model import LinearPredictor
from .gbm_model import GBMPredictor
from ..config import SAVED_MODELS_DIR, calculate_fantasy_points


class MultiModelTrainer:
    """
    Orchestrates training of all model types for NFL prediction.

    This is the main interface for Phase 4 training.
    """

    def __init__(
        self,
        name: str = "nfl_multi_model",
        version: Optional[str] = None,
    ):
        """
        Initialize multi-model trainer.

        Args:
            name: Base name for saved models
            version: Version string (auto-generated if not provided)
        """
        self.name = name
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Model instances
        self.linear_model: Optional[LinearPredictor] = None
        self.gbm_model: Optional[GBMPredictor] = None

        # Training state
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> "MultiModelTrainer":
        """
        Train all model types.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print progress

        Returns:
            self
        """
        self.feature_columns = list(X_train.columns)
        self.target_columns = list(y_train.columns)

        if verbose:
            print("=" * 60)
            print(f"NFL Multi-Model Training - Version: {self.version}")
            print("=" * 60)
            print(f"Training samples: {len(X_train):,}")
            print(f"Validation samples: {len(X_val) if X_val is not None else 0:,}")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Targets: {len(self.target_columns)}")
            print()

        # 1. Train Linear Models
        if verbose:
            print("-" * 60)
            print("Training Linear Models (Ridge + ElasticNet)")
            print("-" * 60)

        self.linear_model = LinearPredictor(name=f"{self.name}_linear")
        self.linear_model.fit(X_train, y_train, verbose=verbose)

        if X_val is not None and y_val is not None and verbose:
            print("\nLinear Model Validation (Ridge):")
            self.linear_model.evaluate(X_val, y_val, model_type="ridge", verbose=True)

        # 2. Train GBM Models
        if verbose:
            print()
            print("-" * 60)
            print("Training Gradient Boosting Models (XGBoost + LightGBM)")
            print("-" * 60)

        eval_set = (X_val, y_val) if X_val is not None else None

        self.gbm_model = GBMPredictor(name=f"{self.name}_gbm")
        self.gbm_model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

        if X_val is not None and y_val is not None and verbose:
            print("\nGBM Model Validation (XGBoost):")
            self.gbm_model.evaluate(X_val, y_val, model_type="xgboost", verbose=True)

            if self.gbm_model.has_lightgbm:
                print("\nGBM Model Validation (LightGBM):")
                self.gbm_model.evaluate(X_val, y_val, model_type="lightgbm", verbose=True)

        # Store metadata
        self.training_metadata = {
            "version": self.version,
            "trained_at": datetime.now().isoformat(),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val) if X_val is not None else 0,
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
            "models_trained": self._get_trained_models_list(),
        }

        self.is_trained = True

        if verbose:
            print()
            print("=" * 60)
            print("Multi-Model Training Complete!")
            print(f"Models trained: {', '.join(self.training_metadata['models_trained'])}")
            print("=" * 60)

        return self

    def _get_trained_models_list(self) -> List[str]:
        """Get list of trained model names."""
        models = []
        if self.linear_model:
            models.extend(["ridge", "elasticnet"])
        if self.gbm_model:
            if self.gbm_model.has_xgboost:
                models.append("xgboost")
            if self.gbm_model.has_lightgbm:
                models.append("lightgbm")
        return models

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all trained models.

        Returns DataFrame with predictions from each model type.
        Columns are named: pred_{model}_{stat}
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call fit() first.")

        predictions = []

        # Linear model predictions
        if self.linear_model:
            linear_preds = self.linear_model.predict_all(X)
            predictions.append(linear_preds)

        # GBM model predictions
        if self.gbm_model:
            gbm_preds = self.gbm_model.predict_all(X)
            predictions.append(gbm_preds)

        if not predictions:
            raise ValueError("No predictions generated")

        return pd.concat(predictions, axis=1)

    def predict_with_ensemble(
        self,
        X: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Get ensemble predictions by averaging all models.

        Args:
            X: Feature DataFrame
            weights: Optional model weights (default: equal weights)

        Returns:
            DataFrame with ensemble predictions
        """
        all_preds = self.predict(X)

        # Default equal weights
        if weights is None:
            weights = {model: 1.0 for model in self._get_trained_models_list()}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate ensemble for each target
        ensemble_preds = {}
        for target in self.target_columns:
            stat = target.replace("label_", "")
            weighted_sum = np.zeros(len(X))
            total_w = 0

            # Ridge
            col = f"pred_ridge_{stat}"
            if col in all_preds.columns and "ridge" in weights:
                weighted_sum += all_preds[col].fillna(0) * weights["ridge"]
                total_w += weights["ridge"]

            # ElasticNet
            col = f"pred_elasticnet_{stat}"
            if col in all_preds.columns and "elasticnet" in weights:
                weighted_sum += all_preds[col].fillna(0) * weights["elasticnet"]
                total_w += weights["elasticnet"]

            # XGBoost
            col = f"pred_xgb_{stat}"
            if col in all_preds.columns and "xgboost" in weights:
                weighted_sum += all_preds[col].fillna(0) * weights["xgboost"]
                total_w += weights["xgboost"]

            # LightGBM
            col = f"pred_lgb_{stat}"
            if col in all_preds.columns and "lightgbm" in weights:
                weighted_sum += all_preds[col].fillna(0) * weights["lightgbm"]
                total_w += weights["lightgbm"]

            if total_w > 0:
                ensemble_preds[f"pred_ensemble_{stat}"] = weighted_sum / total_w

        return pd.DataFrame(ensemble_preds, index=X.index)

    def predict_fantasy_points(
        self,
        X: pd.DataFrame,
        scoring_type: str = "ppr",
        use_ensemble: bool = True,
    ) -> pd.DataFrame:
        """
        Get fantasy point predictions.

        Args:
            X: Feature DataFrame
            scoring_type: 'ppr', 'half_ppr', or 'standard'
            use_ensemble: Use ensemble (True) or all individual models (False)

        Returns:
            DataFrame with fantasy point predictions
        """
        if use_ensemble:
            preds = self.predict_with_ensemble(X)
            prefix = "pred_ensemble_"
        else:
            preds = self.predict(X)
            # Use XGBoost as primary if available
            prefix = "pred_xgb_" if "pred_xgb_targets" in preds.columns else "pred_ridge_"

        # Calculate fantasy points for each row
        fp_values = []
        for idx in preds.index:
            row = preds.loc[idx]
            stats = {
                "pass_yards": row.get(f"{prefix}pass_yards", 0),
                "pass_tds": row.get(f"{prefix}pass_tds", 0),
                "interceptions": row.get(f"{prefix}interceptions", 0),
                "rush_yards": row.get(f"{prefix}rush_yards", 0),
                "rush_tds": row.get(f"{prefix}rush_tds", 0),
                "receptions": row.get(f"{prefix}receptions", 0),
                "rec_yards": row.get(f"{prefix}rec_yards", 0),
                "rec_tds": row.get(f"{prefix}rec_tds", 0),
            }
            fp_values.append(calculate_fantasy_points(stats, scoring_type))

        preds[f"pred_fp_{scoring_type}"] = fp_values

        return preds

    def evaluate_all(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate all models.

        Returns nested dict: model -> target -> metric -> value
        """
        results = {}

        if verbose:
            print("\n" + "=" * 60)
            print("Model Evaluation Results")
            print("=" * 60)

        # Linear models
        if self.linear_model:
            if verbose:
                print("\n--- Ridge Regression ---")
            results["ridge"] = self.linear_model.evaluate(X, y, "ridge", verbose)

            if verbose:
                print("\n--- ElasticNet ---")
            results["elasticnet"] = self.linear_model.evaluate(X, y, "elasticnet", verbose)

        # GBM models
        if self.gbm_model:
            if self.gbm_model.has_xgboost:
                if verbose:
                    print("\n--- XGBoost ---")
                results["xgboost"] = self.gbm_model.evaluate(X, y, "xgboost", verbose)

            if self.gbm_model.has_lightgbm:
                if verbose:
                    print("\n--- LightGBM ---")
                results["lightgbm"] = self.gbm_model.evaluate(X, y, "lightgbm", verbose)

        return results

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save all models to disk.

        Creates a directory with all model artifacts.
        """
        if path is None:
            path = SAVED_MODELS_DIR / f"{self.name}_{self.version}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save individual models
        if self.linear_model:
            self.linear_model.save(path / "linear_model.pkl")

        if self.gbm_model:
            self.gbm_model.save(path / "gbm_model.pkl")

        # Save master metadata
        metadata = {
            "name": self.name,
            "version": self.version,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save master pickle
        with open(path / "multi_model.pkl", "wb") as f:
            pickle.dump({
                "name": self.name,
                "version": self.version,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "training_metadata": self.training_metadata,
                "is_trained": self.is_trained,
            }, f)

        print(f"\nMulti-model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "MultiModelTrainer":
        """
        Load all models from disk.
        """
        path = Path(path)

        # Load metadata
        with open(path / "multi_model.pkl", "rb") as f:
            data = pickle.load(f)

        trainer = cls(name=data["name"], version=data["version"])
        trainer.feature_columns = data["feature_columns"]
        trainer.target_columns = data["target_columns"]
        trainer.training_metadata = data["training_metadata"]
        trainer.is_trained = data["is_trained"]

        # Load individual models
        linear_path = path / "linear_model.pkl"
        if linear_path.exists():
            trainer.linear_model = LinearPredictor.load(linear_path)

        gbm_path = path / "gbm_model.pkl"
        if gbm_path.exists():
            trainer.gbm_model = GBMPredictor.load(gbm_path)

        return trainer

    def get_feature_importance(self, target: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get combined feature importance across all models.
        """
        importance_dfs = []

        if self.linear_model and target in self.linear_model.feature_importance:
            df = self.linear_model.feature_importance[target][["feature", "avg_importance"]].copy()
            df.columns = ["feature", "linear_importance"]
            importance_dfs.append(df)

        if self.gbm_model and target in self.gbm_model.feature_importance:
            df = self.gbm_model.feature_importance[target][["feature", "avg_importance"]].copy()
            df.columns = ["feature", "gbm_importance"]
            importance_dfs.append(df)

        if not importance_dfs:
            return pd.DataFrame()

        # Merge all importance scores
        result = importance_dfs[0]
        for df in importance_dfs[1:]:
            result = result.merge(df, on="feature", how="outer")

        # Calculate overall importance
        imp_cols = [c for c in result.columns if c.endswith("_importance")]
        result["overall_importance"] = result[imp_cols].mean(axis=1)
        result = result.sort_values("overall_importance", ascending=False)

        return result.head(top_n)
