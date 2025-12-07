"""
Colab Training Module

Functions for training ML models in Google Colab.
"""

import os
import sys
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from . import data


# Target columns
TARGETS = [
    "label_targets", "label_receptions", "label_rec_yards", "label_rec_tds",
    "label_carries", "label_rush_yards", "label_rush_tds",
    "label_pass_attempts", "label_pass_completions", "label_pass_yards",
    "label_pass_tds", "label_interceptions"
]

# Model output directory
MODELS_DIR = Path("saved_models")


class NFLMultiModelTrainer:
    """
    Multi-model trainer for NFL predictions.

    Trains Ridge, ElasticNet, XGBoost, and LightGBM models for each target stat.
    """

    def __init__(self, name: str = "nfl_model", version: str = None):
        self.name = name
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.target_cols = []
        self.training_metadata = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        model_types: List[str] = None,
        verbose: bool = True
    ):
        """
        Train all models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_types: Which model types to train (default: all available)
            verbose: Print progress
        """
        self.feature_cols = list(X_train.columns)
        self.target_cols = list(y_train.columns)

        # Determine model types
        if model_types is None:
            model_types = ["ridge", "elasticnet"]
            if HAS_XGB:
                model_types.append("xgboost")
            if HAS_LGB:
                model_types.append("lightgbm")

        if verbose:
            print(f"\nTraining {len(model_types)} model types on {len(self.target_cols)} targets")
            print(f"Features: {len(self.feature_cols)}")
            print(f"Training samples: {len(X_train):,}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val):,}")
            print()

        # Scale features
        self.scalers["features"] = StandardScaler()
        X_train_scaled = self.scalers["features"].fit_transform(X_train.fillna(0))
        if X_val is not None:
            X_val_scaled = self.scalers["features"].transform(X_val.fillna(0))

        # Train models for each target
        models_trained = []

        for target in self.target_cols:
            if verbose:
                print(f"Training {target}...")

            y_train_target = y_train[target].fillna(0)
            y_val_target = y_val[target].fillna(0) if y_val is not None else None

            self.models[target] = {}

            # Ridge regression
            if "ridge" in model_types:
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_scaled, y_train_target)
                self.models[target]["ridge"] = ridge

            # ElasticNet
            if "elasticnet" in model_types:
                en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
                en.fit(X_train_scaled, y_train_target)
                self.models[target]["elasticnet"] = en

            # XGBoost
            if "xgboost" in model_types and HAS_XGB:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                xgb_model.fit(X_train_scaled, y_train_target)
                self.models[target]["xgboost"] = xgb_model

            # LightGBM
            if "lightgbm" in model_types and HAS_LGB:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                lgb_model.fit(X_train_scaled, y_train_target)
                self.models[target]["lightgbm"] = lgb_model

            models_trained.append(target)

        # Store metadata
        self.training_metadata = {
            "name": self.name,
            "version": self.version,
            "n_features": len(self.feature_cols),
            "n_targets": len(self.target_cols),
            "n_train_samples": len(X_train),
            "model_types": model_types,
            "models_trained": models_trained,
            "trained_at": datetime.now().isoformat()
        }

        if verbose:
            print(f"\nTraining complete!")
            print(f"Models trained: {len(models_trained)} targets x {len(model_types)} types")

    def predict(self, X: pd.DataFrame, model_type: str = None) -> pd.DataFrame:
        """
        Generate predictions.

        Args:
            X: Feature DataFrame
            model_type: Specific model type (None = all)

        Returns:
            DataFrame with predictions for all targets
        """
        X_scaled = self.scalers["features"].transform(X[self.feature_cols].fillna(0))

        predictions = {}

        for target in self.target_cols:
            target_models = self.models.get(target, {})

            if model_type:
                if model_type in target_models:
                    model = target_models[model_type]
                    pred = model.predict(X_scaled)
                    clean_name = target.replace("label_", "")
                    predictions[f"pred_{model_type}_{clean_name}"] = pred
            else:
                # Predict with all models
                for mtype, model in target_models.items():
                    pred = model.predict(X_scaled)
                    clean_name = target.replace("label_", "")
                    predictions[f"pred_{mtype}_{clean_name}"] = pred

        return pd.DataFrame(predictions)

    def predict_ensemble(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble predictions (average of all models).

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with ensemble predictions
        """
        all_preds = self.predict(X)

        ensemble = {}
        for target in self.target_cols:
            clean_name = target.replace("label_", "")
            # Get all predictions for this target
            target_preds = [c for c in all_preds.columns if c.endswith(f"_{clean_name}")]
            if target_preds:
                ensemble[f"pred_ensemble_{clean_name}"] = all_preds[target_preds].mean(axis=1)

        return pd.DataFrame(ensemble)

    def predict_fantasy_points(
        self,
        X: pd.DataFrame,
        scoring: str = "ppr"
    ) -> pd.DataFrame:
        """
        Predict fantasy points.

        Args:
            X: Feature DataFrame
            scoring: Scoring format (ppr, half_ppr, standard)

        Returns:
            DataFrame with fantasy point predictions
        """
        # Get ensemble predictions
        preds = self.predict_ensemble(X)

        # Scoring rules
        scoring_rules = {
            "ppr": {"receptions": 1.0, "rec_yards": 0.1, "rec_tds": 6, "rush_yards": 0.1, "rush_tds": 6, "pass_yards": 0.04, "pass_tds": 4, "interceptions": -2},
            "half_ppr": {"receptions": 0.5, "rec_yards": 0.1, "rec_tds": 6, "rush_yards": 0.1, "rush_tds": 6, "pass_yards": 0.04, "pass_tds": 4, "interceptions": -2},
            "standard": {"receptions": 0.0, "rec_yards": 0.1, "rec_tds": 6, "rush_yards": 0.1, "rush_tds": 6, "pass_yards": 0.04, "pass_tds": 4, "interceptions": -2}
        }

        rules = scoring_rules.get(scoring, scoring_rules["ppr"])

        # Calculate fantasy points
        fp = pd.Series(0.0, index=preds.index)

        for stat, multiplier in rules.items():
            col = f"pred_ensemble_{stat}"
            if col in preds.columns:
                fp += preds[col].fillna(0) * multiplier

        return pd.DataFrame({f"pred_fp_{scoring}": fp})

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            verbose: Print results

        Returns:
            Dictionary of metrics by target and model type
        """
        all_preds = self.predict(X)
        ensemble_preds = self.predict_ensemble(X)

        results = {}

        for target in self.target_cols:
            clean_name = target.replace("label_", "")
            y_true = y[target].fillna(0)
            results[clean_name] = {}

            # Evaluate each model type
            for col in all_preds.columns:
                if col.endswith(f"_{clean_name}"):
                    model_type = col.replace(f"pred_", "").replace(f"_{clean_name}", "")
                    y_pred = all_preds[col]

                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)

                    results[clean_name][model_type] = {
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2
                    }

            # Evaluate ensemble
            ens_col = f"pred_ensemble_{clean_name}"
            if ens_col in ensemble_preds.columns:
                y_pred = ensemble_preds[ens_col]
                results[clean_name]["ensemble"] = {
                    "mae": mean_absolute_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "r2": r2_score(y_true, y_pred)
                }

        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)

            for target, models in results.items():
                print(f"\n{target}:")
                for model, metrics in models.items():
                    print(f"  {model:12} MAE: {metrics['mae']:.3f}  RMSE: {metrics['rmse']:.3f}  R2: {metrics['r2']:.3f}")

        return results

    def save(self, path: str = None) -> Path:
        """
        Save trained models.

        Args:
            path: Save directory (default: saved_models/{name}_{version})

        Returns:
            Path to saved model directory
        """
        if path is None:
            MODELS_DIR.mkdir(exist_ok=True)
            path = MODELS_DIR / f"{self.name}_{self.version}"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(path / "models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        # Save scalers
        with open(path / "scalers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)

        # Save metadata
        metadata = {
            **self.training_metadata,
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "NFLMultiModelTrainer":
        """
        Load trained models.

        Args:
            path: Model directory path

        Returns:
            Loaded trainer instance
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        trainer = cls(name=metadata["name"], version=metadata["version"])

        # Load models
        with open(path / "models.pkl", "rb") as f:
            trainer.models = pickle.load(f)

        # Load scalers
        with open(path / "scalers.pkl", "rb") as f:
            trainer.scalers = pickle.load(f)

        trainer.feature_cols = metadata["feature_cols"]
        trainer.target_cols = metadata["target_cols"]
        trainer.training_metadata = metadata

        print(f"Model loaded from: {path}")
        return trainer


def train_models(
    train_seasons: List[int],
    val_season: int = None,
    test_season: int = None,
    positions: List[str] = None,
    model_types: List[str] = None,
    quick: bool = False,
    verbose: bool = True
) -> NFLMultiModelTrainer:
    """
    Train NFL prediction models.

    Args:
        train_seasons: Seasons for training
        val_season: Season for validation
        test_season: Season for test evaluation
        positions: Position filter
        model_types: Model types to train
        quick: Use quick settings
        verbose: Print progress

    Returns:
        Trained model trainer

    Example:
        trainer = train_models([2021, 2022, 2023], val_season=2023, test_season=2024)
    """
    if verbose:
        print("=" * 60)
        print("NFL MULTI-MODEL TRAINING")
        print("=" * 60)
        print(f"Training seasons: {train_seasons}")
        print(f"Validation season: {val_season}")
        print(f"Test season: {test_season}")
        print()

    # Load data
    all_seasons = list(train_seasons)
    if val_season and val_season not in all_seasons:
        all_seasons.append(val_season)
    if test_season and test_season not in all_seasons:
        all_seasons.append(test_season)

    df, feature_cols, target_cols = data.get_training_data(all_seasons, positions)

    if verbose:
        print(f"Loaded {len(df):,} samples")
        print(f"Features: {len(feature_cols)}")

    # Split data
    train_df = df[df["season"].isin(train_seasons)]
    if val_season:
        train_df = train_df[train_df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        val_df = None

    if test_season:
        test_df = df[df["season"] == test_season]
    else:
        test_df = None

    if verbose:
        print(f"\nTraining: {len(train_df):,} samples")
        if val_df is not None:
            print(f"Validation: {len(val_df):,} samples")
        if test_df is not None:
            print(f"Test: {len(test_df):,} samples")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols] if val_df is not None else None
    y_val = val_df[target_cols] if val_df is not None else None

    # Create and train model
    trainer = NFLMultiModelTrainer()
    trainer.fit(X_train, y_train, X_val, y_val, model_types, verbose)

    # Evaluate on test set
    if test_df is not None:
        X_test = test_df[feature_cols]
        y_test = test_df[target_cols]
        trainer.evaluate(X_test, y_test, verbose)

    # Save model
    model_path = trainer.save()

    if verbose:
        print(f"\nTraining complete! Model saved to: {model_path}")

    return trainer


def list_saved_models() -> List[str]:
    """List all saved models."""
    if not MODELS_DIR.exists():
        return []
    return [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]


def load_latest_model() -> NFLMultiModelTrainer:
    """Load the most recently saved model."""
    models = list_saved_models()
    if not models:
        raise FileNotFoundError("No saved models found")

    # Sort by modification time
    model_paths = [MODELS_DIR / m for m in models]
    model_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return NFLMultiModelTrainer.load(model_paths[0])
