# NFL Fantasy Prediction Engine - Google Colab Guide

This guide explains how to use the NFL Fantasy Prediction Engine in Google Colab for free GPU/CPU compute resources.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Weekly Workflow](#weekly-workflow)
3. [Training New Models](#training-new-models)
4. [Making Predictions](#making-predictions)
5. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### Step 1: Upload Your Project to Google Drive

1. Zip your `nfl_prediction_engine` folder on your local machine
2. Upload the zip file to Google Drive
3. Note the path (e.g., `/content/drive/MyDrive/nfl_prediction_engine.zip`)

### Step 2: Create a New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 3: Mount Google Drive and Extract Files

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract the project (first time only)
!unzip -q /content/drive/MyDrive/nfl_prediction_engine.zip -d /content/

# Navigate to project directory
%cd /content/nfl_prediction_engine
```

### Step 4: Install Dependencies

```python
# Install required packages
!pip install -q pandas numpy scikit-learn xgboost lightgbm nfl_data_py joblib

# Verify installation
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
print("All dependencies installed successfully!")
```

### Step 5: Verify Database

```python
import sqlite3
from pathlib import Path

db_path = Path("phase_1/database/nfl_data.db")
conn = sqlite3.connect(db_path)

# Check data counts
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM player_game_features")
features = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM play_by_play")
pbp = cursor.fetchone()[0]

print(f"Player-game features: {features:,}")
print(f"Play-by-play records: {pbp:,}")
conn.close()
```

---

## Weekly Workflow

### Quick Reference: Weekly Prediction Steps

```python
# === WEEKLY PREDICTION WORKFLOW ===
# Run this each week to get fresh predictions

# Step 1: Update current week's data
!python -m phase_1.ingestion_scripts.ingest_all --current

# Step 2: Build features for the new week
from phase_2.pipeline.build_week import build_features_for_week
build_features_for_week(season=2024, week=15)  # <-- Change week number

# Step 3: Generate predictions
!python -m phase_4.predict --season 2024 --week 15 --output predictions_week15.csv
```

### Detailed Weekly Steps

#### 1. Ingest New Week's Data

```python
# Ingest latest data from all sources
!python -m phase_1.ingestion_scripts.ingest_all --current

# Or ingest specific sources
!python -m phase_1.ingestion_scripts.ingest_sleeper --week 15
!python -m phase_1.ingestion_scripts.ingest_nflverse --current
!python -m phase_1.ingestion_scripts.ingest_pbp --current
```

#### 2. Build Features for the Week

```python
import sys
sys.path.insert(0, '/content/nfl_prediction_engine')

from phase_2.pipeline.build_week import build_features_for_week

# Build features for week 15 (includes all 259 features with PBP data)
df = build_features_for_week(season=2024, week=15)
print(f"Built {len(df)} player-game features with {len(df.columns)} columns")
```

#### 3. Generate Predictions

```python
# Generate predictions using trained multi-model ensemble
!python -m phase_4.predict --season 2024 --week 15

# With CSV output
!python -m phase_4.predict --season 2024 --week 15 --output week15_predictions.csv

# Filter by position
!python -m phase_4.predict --season 2024 --week 15 --positions WR RB

# Different scoring formats
!python -m phase_4.predict --season 2024 --week 15 --scoring half_ppr
```

#### 4. Download Predictions

```python
# Download to your computer
from google.colab import files
files.download('week15_predictions.csv')
```

---

## Training New Models

### When to Retrain

Retrain your models:
- At the start of each season (major retrain)
- Mid-season if performance degrades
- When new features are added

### Training Command

```python
# Train multi-model ensemble (Ridge, ElasticNet, XGBoost, LightGBM)
!python -m phase_4.training.train \
    --seasons 2021 2022 2023 \
    --val-season 2023 \
    --min-games 3

# Quick training (Ridge + XGBoost only)
!python -m phase_4.training.train \
    --seasons 2022 2023 \
    --val-season 2023 \
    --quick
```

### Training in Python

```python
import sys
sys.path.insert(0, '/content/nfl_prediction_engine')

from phase_4.training.train import train_multi_model

# Train with custom settings
trainer = train_multi_model(
    train_seasons=[2021, 2022, 2023],
    val_season=2023,
    test_season=2024,
    positions=None,  # All positions
    min_games=3,
    quick=False,  # Train all 4 model types
    verbose=True
)

print(f"Model saved to: {trainer.model_path}")
```

### Model Evaluation

```python
# Evaluate model performance
!python -m phase_4.evaluation.evaluate --season 2024 --compare

# Per-position analysis
!python -m phase_4.evaluation.evaluate --season 2024 --positions QB
```

---

## Making Predictions

### Programmatic Predictions

```python
import sys
sys.path.insert(0, '/content/nfl_prediction_engine')

from phase_4.predict import predict_week

# Get predictions DataFrame
predictions = predict_week(
    season=2024,
    week=15,
    scoring_type='ppr',
    verbose=True
)

# View top predictions by position
for pos in ['QB', 'RB', 'WR', 'TE']:
    print(f"\n=== Top 5 {pos}s ===")
    pos_preds = predictions[predictions['position'] == pos]
    print(pos_preds[['player_name', 'team', 'opponent', 'pred_fp_ppr']].head())
```

### Understanding Prediction Columns

| Column | Description |
|--------|-------------|
| `pred_ridge_*` | Ridge regression predictions |
| `pred_elasticnet_*` | ElasticNet predictions |
| `pred_xgb_*` | XGBoost predictions |
| `pred_lgb_*` | LightGBM predictions |
| `pred_ensemble_*` | Weighted average of all models |
| `pred_fp_ppr` | Predicted fantasy points (PPR) |
| `pred_fp_half_ppr` | Predicted fantasy points (Half PPR) |
| `pred_fp_standard` | Predicted fantasy points (Standard) |

### Stat Predictions

Each model predicts these individual stats:
- `targets`, `receptions`, `receiving_yards`, `receiving_tds`
- `carries`, `rushing_yards`, `rushing_tds`
- `completions`, `passing_yards`, `passing_tds`, `interceptions`

---

## Complete Weekly Notebook Template

Copy this entire cell into a new Colab notebook for weekly use:

```python
#@title NFL Fantasy Prediction Engine - Weekly Predictions

# Configuration
SEASON = 2024  #@param {type:"integer"}
WEEK = 15  #@param {type:"integer"}
SCORING = "ppr"  #@param ["ppr", "half_ppr", "standard"]

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup
%cd /content/drive/MyDrive/nfl_prediction_engine

# Install dependencies (if needed)
!pip install -q pandas numpy scikit-learn xgboost lightgbm nfl_data_py joblib

# Import modules
import sys
sys.path.insert(0, '.')

# Step 1: Update data
print("=" * 60)
print("STEP 1: Updating Data")
print("=" * 60)
!python -m phase_1.ingestion_scripts.ingest_all --current

# Step 2: Build features
print("\n" + "=" * 60)
print("STEP 2: Building Features")
print("=" * 60)
from phase_2.pipeline.build_week import build_features_for_week
df = build_features_for_week(SEASON, WEEK)
print(f"Built features for {len(df)} players")

# Step 3: Generate predictions
print("\n" + "=" * 60)
print("STEP 3: Generating Predictions")
print("=" * 60)
from phase_4.predict import predict_week
predictions = predict_week(SEASON, WEEK, scoring_type=SCORING)

# Step 4: Display results
print("\n" + "=" * 60)
print(f"TOP 20 PLAYERS - WEEK {WEEK} ({SCORING.upper()})")
print("=" * 60)
display_cols = ['player_name', 'position', 'team', 'opponent', f'pred_fp_{SCORING}']
print(predictions[display_cols].head(20).to_string(index=False))

# Step 5: Save and download
output_file = f'predictions_week{WEEK}_{SCORING}.csv'
predictions.to_csv(output_file, index=False)
from google.colab import files
files.download(output_file)
print(f"\nPredictions saved to: {output_file}")
```

---

## Troubleshooting

### Common Issues

**Issue: "Module not found" errors**
```python
import sys
sys.path.insert(0, '/content/nfl_prediction_engine')
```

**Issue: Database not found**
```python
# Verify path
import os
os.chdir('/content/nfl_prediction_engine')
!ls -la phase_1/database/
```

**Issue: Out of memory during training**
```python
# Use quick training mode
!python -m phase_4.training.train --seasons 2023 --val-season 2023 --quick

# Or train individual positions
!python -m phase_4.training.train --seasons 2022 2023 --positions WR
```

**Issue: Slow performance**
- Use GPU runtime: Runtime → Change runtime type → GPU
- XGBoost and LightGBM benefit from GPU acceleration

### Saving Your Work

```python
# Save updated database back to Drive
!cp phase_1/database/nfl_data.db /content/drive/MyDrive/nfl_data_backup.db

# Save trained models
!cp -r phase_4/saved_models /content/drive/MyDrive/nfl_models/
```

---

## Resource Requirements

| Task | Time | Memory |
|------|------|--------|
| Weekly predictions | 1-2 min | 2 GB |
| Feature building | 5-10 min | 4 GB |
| Full model training | 15-30 min | 8 GB |
| Quick model training | 5-10 min | 4 GB |

Google Colab Free Tier provides:
- 12 GB RAM (usually sufficient)
- GPU access (helps with XGBoost/LightGBM)
- ~12 hour session limit

---

## Tips for Best Results

1. **Save your database to Drive** after each update to preserve data
2. **Run predictions early in the week** when player news breaks
3. **Retrain models monthly** during the season for best accuracy
4. **Check the ensemble predictions** (`pred_ensemble_*`) for most robust results
5. **Download predictions as CSV** for use in your fantasy platform

---

## Questions?

For issues or feature requests, please open an issue in the repository.
