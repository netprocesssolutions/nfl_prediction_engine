"""
Colab Save Module

Functions for saving predictions and data to the database.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd

from . import setup
from . import data


def save_ml_predictions(
    predictions: pd.DataFrame,
    model_version: str = None,
    replace_existing: bool = True,
    verbose: bool = True
) -> int:
    """
    Save ML predictions to database.

    Args:
        predictions: DataFrame with predictions (from predict_week)
        model_version: Version string for tracking
        replace_existing: Replace existing predictions for same season/week
        verbose: Print summary

    Returns:
        Number of rows saved

    Example:
        preds = predict.predict_week(2024, 14)
        save.save_ml_predictions(preds, model_version="v1.0")
    """
    if predictions.empty:
        print("No predictions to save")
        return 0

    conn = sqlite3.connect(setup.get_db_path())

    # Prepare data
    save_df = predictions.copy()

    # Add metadata
    if model_version:
        save_df["model_version"] = model_version
    else:
        save_df["model_version"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_df["created_at"] = datetime.now().isoformat()

    # Get season/week for potential replacement
    season = save_df["season"].iloc[0]
    week = save_df["week"].iloc[0]

    if replace_existing:
        # Delete existing predictions for this season/week
        conn.execute(
            "DELETE FROM ml_predictions WHERE season = ? AND week = ?",
            (season, week)
        )
        conn.commit()

    # Select columns to save (only prediction columns + metadata)
    pred_cols = [c for c in save_df.columns if c.startswith("pred_")]
    meta_cols = ["season", "week", "game_id", "player_id", "player_name",
                 "position", "team", "opponent", "model_version", "created_at"]

    save_cols = [c for c in meta_cols + pred_cols if c in save_df.columns]
    save_df = save_df[save_cols]

    # Save to database
    save_df.to_sql("ml_predictions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    if verbose:
        print(f"Saved {len(save_df)} predictions to ml_predictions")
        print(f"Season: {season}, Week: {week}")
        print(f"Model version: {save_df['model_version'].iloc[0]}")

    return len(save_df)


def save_baseline_predictions(
    predictions: pd.DataFrame,
    version: str = None,
    replace_existing: bool = True,
    verbose: bool = True
) -> int:
    """
    Save baseline predictions to database.

    Args:
        predictions: DataFrame with baseline predictions
        version: Version string
        replace_existing: Replace existing predictions
        verbose: Print summary

    Returns:
        Number of rows saved
    """
    if predictions.empty:
        print("No predictions to save")
        return 0

    conn = sqlite3.connect(setup.get_db_path())

    save_df = predictions.copy()
    save_df["baseline_version"] = version or "3.0.0"
    save_df["created_at"] = datetime.now().isoformat()

    season = save_df["season"].iloc[0]
    week = save_df["week"].iloc[0]

    if replace_existing:
        conn.execute(
            "DELETE FROM baseline_predictions WHERE season = ? AND week = ?",
            (season, week)
        )
        conn.commit()

    save_df.to_sql("baseline_predictions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    if verbose:
        print(f"Saved {len(save_df)} predictions to baseline_predictions")

    return len(save_df)


def save_custom_predictions(
    predictions: pd.DataFrame,
    table_name: str,
    replace_existing: bool = False,
    verbose: bool = True
) -> int:
    """
    Save predictions to a custom table.

    Args:
        predictions: DataFrame to save
        table_name: Name of table to create/append
        replace_existing: Replace entire table if True
        verbose: Print summary

    Returns:
        Number of rows saved

    Example:
        # Save weekly prop analysis
        save.save_custom_predictions(prop_analysis, "weekly_props")
    """
    if predictions.empty:
        print("No data to save")
        return 0

    conn = sqlite3.connect(setup.get_db_path())

    save_df = predictions.copy()
    save_df["saved_at"] = datetime.now().isoformat()

    if_exists = "replace" if replace_existing else "append"
    save_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    conn.close()

    if verbose:
        print(f"Saved {len(save_df)} rows to {table_name}")

    return len(save_df)


def save_betting_analysis(
    analysis: pd.DataFrame,
    analysis_type: str = "props",
    season: int = None,
    week: int = None,
    verbose: bool = True
) -> int:
    """
    Save betting analysis to database.

    Args:
        analysis: DataFrame with betting analysis
        analysis_type: Type of analysis (props, lines, value)
        season: Season (extracted from data if not provided)
        week: Week (extracted from data if not provided)
        verbose: Print summary

    Returns:
        Number of rows saved

    Example:
        value_props = betting.find_value_props(...)
        save.save_betting_analysis(value_props, "props", 2024, 14)
    """
    if analysis.empty:
        print("No analysis to save")
        return 0

    table_name = f"betting_analysis_{analysis_type}"

    conn = sqlite3.connect(setup.get_db_path())

    save_df = analysis.copy()
    save_df["analysis_type"] = analysis_type
    save_df["analyzed_at"] = datetime.now().isoformat()

    if season:
        save_df["season"] = season
    if week:
        save_df["week"] = week

    save_df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    if verbose:
        print(f"Saved {len(save_df)} rows to {table_name}")

    return len(save_df)


def export_predictions_csv(
    predictions: pd.DataFrame,
    filename: str = None,
    season: int = None,
    week: int = None
) -> str:
    """
    Export predictions to CSV file.

    Args:
        predictions: DataFrame to export
        filename: Output filename (auto-generated if None)
        season: Season for filename
        week: Week for filename

    Returns:
        Path to saved file

    Example:
        path = save.export_predictions_csv(preds, season=2024, week=14)
    """
    if filename is None:
        if season and week:
            filename = f"predictions_{season}_week{week}.csv"
        else:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    predictions.to_csv(filename, index=False)
    print(f"Exported to: {filename}")

    return filename


def export_to_drive(
    predictions: pd.DataFrame,
    drive_path: str = "/content/drive/MyDrive/NFL/predictions",
    filename: str = None,
    season: int = None,
    week: int = None
) -> str:
    """
    Export predictions to Google Drive.

    Args:
        predictions: DataFrame to export
        drive_path: Path in Google Drive
        filename: Output filename
        season: Season for filename
        week: Week for filename

    Returns:
        Path to saved file

    Example:
        save.export_to_drive(preds, season=2024, week=14)
    """
    import os

    # Ensure directory exists
    os.makedirs(drive_path, exist_ok=True)

    if filename is None:
        if season and week:
            filename = f"predictions_{season}_week{week}.csv"
        else:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    filepath = os.path.join(drive_path, filename)
    predictions.to_csv(filepath, index=False)
    print(f"Exported to Drive: {filepath}")

    return filepath


def sync_database_to_drive(
    drive_path: str = "/content/drive/MyDrive/NFL"
) -> str:
    """
    Copy database to Google Drive for backup.

    Args:
        drive_path: Destination in Google Drive

    Returns:
        Path to backed up database
    """
    import shutil
    import os

    os.makedirs(drive_path, exist_ok=True)

    db_path = setup.get_db_path()
    backup_name = f"nfl_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    backup_path = os.path.join(drive_path, backup_name)

    shutil.copy(db_path, backup_path)
    print(f"Database backed up to: {backup_path}")

    return backup_path


def download_predictions(predictions: pd.DataFrame, filename: str = None):
    """
    Download predictions as CSV (Colab only).

    Args:
        predictions: DataFrame to download
        filename: Output filename
    """
    try:
        from google.colab import files

        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        predictions.to_csv(filename, index=False)
        files.download(filename)
        print(f"Downloaded: {filename}")

    except ImportError:
        print("Download only works in Google Colab")
        print("Use export_predictions_csv() instead")


def get_saved_predictions(
    prediction_type: str = "ml",
    season: int = None,
    week: int = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Retrieve saved predictions from database.

    Args:
        prediction_type: "ml" or "baseline"
        season: Filter by season
        week: Filter by week
        limit: Max rows to return

    Returns:
        DataFrame with predictions
    """
    table = "ml_predictions" if prediction_type == "ml" else "baseline_predictions"

    sql = f"SELECT * FROM {table} WHERE 1=1"
    params = []

    if season:
        sql += " AND season = ?"
        params.append(season)
    if week:
        sql += " AND week = ?"
        params.append(week)

    sql += " ORDER BY created_at DESC"

    if limit:
        sql += f" LIMIT {limit}"

    return data.query(sql, params)


def list_saved_predictions(verbose: bool = True) -> pd.DataFrame:
    """
    List all saved prediction sets.

    Args:
        verbose: Print summary

    Returns:
        DataFrame with prediction metadata
    """
    conn = sqlite3.connect(setup.get_db_path())

    # Check ML predictions
    try:
        ml_summary = pd.read_sql_query("""
            SELECT 'ml' as type, season, week, model_version,
                   COUNT(*) as players, MAX(created_at) as created_at
            FROM ml_predictions
            GROUP BY season, week, model_version
            ORDER BY created_at DESC
        """, conn)
    except:
        ml_summary = pd.DataFrame()

    # Check baseline predictions
    try:
        baseline_summary = pd.read_sql_query("""
            SELECT 'baseline' as type, season, week, baseline_version as model_version,
                   COUNT(*) as players, MAX(created_at) as created_at
            FROM baseline_predictions
            GROUP BY season, week, baseline_version
            ORDER BY created_at DESC
        """, conn)
    except:
        baseline_summary = pd.DataFrame()

    conn.close()

    summary = pd.concat([ml_summary, baseline_summary], ignore_index=True)

    if verbose and len(summary) > 0:
        print("\n" + "=" * 60)
        print("SAVED PREDICTIONS")
        print("=" * 60)
        print(summary.to_string(index=False))

    return summary
