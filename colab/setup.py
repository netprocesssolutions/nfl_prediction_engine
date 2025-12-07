"""
Colab Setup Module

Handles environment setup, dependency installation, and database configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

# Default paths
DEFAULT_DB_NAME = "nfl_data.db"
COLAB_DB_PATH = None  # Set after setup


def install_dependencies(quiet: bool = True):
    """Install required Python packages."""
    packages = [
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ]

    q_flag = "-q" if quiet else ""
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", q_flag, pkg],
                      capture_output=quiet)

    print("Dependencies installed successfully!")


def setup_from_drive(drive_path: str = "/content/drive/MyDrive/NFL"):
    """
    Setup using database from Google Drive.

    Args:
        drive_path: Path to NFL folder in Google Drive

    Returns:
        Path to database
    """
    global COLAB_DB_PATH

    # Mount drive if not mounted
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            drive.mount("/content/drive")
    except ImportError:
        print("Not running in Colab - skipping drive mount")

    # Find database
    db_path = Path(drive_path) / DEFAULT_DB_NAME
    if not db_path.exists():
        # Try alternate locations
        alternates = [
            Path(drive_path) / "nfl_data (1).db",
            Path(drive_path) / "database" / DEFAULT_DB_NAME,
        ]
        for alt in alternates:
            if alt.exists():
                db_path = alt
                break

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {drive_path}")

    COLAB_DB_PATH = str(db_path)
    print(f"Database loaded: {COLAB_DB_PATH}")
    return COLAB_DB_PATH


def setup_from_github(repo_url: str = "https://github.com/netprocesssolutions/nfl_prediction_engine.git"):
    """
    Setup by cloning from GitHub and using LFS database.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Path to database
    """
    global COLAB_DB_PATH

    repo_name = "nfl_prediction_engine"

    # Clone if not exists
    if not os.path.exists(f"/content/{repo_name}"):
        subprocess.run(["git", "clone", repo_url], cwd="/content")
        # Pull LFS files
        subprocess.run(["git", "lfs", "pull"], cwd=f"/content/{repo_name}")

    os.chdir(f"/content/{repo_name}")
    sys.path.insert(0, f"/content/{repo_name}")

    # Find database
    db_candidates = [
        f"/content/{repo_name}/phase_1/database/nfl_data.db",
        f"/content/{repo_name}/phase_1/database/nfl_data (1).db",
    ]

    for db_path in db_candidates:
        if os.path.exists(db_path) and os.path.getsize(db_path) > 1000:
            COLAB_DB_PATH = db_path
            print(f"Database loaded: {COLAB_DB_PATH}")
            return COLAB_DB_PATH

    raise FileNotFoundError("Database not found in repo. Run 'git lfs pull' first.")


def setup_from_upload():
    """
    Setup by uploading database directly.

    Returns:
        Path to database
    """
    global COLAB_DB_PATH

    try:
        from google.colab import files
        print("Select your nfl_data.db file:")
        uploaded = files.upload()

        for filename in uploaded.keys():
            if filename.endswith('.db'):
                COLAB_DB_PATH = f"/content/{filename}"
                # Move to content root
                with open(COLAB_DB_PATH, 'wb') as f:
                    f.write(uploaded[filename])
                print(f"Database uploaded: {COLAB_DB_PATH}")
                return COLAB_DB_PATH

        raise ValueError("No .db file found in upload")

    except ImportError:
        raise RuntimeError("Upload only works in Google Colab")


def get_db_path() -> str:
    """Get the current database path."""
    if COLAB_DB_PATH is None:
        raise RuntimeError("Database not set up. Run setup_from_github(), setup_from_drive(), or setup_from_upload() first.")
    return COLAB_DB_PATH


def set_db_path(path: str):
    """Manually set the database path."""
    global COLAB_DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    COLAB_DB_PATH = path
    print(f"Database path set: {COLAB_DB_PATH}")


def verify_database():
    """Verify database connection and show stats."""
    import sqlite3

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)

    print("\n" + "=" * 50)
    print("DATABASE VERIFICATION")
    print("=" * 50)

    # Get all tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print(f"\nTables: {len(tables)}")

    # Key table stats
    key_tables = [
        ('player_game_features', 'Player features'),
        ('player_game_stats', 'Player stats'),
        ('ml_predictions', 'ML predictions'),
        ('baseline_predictions', 'Baseline predictions'),
        ('pbp_game_rec', 'PBP receiving'),
        ('pbp_game_rush', 'PBP rushing'),
        ('betting_lines', 'Betting lines'),
    ]

    print("\nKey tables:")
    for table, desc in key_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            status = "OK" if count > 0 else "EMPTY"
            print(f"  [{status}] {desc}: {count:,} rows")
        except:
            print(f"  [MISSING] {desc}")

    conn.close()
    print("\nDatabase ready!")
