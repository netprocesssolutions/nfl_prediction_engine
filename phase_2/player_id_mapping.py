# phase_2/player_id_mapping.py
"""
Player ID Mapping Utility

This module handles the mapping between different player ID formats:
- Sleeper IDs (e.g., "10210") - used in players, player_game_stats
- GSIS IDs (e.g., "00-0022531") - used in nflverse_weekly_stats, NGS tables

The mapping is extracted from players.metadata_json which contains gsis_id.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, Optional, Tuple

import pandas as pd

from .db import get_connection, read_sql


def create_player_id_mapping_table() -> None:
    """
    Create the player_id_mapping table if it doesn't exist.

    Schema:
        sleeper_id TEXT PRIMARY KEY - Sleeper format ID (e.g., "10210")
        gsis_id TEXT - NFL GSIS format ID (e.g., "00-0022531")
        player_name TEXT - Player name for debugging
        position TEXT - Player position
        created_at TEXT - Timestamp
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS player_id_mapping (
        sleeper_id TEXT PRIMARY KEY,
        gsis_id TEXT,
        player_name TEXT,
        position TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_player_id_mapping_gsis
    ON player_id_mapping(gsis_id);
    """

    with get_connection(readonly=False) as conn:
        conn.executescript(create_sql)
        conn.commit()

    print("Created player_id_mapping table")


def populate_player_id_mapping() -> Tuple[int, int]:
    """
    Populate the player_id_mapping table from players.metadata_json.

    Returns:
        Tuple of (total_players, players_with_gsis_id)
    """
    # Extract gsis_id from metadata_json for all players
    query = """
        SELECT
            player_id as sleeper_id,
            full_name as player_name,
            position,
            metadata_json
        FROM players
        WHERE metadata_json IS NOT NULL
    """

    df = read_sql(query)

    if df.empty:
        print("No players found in database")
        return 0, 0

    # Parse gsis_id from metadata_json
    def extract_gsis_id(metadata_json: str) -> Optional[str]:
        if not metadata_json:
            return None
        try:
            data = json.loads(metadata_json)
            gsis_id = data.get("gsis_id")
            if gsis_id:
                # Clean up the gsis_id (remove leading/trailing spaces)
                return gsis_id.strip()
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    df["gsis_id"] = df["metadata_json"].apply(extract_gsis_id)

    # Drop metadata_json column before inserting
    df = df[["sleeper_id", "gsis_id", "player_name", "position"]]

    # Count stats
    total_players = len(df)
    players_with_gsis = df["gsis_id"].notna().sum()

    # Insert into mapping table (replace existing)
    with get_connection(readonly=False) as conn:
        # Clear existing data
        conn.execute("DELETE FROM player_id_mapping")

        # Insert new mappings
        df.to_sql("player_id_mapping", conn, if_exists="append", index=False)
        conn.commit()

    print(f"Populated player_id_mapping: {total_players} players, {players_with_gsis} with GSIS ID")
    return total_players, players_with_gsis


def get_sleeper_to_gsis_map() -> Dict[str, str]:
    """
    Get a dictionary mapping Sleeper IDs to GSIS IDs.

    Returns:
        Dict[sleeper_id, gsis_id]
    """
    query = """
        SELECT sleeper_id, gsis_id
        FROM player_id_mapping
        WHERE gsis_id IS NOT NULL
    """
    df = read_sql(query)
    return dict(zip(df["sleeper_id"].astype(str), df["gsis_id"]))


def get_gsis_to_sleeper_map() -> Dict[str, str]:
    """
    Get a dictionary mapping GSIS IDs to Sleeper IDs.

    Returns:
        Dict[gsis_id, sleeper_id]
    """
    query = """
        SELECT sleeper_id, gsis_id
        FROM player_id_mapping
        WHERE gsis_id IS NOT NULL
    """
    df = read_sql(query)
    return dict(zip(df["gsis_id"], df["sleeper_id"].astype(str)))


def map_gsis_to_sleeper(df: pd.DataFrame, gsis_col: str = "player_id") -> pd.DataFrame:
    """
    Add a sleeper_id column to a DataFrame that has GSIS IDs.

    Args:
        df: DataFrame with GSIS format player IDs
        gsis_col: Name of the column containing GSIS IDs

    Returns:
        DataFrame with added 'sleeper_id' column
    """
    mapping = get_gsis_to_sleeper_map()
    df = df.copy()
    df["sleeper_id"] = df[gsis_col].map(mapping)
    return df


def map_sleeper_to_gsis(df: pd.DataFrame, sleeper_col: str = "player_id") -> pd.DataFrame:
    """
    Add a gsis_id column to a DataFrame that has Sleeper IDs.

    Args:
        df: DataFrame with Sleeper format player IDs
        sleeper_col: Name of the column containing Sleeper IDs

    Returns:
        DataFrame with added 'gsis_id' column
    """
    mapping = get_sleeper_to_gsis_map()
    df = df.copy()
    df["gsis_id"] = df[sleeper_col].astype(str).map(mapping)
    return df


def ensure_mapping_exists() -> None:
    """
    Ensure the player ID mapping table exists and is populated.
    Creates and populates if necessary.
    """
    # Check if table exists and has data
    try:
        query = "SELECT COUNT(*) as cnt FROM player_id_mapping"
        df = read_sql(query)
        count = df["cnt"].iloc[0]
        if count > 0:
            return  # Already populated
    except Exception:
        pass  # Table doesn't exist

    # Create and populate
    create_player_id_mapping_table()
    populate_player_id_mapping()


if __name__ == "__main__":
    print("Setting up player ID mapping...")
    create_player_id_mapping_table()
    total, with_gsis = populate_player_id_mapping()

    print(f"\nMapping statistics:")
    print(f"  Total players: {total}")
    print(f"  With GSIS ID: {with_gsis}")
    print(f"  Missing GSIS ID: {total - with_gsis}")

    # Test the mapping
    print("\nTesting mapping...")
    gsis_to_sleeper = get_gsis_to_sleeper_map()
    sleeper_to_gsis = get_sleeper_to_gsis_map()

    print(f"  GSIS->Sleeper mappings: {len(gsis_to_sleeper)}")
    print(f"  Sleeper->GSIS mappings: {len(sleeper_to_gsis)}")

    # Show a few examples
    print("\nSample mappings (first 5):")
    for i, (gsis, sleeper) in enumerate(list(gsis_to_sleeper.items())[:5]):
        print(f"  GSIS {gsis} -> Sleeper {sleeper}")
