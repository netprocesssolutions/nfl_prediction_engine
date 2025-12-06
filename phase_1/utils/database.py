"""
Database Connection Utility Module for NFL Fantasy Prediction Engine - Phase 1

This module provides centralized database connection management with:
- PRAGMA settings as specified in Phase 1 v2 (foreign_keys, WAL mode, synchronous)
- Connection context manager for safe resource handling
- Idempotent schema initialization
- Data versioning support

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator, Any
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import from config.settings (database.py is in utils/, config/ is sibling)
from config.settings import DATABASE_PATH, SQLITE_PRAGMAS, DATABASE_DIR


class DatabaseConnection:
    """
    Manages SQLite database connections with proper PRAGMA settings
    as specified in Phase 1 v2 documentation.
    
    PRAGMA settings applied:
    - foreign_keys = ON (enforce referential integrity)
    - journal_mode = WAL (Write-Ahead Logging for concurrent reads)
    - synchronous = NORMAL (balance safety/performance)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection manager.
        
        Args:
            db_path: Optional custom database path. Defaults to settings.DATABASE_PATH
        """
        self.db_path = db_path or DATABASE_PATH
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _apply_pragmas(self, conn: sqlite3.Connection):
        """
        Apply SQLite PRAGMA settings as specified in Phase 1 v2.
        
        These settings ensure:
        - Foreign key constraints are enforced
        - WAL mode for safe concurrent reads
        - Normal synchronous mode for balanced safety/performance
        """
        cursor = conn.cursor()
        for pragma, value in SQLITE_PRAGMAS.items():
            cursor.execute(f"PRAGMA {pragma} = {value};")
        cursor.close()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.
        
        Ensures:
        - PRAGMA settings are applied
        - Connection is properly closed
        - Transactions are committed or rolled back
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM players")
        
        Yields:
            sqlite3.Connection with PRAGMA settings applied
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like row access
        try:
            self._apply_pragmas(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context manager for database cursors.
        
        Convenience method that handles both connection and cursor.
        
        Usage:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM players")
                rows = cursor.fetchall()
        
        Yields:
            sqlite3.Cursor with PRAGMA settings applied to connection
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a single SQL statement and return cursor.
        
        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement
        
        Returns:
            sqlite3.Cursor with results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor
    
    def execute_many(self, sql: str, params_list: list) -> int:
        """
        Execute a SQL statement with multiple parameter sets.
        
        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples
        
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            return cursor.rowcount
    
    def execute_script(self, script: str):
        """
        Execute a multi-statement SQL script.
        
        Args:
            script: SQL script with multiple statements
        """
        with self.get_connection() as conn:
            conn.executescript(script)
    
    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """
        Execute query and fetch single result.
        
        Args:
            sql: SQL query
            params: Query parameters
        
        Returns:
            Single row or None
        """
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()
    
    def fetch_all(self, sql: str, params: tuple = ()) -> list:
        """
        Execute query and fetch all results.
        
        Args:
            sql: SQL query
            params: Query parameters
        
        Returns:
            List of rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
        
        Returns:
            True if table exists, False otherwise
        """
        result = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return result is not None
    
    def get_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Row count
        """
        result = self.fetch_one(f"SELECT COUNT(*) as count FROM {table_name}")
        return result['count'] if result else 0
    
    def get_table_schema(self, table_name: str) -> list:
        """
        Get the schema information for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            List of column information dictionaries
        """
        return self.fetch_all(f"PRAGMA table_info({table_name})")


class DataVersionManager:
    """
    Manages data versions as specified in Phase 1 v2 Section 8.
    
    Each ingestion cycle creates a version label: data_version = season_week
    Examples: 2025_01, 2025_06, 2026_14
    """
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize the data version manager.
        
        Args:
            db: DatabaseConnection instance
        """
        self.db = db
    
    def create_version(self, season: int, week: int,
                       offensive_row_count: int = 0,
                       defensive_row_count: int = 0,
                       defender_row_count: int = 0,
                       notes: str = "") -> str:
        """
        Create a new data version record.
        
        Args:
            season: NFL season year
            week: NFL week number
            offensive_row_count: Number of offensive stat rows
            defensive_row_count: Number of team defense stat rows
            defender_row_count: Number of defender stat rows
            notes: Optional notes about this version
        
        Returns:
            Version name string (e.g., "2025_06")
        """
        version_name = f"{season}_{week:02d}"
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO data_versions 
                (version_name, timestamp, notes, offensive_row_count, 
                 defensive_row_count, defender_row_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (version_name, timestamp, notes, offensive_row_count,
                  defensive_row_count, defender_row_count))
        
        return version_name
    
    def get_version(self, version_name: str) -> Optional[dict]:
        """
        Get a specific data version record.
        
        Args:
            version_name: Version name (e.g., "2025_06")
        
        Returns:
            Version record as dictionary or None
        """
        result = self.db.fetch_one(
            "SELECT * FROM data_versions WHERE version_name = ?",
            (version_name,)
        )
        return dict(result) if result else None
    
    def get_latest_version(self) -> Optional[dict]:
        """
        Get the most recent data version.
        
        Returns:
            Latest version record or None
        """
        result = self.db.fetch_one(
            "SELECT * FROM data_versions ORDER BY timestamp DESC LIMIT 1"
        )
        return dict(result) if result else None
    
        def list_versions(self, limit: int = 10) -> list:
            """
            List recent data versions.
            
            Args:
                limit: Maximum number of versions to return
            
            Returns:
                List of version records
            """
            results = self.db.fetch_all(
                "SELECT * FROM data_versions ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in results]
    
    
    class PredictionRepository:
        
        def __init__(self, db: DatabaseConnection):
            self.db = db

        def _find_prediction_row(
            self,
            model_version: str,
            season: int,
            week: int,
            prediction_type: str,
            player_id: Optional[str],
            team_id: Optional[str],
            game_id: Optional[str],
        ) -> Optional[Any]:
            query = """
                SELECT *
                FROM predictions
                WHERE model_version = ?
                AND season = ?
                AND week = ?
                AND prediction_type = ?
                AND COALESCE(player_id, '') = COALESCE(?, '')
                AND COALESCE(team_id, '') = COALESCE(?, '')
                AND COALESCE(game_id, '') = COALESCE(?, '')
            """
            params = (
                model_version,
                season,
                week,
                prediction_type,
                player_id,
                team_id,
                game_id,
            )
            with self.db.get_cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
            return row

        def upsert_prediction(
            self,
            model_version: str,
            season: int,
            week: int,
            prediction_type: str,
            predicted_value: float,
            player_id: Optional[str] = None,
            team_id: Optional[str] = None,
            game_id: Optional[str] = None,
            position: Optional[str] = None,
        ) -> int:
            existing = self._find_prediction_row(
                model_version=model_version,
                season=season,
                week=week,
                prediction_type=prediction_type,
                player_id=player_id,
                team_id=team_id,
                game_id=game_id,
            )

            if existing:
                prediction_id = existing["prediction_id"]
                update_sql = """
                    UPDATE predictions
                    SET predicted_value = ?,
                        position = COALESCE(?, position),
                        prediction_timestamp = CURRENT_TIMESTAMP
                    WHERE prediction_id = ?
                """
                with self.db.get_cursor() as cursor:
                    cursor.execute(
                        update_sql,
                        (predicted_value, position, prediction_id),
                    )
                return prediction_id

            insert_sql = """
                INSERT INTO predictions (
                    model_version,
                    season,
                    week,
                    game_id,
                    player_id,
                    team_id,
                    position,
                    prediction_type,
                    predicted_value
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                model_version,
                season,
                week,
                game_id,
                player_id,
                team_id,
                position,
                prediction_type,
                predicted_value,
            )
            with self.db.get_cursor() as cursor:
                cursor.execute(insert_sql, params)
                prediction_id = cursor.lastrowid
            return prediction_id

    def update_prediction_outcome(
        self,
        model_version: str,
        season: int,
        week: int,
        prediction_type: str,
        actual_value: float,
        player_id: Optional[str] = None,
        team_id: Optional[str] = None,
        game_id: Optional[str] = None,
        evaluation_metric: str = "fantasy_points_ppr",
        error: Optional[float] = None,
    ) -> Optional[int]:
        """
        Update an existing prediction row with the actual outcome and error.

        If error is not provided, it will be computed as:
            error = actual_value - predicted_value

        Args:
            model_version: Identifier for the model
            season: NFL season year
            week: NFL week number
            prediction_type: Type of prediction, e.g. "fantasy_points_ppr"
            actual_value: Actual outcome (points, probability, etc.)
            player_id: Optional player_id (for player-level predictions)
            team_id: Optional team_id (for team-level predictions)
            game_id: Optional game_id (for game-level predictions)
            evaluation_metric: Label describing how actual_value/error are interpreted
            error: Optional precomputed error. If None, it will be derived.

        Returns:
            prediction_id if the row was found and updated, otherwise None.
        """
        existing = self._find_prediction_row(
            model_version=model_version,
            season=season,
            week=week,
            prediction_type=prediction_type,
            player_id=player_id,
            team_id=team_id,
            game_id=game_id,
        )
        if not existing:
            # No matching prediction row found; nothing to update
            return None

        prediction_id = existing["prediction_id"]
        if error is None:
            predicted_value = existing["predicted_value"]
            if actual_value is not None and predicted_value is not None:
                error = actual_value - predicted_value
            else:
                error = None

        update_sql = """
            UPDATE predictions
               SET actual_value = ?,
                   evaluation_metric = ?,
                   error = ?
             WHERE prediction_id = ?
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                update_sql,
                (actual_value, evaluation_metric, error, prediction_id),
            )
        return prediction_id

# =============================================================================
# Module-level convenience functions
# =============================================================================

# Global database instance
_db_instance: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """
    Get the global database connection instance.
    
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


def get_version_manager() -> DataVersionManager:
    """
    Get a data version manager instance.
    
    Returns:
        DataVersionManager instance
    """
    return DataVersionManager(get_db())


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    
    db = get_db()
    print(f"Database path: {db.db_path}")
    print(f"Database exists: {db.db_path.exists()}")
    
    # Test connection
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        print(f"Foreign keys enabled: {result[0]}")
        
        cursor.execute("PRAGMA journal_mode")
        result = cursor.fetchone()
        print(f"Journal mode: {result[0]}")
    
    print("Database connection test complete!")
