"""
Logging Utility Module for NFL Fantasy Prediction Engine - Phase 1

This module implements the logging architecture defined in Plan v2 Section 10.
All logs use structured JSON format for easy searching, parsing, and debugging.

Features:
- Structured JSON line format
- Separate log files per category (ingestion, validation, errors)
- Timestamped entries
- Source and event tracking
- Error traceback logging

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Try to import from config.settings, with fallbacks for missing constants
try:
    from config.settings import (
        LOGS_DIR, LOG_INGESTION_DIR, LOG_SYSTEM_DIR,
        LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL
    )
except ImportError:
    # Define fallbacks if not all constants exist
    from config.settings import LOGS_DIR
    LOG_INGESTION_DIR = LOGS_DIR / "data_ingestion"
    LOG_SYSTEM_DIR = LOGS_DIR / "system"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_LEVEL = "INFO"


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON lines.
    This matches the format specified in Plan v2 Section 10.3:
    
    {
        "timestamp": "2025-09-14T03:14:07Z",
        "level": "INFO",
        "source": "feature_engineering",
        "event": "delta_computation",
        "player_id": "12345",
        "detail": "computed deltas for WR; window=3 weeks"
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "source": getattr(record, 'source', record.name),
            "event": getattr(record, 'event', record.funcName),
            "detail": record.getMessage(),
        }
        
        # Add optional fields if present
        for field in ['player_id', 'defender_id', 'team_id', 'game_id', 
                      'season', 'week', 'row_count', 'error_type', 'data_version']:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class IngestionLogger:
    """
    Specialized logger for data ingestion operations.
    
    Creates structured logs as per Plan v2 Section 10.4:
    - Count of rows ingested vs expected
    - Null rates per column
    - New player IDs discovered
    - Player-team mismatch count
    - Missing opponent_team_ids
    - Rows failing validation checks
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize the ingestion logger.
        
        Args:
            name: Logger name (e.g., 'ingest_players', 'ingest_games')
            log_dir: Optional custom log directory
        """
        self.name = name
        self.log_dir = log_dir or LOG_INGESTION_DIR
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"ingestion.{name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create file handler with JSON formatter
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
        
        # Create console handler with standard formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        )
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method that attaches extra fields."""
        extra = {'source': self.name}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    # Specialized logging methods for data ingestion
    
    def log_ingestion_start(self, source: str, season: Optional[int] = None, 
                           week: Optional[int] = None):
        """Log the start of an ingestion operation."""
        self.info(
            f"Starting ingestion from {source}",
            event="ingestion_start",
            season=season,
            week=week
        )
    
    def log_ingestion_complete(self, row_count: int, expected_count: Optional[int] = None,
                               duration_seconds: Optional[float] = None):
        """Log successful completion of ingestion."""
        detail = f"Ingested {row_count} rows"
        if expected_count:
            detail += f" (expected: {expected_count})"
        if duration_seconds:
            detail += f" in {duration_seconds:.2f}s"
        
        self.info(
            detail,
            event="ingestion_complete",
            row_count=row_count
        )
    
    def log_row_inserted(self, table: str, row_id: str, **kwargs):
        """Log a single row insertion."""
        self.debug(
            f"Inserted row into {table}: {row_id}",
            event="row_inserted",
            **kwargs
        )
    
    def log_null_warning(self, column: str, null_count: int, total_count: int):
        """Log a warning about null values in a column."""
        null_rate = null_count / total_count if total_count > 0 else 0
        self.warning(
            f"Column '{column}' has {null_count}/{total_count} null values ({null_rate:.1%})",
            event="null_warning"
        )
    
    def log_validation_failure(self, validation_type: str, detail: str, **kwargs):
        """Log a validation failure. Ingestion should STOP when this is called."""
        self.error(
            f"VALIDATION FAILED [{validation_type}]: {detail}",
            event="validation_failure",
            error_type=validation_type,
            **kwargs
        )
    
    def log_missing_data(self, entity_type: str, entity_id: str, missing_fields: list):
        """Log missing data for an entity."""
        self.warning(
            f"Missing data for {entity_type} {entity_id}: {missing_fields}",
            event="missing_data"
        )
    
    def log_duplicate_detected(self, table: str, key: str, action: str = "skipped"):
        """Log detection of a duplicate entry."""
        self.warning(
            f"Duplicate detected in {table} for key {key}, action: {action}",
            event="duplicate_detected"
        )
    
    def log_api_call(self, endpoint: str, status_code: int, 
                     response_time_ms: Optional[float] = None):
        """Log an API call."""
        detail = f"API call to {endpoint} returned {status_code}"
        if response_time_ms:
            detail += f" in {response_time_ms:.0f}ms"
        
        if status_code >= 400:
            self.error(detail, event="api_call")
        else:
            self.debug(detail, event="api_call")
    
    def log_data_version_created(self, version: str, notes: str = ""):
        """Log creation of a new data version."""
        self.info(
            f"Created data version: {version}. {notes}",
            event="data_version_created",
            data_version=version
        )


def setup_system_logger() -> logging.Logger:
    """
    Set up the system-wide logger for general system events.
    
    Returns:
        Configured logger instance
    """
    LOG_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("nfl_prediction_engine")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    log_file = LOG_SYSTEM_DIR / f"system_{datetime.now().strftime('%Y%m%d')}.jsonl"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    )
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_ingestion_logger(name: str) -> IngestionLogger:
    """
    Get or create an ingestion logger with the given name.
    
    Args:
        name: Logger name (e.g., 'ingest_teams', 'ingest_players')
    
    Returns:
        Configured IngestionLogger instance
    """
    return IngestionLogger(name)


def get_validation_logger(name: str = "validation") -> IngestionLogger:
    """
    Get or create a validation logger with the given name.
    
    Args:
        name: Logger name (default: 'validation')
    
    Returns:
        Configured IngestionLogger instance for validation
    """
    return IngestionLogger(f"validation.{name}")


def get_system_logger(name: str = "system") -> IngestionLogger:
    """
    Get or create a system logger with the given name.
    
    Args:
        name: Logger name (default: 'system')
    
    Returns:
        Configured IngestionLogger instance for system operations
    """
    return IngestionLogger(f"system.{name}")


if __name__ == "__main__":
    # Test the logging system
    print("Testing logging system...")
    
    # Test ingestion logger
    logger = get_ingestion_logger("test_ingestion")
    logger.log_ingestion_start("Sleeper API", season=2024, week=1)
    logger.info("This is a test info message")
    logger.warning("This is a test warning")
    logger.log_ingestion_complete(row_count=100, expected_count=100, duration_seconds=2.5)
    
    print(f"\nLog file created at: {logger.log_file}")
    print("Logging system test complete!")
