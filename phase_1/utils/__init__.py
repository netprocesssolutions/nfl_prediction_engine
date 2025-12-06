"""
Utilities package for NFL Fantasy Prediction Engine.

This package contains shared utility modules:
- database: Database connection management
- logger: Structured logging
"""

from .database import (
    DatabaseConnection,
    DataVersionManager,
    get_db,
    get_version_manager,
)

from .logger import (
    get_ingestion_logger,
    get_validation_logger,
    get_system_logger,
)

__all__ = [
    # Database
    "DatabaseConnection",
    "DataVersionManager", 
    "get_db",
    "get_version_manager",
    
    # Logger
    "get_ingestion_logger",
    "get_validation_logger",
    "get_system_logger",
]
