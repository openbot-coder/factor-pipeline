"""Data module for factor-pipeline.

This module provides:
- DuckDB storage management
- Data loading and querying
- CSV/Parquet import tools
- Expression evaluation
"""

from data.storage import DuckDBStorage

__all__ = ["DuckDBStorage"]
