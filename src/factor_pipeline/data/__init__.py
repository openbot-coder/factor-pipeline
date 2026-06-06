"""Data module - storage and data management."""

from factor_pipeline.data.storage import (
    DuckDBStorage,
    DataPreprocessor,
    SCHEMA_DAILY_OHLCV,
    SCHEMA_INSTRUMENTS,
    SCHEMA_CALENDARS,
    SCHEMA_FACTOR_CACHE,
)

# QuantDB - 量化数据库 (新)
from factor_pipeline.data.quantdb import (
    QuantDB,
    Market,
    IndexCode,
    UpdateResult,
)

__all__ = [
    # DuckDBStorage (legacy)
    "DuckDBStorage",
    "DataPreprocessor",
    "SCHEMA_DAILY_OHLCV",
    "SCHEMA_INSTRUMENTS",
    "SCHEMA_CALENDARS",
    "SCHEMA_FACTOR_CACHE",
    # QuantDB (new)
    "QuantDB",
    "Market",
    "IndexCode",
    "UpdateResult",
]
