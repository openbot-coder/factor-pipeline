"""Data module - storage and data management."""

# QuantDB - 量化数据库 (新)
from factor_pipeline.data.quantdb import (
    IndexCode,
    Market,
    QuantDB,
    UpdateResult,
)
from factor_pipeline.data.storage import (
    SCHEMA_CALENDARS,
    SCHEMA_DAILY_OHLCV,
    SCHEMA_FACTOR_CACHE,
    SCHEMA_INSTRUMENTS,
    DataPreprocessor,
    DuckDBStorage,
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
