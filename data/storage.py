"""DuckDB Storage Layer - Qlib-style data management with DuckDB backend.

This module provides:
- Database schema management (OHLCV, instruments, calendars, factor_cache)
- CSV import/export functionality
- Query utilities
- Connection pooling

Usage:
    from data.storage import DuckDBStorage

    # Initialize
    db = DuckDBStorage("data/ohlcv.duckdb")

    # Import CSV
    db.import_csv("daily.csv", table="daily_ohlcv")

    # Query
    df = db.query("SELECT * FROM daily_ohlcv WHERE date >= '2024-01-01'")

    # Export
    db.export_csv("SELECT * FROM daily_ohlcv", "output.csv")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import pandas as pd


# =============================================================================
# Schema Definitions
# =============================================================================

SCHEMA_DAILY_OHLCV = """
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    date DATE,
    symbol VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    amount DOUBLE,
    factor DOUBLE DEFAULT 1.0,
    PRIMARY KEY (date, symbol)
)
"""

SCHEMA_INSTRUMENTS = """
CREATE TABLE IF NOT EXISTS instruments (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    market VARCHAR,
    industry VARCHAR
)
"""

SCHEMA_CALENDARS = """
CREATE TABLE IF NOT EXISTS calendars (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN DEFAULT TRUE
)
"""

SCHEMA_FACTOR_CACHE = """
CREATE TABLE IF NOT EXISTS factor_cache (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE,
    expression TEXT,
    result_bytes BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON daily_ohlcv(date)",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON daily_ohlcv(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_date_symbol ON daily_ohlcv(date, symbol)",
    "CREATE INDEX IF NOT EXISTS idx_instruments_market ON instruments(market)",
]


# =============================================================================
# DuckDBStorage Class
# =============================================================================

class DuckDBStorage:
    """DuckDB storage manager for financial data.

    Provides a Qlib-compatible interface with DuckDB backend.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        config: Optional[dict] = None,
    ):
        """Initialize DuckDB storage.

        Args:
            db_path: Path to DuckDB file, or ":memory:" for in-memory.
            read_only: Open in read-only mode.
            config: DuckDB configuration options.
        """
        self.db_path = db_path
        self.read_only = read_only
        self.config = config or {}
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

        # Auto-create if file doesn't exist
        if db_path != ":memory:" and not os.path.exists(db_path):
            self._init_database()
        elif db_path != ":memory:":
            self.connect()

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(
                self.db_path,
                read_only=self.read_only,
                config=self.config,
            )
            # Enable auto-commit for writes
            self._conn.begin()
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckDBStorage":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        try:
            # Create tables
            conn.execute(SCHEMA_DAILY_OHLCV)
            conn.execute(SCHEMA_INSTRUMENTS)
            conn.execute(SCHEMA_CALENDARS)
            conn.execute(SCHEMA_FACTOR_CACHE)

            # Create indexes
            for idx_sql in INDEXES:
                conn.execute(idx_sql)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Initialize complete database schema."""
        self._init_database()

    def show_schema(self) -> list[dict]:
        """Show all tables and their schemas."""
        conn = self.connect()
        result = conn.execute("""
            SELECT table_name, column_name, column_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
        """).fetchdf()
        return result.to_dict("records")

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        conn = self.connect()
        result = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchdf()
        return result["table_name"].tolist()

    # -------------------------------------------------------------------------
    # Import / Export
    # -------------------------------------------------------------------------

    def import_csv(
        self,
        csv_path: str,
        table: str,
        date_col: str = "date",
        symbol_col: str = "symbol",
        if_exists: str = "append",
        parse_dates: Optional[list[str]] = None,
        columns_map: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> int:
        """Import CSV file into DuckDB table.

        Args:
            csv_path: Path to CSV file.
            table: Target table name.
            date_col: Name of date column (for filtering during import).
            symbol_col: Name of symbol column.
            if_exists: How to handle existing data ('append', 'replace', 'fail').
            parse_dates: List of date columns to parse.
            columns_map: Rename columns (old_name -> new_name).
            **kwargs: Additional arguments passed to pandas read_csv.

        Returns:
            Number of rows imported.

        Example:
            db.import_csv("daily.csv", "daily_ohlcv",
                          date_col="trade_date",
                          symbol_col="stock_code")
        """
        conn = self.connect()

        # Read CSV
        parse_dates = parse_dates or []
        if date_col not in parse_dates:
            parse_dates.append(date_col)

        df = pd.read_csv(csv_path, parse_dates=parse_dates, **kwargs)

        # Rename columns if needed
        if columns_map:
            df = df.rename(columns=columns_map)

        # Handle if_exists
        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df LIMIT 0")
            if_exists = "append"

        # Import
        if if_exists == "append":
            conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df")
        elif if_exists == "fail":
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count > 0:
                raise ValueError(f"Table {table} already has data. Use if_exists='replace' or 'append'.")

        conn.commit()
        return len(df)

    def import_parquet(
        self,
        parquet_path: str,
        table: str,
        if_exists: str = "append",
    ) -> int:
        """Import Parquet file into DuckDB table."""
        conn = self.connect()
        df = pd.read_parquet(parquet_path)

        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df")
        conn.commit()
        return len(df)

    def import_excel(
        self,
        excel_path: str,
        table: str,
        sheet_name: Union[str, int] = 0,
        if_exists: str = "append",
    ) -> int:
        """Import Excel file into DuckDB table."""
        conn = self.connect()
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df")
        conn.commit()
        return len(df)

    def export_csv(
        self,
        query_or_table: str,
        output_path: str,
        header: bool = True,
        delimiter: str = ",",
    ) -> None:
        """Export query result or table to CSV.

        Args:
            query_or_table: SQL query or table name.
            output_path: Output CSV file path.
            header: Include header row.
            delimiter: CSV delimiter.
        """
        conn = self.connect()

        # Check if it's a table or query
        tables = self.list_tables()
        if query_or_table in tables:
            df = conn.execute(f"SELECT * FROM {query_or_table}").fetchdf()
        else:
            df = conn.execute(query_or_table).fetchdf()

        df.to_csv(output_path, index=False, header=header, sep=delimiter)

    def export_parquet(
        self,
        query_or_table: str,
        output_path: str,
    ) -> None:
        """Export query result or table to Parquet."""
        conn = self.connect()

        tables = self.list_tables()
        if query_or_table in tables:
            df = conn.execute(f"SELECT * FROM {query_or_table}").fetchdf()
        else:
            df = conn.execute(query_or_table).fetchdf()

        df.to_parquet(output_path, index=False)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            Query result as DataFrame.
        """
        conn = self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()

    def execute(self, sql: str, params: Optional[dict] = None) -> duckdb.DuckDBPyConnection:
        """Execute SQL without returning results."""
        conn = self.connect()
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)
        conn.commit()
        return conn

    def fetchone(self, sql: str) -> Optional[tuple]:
        """Execute SQL and fetch one row."""
        conn = self.connect()
        return conn.execute(sql).fetchone()

    # -------------------------------------------------------------------------
    # OHLCV Specific Methods
    # -------------------------------------------------------------------------

    def get_instruments(
        self,
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        """Get list of instruments (symbols).

        Args:
            market: Filter by market (e.g., 'SSE', 'SZSE').
            start_date: Filter by list_date <= start_date.
            end_date: Filter by delist_date >= end_date.

        Returns:
            List of symbol strings.
        """
        conditions = []
        if market:
            conditions.append(f"market = '{market}'")
        if start_date:
            conditions.append(f"(list_date IS NULL OR list_date <= '{start_date}')")
        if end_date:
            conditions.append(f"(delist_date IS NULL OR delist_date >= '{end_date}')")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT symbol FROM instruments WHERE {where}"

        return self.query(sql)["symbol"].tolist()

    def get_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        """Get trading calendar dates.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            List of date strings.
        """
        conditions = ["is_trading_day = TRUE"]
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")

        where = " AND ".join(conditions)
        sql = f"SELECT date FROM calendars WHERE {where} ORDER BY date"

        return [str(d.date()) for d in self.query(sql)["date"].tolist()]

    def get_ohlcv(
        self,
        symbols: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        index: bool = True,
    ) -> pd.DataFrame:
        """Get OHLCV data.

        Args:
            symbols: List of symbols (None = all).
            fields: List of columns (None = all).
            start_date: Start date.
            end_date: End date.
            index: Return with MultiIndex (date, symbol).

        Returns:
            OHLCV DataFrame.
        """
        fields = fields or ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        select_fields = ", ".join(fields)

        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT {select_fields} FROM daily_ohlcv WHERE {where} ORDER BY date, symbol"

        df = self.query(sql)

        if index and "date" in df.columns and "symbol" in df.columns:
            df = df.set_index(["date", "symbol"])

        return df

    def count_ohlcv(self) -> int:
        """Get total number of OHLCV records."""
        return self.fetchone("SELECT COUNT(*) FROM daily_ohlcv")[0]

    def date_range(self) -> tuple[str, str]:
        """Get min and max dates in OHLCV data."""
        row = self.fetchone("SELECT MIN(date), MAX(date) FROM daily_ohlcv")
        return (str(row[0]), str(row[1])) if row else (None, None)

    # -------------------------------------------------------------------------
    # Database Info
    # -------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Get database information."""
        tables = self.list_tables()

        info_dict = {
            "db_path": self.db_path,
            "db_size_mb": 0,
            "tables": {},
        }

        # File size
        if self.db_path != ":memory:" and os.path.exists(self.db_path):
            info_dict["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)

        # Table info
        for table in tables:
            try:
                count = self.fetchone(f"SELECT COUNT(*) FROM {table}")[0]
                info_dict["tables"][table] = {"rows": count}
            except Exception:
                info_dict["tables"][table] = {"rows": -1}

        # OHLCV summary
        if "daily_ohlcv" in tables:
            date_range = self.date_range()
            info_dict["date_range"] = date_range
            info_dict["symbols_count"] = len(self.get_instruments())

        return info_dict

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def vacuum(self) -> None:
        """Optimize database storage."""
        self.execute("VACUUM")

    def checkpoint(self) -> None:
        """Checkpoint WAL to main database."""
        self.execute("CHECKPOINT")

    def drop_table(self, table: str) -> None:
        """Drop a table."""
        self.execute(f"DROP TABLE IF EXISTS {table}")

    def truncate_table(self, table: str) -> None:
        """Truncate a table."""
        self.execute(f"DELETE FROM {table}")

    def copy_table(self, source: str, target: str) -> None:
        """Copy a table."""
        self.execute(f"CREATE TABLE {target} AS SELECT * FROM {source}")
