"""DuckDB Storage Layer - Qlib-style data management with DuckDB backend.

This module provides:
- Database schema management (OHLCV, instruments, calendars, factor_cache)
- CSV import/export functionality
- Data loading from various sources (DuckDB, Parquet, CSV)
- Data preprocessing (alignment, NaN handling, normalization)
- Query utilities

Usage:
    from data.storage import DuckDBStorage

    # Initialize
    db = DuckDBStorage("data/ohlcv.duckdb")

    # Import CSV
    db.import_csv("daily.csv", table="daily_ohlcv")

    # Load as MultiIndex dict (for factor computation)
    data = db.load(symbols=["000001"], start="2024-01-01", end="2024-12-31")

    # Preprocess
    data = db.preprocess.align(data)
    data = db.preprocess.winsorize(data)

    # Export
    db.export_csv("SELECT * FROM daily_ohlcv", "output.csv")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import numpy as np
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
# DataPreprocessor Class (merged from data/preprocessor.py)
# =============================================================================

class DataPreprocessor:
    """Align, clean, and prepare data for factor computation."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self._raw = data

    def align(self) -> dict[str, pd.DataFrame]:
        """Ensure all DataFrames share the same MultiIndex (date, stock)."""
        common_idx = None
        for name, df in self._raw.items():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        aligned = {}
        for name, df in self._raw.items():
            aligned[name] = df.loc[common_idx].sort_index()
        return aligned

    def dropna(self, how: str = "any", pct: float = 0.5) -> dict[str, pd.DataFrame]:
        """Drop stocks with too many NaN values."""
        aligned = self.align()
        cols_to_check = ["close", "volume"]
        valid_stocks = set()
        for col in cols_to_check:
            if col not in aligned:
                continue
            df = aligned[col]
            notna_pct = 1 - df.groupby(level=1).apply(lambda x: x.isna().mean())
            valid = notna_pct[notna_pct >= pct].index
            valid_stocks = valid_stocks.intersection(valid) if valid_stocks else set(valid)
        return {k: v[v.index.get_level_values(1).isin(valid_stocks)] for k, v in aligned.items()}

    def winsorize(self, data: dict[str, pd.DataFrame], limits: float = 0.01) -> dict[str, pd.DataFrame]:
        """Winsorize extreme values per cross-section."""
        result = {}
        for name, df in data.items():
            def _clip(x):
                lo, hi = x.quantile(limits), x.quantile(1 - limits)
                return x.clip(lo, hi)
            result[name] = df.groupby(level=0).transform(_clip) if not df.empty else df
        return result

    def neutralise(self, data: pd.DataFrame, groupby: pd.Series = None) -> pd.DataFrame:
        """Cross-sectional neutralisation (e.g. market/sector)."""
        if groupby is None:
            return data - data.groupby(level=0).mean()
        return data

    def standardise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardisation per date."""
        mu = data.groupby(level=0).mean()
        sd = data.groupby(level=0).std()
        return (data - mu) / sd


# =============================================================================
# DuckDBStorage Class
# =============================================================================

class DuckDBStorage:
    """DuckDB storage manager for financial data.

    Provides a Qlib-compatible interface with DuckDB backend.
    Includes data loading and preprocessing capabilities.
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
        self.preprocess = DataPreprocessor({})  # Placeholder, updated on load

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
    # Data Loading (merged from data/loader.py)
    # -------------------------------------------------------------------------

    def load(
        self,
        symbols: list[str] = None,
        start: str = None,
        end: str = None,
        source: str = "duckdb",
        path: str = None,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data as MultiIndex dict for factor computation.

        Args:
            symbols: List of symbols to load.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            source: Data source ('duckdb', 'parquet', 'csv').
            path: Path to data directory (for parquet/csv sources).

        Returns:
            Dict with keys: open, high, low, close, volume, amount
            Each value is a DataFrame with MultiIndex (date, symbol).

        Example:
            data = db.load(symbols=["000001", "000002"], start="2024-01-01")
            # Returns: {"open": DataFrame, "high": DataFrame, ...}
        """
        if source == "duckdb":
            return self._load_duckdb(symbols, start, end)
        elif source == "parquet":
            return self._load_parquet(symbols, start, end, path)
        elif source == "csv":
            return self._load_csv(symbols, start, end, path)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _load_duckdb(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Load data from DuckDB."""
        db_path = self.db_path
        if db_path == ":memory:":
            raise ValueError("Cannot load from in-memory database. Use a file-based database.")

        con = duckdb.connect(str(db_path))

        where_parts = []
        if symbols:
            sym_list = ",".join(f"'{s}'" for s in symbols)
            where_parts.append(f"symbol IN ({sym_list})")
        if start:
            where_parts.append(f"date >= '{start}'")
        if end:
            where_parts.append(f"date <= '{end}'")

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        query = f"""
            SELECT date, symbol, open, high, low, close, volume, amount
            FROM daily_ohlcv
            WHERE {where_clause}
            ORDER BY date, symbol
        """
        df = con.execute(query).fetchdf()
        con.close()
        return self._to_multiindex(df)

    def _load_parquet(
        self,
        symbols: list[str],
        start: str,
        end: str,
        path: str,
    ) -> dict[str, pd.DataFrame]:
        """Load data from Parquet files."""
        import glob

        if path is None:
            raise ValueError("path is required for parquet source")

        files = glob.glob(str(Path(path) / "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")

        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return self._to_multiindex(df)

    def _load_csv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        path: str,
    ) -> dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        import glob

        if path is None:
            raise ValueError("path is required for csv source")

        files = glob.glob(str(Path(path) / "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {path}")

        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=["date"])
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return self._to_multiindex(df)

    def _to_multiindex(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Convert flat DataFrame to MultiIndex dict for factor computation."""
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "symbol"]).sort_index()

        # Update preprocessor with loaded data
        self.preprocess = DataPreprocessor({col: df[[col]] for col in df.columns})

        result = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                result[col] = df[col].to_frame()
        if "amount" in df.columns:
            result["amount"] = df["amount"].to_frame()
        return result

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

    def validate(self) -> dict[str, Any]:
        """Validate database schema and data integrity.

        Returns:
            Dict with validation results.
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check tables exist
        tables = self.list_tables()
        required_tables = ["daily_ohlcv", "instruments", "calendars"]
        for table in required_tables:
            if table not in tables:
                results["valid"] = False
                results["errors"].append(f"Missing required table: {table}")

        # Check daily_ohlcv structure
        if "daily_ohlcv" in tables:
            schema = self.show_schema()
            ohlcv_cols = [col["column_name"] for col in schema if col["table_name"] == "daily_ohlcv"]
            required_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in ohlcv_cols:
                    results["valid"] = False
                    results["errors"].append(f"Missing required column in daily_ohlcv: {col}")

        # Check for null primary keys
        if "daily_ohlcv" in tables:
            null_count = self.fetchone(
                "SELECT COUNT(*) FROM daily_ohlcv WHERE date IS NULL OR symbol IS NULL"
            )[0]
            if null_count > 0:
                results["warnings"].append(f"Found {null_count} rows with NULL date or symbol")

        # Check for duplicate (date, symbol) pairs
        if "daily_ohlcv" in tables:
            dup_count = self.fetchone(
                "SELECT COUNT(*) FROM (SELECT date, symbol, COUNT(*) as cnt FROM daily_ohlcv GROUP BY date, symbol HAVING cnt > 1)"
            )[0]
            if dup_count > 0:
                results["warnings"].append(f"Found {dup_count} duplicate (date, symbol) pairs")

        return results
