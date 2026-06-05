#!/usr/bin/env python
"""CSV to DuckDB Converter.

A standalone script to import CSV data into DuckDB with Qlib-compatible schema.

Usage:
    python scripts/csv2duckdb.py data.csv --db data/ohlcv.duckdb --table daily_ohlcv
    python scripts/csv2duckdb.py --init-only --db data/ohlcv.duckdb
    python scripts/csv2duckdb.py --batch ./csv_files/ --db data/ohlcv.duckdb

Requirements:
    pip install duckdb pandas click
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON daily_ohlcv(date)",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON daily_ohlcv(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_date_symbol ON daily_ohlcv(date, symbol)",
]


# =============================================================================
# Column Mappings for Common Data Sources
# =============================================================================

# Tushare format
TUSHARE_COLUMNS = {
    "trade_date": "date",
    "ts_code": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "vol": "volume",
    "amount": "amount",
}

# AKShare format
AKSHARE_COLUMNS = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}

# JoinQuant format
JOINQUANT_COLUMNS = {
    "date": "date",
    "code": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "money": "amount",
}

# Default mapping
DEFAULT_COLUMNS = {
    "date": "date",
    "symbol": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "factor": "factor",
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_column_mapping(columns: list[str], source: str = "auto") -> dict[str, str]:
    """Detect and return column mapping based on source format."""
    if source == "tushare":
        return TUSHARE_COLUMNS
    elif source == "akshare":
        return AKSHARE_COLUMNS
    elif source == "joinquant":
        return JOINQUANT_COLUMNS

    # Auto-detect
    col_lower = {c.lower() for c in columns}

    if "trade_date" in col_lower or "ts_code" in col_lower:
        return TUSHARE_COLUMNS
    elif "日期" in columns or "股票代码" in columns:
        return AKSHARE_COLUMNS
    elif "code" in col_lower and "date" in col_lower:
        return JOINQUANT_COLUMNS

    return DEFAULT_COLUMNS


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date string to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Already in correct format
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str

    # Try parsing with pandas
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol."""
    if pd.isna(symbol):
        return None
    symbol = str(symbol).strip()
    # Remove .SH, .SZ prefixes if present, keep just the code
    symbol = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
    return symbol


def init_database(db_path: str) -> duckdb.DuckDBPyConnection:
    """Initialize database schema."""
    print(f"📦 Initializing database: {db_path}")

    conn = duckdb.connect(db_path)

    # Create tables
    conn.execute(SCHEMA_DAILY_OHLCV)
    conn.execute(SCHEMA_INSTRUMENTS)

    # Create indexes
    for idx_sql in INDEXES:
        conn.execute(idx_sql)

    conn.commit()
    print("✅ Database schema initialized!")

    return conn


def preview_csv(csv_path: str, nrows: int = 5) -> pd.DataFrame:
    """Preview CSV file."""
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"\n📄 File: {csv_path}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    print(f"   Shape: {df.shape[0]} rows (preview), {df.shape[1]} columns")
    return df


def import_csv(
    conn: duckdb.DuckDBPyConnection,
    csv_path: str,
    table: str,
    source: str = "auto",
    if_exists: str = "append",
    date_col: str = "date",
    symbol_col: str = "symbol",
    skip_rows: int = 0,
    encoding: str = "utf-8",
    verbose: bool = True,
) -> int:
    """Import CSV file into DuckDB table."""
    if verbose:
        print(f"\n📥 Importing: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path, skiprows=skip_rows, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, skiprows=skip_rows, encoding="gbk")
        if verbose:
            print("   (Using GBK encoding)")

    if df.empty:
        if verbose:
            print("   ⚠️  Empty file, skipping")
        return 0

    # Get column mapping
    columns_map = get_column_mapping(df.columns.tolist(), source)

    # Rename columns
    rename_map = {}
    for old_col, new_col in columns_map.items():
        if old_col in df.columns:
            rename_map[old_col] = new_col

    df = df.rename(columns=rename_map)

    # Keep only relevant columns
    keep_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "factor"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Normalize data
    if "date" in df.columns:
        df["date"] = df["date"].apply(normalize_date)

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].apply(normalize_symbol)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "factor"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle if_exists
    if if_exists == "replace":
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        if table == "daily_ohlcv":
            conn.execute(SCHEMA_DAILY_OHLCV)
        elif table == "instruments":
            conn.execute(SCHEMA_INSTRUMENTS)

    # Import
    try:
        conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df")
    except Exception as e:
        if "duplicate" in str(e).lower() or "primary key" in str(e).lower():
            # Handle duplicates with REPLACE strategy
            conn.execute(f"""
                INSERT INTO {table} BY NAME
                SELECT * FROM df
                WHERE (date, symbol) NOT IN (SELECT date, symbol FROM {table})
            """)
        else:
            raise

    conn.commit()

    if verbose:
        print(f"   ✅ Imported {len(df):,} rows into '{table}'")

    return len(df)


def show_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Show simple progress."""
    percent = 100 * current / total if total > 0 else 100
    print(f"\r{prefix}: {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current >= total:
        print()


# =============================================================================
# Main Functions
# =============================================================================

def cmd_init(args) -> None:
    """Initialize database."""
    db_path = args.db

    # Create parent directory if needed
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = init_database(db_path)
    conn.close()


def cmd_import(args) -> None:
    """Import CSV file."""
    db_path = args.db
    csv_path = args.csv

    # Create connection
    if not os.path.exists(db_path):
        conn = init_database(db_path)
    else:
        conn = duckdb.connect(db_path)

    # Preview
    preview_csv(csv_path)

    # Import
    import_csv(
        conn,
        csv_path,
        table=args.table,
        source=args.source,
        if_exists=args.if_exists,
        skip_rows=args.skip_rows,
    )

    conn.close()


def cmd_batch(args) -> None:
    """Batch import directory."""
    db_path = args.db
    dir_path = args.dir

    # Get files
    pattern = os.path.join(dir_path, f"*.{args.pattern}")
    files = glob.glob(pattern)

    if not files:
        print(f"⚠️  No files found matching: {pattern}")
        return

    print(f"📦 Found {len(files)} files to import")

    # Create connection
    if not os.path.exists(db_path):
        conn = init_database(db_path)
    else:
        conn = duckdb.connect(db_path)

    total_rows = 0
    for i, f in enumerate(files):
        rows = import_csv(conn, f, table=args.table, source=args.source)
        total_rows += rows
        show_progress(i + 1, len(files), "Importing")

    conn.close()

    print(f"\n✅ Total imported: {total_rows:,} rows from {len(files)} files")


def cmd_stats(args) -> None:
    """Show database statistics."""
    db_path = args.db

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    conn = duckdb.connect(db_path)

    print(f"\n📊 Database: {db_path}")
    print(f"   Size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")

    # Tables
    tables = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()

    print(f"\n📋 Tables:")
    for (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"   - {table}: {count:,} rows")

    # OHLCV summary
    if conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'daily_ohlcv'
    """).fetchone()[0]:
        date_range = conn.execute("SELECT MIN(date), MAX(date) FROM daily_ohlcv").fetchone()
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_ohlcv").fetchone()[0]

        print(f"\n📈 OHLCV Data:")
        print(f"   Date Range: {date_range[0]} to {date_range[1]}")
        print(f"   Symbols: {symbols:,}")

    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CSV to DuckDB Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize empty database
  python csv2duckdb.py --init-only --db data/ohlcv.duckdb

  # Import single CSV
  python csv2duckdb.py data.csv --db data/ohlcv.duckdb --table daily_ohlcv

  # Import with auto-detected format
  python csv2duckdb.py data.csv --db data/ohlcv.duckdb --table daily_ohlcv --source tushare

  # Batch import directory
  python csv2duckdb.py --batch ./csv_data/ --db data/ohlcv.duckdb --table daily_ohlcv

  # Show statistics
  python csv2duckdb.py --stats --db data/ohlcv.duckdb
        """
    )

    parser.add_argument("--db", default="data/ohlcv.duckdb", help="DuckDB database path")
    parser.add_argument("--init-only", action="store_true", help="Only initialize schema")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--batch", help="Directory to batch import")

    # Import options
    parser.add_argument("csv", nargs="?", help="CSV file to import")
    parser.add_argument("--table", default="daily_ohlcv", help="Target table name")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "tushare", "akshare", "joinquant"],
                        help="Data source format")
    parser.add_argument("--if-exists", default="append",
                        choices=["append", "replace"],
                        help="How to handle existing data")
    parser.add_argument("--skip-rows", type=int, default=0, help="Rows to skip")
    parser.add_argument("--pattern", default="csv", help="File pattern for batch import")

    args = parser.parse_args()

    # Execute command
    if args.init_only:
        cmd_init(args)
    elif args.stats:
        cmd_stats(args)
    elif args.batch:
        cmd_batch(args)
    elif args.csv:
        cmd_import(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
