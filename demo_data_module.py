#!/usr/bin/env python
"""Demo script for DuckDB data module.

This script demonstrates:
1. Creating a DuckDB database
2. Importing CSV data
3. Running factor expressions
4. Querying and exporting data

Usage:
    python demo_data_module.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.storage import DuckDBStorage
from factors.ops import rolling_mean, ts_rank, cs_rank, ts_corr


def create_sample_data(n_days: int = 100, n_symbols: int = 10) -> pd.DataFrame:
    """Create sample OHLCV data for demo."""
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]

    records = []
    for symbol in symbols:
        # Generate random price data
        price = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "symbol": symbol,
                "open": price[i] * (1 + np.random.randn() * 0.01),
                "high": price[i] * (1 + abs(np.random.randn()) * 0.02),
                "low": price[i] * (1 - abs(np.random.randn()) * 0.02),
                "close": price[i],
                "volume": np.random.randint(1e6, 1e8),
                "amount": price[i] * np.random.randint(1e6, 1e8),
            })

    return pd.DataFrame(records)


def demo_basic_operations():
    """Demo basic storage operations."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Storage Operations")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    try:
        # Initialize
        db = DuckDBStorage(db_path)
        print(f"\n✅ Created database: {db_path}")

        # Create sample data
        print("\n📊 Creating sample data...")
        df = create_sample_data(n_days=100, n_symbols=10)
        df.to_csv("sample_data.csv", index=False)
        print(f"   Saved {len(df):,} rows to sample_data.csv")

        # Import CSV
        rows = db.import_csv("sample_data.csv", table="daily_ohlcv")
        print(f"\n✅ Imported {rows:,} rows")

        # Query
        print("\n📋 Sample query:")
        result = db.query("""
            SELECT date, symbol, close
            FROM daily_ohlcv
            WHERE symbol = '000001'
            ORDER BY date
            LIMIT 5
        """)
        print(result.to_string(index=False))

        # Stats
        info = db.info()
        print(f"\n📊 Database info:")
        print(f"   Tables: {list(info['tables'].keys())}")
        print(f"   OHLCV rows: {info['tables']['daily_ohlcv']['rows']:,}")

        # Get instruments
        symbols = db.get_instruments()
        print(f"   Symbols: {len(symbols)}")

        # Date range
        date_range = db.date_range()
        print(f"   Date range: {date_range[0]} to {date_range[1]}")

        # Export
        db.export_csv(
            "SELECT * FROM daily_ohlcv WHERE symbol = '000001' LIMIT 10",
            "export_test.csv"
        )
        print("\n✅ Exported to export_test.csv")

        os.unlink("sample_data.csv")
        os.unlink("export_test.csv")

    finally:
        os.unlink(db_path)
        print(f"\n🧹 Cleaned up temporary database")


def demo_expression_evaluation():
    """Demo factor expression evaluation."""
    print("\n" + "=" * 60)
    print("Demo 2: Factor Expression Evaluation")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    try:
        db = DuckDBStorage(db_path)

        # Create sample data
        df = create_sample_data(n_days=100, n_symbols=10)

        # Group by symbol for time series operations
        df = df.sort_values(["symbol", "date"])
        df = df.set_index(["symbol", "date"])

        print("\n📊 Sample data shape:", df.shape)

        # Calculate factors using operators
        print("\n🧮 Calculating factors...")

        # 1. Simple rolling mean (time series)
        df["ma5"] = df.groupby(level="symbol")["close"].transform(
            lambda x: rolling_mean(x.values, 5)
        )
        print("   ✅ MA5 (5-day moving average)")

        df["ma20"] = df.groupby(level="symbol")["close"].transform(
            lambda x: rolling_mean(x.values, 20)
        )
        print("   ✅ MA20 (20-day moving average)")

        # 2. Time series rank
        df["ts_rank_10"] = df.groupby(level="symbol")["close"].transform(
            lambda x: ts_rank(x.values, 10)
        )
        print("   ✅ TS_Rank_10 (10-day time series rank)")

        # 3. Cross-sectional rank
        df["cs_rank"] = df.groupby(level="date")["close"].transform(
            lambda x: cs_rank(x.values)
        )
        print("   ✅ CS_Rank (cross-sectional rank)")

        # 4. Rolling correlation
        df["corr_10"] = df.groupby(level="symbol").apply(
            lambda g: pd.Series(
                ts_corr(g["close"].values, g["volume"].values, 10),
                index=g.index
            )
        ).reset_index(level=0, drop=True)
        print("   ✅ Corr_10 (10-day correlation with volume)")

        # Show results
        print("\n📋 Sample results:")
        sample = df.reset_index()
        sample = sample[sample["symbol"] == "000001"].head(10)
        print(sample[["date", "symbol", "close", "ma5", "ma20", "ts_rank_10", "cs_rank"]].to_string(index=False))

        # Statistics
        print("\n📊 Factor statistics:")
        factors = ["ma5", "ma20", "ts_rank_10", "cs_rank", "corr_10"]
        for f in factors:
            if f in df.columns:
                valid = df[f].dropna()
                print(f"   {f:12} mean={valid.mean():8.4f}  std={valid.std():8.4f}  non-null={len(valid):,}")

    finally:
        os.unlink(db_path)
        print(f"\n🧹 Cleaned up temporary database")


def demo_duckdb_query():
    """Demo DuckDB SQL features."""
    print("\n" + "=" * 60)
    print("Demo 3: DuckDB SQL Features")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    try:
        db = DuckDBStorage(db_path)

        # Create sample data
        df = create_sample_data(n_days=100, n_symbols=10)
        db.import_csv("sample_data.csv", table="daily_ohlcv")

        # Example 1: Daily returns
        print("\n📈 Calculating daily returns...")
        returns = db.query("""
            WITH ranked AS (
                SELECT
                    symbol,
                    date,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                FROM daily_ohlcv
            )
            SELECT
                date,
                AVG((close - prev_close) / prev_close * 100) as avg_return,
                STDDEV((close - prev_close) / prev_close * 100) as vol
            FROM ranked
            GROUP BY date
            ORDER BY date
            LIMIT 10
        """)
        print(returns.to_string(index=False))

        # Example 2: Monthly aggregation
        print("\n📅 Monthly aggregation...")
        monthly = db.query("""
            SELECT
                STRFTIME(date, '%Y-%m') as month,
                symbol,
                COUNT(*) as days,
                MIN(low) as month_low,
                MAX(high) as month_high,
                LAST(close) as month_close,
                SUM(amount) as total_amount
            FROM daily_ohlcv
            GROUP BY month, symbol
            ORDER BY month DESC, symbol
            LIMIT 15
        """)
        print(monthly.to_string(index=False))

        # Example 3: Rolling statistics
        print("\n📊 Rolling 5-day statistics...")
        rolling = db.query("""
            WITH windowed AS (
                SELECT
                    symbol,
                    date,
                    close,
                    AVG(close) OVER (
                        PARTITION BY symbol
                        ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) as ma5,
                    STDDEV(close) OVER (
                        PARTITION BY symbol
                        ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) as std5
                FROM daily_ohlcv
            )
            SELECT *
            FROM windowed
            WHERE symbol = '000001'
            ORDER BY date
            LIMIT 10
        """)
        print(rolling.to_string(index=False))

        os.unlink("sample_data.csv")

    finally:
        os.unlink(db_path)
        print(f"\n🧹 Cleaned up temporary database")


def demo_cli_commands():
    """Show CLI command examples."""
    print("\n" + "=" * 60)
    print("CLI Commands Reference")
    print("=" * 60)

    print("""
# Initialize database
python -m data.cli init --db data/ohlcv.duckdb

# Show database info
python -m data.cli info --db data/ohlcv.duckdb

# Import CSV
python -m data.cli import-csv data.csv --db data/ohlcv.duckdb --table daily_ohlcv

# Batch import directory
python -m data.cli import-dir ./csv_data/ --db data/ohlcv.duckdb --table daily_ohlcv

# Query data
python -m data.cli query "SELECT * FROM daily_ohlcv LIMIT 10"

# Export to CSV
python -m data.cli export "SELECT * FROM daily_ohlcv" --output data.csv

# Show statistics
python -m data.cli stats --table daily_ohlcv

# List instruments
python -m data.cli instruments --db data/ohlcv.duckdb

# Show trading calendar
python -m data.cli calendar --db data/ohlcv.duckdb --start 2024-01-01 --end 2024-12-31

# Or use standalone script
python scripts/csv2duckdb.py --init-only --db data/ohlcv.duckdb
python scripts/csv2duckdb.py data.csv --db data/ohlcv.duckdb --table daily_ohlcv
python scripts/csv2duckdb.py --batch ./csv_data/ --db data/ohlcv.duckdb --table daily_ohlcv
    """)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("DuckDB Data Module Demo")
    print("=" * 60)

    demo_basic_operations()
    demo_expression_evaluation()
    demo_duckdb_query()
    demo_cli_commands()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
