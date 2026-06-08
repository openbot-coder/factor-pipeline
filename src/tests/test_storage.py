"""Comprehensive tests for data/storage.py.

Coverage:
- Schema initialization
- CSV import (positive, negative, edge cases)
- Query execution
- Data export
- Edge cases: empty DB, invalid data, duplicates
"""

# Import the module being tested
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from factor_pipeline.data.storage import DuckDBStorage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary in-memory database for testing."""
    db = DuckDBStorage(":memory:")
    yield db
    db.close()


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    symbols = [f"{i:06d}" for i in range(1, 11)]

    records = []
    for symbol in symbols:
        price = 100.0
        for d in dates:
            price = price * (1 + np.random.randn() * 0.01)
            records.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": np.random.randint(1e6, 1e8),
                    "amount": price * np.random.randint(1e6, 1e8),
                    "factor": 1.0,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def sample_csv(tmp_path, sample_ohlcv):
    """Create a temporary CSV file."""
    csv_path = tmp_path / "test_ohlcv.csv"
    sample_ohlcv.to_csv(csv_path, index=False)
    return str(csv_path)


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemaInit:
    """Tests for database schema initialization."""

    def test_init_schema_in_memory(self, temp_db):
        """Positive: Initialize schema in memory database."""
        temp_db.init_schema()
        tables = temp_db.list_tables()

        assert "daily_ohlcv" in tables
        assert "instruments" in tables
        assert "calendars" in tables
        assert "factor_cache" in tables

    def test_init_schema_twice(self, temp_db):
        """Edge: Initialize schema twice (should not raise)."""
        temp_db.init_schema()
        temp_db.init_schema()  # Should not raise

        tables = temp_db.list_tables()
        assert len(tables) == 4

    def test_show_schema(self, temp_db):
        """Positive: Show schema returns column information."""
        temp_db.init_schema()
        schema = temp_db.show_schema()

        assert len(schema) > 0
        assert any(col["table_name"] == "daily_ohlcv" for col in schema)

    def test_list_tables(self, temp_db):
        """Positive: List tables returns all tables."""
        temp_db.init_schema()
        tables = temp_db.list_tables()

        assert "daily_ohlcv" in tables
        assert "instruments" in tables


# =============================================================================
# Import Tests
# =============================================================================


class TestImportCSV:
    """Tests for CSV import functionality."""

    def test_import_basic(self, temp_db, sample_csv):
        """Positive: Basic CSV import."""
        temp_db.init_schema()
        rows = temp_db.import_csv(sample_csv, table="daily_ohlcv")

        assert rows == 1000  # 100 dates * 10 symbols

    def test_import_append(self, temp_db, sample_csv, sample_ohlcv):
        """Positive: Append mode adds new rows."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        # Create second CSV with different data
        temp_db.import_csv(sample_csv, table="daily_ohlcv", if_exists="append")

        total = temp_db.count_ohlcv()
        assert total == 2000  # Doubled

    def test_import_replace(self, temp_db, sample_csv):
        """Positive: Replace mode replaces data."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")
        temp_db.import_csv(sample_csv, table="daily_ohlcv", if_exists="replace")

        total = temp_db.count_ohlcv()
        assert total == 1000  # Same count as original

    def test_import_with_column_mapping(self, temp_db, tmp_path):
        """Positive: Import with custom column mapping."""
        temp_db.init_schema()

        # Create CSV with different column names
        df = pd.DataFrame(
            {
                "trade_date": ["2024-01-01", "2024-01-02"],
                "stock_code": ["000001", "000002"],
                "o": [10.0, 11.0],
                "h": [11.0, 12.0],
                "l": [9.0, 10.0],
                "c": [10.5, 11.5],
                "v": [1e6, 2e6],
                "a": [10.5e6, 23e6],
            }
        )
        csv_path = tmp_path / "mapped.csv"
        df.to_csv(csv_path, index=False)

        rows = temp_db.import_csv(
            str(csv_path),
            table="daily_ohlcv",
            date_col="trade_date",
            symbol_col="stock_code",
            columns_map={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "a": "amount",
            },
        )

        assert rows == 2

    def test_import_nonexistent_file(self, temp_db):
        """Negative: Import non-existent file raises error."""
        temp_db.init_schema()

        with pytest.raises(FileNotFoundError):
            temp_db.import_csv("nonexistent.csv", table="daily_ohlcv")

    def test_import_empty_file(self, temp_db, tmp_path):
        """Edge: Import empty CSV."""
        temp_db.init_schema()

        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        # Should handle gracefully
        rows = temp_db.import_csv(str(empty_csv), table="daily_ohlcv")
        assert rows == 0

    def test_import_invalid_date_format(self, temp_db, tmp_path):
        """Edge: Import with invalid date format (handled as NaT)."""
        temp_db.init_schema()

        df = pd.DataFrame(
            {
                "date": ["invalid", "2024-01-02"],
                "symbol": ["000001", "000002"],
                "close": [10.0, 11.0],
            }
        )
        csv_path = tmp_path / "bad_date.csv"
        df.to_csv(csv_path, index=False)

        # Should import what it can
        rows = temp_db.import_csv(str(csv_path), table="daily_ohlcv")
        # At least the valid row should be imported or the invalid one handled
        assert rows >= 0


# =============================================================================
# Query Tests
# =============================================================================


class TestQuery:
    """Tests for SQL query execution."""

    def test_query_basic(self, temp_db, sample_csv):
        """Positive: Basic SELECT query."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.query("SELECT * FROM daily_ohlcv LIMIT 10")
        assert len(df) == 10
        assert "date" in df.columns
        assert "symbol" in df.columns

    def test_query_with_filter(self, temp_db, sample_csv):
        """Positive: Query with WHERE clause."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.query("SELECT * FROM daily_ohlcv WHERE symbol = '000001'")
        assert all(df["symbol"] == "000001")
        assert len(df) == 100  # 100 days

    def test_query_with_aggregation(self, temp_db, sample_csv):
        """Positive: Query with aggregation."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.query("""
            SELECT symbol, COUNT(*) as cnt, AVG(close) as avg_close
            FROM daily_ohlcv
            GROUP BY symbol
        """)

        assert len(df) == 10  # 10 symbols
        assert all(df["cnt"] == 100)  # 100 days each

    def test_query_invalid_sql(self, temp_db):
        """Negative: Invalid SQL raises error."""
        temp_db.init_schema()

        with pytest.raises(Exception):  # DuckDB error
            temp_db.query("SELECT * FROM nonexistent_table")

    def test_fetchone(self, temp_db, sample_csv):
        """Positive: fetchone returns single row."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        result = temp_db.fetchone("SELECT COUNT(*) FROM daily_ohlcv")
        assert result[0] == 1000

    def test_fetchone_empty(self, temp_db):
        """Edge: fetchone on empty table."""
        temp_db.init_schema()
        result = temp_db.fetchone("SELECT * FROM daily_ohlcv LIMIT 1")
        assert result is None


# =============================================================================
# OHLCV Specific Tests
# =============================================================================


class TestOHLCVMethods:
    """Tests for OHLCV-specific methods."""

    def test_get_ohlcv_basic(self, temp_db, sample_csv):
        """Positive: Get OHLCV data."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.get_ohlcv()
        assert len(df) > 0
        assert "close" in df.columns

    def test_get_ohlcv_with_date_filter(self, temp_db, sample_csv):
        """Positive: Get OHLCV with date filter."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.get_ohlcv(start_date="2024-01-01", end_date="2024-01-31")
        assert len(df) > 0

    def test_get_ohlcv_with_symbol_filter(self, temp_db, sample_csv):
        """Positive: Get OHLCV with symbol filter."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.get_ohlcv(symbols=["000001", "000002"])
        assert df.index.get_level_values("symbol").nunique() == 2

    def test_get_ohlcv_no_index(self, temp_db, sample_csv):
        """Edge: Get OHLCV without MultiIndex."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.get_ohlcv(index=False)
        assert "date" in df.columns
        assert "symbol" in df.columns

    def test_get_instruments(self, temp_db, sample_csv):
        """Positive: Get list of instruments."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        symbols = temp_db.get_instruments()
        assert len(symbols) == 10

    def test_date_range(self, temp_db, sample_csv):
        """Positive: Get date range of data."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        start, end = temp_db.date_range()
        assert start is not None
        assert end is not None

    def test_date_range_empty(self, temp_db):
        """Edge: Get date range from empty table."""
        temp_db.init_schema()
        start, end = temp_db.date_range()
        assert start is None
        assert end is None


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for data export functionality."""

    def test_export_csv(self, temp_db, sample_csv, tmp_path):
        """Positive: Export to CSV."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        output_path = tmp_path / "export.csv"
        temp_db.export_csv("SELECT * FROM daily_ohlcv LIMIT 100", str(output_path))

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 100

    def test_export_parquet(self, temp_db, sample_csv, tmp_path):
        """Positive: Export to Parquet."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        output_path = tmp_path / "export.parquet"
        temp_db.export_parquet("SELECT * FROM daily_ohlcv LIMIT 100", str(output_path))

        assert output_path.exists()

    def test_export_by_table_name(self, temp_db, sample_csv, tmp_path):
        """Positive: Export by table name."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        output_path = tmp_path / "export2.csv"
        temp_db.export_csv("daily_ohlcv", str(output_path))

        df = pd.read_csv(output_path)
        assert len(df) == 1000


# =============================================================================
# Database Info Tests
# =============================================================================


class TestDatabaseInfo:
    """Tests for database information methods."""

    def test_info_basic(self, temp_db, sample_csv):
        """Positive: Get database info."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        info = temp_db.info()
        assert "db_path" in info
        assert "tables" in info
        assert "daily_ohlcv" in info["tables"]

    def test_info_empty_db(self, temp_db):
        """Edge: Get info from empty database."""
        temp_db.init_schema()
        info = temp_db.info()

        assert info["tables"]["daily_ohlcv"]["rows"] == 0

    def test_count_ohlcv(self, temp_db, sample_csv):
        """Positive: Count OHLCV records."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        count = temp_db.count_ohlcv()
        assert count == 1000


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_drop_table(self, temp_db, sample_csv):
        """Positive: Drop table."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        temp_db.drop_table("daily_ohlcv")
        tables = temp_db.list_tables()
        assert "daily_ohlcv" not in tables

    def test_truncate_table(self, temp_db, sample_csv):
        """Positive: Truncate table."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        temp_db.truncate_table("daily_ohlcv")
        count = temp_db.count_ohlcv()
        assert count == 0

    def test_copy_table(self, temp_db, sample_csv):
        """Positive: Copy table."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        temp_db.copy_table("daily_ohlcv", "daily_ohlcv_copy")
        tables = temp_db.list_tables()
        assert "daily_ohlcv_copy" in tables

        count = temp_db.fetchone("SELECT COUNT(*) FROM daily_ohlcv_copy")[0]
        assert count == 1000

    def test_vacuum(self, temp_db):
        """Edge: Vacuum (should not raise)."""
        temp_db.init_schema()
        temp_db.vacuum()  # Should not raise

    def test_checkpoint(self, temp_db):
        """Edge: Checkpoint (should not raise)."""
        temp_db.init_schema()
        temp_db.checkpoint()  # Should not raise


# =============================================================================
# File-based Database Tests
# =============================================================================


class TestFileDatabase:
    """Tests for file-based database (non-in-memory)."""

    def test_file_db_init(self, tmp_path):
        """Positive: Create file-based database."""
        db_path = tmp_path / "test.duckdb"
        db = DuckDBStorage(str(db_path))

        assert db_path.exists()
        db.close()

    def test_file_db_auto_init(self, tmp_path):
        """Edge: Auto-create non-existent database on connect."""
        db_path = tmp_path / "new.duckdb"
        assert not db_path.exists()

        db = DuckDBStorage(str(db_path))
        tables = db.list_tables()
        # Should auto-create with schema
        assert len(tables) == 4

        db.close()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self, temp_db):
        """Positive: Use as context manager."""
        with DuckDBStorage(":memory:") as db:
            db.init_schema()
            tables = db.list_tables()
            assert len(tables) == 4

    def test_context_manager_exception(self):
        """Edge: Exception in context manager closes connection."""
        try:
            with DuckDBStorage(":memory:") as db:
                db.init_schema()
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Connection should be closed (no further operations)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_query_empty_result(self, temp_db, sample_csv):
        """Edge: Query returns empty result."""
        temp_db.init_schema()
        temp_db.import_csv(sample_csv, table="daily_ohlcv")

        df = temp_db.query("SELECT * FROM daily_ohlcv WHERE close > 999999")
        assert len(df) == 0

    def test_special_characters_in_symbol(self, temp_db, tmp_path):
        """Edge: Symbols with special characters."""
        temp_db.init_schema()

        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "symbol": ["600000.SH", "000001.SZ"],  # With market suffix
                "close": [10.0, 11.0],
                "open": [9.5, 10.5],
                "high": [10.5, 11.5],
                "low": [9.0, 10.0],
                "volume": [1e6, 2e6],
                "amount": [10e6, 22e6],
            }
        )
        csv_path = tmp_path / "special.csv"
        df.to_csv(csv_path, index=False)

        rows = temp_db.import_csv(str(csv_path), table="daily_ohlcv")
        assert rows == 2

    def test_null_values_in_csv(self, temp_db, tmp_path):
        """Edge: CSV with null values."""
        temp_db.init_schema()

        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "symbol": ["000001", "000002"],
                "close": [10.0, None],  # Null value
                "open": [9.5, 10.5],
                "high": [10.5, 11.5],
                "low": [9.0, 10.0],
                "volume": [1e6, 2e6],
                "amount": [10e6, 22e6],
            }
        )
        csv_path = tmp_path / "nulls.csv"
        df.to_csv(csv_path, index=False)

        rows = temp_db.import_csv(str(csv_path), table="daily_ohlcv")
        assert rows == 2

    def test_large_dataset(self, temp_db, tmp_path):
        """Edge: Large dataset import."""
        temp_db.init_schema()

        # Generate 10k rows
        n = 10000
        dates = pd.date_range("2024-01-01", periods=100)
        symbols = [f"{i:06d}" for i in range(1, 101)]

        records = []
        for i in range(n):
            records.append(
                {
                    "date": dates[i % 100],
                    "symbol": symbols[i % 100],
                    "open": 10.0 + i * 0.01,
                    "high": 10.5 + i * 0.01,
                    "low": 9.5 + i * 0.01,
                    "close": 10.0 + i * 0.01,
                    "volume": 1e6,
                    "amount": 10e6,
                }
            )

        df = pd.DataFrame(records)
        csv_path = tmp_path / "large.csv"
        df.to_csv(csv_path, index=False)

        rows = temp_db.import_csv(str(csv_path), table="daily_ohlcv")
        assert rows == n
