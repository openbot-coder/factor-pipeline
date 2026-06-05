"""Data loader — load OHLCV from CSV/Parquet/DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """Load daily OHLCV data from various sources."""

    def __init__(self, source: str = "duckdb", path: Optional[str] = None):
        """
        Args:
            source: "duckdb" | "parquet" | "csv"
            path: path to DB file or directory
        """
        self.source = source
        self.path = Path(path) if path else None

    def load(self, symbols: list[str] = None, start: str = None, end: str = None) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for given symbols and date range.

        Returns dict with keys: open, high, low, close, volume, amount (optional)
        Each value is a DataFrame indexed by (date, stock) MultiIndex.
        """
        if self.source == "duckdb":
            return self._load_duckdb(symbols, start, end)
        elif self.source == "parquet":
            return self._load_parquet(symbols, start, end)
        elif self.source == "csv":
            return self._load_csv(symbols, start, end)
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_duckdb(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        db_path = self.path or "data/ohlcv.duckdb"
        con = duckdb.connect(str(db_path))

        where_parts = []
        params = []
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

    def _load_parquet(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        files = list(self.path.glob("*.parquet"))
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No parquet files in {self.path}")
        df = pd.concat(dfs, ignore_index=True)
        return self._to_multiindex(df)

    def _load_csv(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        files = list(self.path.glob("*.csv"))
        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=["date"])
            dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No CSV files in {self.path}")
        df = pd.concat(dfs, ignore_index=True)
        return self._to_multiindex(df)

    def _to_multiindex(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Convert flat DataFrame to MultiIndex dict."""
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "symbol"]).sort_index()
        result = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                result[col] = df[col].to_frame()
        if "amount" in df.columns:
            result["amount"] = df["amount"].to_frame()
        return result
