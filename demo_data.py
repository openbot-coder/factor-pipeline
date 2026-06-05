#!/usr/bin/env python3
"""Generate synthetic OHLCV data for demo/testing — no API key needed."""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path


def generate_ohlcv(
    symbols: list[str],
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV DataFrame for given symbols."""
    np.random.seed(seed)
    dates = pd.bdate_range(start, end)  # business days only
    n = len(dates)
    rows = []
    for sym in symbols:
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
        open_ = low + np.random.rand(n) * (high - low)
        volume = np.random.lognormal(15, 1, n).astype(int) + 1
        amount = close * volume
        df_sym = pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        })
        rows.append(df_sym)
    return pd.concat(rows, ignore_index=True)


def save_to_duckdb(df: pd.DataFrame, path: str = "data/ohlcv.duckdb"):
    """Save DataFrame to DuckDB."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute("DROP TABLE IF EXISTS daily_ohlcv")
    con.execute(
        "CREATE TABLE daily_ohlcv (date DATE, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT, amount DOUBLE)"
    )
    con.execute("INSERT INTO daily_ohlcv SELECT * FROM df")
    con.close()
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    # 50 synthetic stocks
    symbols = [f"SH{str(i).zfill(6)}" for i in range(600000, 600050)]
    df = generate_ohlcv(symbols, start="2020-01-01", end="2025-05-01")
    save_to_duckdb(df, "data/ohlcv.duckdb")
    print(f"Generated {len(df)} rows, {df['symbol'].nunique()} stocks")
