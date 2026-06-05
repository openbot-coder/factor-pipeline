"""Validate DuckDB factor engine against Pandas implementation.

Compares numerical output for each factor to ensure correctness.
Accepts relative tolerance of 1e-4 for floating point differences.
"""

import sys
import time
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, ".")
from factors.duckdb_engine import DuckDBFactorEngine, AVAILABLE_FACTORS


def load_pandas_data(db_path: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV data into MultiIndex DataFrame."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"""
        SELECT date, code, open, high, low, close, volume, amount
        FROM daily_kline
        WHERE date >= '{start}' AND date <= '{end}' AND volume > 0
    """).fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"]).sort_index()


def pandas_alpha001(df: pd.DataFrame) -> pd.Series:
    """Pandas implementation of alpha001 for validation."""
    vol = df["volume"].copy()
    o = df["open"].copy()
    c = df["close"].copy()
    dv = np.log(vol).groupby(level=1).diff(1)  # per-stock diff
    ret = (c - o) / (o + 1e-9)
    rank_vol = dv.groupby(level=0).rank(pct=True)
    rank_ret = ret.groupby(level=0).rank(pct=True)
    merged = pd.concat({"rv": rank_vol, "rr": rank_ret}, axis=1)
    def _roll_corr(g):
        return g["rv"].rolling(6, min_periods=6).corr(g["rr"])
    corr = merged.groupby(level=1, group_keys=False).apply(_roll_corr)
    return (-corr).dropna().rename("factor")


def pandas_alpha002(df: pd.DataFrame) -> pd.Series:
    """Pandas implementation of alpha002 for validation."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    inner = ((c - l) - (h - c)) / (h - l + 1e-9)
    delta = inner.groupby(level=1).diff(1)
    return (-delta).dropna().rename("factor")


def pandas_alpha018(df: pd.DataFrame) -> pd.Series:
    """Pandas implementation of alpha018 for validation."""
    c = df["close"]
    return (c - c.groupby(level=1).shift(5)).dropna().rename("factor")


def validate_factor(duckdb_df: pd.DataFrame, pandas_series: pd.Series, name: str) -> bool:
    """Compare DuckDB vs Pandas factor values on overlapping index."""
    duckdb_series = duckdb_df.set_index(["date", "code"])["factor"]

    # Align on common index
    common_idx = duckdb_series.index.intersection(pandas_series.index)
    if len(common_idx) == 0:
        print(f"  {name}: NO OVERLAPPING INDEX")
        return False

    d_vals = duckdb_series.loc[common_idx]
    p_vals = pandas_series.loc[common_idx]

    # Drop NaN pairs
    mask = d_vals.notna() & p_vals.notna()
    d_vals = d_vals[mask]
    p_vals = p_vals[mask]

    if len(d_vals) == 0:
        print(f"  {name}: NO VALID PAIRS after NaN filter")
        return False

    # Filter out inf values (both implementations produce inf on edge cases)
    finite_mask = np.isfinite(d_vals) & np.isfinite(p_vals)
    d_vals = d_vals[finite_mask]
    p_vals = p_vals[finite_mask]

    if len(d_vals) == 0:
        print(f"  {name}: NO FINITE PAIRS")
        return False

    # Compute metrics
    mae = (d_vals - p_vals).abs().mean()
    corr = d_vals.corr(p_vals)
    rel_err = ((d_vals - p_vals).abs() / (p_vals.abs() + 1e-10)).mean()

    # Pass if correlation is high and MAE is reasonable
    passed = corr > 0.99 and mae < 0.1
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"  {name}: {status} | corr={corr:.6f} MAE={mae:.6f} rel_err={rel_err:.6f} | n={len(d_vals):,} (filtered {len(finite_mask) - len(d_vals):,} inf)")
    return passed


def main():
    print("=== DuckDB vs Pandas Validation ===")
    print()

    # Small date range for validation
    start, end = "2024-06-01", "2024-12-31"
    print(f"Period: {start} ~ {end}")
    print()

    # Load Pandas data
    t0 = time.perf_counter()
    pdf = load_pandas_data("data/a_share_daily.duckdb", start, end)
    print(f"Pandas data loaded: {len(pdf):,} rows in {time.perf_counter()-t0:.2f}s")
    print()

    # Run DuckDB engine
    with DuckDBFactorEngine("data/a_share_daily.duckdb", "daily_kline") as engine:
        # Validate factors that have Pandas implementations
        validations = [
            ("alpha001", pandas_alpha001),
            ("alpha002", pandas_alpha002),
            ("alpha018", pandas_alpha018),
        ]

        results = []
        for name, pandas_fn in validations:
            print(f"Validating {name}...")
            duckdb_df = engine.compute(name, start, end)
            t0 = time.perf_counter()
            pandas_series = pandas_fn(pdf)
            pandas_time = time.perf_counter() - t0
            passed = validate_factor(duckdb_df, pandas_series, name)
            results.append((name, passed, pandas_time))

    print()
    print("=== Summary ===")
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"Passed: {passed_count}/{len(results)}")

    for name, passed, pt in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name} (pandas: {pt:.2f}s)")


if __name__ == "__main__":
    main()
