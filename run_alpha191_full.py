#!/usr/bin/env python3
"""Alpha191 全量 IC/ICIR 评估 — 使用 Vibe-Trading 的 191 个 Pandas 实现作为 ground truth。

数据: ohlcv_csi500.duckdb (CSI500 成分股日线, ~1.5M rows)
输出: /tmp/alpha191_full_report.csv
"""

import importlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ──────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "data" / "ohlcv_csi500.duckdb"
VIBE_ROOT = "/home/openbot/workspace/projects/Vibe-Trading/agent"
FORWARD_DAYS = [1, 5, 10, 20]  # forward return horizons
START_DATE = "2018-01-01"       # enough warmup for 60d factors
END_DATE = "2026-05-22"

# ── Load data ───────────────────────────────────────────────────────
def load_data() -> dict[str, pd.DataFrame]:
    """Load OHLCV from DuckDB, return wide-panel dict."""
    db = duckdb.connect(str(DB_PATH), read_only=True)
    df = db.sql(f"""
        SELECT date, symbol, open, high, low, close, volume, amount
        FROM daily_ohlcv
        WHERE date >= '{START_DATE}' AND date <= '{END_DATE}'
        ORDER BY date, symbol
    """).fetchdf()
    db.close()

    print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} stocks, "
          f"{df['date'].min()} ~ {df['date'].max()}")

    # Pivot to wide panel: index=date (DatetimeIndex), columns=symbol
    df["date"] = pd.to_datetime(df["date"])
    panel = {}
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        wide = df.pivot(index="date", columns="symbol", values=col)
        panel[col] = wide.astype(np.float64)

    print(f"Panel shape: {panel['close'].shape[0]} dates × {panel['close'].shape[1]} stocks")
    return panel


# ── Load all 191 factors ────────────────────────────────────────────
def load_factors():
    """Import and return list of (idx, compute_fn, meta)."""
    sys.path.insert(0, VIBE_ROOT)
    factors = []
    errors = []
    for i in range(1, 192):
        mod_name = f"src.factors.zoo.gtja191.alpha_{i:03d}"
        try:
            mod = importlib.import_module(mod_name)
            factors.append((i, mod.compute, getattr(mod, "__alpha_meta__", {})))
        except Exception as e:
            errors.append((i, str(e)[:100]))
    print(f"Loaded {len(factors)}/191 factors, errors: {len(errors)}")
    if errors:
        for idx, err in errors[:5]:
            print(f"  Error alpha_{idx:03d}: {err}")
    return factors


# ── Forward returns ─────────────────────────────────────────────────
def compute_forward_returns(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Forward return: ret[t] = close[t+n] / close[t] - 1."""
    return close.shift(-days) / close - 1


# ── IC computation ──────────────────────────────────────────────────
def compute_ic_series(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Per-date rank IC (Spearman) between factor values and forward returns."""
    common_dates = factor_df.index.intersection(fwd_ret.index)
    common_cols = factor_df.columns.intersection(fwd_ret.columns)
    f = factor_df.loc[common_dates, common_cols]
    r = fwd_ret.loc[common_dates, common_cols]

    ic = f.corrwith(r, axis=1, method="spearman")
    return ic.dropna()


# ── Main evaluation ─────────────────────────────────────────────────
def main():
    t0 = time.time()

    # 1. Load data
    print("=" * 60)
    print("Step 1: Loading data...")
    panel = load_data()
    close = panel["close"]

    # 2. Load factors
    print("=" * 60)
    print("Step 2: Loading 191 factors...")
    factors = load_factors()

    # 3. Compute forward returns
    print("=" * 60)
    print("Step 3: Computing forward returns...")
    fwd_rets = {}
    for d in FORWARD_DAYS:
        fwd_rets[d] = compute_forward_returns(close, d)
        print(f"  {d}d forward returns ready")

    # 4. Run all factors + IC
    print("=" * 60)
    print(f"Step 4: Running {len(factors)} factors × {len(FORWARD_DAYS)} horizons...")
    results = []
    factor_values_cache = {}
    run_errors = []

    for idx, compute_fn, meta in factors:
        t1 = time.time()
        fid = f"alpha_{idx:03d}"
        try:
            fv = compute_fn(panel)
            factor_values_cache[idx] = fv
            elapsed = time.time() - t1

            # Check valid coverage
            valid_pct = fv.notna().sum().sum() / fv.size * 100

            row = {
                "factor": fid,
                "idx": idx,
                "elapsed_s": round(elapsed, 2),
                "valid_pct": round(valid_pct, 1),
            }

            for d in FORWARD_DAYS:
                ic = compute_ic_series(fv, fwd_rets[d])
                if len(ic) > 10:
                    ic_mean = ic.mean()
                    ic_std = ic.std()
                    icir = ic_mean / ic_std if ic_std > 0 else 0
                    ic_pos = (ic > 0).sum() / len(ic) * 100
                    row[f"IC_{d}d"] = round(ic_mean, 5)
                    row[f"ICIR_{d}d"] = round(icir, 4)
                    row[f"ICpos_{d}d"] = round(ic_pos, 1)
                    row[f"ICcount_{d}d"] = len(ic)
                else:
                    row[f"IC_{d}d"] = np.nan
                    row[f"ICIR_{d}d"] = np.nan
                    row[f"ICpos_{d}d"] = np.nan
                    row[f"ICcount_{d}d"] = 0

            results.append(row)
            if idx % 20 == 0 or idx == 1:
                print(f"  [{idx}/191] {fid} elapsed={elapsed:.1f}s valid={valid_pct:.0f}% "
                      f"IC_5d={row.get('IC_5d', 'N/A')} ICIR_5d={row.get('ICIR_5d', 'N/A')}")

        except Exception as e:
            elapsed = time.time() - t1
            run_errors.append((idx, str(e)[:200]))
            results.append({
                "factor": fid, "idx": idx, "elapsed_s": round(elapsed, 2),
                "valid_pct": 0,
                **{f"IC_{d}d": np.nan for d in FORWARD_DAYS},
                **{f"ICIR_{d}d": np.nan for d in FORWARD_DAYS},
                **{f"ICpos_{d}d": np.nan for d in FORWARD_DAYS},
                **{f"ICcount_{d}d": 0 for d in FORWARD_DAYS},
            })
            if idx % 20 == 0 or idx <= 3:
                print(f"  [{idx}/191] {fid} ERROR: {str(e)[:80]}")

    # 5. Summary
    elapsed_total = time.time() - t0
    print("=" * 60)
    print(f"Step 5: Results summary (total time: {elapsed_total:.0f}s)")
    df = pd.DataFrame(results).sort_values("idx")
    df.to_csv("/tmp/alpha191_full_report.csv", index=False)
    print(f"Saved: /tmp/alpha191_full_report.csv ({len(df)} rows)")

    # Stats
    ok = df[df["ICcount_5d"] > 0]
    err = df[df["ICcount_5d"] == 0]
    print(f"\nSuccess: {len(ok)}/191, Errors/Skipped: {len(err)}/191")
    if run_errors:
        print(f"\nRun errors ({len(run_errors)}):")
        for idx, e in run_errors[:10]:
            print(f"  alpha_{idx:03d}: {e[:120]}")

    # Top factors by |ICIR_5d|
    if len(ok) > 0:
        ok_sorted = ok.copy()
        ok_sorted["abs_ICIR_5d"] = ok_sorted["ICIR_5d"].abs()
        top = ok_sorted.nlargest(20, "abs_ICIR_5d")
        print(f"\n{'='*80}")
        print("Top 20 factors by |ICIR_5d|:")
        print(f"{'='*80}")
        cols_show = ["factor", "IC_5d", "ICIR_5d", "ICpos_5d", "valid_pct"]
        for h in [1, 10, 20]:
            cols_show.extend([f"IC_{h}d", f"ICIR_{h}d"])
        print(top[cols_show].to_string(index=False))

        # Distribution
        print(f"\n{'='*80}")
        print("ICIR_5d Distribution:")
        icir_vals = ok["ICIR_5d"].dropna()
        if len(icir_vals) > 0:
            for threshold in [0.3, 0.2, 0.1, 0.05]:
                n = (icir_vals.abs() >= threshold).sum()
                print(f"  |ICIR| >= {threshold}: {n} factors ({n/len(ok)*100:.0f}%)")

    return df


if __name__ == "__main__":
    main()
