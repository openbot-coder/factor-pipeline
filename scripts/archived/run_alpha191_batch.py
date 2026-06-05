#!/usr/bin/env python3
"""Alpha191 全量 IC/ICIR 评估 — 分批跑，每批 20 个因子，增量保存。

数据: ohlcv_csi500.duckdb
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
FORWARD_DAYS = [1, 5, 10, 20]
START_DATE = "2020-01-01"   # Use 6 years for speed, enough warmup
END_DATE = "2026-05-22"
BATCH_SIZE = 20
REPORT_PATH = "/tmp/alpha191_full_report.csv"
PROGRESS_PATH = "/tmp/alpha191_progress.json"
RUN_ERRORS_PATH = "/tmp/alpha191_errors.json"

# ── Load data (cached to /tmp) ──────────────────────────────────────
CACHE_PICKLE = "/tmp/alpha191_panel.pkl"
CACHE_FWD = "/tmp/alpha191_fwd.pkl"

def compute_forward_returns(close, days):
    return close.shift(-days) / close - 1

def load_data():
    if Path(CACHE_PICKLE).exists() and Path(CACHE_FWD).exists():
        print("Loading cached panel...")
        panel = pd.read_pickle(CACHE_PICKLE)
        fwd_rets = pd.read_pickle(CACHE_FWD)
        print(f"Panel shape: {panel['close'].shape}")
        return panel, fwd_rets

    print("Loading from DuckDB...")
    db = duckdb.connect(str(DB_PATH), read_only=True)
    df = db.sql(f"""
        SELECT date, symbol, open, high, low, close, volume, amount
        FROM daily_ohlcv
        WHERE date >= '{START_DATE}' AND date <= '{END_DATE}'
        ORDER BY date, symbol
    """).fetchdf()
    db.close()
    print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} stocks")

    df["date"] = pd.to_datetime(df["date"])
    panel = {}
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        wide = df.pivot(index="date", columns="symbol", values=col)
        panel[col] = wide.astype(np.float64)

    fwd_rets = {}
    for d in FORWARD_DAYS:
        fwd_rets[d] = compute_forward_returns(panel["close"], d)

    pd.to_pickle(panel, CACHE_PICKLE)
    pd.to_pickle(fwd_rets, CACHE_FWD)
    print(f"Panel: {panel['close'].shape[0]} dates × {panel['close'].shape[1]} stocks")
    print(f"Cached to /tmp")
    return panel, fwd_rets


def load_factors():
    """Import and return list of (idx, compute_fn, meta)."""
    sys.path.insert(0, VIBE_ROOT)
    factors = []
    for i in range(1, 192):
        mod_name = f"src.factors.zoo.gtja191.alpha_{i:03d}"
        try:
            mod = importlib.import_module(mod_name)
            factors.append((i, mod.compute, getattr(mod, "__alpha_meta__", {})))
        except Exception as e:
            print(f"  LOAD ERROR alpha_{i:03d}: {e}")
    return factors


def compute_ic_series(factor_df, fwd_ret):
    common_dates = factor_df.index.intersection(fwd_ret.index)
    common_cols = factor_df.columns.intersection(fwd_ret.columns)
    f = factor_df.loc[common_dates, common_cols]
    r = fwd_ret.loc[common_dates, common_cols]
    ic = f.corrwith(r, axis=1, method="spearman")
    return ic.dropna()


def evaluate_factor(idx, compute_fn, panel, fwd_rets):
    """Evaluate one factor, return result dict or None."""
    t1 = time.time()
    fid = f"alpha_{idx:03d}"
    fv = compute_fn(panel)
    elapsed = time.time() - t1
    valid_pct = fv.notna().sum().sum() / fv.size * 100
    row = {"factor": fid, "idx": idx, "elapsed_s": round(elapsed, 2), "valid_pct": round(valid_pct, 1)}
    for d in FORWARD_DAYS:
        ic = compute_ic_series(fv, fwd_rets[d])
        if len(ic) > 10:
            ic_mean, ic_std = ic.mean(), ic.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos = (ic > 0).sum() / len(ic) * 100
            row.update({f"IC_{d}d": round(ic_mean, 5), f"ICIR_{d}d": round(icir, 4),
                        f"ICpos_{d}d": round(ic_pos, 1), f"ICcount_{d}d": len(ic)})
        else:
            row.update({f"IC_{d}d": np.nan, f"ICIR_{d}d": np.nan,
                        f"ICpos_{d}d": np.nan, f"ICcount_{d}d": 0})
    return row


def main():
    import json
    t0 = time.time()

    # Check progress
    done_indices = set()
    results = []
    run_errors = []
    if Path(REPORT_PATH).exists():
        df_existing = pd.read_csv(REPORT_PATH)
        for _, r in df_existing.iterrows():
            if pd.notna(r.get("IC_5d")):
                done_indices.add(int(r["idx"]))
                results.append(r.to_dict())
            elif pd.isna(r.get("IC_5d")) and r.get("valid_pct", 0) == 0:
                # Was an error before, retry
                pass
        print(f"Resuming: {len(done_indices)} factors already done")

    panel, fwd_rets = load_data()
    factors = load_factors()
    print(f"Loaded {len(factors)} factors")

    # Filter out already done
    remaining = [(idx, fn, meta) for idx, fn, meta in factors if idx not in done_indices]
    print(f"Remaining: {len(remaining)} factors")

    # Run in batches
    batch_results = []
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_t0 = time.time()

        for idx, compute_fn, meta in batch:
            try:
                row = evaluate_factor(idx, compute_fn, panel, fwd_rets)
                batch_results.append(row)
                results.append(row)
                print(f"  ✓ alpha_{idx:03d} {row['elapsed_s']:.1f}s ICIR_5d={row.get('ICIR_5d', 'N/A')}")
            except Exception as e:
                err_msg = str(e)[:200]
                run_errors.append((idx, err_msg))
                row = {"factor": f"alpha_{idx:03d}", "idx": idx, "elapsed_s": 0, "valid_pct": 0,
                       **{f"IC_{d}d": np.nan for d in FORWARD_DAYS},
                       **{f"ICIR_{d}d": np.nan for d in FORWARD_DAYS},
                       **{f"ICpos_{d}d": np.nan for d in FORWARD_DAYS},
                       **{f"ICcount_{d}d": 0 for d in FORWARD_DAYS}}
                batch_results.append(row)
                results.append(row)
                print(f"  ✗ alpha_{idx:03d} ERROR: {err_msg[:80]}")

        # Save after each batch
        df_save = pd.DataFrame(batch_results).sort_values("idx") if batch_results else pd.DataFrame()
        # Append all results
        all_df = pd.DataFrame(results).sort_values("idx").drop_duplicates(subset=["idx"], keep="last")
        all_df.to_csv(REPORT_PATH, index=False)
        batch_elapsed = time.time() - batch_t0
        print(f"  Batch saved ({len(results)} total, batch took {batch_elapsed:.0f}s)")
        batch_results = []

    # Final summary
    elapsed_total = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed_total:.0f}s")
    all_df = pd.DataFrame(results).sort_values("idx").drop_duplicates(subset=["idx"], keep="last")
    all_df.to_csv(REPORT_PATH, index=False)

    ok = all_df[all_df["ICcount_5d"] > 0]
    err = all_df[all_df["ICcount_5d"] == 0]
    print(f"Success: {len(ok)}/191, Errors: {len(err)}/191")
    if run_errors:
        print(f"\nRun errors ({len(run_errors)}):")
        for idx, e in run_errors:
            print(f"  alpha_{idx:03d}: {e[:120]}")

    if len(ok) > 0:
        ok_s = ok.copy()
        ok_s["abs_ICIR_5d"] = ok_s["ICIR_5d"].abs()
        top = ok_s.nlargest(20, "abs_ICIR_5d")
        print(f"\nTop 20 by |ICIR_5d|:")
        print(top[["factor", "IC_5d", "ICIR_5d", "ICpos_5d", "valid_pct", "ICIR_1d", "ICIR_10d", "ICIR_20d"]].to_string(index=False))
        print(f"\nICIR_5d distribution:")
        icir = ok["ICIR_5d"].dropna()
        for th in [0.3, 0.2, 0.1, 0.05]:
            n = (icir.abs() >= th).sum()
            print(f"  |ICIR| >= {th}: {n} ({n/len(ok)*100:.0f}%)")

    if run_errors:
        with open(RUN_ERRORS_PATH, "w") as f:
            json.dump(run_errors, f)


if __name__ == "__main__":
    main()
