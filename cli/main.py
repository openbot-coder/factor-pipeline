"""CLI main entry — argparse dispatch."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PipelineConfig, load_config
from factors.registry import FactorRegistry


def cmd_run(cfg: PipelineConfig):
    """Run full pipeline: load → factor → IC → layered → report."""
    from data.loader import DataLoader
    from data.preprocessor import DataPreprocessor
    from analysis.report import FactorReport

    logging.basicConfig(level=cfg.output.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pipeline")

    # 1. Load data
    log.info(f"Loading data from {cfg.data.source}...")
    loader = DataLoader(cfg.data.source, cfg.data.path)
    symbols = cfg.data.symbols or None
    data = loader.load(symbols, cfg.data.start_date, cfg.data.end_date)

    pp = DataPreprocessor(data)
    data = pp.align()
    n_dates = data["close"].index.get_level_values(0).nunique()
    n_stocks = data["close"].index.get_level_values(1).nunique()
    log.info(f"Loaded {len(data['close'].index)} rows ({n_dates} dates × {n_stocks} stocks)")

    # 2. Run factors
    from factors.registry import FactorRegistry
    from analysis.ic import ICAnalysis
    from analysis.layered import LayeredBacktest
    import importlib
    importlib.import_module("factors.gtja191")
    importlib.import_module("factors.technical")

    results = []
    close = data["close"]
    period = cfg.backtest.periods[0]

    def _compute_fwd_ret(close_df: pd.DataFrame, period: int) -> pd.Series:
        """Compute forward returns from close prices."""
        close_s = close_df["close"].copy()
        shifted = close_s.groupby(level=1).shift(-period)
        return (shifted / close_s - 1).dropna()

    for factor_name in cfg.factor.names:
        factor_fn = FactorRegistry.get(factor_name)
        if factor_fn is None:
            log.warning(f"Factor '{factor_name}' not found, skipping.")
            continue
        log.info(f"Computing {factor_name}...")
        try:
            if isinstance(factor_fn, type):
                # FactorABC class — instantiate with data dict
                inst = factor_fn(data)
                values = inst.compute()
            else:
                # plain function
                result = factor_fn(data)
                if isinstance(result, pd.Series):
                    values = result
                elif isinstance(result, pd.DataFrame):
                    values = result.iloc[:, 0] if result.shape[1] > 1 else result.iloc[:, 0]
                else:
                    values = result
            # Ensure MultiIndex (date, stock)
            if not isinstance(values.index, pd.MultiIndex):
                log.warning(f"{factor_name} returned non-MultiIndex, stacking...")
                values = values.stack()
            # Ensure column name is 'factor'
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            values = values.rename("factor")
        except Exception as e:
            log.warning(f"Factor {factor_name} failed: {e}")
            continue

        # IC analysis — handle MultiIndex mismatches safely
        fwd_ret = _compute_fwd_ret(close, period)
        factor_clean = values.dropna()
        # Reindex both to a common set to avoid index incompatibility
        common_date_stock = factor_clean.index.intersection(fwd_ret.index)
        if len(common_date_stock) < 30:
            log.warning(f"{factor_name}: only {len(common_date_stock)} common (date,stock) pairs, skipping.")
            continue
        ic = ICAnalysis(factor_clean.reindex(common_date_stock), fwd_ret.reindex(common_date_stock))
        ic_res = ic.run("spearman")
        log.info(f"  {factor_name}: IC_mean={ic_res.ic_mean:.4f}, IR={ic_res.ir:.3f}, IC>0={ic_res.ic_positive_ratio:.1%}")

        # Layered backtest
        lb = LayeredBacktest(factor_clean.reindex(common_date_stock), fwd_ret.reindex(common_date_stock), cfg.backtest.n_quantiles)
        lb_res = lb.run()

        # Report
        report = FactorReport(values, close)
        out_path = f"{cfg.output.report_dir}/{factor_name}_report.html"
        report.to_html(output_path=out_path, n_quantiles=cfg.backtest.n_quantiles)
        log.info(f"  Report → {out_path}")

        results.append({
            "factor": factor_name,
            "ic_mean": ic_res.ic_mean,
            "ir": ic_res.ir,
            "ic_positive_ratio": ic_res.ic_positive_ratio,
            "spread_ir": lb_res["spread_ir"],
        })

    # Summary table
    if results:
        df = pd.DataFrame(results).sort_values("ir", ascending=False)
        print("\n" + "=" * 70)
        print("Factor Ranking Summary")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
        summary_path = f"{cfg.output.report_dir}/summary.csv"
        Path(cfg.output.report_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
        log.info(f"Summary → {summary_path}")
    else:
        log.error("No factors computed successfully.")


def cmd_list_factors():
    """List all registered factors."""
    from factors.registry import FactorRegistry
    import importlib
    importlib.import_module("factors.gtja191")
    importlib.import_module("factors.technical")
    names = FactorRegistry.list()
    print(f"\nAvailable factors ({len(names)}):\n")
    for i, n in enumerate(names, 1):
        info = FactorRegistry.info(n)
        doc = info.get("doc", "")[:60]
        print(f"  {i:3d}. {n:20s}  {doc}")
    print()


def cmd_factor_doc(name: str):
    """Show factor formula/doc."""
    import importlib
    importlib.import_module("factors.gtja191")
    importlib.import_module("factors.technical")
    info = FactorRegistry.info(name)
    if not info["found"]:
        print(f"Factor '{name}' not found.")
        return
    print(f"\n{name}")
    print(f"Signature: {info['signature']}")
    print(f"Doc: {info['doc']}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="factor-pipeline",
        description="Factor detection pipeline: calculate → IC/IR → layered → report",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--config", "-c", help="YAML config path", default=None)
    p_run.add_argument("--factors", nargs="+", help="Factor names to compute")
    p_run.add_argument("--data", help="Data path (duckdb/parquet/csv)")
    p_run.add_argument("--start", help="Start date (YYYY-MM-DD)")
    p_run.add_argument("--end", help="End date (YYYY-MM-DD)")
    p_run.add_argument("--out", help="Report output dir", default="reports")

    sub.add_parser("factors", help="List available factors")

    p_doc = sub.add_parser("doc", help="Show factor formula/doc")
    p_doc.add_argument("name", help="Factor name (e.g. alpha001)")

    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config(args.config)
        if args.factors:
            cfg.factor.names = args.factors
        if args.data:
            cfg.data.path = args.data
        if args.start:
            cfg.data.start_date = args.start
        if args.end:
            cfg.data.end_date = args.end
        if args.out:
            cfg.output.report_dir = args.out
        cmd_run(cfg)
    elif args.command == "factors":
        cmd_list_factors()
    elif args.command == "doc":
        cmd_factor_doc(args.name)
    else:
        parser.print_help()
