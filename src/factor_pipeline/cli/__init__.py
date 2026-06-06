#!/usr/bin/env python
"""Factor Pipeline CLI - Unified command-line interface.

A comprehensive CLI tool for factor analysis with DuckDB backend.

Usage:
    fp data init
    fp data import data.csv
    fp data info
    fp factor run "Mean($close, 20)"
    fp factor list
    fp factor doc alpha001
    fp backtest run --factors alpha001,alpha014

Installation:
    pip install -e .

Examples:
    # Data management
    fp data init --db data/ohlcv.duckdb
    fp data import data.csv --table daily_ohlcv
    fp data query "SELECT * FROM daily_ohlcv LIMIT 10"
    fp data stats --table daily_ohlcv

    # Factor analysis
    fp factor run "Corr($close, $volume, 20)" --start 2020-01-01
    fp factor batch factors.txt --output results/
    fp factor list
    fp factor doc alpha001

    # Backtesting
    fp backtest run --factors alpha001,alpha014 --start 2020-01-01 --end 2024-12-31
    fp backtest ic --factor alpha001 --output ic_report.html
    fp backtest layered --factor alpha001 --output layered_report.html

    # Reports
    fp report generate --factors alpha001,alpha014 --output report.html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click

# Import from factor_pipeline package
from factor_pipeline import __version__

from data.storage import DuckDBStorage


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB = os.environ.get("FACTOR_PIPELINE_DB", "data/ohlcv.duckdb")
VERSION = "0.1.0"


# =============================================================================
# Utility Functions
# =============================================================================

def echo_header(text: str) -> None:
    """Print section header."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f" {text}")
    click.echo(f"{'=' * 60}\n")


def echo_success(text: str) -> None:
    """Print success message."""
    click.echo(f"✅ {text}")


def echo_error(text: str) -> None:
    """Print error message."""
    click.echo(f"❌ {text}", err=True)


def echo_info(text: str) -> None:
    """Print info message."""
    click.echo(f"ℹ️  {text}")


def get_db(db_path: Optional[str] = None) -> DuckDBStorage:
    """Get database connection."""
    path = db_path or DEFAULT_DB
    if not os.path.exists(path) and path != ":memory:":
        echo_error(f"Database not found: {path}")
        echo_info(f"Run 'fp data init --db {path}' to create it")
        sys.exit(1)
    return DuckDBStorage(path)


# =============================================================================
# Data Commands
# =============================================================================

@click.group(name="data")
@click.help_option("-h", "--help")
def data_group():
    """Data management commands."""
    pass


@data_group.command("init")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--force", is_flag=True, help="Overwrite existing database")
def data_init(db: str, force: bool):
    """Initialize a new database with schema."""
    if os.path.exists(db) and not force:
        echo_error(f"Database already exists: {db}")
        echo_info("Use --force to overwrite")
        sys.exit(1)

    # Create directory
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)

    storage = DuckDBStorage(db)
    storage.init_schema()

    echo_success(f"Database initialized: {db}")
    echo_info("Tables: daily_ohlcv, instruments, calendars, factor_cache")


@data_group.command("info")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def data_info(db: str):
    """Show database information."""
    storage = get_db(db)
    info = storage.info()

    echo_header("Database Info")
    click.echo(f"Path:     {info['db_path']}")
    click.echo(f"Size:     {info.get('db_size_mb', 0):.2f} MB")

    if info.get("date_range"):
        click.echo(f"Date:     {info['date_range'][0]} to {info['date_range'][1]}")
    if info.get("symbols_count"):
        click.echo(f"Symbols:  {info['symbols_count']}")

    click.echo("\n📋 Tables:")
    for table, stats in info["tables"].items():
        click.echo(f"   {table:20} {stats['rows']:>10,} rows")


@data_group.command("import")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--table", required=True, help="Target table name")
@click.option("--date-col", default="date", help="Date column name")
@click.option("--symbol-col", default="symbol", help="Symbol column name")
@click.option("--if-exists", default="append", type=click.Choice(["append", "replace", "fail"]))
@click.option("--preview/--no-preview", default=True, help="Show preview before import")
def data_import(
    csv_path: str,
    db: str,
    table: str,
    date_col: str,
    symbol_col: str,
    if_exists: str,
    preview: bool,
):
    """Import CSV file into database."""
    import pandas as pd

    storage = get_db(db)

    # Preview
    if preview:
        df = pd.read_csv(csv_path, nrows=5)
        echo_header(f"Preview: {os.path.basename(csv_path)}")
        click.echo(f"Columns: {', '.join(df.columns.tolist())}")
        click.echo(f"\nFirst 5 rows:")
        click.echo(df.to_string(index=False))
        click.echo()

    # Import
    try:
        rows = storage.import_csv(
            csv_path,
            table=table,
            date_col=date_col,
            symbol_col=symbol_col,
            if_exists=if_exists,
        )
        echo_success(f"Imported {rows:,} rows into '{table}'")
    except Exception as e:
        echo_error(f"Import failed: {e}")
        sys.exit(1)


@data_group.command("query")
@click.argument("sql")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--limit", default=100, help="Result limit")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "csv", "json"]))
@click.option("--output", "-o", help="Output file path")
def data_query(sql: str, db: str, limit: int, fmt: str, output: Optional[str]):
    """Execute SQL query."""
    storage = get_db(db)

    # Add LIMIT if not present
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    try:
        df = storage.query(sql)

        if output:
            if output.endswith(".csv"):
                df.to_csv(output, index=False)
            elif output.endswith(".json"):
                df.to_json(output, orient="records", indent=2)
            echo_success(f"Results saved to {output}")
        elif fmt == "csv":
            click.echo(df.to_csv(index=False))
        elif fmt == "json":
            click.echo(df.to_json(orient="records", indent=2))
        else:
            click.echo(df.to_string())
    except Exception as e:
        echo_error(f"Query failed: {e}")
        sys.exit(1)


@data_group.command("stats")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--table", help="Specific table")
def data_stats(db: str, table: Optional[str]):
    """Show table statistics."""
    import pandas as pd

    storage = get_db(db)
    tables = [table] if table else storage.list_tables()

    for t in tables:
        click.echo(f"\n📊 {t}:")
        df = storage.query(f"SELECT * FROM {t} LIMIT 0")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = storage.query(f"SELECT AVG({col}), STD({col}), MIN({col}), MAX({col}) FROM {t}")
                click.echo(f"   {col}: avg={stats.iloc[0,0]:.2f}, std={stats.iloc[0,1]:.2f}, min={stats.iloc[0,2]:.2f}, max={stats.iloc[0,3]:.2f}")


@data_group.command("tables")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def data_tables(db: str):
    """List all tables."""
    storage = get_db(db)
    tables = storage.list_tables()

    echo_header("Tables")
    for t in tables:
        count = storage.fetchone(f"SELECT COUNT(*) FROM {t}")[0]
        click.echo(f"   {t:20} {count:>10,} rows")


@data_group.command("instruments")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--market", help="Filter by market")
@click.option("--active/--all", default=False, help="Only active instruments")
@click.option("--limit", default=50, help="Limit results")
@click.option("--output", "-o", help="Output file")
def data_instruments(db: str, market: Optional[str], active: bool, limit: int, output: Optional[str]):
    """List instruments (symbols)."""
    storage = get_db(db)
    symbols = storage.get_instruments(market=market)

    if active:
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        symbols = [s for s in symbols if storage.query(f"""
            SELECT 1 FROM instruments
            WHERE symbol = '{s}'
            AND list_date <= '{today}'
            AND (delist_date IS NULL OR delist_date >= '{today}')
        """).shape[0] > 0]

    echo_header(f"Instruments ({len(symbols)} total)")
    for i, s in enumerate(symbols[:limit]):
        click.echo(f"   {s}")
    if len(symbols) > limit:
        click.echo(f"   ... and {len(symbols) - limit} more")


@data_group.command("export")
@click.argument("sql")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", "-o", required=True, help="Output file")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "parquet"]))
def data_export(sql: str, db: str, output: str, fmt: str):
    """Export query results to file."""
    storage = get_db(db)

    try:
        if fmt == "csv":
            storage.export_csv(sql, output)
        else:
            storage.export_parquet(sql, output)
        echo_success(f"Exported to {output}")
    except Exception as e:
        echo_error(f"Export failed: {e}")
        sys.exit(1)


# =============================================================================
# Factor Commands
# =============================================================================

@click.group(name="factor")
@click.help_option("-h", "--help")
def factor_group():
    """Factor analysis commands."""
    pass


@factor_group.command("list")
def factor_list():
    """List all available factors."""
    from factors.registry import FactorRegistry
    import importlib

    # Load factor modules
    importlib.import_module("factors.gtja191")
    importlib.import_module("factors.technical")

    names = FactorRegistry.list()

    echo_header(f"Available Factors ({len(names)} total)")

    # Categorize
    gtja = [n for n in names if n.startswith("alpha")]
    tech = [n for n in names if not n.startswith("alpha")]

    if gtja:
        click.echo("\n📈 GTJA 191 Factors:")
        for n in sorted(gtja):
            f = FactorRegistry.get(n)
            click.echo(f"   {n:15} - {f.description[:60] if f.description else 'No description'}")

    if tech:
        click.echo("\n📊 Technical Indicators:")
        for n in sorted(tech):
            f = FactorRegistry.get(n)
            click.echo(f"   {n:15} - {f.description[:60] if f.description else 'No description'}")


@factor_group.command("doc")
@click.argument("factor_name")
def factor_doc(factor_name: str):
    """Show factor documentation."""
    from factors.registry import FactorRegistry

    f = FactorRegistry.get(factor_name)
    if not f:
        echo_error(f"Factor not found: {factor_name}")
        sys.exit(1)

    echo_header(f"Factor: {factor_name}")
    click.echo(f"Expression: {f.expression if hasattr(f, 'expression') else 'N/A'}")
    click.echo(f"Category: {f.category if hasattr(f, 'category') else 'N/A'}")
    click.echo(f"\nDescription:")
    click.echo(f"   {f.description if f.description else 'No description available'}")


@factor_group.command("run")
@click.argument("expression")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--symbols", help="Comma-separated symbols")
@click.option("--output", "-o", help="Output file")
def factor_run(expression: str, db: str, start: Optional[str], end: Optional[str],
                symbols: Optional[str], output: Optional[str]):
    """Run a factor expression."""
    from factors.expr_engine import ExprEngine

    storage = get_db(db)

    try:
        engine = ExprEngine(db)
        result = engine.factor(expression, start=start, end=end)

        if output:
            result.to_csv(output)
            echo_success(f"Results saved to {output}")
        else:
            echo_header(f"Expression: {expression}")
            click.echo(result.head(20).to_string())
    except Exception as e:
        echo_error(f"Factor calculation failed: {e}")
        sys.exit(1)


@factor_group.command("batch")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", help="Start date")
@click.option("--end", help="End date")
@click.option("--output", "-o", default="results/", help="Output directory")
def factor_batch(file_path: str, db: str, start: Optional[str], end: Optional[str], output: str):
    """Run multiple factors from a file (one expression per line)."""
    from factors.expr_engine import ExprEngine

    storage = get_db(db)
    engine = ExprEngine(db)

    # Read expressions
    with open(file_path) as f:
        expressions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    os.makedirs(output, exist_ok=True)

    echo_header(f"Running {len(expressions)} factors")

    for i, expr in enumerate(expressions):
        click.echo(f"[{i+1}/{len(expressions)}] {expr[:50]}...", nl=False)
        try:
            result = engine.factor(expr, start=start, end=end)
            safe_name = expr.replace("/", "_").replace("(", "_").replace(")", "_")[:30]
            result.to_csv(os.path.join(output, f"{safe_name}.csv"))
            click.echo(" ✅")
        except Exception as e:
            click.echo(f" ❌ ({e})")

    echo_success(f"Results saved to {output}")


# =============================================================================
# Backtest Commands
# =============================================================================

@click.group(name="backtest")
@click.help_option("-h", "--help")
def backtest_group():
    """Backtesting commands."""
    pass


@backtest_group.command("run")
@click.option("--factors", required=True, help="Comma-separated factor names")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", default="2020-01-01", help="Start date")
@click.option("--end", default="2024-12-31", help="End date")
@click.option("--output", "-o", default="backtest_results/", help="Output directory")
def backtest_run(factors: str, db: str, start: str, end: str, output: str):
    """Run backtest for factors."""
    import pandas as pd
    from analysis.ic import ICAnalysis
    from analysis.layered import LayeredBacktest

    storage = get_db(db)

    factor_list = [f.strip() for f in factors.split(",")]

    echo_header(f"Running Backtest")
    click.echo(f"Factors: {', '.join(factor_list)}")
    click.echo(f"Period: {start} to {end}")
    click.echo(f"Output: {output}")

    os.makedirs(output, exist_ok=True)

    # Calculate factors
    for factor_name in factor_list:
        click.echo(f"\n📊 {factor_name}:")

        try:
            # Get factor values
            from factors.registry import FactorRegistry
            factor = FactorRegistry.get(factor_name)
            if factor:
                data = storage.get_ohlcv(start_date=start, end_date=end)
                result = factor.calculate(data)
                result.to_csv(os.path.join(output, f"{factor_name}_values.csv"))
                click.echo(f"   ✅ Factor values saved")

            # IC Analysis
            ic = ICAnalysis()
            ic_result = ic.calculate(result, storage)
            ic_result.to_csv(os.path.join(output, f"{factor_name}_ic.csv"))
            click.echo(f"   ✅ IC analysis saved")

            # Layered Backtest
            lb = LayeredBacktest(n_quantiles=5)
            lb_result = lb.calculate(result, storage)
            lb_result.to_csv(os.path.join(output, f"{factor_name}_layered.csv"))
            click.echo(f"   ✅ Layered backtest saved")

        except Exception as e:
            echo_error(f"Failed: {e}")

    echo_success(f"\nResults saved to {output}")


@backtest_group.command("ic")
@click.option("--factor", required=True, help="Factor name")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", default="2020-01-01", help="Start date")
@click.option("--end", default="2024-12-31", help="End date")
@click.option("--output", "-o", help="Output HTML file")
def backtest_ic(factor: str, db: str, start: str, end: str, output: Optional[str]):
    """Run IC analysis for a factor."""
    from analysis.ic import ICAnalysis

    storage = get_db(db)

    echo_header(f"IC Analysis: {factor}")

    try:
        # Get factor data
        from factors.registry import FactorRegistry
        factor_obj = FactorRegistry.get(factor)
        if not factor_obj:
            echo_error(f"Factor not found: {factor}")
            sys.exit(1)

        data = storage.get_ohlcv(start_date=start, end_date=end)
        factor_values = factor_obj.calculate(data)

        # Calculate IC
        ic = ICAnalysis()
        ic_result = ic.calculate(factor_values, storage)

        click.echo(f"\n📊 IC Statistics:")
        click.echo(f"   Mean IC:     {ic_result['ic'].mean():.4f}")
        click.echo(f"   IC Std:      {ic_result['ic'].std():.4f}")
        click.echo(f"   IR (Mean/Std): {ic_result['ic'].mean() / ic_result['ic'].std():.4f}")
        click.echo(f"   Positive %:  {(ic_result['ic'] > 0).sum() / len(ic_result) * 100:.1f}%")

        if output:
            ic_result.to_csv(output)
            echo_success(f"Results saved to {output}")

    except Exception as e:
        echo_error(f"IC analysis failed: {e}")
        sys.exit(1)


@backtest_group.command("layered")
@click.option("--factor", required=True, help="Factor name")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", default="2020-01-01", help="Start date")
@click.option("--end", default="2024-12-31", help="End date")
@click.option("--n-quintiles", default=5, help="Number of quantiles")
@click.option("--output", "-o", help="Output HTML file")
def backtest_layered(factor: str, db: str, start: str, end: str,
                     n_quintiles: int, output: Optional[str]):
    """Run layered backtest for a factor."""
    from analysis.layered import LayeredBacktest

    storage = get_db(db)

    echo_header(f"Layered Backtest: {factor}")

    try:
        from factors.registry import FactorRegistry
        factor_obj = FactorRegistry.get(factor)
        if not factor_obj:
            echo_error(f"Factor not found: {factor}")
            sys.exit(1)

        data = storage.get_ohlcv(start_date=start, end_date=end)
        factor_values = factor_obj.calculate(data)

        lb = LayeredBacktest(n_quantiles=n_quintiles)
        result = lb.calculate(factor_values, storage)

        click.echo(f"\n📊 Layered Backtest Results:")
        click.echo(result.to_string())

        if output:
            result.to_csv(output)
            echo_success(f"Results saved to {output}")

    except Exception as e:
        echo_error(f"Layered backtest failed: {e}")
        sys.exit(1)


# =============================================================================
# Report Commands
# =============================================================================

@click.group(name="report")
@click.help_option("-h", "--help")
def report_group():
    """Report generation commands."""
    pass


@report_group.command("generate")
@click.option("--factors", required=True, help="Comma-separated factor names")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", default="2020-01-01", help="Start date")
@click.option("--end", default="2024-12-31", help="End date")
@click.option("--output", "-o", default="report.html", help="Output HTML file")
def report_generate(factors: str, db: str, start: str, end: str, output: str):
    """Generate HTML report for factors."""
    from analysis.report import HTMLReportGenerator

    storage = get_db(db)
    factor_list = [f.strip() for f in factors.split(",")]

    echo_header(f"Generating Report")
    click.echo(f"Factors: {', '.join(factor_list)}")
    click.echo(f"Period: {start} to {end}")

    try:
        generator = HTMLReportGenerator()
        generator.generate(
            storage=storage,
            factors=factor_list,
            start_date=start,
            end_date=end,
            output_path=output,
        )
        echo_success(f"Report saved to {output}")
    except Exception as e:
        echo_error(f"Report generation failed: {e}")
        sys.exit(1)


# =============================================================================
# Main CLI Entry Point
# =============================================================================

@click.group()
@click.version_option(version=VERSION)
@click.help_option("-h", "--help")
def cli():
    """Factor Pipeline CLI - Quantitative factor analysis toolkit.

    A comprehensive CLI for factor research with DuckDB backend.
    """
    pass


# Register all command groups
cli.add_command(data_group)
cli.add_command(factor_group)
cli.add_command(backtest_group)
cli.add_command(report_group)


# Aliases for convenience
@cli.command("init", help="Initialize database (alias for 'fp data init')")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def init_alias(db: str):
    """Initialize database."""
    storage = DuckDBStorage(db)
    storage.init_schema()
    echo_success(f"Database initialized: {db}")


@cli.command("shell", help="Interactive SQL shell")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def shell(db: str):
    """Interactive SQL shell."""
    storage = get_db(db)
    click.echo("Factor Pipeline SQL Shell")
    click.echo("Type 'exit' or 'quit' to exit\n")

    while True:
        try:
            sql = click.prompt("sql> ", default="")
            if sql.lower() in ("exit", "quit"):
                break
            if sql.strip():
                df = storage.query(sql)
                click.echo(df.to_string())
                click.echo()
        except Exception as e:
            echo_error(f"Error: {e}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
