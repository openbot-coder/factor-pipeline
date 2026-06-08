"""Command-line interface for data management.

Usage:
    python -m data.cli --help
    python -m data.cli info --db data/ohlcv.duckdb
    python -m data.cli import-csv data.csv --db data/ohlcv.duckdb
    python -m data.cli run "Mean($close, 20)" --db data/ohlcv.duckdb
"""

from __future__ import annotations

import os
import sys

import click
import pandas as pd

from factor_pipeline.data.storage import DuckDBStorage

# =============================================================================
# CLI Configuration
# =============================================================================

DEFAULT_DB = os.environ.get("FACTOR_PIPELINE_DB", "data/ohlcv.duckdb")


# =============================================================================
# Helper Functions
# =============================================================================


def get_db(db_path: str | None) -> DuckDBStorage:
    """Get database connection."""
    path = db_path or DEFAULT_DB
    if path == ":memory:":
        return DuckDBStorage(":memory:")
    return DuckDBStorage(path)


def print_info(info: dict) -> None:
    """Print database info in formatted style."""
    click.echo(f"\n📊 Database: {info['db_path']}")
    click.echo(f"   Size: {info.get('db_size_mb', 0):.2f} MB")

    if info.get("date_range"):
        click.echo(f"   Date Range: {info['date_range'][0]} to {info['date_range'][1]}")
    if info.get("symbols_count"):
        click.echo(f"   Symbols: {info['symbols_count']}")

    click.echo(f"\n📋 Tables ({len(info['tables'])}):")
    for table, stats in info["tables"].items():
        click.echo(f"   - {table}: {stats['rows']:,} rows")

    click.echo()


def print_table_info(tables: list[dict]) -> None:
    """Print table schema."""
    current_table = None
    for col in tables:
        if col["table_name"] != current_table:
            if current_table is not None:
                click.echo("")
            current_table = col["table_name"]
            click.echo(f"📋 {current_table}:")
        nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
        click.echo(f"   {col['column_name']:20} {col['column_type']:15} {nullable}")


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Factor Pipeline Data CLI - DuckDB data management tool.

    Environment Variables:
        FACTOR_PIPELINE_DB: Default database path

    Examples:
        python -m data.cli info
        python -m data.cli import-csv data.csv --table daily_ohlcv
        python -m data.cli query "SELECT * FROM daily_ohlcv LIMIT 10"
    """
    pass


@cli.command("info")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--schema", is_flag=True, help="Show table schemas")
def info_cmd(db: str, schema: bool):
    """Show database information."""
    try:
        storage = get_db(db)
        info = storage.info()
        print_info(info)

        if schema:
            tables = storage.show_schema()
            print_table_info(tables)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("schema")
@click.argument("table")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def schema_cmd(table: str, db: str):
    """Show table schema."""
    try:
        storage = get_db(db)
        result = storage.query(f"DESCRIBE {table}")
        click.echo(f"\n📋 Schema for '{table}':\n")
        click.echo(result.to_string(index=False))
        click.echo()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("tables")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def tables_cmd(db: str):
    """List all tables."""
    try:
        storage = get_db(db)
        tables = storage.list_tables()
        click.echo("\n📋 Tables:")
        for t in tables:
            count = storage.fetchone(f"SELECT COUNT(*) FROM {t}")[0]
            click.echo(f"   - {t}: {count:,} rows")
        click.echo()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("init")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def init_cmd(db: str):
    """Initialize database schema."""
    try:
        storage = get_db(db)
        storage.init_schema()
        click.echo("✅ Database schema initialized successfully!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("import-csv")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--table", required=True, help="Target table name")
@click.option("--date-col", default="date", help="Date column name")
@click.option("--symbol-col", default="symbol", help="Symbol column name")
@click.option(
    "--if-exists",
    default="append",
    type=click.Choice(["append", "replace", "fail"]),
    help="How to handle existing data",
)
@click.option("--skip-rows", default=0, help="Number of rows to skip")
@click.option("--encoding", default="utf-8", help="File encoding")
def import_csv_cmd(
    csv_path: str,
    db: str,
    table: str,
    date_col: str,
    symbol_col: str,
    if_exists: str,
    skip_rows: int,
    encoding: str,
):
    """Import CSV file into database.

    Examples:
        python -m data.cli import-csv data.csv --table daily_ohlcv
        python -m data.cli import-csv data.csv --table daily_ohlcv --date-col trade_date
        python -m data.cli import-csv data.csv --table daily_ohlcv --if-exists replace
    """
    try:
        click.echo(f"📥 Importing {csv_path} into '{table}'...")

        storage = get_db(db)

        # Read CSV to preview
        df_preview = pd.read_csv(csv_path, nrows=5, encoding=encoding)
        click.echo(f"   Columns: {', '.join(df_preview.columns.tolist())}")
        click.echo(f"   Preview:\n{df_preview.head().to_string()}\n")

        # Import
        rows = storage.import_csv(
            csv_path,
            table=table,
            date_col=date_col,
            symbol_col=symbol_col,
            if_exists=if_exists,
            skiprows=skip_rows,
        )

        click.echo(f"✅ Imported {rows:,} rows into '{table}'!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("import-dir")
@click.argument("dir_path", type=click.Path(exists=True))
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--table", required=True, help="Target table name")
@click.option("--pattern", default="*.csv", help="File pattern to match")
@click.option("--recursive", is_flag=True, help="Search recursively")
def import_dir_cmd(
    dir_path: str,
    db: str,
    table: str,
    pattern: str,
    recursive: bool,
):
    """Import all CSV files from directory.

    Examples:
        python -m data.cli import-dir ./csv_data --table daily_ohlcv
        python -m data.cli import-dir ./data --table daily_ohlcv --pattern "*.csv.gz"
    """
    try:
        from glob import glob

        if recursive:
            files = glob(os.path.join(dir_path, "**", pattern), recursive=True)
        else:
            files = glob(os.path.join(dir_path, pattern))

        if not files:
            click.echo(f"⚠️  No files found matching '{pattern}' in '{dir_path}'")
            return

        click.echo(f"📥 Found {len(files)} files to import...")

        storage = get_db(db)
        total_rows = 0

        for f in files:
            click.echo(f"   Importing {os.path.basename(f)}...")
            rows = storage.import_csv(f, table=table)
            total_rows += rows

        click.echo(f"✅ Imported {total_rows:,} rows from {len(files)} files!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("query")
@click.argument("sql")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--limit", default=100, help="Result limit")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "csv", "json"]),
    help="Output format",
)
@click.option("--output", "-o", help="Output file path")
def query_cmd(sql: str, db: str, limit: int, output_format: str, output: str | None):
    """Execute SQL query.

    Examples:
        python -m data.cli query "SELECT * FROM daily_ohlcv LIMIT 10"
        python -m data.cli query "SELECT symbol, COUNT(*) FROM daily_ohlcv GROUP BY symbol"
    """
    try:
        storage = get_db(db)

        # Add LIMIT if not present
        if "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        df = storage.query(sql)

        if output:
            if output.endswith(".csv"):
                df.to_csv(output, index=False)
            elif output.endswith(".json"):
                df.to_json(output, orient="records", indent=2)
            else:
                df.to_csv(output, index=False)
            click.echo(f"✅ Results saved to {output}")
        elif output_format == "csv":
            click.echo(df.to_csv(index=False))
        elif output_format == "json":
            click.echo(df.to_json(orient="records", indent=2))
        else:
            click.echo(df.to_string())

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("export")
@click.argument("sql")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option(
    "--format",
    "output_format",
    default="csv",
    type=click.Choice(["csv", "parquet"]),
    help="Output format",
)
def export_cmd(sql: str, db: str, output: str, output_format: str):
    """Export query results to file.

    Examples:
        python -m data.cli export "SELECT * FROM daily_ohlcv" -o output.csv
        python -m data.cli export "SELECT * FROM daily_ohlcv" -o output.parquet --format parquet
    """
    try:
        storage = get_db(db)

        if output_format == "csv":
            storage.export_csv(sql, output)
        else:
            storage.export_parquet(sql, output)

        click.echo(f"✅ Exported to {output}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("stats")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--table", help="Specific table to show stats")
def stats_cmd(db: str, table: str | None):
    """Show statistics for tables.

    Examples:
        python -m data.cli stats
        python -m data.cli stats --table daily_ohlcv
    """
    try:
        storage = get_db(db)

        tables = [table] if table else storage.list_tables()

        for t in tables:
            click.echo(f"\n📊 Statistics for '{t}':\n")
            df = storage.query(f"SELECT * FROM {t} LIMIT 0")

            for col in df.columns:
                col_data = storage.query(f"SELECT {col} FROM {t}")
                if pd.api.types.is_numeric_dtype(col_data[col]):
                    stats = col_data[col].describe()
                    click.echo(f"   {col}:")
                    click.echo(f"      count: {stats['count']:.0f}")
                    click.echo(f"      mean:  {stats['mean']:.4f}")
                    click.echo(f"      std:   {stats['std']:.4f}")
                    click.echo(f"      min:   {stats['min']:.4f}")
                    click.echo(f"      25%:   {stats['25%']:.4f}")
                    click.echo(f"      50%:   {stats['50%']:.4f}")
                    click.echo(f"      75%:   {stats['75%']:.4f}")
                    click.echo(f"      max:   {stats['max']:.4f}")
                else:
                    unique = col_data[col].nunique()
                    click.echo(f"   {col}: {unique} unique values")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("instruments")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--market", help="Filter by market (SSE, SZSE)")
@click.option("--active", is_flag=True, help="Only show active instruments")
@click.option("--output", "-o", help="Output file path")
def instruments_cmd(db: str, market: str | None, active: bool, output: str | None):
    """List instruments (symbols).

    Examples:
        python -m data.cli instruments
        python -m data.cli instruments --market SSE
        python -m data.cli instruments --active -o instruments.csv
    """
    try:
        storage = get_db(db)

        if active:
            today = pd.Timestamp.today().strftime("%Y-%m-%d")
            sql = f"""
                SELECT symbol, name, market, list_date
                FROM instruments
                WHERE list_date <= '{today}'
                AND (delist_date IS NULL OR delist_date >= '{today}')
            """
        else:
            sql = "SELECT symbol, name, market, list_date, delist_date FROM instruments"

        if market:
            sql = f"{sql.rstrip()} AND market = '{market}'"

        df = storage.query(sql)

        if output:
            df.to_csv(output, index=False)
            click.echo(f"✅ Exported {len(df)} instruments to {output}")
        else:
            click.echo(f"\n📋 Instruments ({len(df)} total):\n")
            click.echo(df.to_string(index=False))

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("calendar")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", help="Output file path")
def calendar_cmd(db: str, start: str | None, end: str | None, output: str | None):
    """Show trading calendar.

    Examples:
        python -m data.cli calendar
        python -m data.cli calendar --start 2024-01-01 --end 2024-12-31
    """
    try:
        storage = get_db(db)

        dates = storage.get_calendar(start, end)

        if output:
            df = pd.DataFrame({"date": dates})
            df.to_csv(output, index=False)
            click.echo(f"✅ Exported {len(dates)} dates to {output}")
        else:
            click.echo(f"\n📅 Trading Calendar ({len(dates)} days):\n")
            for d in dates[:10]:
                click.echo(f"   {d}")
            if len(dates) > 10:
                click.echo(f"   ... ({len(dates) - 10} more)")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("drop")
@click.argument("table")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.confirmation_option(prompt="Are you sure you want to drop the table?")
def drop_cmd(table: str, db: str):
    """Drop a table."""
    try:
        storage = get_db(db)
        storage.drop_table(table)
        click.echo(f"✅ Table '{table}' dropped!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("vacuum")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def vacuum_cmd(db: str):
    """Optimize database storage."""
    try:
        storage = get_db(db)
        storage.vacuum()
        click.echo("✅ Database optimized!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command("copy-table")
@click.argument("source")
@click.argument("target")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def copy_table_cmd(source: str, target: str, db: str):
    """Copy a table."""
    try:
        storage = get_db(db)
        storage.copy_table(source, target)
        click.echo(f"✅ Table '{source}' copied to '{target}'!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
