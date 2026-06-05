"""
Download all A-share daily OHLCV data from baostock (2015–present) → DuckDB.

Usage:
    python scripts/download_baostock.py
    python scripts/download_baostock.py --workers 4 --start 2015-01-01
    python scripts/download_baostock.py --reset
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date

import duckdb
from tqdm import tqdm

logger = logging.getLogger("download_baostock")

# ── Constants ──────────────────────────────────────────────────────
_DEFAULT_DB = "data/a_share_daily.duckdb"
_DEFAULT_START = "2015-01-01"
_TODAY = date.today().isoformat()
_BATCH_SIZE = 50          # commit interval
_RETRY_MAX = 3
_STOCK_FILTERS = ("sh.6", "sz.0", "sz.2", "sz.3", "bj.8")


# ── Config ─────────────────────────────────────────────────────────
@dataclass
class BaostockConfig:
    """Configuration for the baostock downloader."""
    db_path: str = _DEFAULT_DB
    start_date: str = _DEFAULT_START
    end_date: str = _TODAY
    workers: int = 4
    reset: bool = False
    verbose: bool = False
    batch_size: int = _BATCH_SIZE
    retry_max: int = _RETRY_MAX

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> BaostockConfig:
        parser = argparse.ArgumentParser(
            description="Download all A-share daily OHLCV data via baostock → DuckDB",
        )
        parser.add_argument(
            "--db", default=_DEFAULT_DB,
            help=f"DuckDB output path (default: {_DEFAULT_DB})",
        )
        parser.add_argument(
            "--start", default=_DEFAULT_START,
            help=f"Start date YYYY-MM-DD (default: {_DEFAULT_START})",
        )
        parser.add_argument(
            "--end", default=_TODAY,
            help=f"End date YYYY-MM-DD (default: {_TODAY})",
        )
        parser.add_argument(
            "--workers", type=int, default=4,
            help="Concurrent worker processes (default: 4, 1 = sequential)",
        )
        parser.add_argument(
            "--reset", action="store_true",
            help="Drop existing table and re-download everything",
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true",
            help="Enable debug logging",
        )
        ns = parser.parse_args(argv)
        return cls(
            db_path=ns.db,
            start_date=ns.start,
            end_date=ns.end,
            workers=ns.workers,
            reset=ns.reset,
            verbose=ns.verbose,
        )


# ── baostock helpers ───────────────────────────────────────────────
def _fetch_stock(args: tuple[str, str, str]) -> tuple[str, list[list[str]]]:
    """Fetch daily k-line for a single stock (runs in a separate process).

    Returns (code, rows) where rows is a list of raw baostock row-lists.
    """
    code, start, end = args
    import baostock as bs  # lazy-import so each worker process loads it fresh

    for attempt in range(_RETRY_MAX):
        try:
            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("%s login failed (attempt %d): %s", code, attempt + 1, lg.error_msg)
                time.sleep(1.0 + attempt * 0.5)
                continue

            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,pctChg,isST",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3",  # 不复权
            )
            rows: list[list[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            bs.logout()
            return code, rows

        except Exception as exc:
            logger.warning("%s fetch error (attempt %d): %s", code, attempt + 1, exc)
            time.sleep(1.0 + attempt * 0.5)
            try:
                bs.logout()
            except Exception:
                pass

    return code, []


def _list_stocks() -> list[str]:
    """Return all A-share stock codes from baostock (excludes indices)."""
    import baostock as bs

    bs.login()
    codes: list[str] = []
    rs = bs.query_all_stock(day="2025-01-02")
    while rs.error_code == "0" and rs.next():
        code = rs.get_row_data()[0]
        if any(code.startswith(p) for p in _STOCK_FILTERS):
            codes.append(code)
    bs.logout()
    return codes


# ── DuckDB helpers ─────────────────────────────────────────────────
def _init_db(db_path: str, reset: bool = False) -> duckdb.DuckDBPyConnection:
    """Create or connect to DuckDB, optionally dropping existing data."""
    con = duckdb.connect(db_path)
    if reset:
        con.execute("DROP TABLE IF EXISTS daily_kline")
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_kline (
            date        DATE,
            code        VARCHAR,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      BIGINT,
            amount      DOUBLE,
            pct_chg     DOUBLE,
            is_st       INTEGER
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_code_date ON daily_kline (code, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date     ON daily_kline (date)")
    return con


def _rows_to_inserts(rows: list[list[str]]) -> list[tuple]:
    """Convert raw baostock rows to DuckDB INSERT tuples."""
    out: list[tuple] = []
    for r in rows:
        try:
            out.append((
                r[0],                                    # date
                r[1],                                    # code
                float(r[2])  if r[2] else None,          # open
                float(r[3])  if r[3] else None,          # high
                float(r[4])  if r[4] else None,          # low
                float(r[5])  if r[5] else None,          # close
                int(float(r[6])) if r[6] else None,      # volume
                float(r[7])  if r[7] else None,          # amount
                float(r[8])  if r[8] else None,          # pct_chg
                int(float(r[9])) if r[9] else None,      # is_st
            ))
        except (ValueError, IndexError):
            continue
    return out


def _existing_codes(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Return set of stock codes already stored."""
    try:
        return {r[0] for r in con.execute("SELECT DISTINCT code FROM daily_kline").fetchall()}
    except Exception:
        return set()


def _insert_batch(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    con.executemany(
        "INSERT INTO daily_kline VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def _summary(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute("""
        SELECT
            COUNT(*)                                          AS total_rows,
            COUNT(DISTINCT code)                              AS stocks,
            COUNT(DISTINCT date)                              AS days,
            MIN(date)::VARCHAR                                 AS min_date,
            MAX(date)::VARCHAR                                 AS max_date
        FROM daily_kline
    """).fetchdf()
    logger.info("Database summary:\n%s", df.to_string(index=False))
    print(df.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────
def main() -> None:
    cfg = BaostockConfig.from_cli()
    started_at = time.time()

    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle SIGINT gracefully
    shutdown_flag = False

    def _handle_sigint(*_):
        nonlocal shutdown_flag
        if not shutdown_flag:
            logger.warning("Received SIGINT — finishing current batch then exiting…")
            shutdown_flag = True
        else:
            logger.warning("Force exit")
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sigint)

    # 1. Initialise DuckDB
    con = _init_db(cfg.db_path, reset=cfg.reset)
    ingested = 0  # rows inserted in this run
    inserted = 0  # stocks inserted in this run
    failed = 0

    try:
        # 2. Determine pending stocks
        all_codes = _list_stocks()
        logger.info("Total A-share stocks found: %d", len(all_codes))

        if cfg.reset:
            pending = all_codes
        else:
            done = _existing_codes(con)
            pending = [c for c in all_codes if c not in done]
            logger.info("Already in DB: %d  Pending: %d", len(done), len(pending))

        if not pending:
            logger.info("Nothing to download.")
            _summary(con)
            return

        # 3. Build task list
        tasks = [(code, cfg.start_date, cfg.end_date) for code in pending]
        logger.info(
            "Downloading %d stocks (%s – %s) with %d worker(s) …",
            len(tasks), cfg.start_date, cfg.end_date, cfg.workers,
        )

        # 4. Fetch & insert
        if cfg.workers == 1:
            # ── Sequential ──
            logger.debug("Sequential mode (workers=1)")
            pbar = tqdm(tasks, desc="Downloading", unit="stock", ncols=90)
            for task in pbar:
                if shutdown_flag:
                    break
                code, rows = _fetch_stock(task)
                if rows:
                    data = _rows_to_inserts(rows)
                    if data:
                        _insert_batch(con, data)
                        ingested += len(data)
                        inserted += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                pbar.set_postfix(ok=inserted, fail=failed, rows=ingested)
        else:
            # ── Parallel (multiprocessing) ──
            logger.debug("Parallel mode (workers=%d)", cfg.workers)
            with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {pool.submit(_fetch_stock, t): t[0] for t in tasks}
                pbar = tqdm(
                    total=len(tasks), desc="Downloading",
                    unit="stock", ncols=90,
                )
                batch: list[tuple] = []

                for fut in as_completed(futures):
                    if shutdown_flag:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break

                    code, rows = fut.result()
                    if rows:
                        data = _rows_to_inserts(rows)
                        if data:
                            batch.extend(data)
                            inserted += 1
                        else:
                            failed += 1
                    else:
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix(ok=inserted, fail=failed, rows=ingested)

                    # Periodic commit
                    if len(batch) >= cfg.batch_size and not shutdown_flag:
                        _insert_batch(con, batch)
                        ingested += len(batch)
                        batch = []

                # Flush remaining
                if batch:
                    _insert_batch(con, batch)
                    ingested += len(batch)

                pbar.close()

    finally:
        con.close()

    elapsed = time.time() - started_at
    logger.info(
        "Done — %d stocks (%d rows) ingested in %.1f min (failed: %d)",
        inserted, ingested, elapsed / 60, failed,
    )

    # 5. Verify
    con = _init_db(cfg.db_path)
    try:
        _summary(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
