"""
Fill missing HS300 OHLCV data in ohlcv_hs300.duckdb.
Single worker, low memory, batch commit, WAL checkpoint at end.

Usage:
    python scripts/fill_hs300_gaps.py
    python scripts/fill_hs300_gaps.py --start 2015-01-01
"""
from __future__ import annotations

import argparse
import gc
import logging
import signal
import sys
import time
from datetime import date, datetime
from pathlib import Path

import duckdb

logger = logging.getLogger("fill_hs300")

TODAY = date.today().isoformat()
BATCH_SIZE = 200  # rows per commit
RETRY_MAX = 3


# ── baostock helpers ──────────────────────────────────────────────
def _fetch_stock(symbol: str, start: str, end: str) -> list[dict]:
    """Fetch daily k-line for one stock. Returns list of dicts."""
    import baostock as bs

    for attempt in range(RETRY_MAX):
        try:
            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("%s login fail attempt %d: %s", symbol, attempt+1, lg.error_msg)
                time.sleep(1 + attempt)
                continue

            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume,amount",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3",
            )
            rows = []
            while rs.next():
                r = rs.get_row_data()
                rows.append({
                    "date": r[0],
                    "open": r[1] or None,
                    "high": r[2] or None,
                    "low": r[3] or None,
                    "close": r[4] or None,
                    "volume": r[5] or None,
                    "amount": r[6] or None,
                })
            bs.logout()
            return rows

        except Exception as exc:
            logger.warning("%s fetch error attempt %d: %s", symbol, attempt+1, exc)
            time.sleep(1 + attempt)
            try:
                bs.logout()
            except Exception:
                pass

    return []


def _symbol_from_code(code: str) -> str:
    """Convert raw code like '000001' to baostock format 'sz.000001'."""
    c = code.strip()
    if c.startswith("6"):
        return f"sh.{c}"
    elif c.startswith(("0", "3")):
        return f"sz.{c}"
    elif c.startswith("9"):
        return f"sh.{c}"
    else:
        return f"sz.{c}"


# ── main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fill HS300 gaps")
    parser.add_argument("--start", default="2015-01-01", help="Global start date")
    parser.add_argument("--codes", default="data/hs300_codes.txt", help="Codes list file")
    parser.add_argument("--db", default="data/ohlcv_hs300.duckdb", help="DuckDB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # SIGINT handling
    shutdown = False
    def _sigint(*_):
        nonlocal shutdown
        if shutdown:
            sys.exit(1)
        logger.warning("SIGINT received — finishing current stock then exiting…")
        shutdown = True
    signal.signal(signal.SIGINT, _sigint)

    # 1. Load codes
    codes_path = Path(args.codes)
    raw_codes = [line.strip() for line in codes_path.read_text().splitlines() if line.strip()]
    symbols = sorted(set(_symbol_from_code(c) for c in raw_codes))
    logger.info("Loaded %d unique symbols from %d codes", len(symbols), len(raw_codes))

    # 2. Connect DB, get per-symbol latest date
    db_path = args.db
    con = duckdb.connect(db_path)
    try:
        existing = {}
        try:
            rows = con.execute(
                "SELECT symbol, MAX(date) FROM daily_ohlcv GROUP BY symbol"
            ).fetchall()
            existing = {r[0]: r[1] for r in rows}
        except Exception:
            con.execute("""
                CREATE TABLE IF NOT EXISTS daily_ohlcv (
                    date   DATE,
                    symbol VARCHAR,
                    open   DOUBLE,
                    high   DOUBLE,
                    low    DOUBLE,
                    close  DOUBLE,
                    volume BIGINT,
                    amount DOUBLE
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON daily_ohlcv (symbol)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_ohlcv (date)")

        logger.info("Already in DB: %d symbols", len(existing))

        # 3. Download per stock — single worker, sequential
        total_inserted = 0
        total_rows = 0
        failed = 0
        skipped = 0
        start_time = time.time()

        for i, symbol in enumerate(symbols, 1):
            if shutdown:
                break

            # Determine start date for this stock
            last_date = existing.get(symbol)
            if last_date is not None:
                if hasattr(last_date, 'strftime'):
                    last_str = last_date.strftime('%Y-%m-%d')
                else:
                    last_str = str(last_date)
                # If already up to today, skip
                if last_str >= TODAY:
                    skipped += 1
                    if skipped % 50 == 0:
                        logger.info("Progress %d/%d — skipped=%d ok=%d fail=%d rows=%d",
                                   i, len(symbols), skipped, total_inserted, failed, total_rows)
                    continue
                fetch_start = last_str
            else:
                fetch_start = args.start

            # Fetch
            rows = _fetch_stock(symbol, fetch_start, TODAY)
            if not rows:
                failed += 1
                logger.debug("%s: no data", symbol)
                continue

            # Insert immediately (low memory)
            batch = []
            for r in rows:
                try:
                    batch.append((
                        r["date"],
                        symbol,
                        float(r["open"]) if r["open"] else None,
                        float(r["high"]) if r["high"] else None,
                        float(r["low"]) if r["low"] else None,
                        float(r["close"]) if r["close"] else None,
                        int(float(r["volume"])) if r["volume"] else None,
                        float(r["amount"]) if r["amount"] else None,
                    ))
                except (ValueError, TypeError):
                    continue

            if batch:
                con.executemany(
                    "INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)",
                    batch,
                )
                total_inserted += 1
                total_rows += len(batch)
                # Free memory
                del batch, rows
                gc.collect()

            # Progress every 20 stocks
            if i % 20 == 0:
                elapsed = time.time() - start_time
                logger.info("Progress %d/%d — ok=%d fail=%d skipped=%d rows=%d (%.1f min)",
                           i, len(symbols), total_inserted, failed, skipped, total_rows, elapsed/60)

    finally:
        con.close()

    elapsed = time.time() - start_time
    logger.info(
        "Done — inserted %d stocks, %d rows, failed %d, skipped %d (%.1f min)",
        total_inserted, total_rows, failed, skipped, elapsed / 60,
    )

    # 4. WAL checkpoint
    logger.info("Running WAL checkpoint…")
    try:
        con2 = duckdb.connect(db_path)
        con2.execute("CHECKPOINT")
        con2.close()
        logger.info("WAL checkpoint done")
    except Exception as e:
        logger.warning("Checkpoint failed: %s", e)

    # 5. Final summary
    logger.info("=== Summary ===")
    try:
        con3 = duckdb.connect(db_path, read_only=True)
        r = con3.execute("""
            SELECT COUNT(*) as rows,
                   COUNT(DISTINCT symbol) as stocks,
                   MIN(date) as min_date,
                   MAX(date) as max_date
            FROM daily_ohlcv
        """).fetchone()
        logger.info("Total: %d rows, %d stocks, %s ~ %s", r[0], r[1], r[2], r[3])
        con3.close()
    except Exception as e:
        logger.warning("Summary error: %s", e)


if __name__ == "__main__":
    main()
