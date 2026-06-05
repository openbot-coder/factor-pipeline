"""DuckDB SQL Factor Engine — compile Alpha191 expressions to window SQL.

Core operators → SQL mapping:
  Ref(x, n)    → LAG(x, n) OVER (PARTITION BY code ORDER BY date)
  Mean(x, n)   → AVG(x) OVER (PARTITION BY code ORDER BY date ROWS n-1 PRECEDING)
  Std(x, n)    → STDDEV_SAMP(x) OVER (...)
  Sum(x, n)    → SUM(x) OVER (...)
  Delta(x, n)  → x - LAG(x, n) OVER (...)
  Max(x, n)    → MAX(x) OVER (...)
  Min(x, n)    → MIN(x) OVER (...)
  Rank(x)      → PERCENT_RANK() OVER (PARTITION BY date ORDER BY x)
  CSRank(x)    → same as Rank (cross-section)
  Corr(x,y,n)  → CORR(x, y) OVER (... ROWS n-1 PRECEDING)
  Iif(c,a,b)   → CASE WHEN c THEN a ELSE b END
  Sign(x)      → SIGN(x)

CTE decomposition for nested windows:
  Std(Mean($close, 20), 60) needs:
    WITH cte_mean AS (
      SELECT *, AVG(close) OVER (PARTITION BY code ORDER BY date ROWS 19 PRECEDING) AS mean_20
    )
    SELECT *, STDDEV_SAMP(mean_20) OVER (...) AS factor
    FROM cte_mean

Usage:
    engine = DuckDBFactorEngine("data/a_share_daily.duckdb")
    factor_df = engine.compute_alpha001(start="2024-01-01", end="2025-12-31")
"""

from __future__ import annotations

import re
from typing import Optional
import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Column mapping: DB columns → factor variable names
# ---------------------------------------------------------------------------
COL_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "pct_chg": "pct_chg",
}

# Columns that need $ prefix in Qlib expressions
QLIB_VARS = {"open", "high", "low", "close", "volume", "amount"}


class DuckDBFactorEngine:
    """Execute factor expressions as DuckDB SQL on a single database."""

    def __init__(self, db_path: str, table: str = "daily_kline"):
        self.db_path = db_path
        self.table = table
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self.db_path, read_only=True)
        return self._con

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _base_sql(self, start: str = "2020-01-01", end: str = "2026-12-31") -> str:
        """Base CTE selecting relevant columns."""
        return f"""
        base AS (
            SELECT
                date,
                code,
                COALESCE(open, 0) AS open,
                COALESCE(high, 0) AS high,
                COALESCE(low, 0) AS low,
                COALESCE(close, 0) AS close,
                COALESCE(volume, 0) AS volume,
                COALESCE(amount, 0) AS amount,
                COALESCE(pct_chg, 0) AS pct_chg
            FROM {self.table}
            WHERE date >= '{start}' AND date <= '{end}'
              AND volume > 0
        )
        """

    def _run_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""
        con = self.connect()
        result = con.execute(sql).fetchdf()
        return result

    # -----------------------------------------------------------------------
    # Individual factor SQL generators
    # -----------------------------------------------------------------------

    def sql_alpha001(self, start: str, end: str) -> str:
        """Alpha001: -1 * CORR(RANK(DELTA(LN(VOLUME), 1)), RANK((CLOSE - OPEN) / OPEN), 6)

        Nested windows: need CTE for inner ranks, then rolling corr.
        Note: DuckDB LOG() is base-10, we need LN() for natural log to match Python np.log().
        """
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code,
                LN(NULLIF(volume, 0)) AS log_vol,
                (close - open) / NULLIF(open, 0) AS co_ret
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                log_vol - LAG(log_vol, 1) OVER (PARTITION BY code ORDER BY date) AS dlv,
                co_ret
            FROM step1
        ),
        step3 AS (
            SELECT
                date, code,
                dlv,
                co_ret,
                PERCENT_RANK() OVER (PARTITION BY date ORDER BY dlv) AS rank_dlv,
                PERCENT_RANK() OVER (PARTITION BY date ORDER BY co_ret) AS rank_cor
            FROM step2
            WHERE dlv IS NOT NULL
        ),
        step4 AS (
            SELECT
                date, code,
                rank_dlv,
                rank_cor,
                CORR(rank_dlv, rank_cor) OVER (
                    PARTITION BY code
                    ORDER BY date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                ) AS roll_corr
            FROM step3
        )
        SELECT
            date, code,
            -roll_corr AS factor
        FROM step4
        WHERE roll_corr IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha002(self, start: str, end: str) -> str:
        """Alpha002: -1 * delta(((close - low) - (high - close)) / (high - low), 1)
        Single window — no nesting needed.
        """
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code,
                ((close - low) - (high - close)) / NULLIF(high - low, 0) AS inner_val
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                inner_val,
                inner_val - LAG(inner_val, 1) OVER (PARTITION BY code ORDER BY date) AS delta_val
            FROM step1
        )
        SELECT
            date, code,
            -delta_val AS factor
        FROM step2
        WHERE delta_val IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha003(self, start: str, end: str) -> str:
        """Alpha003: -1 * SUM(term, 6) where term uses LAG logic."""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code, close, low, high,
                LAG(close, 1) OVER (PARTITION BY code ORDER BY date) AS prev_c
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                CASE
                    WHEN close = prev_c THEN 0
                    WHEN close > prev_c THEN close - LEAST(low, prev_c)
                    ELSE close - GREATEST(high, prev_c)
                END AS term
            FROM step1
            WHERE prev_c IS NOT NULL
        ),
        step3 AS (
            SELECT
                date, code,
                SUM(term) OVER (
                    PARTITION BY code
                    ORDER BY date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                ) AS roll_sum
            FROM step2
        )
        SELECT
            date, code,
            -roll_sum AS factor
        FROM step3
        WHERE roll_sum IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha004(self, start: str, end: str) -> str:
        """Alpha004: Conditional based on MA8, STD8, MA2, VOL20."""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code, close, volume,
                AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS ma8,
                STDDEV_SAMP(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS sd8,
                AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS ma2,
                AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS mv20
            FROM base
        )
        SELECT
            date, code,
            CASE
                WHEN (ma8 + sd8) < ma2 THEN -1
                WHEN ma2 < (ma8 - sd8) THEN 1
                WHEN volume >= mv20 THEN 1
                ELSE -1
            END AS factor
        FROM step1
        WHERE ma8 IS NOT NULL AND sd8 IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha005(self, start: str, end: str) -> str:
        """Alpha005: -1 * TSMAX(CORR(TSRANK(volume,5), TSRANK(high,5), 5), 3)
        Requires CTE decomposition: TSRANK → CORR → TSMAX.
        """
        return f"""
        WITH
        {self._base_sql(start, end)},
        tsrank AS (
            SELECT
                date, code, volume, high,
                -- TSRANK(volume, 5) = percentile of current value in last 5 rows
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY volume
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS trk_v_raw,
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY high
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS trk_h_raw
            FROM base
        ),
        tsrank_windowed AS (
            SELECT
                date, code,
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY volume
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS trk_v,
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY high
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS trk_h
            FROM base
        ),
        corr_step AS (
            SELECT
                date, code,
                CORR(trk_v, trk_h) OVER (
                    PARTITION BY code
                    ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS roll_corr
            FROM tsrank_windowed
        ),
        tsmax AS (
            SELECT
                date, code,
                MAX(roll_corr) OVER (
                    PARTITION BY code
                    ORDER BY date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS tsmax_val
            FROM corr_step
        )
        SELECT
            date, code,
            -tsmax_val AS factor
        FROM tsmax
        WHERE tsmax_val IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha006(self, start: str, end: str) -> str:
        """Alpha006: -1 * RANK(SIGN(DELTA(open*0.85 + high*0.15, 4)))"""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code,
                open * 0.85 + high * 0.15 AS weighted
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                weighted - LAG(weighted, 4) OVER (PARTITION BY code ORDER BY date) AS delta_val
            FROM step1
        ),
        step3 AS (
            SELECT
                date, code,
                SIGN(delta_val) AS sign_val
            FROM step2
            WHERE delta_val IS NOT NULL
        ),
        step4 AS (
            SELECT
                date, code,
                sign_val,
                -PERCENT_RANK() OVER (PARTITION BY date ORDER BY sign_val) AS factor
            FROM step3
        )
        SELECT date, code, factor
        FROM step4
        ORDER BY date, code
        """

    def sql_alpha008(self, start: str, end: str) -> str:
        """Alpha008: -1 * RANK(DELTA((high+low)/10 + VWAP*0.8, 4))"""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code,
                (high + low) / 10.0 + (amount / NULLIF(volume, 0)) * 0.8 AS mid_vwap
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                mid_vwap - LAG(mid_vwap, 4) OVER (PARTITION BY code ORDER BY date) AS delta_val
            FROM step1
        )
        SELECT
            date, code,
            -PERCENT_RANK() OVER (PARTITION BY date ORDER BY delta_val) AS factor
        FROM step2
        WHERE delta_val IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha012(self, start: str, end: str) -> str:
        """Alpha012: SIGN(DELTA(volume,1)) * (-1 * DELTA(close,1))
        Simple: two LAGs, no nesting.
        """
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code, close, volume,
                volume - LAG(volume, 1) OVER (PARTITION BY code ORDER BY date) AS d_vol,
                close - LAG(close, 1) OVER (PARTITION BY code ORDER BY date) AS d_close
            FROM base
        )
        SELECT
            date, code,
            SIGN(d_vol) * (-d_close) AS factor
        FROM step1
        WHERE d_vol IS NOT NULL AND d_close IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha014(self, start: str, end: str) -> str:
        """Alpha014: -1 * RANK(DELTA(RET, 3)) where RET = -1 * (low - close) / open"""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code,
                -1.0 * (low - close) / NULLIF(open, 0) AS ret_val
            FROM base
        ),
        step2 AS (
            SELECT
                date, code,
                ret_val - LAG(ret_val, 3) OVER (PARTITION BY code ORDER BY date) AS delta_val
            FROM step1
        )
        SELECT
            date, code,
            -PERCENT_RANK() OVER (PARTITION BY date ORDER BY delta_val) AS factor
        FROM step2
        WHERE delta_val IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha018(self, start: str, end: str) -> str:
        """Alpha018: CLOSE - DELAY(CLOSE, 5)
        5-day return as raw factor."""
        return f"""
        WITH
        {self._base_sql(start, end)}
        SELECT
            date, code,
            close - LAG(close, 5) OVER (PARTITION BY code ORDER BY date) AS factor
        FROM base
        ORDER BY date, code
        """

    def sql_alpha020(self, start: str, end: str) -> str:
        """Alpha020: (CLOSE - OPEN) / ((HIGH - LOW) + 0.001)
        No window — single-period factor.
        """
        return f"""
        WITH
        {self._base_sql(start, end)}
        SELECT
            date, code,
            (close - open) / (high - low + 0.001) AS factor
        FROM base
        ORDER BY date, code
        """

    def sql_alpha021(self, start: str, end: str) -> str:
        """Alpha021: Multi-step: MEAN(volume,8) → MEAN(term,2) → condition"""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code, close, open, high, low, volume,
                AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS mv8,
                AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS mc8
            FROM base
        ),
        step2 AS (
            SELECT
                date, code, volume, mv8, mc8,
                CASE
                    WHEN close > mc8 THEN (low - mc8) * volume
                    ELSE 0
                END AS term1,
                CASE
                    WHEN close < mc8 THEN (high - mc8) * volume
                    ELSE 0
                END AS term2,
                CASE
                    WHEN close > mc8 THEN (low - close) * volume / NULLIF(mv8, 0)
                    ELSE (high - close) * volume / NULLIF(mv8, 0)
                END AS term3
            FROM step1
            WHERE mv8 > 0
        ),
        step3 AS (
            SELECT
                date, code,
                AVG(term1) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS mt1,
                AVG(term2) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS mt2,
                AVG(term3) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS mt3
            FROM step2
        )
        SELECT
            date, code,
            mt1 - mt2 + mt3 AS factor
        FROM step3
        WHERE mt1 IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha023(self, start: str, end: str) -> str:
        """Alpha023: Simple MA ratio"""
        return f"""
        WITH
        {self._base_sql(start, end)},
        step1 AS (
            SELECT
                date, code, high,
                AVG(high) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20
            FROM base
        )
        SELECT
            date, code,
            (high - ma20) / NULLIF(ma20, 0) AS factor
        FROM step1
        WHERE ma20 > 0
        ORDER BY date, code
        """

    def sql_alpha024(self, start: str, end: str) -> str:
        """Alpha024: -1 * DELTA(MEAN(close,5), 5)
        MA5 → Delta of MA5 by 5. One nested window (MA inside Delta).
        Need CTE.
        """
        return f"""
        WITH
        {self._base_sql(start, end)},
        ma5 AS (
            SELECT
                date, code,
                AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma_val
            FROM base
        )
        SELECT
            date, code,
            -(ma_val - LAG(ma_val, 5) OVER (PARTITION BY code ORDER BY date)) AS factor
        FROM ma5
        WHERE ma_val IS NOT NULL
        ORDER BY date, code
        """

    def sql_alpha026(self, start: str, end: str) -> str:
        """Alpha026: -1 * MAX(CORR(TSRANK(volume,5), TSRANK(high,5), 5), 3)
        Similar to alpha005 but max instead of rank of max."""
        return f"""
        WITH
        {self._base_sql(start, end)},
        tsrank AS (
            SELECT
                date, code, volume, high,
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY volume
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS trk_v,
                PERCENT_RANK() OVER (
                    PARTITION BY code
                    ORDER BY high
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS trk_h
            FROM base
        ),
        corr_step AS (
            SELECT
                date, code,
                CORR(trk_v, trk_h) OVER (
                    PARTITION BY code
                    ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS roll_corr
            FROM tsrank
        )
        SELECT
            date, code,
            -MAX(roll_corr) OVER (
                PARTITION BY code
                ORDER BY date
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) AS factor
        FROM corr_step
        ORDER BY date, code
        """

    # -----------------------------------------------------------------------
    # Batch execution
    # -----------------------------------------------------------------------

    def compute(self, factor_name: str, start: str = "2020-01-01", end: str = "2026-12-31") -> pd.DataFrame:
        """Compute a factor by name, return DataFrame with date/code/factor."""
        method = getattr(self, f"sql_{factor_name}", None)
        if method is None:
            raise ValueError(f"Factor {factor_name} not implemented in DuckDB engine")
        sql = method(start, end)
        df = self._run_sql(sql)
        return df

    def compute_all(self, factors: list[str], start: str = "2020-01-01", end: str = "2026-12-31") -> dict[str, pd.DataFrame]:
        """Compute multiple factors, return dict of name → DataFrame."""
        results = {}
        for name in factors:
            try:
                df = self.compute(name, start, end)
                results[name] = df
                n_rows = len(df)
                n_stocks = df["code"].nunique()
                print(f"  {name}: {n_rows:,} rows, {n_stocks} stocks, factor range [{df['factor'].min():.4f}, {df['factor'].max():.4f}]")
            except Exception as e:
                print(f"  {name}: FAILED — {e}")
                results[name] = None
        return results


# ---------------------------------------------------------------------------
# Convenience: available factors
# ---------------------------------------------------------------------------

AVAILABLE_FACTORS = [
    "alpha001", "alpha002", "alpha003", "alpha004", "alpha005",
    "alpha006", "alpha008", "alpha012", "alpha014", "alpha018",
    "alpha020", "alpha021", "alpha023", "alpha024", "alpha026",
]


def benchmark(engine: DuckDBFactorEngine, factors: list[str] | None = None,
              start: str = "2024-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Benchmark DuckDB factor computation, return timing DataFrame."""
    import time

    if factors is None:
        factors = AVAILABLE_FACTORS

    rows = []
    for name in factors:
        t0 = time.perf_counter()
        try:
            df = engine.compute(name, start, end)
            elapsed = time.perf_counter() - t0
            rows.append({
                "factor": name,
                "rows": len(df),
                "stocks": df["code"].nunique(),
                "time_s": round(elapsed, 3),
                "status": "ok",
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            rows.append({
                "factor": name,
                "rows": 0,
                "stocks": 0,
                "time_s": round(elapsed, 3),
                "status": f"error: {e}",
            })

    return pd.DataFrame(rows)
