"""Tests for Qlib Expression Engine — tokenizer, parser, SQL compiler, Pandas compiler."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "factors"))

import numpy as np
import pandas as pd
import pytest
from expr_engine import (
    TT,
    ExprEngine,
    NodeKind,
    SQLCompiler,
    compile_sql,
    compute_pandas,
    parse,
    tokenize,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Multi-stock OHLCV data for testing (pre-sorted by date, symbol)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    symbols = ["SH.600000", "SH.600001"]
    rows = []
    for d in dates:
        for j, sym in enumerate(symbols):
            base = 100 if j == 0 else 50
            (d - dates[0]).days // 7  # rough day index within series
            o = base + int((d - dates[0]).days / 7)
            h = o + 2
            l = o - 1
            c = o + 1
            v = 1000 * (list(dates).index(d) + 1)
            a = c * v
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "amount": a,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def engine():
    return ExprEngine(table="daily_ohlcv", code_col="symbol")


# =============================================================================
# Tokenizer tests
# =============================================================================


class TestTokenizer:
    def test_simple_expr(self):
        tokens = tokenize("$close + $open")
        types = [t.tt for t in tokens if t.tt != TT.EOF]
        assert types == [TT.DOLLAR_VAR, TT.PLUS, TT.DOLLAR_VAR]

    def test_function_call(self):
        tokens = tokenize("Ref($close, 1)")
        types = [t.tt for t in tokens if t.tt != TT.EOF]
        assert types == [TT.IDENT, TT.LPAREN, TT.DOLLAR_VAR, TT.COMMA, TT.NUMBER, TT.RPAREN]

    def test_negative_number(self):
        tokens = tokenize("-1")
        types = [t.tt for t in tokens if t.tt != TT.EOF]
        assert types == [TT.MINUS, TT.NUMBER]

    def test_comparison(self):
        tokens = tokenize("$close >= 100")
        types = [t.tt for t in tokens if t.tt != TT.EOF]
        assert types == [TT.DOLLAR_VAR, TT.GTE, TT.NUMBER]

    def test_unknown_var(self):
        with pytest.raises(SyntaxError, match="Unknown"):
            tokenize("$foo")

    def test_invalid_char(self):
        with pytest.raises(SyntaxError, match="Unexpected"):
            tokenize("$close & $open")


# =============================================================================
# Parser tests
# =============================================================================


class TestParser:
    def test_column(self):
        ast = parse("$close")
        assert ast.kind == NodeKind.COLUMN
        assert ast.value == "close"

    def test_literal(self):
        ast = parse("42")
        assert ast.kind == NodeKind.LITERAL
        assert ast.value == 42

    def test_binary(self):
        ast = parse("$close + $open")
        assert ast.kind == NodeKind.BINARY
        assert ast.value == "+"
        assert ast.children[0].kind == NodeKind.COLUMN
        assert ast.children[1].kind == NodeKind.COLUMN

    def test_unary(self):
        ast = parse("-$close")
        assert ast.kind == NodeKind.UNARY
        assert ast.children[0].kind == NodeKind.COLUMN

    def test_precedence(self):
        # * binds tighter than +
        ast = parse("$close + $open * 2")
        assert ast.kind == NodeKind.BINARY  # +
        assert ast.children[1].kind == NodeKind.BINARY  # *
        assert ast.children[1].value == "*"

    def test_function(self):
        ast = parse("Ref($close, 1)")
        assert ast.kind == NodeKind.FUNC
        assert ast.value == "Ref"
        assert len(ast.children) == 2
        assert ast.children[0].kind == NodeKind.COLUMN
        assert ast.children[1].kind == NodeKind.LITERAL

    def test_nested(self):
        ast = parse("($close - $open) / $open")
        assert ast.kind == NodeKind.BINARY  # /
        assert ast.children[0].kind == NodeKind.BINARY  # -
        assert ast.children[0].children[0].kind == NodeKind.COLUMN

    def test_comparison(self):
        ast = parse("$close > $open")
        assert ast.kind == NodeKind.CMP
        assert ast.value == ">"


# =============================================================================
# SQL Compiler tests
# =============================================================================


class TestSQLCompiler:
    def test_column(self):
        sql = compile_sql("$close")
        assert "SELECT" in sql
        assert "factor" in sql

    def test_binary(self):
        sql = compile_sql("($close - $high) / $close")
        assert "close" in sql
        assert "high" in sql

    def test_ref(self):
        sql = compile_sql("Ref($close, 1)")
        assert "LAG(close, 1)" in sql
        assert "PARTITION BY" in sql
        assert "ORDER BY" in sql

    def test_mean(self):
        sql = compile_sql("Mean($close, 20)")
        assert "AVG(close)" in sql
        assert "ROWS BETWEEN 19 PRECEDING" in sql

    def test_std(self):
        sql = compile_sql("Std($close, 60)")
        assert "STDDEV_SAMP(close)" in sql
        assert "ROWS BETWEEN 59 PRECEDING" in sql

    def test_delta(self):
        sql = compile_sql("Delta($close, 1)")
        assert "LAG(close, 1)" in sql
        assert "close -" in sql
        assert "step1" in sql  # CTE decomposition

    def test_rank(self):
        sql = compile_sql("Rank($close)")
        assert "PERCENT_RANK()" in sql
        assert "PARTITION BY date" in sql
        assert "ORDER BY close" in sql

    def test_corr(self):
        sql = compile_sql("Corr($high, $volume, 20)")
        assert "CORR(high, volume)" in sql
        assert "ROWS BETWEEN 19 PRECEDING" in sql

    def test_log(self):
        sql = compile_sql("Log($close)")
        assert "LN(close)" in sql
        assert "LOG(" not in sql  # must be LN, not LOG

    def test_iif(self):
        sql = compile_sql("Iif($close > 100, 1, -1)")
        assert "CASE WHEN" in sql
        assert "THEN" in sql
        assert "ELSE" in sql

    def test_ts_rank_needs_cte(self):
        sql = compile_sql("Ts_Rank($close, 10)")
        assert "step1" in sql
        assert "PERCENT_RANK()" in sql

    def test_nested_mean_std(self):
        sql = compile_sql("Mean($close, 20) / Std($close, 60)")
        # Should have CTE steps for nested windows
        assert "AVG(close)" in sql
        assert "STDDEV_SAMP(close)" in sql

    def test_complex_alpha001_like(self):
        expr = "-1 * Corr(Rank(Delta(Log($volume), 1)), Rank(($close - $open) / $open), 6)"
        sql = compile_sql(expr)
        # Log($volume) is materialized as _col1 in a CTE, Delta/Rank also CTE'd
        assert "_col1" in sql
        assert "CORR(" in sql
        assert "PERCENT_RANK" in sql


# =============================================================================
# Pandas Compiler tests
# =============================================================================


class TestPandasCompiler:
    def test_column(self, sample_df):
        result = compute_pandas(sample_df, "$close")
        assert len(result) == len(sample_df)
        assert result.iloc[0] == sample_df["close"].iloc[0]

    def test_binary(self, sample_df):
        result = compute_pandas(sample_df, "($close - $high) / $close")
        expected = (sample_df["close"] - sample_df["high"]) / sample_df["close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_ref(self, sample_df):
        result = compute_pandas(sample_df, "Ref($close, 1)")
        # First row of each symbol should be NaN
        # With (date, symbol) sort: idx 0 = date1/sym1, idx 1 = date1/sym2 (both first)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # idx 2 = date2/sym1, Ref = close at idx 0 (date1/sym1)
        assert result.iloc[2] == sample_df["close"].iloc[0]

    def test_mean(self, sample_df):
        result = compute_pandas(sample_df, "Mean($close, 3)")
        # First 2 rows per stock NaN (need 3 periods)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_delta(self, sample_df):
        result = compute_pandas(sample_df, "Delta($close, 1)")
        # delta = close - Ref(close, 1)
        # First two entries (one per symbol) should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # idx 2 = date2/sym1: delta = close[2] - close[0]
        expected = sample_df["close"].iloc[2] - sample_df["close"].iloc[0]
        assert abs(result.iloc[2] - expected) < 1e-10

    def test_log(self, sample_df):
        result = compute_pandas(sample_df, "Log($close)")
        expected = np.log(sample_df["close"])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_rank(self, sample_df):
        result = compute_pandas(sample_df, "Rank($close)")
        # Ranks should be between 0 and 1
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 1).all()

    def test_iif(self, sample_df):
        result = compute_pandas(sample_df, "Iif($close > 100, 1, -1)")
        expected = np.where(sample_df["close"] > 100, 1, -1)
        np.testing.assert_array_equal(result.values, expected)

    def test_unary(self, sample_df):
        result = compute_pandas(sample_df, "-$close")
        expected = -sample_df["close"].astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_complex(self, sample_df):
        expr = "($close - Ref($close, 1)) / Ref($close, 1)"
        result = compute_pandas(sample_df, expr)
        assert len(result) == len(sample_df)
        # First row per stock is NaN
        assert pd.isna(result.iloc[0])


# =============================================================================
# ExprEngine tests
# =============================================================================


class TestExprEngine:
    def test_explain(self, engine):
        info = engine.explain("($close - $high) / $close")
        assert "ast" in info
        assert "sql" in info
        assert "close" in info["sql"]


# =============================================================================
# Edge cases
# =============================================================================


class TestUniverseFilter:
    """Test universe filtering for stock pool membership."""

    def test_universe_sql_generation(self):
        """Test that universe filter generates correct SQL."""
        compiler = SQLCompiler(universe="csi500", instruments_db="/path/to/quantdb.duckdb")

        # Check that ATTACH statement is generated
        ast = parse("Mean($close, 5)")
        sql = compiler.compile(ast, start="2026-01-01", end="2026-06-01")

        assert "ATTACH '/path/to/quantdb.duckdb' AS instruments_db" in sql
        assert "csi500" in sql
        assert "instruments_db.instruments" in sql

    def test_universe_raw_sql(self):
        """Test that raw SQL subquery is used as-is."""
        raw_sql = "SELECT stock_code FROM my_table WHERE pool_id = 'csi300'"
        compiler = SQLCompiler(universe=raw_sql)

        ast = parse("Mean($close, 5)")
        sql = compiler.compile(ast, start="2026-01-01", end="2026-06-01")

        # Should NOT have ATTACH statement for raw SQL
        assert "ATTACH" not in sql
        # Should use the raw SQL directly
        assert raw_sql in sql

    def test_no_universe(self):
        """Test that no universe filter means no ATTACH."""
        compiler = SQLCompiler(universe=None)
        ast = parse("Mean($close, 5)")
        sql = compiler.compile(ast, start="2026-01-01", end="2026-06-01")

        assert "ATTACH" not in sql
        assert "instruments" not in sql


class TestEdgeCases:
    def test_deeply_nested(self):
        sql = compile_sql("Mean(Std($close, 20), 60)")
        assert "AVG(" in sql
        assert "STDDEV_SAMP(" in sql
        # Inner Std materialized in CTE, Mean inlined on top
        assert "step1" in sql

    def test_negative_literal(self):
        ast = parse("-1")
        assert ast.kind == NodeKind.UNARY
        assert ast.children[0].kind == NodeKind.LITERAL
        assert ast.children[0].value == 1

    def test_power(self):
        sql = compile_sql("Power($close, 2)")
        assert "POWER(close, 2)" in sql

    def test_sqrt(self):
        sql = compile_sql("Sqrt($volume)")
        assert "SQRT(volume)" in sql

    def test_multiple_funcs(self, sample_df):
        result = compute_pandas(sample_df, "Log($close) - Log(Ref($close, 1))")
        assert len(result) == len(sample_df)
        assert pd.isna(result.iloc[0])  # Ref gives NaN for first row


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
