"""Smoke tests for factor-pipeline."""

import pandas as pd
import numpy as np
import pytest


def test_data_loader():
    """Test DataLoader with demo data."""
    from data.loader import DataLoader
    loader = DataLoader("duckdb", "data/ohlcv.duckdb")
    data = loader.load(start="2021-01-01", end="2021-12-31")
    assert "close" in data
    assert "open" in data
    assert "high" in data
    assert "low" in data
    assert "volume" in data
    assert isinstance(data["close"].index, pd.MultiIndex)


def test_factor_registry():
    """Test factor registry."""
    from factors.registry import FactorRegistry
    import importlib
    importlib.import_module("factors.gtja191")
    importlib.import_module("factors.technical")
    names = FactorRegistry.list()
    assert len(names) > 10
    assert "alpha001" in names
    assert "alpha014" in names
    assert "rsi14" in names


def test_alpha014():
    """Test alpha014 (5-day momentum) computation."""
    from factors.registry import FactorRegistry
    from data.loader import DataLoader
    import importlib
    importlib.import_module("factors.gtja191")
    loader = DataLoader("duckdb", "data/ohlcv.duckdb")
    data = loader.load(start="2021-01-01", end="2021-06-30")
    alpha_fn = FactorRegistry.get("alpha014")
    result = alpha_fn(data)
    assert isinstance(result, pd.Series)
    assert result.name == "factor"
    assert not result.isna().all()


def test_ic_analysis():
    """Test IC analysis."""
    from analysis.ic import ICAnalysis
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    stocks = [f"S{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])
    factor = pd.Series(np.random.randn(len(idx)), index=idx, name="factor")
    ret = pd.Series(np.random.randn(len(idx)), index=idx, name="ret")
    ic = ICAnalysis(factor, ret)
    result = ic.run("spearman")
    assert result.n_days > 0
    assert -1 <= result.ic_mean <= 1


def test_layered_backtest():
    """Test layered backtest."""
    from analysis.layered import LayeredBacktest
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    stocks = [f"S{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])
    factor = pd.Series(np.random.randn(len(idx)), index=idx, name="factor")
    ret = pd.Series(np.random.randn(len(idx)), index=idx, name="ret")
    lb = LayeredBacktest(factor, ret, n_quantiles=5)
    result = lb.run()
    assert "quantile_returns" in result
    assert "long_short" in result
    assert "spread_ir" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
