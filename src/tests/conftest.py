"""Pytest configuration and shared fixtures for all tests.

This module provides common fixtures and configuration for the test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_data: marks tests that require external data")


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    symbols = [f"{i:06d}" for i in range(1, 21)]

    records = []
    for symbol in symbols:
        price = 100.0
        for d in dates:
            price = price * (1 + np.random.randn() * 0.01)
            records.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": np.random.randint(1e6, 1e8),
                    "amount": price * np.random.randint(1e6, 1e8),
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def multiindex_data():
    """Create MultiIndex (date, stock) data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    stocks = [f"S{i:03d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

    np.random.seed(42)
    data = {
        "open": pd.Series(np.random.randn(len(idx)) * 10 + 100, index=idx),
        "high": pd.Series(np.random.randn(len(idx)) * 10 + 102, index=idx),
        "low": pd.Series(np.random.randn(len(idx)) * 10 + 98, index=idx),
        "close": pd.Series(np.random.randn(len(idx)) * 10 + 100, index=idx),
        "volume": pd.Series(np.abs(np.random.randn(len(idx))) * 1e6, index=idx),
        "amount": pd.Series(np.abs(np.random.randn(len(idx))) * 1e8, index=idx),
    }
    return data


@pytest.fixture
def multiindex_factor_ret():
    """Create MultiIndex factor and return series."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    stocks = [f"S{i:03d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

    np.random.seed(42)
    factor = pd.Series(np.random.randn(len(idx)), index=idx, name="factor")
    ret = pd.Series(np.random.randn(len(idx)), index=idx, name="ret")
    return factor, ret


@pytest.fixture
def aligned_factor_ret():
    """Create aligned factor and return with some correlation."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    stocks = [f"S{i:03d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

    np.random.seed(42)
    # Factor with some signal
    factor_vals = np.random.randn(len(idx))
    factor_vals[::10] = factor_vals[::10] + 0.5

    # Returns correlated with factor
    ret_vals = factor_vals * 0.3 + np.random.randn(len(idx)) * 0.5

    factor = pd.Series(factor_vals, index=idx, name="factor")
    ret = pd.Series(ret_vals, index=idx, name="ret")
    return factor, ret


# =============================================================================
# Array Fixtures
# =============================================================================


@pytest.fixture
def simple_array():
    """Simple 1D array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def array_with_nan():
    """Array with NaN values."""
    return np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0])


@pytest.fixture
def array_with_inf():
    """Array with Inf values."""
    return np.array([1.0, 2.0, np.inf, 4.0, 5.0, -np.inf, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def two_aligned_arrays():
    """Two aligned arrays for binary operations."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    return x, y


@pytest.fixture
def cross_sectional_array():
    """Array for cross-sectional operations."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# =============================================================================
# Date/Time Fixtures
# =============================================================================


@pytest.fixture
def trading_dates():
    """List of trading dates (business days)."""
    return pd.date_range("2024-01-01", periods=100, freq="B").tolist()


@pytest.fixture
def sample_dates():
    """Sample date range."""
    return pd.date_range("2024-01-01", periods=50, freq="B")


# =============================================================================
# Module Fixtures
# =============================================================================


@pytest.fixture
def storage_module():
    """Import and return storage module."""
    from factor_pipeline.data.storage import DuckDBStorage

    return DuckDBStorage


@pytest.fixture
def ops_module():
    """Import and return ops module."""
    from factor_pipeline.factors import ops as ops_module

    return ops_module


@pytest.fixture
def registry_module():
    """Import and return registry module."""
    from factor_pipeline.factors.registry import FactorRegistry

    return FactorRegistry


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_factor_registry():
    """Reset factor registry before each test."""
    from factor_pipeline.factors.registry import _REGISTRY

    # Store original state
    original = dict(_REGISTRY)
    yield
    # Restore original state
    _REGISTRY.clear()
    _REGISTRY.update(original)


# =============================================================================
# Parametrized Fixtures
# =============================================================================


@pytest.fixture(params=[3, 5, 10])
def quantile_values(request):
    """Parametrized quantile values."""
    return request.param


@pytest.fixture(params=["spearman", "pearson"])
def ic_method(request):
    """Parametrized IC methods."""
    return request.param


@pytest.fixture(params=[1, 2, 5, 10])
def window_sizes(request):
    """Parametrized window sizes."""
    return request.param
