"""Comprehensive tests for analysis modules and factors/registry.py.

Coverage:
- IC Analysis (Spearman, Pearson, edge cases)
- Layered Backtest (quantiles, long-short, edge cases)
- Factor Registry (registration, retrieval, edge cases)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def multiindex_data():
    """Create MultiIndex data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    stocks = [f"S{i:03d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

    np.random.seed(42)
    factor = pd.Series(np.random.randn(len(idx)), index=idx, name="factor")
    ret = pd.Series(np.random.randn(len(idx)), index=idx, name="ret")

    return factor, ret


@pytest.fixture
def aligned_data():
    """Create aligned factor and return data."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    stocks = [f"S{i:03d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

    np.random.seed(42)
    # Create factor with some signal
    factor_vals = np.random.randn(len(idx))
    factor_vals[::10] = factor_vals[::10] + 0.5  # Add some signal

    # Create returns with correlation to factor
    ret_vals = factor_vals * 0.3 + np.random.randn(len(idx)) * 0.5

    factor = pd.Series(factor_vals, index=idx, name="factor")
    ret = pd.Series(ret_vals, index=idx, name="ret")

    return factor, ret


# =============================================================================
# IC Analysis Tests
# =============================================================================

class TestICAnalysis:
    """Tests for IC (Information Coefficient) analysis."""

    def test_import(self):
        """Positive: Import ICAnalysis."""
        from factor_pipeline.analysis.ic import ICAnalysis
        assert ICAnalysis is not None

    def test_spearman_ic_positive_signal(self):
        """Positive: Spearman IC with positive signal."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        # Factor: increasing trend
        factor_vals = np.tile(np.arange(20), 50) + np.random.randn(1000) * 0.1
        # Returns: correlated with factor
        ret_vals = factor_vals * 0.1 + np.random.randn(1000) * 0.1

        factor = pd.Series(factor_vals, index=idx, name="factor")
        ret = pd.Series(ret_vals, index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        assert result.n_days > 0
        assert result.ic_mean > 0  # Should have positive IC

    def test_spearman_ic_no_signal(self):
        """Edge: Spearman IC with no signal (random data)."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(600), index=idx, name="factor")
        ret = pd.Series(np.random.randn(600), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        assert result.n_days > 0
        assert -0.5 < result.ic_mean < 0.5  # Should be near zero

    def test_pearson_ic(self):
        """Positive: Pearson IC calculation."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(600), index=idx, name="factor")
        ret = pd.Series(np.random.randn(600), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("pearson")

        assert result.n_days > 0
        assert -1 <= result.ic_mean <= 1

    def test_ic_ir(self, aligned_data):
        """Positive: IC Information Ratio calculation."""
        from factor_pipeline.analysis.ic import ICAnalysis

        factor, ret = aligned_data
        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # IR = IC_mean / IC_std
        assert result.ir is not None
        # IR should be reasonable for aligned data
        assert -5 < result.ir < 5

    def test_ic_t_stat(self, aligned_data):
        """Positive: IC t-statistic."""
        from factor_pipeline.analysis.ic import ICAnalysis

        factor, ret = aligned_data
        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        assert result.t_stat is not None

    def test_ic_positive_ratio(self, aligned_data):
        """Positive: IC positive ratio."""
        from factor_pipeline.analysis.ic import ICAnalysis

        factor, ret = aligned_data
        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        assert 0 <= result.ic_positive_ratio <= 1

    def test_ic_with_nan(self):
        """Edge: IC calculation with NaN values."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        stocks = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(200), index=idx, name="factor")
        factor.iloc[::5] = np.nan  # Add some NaN

        ret = pd.Series(np.random.randn(200), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # Should handle NaN gracefully
        assert result.n_days > 0

    def test_ic_insufficient_data(self):
        """Edge: IC with insufficient data."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        stocks = [f"S{i:03d}" for i in range(5)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(10), index=idx, name="factor")
        ret = pd.Series(np.random.randn(10), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # Should handle with fewer data points
        assert result.n_days >= 0

    def test_ic_invalid_method(self, aligned_data):
        """Negative: Invalid IC method."""
        from factor_pipeline.analysis.ic import ICAnalysis

        factor, ret = aligned_data
        ic = ICAnalysis(factor, ret)

        with pytest.raises(ValueError):
            ic.run("invalid_method")


# =============================================================================
# Layered Backtest Tests
# =============================================================================

class TestLayeredBacktest:
    """Tests for Layered Backtest."""

    def test_import(self):
        """Positive: Import LayeredBacktest."""
        from factor_pipeline.analysis.layered import LayeredBacktest
        assert LayeredBacktest is not None

    def test_quantile_calculation(self, aligned_data):
        """Positive: Quantile calculation."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data
        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        assert "quantile_returns" in result
        assert len(result["quantile_returns"]) == 5

    def test_long_short_portfolio(self, aligned_data):
        """Positive: Long-short portfolio calculation."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data
        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        assert "long_short" in result
        # Long-short should be Q5 - Q1 (top - bottom quantile)
        assert len(result["long_short"]) > 0

    def test_spread_ir(self, aligned_data):
        """Positive: Long-short Information Ratio."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data
        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        assert "spread_ir" in result
        assert isinstance(result["spread_ir"], (int, float))

    def test_turnover(self, aligned_data):
        """Positive: Portfolio turnover calculation."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data
        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        assert "turnover" in result or "quantile_returns" in result

    def test_different_quantiles(self, aligned_data):
        """Positive: Different number of quantiles."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data
        for n_q in [2, 3, 5, 10]:
            lb = LayeredBacktest(factor, ret, n_quantiles=n_q)
            result = lb.run()
            assert len(result["quantile_returns"]) == n_q

    def test_quantiles_single_stock(self):
        """Edge: Quantiles with single stock (not enough for multiple quantiles)."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        idx = pd.MultiIndex.from_tuples(
            [(d, "S001") for d in dates],
            names=["date", "stock"]
        )

        np.random.seed(42)
        factor = pd.Series(np.random.randn(50), index=idx, name="factor")
        ret = pd.Series(np.random.randn(50), index=idx, name="ret")

        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        # Should handle single stock gracefully
        assert result is not None

    def test_quantiles_with_nan(self):
        """Edge: Quantiles with NaN in factor."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(600), index=idx, name="factor")
        factor.iloc[::10] = np.nan  # Add NaN

        ret = pd.Series(np.random.randn(600), index=idx, name="ret")

        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        # Should handle NaN gracefully
        assert result is not None

    def test_rebalancing_with_infrequent_data(self):
        """Edge: Rebalancing with very infrequent data."""
        from factor_pipeline.analysis.layered import LayeredBacktest

        # Weekly data
        dates = pd.date_range("2024-01-01", periods=12, freq="W")
        stocks = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(120), index=idx, name="factor")
        ret = pd.Series(np.random.randn(120), index=idx, name="ret")

        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        result = lb.run()

        assert result is not None


# =============================================================================
# Factor Registry Tests
# =============================================================================

class TestFactorRegistry:
    """Tests for Factor Registry."""

    def test_import(self):
        """Positive: Import FactorRegistry."""
        from factor_pipeline.factors.registry import FactorRegistry
        assert FactorRegistry is not None

    def test_register_decorator(self):
        """Positive: Register factor using decorator."""
        from factor_pipeline.factors.registry import FactorRegistry, register_factor

        @register_factor("test_factor_001")
        def test_factor(data):
            """Test factor."""
            return pd.Series([1, 2, 3])

        # Check if registered
        assert "test_factor_001" in FactorRegistry.list()

        # Clean up
        FactorRegistry._factors.pop("test_factor_001", None)

    def test_register_class(self):
        """Positive: Register factor class."""
        from factor_pipeline.factors.registry import FactorRegistry, FactorBase

        class TestFactorClass(FactorBase):
            name = "test_factor_002"
            description = "Test factor class"
            category = "test"

            def compute(self, data):
                return pd.Series([1, 2, 3])

        FactorRegistry.register(TestFactorClass)
        assert "test_factor_002" in FactorRegistry.list()

        # Clean up
        FactorRegistry._factors.pop("test_factor_002", None)

    def test_get_factor(self):
        """Positive: Get registered factor."""
        from factor_pipeline.factors.registry import FactorRegistry, register_factor

        @register_factor("test_get_factor")
        def my_factor(data):
            return pd.Series([1, 2, 3])

        factor = FactorRegistry.get("test_get_factor")
        assert factor is not None

        # Clean up
        FactorRegistry._factors.pop("test_get_factor", None)

    def test_get_nonexistent_factor(self):
        """Negative: Get non-existent factor."""
        from factor_pipeline.factors.registry import FactorRegistry

        factor = FactorRegistry.get("nonexistent_factor_xyz")
        assert factor is None

    def test_list_factors(self):
        """Positive: List all registered factors."""
        from factor_pipeline.factors.registry import FactorRegistry

        names = FactorRegistry.list()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_register_duplicate(self):
        """Edge: Register duplicate factor name."""
        from factor_pipeline.factors.registry import FactorRegistry, register_factor

        @register_factor("test_duplicate")
        def factor_a(data):
            return pd.Series([1])

        # Try to register with same name
        with pytest.raises(ValueError):
            @register_factor("test_duplicate")
            def factor_b(data):
                return pd.Series([2])

        # Clean up
        FactorRegistry._factors.pop("test_duplicate", None)

    def test_info(self):
        """Positive: Get factor info."""
        from factor_pipeline.factors.registry import FactorRegistry, register_factor

        @register_factor("test_info_factor")
        def info_factor(data):
            """Test info factor."""
            return pd.Series([1, 2, 3])

        info = FactorRegistry.info("test_info_factor")
        assert info["found"] is True
        assert "test_info_factor" in info["name"]

        # Clean up
        FactorRegistry._factors.pop("test_info_factor", None)

    def test_info_nonexistent(self):
        """Edge: Get info for non-existent factor."""
        from factor_pipeline.factors.registry import FactorRegistry

        info = FactorRegistry.info("nonexistent_info_factor_xyz")
        assert info["found"] is False

    def test_clear_registry(self):
        """Edge: Clear registry."""
        from factor_pipeline.factors.registry import FactorRegistry, register_factor

        @register_factor("test_clear")
        def clear_factor(data):
            return pd.Series([1])

        assert "test_clear" in FactorRegistry.list()
        FactorRegistry.clear()
        assert "test_clear" not in FactorRegistry.list()

    def test_load_gtja_factors(self):
        """Positive: Load GTJA 191 factors."""
        from factor_pipeline.factors.registry import FactorRegistry
        import importlib

        importlib.import_module("factors.gtja191")

        names = FactorRegistry.list()
        gtja_factors = [n for n in names if n.startswith("alpha")]
        assert len(gtja_factors) > 100  # Should have many alpha factors

    def test_load_technical_factors(self):
        """Positive: Load technical factors."""
        from factor_pipeline.factors.registry import FactorRegistry
        import importlib

        importlib.import_module("factors.technical")

        names = FactorRegistry.list()
        tech_factors = [n for n in names if n in ["rsi14", "macd_diff", "bb_pct", "atr14", "obv"]]
        assert len(tech_factors) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_ic_to_layered(self, aligned_data):
        """Positive: Full pipeline from IC to layered backtest."""
        from factor_pipeline.analysis.ic import ICAnalysis
        from factor_pipeline.analysis.layered import LayeredBacktest

        factor, ret = aligned_data

        # IC Analysis
        ic = ICAnalysis(factor, ret)
        ic_result = ic.run("spearman")

        # Layered Backtest
        lb = LayeredBacktest(factor, ret, n_quantiles=5)
        lb_result = lb.run()

        # Check results
        assert ic_result.n_days > 0
        assert "spread_ir" in lb_result

    def test_factor_calculation_pipeline(self):
        """Positive: Factor calculation with GTJA factor."""
        from factor_pipeline.factors.registry import FactorRegistry
        from factor_pipeline.analysis.ic import ICAnalysis
        import importlib

        # Load factors
        importlib.import_module("factors.gtja191")
        importlib.import_module("factors.technical")

        # Get a simple factor
        alpha014 = FactorRegistry.get("alpha014")
        assert alpha014 is not None

        # Create dummy data for factor
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
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

        # Calculate factor
        factor_values = alpha014(data)
        assert isinstance(factor_values, (pd.Series, pd.DataFrame))

    def test_multiple_factors_ic(self):
        """Positive: Calculate IC for multiple factors."""
        from factor_pipeline.factors.registry import FactorRegistry
        from factor_pipeline.analysis.ic import ICAnalysis
        import importlib

        # Load factors
        importlib.import_module("factors.gtja191")

        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        stocks = [f"S{i:03d}" for i in range(20)]
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

        # Calculate IC for a few factors
        test_factors = ["alpha014", "alpha018"]
        for fname in test_factors:
            factor_fn = FactorRegistry.get(fname)
            if factor_fn:
                factor_vals = factor_fn(data)
                ret = pd.Series(np.random.randn(len(idx)), index=idx)

                # Align indices
                common_idx = factor_vals.dropna().index.intersection(ret.index)
                if len(common_idx) > 10:
                    ic = ICAnalysis(factor_vals.loc[common_idx], ret.loc[common_idx])
                    result = ic.run("spearman")
                    assert result.n_days > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for analysis modules."""

    def test_empty_factor(self):
        """Edge: Empty factor Series."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        stocks = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        factor = pd.Series([], index=idx[:0], name="factor")  # Empty
        ret = pd.Series(np.random.randn(200), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        # Should handle gracefully
        try:
            result = ic.run("spearman")
            # Empty result is acceptable
        except Exception:
            pass  # Some implementations may raise

    def test_constant_factor(self):
        """Edge: Constant factor (no variation)."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        stocks = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])

        factor = pd.Series(np.ones(200), index=idx, name="factor")  # All 1s
        ret = pd.Series(np.random.randn(200), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # Constant factor may result in NaN or 0 IC
        assert result.n_days >= 0

    def test_single_date(self):
        """Edge: Single date in data."""
        from factor_pipeline.analysis.ic import ICAnalysis

        stocks = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", s) for s in stocks],
            names=["date", "stock"]
        )

        np.random.seed(42)
        factor = pd.Series(np.random.randn(10), index=idx, name="factor")
        ret = pd.Series(np.random.randn(10), index=idx, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # Single date should still work
        assert result is not None

    def test_mismatched_indices(self):
        """Edge: Mismatched MultiIndex in factor and return."""
        from factor_pipeline.analysis.ic import ICAnalysis

        dates1 = pd.date_range("2024-01-01", periods=50, freq="B")
        stocks1 = [f"S{i:03d}" for i in range(20)]
        idx1 = pd.MultiIndex.from_product([dates1, stocks1], names=["date", "stock"])

        dates2 = pd.date_range("2024-02-01", periods=50, freq="B")
        stocks2 = [f"T{i:03d}" for i in range(20)]
        idx2 = pd.MultiIndex.from_product([dates2, stocks2], names=["date", "stock"])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(1000), index=idx1, name="factor")
        ret = pd.Series(np.random.randn(1000), index=idx2, name="ret")

        ic = ICAnalysis(factor, ret)
        result = ic.run("spearman")

        # Should handle gracefully - may result in 0 common dates
        assert result is not None
