"""Comprehensive tests for factors/ops.py.

Coverage:
- All time-series operators (Ref, Mean, Sum, Std, etc.)
- Cross-sectional operators (Rank, Quantile)
- Math operators (Log, Abs, Sqrt, etc.)
- Conditional operators (Iif, Where)
- Edge cases: NaN, Inf, zero division, empty arrays
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.ops import (
    # Operators
    Abs, Sign, Log, LogN, Sqrt, Square, Power, Exp, Tanh, Sigmoid,
    Sin, Cos, Floor, Ceil, Round, Clip,
    Add, Sub, Mul, Div, Mod,
    Ref, Delta, Sum, Mean, Std, Var, Max, Min, Median, Skew, Kurt,
    Prod, Count, Sem, First, Last,
    Corr, Cov, Beta,
    Rank, Quantile, Decile,
    TsRank, TsQuantile,
    DecayLinear, DecayExp, WMA, EMA, SMA,
    Iif, Where, IsNa, NotNa, FillNa,
    TsMax, TsMin, ArgMax, ArgMin, Shift, RollingSum,
    Scale, ZScore, RollingZScore, Return, PctChange,
    # Helper functions
    rolling_mean, rolling_std, ts_rank, ts_corr, cs_rank, cs_zscore, decay_linear,
    # Registry
    OperatorRegistry, REGISTRY,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_series():
    """Sample 1D array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def series_with_nan():
    """Series with NaN values."""
    return np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0])


@pytest.fixture
def sample_2d():
    """Sample 2D array for cross-sectional operations."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def two_series():
    """Two series for binary operations."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    return x, y


# =============================================================================
# Registry Tests
# =============================================================================

class TestRegistry:
    """Tests for operator registry."""

    def test_registry_has_operators(self):
        """Positive: Registry has operators."""
        assert len(REGISTRY.list_operators()) > 50

    def test_registry_get(self):
        """Positive: Get operator by name."""
        op_class = REGISTRY.get("Mean")
        assert op_class is not None

    def test_registry_get_nonexistent(self):
        """Negative: Get non-existent operator."""
        op_class = REGISTRY.get("NonexistentOp")
        assert op_class is None

    def test_registry_contains(self):
        """Positive: Check operator existence."""
        assert "Mean" in REGISTRY
        assert "Ref" in REGISTRY
        assert "Nonexistent" not in REGISTRY

    def test_registry_list_operators(self):
        """Positive: List all operators."""
        operators = REGISTRY.list_operators()
        assert "Mean" in operators
        assert "Sum" in operators
        assert "Rank" in operators


# =============================================================================
# Unary Math Operator Tests
# =============================================================================

class TestUnaryMathOps:
    """Tests for unary math operators."""

    def test_abs_positive(self):
        """Positive: Abs of positive numbers."""
        op = Abs()
        result = op.evaluate(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_abs_negative(self):
        """Positive: Abs of negative numbers."""
        op = Abs()
        result = op.evaluate(np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_abs_mixed(self):
        """Edge: Abs of mixed positive/negative."""
        op = Abs()
        result = op.evaluate(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 1.0])

    def test_sign_positive(self):
        """Positive: Sign of positive numbers."""
        op = Sign()
        result = op.evaluate(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_sign_negative(self):
        """Positive: Sign of negative numbers."""
        op = Sign()
        result = op.evaluate(np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_array_equal(result, [-1.0, -1.0, -1.0])

    def test_sign_zero(self):
        """Edge: Sign of zero."""
        op = Sign()
        result = op.evaluate(np.array([0.0]))
        np.testing.assert_array_equal(result, [0.0])

    def test_log_positive(self):
        """Positive: Log of positive numbers."""
        op = Log()
        result = op.evaluate(np.array([1.0, 2.718281828, 10.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, np.log(10.0)])

    def test_log_zero(self):
        """Edge: Log of zero (should be -inf)."""
        op = Log()
        result = op.evaluate(np.array([0.0]))
        assert np.isinf(result[0]) and result[0] < 0

    def test_sqrt_positive(self):
        """Positive: Sqrt of positive numbers."""
        op = Sqrt()
        result = op.evaluate(np.array([0.0, 1.0, 4.0, 9.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0, 3.0])

    def test_sqrt_negative(self):
        """Edge: Sqrt of negative (should be nan)."""
        op = Sqrt()
        result = op.evaluate(np.array([-1.0]))
        assert np.isnan(result[0])

    def test_square(self):
        """Positive: Square of numbers."""
        op = Square()
        result = op.evaluate(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(result, [4.0, 1.0, 0.0, 1.0, 4.0])

    def test_power(self):
        """Positive: Power operation."""
        op = Power()
        result = op.evaluate(np.array([2.0, 3.0, 4.0]), p=2)
        np.testing.assert_array_almost_equal(result, [4.0, 9.0, 16.0])

    def test_exp(self):
        """Positive: Exponential."""
        op = Exp()
        result = op.evaluate(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_almost_equal(result, [1.0, np.e, np.e**2])

    def test_tanh(self):
        """Positive: Tanh."""
        op = Tanh()
        result = op.evaluate(np.array([0.0, 1.0, -1.0]))
        np.testing.assert_array_almost_equal(result, [0.0, np.tanh(1), np.tanh(-1)])

    def test_sigmoid(self):
        """Positive: Sigmoid."""
        op = Sigmoid()
        result = op.evaluate(np.array([0.0, 1.0, -1.0]))
        np.testing.assert_array_almost_equal(result, [0.5, 0.731, 0.269], decimal=3)

    def test_floor(self):
        """Positive: Floor."""
        op = Floor()
        result = op.evaluate(np.array([1.1, 1.9, -1.1, -1.9]))
        np.testing.assert_array_equal(result, [1.0, 1.0, -2.0, -2.0])

    def test_ceil(self):
        """Positive: Ceil."""
        op = Ceil()
        result = op.evaluate(np.array([1.1, 1.9, -1.1, -1.9]))
        np.testing.assert_array_equal(result, [2.0, 2.0, -1.0, -1.0])

    def test_round(self):
        """Positive: Round."""
        op = Round()
        result = op.evaluate(np.array([1.4, 1.5, 1.6, -1.4, -1.5]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 2.0, -1.0, -2.0])

    def test_round_decimals(self):
        """Positive: Round with decimals."""
        op = Round()
        result = op.evaluate(np.array([1.555]), decimals=2)
        np.testing.assert_array_almost_equal(result, [1.56])

    def test_clip(self):
        """Positive: Clip values."""
        op = Clip()
        result = op.evaluate(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), min_val=1.0, max_val=3.0)
        np.testing.assert_array_equal(result, [1.0, 1.0, 2.0, 3.0, 3.0])

    def test_clip_no_bounds(self):
        """Edge: Clip with no bounds."""
        op = Clip()
        result = op.evaluate(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# =============================================================================
# Binary Math Operator Tests
# =============================================================================

class TestBinaryMathOps:
    """Tests for binary math operators."""

    def test_add(self):
        """Positive: Add two arrays."""
        op = Add()
        result = op.evaluate(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_sub(self):
        """Positive: Subtract arrays."""
        op = Sub()
        result = op.evaluate(np.array([5.0, 6.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [2.0, 2.0])

    def test_mul(self):
        """Positive: Multiply arrays."""
        op = Mul()
        result = op.evaluate(np.array([2.0, 3.0]), np.array([4.0, 5.0]))
        np.testing.assert_array_equal(result, [8.0, 15.0])

    def test_div(self):
        """Positive: Divide arrays."""
        op = Div()
        result = op.evaluate(np.array([6.0, 10.0]), np.array([2.0, 5.0]))
        np.testing.assert_array_almost_equal(result, [3.0, 2.0])

    def test_div_by_zero(self):
        """Edge: Division by zero returns nan."""
        op = Div()
        result = op.evaluate(np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_mod(self):
        """Positive: Modulo."""
        op = Mod()
        result = op.evaluate(np.array([7.0, 8.0]), np.array([3.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0])


# =============================================================================
# Time Series Operator Tests
# =============================================================================

class TestTimeSeriesOps:
    """Tests for time series operators."""

    def test_ref_basic(self, sample_series):
        """Positive: Ref shifts by period."""
        op = Ref()
        result = op.evaluate(sample_series, period=1)
        expected = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(np.isnan(result[:1]), [True])
        np.testing.assert_array_almost_equal(result[1:], sample_series[:-1])

    def test_ref_period_2(self, sample_series):
        """Positive: Ref with period 2."""
        op = Ref()
        result = op.evaluate(sample_series, period=2)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_almost_equal(result[2:], sample_series[:-2])

    def test_delta_basic(self, sample_series):
        """Positive: Delta computes change."""
        op = Delta()
        result = op.evaluate(sample_series, period=1)
        expected = np.array([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result[1:], expected[1:])

    def test_delta_period_5(self, sample_series):
        """Positive: Delta with period 5."""
        op = Delta()
        result = op.evaluate(sample_series, period=5)
        # 6.0 - 1.0 = 5.0 for first valid
        assert np.isnan(result[:5]).all()
        np.testing.assert_array_almost_equal(result[5:], [5.0, 5.0, 5.0, 5.0, 5.0])

    def test_sum_basic(self, sample_series):
        """Positive: Rolling sum."""
        op = Sum()
        result = op.evaluate(sample_series, window=3)
        # First two should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # 1+2+3=6, 2+3+4=9, ...
        np.testing.assert_array_almost_equal(result[2], 6.0)

    def test_mean_basic(self, sample_series):
        """Positive: Rolling mean."""
        op = Mean()
        result = op.evaluate(sample_series, window=3)
        # (1+2+3)/3=2, (2+3+4)/3=3, ...
        np.testing.assert_array_almost_equal(result[2], 2.0)
        np.testing.assert_array_almost_equal(result[3], 3.0)

    def test_std_basic(self, sample_series):
        """Positive: Rolling std."""
        op = Std()
        result = op.evaluate(sample_series, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Std of [1,2,3] = 1.0
        np.testing.assert_array_almost_equal(result[2], 1.0)

    def test_var_basic(self, sample_series):
        """Positive: Rolling variance."""
        op = Var()
        result = op.evaluate(sample_series, window=3)
        assert np.isnan(result[0])
        # Var of [1,2,3] = 1.0
        np.testing.assert_array_almost_equal(result[2], 1.0)

    def test_max_basic(self, sample_series):
        """Positive: Rolling max."""
        op = Max()
        result = op.evaluate(sample_series, window=3)
        np.testing.assert_array_almost_equal(result[2], 3.0)
        np.testing.assert_array_almost_equal(result[3], 4.0)

    def test_min_basic(self, sample_series):
        """Positive: Rolling min."""
        op = Min()
        result = op.evaluate(sample_series, window=3)
        np.testing.assert_array_almost_equal(result[2], 1.0)
        np.testing.assert_array_almost_equal(result[3], 2.0)

    def test_median_basic(self, sample_series):
        """Positive: Rolling median."""
        op = Median()
        result = op.evaluate(sample_series, window=3)
        # Median of [1,2,3] = 2
        np.testing.assert_array_almost_equal(result[2], 2.0)

    def test_skew_basic(self):
        """Positive: Rolling skewness."""
        op = Skew()
        # Skew of constant series = nan
        result = op.evaluate(np.array([1.0, 1.0, 1.0]), window=3)
        assert np.isnan(result[2])

    def test_kurt_basic(self):
        """Positive: Rolling kurtosis."""
        op = Kurt()
        # Kurt of constant series = nan
        result = op.evaluate(np.array([1.0, 1.0, 1.0]), window=3)
        assert np.isnan(result[2])

    def test_prod_basic(self):
        """Positive: Rolling product."""
        op = Prod()
        result = op.evaluate(np.array([1.0, 2.0, 3.0, 4.0]), window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # 1*2*3=6, 2*3*4=24
        np.testing.assert_array_almost_equal(result[2], 6.0)
        np.testing.assert_array_almost_equal(result[3], 24.0)

    def test_count_basic(self):
        """Positive: Rolling count of non-NaN."""
        op = Count()
        result = op.evaluate(np.array([1.0, 2.0, np.nan, 4.0]), window=3)
        assert np.isnan(result[0])
        assert result[1] == 2.0
        assert result[2] == 2.0  # 2.0, nan, 3.0
        assert result[3] == 2.0  # nan, 3.0, 4.0

    def test_sem_basic(self):
        """Positive: Rolling standard error of mean."""
        op = Sem()
        result = op.evaluate(np.array([1.0, 2.0, 3.0]), window=3)
        # SEM = std / sqrt(n) = 1.0 / sqrt(3)
        np.testing.assert_array_almost_equal(result[2], 1.0 / np.sqrt(3))

    def test_first_basic(self):
        """Positive: First value in window."""
        op = First()
        result = op.evaluate(np.array([1.0, 2.0, 3.0, 4.0]), window=3)
        assert np.isnan(result[0])
        np.testing.assert_array_almost_equal(result[2], 1.0)
        np.testing.assert_array_almost_equal(result[3], 2.0)

    def test_last_basic(self):
        """Positive: Last value in window."""
        op = Last()
        result = op.evaluate(np.array([1.0, 2.0, 3.0, 4.0]), window=3)
        assert np.isnan(result[0])
        np.testing.assert_array_almost_equal(result[2], 3.0)
        np.testing.assert_array_almost_equal(result[3], 4.0)


# =============================================================================
# Two Series Operator Tests
# =============================================================================

class TestTwoSeriesOps:
    """Tests for operators with two time series."""

    def test_corr_basic(self, two_series):
        """Positive: Rolling correlation."""
        x, y = two_series  # [1,2,3,4,5] and [5,4,3,2,1]
        op = Corr()
        result = op.evaluate(x, y, window=5)
        # Perfect negative correlation = -1
        np.testing.assert_array_almost_equal(result[4], -1.0)

    def test_corr_insufficient_data(self, two_series):
        """Edge: Correlation with insufficient data."""
        x, y = two_series
        op = Corr()
        result = op.evaluate(x, y, window=5)
        assert np.isnan(result[:4]).all()

    def test_cov_basic(self, two_series):
        """Positive: Rolling covariance."""
        x, y = two_series
        op = Cov()
        result = op.evaluate(x, y, window=5)
        # Covariance of [1,2,3,4,5] and [5,4,3,2,1] = -2.5
        np.testing.assert_array_almost_equal(result[4], -2.5)

    def test_beta_basic(self, two_series):
        """Positive: Rolling beta (regression coefficient)."""
        x, y = two_series
        op = Beta()
        result = op.evaluate(x, y, window=5)
        # Beta of x on y with perfect negative correlation
        # Cov(x,y) / Var(y) = -2.5 / 2.5 = -1
        np.testing.assert_array_almost_equal(result[4], -1.0)


# =============================================================================
# Cross-sectional Operator Tests
# =============================================================================

class TestCrossSectionalOps:
    """Tests for cross-sectional operators."""

    def test_rank_basic(self, sample_2d):
        """Positive: Cross-sectional rank (1 = smallest)."""
        op = Rank()
        result = op.evaluate(sample_2d)
        # Ranks: 1 is smallest, 5 is largest
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_rank_reversed(self):
        """Positive: Rank of reversed array."""
        op = Rank()
        result = op.evaluate(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        np.testing.assert_array_equal(result, [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_rank_with_nan(self):
        """Edge: Rank with NaN values."""
        op = Rank()
        result = op.evaluate(np.array([np.nan, 1.0, 2.0, np.nan, 3.0]))
        # NaN should be ranked as well
        assert not np.isnan(result).all()

    def test_quantile_basic(self, sample_2d):
        """Positive: Cross-sectional quantile (0-1)."""
        op = Quantile()
        result = op.evaluate(sample_2d, q=0.5)
        # Should return percentile rank
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_decile_basic(self, sample_2d):
        """Positive: Cross-sectional decile (1-10)."""
        op = Decile()
        result = op.evaluate(sample_2d)
        # All values are unique, so deciles should be 1, 2, 3, 4, 5
        assert (result >= 1).all()
        assert (result <= 10).all()


# =============================================================================
# Time Series Ranking Tests
# =============================================================================

class TestTimeSeriesRankingOps:
    """Tests for time series ranking operators."""

    def test_ts_rank_basic(self):
        """Positive: Time series rank."""
        # [1, 2, 3, 4, 5] - at position 4 (0-indexed), rank should be high
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        op = TsRank()
        result = op.evaluate(data, window=5)
        # At last position, rank = 1.0 (largest in window)
        np.testing.assert_array_almost_equal(result[4], 1.0)

    def test_ts_rank_increasing(self):
        """Positive: TsRank of strictly increasing series."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        op = TsRank()
        result = op.evaluate(data, window=5)
        # Increasing: position 0->1, 1->2, etc.
        np.testing.assert_array_almost_equal(result[2], 1.0/3)
        np.testing.assert_array_almost_equal(result[4], 1.0)

    def test_ts_rank_decreasing(self):
        """Positive: TsRank of strictly decreasing series."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        op = TsRank()
        result = op.evaluate(data, window=5)
        # Decreasing: position 0->1, 1->2, etc.
        np.testing.assert_array_almost_equal(result[4], 0.0)


# =============================================================================
# Decay Operator Tests
# =============================================================================

class TestDecayOps:
    """Tests for decay operators."""

    def test_decay_linear_basic(self, sample_series):
        """Positive: Linear decay weighted average."""
        op = DecayLinear()
        result = op.evaluate(sample_series, window=3)
        # Weights: [1,2,3]/6, [2,3,4]/9, [3,4,5]/12
        # Index 2: (1*1 + 2*2 + 3*3)/6 = 14/6
        np.testing.assert_array_almost_equal(result[2], 14.0/6)

    def test_decay_exp_basic(self, sample_series):
        """Positive: Exponential decay weighted average."""
        op = DecayExp()
        result = op.evaluate(sample_series, window=3, alpha=0.5)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_wma_basic(self, sample_series):
        """Positive: Weighted moving average."""
        op = WMA()
        result = op.evaluate(sample_series, window=3)
        # Weights: [1,2,3]/6
        # Index 2: (1*1 + 2*2 + 3*3)/6 = 14/6
        np.testing.assert_array_almost_equal(result[2], 14.0/6)

    def test_ema_basic(self, sample_series):
        """Positive: Exponential moving average."""
        op = EMA()
        result = op.evaluate(sample_series, window=3)
        assert not np.isnan(result[0])

    def test_sma_basic(self, sample_series):
        """Positive: Simple moving average (alias for Mean)."""
        op = SMA()
        result = op.evaluate(sample_series, window=3)
        op_mean = Mean()
        result_mean = op_mean.evaluate(sample_series, window=3)
        np.testing.assert_array_almost_equal(result, result_mean)


# =============================================================================
# Conditional Operator Tests
# =============================================================================

class TestConditionalOps:
    """Tests for conditional operators."""

    def test_iif_true(self):
        """Positive: Iif with true condition."""
        op = Iif()
        cond = np.array([True, True, False])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        result = op.evaluate(cond, x, y)
        np.testing.assert_array_equal(result, [1.0, 2.0, 30.0])

    def test_iif_false(self):
        """Positive: Iif with false condition."""
        op = Iif()
        cond = np.array([False, False, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        result = op.evaluate(cond, x, y)
        np.testing.assert_array_equal(result, [10.0, 20.0, 3.0])

    def test_where_basic(self):
        """Positive: Where (alias for Iif)."""
        op = Where()
        cond = np.array([True, False])
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])
        result = op.evaluate(cond, x, y)
        np.testing.assert_array_equal(result, [1.0, 20.0])

    def test_isna(self, series_with_nan):
        """Positive: IsNa check."""
        op = IsNa()
        result = op.evaluate(series_with_nan)
        expected = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_notna(self, series_with_nan):
        """Positive: NotNa check."""
        op = NotNa()
        result = op.evaluate(series_with_nan)
        expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_fillna(self, series_with_nan):
        """Positive: Fill NaN with value."""
        op = FillNa()
        result = op.evaluate(series_with_nan, fill_value=0.0)
        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 10.0])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# Advanced Operator Tests
# =============================================================================

class TestAdvancedOps:
    """Tests for advanced operators."""

    def test_ts_max(self, sample_series):
        """Positive: Time series max."""
        op = TsMax()
        result = op.evaluate(sample_series, window=3)
        # Index 0: max of [1] = 1
        # Index 2: max of [1,2,3] = 3
        # Index 4: max of [3,4,5] = 5
        np.testing.assert_array_almost_equal(result[2], 3.0)

    def test_ts_min(self, sample_series):
        """Positive: Time series min."""
        op = TsMin()
        result = op.evaluate(sample_series, window=3)
        np.testing.assert_array_almost_equal(result[2], 1.0)

    def test_argmax(self, sample_series):
        """Positive: Argmax (position of max)."""
        op = ArgMax()
        result = op.evaluate(sample_series, window=3)
        # Index 2: [1,2,3], argmax = 3 (position 3)
        np.testing.assert_array_equal(result[2], 3.0)

    def test_argmin(self, sample_series):
        """Positive: Argmin (position of min)."""
        op = ArgMin()
        result = op.evaluate(sample_series, window=3)
        # Index 2: [1,2,3], argmin = 1 (position 1)
        np.testing.assert_array_equal(result[2], 1.0)

    def test_shift_positive(self, sample_series):
        """Positive: Shift by positive period."""
        op = Shift()
        result = op.evaluate(sample_series, period=1)
        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1], sample_series[0])

    def test_shift_negative(self, sample_series):
        """Edge: Shift by negative period (future)."""
        op = Shift()
        result = op.evaluate(sample_series, period=-1)
        np.testing.assert_array_equal(result[:-1], sample_series[1:])

    def test_rolling_sum_alias(self, sample_series):
        """Positive: RollingSum alias for Sum."""
        op = RollingSum()
        result_sum = Sum().evaluate(sample_series, window=3)
        result_rolling = op.evaluate(sample_series, window=3)
        np.testing.assert_array_almost_equal(result_sum, result_rolling)

    def test_scale_basic(self, sample_series):
        """Positive: Scale to [0, 1]."""
        op = Scale()
        result = op.evaluate(sample_series)
        assert result.min() >= 0
        assert result.max() <= 1
        np.testing.assert_array_almost_equal(result.min(), 0.0)
        np.testing.assert_array_almost_equal(result.max(), 1.0)

    def test_scale_custom_range(self):
        """Positive: Scale to custom range."""
        op = Scale()
        result = op.evaluate(np.array([0.0, 1.0, 2.0]), new_min=-1.0, new_max=1.0)
        assert result.min() >= -1
        assert result.max() <= 1

    def test_zscore_basic(self, sample_series):
        """Positive: Z-score normalization."""
        op = ZScore()
        result = op.evaluate(sample_series)
        assert abs(result.mean()) < 1e-10  # Mean ≈ 0
        assert abs(result.std() - 1.0) < 1e-10  # Std ≈ 1

    def test_zscore_constant(self):
        """Edge: Z-score of constant series."""
        op = ZScore()
        result = op.evaluate(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_rolling_zscore(self, sample_series):
        """Positive: Rolling Z-score."""
        op = RollingZScore()
        result = op.evaluate(sample_series, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Z-score at index 2 should be 0 (mean of [1,2,3])
        np.testing.assert_array_almost_equal(result[2], 0.0)

    def test_return_basic(self, sample_series):
        """Positive: Return calculation."""
        op = Return()
        result = op.evaluate(sample_series, period=1)
        # (2-1)/1 = 1, (3-2)/2 = 0.5, ...
        np.testing.assert_array_almost_equal(result[1], 1.0)
        np.testing.assert_array_almost_equal(result[2], 0.5)

    def test_pct_change_alias(self, sample_series):
        """Positive: PctChange alias for Return."""
        op = PctChange()
        result_pct = op.evaluate(sample_series, period=1)
        result_ret = Return().evaluate(sample_series, period=1)
        np.testing.assert_array_almost_equal(result_pct, result_ret)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelpers:
    """Tests for helper functions."""

    def test_rolling_mean(self, sample_series):
        """Positive: rolling_mean helper."""
        result = rolling_mean(sample_series, window=3)
        np.testing.assert_array_almost_equal(result[2], 2.0)

    def test_rolling_std(self, sample_series):
        """Positive: rolling_std helper."""
        result = rolling_std(sample_series, window=3)
        np.testing.assert_array_almost_equal(result[2], 1.0)

    def test_ts_rank_helper(self, sample_series):
        """Positive: ts_rank helper."""
        result = ts_rank(sample_series, window=5)
        np.testing.assert_array_almost_equal(result[4], 1.0)

    def test_ts_corr_helper(self, two_series):
        """Positive: ts_corr helper."""
        x, y = two_series
        result = ts_corr(x, y, window=5)
        np.testing.assert_array_almost_equal(result[4], -1.0)

    def test_cs_rank_helper(self, sample_2d):
        """Positive: cs_rank helper."""
        result = cs_rank(sample_2d)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_cs_zscore_helper(self, sample_2d):
        """Positive: cs_zscore helper."""
        result = cs_zscore(sample_2d)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_decay_linear_helper(self,