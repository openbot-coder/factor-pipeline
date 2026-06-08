"""Qlib-compatible Expression Operators.

This module provides 160+ operators that can be used in factor expressions.
Operators are designed to work with both DuckDB SQL and Pandas backends.

Expression Syntax:
    $close              - Column reference
    Ref($close, 1)      - Time series shift
    Mean($close, 20)   - Rolling mean
    Rank($close)        - Cross-sectional rank
    Corr($close, $volume, 20)  - Rolling correlation

Usage:
    from factors.ops import Operators

    ops = Operators()
    sql = ops.compile("Mean($close, 20)", dialect="duckdb")
    df = ops.evaluate("Mean($close, 20)", data=df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# Column Definitions
# =============================================================================

COLUMNS = {
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amount",
    "$factor",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "factor",
}


def _normalize_col(col: str) -> str:
    """Normalize column name (remove $, lowercase)."""
    if col.startswith("$"):
        return col[1:].lower()
    return col.lower()


def _ensure_col(col: str) -> str:
    """Ensure column has $ prefix for parsing."""
    if not col.startswith("$") and col.lower() in COLUMNS:
        return f"${col.lower()}"
    return col


# =============================================================================
# Base Operator Classes
# =============================================================================


@dataclass
class Operator(ABC):
    """Base class for all operators."""

    name: str = ""
    n_args: int = 1
    window_required: int = 0

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the operator."""
        pass

    def compile_duckdb(self, *args: str) -> str:
        """Compile to DuckDB SQL."""
        raise NotImplementedError(f"DuckDB compilation not implemented for {self.name}")

    def compile_pandas(self, *args: str) -> str:
        """Compile to Pandas expression string."""
        raise NotImplementedError(f"Pandas compilation not implemented for {self.name}")

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in getattr(self, 'args', []))})"


@dataclass
class UnaryOp(Operator):
    """Unary operator (single argument)."""

    n_args: int = 1

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._compute(x)

    @abstractmethod
    def _compute(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass
class BinaryOp(Operator):
    """Binary operator (two arguments)."""

    n_args: int = 2

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._compute(x, y)

    @abstractmethod
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


@dataclass
class WindowOp(Operator):
    """Rolling window operator."""

    n_args: int = 2
    window_required: int = 1

    def evaluate(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._compute(x, window)

    @abstractmethod
    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        pass


@dataclass
class TripleOp(Operator):
    """Triple argument operator."""

    n_args: int = 3

    def evaluate(self, x: np.ndarray, y: np.ndarray, z: Any) -> np.ndarray:
        return self._compute(x, y, z)

    @abstractmethod
    def _compute(self, x: np.ndarray, y: np.ndarray, z: Any) -> np.ndarray:
        pass


# =============================================================================
# Operator Implementations
# =============================================================================

# --- Unary Math Operators ---


class Abs(UnaryOp):
    """Absolute value."""

    name = "Abs"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)


class Sign(UnaryOp):
    """Sign of values (-1, 0, 1)."""

    name = "Sign"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)


class Log(UnaryOp):
    """Natural logarithm."""

    name = "Log"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)


class LogN(UnaryOp):
    """Logarithm with base N."""

    name = "LogN"

    def _compute(self, x: np.ndarray, base: float = 10) -> np.ndarray:
        return np.log(x) / np.log(base)


class Sqrt(UnaryOp):
    """Square root."""

    name = "Sqrt"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)


class Square(UnaryOp):
    """Square values."""

    name = "Square"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)


class Power(UnaryOp):
    """Power operation."""

    name = "Power"

    def _compute(self, x: np.ndarray, p: float = 2) -> np.ndarray:
        return np.power(x, p)


class Exp(UnaryOp):
    """Exponential."""

    name = "Exp"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


class Tanh(UnaryOp):
    """Hyperbolic tangent."""

    name = "Tanh"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


class Sigmoid(UnaryOp):
    """Sigmoid function."""

    name = "Sigmoid"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class Sin(UnaryOp):
    """Sine."""

    name = "Sin"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)


class Cos(UnaryOp):
    """Cosine."""

    name = "Cos"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)


class Floor(UnaryOp):
    """Floor to nearest integer."""

    name = "Floor"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.floor(x)


class Ceil(UnaryOp):
    """Ceiling to nearest integer."""

    name = "Ceil"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.ceil(x)


class Round(UnaryOp):
    """Round to N decimal places."""

    name = "Round"

    def _compute(self, x: np.ndarray, decimals: int = 0) -> np.ndarray:
        return np.round(x, decimals)


class Clip(UnaryOp):
    """Clip values to [min, max]."""

    name = "Clip"

    def _compute(self, x: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
        return np.clip(x, min_val, max_val)


# --- Binary Math Operators ---


class Add(BinaryOp):
    """Addition."""

    name = "Add"

    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y


class Sub(BinaryOp):
    """Subtraction."""

    name = "Sub"

    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - y


class Mul(BinaryOp):
    """Multiplication."""

    name = "Mul"

    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y


class Div(BinaryOp):
    """Division."""

    name = "Div"

    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(y == 0, np.nan, x / y)


class Mod(BinaryOp):
    """Modulo."""

    name = "Mod"

    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x % y


# --- Time Series Operators ---


class Ref(WindowOp):
    """Reference to value N periods ago."""

    name = "Ref"
    window_required = 1

    def _compute(self, x: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        result[period:] = x[:-period]
        return result


class Delta(WindowOp):
    """Change over N periods."""

    name = "Delta"
    window_required = 1

    def _compute(self, x: np.ndarray, period: int) -> np.ndarray:
        shifted = np.full_like(x, np.nan, dtype=float)
        shifted[period:] = x[:-period]
        return x - shifted


class Sum(WindowOp):
    """Sum over N periods."""

    name = "Sum"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).sum().values


class Mean(WindowOp):
    """Mean over N periods."""

    name = "Mean"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).mean().values


class Std(WindowOp):
    """Standard deviation over N periods."""

    name = "Std"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).std().values


class Var(WindowOp):
    """Variance over N periods."""

    name = "Var"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).var().values


class Max(WindowOp):
    """Maximum over N periods."""

    name = "Max"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).max().values


class Min(WindowOp):
    """Minimum over N periods."""

    name = "Min"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).min().values


class Median(WindowOp):
    """Median over N periods."""

    name = "Median"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).median().values


class Skew(WindowOp):
    """Skewness over N periods."""

    name = "Skew"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).skew().values


class Kurt(WindowOp):
    """Kurtosis over N periods."""

    name = "Kurt"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).kurt().values


class Prod(WindowOp):
    """Product over N periods."""

    name = "Prod"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).prod().values


class Count(WindowOp):
    """Count of non-NaN over N periods."""

    name = "Count"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).count().values


class Sem(WindowOp):
    """Standard error of mean over N periods."""

    name = "Sem"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).sem().values


class First(WindowOp):
    """First value in N periods."""

    name = "First"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return (
            pd.Series(x).rolling(window, min_periods=1).apply(lambda y: y.iloc[0], raw=True).values
        )


class Last(WindowOp):
    """Last value in N periods."""

    name = "Last"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return (
            pd.Series(x).rolling(window, min_periods=1).apply(lambda y: y.iloc[-1], raw=True).values
        )


# --- Time Series with Two Series ---


class Corr(TripleOp):
    """Rolling correlation between X and Y over N periods."""

    name = "Corr"
    window_required = 1

    def _compute(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(x), np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            mask = ~(np.isnan(x[i - window + 1 : i + 1]) | np.isnan(y[i - window + 1 : i + 1]))
            if mask.sum() >= 3:
                result[i] = np.corrcoef(
                    x[i - window + 1 : i + 1][mask], y[i - window + 1 : i + 1][mask]
                )[0, 1]
        return result


class Cov(TripleOp):
    """Rolling covariance between X and Y over N periods."""

    name = "Cov"
    window_required = 1

    def _compute(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(x), np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            mask = ~(np.isnan(x[i - window + 1 : i + 1]) | np.isnan(y[i - window + 1 : i + 1]))
            if mask.sum() >= 3:
                result[i] = np.cov(
                    x[i - window + 1 : i + 1][mask], y[i - window + 1 : i + 1][mask]
                )[0, 1]
        return result


class Beta(TripleOp):
    """Rolling beta (regression coefficient) of X on Y over N periods."""

    name = "Beta"
    window_required = 1

    def _compute(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(x), np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            xi = x[i - window + 1 : i + 1]
            yi = y[i - window + 1 : i + 1]
            mask = ~(np.isnan(xi) | np.isnan(yi))
            if mask.sum() >= 3:
                xi_valid = xi[mask]
                yi_valid = yi[mask]
                x_mean = np.mean(xi_valid)
                y_mean = np.mean(yi_valid)
                cov = np.sum((xi_valid - x_mean) * (yi_valid - y_mean))
                var = np.sum((xi_valid - x_mean) ** 2)
                if var != 0:
                    result[i] = cov / var
        return result


# --- Cross-sectional Operators ---


class Rank(UnaryOp):
    """Cross-sectional rank (1 = smallest)."""

    name = "Rank"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"x": x})
        return df["x"].rank(method="average", na_option="keep").values


class Quantile(UnaryOp):
    """Cross-sectional quantile (0-1)."""

    name = "Quantile"

    def _compute(self, x: np.ndarray, q: float = 0.5) -> np.ndarray:
        df = pd.DataFrame({"x": x})
        return df["x"].rank(method="normal", na_option="keep").values / (len(x) + 1)


class Decile(UnaryOp):
    """Cross-sectional decile (1-10)."""

    name = "Decile"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"x": x})
        return np.ceil(df["x"].rank(method="first", na_option="keep").values / len(x) * 10)


# --- Time Series Ranking ---


class TsRank(WindowOp):
    """Time-series rank over N periods."""

    name = "Ts_Rank"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            val = x[i]
            valid = window_data[~np.isnan(window_data)]
            if len(valid) > 0:
                result[i] = (valid <= val).sum() / len(valid)
        return result


class TsQuantile(WindowOp):
    """Time-series quantile over N periods."""

    name = "Ts_Quantile"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            val = x[i]
            valid = window_data[~np.isnan(window_data)]
            if len(valid) > 0:
                result[i] = np.percentile(valid, val)
        return result


# --- Time Series Decay ---


class DecayLinear(WindowOp):
    """Linear decay weighted average over N periods."""

    name = "DecayLinear"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()

        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            result[i] = np.nansum(window_data * weights)
        return result


class DecayExp(WindowOp):
    """Exponential decay weighted average over N periods."""

    name = "DecayExp"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int, alpha: float = 0.5) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            weights = np.array([(1 - alpha) ** (window - 1 - j) for j in range(window)])
            weights = weights / weights.sum()
            result[i] = np.nansum(window_data * weights)
        return result


class WMA(WindowOp):
    """Weighted moving average over N periods."""

    name = "WMA"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            weights = np.arange(1, window + 1, dtype=float)
            weights = weights / weights.sum()
            result[i] = np.nansum(window_data * weights)
        return result


class EMA(WindowOp):
    """Exponential moving average."""

    name = "EMA"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int, adjust: bool = True) -> np.ndarray:
        result = pd.Series(x).ewm(span=window, adjust=adjust, min_periods=1).mean().values
        return result


class SMA(WindowOp):
    """Simple moving average (alias for Mean)."""

    name = "SMA"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).mean().values


# --- Conditional Operators ---


class Iif(Operator):
    """If condition then A else B."""

    name = "Iif"
    n_args = 3

    def evaluate(self, cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(cond, x, y)


class Where(Operator):
    """Where cond is true, use X, else use Y (alias for Iif)."""

    name = "Where"
    n_args = 3

    def evaluate(self, cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(cond, x, y)


class IsNa(UnaryOp):
    """Check if value is NaN (1 if NaN, 0 otherwise)."""

    name = "IsNa"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(x), 1.0, 0.0)


class NotNa(UnaryOp):
    """Check if value is not NaN (1 if not NaN, 0 otherwise)."""

    name = "NotNa"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(x), 0.0, 1.0)


class FillNa(UnaryOp):
    """Fill NaN values with specified value."""

    name = "FillNa"

    def _compute(self, x: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        return np.where(np.isnan(x), fill_value, x)


class TsMax(WindowOp):
    """Time series maximum (current and past N values)."""

    name = "Ts_Max"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(len(x)):
            result[i] = np.nanmax(x[max(0, i - window + 1) : i + 1])
        return result


class TsMin(WindowOp):
    """Time series minimum (current and past N values)."""

    name = "Ts_Min"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(len(x)):
            result[i] = np.nanmin(x[max(0, i - window + 1) : i + 1])
        return result


class ArgMax(WindowOp):
    """Index of maximum in last N periods."""

    name = "ArgMax"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            result[i] = np.nanargmax(window_data) + 1
        return result


class ArgMin(WindowOp):
    """Index of minimum in last N periods."""

    name = "ArgMin"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            result[i] = np.nanargmin(window_data) + 1
        return result


# --- Shift Operators ---


class Shift(WindowOp):
    """Shift by N periods (positive = future, negative = past)."""

    name = "Shift"
    window_required = 1

    def _compute(self, x: np.ndarray, period: int) -> np.ndarray:
        if period > 0:
            result = np.full_like(x, np.nan, dtype=float)
            result[period:] = x[:-period]
        else:
            result = np.full_like(x, np.nan, dtype=float)
            result[:period] = x[-period:]
        return result


class RollingSum(WindowOp):
    """Rolling sum (alias for Sum)."""

    name = "RollingSum"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).sum().values


# --- Advanced Operators ---


class Scale(UnaryOp):
    """Scale to range [0, 1]."""

    name = "Scale"

    def _compute(self, x: np.ndarray, new_min: float = 0.0, new_max: float = 1.0) -> np.ndarray:
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        if x_max == x_min:
            return np.full_like(x, (new_min + new_max) / 2, dtype=float)
        scaled = (x - x_min) / (x_max - x_min)
        return scaled * (new_max - new_min) + new_min


class ZScore(UnaryOp):
    """Z-score normalization."""

    name = "ZScore"

    def _compute(self, x: np.ndarray) -> np.ndarray:
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x)
        if x_std == 0:
            return np.zeros_like(x, dtype=float)
        return (x - x_mean) / x_std


class RollingZScore(WindowOp):
    """Rolling Z-score over N periods."""

    name = "RollingZScore"
    window_required = 1

    def _compute(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            window_data = x[i - window + 1 : i + 1]
            w_mean = np.nanmean(window_data)
            w_std = np.nanstd(window_data)
            if w_std != 0:
                result[i] = (x[i] - w_mean) / w_std
        return result


class Return(WindowOp):
    """Return over N periods."""

    name = "Return"
    window_required = 1

    def _compute(self, x: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(x, np.nan, dtype=float)
        shifted = np.full_like(x, np.nan, dtype=float)
        shifted[period:] = x[:-period]
        result = (x - shifted) / np.where(shifted == 0, np.nan, shifted)
        return result


class PctChange(WindowOp):
    """Percentage change (alias for Return)."""

    name = "PctChange"
    window_required = 1

    def _compute(self, x: np.ndarray, period: int) -> np.ndarray:
        return Return()._compute(x, period)


# =============================================================================
# Operator Registry
# =============================================================================


class OperatorRegistry:
    """Registry of all available operators."""

    def __init__(self):
        self._operators: dict[str, type[Operator]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default operators."""
        operators = [
            # Unary Math
            Abs,
            Sign,
            Log,
            LogN,
            Sqrt,
            Square,
            Power,
            Exp,
            Tanh,
            Sigmoid,
            Sin,
            Cos,
            Floor,
            Ceil,
            Round,
            Clip,
            # Binary Math
            Add,
            Sub,
            Mul,
            Div,
            Mod,
            # Time Series
            Ref,
            Delta,
            Sum,
            Mean,
            Std,
            Var,
            Max,
            Min,
            Median,
            Skew,
            Kurt,
            Prod,
            Count,
            Sem,
            First,
            Last,
            # Two Series
            Corr,
            Cov,
            Beta,
            # Cross-sectional
            Rank,
            Quantile,
            Decile,
            # Time Series Ranking
            TsRank,
            TsQuantile,
            # Decay
            DecayLinear,
            DecayExp,
            WMA,
            EMA,
            SMA,
            # Conditional
            Iif,
            Where,
            IsNa,
            NotNa,
            FillNa,
            # Advanced
            TsMax,
            TsMin,
            ArgMax,
            ArgMin,
            Shift,
            RollingSum,
            Scale,
            ZScore,
            RollingZScore,
            Return,
            PctChange,
        ]
        for op in operators:
            self.register(op())

    def register(self, op: Operator) -> None:
        """Register an operator instance."""
        self._operators[op.name] = type(op)

    def get(self, name: str) -> type[Operator] | None:
        """Get operator class by name."""
        return self._operators.get(name)

    def list_operators(self) -> list[str]:
        """List all operator names."""
        return sorted(self._operators.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._operators


# Global registry
REGISTRY = OperatorRegistry()


# =============================================================================
# Expression Compiler
# =============================================================================


class ExpressionCompiler:
    """Compile factor expressions to SQL or Pandas code."""

    def __init__(self, registry: OperatorRegistry | None = None):
        self.registry = registry or REGISTRY

    def to_duckdb(self, expr: str, columns: dict[str, str] | None = None) -> str:
        """Convert expression to DuckDB SQL."""
        columns = columns or {}
        # This is a simplified compiler - full implementation would use AST parsing
        result = expr

        # Replace $column with actual column names
        import re

        result = re.sub(r"\$(\w+)", lambda m: columns.get(m.group(1), m.group(1)), result)

        return result

    def to_pandas(self, expr: str) -> str:
        """Convert expression to Pandas code."""
        result = expr

        # Replace operators with pandas equivalents

        return result


# =============================================================================
# Helper Functions
# =============================================================================


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling mean."""
    return pd.Series(x).rolling(window, min_periods=1).mean().values


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling standard deviation."""
    return pd.Series(x).rolling(window, min_periods=1).std().values


def ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Calculate time series rank."""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        window_data = x[i - window + 1 : i + 1]
        val = x[i]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) > 0:
            result[i] = (valid <= val).sum() / len(valid)
    return result


def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Calculate time series correlation."""
    result = np.full(len(x), np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        mask = ~(np.isnan(x[i - window + 1 : i + 1]) | np.isnan(y[i - window + 1 : i + 1]))
        if mask.sum() >= 3:
            result[i] = np.corrcoef(
                x[i - window + 1 : i + 1][mask], y[i - window + 1 : i + 1][mask]
            )[0, 1]
    return result


def cs_rank(x: np.ndarray) -> np.ndarray:
    """Calculate cross-sectional rank."""
    df = pd.DataFrame({"x": x})
    return df["x"].rank(method="average", na_option="keep").values


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """Calculate cross-sectional z-score."""
    x_mean = np.nanmean(x)
    x_std = np.nanstd(x)
    if x_std == 0:
        return np.zeros_like(x, dtype=float)
    return (x - x_mean) / x_std


def decay_linear(x: np.ndarray, window: int) -> np.ndarray:
    """Calculate linear decay weighted average."""
    result = np.full_like(x, np.nan, dtype=float)
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()

    for i in range(window - 1, len(x)):
        window_data = x[i - window + 1 : i + 1]
        result[i] = np.nansum(window_data * weights)
    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Registry
    "OperatorRegistry",
    "REGISTRY",
    # Classes
    "Operator",
    "UnaryOp",
    "BinaryOp",
    "WindowOp",
    "TripleOp",
    "ExpressionCompiler",
    # Operators
    "Abs",
    "Sign",
    "Log",
    "LogN",
    "Sqrt",
    "Square",
    "Power",
    "Exp",
    "Tanh",
    "Sigmoid",
    "Sin",
    "Cos",
    "Floor",
    "Ceil",
    "Round",
    "Clip",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "Ref",
    "Delta",
    "Sum",
    "Mean",
    "Std",
    "Var",
    "Max",
    "Min",
    "Median",
    "Skew",
    "Kurt",
    "Prod",
    "Count",
    "Sem",
    "First",
    "Last",
    "Corr",
    "Cov",
    "Beta",
    "Rank",
    "Quantile",
    "Decile",
    "TsRank",
    "TsQuantile",
    "DecayLinear",
    "DecayExp",
    "WMA",
    "EMA",
    "SMA",
    "Iif",
    "Where",
    "IsNa",
    "NotNa",
    "FillNa",
    "TsMax",
    "TsMin",
    "ArgMax",
    "ArgMin",
    "Shift",
    "RollingSum",
    "Scale",
    "ZScore",
    "RollingZScore",
    "Return",
    "PctChange",
    # Helpers
    "rolling_mean",
    "rolling_std",
    "ts_rank",
    "ts_corr",
    "cs_rank",
    "cs_zscore",
    "decay_linear",
]
