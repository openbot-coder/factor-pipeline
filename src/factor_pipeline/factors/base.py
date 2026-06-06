"""Factor abstract base class and result dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass
class FactorResult:
    """Raw output of a factor calculation."""

    values: pd.DataFrame  # MultiIndex (date, stock), single column "factor"
    name: str = ""
    dependencies: list[str] = field(default_factory=list)
    max_window: int = 1          # trailing window needed
    description: str = ""
    computed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    @property
    def ic_type(self) -> str:
        return "spearman"  # default

    def validate(self) -> bool:
        if self.values.empty:
            return False
        if not isinstance(self.values.index, pd.MultiIndex):
            raise ValueError("Factor values must be MultiIndex (date, stock)")
        if self.values.isna().all().all():
            raise ValueError(f"Factor {self.name}: all NaN, check dependencies")
        return True


class FactorABC(ABC):
    """Abstract base for stateful factors that need a data dict at init.

    Simple stateless factors can skip this and just be plain functions —
    register them with @register_factor instead.
    """

    name: str = ""
    dependencies: list[str] = []   # required data keys
    max_window: int = 1             # trailing days needed
    description: str = ""

    def __init__(self, data: dict[str, pd.DataFrame]):
        """Initialize with a dict of named DataFrames.

        Expected keys (all indexed by date × stock MultiIndex or date/index):
            open, high, low, close, volume
        Optional:
            vwap, amount, returns, kdj_j, adx, adxr, bbi, ...
        """
        self._data = data
        self._validate_dependencies()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _col(self, key: str) -> pd.DataFrame:
        """Return a column from stored data, ensuring float dtype."""
        if key not in self._data:
            raise KeyError(f"Factor {self.name} requires data key '{key}' but got: {list(self._data.keys())}")
        df = self._data[key]
        return df.astype(float)

    def _validate_dependencies(self):
        missing = [d for d in self.dependencies if d not in self._data]
        if missing:
            raise KeyError(f"Factor {self.name} missing data keys: {missing}")

    # ------------------------------------------------------------------
    # helpers matching GTJA formula language
    # ------------------------------------------------------------------

    def DELAY(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).shift(n)

    def DELTA(self, s: pd.Series, n: int) -> pd.Series:
        return s - self.DELAY(s, n)

    def SUM(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).sum().droplevel(0)

    def MEAN(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).mean().droplevel(0)

    def STD(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).std().droplevel(0)

    def TSMIN(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).min().droplevel(0)

    def TSMAX(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).max().droplevel(0)

    def TSRANK(self, s: pd.Series, n: int) -> pd.Series:
        def _tsrank(x):
            window = x.shape[0]
            if window < n:
                return np.nan
            return (x < x.iloc[-1]).sum() / window
        return s.groupby(level=1).rolling(n, min_periods=n).apply(_tsrank, raw=False).droplevel(0)

    def CORR(self, s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return s1.groupby(level=1).rolling(n, min_periods=n).corr(s2.groupby(level=1).shift(0)).droplevel(0)

    def RANK(self, s: pd.Series) -> pd.Series:
        return s.groupby(level=0).rank(pct=True, ascending=True)

    def ABS(self, s: pd.Series) -> pd.Series:
        return s.abs()

    def LOG(self, s: pd.Series) -> pd.Series:
        return np.log(s)

    def SIGN(self, s: pd.Series) -> pd.Series:
        return np.sign(s)

    def COVIANCE(self, s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return s1.groupby(level=1).rolling(n, min_periods=n).cov(s2.groupby(level=1).shift(0)).droplevel(0)

    def SMA(self, s: pd.Series, n: int, m: int) -> pd.Series:
        """Exponential moving average: SMA(A,n,m) = (A*m + prev_SMA*(n-m)) / n"""
        return s.ewm(alpha=m/n, adjust=False).mean()

    def WMA(self, s: pd.Series, n: int) -> pd.Series:
        w = np.arange(1, n + 1)
        w = w / w.sum()
        return s.groupby(level=1).rolling(n, min_periods=n).apply(
            lambda x: (x * w[: len(x)]).sum(), raw=False
        ).droplevel(0)

    def DECAYLINEAR(self, s: pd.Series, d: int) -> pd.Series:
        """Linear decay weighted average over trailing d days."""
        w = np.arange(1, d + 1)
        w = w / w.sum()
        return s.groupby(level=1).rolling(d, min_periods=d).apply(
            lambda x: (x * w[: len(x)]).sum(), raw=False
        ).droplevel(0)

    def REGBETA(self, s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        """Rolling OLS beta of s1 on s2 over trailing n days."""
        def _beta(x):
            if len(x) < n:
                return np.nan
            x1, x2 = s1.iloc[-n:].values, s2.iloc[-n:].values
            cov = np.cov(x1, x2)[0, 1]
            var = np.var(x2)
            return cov / var if var != 0 else 0.0
        return s1.groupby(level=1).rolling(n, min_periods=n).apply(_beta, raw=False).droplevel(0)

    def HIGHDAY(self, s: pd.Series, n: int) -> pd.Series:
        """Days since high within trailing n."""
        def _hd(x):
            if len(x) < n:
                return np.nan
            return n - 1 - int(np.argmax(x.values[::-1]))
        return s.groupby(level=1).rolling(n, min_periods=n).apply(_hd, raw=False).droplevel(0)

    def LOWDAY(self, s: pd.Series, n: int) -> pd.Series:
        def _ld(x):
            if len(x) < n:
                return np.nan
            return n - 1 - int(np.argmin(x.values[::-1]))
        return s.groupby(level=1).rolling(n, min_periods=n).apply(_ld, raw=False).droplevel(0)

    def PROD(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(level=1).rolling(n, min_periods=n).apply(np.prod, raw=False).droplevel(0)

    def SEQUENCE(self, n: int) -> pd.Series:
        """Returns a simple 1..n sequence (used in REGBETA)."""
        return pd.Series(np.arange(1, n + 1))

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """Return DataFrame (date × stock) with column 'factor'."""
        ...
