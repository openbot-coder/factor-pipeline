"""IC (Information Coefficient) analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy import stats


@dataclass
class ICResult:
    ic_series: pd.Series = field(default_factory=pd.Series)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0  # IC mean / IC std (information ratio)
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ir: float = 0.0
    ic_positive_ratio: float = 0.0
    t_stat: float = 0.0
    p_value: float = 0.0
    n_days: int = 0


class ICAnalysis:
    """Calculate IC (Pearson), Rank IC (Spearman) and summary stats."""

    def __init__(self, factor: pd.Series, forward_returns: pd.Series):
        """
        Args:
            factor: MultiIndex (date, stock) series of factor values
            forward_returns: MultiIndex (date, stock) series of forward returns
        """
        self._factor = factor
        self._fwd_ret = forward_returns

    def run(self, method: str = "spearman") -> ICResult:
        """Compute daily IC time-series and summary stats.

        Args:
            method: "spearman" (Rank IC, default) or "pearson"
        """
        dates = self._factor.index.get_level_values(0).unique()
        ic_values = {}

        for dt in dates:
            try:
                f = self._factor.loc[dt].dropna()
                r = self._fwd_ret.loc[dt].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 5:
                    continue
                if method == "spearman":
                    corr, _ = stats.spearmanr(f[common], r[common])
                else:
                    corr, _ = stats.pearsonr(f[common], r[common])
                ic_values[dt] = corr
            except Exception:
                continue

        ic_series = pd.Series(ic_values).sort_index()
        if len(ic_series) == 0:
            return ICResult()

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std != 0 else 0

        # t-test
        t_stat, p_val = stats.ttest_1samp(ic_series, 0)

        # positive ratio
        pos_ratio = (ic_series > 0).mean()

        return ICResult(
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            rank_ic_mean=ic_mean,
            rank_ic_std=ic_std,
            rank_ir=ir,
            ic_positive_ratio=pos_ratio,
            t_stat=t_stat,
            p_value=p_val,
            n_days=len(ic_series),
        )

    def summary(self) -> pd.DataFrame:
        r = self.run()
        return pd.DataFrame(
            {
                "IC Mean": [r.ic_mean],
                "IC Std": [r.ic_std],
                "IR": [r.ir],
                "IC > 0 %": [r.ic_positive_ratio],
                "t-stat": [r.t_stat],
                "p-value": [r.p_value],
                "N days": [r.n_days],
            }
        )
