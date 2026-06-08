"""Layered backtest — equal-weight portfolio per quantile."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LayeredBacktest:
    """Quantile-based layered backtest.

    Divides stocks into N quantiles by factor value each period,
    then computes equal-weight portfolio return per quantile.
    """

    def __init__(self, factor: pd.Series, forward_returns: pd.Series, n_quantiles: int = 5):
        """
        Args:
            factor: MultiIndex (date, stock) factor values
            forward_returns: MultiIndex (date, stock) forward returns
            n_quantiles: number of quantile groups
        """
        self.factor = factor
        self.returns = forward_returns
        self.n_quantiles = n_quantiles

    def run(self) -> dict:
        """Compute quantile returns and long-short spread.

        Returns:
            dict with keys: quantile_returns (DataFrame), long_short,
                            top_return_mean, bottom_return_mean, ic_by_quantile
        """
        df = pd.DataFrame({"factor": self.factor, "ret": self.returns})
        df = df.dropna()

        # Assign quantile labels per date
        df["quantile"] = df.groupby(level=0)["factor"].transform(
            lambda x: pd.qcut(x, q=self.n_quantiles, labels=False, duplicates="drop") + 1
        )

        # Mean return per quantile per date
        quantile_ret = (
            df.groupby([df.index.get_level_values(0), "quantile"])["ret"].mean().unstack("quantile")
        )
        quantile_ret.columns = [f"Q{int(c)}" for c in quantile_ret.columns]

        # Long-short (Qtop - Qbottom)
        top_col = f"Q{self.n_quantiles}"
        bot_col = "Q1"
        if top_col in quantile_ret.columns and bot_col in quantile_ret.columns:
            long_short = quantile_ret[top_col] - quantile_ret[bot_col]
        else:
            long_short = pd.Series(dtype=float)

        # Cumulative returns
        cum = (1 + quantile_ret).cumprod() - 1
        cum_ls = (1 + long_short).cumprod() - 1

        # IC by quantile
        ic_by_q = {}
        for q in range(1, self.n_quantiles + 1):
            q_data = df[df["quantile"] == q]
            if q_data.empty:
                continue

            def _q_ic(g):
                try:
                    return g["factor"].corr(g["ret"], method="spearman")
                except Exception:
                    return np.nan

            ic_by_q[f"Q{q}"] = q_data.groupby(level=0).apply(_q_ic)

        ic_df = pd.DataFrame(ic_by_q) if ic_by_q else pd.DataFrame()

        return {
            "quantile_returns": quantile_ret,  # daily mean ret per quantile
            "cumulative_returns": cum,  # cumulative ret per quantile
            "long_short": long_short,  # Qtop - Qbottom
            "long_short_cum": cum_ls,  # cumulative long-short
            "top_mean": quantile_ret[top_col].mean() if top_col in quantile_ret.columns else 0,
            "bottom_mean": quantile_ret[bot_col].mean() if bot_col in quantile_ret.columns else 0,
            "spread_mean": long_short.mean(),
            "spread_std": long_short.std(),
            "spread_ir": long_short.mean() / long_short.std() if long_short.std() != 0 else 0,
            "ic_by_quantile": ic_df,
        }

    def turnover(self) -> pd.DataFrame:
        """Compute average portfolio turnover per quantile."""
        df = pd.DataFrame({"factor": self.factor})
        df = df.dropna()
        df["quantile"] = df.groupby(level=0)["factor"].transform(
            lambda x: pd.qcut(x, q=self.n_quantiles, labels=False, duplicates="drop") + 1
        )

        turnover = {}
        dates = sorted(df.index.get_level_values(0).unique())
        n_q = self.n_quantiles
        for i in range(1, len(dates) - 1):
            prev_d = df.xs(dates[i - 1], level=0)
            curr_d = df.xs(dates[i], level=0)
            prev_top = set(prev_d[prev_d["quantile"] == n_q].index)
            curr_top = set(curr_d[curr_d["quantile"] == n_q].index)
            if prev_top or curr_top:
                overlap = len(prev_top & curr_top) / len(prev_top | curr_top)
                turnover[dates[i]] = 1 - overlap
        return pd.Series(turnover)
