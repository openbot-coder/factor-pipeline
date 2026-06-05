"""Data pre-processing — alignment, NaN handling, normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Align, clean, and prepare data for factor computation."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self._raw = data

    def align(self) -> dict[str, pd.DataFrame]:
        """Ensure all DataFrames share the same MultiIndex (date, stock)."""
        common_idx = None
        for name, df in self._raw.items():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        aligned = {}
        for name, df in self._raw.items():
            aligned[name] = df.loc[common_idx].sort_index()
        return aligned

    def dropna(self, how: str = "any", pct: float = 0.5) -> dict[str, pd.DataFrame]:
        """Drop stocks with too many NaN values."""
        aligned = self.align()
        cols_to_check = ["close", "volume"]
        valid_stocks = set()
        for col in cols_to_check:
            if col not in aligned:
                continue
            df = aligned[col]
            notna_pct = 1 - df.groupby(level=1).apply(lambda x: x.isna().mean())
            valid = notna_pct[notna_pct >= pct].index
            valid_stocks = valid_stocks.intersection(valid) if valid_stocks else set(valid)
        return {k: v[v.index.get_level_values(1).isin(valid_stocks)] for k, v in aligned.items()}

    def winsorize(self, data: dict[str, pd.DataFrame], limits: float = 0.01) -> dict[str, pd.DataFrame]:
        """Winsorize extreme values per cross-section."""
        result = {}
        for name, df in data.items():
            def _clip(x):
                lo, hi = x.quantile(limits), x.quantile(1 - limits)
                return x.clip(lo, hi)
            result[name] = df.groupby(level=0).transform(_clip) if not df.empty else df
        return result

    def neutralise(self, data: pd.DataFrame, groupby: pd.Series = None) -> pd.DataFrame:
        """Cross-sectional neutralisation (e.g. market/sector)."""
        if groupby is None:
            # demean per date
            return data - data.groupby(level=0).mean()
        return data

    def standardise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardisation per date."""
        mu = data.groupby(level=0).mean()
        sd = data.groupby(level=0).std()
        return (data - mu) / sd

    @staticmethod
    def prepare(factor_df: pd.DataFrame, prices: pd.DataFrame, quantiles: int = 5, periods=(1, 5, 10)):
        """Prepare factor+price data for Alphalens-style analysis.

        Args:
            factor_df: MultiIndex (date, stock), column "factor"
            prices: MultiIndex (date, stock), column "close"
            quantiles: number of quantiles
            periods: forward return periods (days)

        Returns:
            merged DataFrame ready for IC / layered analysis
        """
        from analysis.report import FactorReport
        fr = FactorReport(factor_df, prices)
        return fr.alphalens_input(quantiles=quantiles, periods=periods)
