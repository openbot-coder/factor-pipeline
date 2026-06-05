"""Technical indicator factors — MA cross, RSI, MACD, BB etc."""

from __future__ import annotations
import pandas as pd
import ta
from factors.registry import register_factor


@register_factor
def rsi14(data: dict) -> pd.Series:
    c = data["close"].iloc[:, 0].copy()
    # ta needs 1D array per stock — apply per group
    def _rsi(s):
        return ta.momentum.RSIIndicator(s.fillna(0), window=14).rsi()
    result = c.groupby(level=1).transform(_rsi)
    return result.rename("rsi14")


@register_factor
def macd_diff(data: dict) -> pd.Series:
    c = data["close"].iloc[:, 0].copy()
    def _macd(s):
        macd = ta.trend.MACD(s.fillna(0), window_slow=26, window_fast=12, window_sign=9)
        return macd.macd_diff()
    result = c.groupby(level=1).transform(_macd)
    return result.rename("macd_diff")


@register_factor
def bb_pct(data: dict) -> pd.Series:
    c = data["close"].iloc[:, 0].copy()
    def _bb(s):
        bb = ta.volatility.BollingerBands(s.fillna(0), window=20, window_dev=2)
        mid = bb.bollinger_mavg()
        std = bb.bollinger_std()
        return (s - mid) / (2 * std + 1e-9)
    result = c.groupby(level=1).transform(_bb)
    return result.rename("bb_pct")


@register_factor
def atr14(data: dict) -> pd.Series:
    h = data["high"].stack()
    l = data["low"].stack()
    c = data["close"].stack()
    atr = ta.volatility.AverageTrueRange(h.fillna(0).unstack(), l.fillna(0).unstack(), c.fillna(0).unstack(), window=14)
    return atr.average_true_range().stack().rename("atr14")


@register_factor
def obv(data: dict) -> pd.Series:
    c = data["close"].stack()
    v = data["volume"].stack()
    obv = ta.volume.OnBalanceVolumeIndicator(c.fillna(0).unstack(), v.fillna(0).unstack())
    return obv.on_balance_volume().stack().rename("obv")
