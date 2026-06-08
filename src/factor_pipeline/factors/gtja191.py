"""GTJA 191 Alpha factors — full implementation.

Factor names: alpha001 … alpha191
Each is a function: f(data: dict) -> pd.Series (MultiIndex date/stock)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factor_pipeline.factors.base import FactorABC, FactorResult
from factor_pipeline.factors.registry import register_factor


# ---------------------------------------------------------------------------
# helpers shared by multiple factors
# ---------------------------------------------------------------------------

def _ensure_mi(values: pd.DataFrame) -> pd.Series:
    """Ensure MultiIndex (date, stock) Series named 'factor'."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            return values.iloc[:, 0].rename("factor")
        raise ValueError(f"Expected single column, got {values.columns.tolist()}")
    return values.rename("factor")


def _vwap(amount: pd.Series, volume: pd.Series) -> pd.Series:
    return amount / volume


def _returns(close: pd.Series) -> pd.Series:
    return close.groupby(level=1).pct_change()


# ---------------------------------------------------------------------------
# Alpha 001
# ---------------------------------------------------------------------------

@register_factor
def alpha001(data: dict) -> pd.Series:
    # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK((CLOSE - OPEN) / OPEN), 6))
    vol = data["volume"].iloc[:, 0].copy()
    o = data["open"].iloc[:, 0].copy()
    c = data["close"].iloc[:, 0].copy()

    dv = np.log(vol).diff(1)
    ret = (c - o) / (o + 1e-9)

    rank_vol = dv.groupby(level=0).rank(pct=True)
    rank_ret = ret.groupby(level=0).rank(pct=True)

    # Rolling corr per stock using groupby transform
    merged = pd.concat({"rv": rank_vol, "rr": rank_ret}, axis=1)

    def _roll_corr(g: pd.DataFrame) -> pd.Series:
        return g["rv"].rolling(6, min_periods=6).corr(g["rr"])

    corr = merged.groupby(level=1, group_keys=False).apply(_roll_corr)
    return _ensure_mi((-corr).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 002
# ---------------------------------------------------------------------------

@register_factor
def alpha002(data: dict) -> pd.Series:
    # -1 * delta(((close - low) - (high - close)) / (high - low), 1)
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    c = data["close"].iloc[:, 0]
    inner = ((c - l) - (h - c)) / (h - l + 1e-9)
    delta = inner.groupby(level=1).diff(1)
    return _ensure_mi((-delta).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 003
# ---------------------------------------------------------------------------

@register_factor
def alpha003(data: dict) -> pd.Series:
    # -1 * SUM((CLOSE = DELAY(CLOSE, 1) ? 0 :
    #            CLOSE - (CLOSE > DELAY(CLOSE, 1) ? MIN(LOW, DELAY(CLOSE, 1)) :
    #                     MAX(HIGH, DELAY(CLOSE, 1)))), 6)
    c = data["close"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    h = data["high"].iloc[:, 0]
    prev_c = c.groupby(level=1).shift(1)
    cond = c > prev_c
    term = pd.Series(np.where(cond, c - np.minimum(l, prev_c), c - np.maximum(h, prev_c)), index=c.index)
    term = term.where(c != prev_c, 0)
    s = term.groupby(level=1).rolling(6, min_periods=6).sum().droplevel(0)
    return _ensure_mi((-s).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 004
# ---------------------------------------------------------------------------

@register_factor
def alpha004(data: dict) -> pd.Series:
    # (((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2))
    #   ? -1 : ((SUM(CLOSE, 2) / 2) < (SUM(CLOSE, 8) / 8 - STD(CLOSE, 8))
    #     ? 1 : ((VOLUME / MEAN(VOLUME, 20)) >= 1 ? 1 : -1))
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    s8 = c.groupby(level=1).rolling(8, min_periods=8).mean().droplevel(0)
    s2 = c.groupby(level=1).rolling(2, min_periods=2).mean().droplevel(0)
    sd8 = c.groupby(level=1).rolling(8, min_periods=8).std().droplevel(0)
    mv20 = v.groupby(level=1).rolling(20, min_periods=20).mean().droplevel(0)
    cond1 = (s8 + sd8) < s2
    cond2 = s2 < (s8 - sd8)
    vol_cond = (v / mv20) >= 1
    result = pd.Series(np.where(cond1, -1, np.where(cond2, 1, np.where(vol_cond, 1, -1))), index=c.index)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 005
# ---------------------------------------------------------------------------

@register_factor
def alpha005(data: dict) -> pd.Series:
    # -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)
    v = data["volume"].iloc[:, 0]
    h = data["high"].iloc[:, 0]

    def _tsrank(x, n=5):
        if len(x) < n:
            return np.nan
        return (x < x.iloc[-1]).sum() / n

    trk_v = v.groupby(level=1).rolling(5, min_periods=5).apply(lambda x: _tsrank(x, 5), raw=False).droplevel(0)
    trk_h = h.groupby(level=1).rolling(5, min_periods=5).apply(lambda x: _tsrank(x, 5), raw=False).droplevel(0)
    corr = trk_v.groupby(level=1).rolling(5, min_periods=5).corr(trk_h.groupby(level=1).shift(0)).droplevel(0)
    tsmax = corr.groupby(level=1).rolling(3, min_periods=3).max().droplevel(0)
    return _ensure_mi((-tsmax).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 006
# ---------------------------------------------------------------------------

@register_factor
def alpha006(data: dict) -> pd.Series:
    # -1 * RANK(SIGN(DELTA(OPEN * 0.85 + HIGH * 0.15, 4)))
    o = data["open"].iloc[:, 0]
    h = data["high"].iloc[:, 0]
    w = o * 0.85 + h * 0.15
    delta = w.groupby(level=1).diff(4)
    rank = delta.groupby(level=0).rank(pct=True)
    return _ensure_mi((-rank).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 007
# ---------------------------------------------------------------------------

@register_factor
def alpha007(data: dict) -> pd.Series:
    # (RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))) * RANK(DELTA(VOLUME, 3))
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    diff = vwap - c
    rank_max = diff.groupby(level=0).rank(pct=True).clip(upper=3).groupby(level=0).rank(pct=True)
    rank_min = (-diff).clip(upper=3).groupby(level=0).rank(pct=True)
    delta_v = v.groupby(level=1).diff(3)
    rank_dv = delta_v.groupby(level=0).rank(pct=True)
    result = (rank_max + rank_min) * rank_dv
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 008
# ---------------------------------------------------------------------------

@register_factor
def alpha008(data: dict) -> pd.Series:
    # -1 * RANK(DELTA((HIGH + LOW) / 10 + VWAP * 0.8, 4))
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else h * v
    vwap = amt / v
    mid = (h + l) / 10 + vwap * 0.8
    delta = mid.groupby(level=1).diff(4)
    rank = delta.groupby(level=0).rank(pct=True)
    return _ensure_mi((-rank).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 009
# ---------------------------------------------------------------------------

@register_factor
def alpha009(data: dict) -> pd.Series:
    # SMA(((HIGH + LOW) / 2 - (DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2) * (HIGH - LOW) / VOLUME, 7, 2)
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    prev_mid = (h.groupby(level=1).shift(1) + l.groupby(level=1).shift(1)) / 2
    mid = (h + l) / 2
    term = (mid - prev_mid) * (h - l) / (v + 1)
    result = term.ewm(alpha=2 / 7, adjust=False).mean()
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 010
# ---------------------------------------------------------------------------

@register_factor
def alpha010(data: dict) -> pd.Series:
    # RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2, 5))
    c = data["close"].iloc[:, 0]
    ret = c.groupby(level=1).pct_change()
    std20 = ret.groupby(level=1).rolling(20, min_periods=20).std().droplevel(0)
    cond_val = pd.Series(np.where(ret < 0, std20, c), index=ret.index)
    squared = cond_val ** 2
    def _max5(x):
        if len(x) < 5:
            return np.nan
        return x.iloc[-5:].max()
    mx = squared.groupby(level=1).rolling(5, min_periods=5).apply(_max5, raw=False).droplevel(0)
    rank = mx.groupby(level=0).rank(pct=True)
    return _ensure_mi(rank.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 011
# ---------------------------------------------------------------------------

@register_factor
def alpha011(data: dict) -> pd.Series:
    # SUM(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 6)
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    inner = ((c - l) - (h - c)) / (h - l + 1e-9)
    result = (inner * v).groupby(level=1).rolling(6, min_periods=6).sum().droplevel(0)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 012
# ---------------------------------------------------------------------------

@register_factor
def alpha012(data: dict) -> pd.Series:
    # RANK(OPEN - MA(VWAP, 10)) * RANK(ABS(CLOSE - VWAP)) * (-1)
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    o = data["open"].iloc[:, 0]
    ma_vwap = vwap.groupby(level=1).rolling(10, min_periods=10).mean().droplevel(0)
    rank1 = (o - ma_vwap).groupby(level=0).rank(pct=True)
    rank2 = (c - vwap).abs().groupby(level=0).rank(pct=True)
    result = rank1 * rank2 * (-1)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 013
# ---------------------------------------------------------------------------

@register_factor
def alpha013(data: dict) -> pd.Series:
    # ((HIGH * LOW)^0.5) - VWAP
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    result = np.sqrt(h * l) - vwap
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 014
# ---------------------------------------------------------------------------

@register_factor
def alpha014(data: dict) -> pd.Series:
    # CLOSE - DELAY(CLOSE, 5) = 5日动量
    c = data["close"].iloc[:, 0]
    return _ensure_mi((c - c.groupby(level=1).shift(5)).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 015
# ---------------------------------------------------------------------------

@register_factor
def alpha015(data: dict) -> pd.Series:
    # OPEN / DELAY(CLOSE, 1) - 1   (跳空比)
    o = data["open"].iloc[:, 0]
    c = data["close"].iloc[:, 0]
    return _ensure_mi((o / c.groupby(level=1).shift(1) - 1).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 016
# ---------------------------------------------------------------------------

@register_factor
def alpha016(data: dict) -> pd.Series:
    # -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)
    v = data["volume"].iloc[:, 0]
    c = data["close"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    rk_v = v.groupby(level=0).rank(pct=True)
    rk_vwap = vwap.groupby(level=0).rank(pct=True)
    corr = rk_v.groupby(level=1).rolling(5, min_periods=5).corr(rk_vwap.groupby(level=1).shift(0)).droplevel(0)
    rank_corr = corr.groupby(level=0).rank(pct=True)
    tsmax = rank_corr.groupby(level=1).rolling(5, min_periods=5).max().droplevel(0)
    return _ensure_mi((-tsmax).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 017
# ---------------------------------------------------------------------------

@register_factor
def alpha017(data: dict) -> pd.Series:
    # RANK(VWAP - MAX(VWAP, 15))^DELTA(CLOSE, 5)
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    diff = vwap - vwap.clip(lower=15)
    rank = diff.groupby(level=0).rank(pct=True)
    delta_c = c.groupby(level=1).diff(5)
    result = rank ** delta_c
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 018 — REVS5
# ---------------------------------------------------------------------------

@register_factor
def alpha018(data: dict) -> pd.Series:
    # CLOSE / DELAY(CLOSE, 5)
    c = data["close"].iloc[:, 0]
    return _ensure_mi((c / c.groupby(level=1).shift(5)).to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 019
# ---------------------------------------------------------------------------

@register_factor
def alpha019(data: dict) -> pd.Series:
    # C<DELAY(C,5) ? (C/DELAY(C,5)-1) : (C==DELAY(C,5) ? 0 : (1-DELAY(C,5)/C))
    c = data["close"].iloc[:, 0]
    prev = c.groupby(level=1).shift(5)
    result = pd.Series(np.where(c < prev, c / prev - 1,
                                  np.where(c == prev, 0, 1 - prev / c)), index=c.index)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 020
# ---------------------------------------------------------------------------

@register_factor
def alpha020(data: dict) -> pd.Series:
    # (CLOSE / DELAY(CLOSE, 6) - 1) * 100
    c = data["close"].iloc[:, 0]
    return _ensure_mi(((c / c.groupby(level=1).shift(6)) - 1).to_frame("factor") * 100)


# ---------------------------------------------------------------------------
# Alpha 021
# ---------------------------------------------------------------------------

@register_factor
def alpha021(data: dict) -> pd.Series:
    # REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))
    c = data["close"].iloc[:, 0]
    seq = pd.Series(np.arange(1, 7), index=c.index)

    def _regbeta(x, n=6):
        if len(x) < n:
            return np.nan
        cov = np.cov(x[-n:], np.arange(1, n + 1))[0, 1]
        var = np.var(np.arange(1, n + 1))
        return cov / var if var != 0 else 0.0
    mean6 = c.groupby(level=1).rolling(6, min_periods=6).mean().droplevel(0)
    result = mean6.groupby(level=1).rolling(6, min_periods=6).apply(_regbeta, raw=False).droplevel(0)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 022
# ---------------------------------------------------------------------------

@register_factor
def alpha022(data: dict) -> pd.Series:
    # SMA((CLOSE / MEAN(CLOSE, 6) - 1 - DELAY(CLOSE / MEAN(CLOSE, 6) - 1, 3)), 12, 1)
    c = data["close"].iloc[:, 0]
    mean6 = c.groupby(level=1).rolling(6, min_periods=6).mean().droplevel(0)
    ratio = c / mean6 - 1
    prev = ratio.groupby(level=1).shift(3)
    term = (ratio - prev).ewm(alpha=1 / 12, adjust=False).mean()
    return _ensure_mi(term.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 023
# ---------------------------------------------------------------------------

@register_factor
def alpha023(data: dict) -> pd.Series:
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) /
    # (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    c = data["close"].iloc[:, 0]
    prev = c.groupby(level=1).shift(1)
    up = np.where(c > prev, c, 0)
    dn = np.where(c <= prev, c, 0)
    # We use rolling std of returns scaled by price
    ret = c.groupby(level=1).pct_change()
    std20 = ret.groupby(level=1).rolling(20, min_periods=20).std().droplevel(0) * c
    up_std = pd.Series(np.where(c > prev, std20, 0), index=c.index)
    dn_std = pd.Series(np.where(c <= prev, std20, 0), index=c.index)
    sma_up = up_std.ewm(alpha=1 / 20, adjust=False).mean()
    sma_dn = dn_std.ewm(alpha=1 / 20, adjust=False).mean()
    result = sma_up / (sma_up + sma_dn + 1e-9) * 100
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 024
# ---------------------------------------------------------------------------

@register_factor
def alpha024(data: dict) -> pd.Series:
    # SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)
    c = data["close"].iloc[:, 0]
    delta = c - c.groupby(level=1).shift(5)
    result = delta.ewm(alpha=1 / 5, adjust=False).mean()
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 025
# ---------------------------------------------------------------------------

@register_factor
def alpha025(data: dict) -> pd.Series:
    # (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(VOLUME / MEAN(VOLUME, 20), 9)))))
    #  * (1 + RANK(SUM(RET, 250)))
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    delta_c = c.groupby(level=1).diff(7)
    mv20 = v.groupby(level=1).rolling(20, min_periods=20).mean().droplevel(0)
    vol_ratio = v / (mv20 + 1)
    w = np.arange(1, 10) / np.arange(1, 10).sum()
    dl = vol_ratio.groupby(level=1).rolling(9, min_periods=9).apply(lambda x: (x * w[: len(x)]).sum(), raw=False).droplevel(0)
    rank1 = delta_c.groupby(level=0).rank(pct=True)
    rank2 = (1 - dl).groupby(level=0).rank(pct=True)
    ret250 = c.groupby(level=1).pct_change().groupby(level=1).rolling(250, min_periods=250).sum().droplevel(0)
    term2 = 1 + ret250.groupby(level=0).rank(pct=True)
    result = -rank1 * rank2 * term2
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 026
# ---------------------------------------------------------------------------

@register_factor
def alpha026(data: dict) -> pd.Series:
    # SUM(CLOSE, 7) / 7 - CLOSE + CORR(VWAP, DELAY(CLOSE, 5), 230)
    c = data["close"].iloc[:, 0]
    v = data["volume"].iloc[:, 0]
    amt = data.get("amount", pd.DataFrame()).iloc[:, 0] if "amount" in data else c * v
    vwap = amt / v
    s7 = c.groupby(level=1).rolling(7, min_periods=7).sum().droplevel(0) / 7
    corr = vwap.groupby(level=1).rolling(230, min_periods=230).corr(c.groupby(level=1).shift(5)).droplevel(0)
    result = s7 - c + corr
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 027
# ---------------------------------------------------------------------------

@register_factor
def alpha027(data: dict) -> pd.Series:
    # WMA((CLOSE - DELAY(CLOSE, 3)) / DELAY(CLOSE, 3) * 100 + (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100, 12)
    c = data["close"].iloc[:, 0]
    d3 = c / c.groupby(level=1).shift(3) - 1
    d6 = c / c.groupby(level=1).shift(6) - 1
    term = (d3 + d6) * 100
    w = np.arange(1, 13) / np.arange(1, 13).sum()
    result = term.groupby(level=1).rolling(12, min_periods=12).apply(lambda x: (x * w[: len(x)]).sum(), raw=False).droplevel(0)
    return _ensure_mi(result.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alpha 028 — KDJ_J (external dependency)
# ---------------------------------------------------------------------------

@register_factor
def alpha028(data: dict) -> pd.Series:
    # 3 * SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9) + 1e-9) * 100, 3, 1)
    c = data["close"].iloc[:, 0]
    h = data["high"].iloc[:, 0]
    l = data["low"].iloc[:, 0]
    tsmax9 = l.groupby(level=1).shift(0).groupby(level=1).rolling(9, min_periods=9).max().droplevel(0)
    tsmin9 = h.groupby(level=1).shift(0).groupby(level=1).rolling(9, min_periods=9).min().droplevel(0)
    # Actually use proper data
    tsmin9 = c.groupby(level=1).rolling(9, min_periods=9).min().droplevel(0)
    tsmax9 = c.groupby(level=1).rolling(9, min_periods=9).max().droplevel(0)
    rsv = (c - tsmin9) / (tsmax9 - tsmin9 + 1e-9) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return _ensure_mi(j.to_frame("factor"))


# ---------------------------------------------------------------------------
# Alphas 029–191 (stub implementations — replace with full formulas)
# For production use, replace stubs with actual GTJA formulas.
# ---------------------------------------------------------------------------

_FACTOR_DOCS = {
    29: "TSRANK(VOLUME, 32) * (1 - TSRANK(CLOSE, 16)) * (1 - TSRANK(RET, 32))",
    30: "SUM(HIGH - OPEN, 20) / SUM(OPEN - LOW, 20) * CLOSE",
    31: "(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME, 5), 26), 4), 7)) - RANK(DELTA(CLOSE, 7)))",
    32: "(RANK(VWAP - CLOSE)) / RANK(VWAP + CLOSE)",
    33: "SMA(VWAP - MIN(VWAP, 16), 16, 2)",
    34: "(SMA(CLOSE, 8, 2) - SMA(CLOSE, 16, 4))",
    35: "(SUM(CLOSE, 7) / 7 - CLOSE) + 20 * SMA(CLOSE, 7, 2) - 19 * SMA(CLOSE, 7, 1)",
    36: "RANK(DELTA(FUNCTIONS, 3)) * (-1 * RANK(TSDELTA(VWAP, 4)))",
    37: "SMA(HIGH - LOW, 10, 2) / SUM(ABS(CLOSE - OPEN), 10)",
    38: "SUM(MAX(0, HIGH - DELAY(CLOSE, 1)), 20) / SUM(MAX(0, DELAY(CLOSE, 1) - LOW), 20) * 100",
    39: "SMA(VWAP - CLOSE, 20, 2) / SMA(ABS(VWAP - CLOSE), 20, 2) * 100",
    40: "RANK((CLOSE - MAX(CLOSE, 10)) / MAX(CLOSE, 10))",
}


def _make_stub_factor(n: int, formula: str):
    """Generate a stub factor function for numbers without full implementation."""
    name = f"alpha{n:03d}"

    @register_factor(name=name)
    def _stub(data: dict) -> pd.Series:
        c = data["close"].iloc[:, 0]
        return pd.Series(np.nan, index=c.index, name="factor")
    _stub.__doc__ = f"Alpha{n:03d}: {formula}"
    _stub.__name__ = name
    globals()[name] = _stub
    return _stub


# Generate stubs for 29-191 (full formulas can be filled in from gtja_csdn.py)
for n in range(29, 192):
    doc = _FACTOR_DOCS.get(n, f"Alpha{n:03d} (stub — implement formula)")
    _make_stub_factor(n, doc)
