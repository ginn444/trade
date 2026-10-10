"""Causal technical indicators. Every value at index t uses only data at or before t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year)


def rate_of_change(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """Channel of the PRIOR n bars (excludes the current bar, so a breakout is detectable)."""
    return df["high"].shift(1).rolling(n, min_periods=n).max(), df["low"].shift(1).rolling(n, min_periods=n).min()


def relative_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    return volume / volume.shift(1).rolling(n, min_periods=n).mean()


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """Session VWAP; resets each calendar day of the index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.Series(df.index.date, index=df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.Series(0, index=df.index)
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def iv_rank(iv_series: pd.Series, n: int = 252) -> pd.Series:
    lo = iv_series.rolling(n, min_periods=20).min()
    hi = iv_series.rolling(n, min_periods=20).max()
    return (iv_series - lo) / (hi - lo).replace(0, np.nan)


def add_standard_features(df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Return a copy with the indicator columns the strategies use."""
    out = df.copy()
    c = out["close"]
    out["ema_fast"] = ema(c, 20)
    out["ema_slow"] = ema(c, 50)
    out["sma_200"] = sma(c, 200)
    out["rsi"] = rsi(c, 14)
    out["rsi2"] = rsi(c, 2)
    out["atr"] = atr(out, 14)
    out["adx"] = adx(out, 14)
    m, s, h = macd(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = m, s, h
    lo, mid, hi = bollinger(c)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = lo, mid, hi
    out["bb_width"] = (hi - lo) / mid
    out["rv20"] = realized_vol(c, 20, periods_per_year)
    out["roc20"] = rate_of_change(c, 20)
    out["roc5"] = rate_of_change(c, 5)
    out["don_high"], out["don_low"] = donchian(out, 20)
    out["rel_vol"] = relative_volume(out["volume"], 20)
    out["dollar_vol20"] = (c * out["volume"]).rolling(20, min_periods=20).mean()
    return out
