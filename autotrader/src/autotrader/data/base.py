"""Market-data provider interface."""
from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

from ..models import OptionContract, Quote


class DataUnavailable(RuntimeError):
    """Raised when data is missing or stale. Callers must NOT trade on the affected symbol."""


class MarketDataProvider(Protocol):
    source: str  # e.g. "alpaca:iex", "SYNTHETIC"

    def daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame: ...
    def intraday_bars(self, symbol: str, minutes: int, lookback_days: int) -> pd.DataFrame: ...
    def quote(self, symbol: str) -> Quote: ...
    def option_chain(self, underlying: str, min_dte: int, max_dte: int,
                     kind: str | None = None, today: date | None = None) -> list[OptionContract]: ...
    def option_quote(self, occ_symbol: str) -> Quote: ...


BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataUnavailable("no bars returned")
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise DataUnavailable(f"bars missing columns {missing}")
    df = df[BAR_COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df[["open", "high", "low", "close"]].le(0).any().any():
        raise DataUnavailable("non-positive prices in bars")
    return df
