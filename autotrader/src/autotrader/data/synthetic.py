"""SYNTHETIC market data — MOCK INTEGRATION.

Generates reproducible, random-walk price series and Black-Scholes-priced option chains so the
full system (strategies, risk engine, order flow, dashboard) can be exercised offline and in tests.

THIS IS NOT REAL MARKET DATA. Nothing produced here says anything about real markets, and any
backtest run on it is labeled ``SYNTHETIC`` and can never approve a strategy for paper/live use.
"""
from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..analysis.options_math import bs
from ..models import OptionContract, Quote
from .alpaca_data import build_occ, parse_occ
from .base import DataUnavailable

NY = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class SyntheticMarketData:
    source = "SYNTHETIC"

    def __init__(self, seed: int = 7, clock: Callable[[], datetime] | None = None,
                 history_days: int = 1500, stale_symbols: set[str] | None = None):
        self.seed = seed
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.history_days = history_days
        self.stale_symbols = set(stale_symbols or ())   # for tests: simulate stale data
        self._daily_cache: dict[str, pd.DataFrame] = {}

    # ---- daily ---------------------------------------------------------------
    def _full_daily(self, symbol: str) -> pd.DataFrame:
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]
        rng = np.random.default_rng(_seed(self.seed, symbol))
        end = self.clock().astimezone(NY).date()
        days = pd.bdate_range(end=end + timedelta(days=30), periods=self.history_days + 22)
        n = len(days)
        start_px = 50 + rng.random() * 400
        # regime-switching drift / vol so strategies see trends, ranges and vol clusters
        drift = np.repeat(rng.normal(0.0003, 0.0015, n // 40 + 1), 40)[:n]
        vol = np.repeat(rng.uniform(0.008, 0.025, n // 30 + 1), 30)[:n]
        rets = rng.normal(drift, vol)
        close = start_px * np.exp(np.cumsum(rets))
        gap = rng.normal(0, vol * 0.3)
        open_ = np.concatenate([[start_px], close[:-1]]) * np.exp(gap)
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.5)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.5)))
        volume = rng.lognormal(16, 0.4, n)
        idx = pd.DatetimeIndex([datetime.combine(d.date(), time(4, 0), tzinfo=timezone.utc) for d in days])
        df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": volume}, index=idx)
        self._daily_cache[symbol] = df
        return df

    def daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        today = self.clock().astimezone(NY).date()
        df = self._full_daily(symbol)
        df = df[df.index.date < today]  # completed sessions only (no partial bar => no look-ahead)
        if df.empty:
            raise DataUnavailable(f"no synthetic history for {symbol}")
        return df.tail(lookback_days).copy()

    def full_history(self, symbol: str) -> pd.DataFrame:
        """All completed synthetic sessions (for backtests)."""
        today = self.clock().astimezone(NY).date()
        df = self._full_daily(symbol)
        return df[df.index.date < today].copy()

    # ---- intraday -------------------------------------------------------------
    def _session_path(self, symbol: str, day: date, minutes: int) -> pd.DataFrame:
        full = self._full_daily(symbol)
        prior = full[full.index.date < day]
        prev_close = float(prior["close"].iloc[-1]) if len(prior) else 100.0
        row = full[full.index.date == day]
        target_close = float(row["close"].iloc[0]) if len(row) else prev_close
        n = int(390 / minutes)
        rng = np.random.default_rng(_seed(self.seed, symbol, day, minutes))
        steps = rng.normal(0, 0.0015 * math.sqrt(minutes), n)
        path = np.cumsum(steps)
        bridge = path - np.linspace(0, 1, n) * path[-1]   # Brownian bridge -> lands on daily close
        closes = prev_close * np.exp(bridge + np.linspace(0, 1, n) * math.log(target_close / prev_close))
        opens = np.concatenate([[prev_close], closes[:-1]])
        wig = np.abs(rng.normal(0, 0.0007, n))
        start = datetime.combine(day, SESSION_OPEN, tzinfo=NY)
        idx = pd.DatetimeIndex([(start + timedelta(minutes=minutes * i)).astimezone(timezone.utc) for i in range(n)])
        return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * (1 + wig),
                             "low": np.minimum(opens, closes) * (1 - wig), "close": closes,
                             "volume": rng.lognormal(12, 0.5, n)}, index=idx)

    def intraday_bars(self, symbol: str, minutes: int, lookback_days: int) -> pd.DataFrame:
        now = self.clock()
        today = now.astimezone(NY).date()
        frames = []
        for d in pd.bdate_range(end=today, periods=max(lookback_days, 1)):
            p = self._session_path(symbol, d.date(), minutes)
            # only bars that have fully closed by `now`
            frames.append(p[p.index + timedelta(minutes=minutes) <= now])
        df = pd.concat(frames)
        if df.empty:
            raise DataUnavailable(f"no synthetic intraday bars for {symbol} yet")
        return df

    def _last_price(self, symbol: str) -> float:
        now = self.clock()
        ny = now.astimezone(NY)
        if SESSION_OPEN <= ny.time() < SESSION_CLOSE and ny.weekday() < 5:
            p = self._session_path(symbol, ny.date(), 1)
            p = p[p.index <= now]
            if len(p):
                return float(p["close"].iloc[-1])
        return float(self.daily_bars(symbol, 1)["close"].iloc[-1])

    def quote(self, symbol: str) -> Quote:
        px = self._last_price(symbol)
        half = px * 0.0002
        ts = self.clock() - (timedelta(hours=2) if symbol in self.stale_symbols else timedelta(seconds=1))
        return Quote(symbol, round(px - half, 2), round(px + half, 2), ts, 100, 100, self.source)

    # ---- options --------------------------------------------------------------
    def _vol(self, underlying: str) -> float:
        d = self.daily_bars(underlying, 30)
        r = np.log(d["close"] / d["close"].shift(1)).dropna()
        return float(max(0.10, r.std() * math.sqrt(252) * 1.1))

    def _price_option(self, underlying: str, exp: date, kind: str, strike: float) -> tuple[Quote, OptionContract]:
        now = self.clock()
        spot = self._last_price(underlying)
        t = max((datetime.combine(exp, SESSION_CLOSE, tzinfo=NY) - now).total_seconds(), 60) / (365 * 86400)
        vol = self._vol(underlying)
        g = bs(spot, strike, t, vol, kind=kind)
        mid = max(g.price, 0.01)
        half = max(0.01, mid * 0.02)
        sym = build_occ(underlying, exp, kind, strike)
        ts = now - (timedelta(hours=2) if underlying in self.stale_symbols else timedelta(seconds=2))
        q = Quote(sym, round(max(mid - half, 0.01), 2), round(mid + half, 2), ts, 10, 10, self.source)
        rng = np.random.default_rng(_seed(self.seed, sym))
        c = OptionContract(sym, underlying, kind, strike, exp, q, vol, g.delta, g.gamma, g.theta, g.vega / 100,
                           int(rng.integers(600, 20000)), int(rng.integers(60, 5000)))
        return q, c

    def option_chain(self, underlying: str, min_dte: int, max_dte: int,
                     kind: str | None = None, today: date | None = None) -> list[OptionContract]:
        today = today or self.clock().astimezone(NY).date()
        spot = self._last_price(underlying)
        step = 1.0 if spot < 200 else 5.0
        exps = [today + timedelta(days=d) for d in range(min_dte, max_dte + 1)
                if (today + timedelta(days=d)).weekday() == 4]  # Friday expirations
        out = []
        for exp in exps:
            for k in np.arange(round(spot * 0.85 / step) * step, spot * 1.15, step):
                for typ in ([kind] if kind else ["call", "put"]):
                    out.append(self._price_option(underlying, exp, typ, float(k))[1])
        return out

    def option_quote(self, occ_symbol: str) -> Quote:
        parsed = parse_occ(occ_symbol)
        if parsed is None:
            raise DataUnavailable(f"bad OCC symbol {occ_symbol}")
        und, exp, typ, strike = parsed
        return self._price_option(und, exp, typ, strike)[0]
