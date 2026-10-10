"""Alpaca market-data integration (stocks + options). Requires real credentials; never fabricates data."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..models import OptionContract, Quote
from .base import DataUnavailable, normalize_bars

log = logging.getLogger(__name__)


class AlpacaMarketData:
    def __init__(self, api_key: str, secret_key: str, stock_feed: str = "iex",
                 options_feed: str = "indicative", trading_client=None):
        from alpaca.data.enums import DataFeed, OptionsFeed
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self._stock = StockHistoricalDataClient(api_key, secret_key)
        self._opt = OptionHistoricalDataClient(api_key, secret_key)
        self._feed = DataFeed(stock_feed)
        self._ofeed = OptionsFeed(options_feed)
        self._trading = trading_client  # used for open interest via contracts endpoint
        self.source = f"alpaca:{stock_feed}/{options_feed}"

    # ---- bars ---------------------------------------------------------------
    def _bars(self, symbol: str, tf, start: datetime) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start,
                               adjustment=Adjustment.ALL, feed=self._feed)
        bars = self._stock.get_stock_bars(req)
        return normalize_bars(bars.df)

    def daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        from alpaca.data.timeframe import TimeFrame
        start = datetime.now(timezone.utc) - timedelta(days=int(lookback_days * 1.5) + 5)
        return self._bars(symbol, TimeFrame.Day, start)

    def intraday_bars(self, symbol: str, minutes: int, lookback_days: int) -> pd.DataFrame:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        return self._bars(symbol, TimeFrame(minutes, TimeFrameUnit.Minute), start)

    # ---- quotes -------------------------------------------------------------
    def quote(self, symbol: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self._stock.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed))
        if symbol not in q:
            raise DataUnavailable(f"no quote for {symbol}")
        r = q[symbol]
        return Quote(symbol, float(r.bid_price or 0), float(r.ask_price or 0), r.timestamp,
                     float(r.bid_size or 0), float(r.ask_size or 0), self.source)

    def option_quote(self, occ_symbol: str) -> Quote:
        from alpaca.data.requests import OptionLatestQuoteRequest
        q = self._opt.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol, feed=self._ofeed))
        if occ_symbol not in q:
            raise DataUnavailable(f"no option quote for {occ_symbol}")
        r = q[occ_symbol]
        return Quote(occ_symbol, float(r.bid_price or 0), float(r.ask_price or 0), r.timestamp,
                     float(r.bid_size or 0), float(r.ask_size or 0), self.source)

    # ---- options chain ------------------------------------------------------
    def option_chain(self, underlying: str, min_dte: int, max_dte: int,
                     kind: str | None = None, today: date | None = None) -> list[OptionContract]:
        from alpaca.data.requests import OptionChainRequest
        today = today or datetime.now(timezone.utc).date()
        kw = dict(underlying_symbol=underlying, feed=self._ofeed,
                  expiration_date_gte=today + timedelta(days=min_dte),
                  expiration_date_lte=today + timedelta(days=max_dte))
        if kind:
            kw["type"] = kind
        snaps = self._opt.get_option_chain(OptionChainRequest(**kw))
        oi = self._open_interest(underlying, today, min_dte, max_dte, kind)
        out: list[OptionContract] = []
        for sym, s in (snaps or {}).items():
            parsed = parse_occ(sym)
            if parsed is None:
                continue
            und, exp, typ, strike = parsed
            q = None
            if s.latest_quote is not None:
                lq = s.latest_quote
                q = Quote(sym, float(lq.bid_price or 0), float(lq.ask_price or 0), lq.timestamp,
                          float(lq.bid_size or 0), float(lq.ask_size or 0), self.source)
            g = s.greeks
            out.append(OptionContract(
                symbol=sym, underlying=und, option_type=typ, strike=strike, expiration=exp, quote=q,
                implied_vol=s.implied_volatility, delta=getattr(g, "delta", None), gamma=getattr(g, "gamma", None),
                theta=getattr(g, "theta", None), vega=getattr(g, "vega", None),
                open_interest=oi.get(sym), volume=None))
        return out

    def _open_interest(self, underlying, today, min_dte, max_dte, kind) -> dict[str, int]:
        if self._trading is None:
            return {}
        from alpaca.trading.requests import GetOptionContractsRequest
        res: dict[str, int] = {}
        token = None
        try:
            for _ in range(10):
                kw = dict(underlying_symbols=[underlying], expiration_date_gte=today + timedelta(days=min_dte),
                          expiration_date_lte=today + timedelta(days=max_dte), limit=1000, page_token=token)
                if kind:
                    kw["type"] = kind
                resp = self._trading.get_option_contracts(GetOptionContractsRequest(**kw))
                for c in resp.option_contracts or []:
                    if c.open_interest is not None:
                        res[c.symbol] = int(float(c.open_interest))
                token = getattr(resp, "next_page_token", None)
                if not token:
                    break
        except Exception as e:  # open interest is a filter input; missing OI => contract filtered out
            log.warning("open interest lookup failed for %s: %s", underlying, e)
        return res


def parse_occ(sym: str):
    """Parse an OCC option symbol like 'SPY241220C00450000'."""
    try:
        root = sym[:-15]
        y, m, d = int(sym[-15:-13]), int(sym[-13:-11]), int(sym[-11:-9])
        typ = {"C": "call", "P": "put"}[sym[-9]]
        strike = int(sym[-8:]) / 1000.0
        return root, date(2000 + y, m, d), typ, strike
    except (ValueError, KeyError, IndexError):
        return None


def build_occ(root: str, expiration: date, kind: str, strike: float) -> str:
    return f"{root}{expiration:%y%m%d}{'C' if kind == 'call' else 'P'}{int(round(strike * 1000)):08d}"
