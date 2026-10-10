"""Alpaca brokerage integration (paper or live).

Paper vs live is fixed at construction and verified against the client's base URL. A live broker
can only be built through ``factory.build_broker`` after ``assert_live_authorized`` passes.
"""
from __future__ import annotations

import logging

from ..models import AccountState, BrokerOrder, BrokerPosition
from .base import BrokerError, DuplicateOrderError, MarketClock, OrderSpec

log = logging.getLogger(__name__)


def _f(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _enum(v) -> str:
    return getattr(v, "value", v) if v is not None else ""


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, *, paper: bool, _live_authorized: bool = False):
        from alpaca.trading.client import TradingClient
        if not paper and not _live_authorized:
            raise BrokerError("live Alpaca broker requested without authorization")
        self.client = TradingClient(api_key, secret_key, paper=paper)
        raw = getattr(self.client, "_base_url", "")
        base = str(getattr(raw, "value", raw) or "").lower()
        if paper and "paper" not in base:
            raise BrokerError(f"paper broker is not pointed at the paper endpoint ({base!r})")
        if not paper and "paper" in base:
            raise BrokerError("live broker is pointed at the paper endpoint")
        self.is_live = not paper
        self.name = "alpaca-live" if self.is_live else "alpaca-paper"

    # ---- reads ----------------------------------------------------------------
    def get_account(self) -> AccountState:
        a = self.client.get_account()
        return AccountState(equity=_f(a.equity), buying_power=_f(a.buying_power), cash=_f(a.cash),
                            last_equity=_f(a.last_equity), options_buying_power=_f(a.options_buying_power, None),
                            trading_blocked=bool(a.trading_blocked or a.account_blocked))

    def get_positions(self) -> list[BrokerPosition]:
        return [BrokerPosition(p.symbol, _f(p.qty), _f(p.avg_entry_price), _f(p.current_price),
                               _f(p.market_value), _f(p.unrealized_pl), _enum(p.asset_class))
                for p in self.client.get_all_positions()]

    def _conv(self, o) -> BrokerOrder:
        return BrokerOrder(
            id=str(o.id), client_order_id=o.client_order_id, symbol=o.symbol or "", side=_enum(o.side),
            qty=_f(o.qty), filled_qty=_f(o.filled_qty), status=_enum(o.status), order_type=_enum(o.order_type),
            limit_price=_f(o.limit_price, None), stop_price=_f(o.stop_price, None),
            filled_avg_price=_f(o.filled_avg_price, None), created_at=o.created_at,
            order_class=_enum(o.order_class) or "simple", legs=[self._conv(l) for l in (o.legs or [])])

    def get_open_orders(self) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        return [self._conv(o) for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500))]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.requests import GetOrderByIdRequest
        try:
            o = self.client.get_order_by_client_id(client_order_id)
            if o.legs is None and _enum(o.order_class) in ("bracket", "oco", "oto", "mleg"):
                o = self.client.get_order_by_id(o.id, GetOrderByIdRequest(nested=True))
            return self._conv(o)
        except APIError as e:
            if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                return None
            raise

    def get_clock(self) -> MarketClock:
        c = self.client.get_clock()
        return MarketClock(c.timestamp, bool(c.is_open), c.next_open, c.next_close)

    # ---- writes ---------------------------------------------------------------
    def submit_order(self, spec: OrderSpec) -> BrokerOrder:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import (LimitOrderRequest, MarketOrderRequest, OptionLegRequest,
                                             StopLossRequest, TakeProfitRequest)
        spec.validate()
        kw: dict = dict(qty=spec.qty, time_in_force=TimeInForce(spec.time_in_force),
                        client_order_id=spec.client_order_id)
        if spec.order_class == "mleg":
            kw["order_class"] = OrderClass.MLEG
            kw["legs"] = [OptionLegRequest(symbol=l.symbol, ratio_qty=l.ratio, side=OrderSide(l.side),
                                           position_intent=PositionIntent(l.position_intent) if l.position_intent else None)
                          for l in spec.legs]
        else:
            kw["symbol"] = spec.symbol
            kw["side"] = OrderSide(spec.side)
            if spec.position_intent:
                kw["position_intent"] = PositionIntent(spec.position_intent)
        if spec.order_class == "bracket":
            kw["order_class"] = OrderClass.BRACKET
            kw["take_profit"] = TakeProfitRequest(limit_price=round(spec.take_profit, 2))
            kw["stop_loss"] = StopLossRequest(stop_price=round(spec.stop_loss, 2))
        try:
            if spec.order_type == "limit":
                req = LimitOrderRequest(limit_price=round(spec.limit_price, 2), **kw)
            else:
                req = MarketOrderRequest(**kw)
            return self._conv(self.client.submit_order(req))
        except APIError as e:
            msg = str(e)
            if "client_order_id" in msg and ("unique" in msg or "exists" in msg or "duplicate" in msg):
                raise DuplicateOrderError(msg) from e
            raise BrokerError(msg) from e

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order_by_id(order_id)

    def replace_order(self, order_id: str, limit_price: float | None = None, qty: int | None = None) -> BrokerOrder:
        from alpaca.trading.requests import ReplaceOrderRequest
        req = ReplaceOrderRequest(limit_price=round(limit_price, 2) if limit_price else None, qty=qty)
        return self._conv(self.client.replace_order_by_id(order_id, req))

    def cancel_all_orders(self) -> None:
        self.client.cancel_orders()
