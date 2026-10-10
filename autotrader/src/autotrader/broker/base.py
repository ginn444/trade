"""Brokerage interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from ..models import AccountState, BrokerOrder, BrokerPosition


class BrokerError(RuntimeError):
    pass


class DuplicateOrderError(BrokerError):
    """The broker already has an order with this client_order_id."""


@dataclass
class LegSpec:
    symbol: str
    side: Literal["buy", "sell"]
    ratio: int = 1
    position_intent: str | None = None   # buy_to_open / sell_to_open / buy_to_close / sell_to_close


@dataclass
class OrderSpec:
    client_order_id: str
    symbol: str                       # stock ticker / OCC symbol; "" for multi-leg
    side: Literal["buy", "sell"]
    qty: int
    order_type: Literal["market", "limit"] = "limit"
    limit_price: float | None = None
    time_in_force: Literal["day", "gtc"] = "day"
    order_class: Literal["simple", "bracket", "mleg"] = "simple"
    take_profit: float | None = None
    stop_loss: float | None = None
    legs: list[LegSpec] = field(default_factory=list)
    position_intent: str | None = None

    def validate(self) -> None:
        if self.qty <= 0 or int(self.qty) != self.qty:
            raise BrokerError("qty must be a positive integer")
        if self.order_type == "limit" and (self.limit_price is None):
            raise BrokerError("limit order requires limit_price")
        if self.order_class == "bracket" and (self.take_profit is None or self.stop_loss is None):
            raise BrokerError("bracket requires take_profit and stop_loss")
        if self.order_class == "mleg":
            if len(self.legs) < 2:
                raise BrokerError("mleg requires >= 2 legs")
            buys = [l for l in self.legs if l.side == "buy"]
            sells = [l for l in self.legs if l.side == "sell"]
            opening = all((l.position_intent or "").endswith("to_open") for l in self.legs)
            # Defined-risk only: every short leg opened must be paired with a long leg.
            if opening and sum(l.ratio for l in sells) > sum(l.ratio for l in buys):
                raise BrokerError("uncovered short option legs are prohibited")
        elif self.position_intent == "sell_to_open":
            raise BrokerError("naked short options are prohibited")


@dataclass
class MarketClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class Broker(Protocol):
    name: str
    is_live: bool

    def get_account(self) -> AccountState: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def get_open_orders(self) -> list[BrokerOrder]: ...
    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None: ...
    def submit_order(self, spec: OrderSpec) -> BrokerOrder: ...
    def cancel_order(self, order_id: str) -> None: ...
    def replace_order(self, order_id: str, limit_price: float | None = None, qty: int | None = None) -> BrokerOrder: ...
    def cancel_all_orders(self) -> None: ...
    def get_clock(self) -> MarketClock: ...
