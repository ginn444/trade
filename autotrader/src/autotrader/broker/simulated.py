"""SIMULATED broker — MOCK INTEGRATION.

An in-process order matcher used for `mode: mock` and tests. It fills against quotes from whatever
MarketDataProvider it is given (normally SyntheticMarketData). Fills, P&L and balances it reports are
simulation results, NOT brokerage responses, and are labeled ``SIMULATED`` throughout.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from ..data.alpaca_data import parse_occ
from ..models import AccountState, BrokerOrder, BrokerPosition
from .base import BrokerError, DuplicateOrderError, MarketClock, OrderSpec

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


@dataclass
class _SimOrder:
    order: BrokerOrder
    spec: OrderSpec
    parent_client_id: str | None = None
    oco_group: str | None = None


@dataclass
class _Pos:
    qty: float = 0.0
    avg: float = 0.0


class SimulatedBroker:
    name = "SIMULATED"
    is_live = False

    def __init__(self, data, starting_cash: float = 100_000.0, clock: Callable[[], datetime] | None = None,
                 option_fee_per_contract: float = 0.05, fail_next: int = 0):
        self.data = data
        self.cash = starting_cash
        self.last_equity = starting_cash
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fee = option_fee_per_contract
        self.positions: dict[str, _Pos] = {}
        self.orders: dict[str, _SimOrder] = {}       # by broker id
        self._ids = itertools.count(1)
        self.fail_next = fail_next                    # tests: simulate connectivity failures
        self.fills: list[dict] = []

    # ---- helpers --------------------------------------------------------------
    def _maybe_fail(self) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ConnectionError("SIMULATED connectivity failure")

    @staticmethod
    def _is_option(sym: str) -> bool:
        return parse_occ(sym) is not None and len(sym) > 15

    def _quote(self, sym: str):
        return self.data.option_quote(sym) if self._is_option(sym) else self.data.quote(sym)

    def _mult(self, sym: str) -> int:
        return 100 if self._is_option(sym) else 1

    def _mark(self, sym: str) -> float:
        try:
            return self._quote(sym).mid
        except Exception:
            return self.positions[sym].avg

    # ---- reads ------------------------------------------------------------------
    def get_account(self) -> AccountState:
        self._maybe_fail()
        mv = sum(p.qty * self._mark(s) * self._mult(s) for s, p in self.positions.items() if p.qty)
        equity = self.cash + mv
        return AccountState(equity=equity, buying_power=max(self.cash, 0.0), cash=self.cash,
                            last_equity=self.last_equity, options_buying_power=max(self.cash, 0.0))

    def get_positions(self) -> list[BrokerPosition]:
        self._maybe_fail()
        out = []
        for s, p in self.positions.items():
            if not p.qty:
                continue
            m = self._mark(s)
            mult = self._mult(s)
            out.append(BrokerPosition(s, p.qty, p.avg, m, p.qty * m * mult, (m - p.avg) * p.qty * mult,
                                      "us_option" if mult == 100 else "us_equity"))
        return out

    def get_open_orders(self) -> list[BrokerOrder]:
        self._maybe_fail()
        return [o.order for o in self.orders.values() if o.order.is_open]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        self._maybe_fail()
        for o in self.orders.values():
            if o.order.client_order_id == client_order_id:
                return o.order
        return None

    def get_clock(self) -> MarketClock:
        now = self.clock()
        ny = now.astimezone(NY)
        is_open = ny.weekday() < 5 and (ny.hour, ny.minute) >= (9, 30) and ny.hour < 16
        d = ny.date()
        nxt_open = datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY)
        if ny >= nxt_open:
            nxt_open += timedelta(days=1)
        while nxt_open.weekday() >= 5:
            nxt_open += timedelta(days=1)
        nxt_close = datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY)
        if ny >= nxt_close:
            nxt_close = nxt_open.replace(hour=16, minute=0)
        return MarketClock(now, is_open, nxt_open.astimezone(timezone.utc), nxt_close.astimezone(timezone.utc))

    # ---- writes ---------------------------------------------------------------
    def submit_order(self, spec: OrderSpec) -> BrokerOrder:
        self._maybe_fail()
        spec.validate()
        if any(o.order.client_order_id == spec.client_order_id for o in self.orders.values()):
            raise DuplicateOrderError(f"client_order_id {spec.client_order_id} already exists")
        bo = self._new_order(spec)
        self.orders[bo.id] = _SimOrder(bo, spec)
        self._try_fill(self.orders[bo.id])
        return bo

    def _new_order(self, spec: OrderSpec, status: str = "new") -> BrokerOrder:
        sym = spec.symbol or "|".join(f"{l.side}:{l.symbol}" for l in spec.legs)
        return BrokerOrder(id=f"sim-{next(self._ids)}", client_order_id=spec.client_order_id, symbol=sym,
                           side=spec.side, qty=spec.qty, filled_qty=0, status=status, order_type=spec.order_type,
                           limit_price=spec.limit_price, stop_price=spec.stop_loss if status == "held" else None,
                           created_at=self.clock(), order_class=spec.order_class)

    def cancel_order(self, order_id: str) -> None:
        self._maybe_fail()
        so = self.orders.get(order_id)
        if so is None:
            raise BrokerError(f"unknown order {order_id}")
        if so.order.is_open:
            so.order.status = "canceled"

    def replace_order(self, order_id: str, limit_price: float | None = None, qty: int | None = None) -> BrokerOrder:
        self._maybe_fail()
        so = self.orders.get(order_id)
        if so is None or not so.order.is_open:
            raise BrokerError(f"cannot replace order {order_id}")
        so.order.status = "replaced"
        spec = so.spec
        new_spec = OrderSpec(**{**spec.__dict__, "client_order_id": spec.client_order_id + "-r",
                                "limit_price": limit_price or spec.limit_price, "qty": qty or spec.qty})
        bo = self._new_order(new_spec)
        self.orders[bo.id] = _SimOrder(bo, new_spec, so.parent_client_id, so.oco_group)
        self._try_fill(self.orders[bo.id])
        return bo

    def cancel_all_orders(self) -> None:
        self._maybe_fail()
        for so in self.orders.values():
            if so.order.is_open:
                so.order.status = "canceled"

    # ---- matching ---------------------------------------------------------------
    def tick(self) -> None:
        """Advance the simulation: attempt fills on every open order."""
        for so in list(self.orders.values()):
            if so.order.is_open:
                self._try_fill(so)

    def _apply_fill(self, sym: str, side: str, qty: float, price: float) -> None:
        mult = self._mult(sym)
        sign = 1 if side == "buy" else -1
        p = self.positions.setdefault(sym, _Pos())
        new_qty = p.qty + sign * qty
        if p.qty == 0 or (p.qty > 0) == (sign > 0):
            p.avg = (p.avg * abs(p.qty) + price * qty) / abs(new_qty) if new_qty else 0.0
        p.qty = new_qty
        if new_qty == 0:
            p.avg = 0.0
        fee = self.fee * qty if mult == 100 else 0.0
        self.cash -= sign * qty * price * mult + fee
        self.fills.append({"ts": self.clock().isoformat(), "symbol": sym, "side": side, "qty": qty,
                           "price": price, "fee": fee, "source": "SIMULATED"})

    def _try_fill(self, so: _SimOrder) -> None:
        o, spec = so.order, so.spec
        if o.status == "held":
            return self._check_child(so)
        try:
            if spec.order_class == "mleg":
                quotes = {l.symbol: self._quote(l.symbol) for l in spec.legs}
                net = sum((quotes[l.symbol].ask if l.side == "buy" else -quotes[l.symbol].bid) * l.ratio for l in spec.legs)
                is_debit = spec.side == "buy"
                price = net if is_debit else -net
                ok = spec.order_type == "market" or (price <= spec.limit_price if is_debit else price >= spec.limit_price)
                if not ok:
                    return
                for l in spec.legs:
                    q = quotes[l.symbol]
                    self._apply_fill(l.symbol, l.side, spec.qty * l.ratio, q.ask if l.side == "buy" else q.bid)
                o.filled_avg_price = round(price, 4)
            else:
                q = self._quote(spec.symbol)
                px = q.ask if spec.side == "buy" else q.bid
                if spec.order_type == "limit":
                    if (spec.side == "buy" and px > spec.limit_price) or (spec.side == "sell" and px < spec.limit_price):
                        return
                self._apply_fill(spec.symbol, spec.side, spec.qty, px)
                o.filled_avg_price = px
        except Exception as e:  # no quote => no fill
            log.debug("SIMULATED fill skipped for %s: %s", o.client_order_id, e)
            return
        o.filled_qty = o.qty
        o.status = "filled"
        if spec.order_class == "bracket":
            self._spawn_bracket_children(so)
        if so.oco_group:
            for other in self.orders.values():
                if other is not so and other.oco_group == so.oco_group and other.order.is_open:
                    other.order.status = "canceled"

    def _spawn_bracket_children(self, parent: _SimOrder) -> None:
        exit_side = "sell" if parent.spec.side == "buy" else "buy"
        group = parent.order.client_order_id
        tp = OrderSpec(client_order_id=group + "-tp", symbol=parent.spec.symbol, side=exit_side,
                       qty=parent.spec.qty, order_type="limit", limit_price=parent.spec.take_profit, time_in_force="gtc")
        sl = OrderSpec(client_order_id=group + "-sl", symbol=parent.spec.symbol, side=exit_side,
                       qty=parent.spec.qty, order_type="market", stop_loss=parent.spec.stop_loss, time_in_force="gtc")
        tpo = self._new_order(tp)
        slo = self._new_order(sl, status="held")
        slo.order_type = "stop"
        slo.stop_price = parent.spec.stop_loss
        for bo, sp in ((tpo, tp), (slo, sl)):
            self.orders[bo.id] = _SimOrder(bo, sp, parent.order.client_order_id, group)
            parent.order.legs.append(bo)

    def _check_child(self, so: _SimOrder) -> None:
        """Stop orders trigger at the stop and fill at the (possibly worse) prevailing bid/ask."""
        q = self._quote(so.spec.symbol)
        stop = so.spec.stop_loss
        triggered = q.bid <= stop if so.spec.side == "sell" else q.ask >= stop
        if triggered:
            px = q.bid if so.spec.side == "sell" else q.ask
            self._apply_fill(so.spec.symbol, so.spec.side, so.spec.qty, px)
            so.order.filled_qty = so.order.qty
            so.order.filled_avg_price = px
            so.order.status = "filled"
            for other in self.orders.values():
                if other is not so and other.oco_group == so.oco_group and other.order.is_open:
                    other.order.status = "canceled"

    def end_of_day(self) -> None:
        self.last_equity = self.get_account().equity

    # ---- persistence (so mock-mode restarts keep simulated positions consistent with the DB) ----
    def save(self, path: Path) -> None:
        def spec_d(sp: OrderSpec) -> dict:
            d = dict(sp.__dict__)
            d["legs"] = [l.__dict__ for l in sp.legs]
            return d
        def order_d(o: BrokerOrder) -> dict:
            d = {k: v for k, v in o.__dict__.items() if k != "legs"}
            d["created_at"] = o.created_at.isoformat() if o.created_at else None
            d["leg_ids"] = [l.id for l in o.legs]
            return d
        state = {
            "label": "SIMULATED broker state - not a brokerage record",
            "cash": self.cash, "last_equity": self.last_equity,
            "positions": {k: [v.qty, v.avg] for k, v in self.positions.items() if v.qty},
            "orders": [{"order": order_d(so.order), "spec": spec_d(so.spec), "parent": so.parent_client_id,
                        "oco": so.oco_group} for so in self.orders.values()],
            "next_id": next(self._ids),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)
        self._ids = itertools.count(state["next_id"] + 1)

    def load(self, path: Path) -> bool:
        from .base import LegSpec
        path = Path(path)
        if not path.exists():
            return False
        st = json.loads(path.read_text())
        self.cash, self.last_equity = st["cash"], st["last_equity"]
        self.positions = {k: _Pos(q, a) for k, (q, a) in st["positions"].items()}
        self.orders = {}
        by_id = {}
        for rec in st["orders"]:
            sp = dict(rec["spec"])
            sp["legs"] = [LegSpec(**l) for l in sp["legs"]]
            od = dict(rec["order"])
            leg_ids = od.pop("leg_ids", [])
            od["created_at"] = datetime.fromisoformat(od["created_at"]) if od.get("created_at") else None
            bo = BrokerOrder(**od)
            so = _SimOrder(bo, OrderSpec(**sp), rec["parent"], rec["oco"])
            self.orders[bo.id] = so
            by_id[bo.id] = (bo, leg_ids)
        for bo, leg_ids in by_id.values():
            bo.legs = [by_id[i][0] for i in leg_ids if i in by_id]
        self._ids = itertools.count(st["next_id"])
        return True
