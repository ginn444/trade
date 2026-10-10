from datetime import timedelta
from types import SimpleNamespace

import pytest

from autotrader.broker.base import BrokerError, LegSpec, OrderSpec
from autotrader.broker.simulated import SimulatedBroker
from autotrader.data.synthetic import SyntheticMarketData
from autotrader.db import Database
from autotrader.execution.orders import OrderManager
from autotrader.execution.positions import evaluate_exit
from autotrader.health import HealthMonitor
from autotrader.models import Quote

from conftest import NOW, option_candidate, proposal, quote, spread_candidate, stock_candidate


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


class FixedData:
    """Quotes under test control."""
    source = "TEST"

    def __init__(self, clock):
        self.clock, self.px = clock, {}

    def quote(self, s):
        b, a = self.px[s]
        return Quote(s, b, a, self.clock(), 1, 1, "TEST")

    option_quote = quote


@pytest.fixture
def env(cfg):
    clock = Clock()
    data = FixedData(clock)
    broker = SimulatedBroker(data, 100_000, clock=clock)
    db = Database(cfg.database_url, "mock", clock=clock)
    om = OrderManager(cfg, broker, db, HealthMonitor(cfg.broker_health), clock)
    return SimpleNamespace(clock=clock, data=data, broker=broker, db=db, om=om, cfg=cfg)


# ---------------------------------------------------------------- order spec safety
def test_order_spec_blocks_naked_and_uncovered_short_options():
    with pytest.raises(BrokerError):
        OrderSpec("x", "AAA261016C00100000", "sell", 1, "limit", 1.0, position_intent="sell_to_open").validate()
    legs = [LegSpec("A", "buy", 1, "buy_to_open"), LegSpec("B", "sell", 2, "sell_to_open")]
    with pytest.raises(BrokerError):
        OrderSpec("x", "", "buy", 1, "limit", 1.0, order_class="mleg", legs=legs).validate()
    with pytest.raises(BrokerError):
        OrderSpec("x", "AAA", "buy", 0, "limit", 1.0).validate()


# ---------------------------------------------------------------- simulated broker
def test_sim_bracket_stop_fills_at_worse_gap_price(env):
    env.data.px["AAA"] = (99.99, 100.01)
    env.broker.submit_order(OrderSpec("e1", "AAA", "buy", 10, "limit", 100.05, order_class="bracket",
                                      take_profit=104, stop_loss=98))
    assert env.broker.positions["AAA"].qty == 10
    env.data.px["AAA"] = (95.00, 95.02)          # gap far through the stop
    env.broker.tick()
    assert env.broker.positions["AAA"].qty == 0
    stop_fill = [f for f in env.broker.fills if f["side"] == "sell"][0]
    assert stop_fill["price"] == 95.00           # worse than the 98 stop
    tp = [o for o in env.broker.orders.values() if o.order.client_order_id == "e1-tp"][0]
    assert tp.order.status == "canceled"         # OCO


def test_sim_duplicate_client_id_rejected(env):
    from autotrader.broker.base import DuplicateOrderError
    env.data.px["AAA"] = (99.99, 100.01)
    env.broker.submit_order(OrderSpec("dup", "AAA", "buy", 1, "limit", 99.0))
    with pytest.raises(DuplicateOrderError):
        env.broker.submit_order(OrderSpec("dup", "AAA", "buy", 1, "limit", 99.0))


def test_sim_broker_persistence_roundtrip(env, tmp_path):
    env.data.px["AAA"] = (99.99, 100.01)
    env.broker.submit_order(OrderSpec("e1", "AAA", "buy", 10, "limit", 100.05, order_class="bracket",
                                      take_profit=104, stop_loss=98))
    env.broker.save(tmp_path / "sim.json")
    b2 = SimulatedBroker(env.data, 0, clock=env.clock)
    assert b2.load(tmp_path / "sim.json")
    assert b2.positions["AAA"].qty == 10 and b2.cash == env.broker.cash
    parent = b2.get_order_by_client_id("e1")
    assert {l.client_order_id for l in parent.legs} == {"e1-tp", "e1-sl"}
    env.data.px["AAA"] = (97.0, 97.02)
    b2.tick()
    assert b2.positions["AAA"].qty == 0


# ---------------------------------------------------------------- order manager
def test_entry_write_ahead_fill_and_duplicate_block(env):
    env.data.px["AAA"] = (99.99, 100.01)
    p = proposal(stock_candidate())
    p.risk_usd = 400
    t = env.om.submit_entry(p, 10)
    assert t.status == "open" and t.entry_price == 100.01
    rec = env.db.get_order(p.client_order_id)
    assert rec.purpose == "entry" and rec.status == "filled"
    assert env.om.submit_entry(p, 10) is None    # same signal again -> blocked
    assert any(a.event == "duplicate_blocked" for a in env.db.recent_audit())


def test_ambiguous_submit_failure_is_resolved_by_lookup(env):
    env.data.px["AAA"] = (99.99, 100.01)
    real_submit = env.broker.submit_order

    def flaky(spec):
        real_submit(spec)                       # order reaches the broker...
        raise TimeoutError("response lost")     # ...but the response is lost
    env.broker.submit_order = flaky
    t = env.om.submit_entry(proposal(stock_candidate()), 5)
    assert t is not None and t.status == "open"
    assert any(a.event == "ambiguous_submit_resolved" for a in env.db.recent_audit())


def test_failed_submit_cancels_trade_and_degrades_health(env):
    env.data.px["AAA"] = (99.99, 100.01)
    env.broker.fail_next = 100
    assert env.om.submit_entry(proposal(stock_candidate()), 5) is None
    assert env.db.open_trades() == []
    for _ in range(3):
        try:
            env.om.reconcile()
        except Exception:
            pass
    assert env.om.health.healthy is False


def test_bracket_exit_detected_and_no_double_sell(env):
    env.data.px["AAA"] = (99.99, 100.01)
    t = env.om.submit_entry(proposal(stock_candidate()), 10)
    env.data.px["AAA"] = (97.5, 97.52)
    env.broker.tick()                          # broker-side stop fills
    env.om.sync()
    t = env.db.get_trade(t.id)
    assert t.status == "closed" and t.exit_reason == "bracket_stop"
    assert t.realized_pnl == pytest.approx((97.5 - 100.01) * 10)
    assert env.broker.positions["AAA"].qty == 0


def test_exit_blocked_when_broker_does_not_hold_position(env):
    env.data.px["AAA"] = (99.99, 100.01)
    t = env.om.submit_entry(proposal(stock_candidate()), 10)
    env.broker.positions["AAA"].qty = 0        # position vanished externally
    assert env.om.submit_exit(env.db.get_trade(t.id), "stop_loss", {"AAA": quote("AAA", 97, 97.1)}) is False
    assert env.broker.positions["AAA"].qty == 0   # never sold short
    assert any(a.event == "exit_blocked_position_mismatch" for a in env.db.recent_audit())


def test_option_and_spread_round_trip(env):
    c = option_candidate()
    leg = c.legs[0].symbol
    env.data.px[leg] = (2.0, 2.1)
    t = env.om.submit_entry(proposal(c), 2)
    assert t.status == "pending_entry"          # limit at mid rests; never chases to the ask
    env.data.px[leg] = (1.98, 2.04)
    env.broker.tick()
    env.om.sync()
    t = env.db.get_trade(t.id)
    assert t.status == "open" and env.broker.positions[leg].qty == 2
    env.data.px[leg] = (3.0, 3.1)
    assert env.om.submit_exit(env.db.get_trade(t.id), "take_profit", {leg: env.data.quote(leg)})
    assert env.db.get_trade(t.id).status == "closing"   # exit limit near mid rests
    env.data.px[leg] = (3.06, 3.12)
    env.broker.tick()
    env.om.sync()
    t = env.db.get_trade(t.id)
    assert t.status == "closed" and t.realized_pnl > 0
    s = spread_candidate()
    for l in s.legs:
        env.data.px[l.symbol] = (l.contract.quote.bid, l.contract.quote.ask)
    ts = env.om.submit_entry(proposal(s), 1)
    assert ts.status == "pending_entry"          # net-debit limit rests until the market reaches it
    env.data.px[s.legs[0].symbol], env.data.px[s.legs[1].symbol] = (3.0, 3.02), (1.02, 1.05)
    env.broker.tick()
    env.om.sync()
    ts = env.db.get_trade(ts.id)
    assert ts.status == "open" and ts.entry_price == pytest.approx(2.0)
    assert env.broker.positions[s.legs[1].symbol].qty == -1   # short leg is covered by the long leg
    quotes = {l.symbol: env.data.quote(l.symbol) for l in s.legs}
    assert env.om.submit_exit(env.db.get_trade(ts.id), "time_stop", quotes, aggressive=True)
    assert all(env.broker.positions[l.symbol].qty == 0 for l in s.legs)


def test_reconcile_detects_unknown_position(env):
    env.data.px["ZZZ"] = (9.99, 10.01)
    env.broker.submit_order(OrderSpec("manual", "ZZZ", "buy", 1, "market"))
    rep = env.om.reconcile()
    assert not rep["ok"] and rep["unknown_positions"] == ["ZZZ"]


def test_stale_entry_cancelled(env):
    env.data.px["AAA"] = (99.99, 100.01)
    p = proposal(stock_candidate(entry=95.0, stop=93.0, target=99.0))   # limit below market: rests
    t = env.om.submit_entry(p, 5)
    assert t.status == "pending_entry"
    env.clock.t = NOW + timedelta(seconds=env.cfg.execution.entry_timeout_seconds + 1)
    env.om.cancel_stale_entries()
    env.om.sync()
    assert env.db.get_trade(t.id).status == "cancelled"


# ---------------------------------------------------------------- exit rules
def _trade(**kw):
    base = dict(status="open", instrument="stock", underlying="AAA", legs=None, stop_price=98, target_price=104,
                entry_price=100, filled_qty=10, qty=10, multiplier=1, intraday_only=False, max_hold_minutes=None,
                opened_at=NOW)
    base.update(kw)
    return SimpleNamespace(**base)


def test_exit_rules(cfg):
    q = lambda b: {"AAA": quote("AAA", b, b + 0.02)}  # noqa: E731
    assert evaluate_exit(_trade(), q(100), NOW, cfg) is None
    assert evaluate_exit(_trade(), q(97.9), NOW, cfg).reason == "stop_loss"
    assert evaluate_exit(_trade(), q(104.5), NOW, cfg).reason == "take_profit"
    assert evaluate_exit(_trade(max_hold_minutes=30), q(100), NOW + timedelta(minutes=31), cfg).reason == "time_stop"
    eod = NOW.replace(hour=19, minute=55)      # 15:55 New York
    assert evaluate_exit(_trade(intraday_only=True), q(100), eod, cfg).reason == "eod_flatten"
    assert evaluate_exit(_trade(), q(100), NOW, cfg, kill_flatten=True).reason == "kill_switch_flatten"
    legs = [{"symbol": "AAA", "side": "buy", "ratio": 1, "expiration": (NOW.date() + timedelta(days=1)).isoformat()}]
    assert evaluate_exit(_trade(instrument="option", legs=legs), q(100), NOW, cfg).reason == "expiry_risk"
