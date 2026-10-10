from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from autotrader.models import AIDecision, Direction, OptionLeg
from autotrader.risk.engine import RiskContext, RiskEngine

from conftest import (NOW, account, option_candidate, proposal, quote, spread_candidate, stock_candidate)

ALL = {"stock_momentum", "option_momentum", "option_debit_spread"}


def ctx_for(c, **over):
    quotes = {}
    if c.legs:
        for l in c.legs:
            q = l.contract.quote
            quotes[l.symbol] = q
    else:
        quotes[c.symbol] = quote(c.symbol, c.entry_price - 0.01, c.entry_price + 0.01)
    base = dict(now=NOW, account=account(), day_start_equity=100_000.0, quotes=quotes, kill_switch_engaged=False,
                broker_healthy=True, market_open=True, session_open_time=NOW.replace(hour=13, minute=30),
                approved_strategies=set(ALL))
    base.update(over)
    return RiskContext(**base)


@pytest.fixture
def eng(cfg):
    return RiskEngine(cfg)


def test_valid_stock_trade_approved_and_sized_within_budget(eng):
    c = stock_candidate()
    p = proposal(c)
    v = eng.validate(p, ctx_for(c))
    assert v.approved, v.reasons
    assert v.adjusted_quantity == 200                  # capped by 20% notional (219 by risk)
    assert p.risk_usd <= 100_000 * 0.005


def test_ai_cannot_increase_size(eng):
    c = stock_candidate()
    p = proposal(c, qty=50)
    assert eng.validate(p, ctx_for(c)).adjusted_quantity == 50
    p2 = proposal(c, qty=10_000)
    assert eng.validate(p2, ctx_for(c)).adjusted_quantity == 200


@pytest.mark.parametrize("over,needle", [
    ({"kill_switch_engaged": True}, "kill switch"),
    ({"broker_healthy": False}, "connectivity"),
    ({"market_open": False}, "market closed"),
    ({"account": None}, "account state"),
    ({"account": account(equity=97_900)}, "daily loss limit"),
    ({"duplicate_signal": True}, "duplicate"),
    ({"approved_strategies": set()}, "backtest validation"),
    ({"entry_orders_today": 20}, "max entry orders"),
    ({"reconcile_ok": False}, "reconciliation"),
    ({"now": NOW.replace(hour=19, minute=45)}, "cutoff"),
    ({"now": NOW.replace(hour=13, minute=32)}, "post-open delay"),
])
def test_hard_limits(eng, over, needle):
    c = stock_candidate()
    v = eng.validate(proposal(c), ctx_for(c, **over))
    assert not v.approved
    assert any(needle in r for r in v.reasons), v.reasons


def test_max_positions_and_no_adding(eng):
    c = stock_candidate()
    trades = [SimpleNamespace(underlying=s) for s in ("X", "Y", "Z")]
    v = eng.validate(proposal(c), ctx_for(c, open_trades=trades))
    assert not v.approved and any("max simultaneous" in r for r in v.reasons)
    v = eng.validate(proposal(c), ctx_for(c, open_trades=[SimpleNamespace(underlying="AAA")]))
    assert not v.approved and any("averaging down" in r for r in v.reasons)


def test_pending_order_same_symbol_blocked(eng):
    c = stock_candidate()
    v = eng.validate(proposal(c), ctx_for(c, open_broker_orders=[SimpleNamespace(symbol="AAA")]))
    assert not v.approved


def test_stale_missing_and_wide_quotes(eng):
    c = stock_candidate()
    v = eng.validate(proposal(c), ctx_for(c, quotes={"AAA": quote("AAA", 99.99, 100.01, age_s=600)}))
    assert not v.approved and any("stale" in r for r in v.reasons)
    v = eng.validate(proposal(c), ctx_for(c, quotes={}))
    assert not v.approved and any("missing" in r for r in v.reasons)
    v = eng.validate(proposal(c), ctx_for(c, quotes={"AAA": quote("AAA", 99.0, 101.0)}))
    assert not v.approved and any("spread" in r for r in v.reasons)


def test_short_stock_prohibited(eng):
    c = stock_candidate(direction=Direction.SHORT, stop=102, target=96)
    v = eng.validate(proposal(c), ctx_for(c))
    assert not v.approved and any("short stock" in r for r in v.reasons)


def test_bad_geometry_and_reward_risk(eng):
    c = stock_candidate(stop=101)
    assert not eng.validate(proposal(c), ctx_for(c)).approved
    c = stock_candidate(stop=98, target=101)   # RR 0.5
    v = eng.validate(proposal(c), ctx_for(c))
    assert not v.approved and any("reward/risk" in r for r in v.reasons)


def test_entry_far_from_market_rejected(eng):
    c = stock_candidate()
    p = proposal(c)
    p.entry_price = 103.0       # e.g. hallucinated price
    p.stop_price, p.target_price = 101.0, 107.0
    v = eng.validate(p, ctx_for(c))
    assert not v.approved and any("too far from market" in r for r in v.reasons)


def test_long_option_sized_on_full_premium(eng):
    c = option_candidate()
    p = proposal(c)
    v = eng.validate(p, ctx_for(c))
    assert v.approved, v.reasons
    # $500 budget / ($2.05 x 100 + fees) -> 2 contracts
    assert v.adjusted_quantity == 2
    assert p.risk_usd <= 500


def test_long_put_for_bearish_view_allowed(eng):
    c = option_candidate(direction=Direction.SHORT)
    assert eng.validate(proposal(c), ctx_for(c)).approved


def test_naked_short_option_rejected(eng):
    c = option_candidate()
    c.legs[0] = OptionLeg(c.legs[0].symbol, "sell", 1, c.legs[0].contract)
    v = eng.validate(proposal(c), ctx_for(c))
    assert not v.approved and any("LONG" in r for r in v.reasons)


def test_debit_spread_structure(eng):
    c = spread_candidate()
    assert eng.validate(proposal(c), ctx_for(c)).approved
    bad = spread_candidate()
    bad.legs = list(reversed([OptionLeg(l.symbol, "sell" if l.side == "buy" else "buy", 1, l.contract) for l in bad.legs]))
    v = eng.validate(proposal(bad), ctx_for(bad))   # short the nearer strike = credit spread -> rejected
    assert not v.approved


def test_ai_mismatch_and_skip_rejected(eng):
    c = stock_candidate()
    d = AIDecision(candidate_id="c0", ticker="BBB", instrument="stock", instrument_symbol="BBB", action="enter",
                   direction="long", strategy="stock_momentum", entry_price=100, stop_loss=98, profit_target=104,
                   position_size=10, confidence=0.5, estimated_risk_usd=20, estimated_costs_usd=1, explanation="x")
    p = proposal(c, 10)
    p.ai_decision = d
    v = eng.validate(p, ctx_for(c))
    assert not v.approved and any("does not match" in r for r in v.reasons)
    p.ai_decision = d.model_copy(update={"ticker": "AAA", "instrument_symbol": "AAA", "action": "skip"})
    assert not eng.validate(p, ctx_for(c)).approved


def test_live_mode_uses_tighter_limits_and_requires_authorization(make_cfg):
    eng = RiskEngine(make_cfg(trading={"mode": "live"}, live={"enabled": True}))
    c = stock_candidate()
    v = eng.validate(proposal(c), ctx_for(c, approved_strategies=set(ALL)))
    assert not v.approved and any("live trading not authorized" in r for r in v.reasons)
    p = proposal(c)
    v = eng.validate(p, ctx_for(c, live_authorized=True))
    assert v.approved, v.reasons
    assert p.risk_usd <= 100_000 * 0.0025
