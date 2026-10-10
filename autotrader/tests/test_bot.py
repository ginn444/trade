"""End-to-end bot cycles: synthetic data + simulated broker + fake Claude (all offline)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from autotrader.ai.claude_engine import ClaudeDecisionEngine
from autotrader.broker.simulated import SimulatedBroker
from autotrader.data.synthetic import SyntheticMarketData
from autotrader.db import Database
from autotrader.engine.bot import TradingBot
from autotrader.health import HealthMonitor
from autotrader.safety import KillSwitch
from autotrader.strategies import registry

from conftest import ALWAYS_LONG, FakeClaude, write_approvals

NY = ZoneInfo("America/New_York")
SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def make_bot(make_cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(registry.STRATEGIES, ALWAYS_LONG.name, ALWAYS_LONG)
    import autotrader.engine.bot as botmod
    monkeypatch.setitem(botmod.STRATEGIES, ALWAYS_LONG.name, ALWAYS_LONG)

    def _make(ai_client=None, approve=True, ai_enabled=True, policy="skip", **cfg_over):
        cfg = make_cfg(strategies={"enabled": [ALWAYS_LONG.name], "validation_file": str(tmp_path / "approvals.json")},
                       universe={"stocks": SYMS, "options_enabled": False, "min_avg_dollar_volume": 1e6},
                       ai={"enabled": ai_enabled, "on_ai_unavailable": policy}, **cfg_over)
        if approve:
            write_approvals(tmp_path / "approvals.json", [ALWAYS_LONG.name])
        state = {"now": datetime(2026, 9, 23, 10, 0, tzinfo=NY).astimezone(timezone.utc)}
        clock = lambda: state["now"]  # noqa: E731
        data = SyntheticMarketData(seed=1, clock=clock)
        broker = SimulatedBroker(data, 100_000, clock=clock)
        db = Database(cfg.database_url, "mock", clock=clock)
        ai = ClaudeDecisionEngine(cfg.ai, None, client=ai_client)
        bot = TradingBot(cfg, broker, data, db, ai, KillSwitch(cfg.state_dir), HealthMonitor(cfg.broker_health),
                         clock=clock)
        return SimpleNamespace(bot=bot, db=db, broker=broker, data=data, state=state, cfg=cfg, ai=ai_client)
    return _make


def test_autonomous_entries_capped_at_three_and_ai_explanations_recorded(make_bot):
    e = make_bot(ai_client=FakeClaude())
    r = e.bot.cycle()
    assert r.submitted == 3                       # max simultaneous positions
    trades = e.db.open_trades()
    assert len(trades) == 3 and all(t.decided_by == "claude" for t in trades)
    assert all(t.risk_usd <= 500 for t in trades)
    assert e.ai.calls, "Claude was consulted"
    sigs = e.db.recent_signals()
    assert any(s.outcome == "approved" and s.ai_explanation == "fake review" for s in sigs)
    # next cycle: no averaging down / no new positions beyond the cap
    e.state["now"] += timedelta(minutes=1)
    assert e.bot.cycle().submitted == 0
    assert len(e.db.open_trades()) == 3


def test_ai_skip_means_no_trade(make_bot):
    e = make_bot(ai_client=FakeClaude(action="skip"))
    assert e.bot.cycle().submitted == 0
    assert all(s.outcome == "ai_skip" for s in e.db.recent_signals())


def test_ai_cannot_bypass_risk_engine(make_bot):
    # Claude tries to widen the stop massively and oversize: the risk engine still sizes within budget
    def greedy(d):
        return d.model_copy(update={"stop_loss": d.entry_price * 0.5, "position_size": 1_000_000})
    e = make_bot(ai_client=FakeClaude(mutate=greedy))
    e.bot.cycle()
    for t in e.db.open_trades():
        assert t.risk_usd <= 100_000 * 0.005 + 1e-6
    # and a hallucinated entry price far from the market is rejected outright
    e2 = make_bot(ai_client=FakeClaude(mutate=lambda d: d.model_copy(update={"entry_price": d.entry_price * 1.2,
                                                                                  "profit_target": d.entry_price * 1.5})))
    assert e2.bot.cycle().submitted == 0


def test_ai_unavailable_policy_skip(make_bot):
    import anthropic, httpx
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    e = make_bot(ai_client=FakeClaude(fail=err))
    assert e.bot.cycle().submitted == 0
    assert any(a.event == "ai_unavailable" for a in e.db.recent_audit())


def test_deterministic_mode_without_ai(make_bot):
    e = make_bot(ai_enabled=False)
    assert e.bot.cycle().submitted == 3
    assert all(t.decided_by == "deterministic" for t in e.db.open_trades())


def test_unvalidated_strategies_do_not_trade(make_bot):
    e = make_bot(ai_client=FakeClaude(), approve=False)
    assert e.bot.cycle().submitted == 0


def test_synthetic_validation_never_authorizes_paper(make_bot):
    e = make_bot(ai_client=FakeClaude())
    e.bot.cfg = e.cfg.model_copy(update={"trading": e.cfg.trading.model_copy(update={"mode": "paper"})})
    assert e.bot.approved_strategies() == set()


def test_kill_switch_blocks_and_flattens(make_bot):
    e = make_bot(ai_enabled=False)
    e.bot.cycle()
    assert len(e.db.open_trades()) == 3
    e.bot.kill.engage("test", flatten=True)
    e.state["now"] += timedelta(minutes=1)
    r = e.bot.cycle()
    assert r.status == "killed" and r.exits == 3
    e.broker.tick()
    e.state["now"] += timedelta(minutes=1)
    e.bot.cycle()
    assert all(p.qty == 0 for p in e.broker.positions.values())
    assert e.db.open_trades() == []


def test_no_entries_after_daily_loss_limit(make_bot):
    e = make_bot(ai_enabled=False)
    e.broker.last_equity = 103_000          # equity 100k vs 103k start => -2.9%
    r = e.bot.cycle()
    assert r.submitted == 0 and "daily_loss_limit" in r.notes
    assert any(a.event == "daily_loss_limit" for a in e.db.recent_audit())


def test_no_entries_when_broker_unreliable(make_bot):
    e = make_bot(ai_enabled=False)
    e.broker.fail_next = 50
    for _ in range(6):   # flapping broker: trips on errors-in-window (5 / 300s)
        e.bot.cycle()
        e.state["now"] += timedelta(seconds=30)
    assert e.bot.health.healthy is False
    e.broker.fail_next = 0
    r = e.bot.cycle()
    assert r.submitted == 0 and "broker_unhealthy" in r.notes


def test_no_trading_on_stale_data(make_bot):
    e = make_bot(ai_enabled=False)
    e.data.stale_symbols.update(SYMS)
    assert e.bot.cycle().submitted == 0
    assert any(a.event == "stale_quote_skip" for a in e.db.recent_audit())


def test_no_new_entries_after_cutoff_and_eod_report(make_bot):
    e = make_bot(ai_enabled=False)
    e.state["now"] = datetime(2026, 9, 23, 15, 40, tzinfo=NY).astimezone(timezone.utc)
    r = e.bot.cycle()
    assert r.submitted == 0 and "past_cutoff" in r.notes
    e.state["now"] = datetime(2026, 9, 23, 16, 10, tzinfo=NY).astimezone(timezone.utc)
    assert e.bot.cycle().status == "market_closed"
    rep = e.db.reports(1)[0].report
    assert rep["date"] == "2026-09-23" and "SIMULATED" in rep["data_source_note"]


def test_restart_reconciles_with_persisted_sim_broker(make_bot, tmp_path):
    e = make_bot(ai_enabled=False)
    e.bot.broker_state_path = tmp_path / "sim.json"
    e.bot.cycle()
    b2 = SimulatedBroker(e.data, 0, clock=e.bot.clock)
    b2.load(tmp_path / "sim.json")
    e.bot.broker = b2
    e.bot.orders.broker = b2
    rep = e.bot.orders.reconcile()
    assert rep["ok"], rep
