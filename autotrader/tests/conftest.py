"""Shared fixtures. Everything here is offline: synthetic data, simulated broker, fake Claude client."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from autotrader.config import Credentials, load_config
from autotrader.models import (AIDecision, AIReview, AccountState, Candidate, Direction, InstrumentType,
                               OptionContract, OptionLeg, Quote, Regime, TradeProposal)
from autotrader.strategies.base import Strategy, UnderlyingSignal

NY = ZoneInfo("America/New_York")
# A Wednesday, 10:30 New York time
NOW = datetime(2026, 9, 23, 10, 30, tzinfo=NY).astimezone(timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for k in ["AUTOTRADER_CONFIG", "AUTOTRADER_MODE", "ANTHROPIC_API_KEY", "ALPACA_PAPER_API_KEY",
              "ALPACA_PAPER_SECRET_KEY", "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY", "DASHBOARD_TOKEN"]:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def make_cfg(tmp_path):
    def _make(**over):
        base = {
            "trading": {"mode": "mock"},
            "storage": {"database_url": f"sqlite:///{tmp_path}/t.db", "state_dir": str(tmp_path),
                        "reports_dir": str(tmp_path / "reports"), "log_dir": str(tmp_path / "logs")},
            "strategies": {"validation_file": str(tmp_path / "approvals.json")},
            "live": {"authorization_file": str(tmp_path / "live_auth.json")},
        }
        for k, v in over.items():
            base.setdefault(k, {}).update(v) if isinstance(v, dict) else base.__setitem__(k, v)
        return load_config(overrides=base)
    return _make


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


def quote(sym, bid, ask, now=NOW, age_s=1):
    return Quote(sym, bid, ask, now - timedelta(seconds=age_s), 100, 100, "TEST")


def stock_candidate(sym="AAA", entry=100.0, stop=98.0, target=104.0, strategy="stock_momentum",
                    direction=Direction.LONG, now=NOW):
    return Candidate(strategy, sym, direction, InstrumentType.STOCK, entry, stop, target, entry, Regime.BULLISH,
                     {"atr": 1.0}, created_at=now, rationale="test")


def option_contract(sym="AAA", kind="call", strike=100.0, exp=date(2026, 10, 16), bid=2.0, ask=2.1,
                    delta=0.5, oi=1000, now=NOW):
    from autotrader.data.alpaca_data import build_occ
    occ = build_occ(sym, exp, kind, strike)
    return OptionContract(occ, sym, kind, strike, exp, quote(occ, bid, ask, now), 0.3,
                          delta if kind == "call" else -delta, 0.05, -0.05, 0.1, oi, 200)


def option_candidate(direction=Direction.LONG, now=NOW, strategy="option_momentum"):
    kind = "call" if direction == Direction.LONG else "put"
    c = option_contract(kind=kind, now=now)
    return Candidate(strategy, "AAA", direction, InstrumentType.OPTION, 2.05, 1.2, 3.6, 100.0, Regime.BULLISH,
                     {}, [OptionLeg(c.symbol, "buy", 1, c)], created_at=now)


def spread_candidate(now=NOW):
    lo = option_contract(strike=100, bid=3.0, ask=3.1, now=now)
    hi = option_contract(strike=105, bid=1.0, ask=1.05, delta=0.3, now=now)
    return Candidate("option_debit_spread", "AAA", Direction.LONG, InstrumentType.DEBIT_SPREAD, 2.0, 1.0, 3.6, 100.0,
                     Regime.BULLISH, {}, [OptionLeg(lo.symbol, "buy", 1, lo), OptionLeg(hi.symbol, "sell", 1, hi)],
                     created_at=now)


def proposal(c, qty=0):
    return TradeProposal(c, qty, c.entry_price, c.stop_price, c.target_price, 0, 0)


def account(equity=100_000.0, last=100_000.0, bp=200_000.0):
    return AccountState(equity=equity, buying_power=bp, cash=equity, last_equity=last)


def write_approvals(path, names, source="SYNTHETIC"):
    path.write_text(json.dumps({n: {"approved": True, "data_source": source} for n in names}))


def always_long_rule(feats, p):
    c = float(feats["close"].iloc[-1])
    return UnderlyingSignal(Direction.LONG, c, c * 0.98, c * 1.06, 5, Regime.BULLISH, "test signal", {"atr": c * 0.01})


ALWAYS_LONG = Strategy("test_always_long", always_long_rule, InstrumentType.STOCK, "1d", {"min_rr": 1.5})


class FakeClaude:
    """Stands in for anthropic.Anthropic: echoes candidates back as structured decisions."""

    def __init__(self, action="enter", mutate=None, fail=None, stop_reason="end_turn"):
        self.action, self.mutate, self.fail, self.stop_reason = action, mutate, fail, stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        payload = json.loads(kw["messages"][0]["content"].split("\n\n", 1)[1])
        decisions = []
        for c in payload["candidates"]:
            d = AIDecision(candidate_id=c["candidate_id"], ticker=c["ticker"], instrument=c["instrument"],
                           instrument_symbol=c["instrument_symbol"], action=self.action, direction=c["direction"],
                           strategy=c["strategy"], entry_price=c["proposed_entry"], stop_loss=c["proposed_stop"],
                           profit_target=c["proposed_target"], position_size=c["max_quantity"], confidence=0.6,
                           estimated_risk_usd=100, estimated_costs_usd=2, explanation="fake review", key_risks=[])
            if self.mutate:
                d = self.mutate(d)
            decisions.append(d)
        return SimpleNamespace(stop_reason=self.stop_reason, model="fake-model",
                               parsed_output=AIReview(market_summary="fake", decisions=decisions))


@pytest.fixture
def creds():
    return Credentials({})
