import logging
from datetime import datetime, timedelta, timezone

import pytest

from autotrader.broker.alpaca_broker import AlpacaBroker
from autotrader.broker.base import BrokerError
from autotrader.config import Credentials, MissingCredentialsError, TradingMode, load_config
from autotrader.logging_setup import RedactingFilter
from autotrader.safety import (CONFIRM_PHRASE, KillSwitch, LiveTradingNotAuthorized, assert_live_authorized,
                               write_live_authorization)

LIVE_KEYS = {"ALPACA_LIVE_API_KEY": "AKLIVE000000000000001", "ALPACA_LIVE_SECRET_KEY": "livesecret",
             "ALPACA_PAPER_API_KEY": "PKPAPER00000000000001", "ALPACA_PAPER_SECRET_KEY": "papersecret"}


def test_default_config_is_conservative_paper():
    c = load_config()
    assert c.trading.mode == TradingMode.PAPER
    assert c.risk.max_risk_per_trade_pct == 0.5
    assert c.risk.max_daily_loss_pct == 2.0
    assert c.risk.max_open_positions == 3
    assert c.risk.allow_averaging_down is False
    assert c.live.enabled is False
    assert c.strategies.require_validation is True


def test_averaging_down_cannot_be_enabled():
    with pytest.raises(Exception):
        load_config(overrides={"risk": {"allow_averaging_down": True}})


def test_env_cannot_select_live(monkeypatch):
    monkeypatch.setenv("AUTOTRADER_MODE", "live")
    with pytest.raises(ValueError):
        load_config()


def test_live_risk_is_tighter(make_cfg):
    c = make_cfg(trading={"mode": "live"}, live={"enabled": True})
    r = c.effective_risk()
    assert r.max_risk_per_trade_pct == 0.25 and r.max_daily_loss_pct == 1.0 and r.max_open_positions == 2


def test_credentials_never_fall_back_across_modes():
    paper_only = Credentials({"ALPACA_PAPER_API_KEY": "PK1", "ALPACA_PAPER_SECRET_KEY": "s"})
    assert paper_only.alpaca(TradingMode.PAPER) == ("PK1", "s")
    with pytest.raises(MissingCredentialsError):
        paper_only.alpaca(TradingMode.LIVE)
    live_only = Credentials({"ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "s"})
    with pytest.raises(MissingCredentialsError):
        live_only.alpaca(TradingMode.PAPER)
    same = Credentials({"ALPACA_PAPER_API_KEY": "K", "ALPACA_PAPER_SECRET_KEY": "s",
                        "ALPACA_LIVE_API_KEY": "K", "ALPACA_LIVE_SECRET_KEY": "s"})
    with pytest.raises(MissingCredentialsError):
        same.alpaca(TradingMode.PAPER)
    assert "secret" not in repr(paper_only)


def test_kill_switch_persists(tmp_path):
    KillSwitch(tmp_path).engage("test", flatten=True)
    k2 = KillSwitch(tmp_path)  # new instance (e.g. after restart / other process)
    assert k2.is_engaged() and k2.info()["flatten"] is True
    k2.release()
    assert not KillSwitch(tmp_path).is_engaged()


def _live_cfg(make_cfg):
    return make_cfg(trading={"mode": "live"}, live={"enabled": True})


def test_live_gate_requires_every_condition(make_cfg):
    creds = Credentials(LIVE_KEYS)
    with pytest.raises(LiveTradingNotAuthorized):
        assert_live_authorized(make_cfg(), creds)                       # not live mode
    with pytest.raises(LiveTradingNotAuthorized):
        assert_live_authorized(make_cfg(trading={"mode": "live"}), creds)  # live.enabled false
    cfg = _live_cfg(make_cfg)
    with pytest.raises(LiveTradingNotAuthorized):
        assert_live_authorized(cfg, creds)                               # no authorization file
    with pytest.raises(LiveTradingNotAuthorized):
        write_live_authorization(cfg, "op", "yes please")                # wrong phrase
    write_live_authorization(cfg, "op", CONFIRM_PHRASE)
    assert_live_authorized(cfg, creds)                                   # all good
    with pytest.raises(MissingCredentialsError):
        assert_live_authorized(cfg, Credentials({}))                     # missing live keys
    later = datetime.now(timezone.utc) + timedelta(hours=25)
    with pytest.raises(LiveTradingNotAuthorized):
        assert_live_authorized(cfg, creds, now=later)                    # expired
    changed = make_cfg(trading={"mode": "live"}, live={"enabled": True}, risk={"max_open_positions": 2})
    with pytest.raises(LiveTradingNotAuthorized):
        assert_live_authorized(changed, creds)                           # config changed after authorizing


def test_live_broker_cannot_be_built_without_authorization():
    with pytest.raises(BrokerError):
        AlpacaBroker("AKX", "s", paper=False)
    b = AlpacaBroker("PKX", "s", paper=True)
    assert b.is_live is False and b.name == "alpaca-paper"


def test_factory_live_mode_refuses_without_authorization(make_cfg):
    from autotrader.engine.factory import build_components
    with pytest.raises(LiveTradingNotAuthorized):
        build_components(_live_cfg(make_cfg), Credentials(LIVE_KEYS))


def test_factory_paper_mode_requires_paper_keys(make_cfg):
    from autotrader.engine.factory import build_components
    with pytest.raises(MissingCredentialsError):
        build_components(make_cfg(trading={"mode": "paper"}),
                         Credentials({k: v for k, v in LIVE_KEYS.items() if "LIVE" in k}))


def test_log_redaction():
    f = RedactingFilter(["supersecretvalue123"])
    rec = logging.LogRecord("x", logging.INFO, __file__, 1,
                            "key=%s api_key=abc123 sk-ant-api03-abcdefghijklmnop PKABCDEFGHIJKLMNOPQR", ("supersecretvalue123",), None)
    f.filter(rec)
    msg = rec.getMessage()
    for leaked in ["supersecretvalue123", "abc123", "sk-ant-api03", "PKABCDEFGHIJKLMNOPQR"]:
        assert leaked not in msg
