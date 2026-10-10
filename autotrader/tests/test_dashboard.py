from fastapi.testclient import TestClient

from autotrader.config import Credentials
from autotrader.dashboard.app import create_app
from autotrader.db import Database
from autotrader.safety import KillSwitch


def _client(cfg, token="tok"):
    creds = Credentials({"DASHBOARD_TOKEN": token} if token else {})
    db = Database(cfg.database_url, cfg.trading.mode.value)
    db.add_equity_snapshot(equity=100_500, buying_power=150_000, cash=90_000, day_start_equity=100_000,
                           daily_pnl=500, open_positions=0, risk_in_use_usd=0)
    return TestClient(create_app(cfg, creds, db))


def test_read_endpoints(cfg):
    c = _client(cfg)
    assert c.get("/").status_code == 200 and "Autotrader" in c.get("/").text
    s = c.get("/api/summary").json()
    assert s["mode"] == "mock" and s["simulated"] is True and s["account"]["equity"] == 100_500
    assert s["risk"]["max_open_positions"] == 3
    for path in ["/api/positions", "/api/trades", "/api/signals", "/api/equity", "/api/strategies", "/api/audit",
                 "/api/reports"]:
        assert c.get(path).status_code == 200, path


def test_kill_switch_requires_token(cfg):
    c = _client(cfg)
    assert c.post("/api/kill", json={}).status_code == 401
    assert c.post("/api/kill", json={}, headers={"X-Dashboard-Token": "wrong"}).status_code == 401
    r = c.post("/api/kill", json={"reason": "t", "flatten": True}, headers={"X-Dashboard-Token": "tok"})
    assert r.status_code == 200 and KillSwitch(cfg.state_dir).info()["flatten"] is True
    assert c.get("/api/summary").json()["kill_switch"]["reason"] == "t"
    assert c.post("/api/resume", headers={"X-Dashboard-Token": "tok"}).status_code == 200
    assert not KillSwitch(cfg.state_dir).is_engaged()


def test_write_endpoints_fail_closed_without_configured_token(cfg):
    c = _client(cfg, token=None)
    assert c.post("/api/kill", json={}, headers={"X-Dashboard-Token": ""}).status_code == 401
