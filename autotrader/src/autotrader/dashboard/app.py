"""Monitoring dashboard (FastAPI). Reads the bot's database; the only write is the kill switch.

Binds to 127.0.0.1 by default. Write endpoints require the DASHBOARD_TOKEN header and fail closed
when no token is configured.
"""
from __future__ import annotations

import hmac
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..backtest.validation import load_approvals
from ..config import AppConfig, Credentials
from ..db import Database, row_to_dict
from ..reporting.daily import strategy_stats
from ..safety import KillSwitch

STATIC = Path(__file__).parent / "static"


class KillRequest(BaseModel):
    reason: str = "dashboard emergency stop"
    flatten: bool = False


def create_app(cfg: AppConfig, creds: Credentials, db: Database | None = None) -> FastAPI:
    db = db or Database(cfg.database_url, cfg.trading.mode.value)
    kill = KillSwitch(cfg.state_dir)
    tz = ZoneInfo(cfg.trading.timezone)
    app = FastAPI(title="autotrader dashboard", docs_url=None, redoc_url=None)

    def _auth(token: str | None) -> None:
        expected = creds.dashboard_token
        if not expected or not token or not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid or missing dashboard token")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        hist = db.equity_history(1)
        snap = hist[-1] if hist else None
        first = db.equity_history(100000)[0] if hist else None
        r = cfg.effective_risk()
        open_trades = db.open_trades()
        risk_in_use = sum(t.risk_usd or 0 for t in open_trades)
        eq = snap.equity if snap else None
        today = datetime.now(tz).date().isoformat()
        realized_all = sum(t.realized_pnl or 0 for t in db.closed_trades())
        loss_limit_usd = (snap.day_start_equity * r.max_daily_loss_pct / 100) if snap else None
        return {
            "mode": cfg.trading.mode.value,
            "simulated": cfg.trading.mode.value == "mock",
            "live": cfg.trading.mode.value == "live",
            "bot_status": db.get_state("bot_status"),
            "kill_switch": kill.info(),
            "account": None if snap is None else {
                "ts": row_to_dict(snap)["ts"], "equity": snap.equity, "buying_power": snap.buying_power,
                "cash": snap.cash, "day_start_equity": snap.day_start_equity, "daily_pnl": snap.daily_pnl,
                "daily_pnl_pct": (snap.daily_pnl / snap.day_start_equity * 100) if snap.day_start_equity else None,
                "cumulative_pnl": (snap.equity - first.equity) if first else None,
            },
            "realized_pnl_total": round(realized_all, 2),
            "risk": {
                "open_positions": len(open_trades), "max_open_positions": r.max_open_positions,
                "risk_in_use_usd": round(risk_in_use, 2),
                "risk_capacity_usd": round((eq or 0) * r.max_risk_per_trade_pct / 100 * r.max_open_positions, 2),
                "daily_loss_used_usd": round(max(0.0, -(snap.daily_pnl if snap else 0)), 2),
                "daily_loss_limit_usd": round(loss_limit_usd, 2) if loss_limit_usd else None,
                "max_risk_per_trade_pct": r.max_risk_per_trade_pct, "max_daily_loss_pct": r.max_daily_loss_pct,
                "loss_limit_hit_today": bool(db.get_state(f"loss_limit_hit:{today}")),
            },
        }

    @app.get("/api/positions")
    def positions():
        return [row_to_dict(t) for t in db.open_trades()]

    @app.get("/api/trades")
    def trades(limit: int = 100):
        return [row_to_dict(t) for t in db.closed_trades(limit=min(limit, 1000))]

    @app.get("/api/signals")
    def signals(limit: int = 50):
        return [row_to_dict(s) for s in db.recent_signals(min(limit, 500))]

    @app.get("/api/equity")
    def equity(limit: int = 2000):
        return [{"ts": row_to_dict(e)["ts"], "equity": e.equity, "daily_pnl": e.daily_pnl}
                for e in db.equity_history(min(limit, 20000))]

    @app.get("/api/strategies")
    def strategies():
        stats = strategy_stats(db.closed_trades())
        approvals = load_approvals(cfg.resolve_path(cfg.strategies.validation_file))
        names = sorted(set(cfg.strategies.enabled) | set(stats) | set(approvals))
        return [{"strategy": n, "enabled": n in cfg.strategies.enabled, "live_stats": stats.get(n),
                 "validation": approvals.get(n)} for n in names]

    @app.get("/api/audit")
    def audit(limit: int = 100):
        return [row_to_dict(a) for a in db.recent_audit(min(limit, 1000))]

    @app.get("/api/reports")
    def reports(limit: int = 10):
        return [r.report for r in db.reports(min(limit, 100))]

    @app.post("/api/kill")
    def engage_kill(req: KillRequest, x_dashboard_token: str | None = Header(default=None)):
        _auth(x_dashboard_token)
        kill.engage(req.reason, actor="dashboard", flatten=req.flatten)
        db.audit("kill_switch_engaged", {"reason": req.reason, "flatten": req.flatten, "actor": "dashboard"}, "WARNING")
        return {"ok": True, "kill_switch": kill.info()}

    @app.post("/api/resume")
    def resume(x_dashboard_token: str | None = Header(default=None)):
        _auth(x_dashboard_token)
        kill.release(actor="dashboard")
        db.audit("kill_switch_released", {"actor": "dashboard"}, "WARNING")
        return {"ok": True}

    @app.get("/api/health")
    def health():
        return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

    return app
