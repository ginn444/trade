"""Daily performance report (built only from recorded trades and broker account state)."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from ..config import AppConfig
from ..db import Database


def strategy_stats(trades) -> dict:
    by = defaultdict(list)
    for t in trades:
        if t.realized_pnl is not None:
            by[t.strategy].append(t.realized_pnl)
    out = {}
    for s, pnl in by.items():
        wins = [p for p in pnl if p > 0]
        losses = [-p for p in pnl if p <= 0]
        out[s] = {"trades": len(pnl), "net_pnl": round(sum(pnl), 2),
                  "win_rate": round(len(wins) / len(pnl), 3) if pnl else 0,
                  "profit_factor": round(sum(wins) / sum(losses), 3) if sum(losses) > 0 else None}
    return out


def build_daily_report(db: Database, cfg: AppConfig, day: date, account, day_start) -> dict:
    tz = ZoneInfo(cfg.trading.timezone)
    since = datetime.combine(day, time(0, 0), tzinfo=tz)
    closed = [t for t in db.closed_trades(since=since)]
    realized = sum(t.realized_pnl or 0 for t in closed)
    signals = db.recent_signals(500)
    sig_today = [s for s in signals if s.ts and s.ts.replace(tzinfo=s.ts.tzinfo or ZoneInfo("UTC")) >= since]
    outcomes = defaultdict(int)
    for s in sig_today:
        outcomes[s.outcome] += 1
    equity = account.equity if account else None
    return {
        "date": day.isoformat(),
        "mode": cfg.trading.mode.value,
        "data_source_note": "SIMULATED/SYNTHETIC — not real results" if cfg.trading.mode.value == "mock" else "broker-reported",
        "start_equity": day_start,
        "end_equity": equity,
        "day_pnl": round(equity - day_start, 2) if (equity is not None and day_start) else None,
        "day_return_pct": round((equity - day_start) / day_start * 100, 3) if (equity is not None and day_start) else None,
        "realized_pnl": round(realized, 2),
        "trades_closed": len(closed),
        "open_positions": len(db.open_trades()),
        "by_strategy": strategy_stats(closed),
        "signals": dict(outcomes),
        "trades": [{"id": t.id, "strategy": t.strategy, "symbol": t.trade_symbol, "qty": t.filled_qty or t.qty,
                    "entry": t.entry_price, "exit": t.exit_price, "pnl": t.realized_pnl, "reason": t.exit_reason}
                   for t in closed],
        "loss_limit_hit": bool(db.get_state(f"loss_limit_hit:{day.isoformat()}")),
    }
