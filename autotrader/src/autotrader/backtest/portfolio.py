"""Portfolio-level replay of per-symbol trades with live-identical sizing and position limits."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .simulator import BTTrade


@dataclass
class PortfolioResult:
    trades: list[dict]               # trades actually taken, with quantity and $ pnl
    equity: pd.Series                # equity after each exit (realized), indexed by time
    initial_equity: float
    skipped_for_limits: int


def run_portfolio(trades: list[BTTrade], initial_equity: float, risk_pct: float, max_positions: int,
                  max_notional_pct: float) -> PortfolioResult:
    """Replay chronologically. Size = floor(equity * risk% / risk_per_unit_$), capped by notional."""
    events = sorted(trades, key=lambda t: (t.entry_time, t.symbol))
    equity = initial_equity
    open_pos: list[tuple[BTTrade, int]] = []
    taken: list[dict] = []
    curve = {pd.Timestamp(events[0].entry_time) if events else pd.Timestamp.utcnow(): equity}
    skipped = 0

    def close_until(ts):
        nonlocal equity, open_pos
        still = []
        for t, q in sorted(open_pos, key=lambda x: x[0].exit_time):
            if t.exit_time <= ts:
                pnl = t.pnl_per_unit * q * t.multiplier
                equity += pnl
                curve[pd.Timestamp(t.exit_time)] = equity
                taken.append({**t.to_dict(), "qty": q, "pnl": round(pnl, 2),
                              "risk_usd": round(t.risk_per_unit * q * t.multiplier, 2),
                              "equity_after": round(equity, 2)})
            else:
                still.append((t, q))
        open_pos = still

    for t in events:
        close_until(t.entry_time)
        if len(open_pos) >= max_positions or any(p.symbol == t.symbol for p, _ in open_pos):
            skipped += 1
            continue
        risk_unit = t.risk_per_unit * t.multiplier
        qty = math.floor(equity * risk_pct / 100 / risk_unit) if risk_unit > 0 else 0
        unit_cost = t.entry_price * t.multiplier
        cap = math.floor(equity * max_notional_pct / 100 / unit_cost) if unit_cost > 0 else 0
        qty = min(qty, cap)
        if qty < 1:
            skipped += 1
            continue
        open_pos.append((t, qty))
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    eq = pd.Series(curve).sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    return PortfolioResult(taken, eq, initial_equity, skipped)
