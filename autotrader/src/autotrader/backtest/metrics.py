"""Performance statistics. Every number is computed from simulated trades; nothing is assumed."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak).min() * -100)


def daily_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(equity.index, utc=True)
    s = pd.Series(equity.values, index=idx)
    d = s.resample("1D").last().ffill()
    d = d[d.index.dayofweek < 5]
    return d.pct_change().dropna()


def sharpe(rets: pd.Series, periods: int = 252) -> float:
    if len(rets) < 2 or rets.std(ddof=1) == 0 or np.isnan(rets.std(ddof=1)):
        return 0.0
    return float(rets.mean() / rets.std(ddof=1) * math.sqrt(periods))


def sortino(rets: pd.Series, periods: int = 252) -> float:
    down = rets[rets < 0]
    if len(down) < 2 or down.std(ddof=1) == 0:
        return 0.0
    return float(rets.mean() / down.std(ddof=1) * math.sqrt(periods))


def summarize(trades: list[dict], equity: pd.Series, initial_equity: float) -> dict:
    n = len(trades)
    pnl = np.array([t["pnl"] for t in trades]) if n else np.array([])
    rs = np.array([t["r_multiple"] for t in trades]) if n else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    rets = daily_returns(equity)
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    years = 0.0
    if len(equity) >= 2:
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365)
    cagr = (final / initial_equity) ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    return {
        "trades": n,
        "win_rate": round(float((pnl > 0).mean()), 4) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "expectancy_usd": round(float(pnl.mean()), 2) if n else 0.0,
        "expectancy_r": round(float(rs.mean()), 4) if n else 0.0,
        "avg_win_usd": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_usd": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "total_return_pct": round((final / initial_equity - 1) * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 3),
        "sharpe": round(sharpe(rets), 3),
        "sortino": round(sortino(rets), 3),
        "final_equity": round(final, 2),
    }


def benchmark_buy_hold(closes: dict[str, pd.Series], start, end, initial_equity: float) -> dict:
    """Equal-weight buy & hold of the same universe over the same window (costs ignored: favours benchmark)."""
    frames = []
    for s in closes.values():
        w = s[(s.index >= start) & (s.index <= end)]
        if len(w) > 1:
            frames.append(w / w.iloc[0])
    if not frames:
        return {"total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0}
    eq = pd.concat(frames, axis=1).ffill().mean(axis=1) * initial_equity
    rets = eq.pct_change().dropna()
    return {"total_return_pct": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 3), "sharpe": round(sharpe(rets), 3),
            "max_drawdown_pct": round(max_drawdown_pct(eq), 3)}
