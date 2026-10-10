"""Bar-level trade simulation for a single symbol.

Look-ahead / leakage controls:
* Indicators are causal (tested), and the strategy receives ``feats.iloc[:i+1]`` — data through bar i.
* A signal on bar i is filled at the OPEN of bar i+1 (never at bar i's close).
* Stop/target are re-anchored to the actual fill price.
* If a bar's range touches both stop and target, the STOP is assumed to fill first (pessimistic).
* Gaps through the stop fill at the open (worse than the stop), plus slippage.

OPTION PRICES ARE MODELED, not historical: Black-Scholes with IV = realized vol x ``option_iv_premium``,
held constant over the trade, plus modeled spreads/slippage/fees. Results for option strategies are
labeled ``option_pricing: modeled_black_scholes`` and should be re-checked against real option
history before being trusted.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..analysis.options_math import bs, strike_for_delta
from ..models import Direction, InstrumentType
from ..strategies.base import Strategy, UnderlyingSignal
from .costs import CostModel


@dataclass
class BTTrade:
    symbol: str
    strategy: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    instrument: str
    entry_price: float       # per unit actually paid (incl. costs)
    exit_price: float        # per unit actually received (incl. costs)
    risk_per_unit: float     # $ at risk per unit (stock: stop distance+slippage; options: full premium)
    pnl_per_unit: float
    multiplier: int
    reason: str
    bars_held: int

    @property
    def r_multiple(self) -> float:
        return self.pnl_per_unit / self.risk_per_unit if self.risk_per_unit > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"], d["exit_time"] = str(self.entry_time), str(self.exit_time)
        d["r_multiple"] = round(self.r_multiple, 4)
        return d


def precompute_signals(strategy: Strategy, feats: pd.DataFrame, warmup: int = 60) -> dict[int, UnderlyingSignal]:
    """Evaluate the strategy at every bar using only data through that bar."""
    out: dict[int, UnderlyingSignal] = {}
    for i in range(warmup, len(feats) - 1):
        sig = strategy.evaluate(feats.iloc[: i + 1])
        if sig is not None:
            out[i] = sig
    return out


def _bars_per_year(strategy: Strategy) -> float:
    return 252 * 78 if strategy.timeframe == "5m" else 252


def _years_per_bar(strategy: Strategy) -> float:
    # calendar time elapsed per bar (for option theta)
    return (5 / (60 * 24 * 365)) if strategy.timeframe == "5m" else (1.4 / 365)


def simulate_symbol(symbol: str, strategy: Strategy, feats: pd.DataFrame, signals: dict[int, UnderlyingSignal],
                    costs: CostModel, iv_premium: float = 1.1, start_idx: int = 0, end_idx: int | None = None
                    ) -> list[BTTrade]:
    """Walk forward through bars, entering on signals (entry bar in [start_idx, end_idx))."""
    end_idx = len(feats) if end_idx is None else end_idx
    o, h, l, c = (feats[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = feats.index
    trades: list[BTTrade] = []
    i = max(start_idx - 1, 0)
    while i < min(end_idx, len(feats)) - 1:
        sig = signals.get(i)
        if sig is None or i + 1 < start_idx:
            i += 1
            continue
        e = i + 1  # fill on next bar's open
        if strategy.instrument == InstrumentType.STOCK:
            t = _sim_stock(symbol, strategy, sig, e, o, h, l, c, idx, costs)
        else:
            t = _sim_option(symbol, strategy, sig, e, feats, o, h, l, c, idx, costs, iv_premium)
        if t is None:
            i += 1
            continue
        trades.append(t)
        i = idx.get_loc(t.exit_time) + 1  # no overlapping trades per symbol; no re-entry on exit bar
        # intraday scalps: never carry overnight
    return trades


def _sim_stock(symbol, strategy, sig, e, o, h, l, c, idx, costs: CostModel) -> BTTrade | None:
    if sig.direction != Direction.LONG:
        return None
    raw_fill = o[e]
    stop_d, tgt_d = sig.entry - sig.stop, sig.target - sig.entry
    stop, target = raw_fill - stop_d, raw_fill + tgt_d
    if stop <= 0:
        return None
    entry = costs.stock_buy(raw_fill)
    risk = entry - costs.stock_sell(stop) + raw_fill * costs.stock_slippage_bps / 10_000
    n = len(c)
    for j in range(e, n):
        held = j - e
        if j > e and o[j] <= stop:                        # gap through stop
            return _mk(symbol, strategy, idx, e, j, sig, entry, costs.stock_sell(o[j]), risk, 1, "stop_gap", held)
        if l[j] <= stop:                                   # stop first if both touched
            return _mk(symbol, strategy, idx, e, j, sig, entry, costs.stock_sell(stop), risk, 1, "stop", held)
        if h[j] >= target:
            px = max(target, o[j]) if j > e else target
            return _mk(symbol, strategy, idx, e, j, sig, entry, costs.stock_sell(px), risk, 1, "target", held)
        if held + 1 >= sig.max_hold_bars or _session_end(strategy, idx, j):
            return _mk(symbol, strategy, idx, e, j, sig, entry, costs.stock_sell(c[j]), risk, 1, "time", held)
    return None  # ran out of data: open trade not counted


def _session_end(strategy: Strategy, idx, j: int) -> bool:
    if not strategy.intraday_only:
        return False
    if j + 1 >= len(idx):
        return True
    return idx[j + 1].date() != idx[j].date()


def _mk(symbol, strategy, idx, e, j, sig, entry, exit_, risk, mult, reason, held) -> BTTrade:
    return BTTrade(symbol, strategy.name, idx[e], idx[j], sig.direction.value, strategy.instrument.value,
                   round(entry, 4), round(exit_, 4), max(risk, 1e-9), exit_ - entry, mult, reason, held)


def _sim_option(symbol, strategy, sig, e, feats, o, h, l, c, idx, costs: CostModel, iv_premium) -> BTTrade | None:
    rv = feats["rv20"].iloc[e - 1] if "rv20" in feats.columns else np.nan
    if pd.isna(rv) or rv <= 0:
        r = np.log(feats["close"].iloc[max(0, e - 60):e]).diff().dropna()
        rv = float(r.std() * math.sqrt(_bars_per_year(strategy))) if len(r) > 10 else 0.3
    iv = max(0.08, float(rv) * iv_premium)
    kind = "call" if sig.direction == Direction.LONG else "put"
    spot0 = o[e]
    step = 1.0 if spot0 < 200 else 5.0
    dte_days = (strategy.option_dte[0] + strategy.option_dte[1]) / 2
    if strategy.intraday_only:
        dte_days = max(strategy.option_dte[0], 2)
    T0 = dte_days / 365
    k1 = strike_for_delta(spot0, T0, iv, strategy.option_target_delta, kind, step=step)
    k2 = None
    if strategy.instrument == InstrumentType.DEBIT_SPREAD:
        atr = float(feats["atr"].iloc[e - 1]) if "atr" in feats.columns else spot0 * 0.02
        w = max(step, round(atr * strategy.spread_width_atr / step) * step)
        k2 = k1 + w if kind == "call" else k1 - w

    def value(spot, T, buy_side: bool) -> float:
        """Executable value: longs at (mid - costs) when selling, (mid + costs) when buying."""
        m1 = bs(spot, k1, max(T, 1e-6), iv, kind=kind).price
        if k2 is None:
            return costs.option_buy(m1) if buy_side else costs.option_sell(m1)
        m2 = bs(spot, k2, max(T, 1e-6), iv, kind=kind).price
        if buy_side:   # pay ask on long, receive bid on short
            return costs.option_buy(m1) - costs.option_sell(m2)
        return costs.option_sell(m1) - costs.option_buy(m2)

    entry = value(spot0, T0, True)
    if entry <= 0.05:
        return None
    max_val = abs(k2 - k1) if k2 is not None else None
    stop_d, tgt_d = abs(sig.entry - sig.stop), abs(sig.target - sig.entry)
    sgn = 1 if sig.direction == Direction.LONG else -1
    u_stop, u_tgt = spot0 - sgn * stop_d, spot0 + sgn * tgt_d
    ypb = _years_per_bar(strategy)
    n = len(c)
    for j in range(e, n):
        held = j - e
        T = T0 - (held + 1) * ypb
        adverse = l[j] if sgn > 0 else h[j]
        favorable = h[j] if sgn > 0 else l[j]
        gap_open = o[j]
        if j > e and ((gap_open <= u_stop) if sgn > 0 else (gap_open >= u_stop)):
            return _mk(symbol, strategy, idx, e, j, sig, entry, _stop_fill(value(gap_open, T, False), costs), entry, 100, "stop_gap", held)
        if (adverse <= u_stop) if sgn > 0 else (adverse >= u_stop):
            return _mk(symbol, strategy, idx, e, j, sig, entry, _stop_fill(value(u_stop, T, False), costs), entry, 100, "stop", held)
        if (favorable >= u_tgt) if sgn > 0 else (favorable <= u_tgt):
            return _mk(symbol, strategy, idx, e, j, sig, entry, value(u_tgt, T, False), entry, 100, "target", held)
        v_close = value(c[j], T, False)
        if v_close <= entry * 0.5:
            return _mk(symbol, strategy, idx, e, j, sig, entry, _stop_fill(v_close, costs), entry, 100, "premium_stop", held)
        if max_val is not None and v_close >= entry + 0.75 * (max_val - entry):
            return _mk(symbol, strategy, idx, e, j, sig, entry, v_close, entry, 100, "spread_target", held)
        if held + 1 >= sig.max_hold_bars or T * 365 <= 1 or _session_end(strategy, idx, j):
            return _mk(symbol, strategy, idx, e, j, sig, entry, v_close, entry, 100, "time", held)
    return None


def _stop_fill(v: float, costs: CostModel) -> float:
    """Stops on options fill worse than modeled: extra slippage equal to the modeled half-spread."""
    return max(0.0, v * (1 - costs.option_half_spread_pct / 100))


def param_combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
