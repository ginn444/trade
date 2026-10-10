"""Strategy research harness: in-sample/holdout split, walk-forward optimisation, benchmarks.

Output is a JSON-serialisable report per strategy. Acceptance against predefined criteria is
decided in ``validation.py``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..strategies.base import Strategy, daily_features, intraday_features
from .costs import CostModel
from .metrics import benchmark_buy_hold, summarize
from .portfolio import run_portfolio
from .simulator import BTTrade, param_combos, precompute_signals, simulate_symbol

log = logging.getLogger(__name__)


@dataclass
class SymbolData:
    symbol: str
    feats: pd.DataFrame


def prepare(strategy: Strategy, bars_by_symbol: dict[str, pd.DataFrame]) -> list[SymbolData]:
    fn = intraday_features if strategy.timeframe == "5m" else daily_features
    return [SymbolData(s, fn(b)) for s, b in bars_by_symbol.items() if len(b) > 100]


class _SignalCache:
    def __init__(self, strategy: Strategy, data: list[SymbolData], warmup: int):
        self.strategy, self.data, self.warmup = strategy, data, warmup
        self._c: dict[tuple, dict] = {}

    def get(self, params: dict, sd: SymbolData) -> dict:
        key = (tuple(sorted(params.items())), sd.symbol)
        if key not in self._c:
            self._c[key] = precompute_signals(self.strategy.with_params(**params), sd.feats, self.warmup)
        return self._c[key]


def _trades_between(strategy, cache, params, data, costs, iv_prem, t0, t1) -> list[BTTrade]:
    out: list[BTTrade] = []
    strat = strategy.with_params(**params)
    for sd in data:
        idx = sd.feats.index
        s = int(idx.searchsorted(t0, side="left"))
        e = int(idx.searchsorted(t1, side="left"))
        if e - s < 2:
            continue
        out.extend(simulate_symbol(sd.symbol, strat, sd.feats, cache.get(params, sd), costs, iv_prem, s, e))
    return out


def _score(trades: list[BTTrade]) -> float:
    if len(trades) < 5:
        return -math.inf
    r = np.array([t.r_multiple for t in trades])
    return float(r.mean() * math.sqrt(len(r)))


def research_strategy(strategy: Strategy, bars_by_symbol: dict[str, pd.DataFrame], cfg: AppConfig,
                      data_source: str) -> dict:
    bt = cfg.backtest
    risk = cfg.risk
    costs = CostModel.from_cfg(bt)
    data = prepare(strategy, bars_by_symbol)
    if not data:
        return {"strategy": strategy.name, "error": "no usable data"}
    intraday = strategy.timeframe == "5m"
    warmup = 30 if intraday else 200
    cache = _SignalCache(strategy, data, warmup)
    timeline = pd.DatetimeIndex(sorted(set().union(*[set(sd.feats.index) for sd in data])))
    if len(timeline) <= warmup + 10:
        return {"strategy": strategy.name, "error": "insufficient history"}
    closes = {sd.symbol: sd.feats["close"] for sd in data}
    port = lambda tr: run_portfolio(tr, bt.initial_equity, risk.max_risk_per_trade_pct,  # noqa: E731
                                    risk.max_open_positions, risk.max_position_notional_pct)
    default = dict(strategy.params)

    # ---- 1. in-sample / holdout split with DEFAULT params (no fitting) --------------
    start = timeline[warmup]
    split = timeline[warmup + int((len(timeline) - warmup) * 0.7)]
    end = timeline[-1] + pd.Timedelta(seconds=1)
    is_res = port(_trades_between(strategy, cache, {}, data, costs, bt.option_iv_premium, start, split))
    ho_res = port(_trades_between(strategy, cache, {}, data, costs, bt.option_iv_premium, split, end))
    in_sample = summarize(is_res.trades, is_res.equity, bt.initial_equity)
    holdout = summarize(ho_res.trades, ho_res.equity, bt.initial_equity)

    # ---- 2. walk-forward optimisation over the parameter grid ---------------------------
    train_n = bt.walk_forward_train_bars_intraday if intraday else bt.walk_forward_train_bars
    test_n = bt.walk_forward_test_bars_intraday if intraday else bt.walk_forward_test_bars
    combos = param_combos(strategy.param_grid)
    windows = []
    oos_trades: list[BTTrade] = []
    pos = warmup
    while pos + train_n + test_n <= len(timeline):
        tr0, tr1 = timeline[pos], timeline[pos + train_n]
        te1 = timeline[pos + train_n + test_n] if pos + train_n + test_n < len(timeline) else end
        best, best_score = {}, -math.inf
        for p in combos:
            sc = _score(_trades_between(strategy, cache, p, data, costs, bt.option_iv_premium, tr0, tr1))
            if sc > best_score:
                best, best_score = p, sc
        test_tr = _trades_between(strategy, cache, best, data, costs, bt.option_iv_premium, tr1, te1)
        oos_trades.extend(test_tr)
        windows.append({"train_start": str(tr0), "test_start": str(tr1), "test_end": str(te1),
                        "params": best, "train_score": None if best_score == -math.inf else round(best_score, 3),
                        "test_trades": len(test_tr),
                        "test_r_sum": round(sum(t.r_multiple for t in test_tr), 3)})
        pos += test_n
    wf = port(oos_trades) if oos_trades else None
    wf_metrics = summarize(wf.trades, wf.equity, bt.initial_equity) if wf else summarize([], pd.Series(dtype=float), bt.initial_equity)
    traded_windows = [w for w in windows if w["test_trades"] > 0]
    pos_frac = (sum(1 for w in traded_windows if w["test_r_sum"] > 0) / len(traded_windows)) if traded_windows else 0.0

    oos_start = pd.Timestamp(windows[0]["test_start"]) if windows else split
    bench = benchmark_buy_hold(closes, oos_start, timeline[-1], bt.initial_equity)
    return {
        "strategy": strategy.name,
        "description": strategy.description,
        "instrument": strategy.instrument.value,
        "timeframe": strategy.timeframe,
        "data_source": data_source,
        "option_pricing": "modeled_black_scholes" if strategy.instrument.value != "stock" else "n/a",
        "symbols": [sd.symbol for sd in data],
        "period": {"start": str(timeline[0]), "end": str(timeline[-1]), "bars": len(timeline)},
        "default_params": default,
        "in_sample": in_sample,
        "holdout": holdout,
        "walk_forward": {**wf_metrics, "windows": len(windows), "positive_windows_frac": round(pos_frac, 3),
                         "window_detail": windows},
        "benchmark_buy_hold": bench,
        "cost_model": costs.__dict__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_trades": (wf.trades[-20:] if wf else []),
    }
