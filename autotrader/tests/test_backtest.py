import json

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest.costs import CostModel
from autotrader.backtest.metrics import max_drawdown_pct, summarize
from autotrader.backtest.portfolio import run_portfolio
from autotrader.backtest.simulator import BTTrade, precompute_signals, simulate_symbol
from autotrader.backtest.validation import evaluate, record_result, load_approvals
from autotrader.backtest.walkforward import research_strategy
from autotrader.config import AcceptanceCfg
from autotrader.data.synthetic import SyntheticMarketData
from autotrader.models import Direction, InstrumentType, Regime
from autotrader.strategies.base import Strategy, UnderlyingSignal
from autotrader.strategies.registry import STRATEGIES

ZERO = CostModel(0, 0, 0, 0, 0, 0)


def frame(opens, highs, lows, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="B", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6}, index=idx)


def signal_on(bar, stop=98.0, target=104.0, hold=10):
    seen = []

    def rule(feats, p):
        seen.append(feats.index[-1])
        if len(feats) - 1 == bar:
            c = float(feats["close"].iloc[-1])
            return UnderlyingSignal(Direction.LONG, c, stop, target, hold, Regime.BULLISH, "t")
        return None
    return Strategy("t", rule, InstrumentType.STOCK, "1d", {"min_rr": 1.0}), seen


def test_strategy_only_sees_data_through_current_bar():
    n = 30
    df = frame([100.0] * n, [101.0] * n, [99.0] * n, [100.0] * n)
    strat, seen = signal_on(-1)
    precompute_signals(strat, df, warmup=5)
    # at evaluation step i the last visible bar is exactly bar i, never later
    assert seen == list(df.index[5:n - 1])


def test_fill_at_next_open_and_stop_first_when_both_touched():
    n = 20
    o = [100.0] * n; h = [100.5] * n; l = [99.5] * n; c = [100.0] * n
    o[11] = 101.0                       # next-bar open differs from signal close
    h[12], l[12] = 110.0, 90.0          # bar touches both stop and target
    df = frame(o, h, l, c)
    strat, _ = signal_on(10)
    sigs = precompute_signals(strat, df, warmup=5)
    trades = simulate_symbol("X", strat, df, sigs, ZERO)
    t = trades[0]
    assert t.entry_time == df.index[11] and t.entry_price == 101.0     # filled at NEXT bar open
    assert t.reason == "stop" and t.exit_price == pytest.approx(101.0 - 2.0)  # stop re-anchored, pessimistic


def test_gap_through_stop_fills_at_open():
    n = 20
    o = [100.0] * n; h = [100.5] * n; l = [99.5] * n; c = [100.0] * n
    o[13], h[13], l[13], c[13] = 90.0, 91.0, 89.0, 90.0
    df = frame(o, h, l, c)
    strat, _ = signal_on(10)
    t = simulate_symbol("X", strat, df, precompute_signals(strat, df, 5), ZERO)[0]
    assert t.reason == "stop_gap" and t.exit_price == 90.0


def test_costs_reduce_pnl():
    n = 20
    o = [100.0] * n; h = [100.5] * n; l = [99.5] * n; c = [100.0] * n
    h[13] = 110
    df = frame(o, h, l, c)
    strat, _ = signal_on(10)
    sigs = precompute_signals(strat, df, 5)
    free = simulate_symbol("X", strat, df, sigs, ZERO)[0]
    costly = simulate_symbol("X", strat, df, sigs, CostModel(10, 5, 0.01, 4, 1, 0.65))[0]
    assert costly.pnl_per_unit < free.pnl_per_unit


def _bt(sym, day, pnl, risk=1.0):
    ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
    return BTTrade(sym, "s", ts, ts + pd.Timedelta(days=2), "long", "stock", 100, 100 + pnl, risk, pnl, 1, "t", 2)


def test_portfolio_respects_position_limits_and_risk():
    trades = [_bt(s, 0, 1.0) for s in "ABCDE"] + [_bt("A", 0, 1.0)]
    res = run_portfolio(trades, 100_000, 0.5, 3, 100)
    assert len(res.trades) == 3 and res.skipped_for_limits == 3
    assert all(t["risk_usd"] <= 500 for t in res.trades)


def test_metrics():
    eq = pd.Series([100, 110, 99, 120], index=pd.date_range("2024-01-01", periods=4, tz="UTC"))
    assert max_drawdown_pct(eq) == pytest.approx(10.0)
    trades = [{"pnl": 10, "r_multiple": 1.0}, {"pnl": -5, "r_multiple": -0.5}, {"pnl": 20, "r_multiple": 2.0}]
    m = summarize(trades, eq, 100)
    assert m["trades"] == 3 and m["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["profit_factor"] == pytest.approx(6.0) and m["expectancy_r"] == pytest.approx(2.5 / 3, abs=1e-3)


def test_acceptance_rejects_weak_or_degraded_results():
    acc = AcceptanceCfg()
    good = {"walk_forward": {"trades": 80, "profit_factor": 1.6, "max_drawdown_pct": 8, "expectancy_r": 0.2,
                             "sharpe": 1.1, "positive_windows_frac": 0.7},
            "in_sample": {"sharpe": 1.3}, "benchmark_buy_hold": {"sharpe": 0.8}}
    assert evaluate(good, acc) == (True, [])
    for k, v in [("trades", 10), ("profit_factor", 1.0), ("max_drawdown_pct", 35), ("expectancy_r", -0.1),
                 ("sharpe", 0.2), ("positive_windows_frac", 0.3)]:
        bad = json.loads(json.dumps(good)); bad["walk_forward"][k] = v
        assert evaluate(bad, acc)[0] is False
    overfit = json.loads(json.dumps(good)); overfit["in_sample"]["sharpe"] = 5.0
    assert evaluate(overfit, acc)[0] is False


def test_research_pipeline_runs_and_records(tmp_path, cfg):
    data = SyntheticMarketData(seed=11)
    bars = {s: data.full_history(s).tail(700) for s in ["AAA", "BBB", "CCC"]}
    rep = research_strategy(STRATEGIES["stock_breakout"], bars, cfg, data.source)
    assert rep["data_source"] == "SYNTHETIC"
    for section in ("in_sample", "holdout", "walk_forward", "benchmark_buy_hold"):
        assert section in rep
    assert rep["walk_forward"]["windows"] >= 3
    ok, reasons = evaluate(rep, cfg.backtest.acceptance)
    record_result(tmp_path / "a.json", rep, ok, reasons)
    assert load_approvals(tmp_path / "a.json")["stock_breakout"]["data_source"] == "SYNTHETIC"


def test_option_strategy_backtest_is_labeled_modeled(cfg):
    data = SyntheticMarketData(seed=5)
    bars = {s: data.full_history(s).tail(600) for s in ["AAA", "BBB"]}
    rep = research_strategy(STRATEGIES["option_momentum"], bars, cfg, data.source)
    assert rep["option_pricing"] == "modeled_black_scholes"
