import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from autotrader.analysis.indicators import add_standard_features, donchian, rsi
from autotrader.analysis.options_math import bs, implied_vol
from autotrader.analysis.regime import classify
from autotrader.data.alpaca_data import build_occ, parse_occ
from autotrader.data.synthetic import SyntheticMarketData
from autotrader.models import Regime


@pytest.fixture(scope="module")
def bars():
    return SyntheticMarketData(seed=3).full_history("AAA").tail(400)


def test_indicators_are_causal(bars):
    """Values at bar t must not change when future bars are appended (no look-ahead)."""
    full = add_standard_features(bars)
    cut = 300
    prefix = add_standard_features(bars.iloc[:cut])
    cols = ["ema_fast", "sma_200", "rsi", "atr", "adx", "macd_hist", "bb_upper", "rv20", "don_high", "rel_vol"]
    pd.testing.assert_frame_equal(full[cols].iloc[:cut], prefix[cols], check_exact=False, rtol=1e-9)


def test_donchian_excludes_current_bar(bars):
    hi, _ = donchian(bars, 20)
    assert hi.iloc[-1] == pytest.approx(bars["high"].iloc[-21:-1].max())


def test_rsi_bounds(bars):
    r = rsi(bars["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_regime_classification():
    row = pd.Series({"close": 110, "ema_fast": 105, "ema_slow": 100, "adx": 30, "roc20": 0.05, "rv20": 0.2,
                     "rel_vol": 1.1, "macd_hist": 0.4})
    assert classify(row).regime == Regime.BULLISH
    row2 = row.copy(); row2.update({"close": 90, "ema_fast": 95, "roc20": -0.05, "macd_hist": -0.4})
    assert classify(row2).regime == Regime.BEARISH
    row3 = row.copy(); row3["adx"] = 10
    assert classify(row3).regime == Regime.NEUTRAL
    row4 = row.copy(); row4["adx"] = float("nan")
    assert classify(row4) is None


def test_put_call_parity():
    s, k, t, v, r = 100, 105, 0.25, 0.3, 0.04
    c, p = bs(s, k, t, v, r, "call").price, bs(s, k, t, v, r, "put").price
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-8)


def test_greeks_signs():
    g = bs(100, 100, 0.1, 0.25, kind="call")
    assert 0.45 < g.delta < 0.6 and g.gamma > 0 and g.theta < 0 and g.vega > 0
    assert bs(100, 100, 0.1, 0.25, kind="put").delta < 0


def test_implied_vol_roundtrip():
    px = bs(100, 95, 0.2, 0.37, kind="put").price
    assert implied_vol(px, 100, 95, 0.2, kind="put") == pytest.approx(0.37, abs=1e-4)
    assert implied_vol(0.0, 100, 95, 0.2) is None


def test_occ_roundtrip():
    sym = build_occ("SPY", date(2026, 12, 18), "call", 450.5)
    assert sym == "SPY261218C00450500"
    assert parse_occ(sym) == ("SPY", date(2026, 12, 18), "call", 450.5)
    assert parse_occ("SPY") is None
