"""Strategy framework.

A strategy = an *underlying signal rule* + an *instrument mapping*.

* Signal rules look only at the LAST row of a causal feature frame (data through bar t) and return an
  ``UnderlyingSignal`` with direction, stop and target expressed in underlying-price terms.
* The instrument mapping turns that into a stock trade (long only — short stock has unlimited loss),
  a long call/put, or a defined-risk debit spread. Short/naked options are never produced.

The same rule objects are used by the live engine and the backtester, so what is backtested is
exactly what trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..analysis.indicators import add_standard_features, ema, rsi, atr, vwap_intraday
from ..analysis.regime import classify
from ..models import Direction, InstrumentType, Regime


@dataclass
class UnderlyingSignal:
    direction: Direction
    entry: float
    stop: float
    target: float
    max_hold_bars: int
    regime: Regime
    rationale: str
    features: dict = field(default_factory=dict)

    @property
    def reward_risk(self) -> float:
        risk = abs(self.entry - self.stop)
        return abs(self.target - self.entry) / risk if risk > 0 else 0.0


SignalRule = Callable[[pd.DataFrame, dict], "UnderlyingSignal | None"]


@dataclass
class Strategy:
    name: str
    rule: SignalRule
    instrument: InstrumentType
    timeframe: str                      # "1d" or "5m"
    params: dict
    param_grid: dict[str, list] = field(default_factory=dict)
    allow_short: bool = False           # bearish signals (only meaningful for option instruments)
    intraday_only: bool = False
    option_target_delta: float = 0.50
    option_dte: tuple[int, int] = (14, 45)
    spread_width_atr: float = 1.0       # debit-spread short-strike distance in ATRs
    description: str = ""

    def with_params(self, **overrides) -> "Strategy":
        return Strategy(**{**self.__dict__, "params": {**self.params, **overrides}})

    def evaluate(self, feats: pd.DataFrame) -> UnderlyingSignal | None:
        if len(feats) < 2:
            return None
        sig = self.rule(feats, self.params)
        if sig is None:
            return None
        if sig.direction == Direction.SHORT and (self.instrument == InstrumentType.STOCK or not self.allow_short):
            return None  # never short stock; bearish views only via long puts / put debit spreads
        if sig.reward_risk < self.params.get("min_rr", 1.5):
            return None
        return sig


# ----------------------------------------------------------------------------- features

def daily_features(bars: pd.DataFrame) -> pd.DataFrame:
    return add_standard_features(bars, 252)


def intraday_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["ema9"] = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    out["rsi"] = rsi(out["close"], 14)
    out["atr"] = atr(out, 14)
    out["vwap"] = vwap_intraday(out)
    out["rel_vol"] = out["volume"] / out["volume"].shift(1).rolling(20, min_periods=20).mean()
    return out


def _last(feats: pd.DataFrame, cols: list[str]) -> pd.Series | None:
    row = feats.iloc[-1]
    if any(pd.isna(row.get(c)) for c in cols):
        return None
    return row


def _feat_dict(row: pd.Series, cols: list[str]) -> dict:
    return {c: (round(float(row[c]), 4) if not pd.isna(row[c]) else None) for c in cols}


# ----------------------------------------------------------------------------- rules

def momentum_rule(feats: pd.DataFrame, p: dict) -> UnderlyingSignal | None:
    """Trend continuation: strong trend (ADX), aligned EMAs, positive/negative momentum, pullback not overextended."""
    row = _last(feats, ["close", "ema_fast", "ema_slow", "adx", "roc20", "rsi", "atr", "rv20", "rel_vol", "macd_hist"])
    if row is None:
        return None
    reg = classify(row)
    if reg is None or row["adx"] < p.get("adx_min", 22):
        return None
    c, a = float(row["close"]), float(row["atr"])
    cols = ["close", "ema_fast", "ema_slow", "adx", "roc20", "rsi", "atr", "rv20", "rel_vol"]
    if reg.regime == Regime.BULLISH and p.get("rsi_lo", 50) <= row["rsi"] <= p.get("rsi_hi", 70) \
            and row["macd_hist"] > 0 and row["roc20"] > p.get("roc_min", 0.03):
        return UnderlyingSignal(Direction.LONG, c, c - p.get("stop_atr", 2.0) * a, c + p.get("target_atr", 4.0) * a,
                                p.get("max_hold", 15), reg.regime, "bullish trend continuation", _feat_dict(row, cols))
    if reg.regime == Regime.BEARISH and 100 - p.get("rsi_hi", 70) <= row["rsi"] <= 100 - p.get("rsi_lo", 50) \
            and row["macd_hist"] < 0 and row["roc20"] < -p.get("roc_min", 0.03):
        return UnderlyingSignal(Direction.SHORT, c, c + p.get("stop_atr", 2.0) * a, c - p.get("target_atr", 4.0) * a,
                                p.get("max_hold", 15), reg.regime, "bearish trend continuation", _feat_dict(row, cols))
    return None


def breakout_rule(feats: pd.DataFrame, p: dict) -> UnderlyingSignal | None:
    """Donchian breakout on expanding volume with trend confirmation."""
    row = _last(feats, ["close", "don_high", "don_low", "rel_vol", "adx", "atr", "ema_slow", "ema_fast", "roc20", "rv20", "macd_hist"])
    if row is None:
        return None
    reg = classify(row)
    c, a = float(row["close"]), float(row["atr"])
    cols = ["close", "don_high", "don_low", "rel_vol", "adx", "atr", "rv20"]
    if row["rel_vol"] < p.get("rel_vol_min", 1.5):
        return None
    if c > row["don_high"] and c > row["ema_slow"]:
        return UnderlyingSignal(Direction.LONG, c, c - p.get("stop_atr", 1.5) * a, c + p.get("target_atr", 3.5) * a,
                                p.get("max_hold", 10), reg.regime if reg else Regime.NEUTRAL,
                                "20-bar high breakout on volume", _feat_dict(row, cols))
    if c < row["don_low"] and c < row["ema_slow"]:
        return UnderlyingSignal(Direction.SHORT, c, c + p.get("stop_atr", 1.5) * a, c - p.get("target_atr", 3.5) * a,
                                p.get("max_hold", 10), reg.regime if reg else Regime.NEUTRAL,
                                "20-bar low breakdown on volume", _feat_dict(row, cols))
    return None


def reversal_rule(feats: pd.DataFrame, p: dict) -> UnderlyingSignal | None:
    """Short-term mean reversion in the direction of the long-term trend (RSI(2) extremes)."""
    row = _last(feats, ["close", "sma_200", "rsi2", "atr", "bb_mid", "bb_lower", "bb_upper", "roc20", "adx",
                        "ema_fast", "ema_slow", "rv20", "rel_vol", "macd_hist"])
    if row is None:
        return None
    c, a = float(row["close"]), float(row["atr"])
    reg = classify(row)
    cols = ["close", "sma_200", "rsi2", "atr", "bb_mid", "rv20"]
    if c > row["sma_200"] and row["rsi2"] < p.get("rsi2_lo", 10) and c < row["bb_lower"] * p.get("bb_tol", 1.01):
        target = max(float(row["bb_mid"]), c + p.get("min_target_atr", 1.5) * a)
        return UnderlyingSignal(Direction.LONG, c, c - p.get("stop_atr", 1.0) * a, target, p.get("max_hold", 5),
                                reg.regime if reg else Regime.NEUTRAL, "oversold pullback in uptrend", _feat_dict(row, cols))
    if c < row["sma_200"] and row["rsi2"] > 100 - p.get("rsi2_lo", 10) and c > row["bb_upper"] / p.get("bb_tol", 1.01):
        target = min(float(row["bb_mid"]), c - p.get("min_target_atr", 1.5) * a)
        return UnderlyingSignal(Direction.SHORT, c, c + p.get("stop_atr", 1.0) * a, target, p.get("max_hold", 5),
                                reg.regime if reg else Regime.NEUTRAL, "overbought bounce in downtrend", _feat_dict(row, cols))
    return None


def volatility_squeeze_rule(feats: pd.DataFrame, p: dict) -> UnderlyingSignal | None:
    """Volatility compression (Bollinger width in low percentile) followed by directional expansion.

    Paired with long options: buys convexity when realized vol is compressed (options relatively cheap).
    """
    if len(feats) < 130:
        return None
    row = _last(feats, ["close", "bb_width", "bb_upper", "bb_lower", "atr", "rel_vol", "rv20", "adx",
                        "ema_fast", "ema_slow", "roc20", "macd_hist"])
    if row is None:
        return None
    width_hist = feats["bb_width"].iloc[-126:-1].dropna()
    if len(width_hist) < 60:
        return None
    pct = float((width_hist < feats["bb_width"].iloc[-2]).mean())  # compression measured on prior bar
    if pct > p.get("squeeze_pct", 0.2):
        return None
    c, a = float(row["close"]), float(row["atr"])
    reg = classify(row)
    cols = ["close", "bb_width", "atr", "rel_vol", "rv20"]
    f = {**_feat_dict(row, cols), "bb_width_pctile": round(pct, 3)}
    if c > row["bb_upper"] and row["rel_vol"] > p.get("rel_vol_min", 1.2):
        return UnderlyingSignal(Direction.LONG, c, c - p.get("stop_atr", 1.5) * a, c + p.get("target_atr", 3.5) * a,
                                p.get("max_hold", 10), reg.regime if reg else Regime.NEUTRAL, "squeeze release upward", f)
    if c < row["bb_lower"] and row["rel_vol"] > p.get("rel_vol_min", 1.2):
        return UnderlyingSignal(Direction.SHORT, c, c + p.get("stop_atr", 1.5) * a, c - p.get("target_atr", 3.5) * a,
                                p.get("max_hold", 10), reg.regime if reg else Regime.NEUTRAL, "squeeze release downward", f)
    return None


def scalp_rule(feats: pd.DataFrame, p: dict) -> UnderlyingSignal | None:
    """Intraday 5-minute momentum relative to VWAP with fresh EMA cross. Short holding period."""
    row = _last(feats, ["close", "ema9", "ema21", "vwap", "rsi", "atr", "rel_vol"])
    if row is None:
        return None
    prev = feats.iloc[-2]
    if pd.isna(prev.get("ema9")) or pd.isna(prev.get("ema21")):
        return None
    ts = feats.index[-1]
    if hasattr(ts, "tz_convert"):
        ny = ts.tz_convert("America/New_York")
        if (ny.hour, ny.minute) < (9, 45) or (ny.hour, ny.minute) >= (15, 15):
            return None
    c, a = float(row["close"]), float(row["atr"])
    cols = ["close", "ema9", "ema21", "vwap", "rsi", "atr", "rel_vol"]
    crossed_up = prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]
    crossed_dn = prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]
    if row["rel_vol"] < p.get("rel_vol_min", 1.2):
        return None
    if crossed_up and c > row["vwap"] and 50 < row["rsi"] < 75:
        return UnderlyingSignal(Direction.LONG, c, c - p.get("stop_atr", 1.0) * a, c + p.get("target_atr", 2.0) * a,
                                p.get("max_hold", 6), Regime.BULLISH, "5m EMA cross above VWAP", _feat_dict(row, cols))
    if crossed_dn and c < row["vwap"] and 25 < row["rsi"] < 50:
        return UnderlyingSignal(Direction.SHORT, c, c + p.get("stop_atr", 1.0) * a, c - p.get("target_atr", 2.0) * a,
                                p.get("max_hold", 6), Regime.BEARISH, "5m EMA cross below VWAP", _feat_dict(row, cols))
    return None
