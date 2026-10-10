"""Market-condition classification (bullish / bearish / neutral) from causal features."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..models import Regime


@dataclass
class RegimeAssessment:
    regime: Regime
    trend_strength: float     # ADX
    momentum: float           # 20-bar rate of change
    volatility: float         # annualized realized vol
    volume_ratio: float       # relative volume
    score: int                # -3..+3 vote

    def as_dict(self) -> dict:
        return {"regime": self.regime.value, "trend_strength": round(self.trend_strength, 2),
                "momentum": round(self.momentum, 4), "volatility": round(self.volatility, 4),
                "volume_ratio": round(self.volume_ratio, 2), "score": self.score}


def classify(row: pd.Series) -> RegimeAssessment | None:
    """Classify using a single feature row (output of add_standard_features)."""
    needed = ["close", "ema_fast", "ema_slow", "adx", "roc20", "rv20", "rel_vol", "macd_hist"]
    if any(pd.isna(row.get(k)) for k in needed):
        return None
    score = 0
    score += 1 if row["close"] > row["ema_fast"] > row["ema_slow"] else -1 if row["close"] < row["ema_fast"] < row["ema_slow"] else 0
    score += 1 if row["roc20"] > 0.02 else -1 if row["roc20"] < -0.02 else 0
    score += 1 if row["macd_hist"] > 0 else -1 if row["macd_hist"] < 0 else 0
    trending = row["adx"] >= 20
    if score >= 2 and trending:
        regime = Regime.BULLISH
    elif score <= -2 and trending:
        regime = Regime.BEARISH
    else:
        regime = Regime.NEUTRAL
    return RegimeAssessment(regime, float(row["adx"]), float(row["roc20"]), float(row["rv20"]),
                            float(row["rel_vol"]), score)
