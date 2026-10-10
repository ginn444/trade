"""All tradable strategies. Each must pass backtest validation before autonomous execution."""
from __future__ import annotations

from ..models import InstrumentType
from .base import (Strategy, breakout_rule, momentum_rule, reversal_rule, scalp_rule,
                   volatility_squeeze_rule)

_MOM = {"adx_min": 22, "rsi_lo": 50, "rsi_hi": 70, "roc_min": 0.03, "stop_atr": 2.0, "target_atr": 4.0,
        "max_hold": 15, "min_rr": 1.5}
_MOM_GRID = {"adx_min": [20, 25], "stop_atr": [1.5, 2.0], "target_atr": [3.0, 4.0]}
_BO = {"rel_vol_min": 1.5, "stop_atr": 1.5, "target_atr": 3.5, "max_hold": 10, "min_rr": 1.5}
_BO_GRID = {"rel_vol_min": [1.3, 1.8], "stop_atr": [1.5, 2.0], "target_atr": [3.0, 4.0]}
_REV = {"rsi2_lo": 10, "bb_tol": 1.01, "stop_atr": 1.0, "min_target_atr": 1.5, "max_hold": 5, "min_rr": 1.5}
_REV_GRID = {"rsi2_lo": [5, 10], "stop_atr": [1.0, 1.5]}
_VOL = {"squeeze_pct": 0.2, "rel_vol_min": 1.2, "stop_atr": 1.5, "target_atr": 3.5, "max_hold": 10, "min_rr": 1.5}
_VOL_GRID = {"squeeze_pct": [0.15, 0.25], "target_atr": [3.0, 4.0]}
_SCALP = {"rel_vol_min": 1.2, "stop_atr": 1.0, "target_atr": 2.0, "max_hold": 6, "min_rr": 1.5}
_SCALP_GRID = {"rel_vol_min": [1.0, 1.5], "target_atr": [1.5, 2.5]}

STRATEGIES: dict[str, Strategy] = {s.name: s for s in [
    Strategy("stock_momentum", momentum_rule, InstrumentType.STOCK, "1d", _MOM, _MOM_GRID,
             description="Long-only trend continuation in liquid stocks/ETFs"),
    Strategy("stock_breakout", breakout_rule, InstrumentType.STOCK, "1d", _BO, _BO_GRID,
             description="Long-only 20-day breakout on relative volume"),
    Strategy("stock_reversal", reversal_rule, InstrumentType.STOCK, "1d", _REV, _REV_GRID,
             description="Long-only RSI(2) pullback in a 200-day uptrend"),
    Strategy("option_momentum", momentum_rule, InstrumentType.OPTION, "1d", _MOM, _MOM_GRID, allow_short=True,
             option_target_delta=0.50, option_dte=(21, 45),
             description="Long calls (bullish) / long puts (bearish) on trend continuation"),
    Strategy("option_breakout", breakout_rule, InstrumentType.OPTION, "1d", _BO, _BO_GRID, allow_short=True,
             option_target_delta=0.55, option_dte=(14, 45),
             description="Long calls/puts on volume-confirmed range breakouts"),
    Strategy("option_reversal", reversal_rule, InstrumentType.OPTION, "1d", _REV, _REV_GRID, allow_short=True,
             option_target_delta=0.60, option_dte=(14, 30),
             description="Long calls/puts on short-term exhaustion against the long-term trend"),
    Strategy("option_debit_spread", momentum_rule, InstrumentType.DEBIT_SPREAD, "1d", _MOM, _MOM_GRID, allow_short=True,
             option_target_delta=0.55, option_dte=(21, 45), spread_width_atr=1.5,
             description="Bull call / bear put debit spreads (defined risk) on trend continuation"),
    Strategy("option_volatility", volatility_squeeze_rule, InstrumentType.OPTION, "1d", _VOL, _VOL_GRID, allow_short=True,
             option_target_delta=0.50, option_dte=(21, 45),
             description="Long options on volatility-compression breakouts (buying convexity when vol is cheap)"),
    Strategy("option_scalp", scalp_rule, InstrumentType.OPTION, "5m", _SCALP, _SCALP_GRID, allow_short=True,
             intraday_only=True, option_target_delta=0.50, option_dte=(1, 7),
             description="Intraday 5-minute VWAP/EMA momentum scalps in near-dated options; flat by EOD"),
]}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}") from None
