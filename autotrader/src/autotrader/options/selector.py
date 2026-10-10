"""Option contract selection and conversion of underlying signals into option candidates.

Filters on: quote validity/freshness, bid-ask spread, open interest, volume (when the feed provides
it), DTE window, and delta band. Debit spreads pair a long leg with a further-OTM short leg of the
same type and expiry, so maximum loss is the net debit paid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

from ..analysis.options_math import bs
from ..config import RiskCfg
from ..models import Candidate, Direction, InstrumentType, OptionContract, OptionLeg
from ..strategies.base import Strategy, UnderlyingSignal

log = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    candidate: Candidate | None
    reasons: list[str]


def hold_minutes(strategy: Strategy, bars: int) -> int:
    """Max holding period in CALENDAR minutes (daily bars -> trading days incl. weekends)."""
    if strategy.timeframe == "5m":
        return bars * 5
    return int(bars * 1440 * 7 / 5)


def _t_years(exp: date, now: datetime) -> float:
    close = datetime(exp.year, exp.month, exp.day, 20, 0, tzinfo=timezone.utc)
    return max((close - now).total_seconds(), 3600) / (365 * 86400)


def contract_delta(c: OptionContract, spot: float, now: datetime) -> float | None:
    if c.delta is not None:
        return c.delta
    if c.implied_vol:
        return bs(spot, c.strike, _t_years(c.expiration, now), c.implied_vol, kind=c.option_type).delta
    return None


def liquidity_reasons(c: OptionContract, risk: RiskCfg, now: datetime, max_age_s: int) -> list[str]:
    r = []
    q = c.quote
    if q is None or not q.is_valid():
        return ["no valid quote"]
    if q.age_seconds(now) > max_age_s:
        r.append(f"stale quote ({q.age_seconds(now):.0f}s)")
    if q.spread_pct > risk.max_spread_pct_option:
        r.append(f"spread {q.spread_pct:.1f}% > {risk.max_spread_pct_option}%")
    if c.open_interest is None or c.open_interest < risk.min_option_open_interest:
        r.append(f"open interest {c.open_interest} < {risk.min_option_open_interest}")
    if c.volume is not None and c.volume < risk.min_option_volume:
        r.append(f"volume {c.volume} < {risk.min_option_volume}")
    return r


def _premium_levels(c: OptionContract, sig: UnderlyingSignal, spot: float, now: datetime, entry: float) -> tuple[float, float]:
    """Translate underlying stop/target into option premium levels using Black-Scholes at current IV."""
    iv = c.implied_vol or 0.3
    t = _t_years(c.expiration, now)
    stop_model = bs(sig.stop, c.strike, t, iv, kind=c.option_type).price
    tgt_model = bs(sig.target, c.strike, t, iv, kind=c.option_type).price
    stop = max(stop_model, entry * 0.5, 0.01)   # premium stop no looser than -50%
    return round(stop, 2), round(max(tgt_model, entry * 1.01), 2)


def select_option_candidate(strategy: Strategy, sig: UnderlyingSignal, underlying: str, spot: float,
                            chain: list[OptionContract], risk: RiskCfg, now: datetime,
                            max_quote_age_s: int) -> SelectionResult:
    kind = "call" if sig.direction == Direction.LONG else "put"
    today = now.date()
    dmin, dmax = strategy.option_dte
    if strategy.intraday_only:
        dmin, dmax = max(dmin, risk.scalp_dte_min), min(dmax, risk.scalp_dte_max)
    else:
        dmin, dmax = max(dmin, risk.option_dte_min), min(dmax, risk.option_dte_max)
    reasons: list[str] = []
    pool = [c for c in chain if c.option_type == kind and dmin <= c.dte(today) <= dmax]
    if not pool:
        return SelectionResult(None, [f"no {kind}s with DTE in [{dmin},{dmax}]"])
    scored = []
    for c in pool:
        lr = liquidity_reasons(c, risk, now, max_quote_age_s)
        if lr:
            continue
        d = contract_delta(c, spot, now)
        if d is None or not (risk.option_delta_min <= abs(d) <= risk.option_delta_max):
            continue
        scored.append((abs(abs(d) - strategy.option_target_delta), c.quote.spread_pct, c, d))
    if not scored:
        return SelectionResult(None, [f"no {kind} passed liquidity/delta filters (pool={len(pool)})"])
    scored.sort(key=lambda x: (x[0], x[1]))
    _, _, long_c, long_delta = scored[0]
    features = {**sig.features, "option_delta": round(long_delta, 3), "option_iv": long_c.implied_vol,
                "option_theta": long_c.theta, "option_gamma": long_c.gamma, "option_oi": long_c.open_interest,
                "option_spread_pct": round(long_c.quote.spread_pct, 2), "dte": long_c.dte(today)}

    if strategy.instrument == InstrumentType.OPTION:
        entry = round(long_c.quote.mid, 2)
        stop, target = _premium_levels(long_c, sig, spot, now, entry)
        cand = Candidate(strategy.name, underlying, sig.direction, InstrumentType.OPTION, entry, stop, target, spot,
                         sig.regime, features, [OptionLeg(long_c.symbol, "buy", 1, long_c)],
                         intraday_only=strategy.intraday_only, rationale=sig.rationale,
                         max_hold_minutes=hold_minutes(strategy, sig.max_hold_bars))
        return SelectionResult(cand, reasons)

    # ---- debit spread: short leg further OTM, same expiry/type ------------------
    atr = float(sig.features.get("atr") or abs(sig.entry - sig.stop))
    width_target = max(atr * strategy.spread_width_atr, 1.0)
    want = long_c.strike + width_target if kind == "call" else long_c.strike - width_target
    shorts = [c for c in chain if c.option_type == kind and c.expiration == long_c.expiration
              and ((c.strike > long_c.strike) if kind == "call" else (c.strike < long_c.strike))
              and not liquidity_reasons(c, risk, now, max_quote_age_s)]
    if not shorts:
        return SelectionResult(None, ["no liquid short leg for debit spread"])
    short_c = min(shorts, key=lambda c: abs(c.strike - want))
    width = abs(short_c.strike - long_c.strike)
    debit_mid = round(long_c.quote.mid - short_c.quote.mid, 2)
    debit_worst = long_c.quote.ask - short_c.quote.bid
    if debit_mid <= 0 or debit_worst >= width:
        return SelectionResult(None, [f"spread pricing invalid (debit {debit_worst:.2f} vs width {width:.2f})"])
    max_profit = width - debit_mid
    target = round(debit_mid + 0.75 * max_profit, 2)
    stop = round(debit_mid * 0.5, 2)
    features.update({"spread_width": width, "short_strike": short_c.strike, "long_strike": long_c.strike,
                     "max_profit_per_spread": round(max_profit * 100, 2), "max_loss_per_spread": round(debit_mid * 100, 2)})
    cand = Candidate(strategy.name, underlying, sig.direction, InstrumentType.DEBIT_SPREAD, debit_mid, stop, target,
                     spot, sig.regime, features,
                     [OptionLeg(long_c.symbol, "buy", 1, long_c), OptionLeg(short_c.symbol, "sell", 1, short_c)],
                     intraday_only=strategy.intraday_only, rationale=sig.rationale,
                     max_hold_minutes=hold_minutes(strategy, sig.max_hold_bars))
    return SelectionResult(cand, reasons)
