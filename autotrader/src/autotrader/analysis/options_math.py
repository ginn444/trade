"""Black-Scholes pricing, greeks and implied volatility (European approximation).

Used to (1) fill in greeks when the data feed omits them and (2) MODEL option prices in backtests
where real historical option quotes are unavailable. Modeled prices are approximations and are
labeled as such in every backtest report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1.00 vol point (i.e. per 100%); divide by 100 for per-1%


def bs(spot: float, strike: float, t_years: float, vol: float, rate: float = 0.04,
       kind: str = "call", div: float = 0.0) -> Greeks:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t_years <= 0 or vol <= 0:
        intrinsic = max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
        d = (1.0 if spot > strike else 0.0) if kind == "call" else (-1.0 if spot < strike else 0.0)
        return Greeks(intrinsic, d, 0.0, 0.0, 0.0)
    st = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * t_years) / st
    d2 = d1 - st
    df_r, df_q = math.exp(-rate * t_years), math.exp(-div * t_years)
    if kind == "call":
        price = spot * df_q * _cdf(d1) - strike * df_r * _cdf(d2)
        delta = df_q * _cdf(d1)
        theta = (-spot * df_q * _pdf(d1) * vol / (2 * math.sqrt(t_years))
                 - rate * strike * df_r * _cdf(d2) + div * spot * df_q * _cdf(d1))
    else:
        price = strike * df_r * _cdf(-d2) - spot * df_q * _cdf(-d1)
        delta = -df_q * _cdf(-d1)
        theta = (-spot * df_q * _pdf(d1) * vol / (2 * math.sqrt(t_years))
                 + rate * strike * df_r * _cdf(-d2) - div * spot * df_q * _cdf(-d1))
    gamma = df_q * _pdf(d1) / (spot * st)
    vega = spot * df_q * _pdf(d1) * math.sqrt(t_years)
    return Greeks(max(price, 0.0), delta, gamma, theta / 365.0, vega)


def implied_vol(price: float, spot: float, strike: float, t_years: float, rate: float = 0.04,
                kind: str = "call", lo: float = 0.01, hi: float = 5.0, tol: float = 1e-6) -> float | None:
    """Bisection IV solver. Returns None when price is outside no-arbitrage bounds."""
    if price <= 0 or t_years <= 0:
        return None
    if bs(spot, strike, t_years, lo, rate, kind).price > price or bs(spot, strike, t_years, hi, rate, kind).price < price:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs(spot, strike, t_years, mid, rate, kind).price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def strike_for_delta(spot: float, t_years: float, vol: float, target_delta: float, kind: str,
                     rate: float = 0.04, step: float = 1.0) -> float:
    """Closest strike on a `step` grid to the absolute target delta."""
    best, best_err = spot, float("inf")
    k = max(step, round(spot * 0.5 / step) * step)
    while k <= spot * 1.5:
        d = abs(bs(spot, k, t_years, vol, rate, kind).delta)
        if abs(d - target_delta) < best_err:
            best, best_err = k, abs(d - target_delta)
        k += step
    return best
