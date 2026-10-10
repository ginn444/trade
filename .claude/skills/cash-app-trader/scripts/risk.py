"""Risk metrics and position sizing for one ticker."""
import argparse
import json
import math
import statistics

from data import history

TRADING_DAYS = 252


def sma(xs, n):
    return sum(xs[-n:]) / min(n, len(xs))


def analyze(closes, budget=10.0, risk_pct=0.02, reward_ratio=2.0):
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    daily_vol = statistics.stdev(rets[-60:])
    peak, max_dd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        max_dd = min(max_dd, c / peak - 1)
    price = closes[-1]
    # Stop ~2 daily standard deviations below price (ATR-like), at least 3%.
    stop_dist = max(2 * daily_vol, 0.03)
    stop = price * (1 - stop_dist)
    target = price * (1 + stop_dist * reward_ratio)
    max_loss = budget * risk_pct
    # Dollars to invest so a stop-out loses at most max_loss; capped at budget.
    dollars = min(budget, max_loss / stop_dist)
    # Tiny accounts: Cash App minimum is $1, so allow $1 if the loss stays small.
    if dollars < 1 and budget >= 1:
        dollars = 1.0
    var95 = 1.645 * daily_vol * dollars
    return {
        "price": round(price, 2),
        "sma20": round(sma(closes, 20), 2),
        "sma50": round(sma(closes, 50), 2),
        "ret_1m_pct": round((price / closes[-22] - 1) * 100, 2),
        "ann_vol_pct": round(daily_vol * math.sqrt(TRADING_DAYS) * 100, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": reward_ratio,
        "invest_dollars": round(dollars, 2),
        "loss_at_stop": round(dollars * stop_dist, 2),
        "var95_1day_dollars": round(var95, 2),
        "risk_level": "low" if daily_vol < 0.015 else "medium" if daily_vol < 0.03 else "high",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--risk-pct", type=float, default=0.02, help="max account fraction lost at stop")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--csv")
    a = p.parse_args()
    closes = [r["close"] for r in history(a.ticker, a.demo, a.csv)]
    print(json.dumps({"ticker": a.ticker.upper(), **analyze(closes, a.budget, a.risk_pct)}, indent=2))


if __name__ == "__main__":
    main()
