"""Scan a watchlist and rank trade candidates."""
import argparse
import json
import os

from data import DEFAULT_WATCHLIST, history
from risk import analyze

LEVELS = ["low", "medium", "high"]


def score(m):
    s = 0.0
    reasons = []
    if m["price"] > m["sma20"] > m["sma50"]:
        s += 2; reasons.append("uptrend (price > 20d > 50d avg)")
    elif m["price"] > m["sma50"]:
        s += 1; reasons.append("above 50d avg")
    else:
        s -= 1; reasons.append("below 50d avg")
    s += max(-2, min(2, m["ret_1m_pct"] / 5))
    reasons.append(f"1-month {m['ret_1m_pct']:+.1f}%")
    s -= m["ann_vol_pct"] / 50
    reasons.append(f"volatility {m['ann_vol_pct']}%/yr")
    # Chasing: penalize if extended far above the 20d average.
    ext = m["price"] / m["sma20"] - 1
    if ext > 0.08:
        s -= 1; reasons.append(f"extended {ext:.0%} above 20d avg")
    return round(s, 2), reasons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", help="comma separated; default is a large-cap watchlist")
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--risk", choices=LEVELS, default="medium", help="max risk level to allow")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    tickers = a.tickers.split(",") if a.tickers else DEFAULT_WATCHLIST
    rows = []
    for t in tickers:
        m = analyze([r["close"] for r in history(t, a.demo)], a.budget)
        if LEVELS.index(m["risk_level"]) > LEVELS.index(a.risk):
            continue
        s, why = score(m)
        rows.append({"ticker": t.upper(), "score": s, "why": why, **m})
    rows.sort(key=lambda r: r["score"], reverse=True)
    picks = [r for r in rows if r["score"] > 0][: a.top]
    os.makedirs(".cache", exist_ok=True)
    with open(".cache/scan.json", "w") as f:
        json.dump({"demo": a.demo, "picks": picks, "all": rows}, f, indent=2)
    print(f"{'TICKER':7}{'SCORE':>6}{'PRICE':>10}{'INVEST':>8}{'STOP':>10}{'TARGET':>10}  RISK")
    for r in picks:
        print(f"{r['ticker']:7}{r['score']:>6}{r['price']:>10}{r['invest_dollars']:>8}"
              f"{r['stop']:>10}{r['target']:>10}  {r['risk_level']}")
        print("        " + "; ".join(r["why"]))
    if not picks:
        print("No positive-scoring setups today. Holding cash is a valid trade.")
    if a.demo:
        print("\n(DEMO DATA — prices are synthetic, not real quotes.)")


if __name__ == "__main__":
    main()
