"""Build dashboard.html: animated price charts, risk gauges, and portfolio for the scan picks + holdings."""
import argparse
import json
import os

from data import history
from risk import analyze

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", help="comma separated; default = scan picks + holdings")
    p.add_argument("--out", default="dashboard.html")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    led = None
    if os.path.exists("ledger.json"):
        with open("ledger.json") as f:
            led = json.load(f)
    tickers = a.tickers.split(",") if a.tickers else []
    if not tickers and os.path.exists(".cache/scan.json"):
        with open(".cache/scan.json") as f:
            tickers = [r["ticker"] for r in json.load(f)["picks"]]
    if led:
        tickers += [t for t in led["holdings"] if t not in tickers]
    if not tickers:
        raise SystemExit("Nothing to chart. Run scan.py first or pass --tickers.")

    budget = led["cash"] if led else 10.0
    cards = []
    for t in tickers:
        closes = [r["close"] for r in history(t, a.demo)][-120:]
        m = analyze(closes, max(budget, 1.0))
        h = (led or {}).get("holdings", {}).get(t.upper())
        if h:
            m["stop"], m["target"] = h["stop"] or m["stop"], h["target"] or m["target"]
        cards.append({"ticker": t.upper(), "closes": closes, "m": m, "holding": h})

    payload = {"demo": a.demo, "cards": cards, "ledger": led}
    with open(os.path.join(HERE, "dashboard_template.html")) as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"Wrote {a.out} ({len(cards)} tickers). Open it in a browser.")


if __name__ == "__main__":
    main()
