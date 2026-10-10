"""Price history loader: Yahoo Finance chart API, CSV fallback, or synthetic demo data."""
import csv
import json
import math
import os
import random
import urllib.request

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ", "KO", "JPM", "COST"]
CACHE_DIR = os.path.join(os.getcwd(), ".cache", "prices")


def _yahoo(ticker, rng="1y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={rng}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.load(r)["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    rows = [(t, c) for t, c in zip(res["timestamp"], closes) if c is not None]
    return [{"t": t, "close": round(c, 4)} for t, c in rows]


def _csv(path):
    with open(path) as f:
        return [{"t": r.get("Date") or r.get("date"), "close": float(r.get("Close") or r.get("close"))}
                for r in csv.DictReader(f)]


def _demo(ticker, n=252):
    rnd = random.Random(ticker)
    price = rnd.uniform(30, 400)
    drift, vol = rnd.uniform(-0.0005, 0.0015), rnd.uniform(0.01, 0.035)
    out = []
    for i in range(n):
        price *= math.exp(drift + vol * rnd.gauss(0, 1))
        out.append({"t": i, "close": round(price, 2)})
    return out


def history(ticker, demo=False, csv_path=None):
    """Return list of {t, close}, oldest first."""
    ticker = ticker.upper()
    if demo:
        return _demo(ticker)
    if csv_path:
        return _csv(csv_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{ticker}.json")
    try:
        rows = _yahoo(ticker)
        with open(cache, "w") as f:
            json.dump(rows, f)
        return rows
    except Exception as e:
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)
        raise SystemExit(f"Could not fetch {ticker}: {e}. Re-run with --demo or --csv FILE.")
