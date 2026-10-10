"""Ledger + order tickets. Claude preps tickets; the user places them in Cash App."""
import argparse
import datetime as dt
import json
import os

from data import history
from risk import analyze

LEDGER = os.path.join(os.getcwd(), "ledger.json")


def load():
    if not os.path.exists(LEDGER):
        raise SystemExit("No ledger.json yet. Run: ticket.py init --cash 10")
    with open(LEDGER) as f:
        return json.load(f)


def save(led):
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def cmd_init(a):
    if os.path.exists(LEDGER) and not a.force:
        raise SystemExit("ledger.json exists; pass --force to reset.")
    save({"start_cash": a.cash, "cash": a.cash, "holdings": {}, "tickets": []})
    print(f"Ledger started with ${a.cash:.2f}. Add the money to Cash App yourself.")


def cmd_prep(a):
    led = load()
    t = a.ticker.upper()
    m = analyze([r["close"] for r in history(t, a.demo)], led["cash"])
    if a.dollars:
        dollars = a.dollars
    elif a.side == "sell" and t in led["holdings"]:
        dollars = led["holdings"][t]["shares"] * m["price"]  # sell everything
    else:
        dollars = m["invest_dollars"]
    if a.side == "buy":
        if dollars > led["cash"] + 1e-9:
            raise SystemExit(f"Only ${led['cash']:.2f} cash available.")
        if dollars < 1:
            raise SystemExit("Cash App minimum buy is $1.")
    elif t not in led["holdings"]:
        raise SystemExit(f"No {t} holding to sell.")
    tk = {
        "id": len(led["tickets"]) + 1, "ticker": t, "side": a.side, "dollars": round(dollars, 2),
        "order": f"custom/limit @ ${a.limit:.2f}" if a.limit else "market (dollar amount)",
        "ref_price": m["price"], "stop": m["stop"], "target": m["target"],
        "loss_at_stop": round(dollars * (1 - m["stop"] / m["price"]), 2),
        "status": "pending", "created": now(), "demo": a.demo,
    }
    led["tickets"].append(tk)
    save(led)
    verb = "Buy" if a.side == "buy" else "Sell"
    print(f"""
ORDER TICKET #{tk['id']}  {'(DEMO PRICES)' if a.demo else ''}
  {verb} ${tk['dollars']:.2f} of ${t}   ({tk['order']})
  Ref price ${tk['ref_price']}   Stop ${tk['stop']}   Target ${tk['target']}
  {'Max expected loss at stop: $%.2f' % tk['loss_at_stop'] if a.side == 'buy' else 'Closing position.'}

In Cash App: Money tab -> Stocks -> search {t} -> {verb} -> enter ${tk['dollars']:.2f}
  {'-> tap ... / Custom order -> set price $%.2f -> ' % a.limit if a.limit else '-> '}confirm with your PIN.
Then tell Claude the fill: ticket.py fill {tk['id']} --price <fill price>
""")


def cmd_fill(a):
    led = load()
    tk = next(x for x in led["tickets"] if x["id"] == a.id)
    if tk["status"] != "pending":
        raise SystemExit(f"Ticket #{a.id} is already {tk['status']}.")
    h = led["holdings"].setdefault(tk["ticker"], {"shares": 0.0, "cost": 0.0, "stop": None, "target": None})
    if tk["side"] == "buy":
        h["shares"] += tk["dollars"] / a.price
        h["cost"] += tk["dollars"]
        h["stop"], h["target"] = tk["stop"], tk["target"]
        led["cash"] -= tk["dollars"]
    else:
        sold = min(h["shares"], tk["dollars"] / a.price)
        frac = sold / h["shares"]
        h["cost"] -= h["cost"] * frac
        h["shares"] -= sold
        led["cash"] += sold * a.price
        if h["shares"] < 1e-9:
            del led["holdings"][tk["ticker"]]
    tk.update(status="filled", fill_price=a.price, filled=now())
    led["cash"] = round(led["cash"], 4)
    save(led)
    print(f"Recorded #{a.id} {tk['side']} {tk['ticker']} @ ${a.price}. Cash now ${led['cash']:.2f}.")


def cmd_cancel(a):
    led = load()
    tk = next(x for x in led["tickets"] if x["id"] == a.id)
    tk["status"] = "cancelled"
    save(led)
    print(f"Ticket #{a.id} cancelled.")


def cmd_status(a):
    led = load()
    total = led["cash"]
    print(f"Cash ${led['cash']:.2f}")
    for t, h in led["holdings"].items():
        px = history(t, a.demo)[-1]["close"]
        val = h["shares"] * px
        total += val
        flag = " <- STOP HIT, prep a sell" if h["stop"] and px <= h["stop"] else \
               " <- TARGET HIT, consider selling" if h["target"] and px >= h["target"] else ""
        print(f"  {t}: {h['shares']:.5f} sh @ ${px} = ${val:.2f} (cost ${h['cost']:.2f}){flag}")
    print(f"Total ${total:.2f}  P&L ${total - led['start_cash']:+.2f}")
    pend = [x for x in led["tickets"] if x["status"] == "pending"]
    for x in pend:
        print(f"  pending #{x['id']}: {x['side']} ${x['dollars']} {x['ticker']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--cash", type=float, default=10.0); s.add_argument("--force", action="store_true")
    s = sub.add_parser("prep"); s.add_argument("ticker"); s.add_argument("--side", choices=["buy", "sell"], default="buy")
    s.add_argument("--dollars", type=float); s.add_argument("--limit", type=float)
    s = sub.add_parser("fill"); s.add_argument("id", type=int); s.add_argument("--price", type=float, required=True)
    s = sub.add_parser("cancel"); s.add_argument("id", type=int)
    sub.add_parser("status")
    a = p.parse_args()
    {"init": cmd_init, "prep": cmd_prep, "fill": cmd_fill, "cancel": cmd_cancel, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
