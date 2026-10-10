---
name: cash-app-trader
description: Manage a small (e.g. $10) Cash App Investing stock budget. Scans a watchlist for trade ideas, calculates risk and position size, renders an animated stock chart dashboard, and preps ready-to-enter order tickets the user places themselves in Cash App. Use when the user asks to find trades, check risk, size a position, review their Cash App portfolio, or update the trade ledger.
---

# Cash App Trader

Helps the user run a tiny stock budget (default **$10 USD**) in **Cash App Investing**.

## Hard limits — say these plainly, never pretend otherwise

- Cash App has **no public trading API**. Claude cannot log in, hold money, move money,
  or place orders. Claude **finds and preps** trades; **the user taps Buy/Sell** in Cash App.
- This is not financial advice. Every ticket can lose money. With $10, fees are $0 in
  Cash App for standard orders, but spreads and price moves still apply.
- Never ask for Cash App login, PIN, $Cashtag sign-in codes, or card numbers.

## Files

- `ledger.json` (repo root, created on first use) — cash, holdings, and every ticket/trade.
- `scripts/data.py` — price history (Yahoo Finance chart API; CSV fallback; `--demo` synthetic).
- `scripts/scan.py` — ranks a watchlist and writes trade candidates.
- `scripts/risk.py` — risk metrics + position sizing for one ticker.
- `scripts/ticket.py` — preps an order ticket and records it in the ledger.
- `scripts/dashboard.py` — builds `dashboard.html`: animated price charts, risk, portfolio.

All scripts are Python 3 stdlib only. Run them from the repo root, e.g.
`python3 .claude/skills/cash-app-trader/scripts/scan.py`. Add `--demo` when no
internet access to market data (e.g. sandboxed sessions) — say clearly that demo
numbers are fake.

## Workflow

1. **Fund check.** Read `ledger.json` (`python3 scripts/ticket.py status`). If missing,
   init with the user's budget: `ticket.py init --cash 10`. The user moves the real
   $10 into Cash App themselves.
2. **Find trades.** `scan.py [--tickers AAPL,MSFT,...] [--risk low|medium|high]`.
   It scores trend (20/50-day SMA), momentum (1-month return), and volatility,
   and drops anything whose risk exceeds the profile. Present the top 3 with the reasons.
3. **Calculate risk.** For each pick: `risk.py TICKER --budget <cash>`. Report:
   annualized volatility, max drawdown, 1-day 95% VaR in dollars, suggested stop
   price, take-profit price, reward/risk ratio, and dollar amount to invest.
   Default rule: risk at most **2% of the account per idea** to the stop, capped at
   the cash available; Cash App fractional buys start at **$1**, so if sizing says < $1, skip.
4. **Prep tickets.** `ticket.py prep TICKER --side buy --dollars 3.00` prints a ticket:
   ticker, side, dollar amount, order type (market dollar-buy, or a custom/limit price),
   stop & target to watch, and the exact Cash App taps. It records it as `pending`.
5. **User executes.** User places it in Cash App, then tells you the fill.
   Record it: `ticket.py fill <id> --price 187.42` (or `ticket.py cancel <id>`).
6. **Animate & review.** `dashboard.py` → open `dashboard.html` in a browser
   (animated line charts drawing in, stop/target bands, risk gauges, P&L). In cloud
   sessions, publish it as an Artifact so the user can view it on their phone.
7. **Exit check.** On later runs, compare current price vs each holding's stop and
   target; prep a `--side sell` ticket when one is hit. Cash App may not support stop
   orders on every position — tell the user to set a price alert or check back.

## Style

Keep it short and phone-friendly: one table of picks, one ticket per trade, a single
line on overall account risk. Always repeat that the user places the trade themselves.
