# autotrader

An autonomous, risk-gated trading bot for **US stocks, ETFs and options** on **Alpaca**, with **Claude** as an
analysis layer and a **deterministic risk engine** that has the final say on every order.

* **Paper trading by default.** Live trading is disabled. It needs separate credentials, its own config,
  tighter limits, and an explicit, expiring, config-bound authorization step.
* **No strategy trades until it passes backtest validation** on real historical data: out-of-sample
  walk-forward results with realistic costs, compared against buy-and-hold.
* **Claude can't bypass risk controls.** Its structured output is re-validated, and position size is
  recomputed by the engine (it can only shrink). Any AI error, refusal or malformed output means *no trade*.
* **Defined risk only.** Stocks are long-only. Options are long calls, long puts, or debit spreads. No naked
  or uncovered short options, and no unlimited-loss structures.

> ⚠️ Nothing in this repository is a claim that any strategy is profitable. The included backtests on
> **synthetic** data only show that the pipeline works. Validate on real data, then paper-trade, before
> drawing conclusions. Not financial advice.

---

## Architecture

```
                 ┌────────────────────────── TradingBot.cycle() (every 30s) ──────────────────────────┐
 Kill switch ──► │ 1 kill switch? → cancel entries (+flatten)                                          │
 (file, persists)│ 2 broker clock + account → start-of-day equity                                      │
                 │ 3 sync orders · cancel stale entries · escalate stuck exits · reconcile vs broker  │
                 │ 4 manage positions: stop / target / time stop / EOD flatten / expiry               │
                 │ 5 gates: daily-loss · connectivity · reconciliation · cutoff · open delay · slots  │
                 │ 6 scan: strategies(validated only) → option selector → candidates                  │
                 │        → Claude review (structured AIReview) → RiskEngine.validate → OrderManager  │
                 │ 7 equity snapshot · heartbeat · daily report after close                           │
                 └────────────────────────────────────────────────────────────────────────────────────┘
   data/            broker/             risk/engine.py        execution/          db.py (SQLite/any SQLAlchemy URL)
   alpaca_data.py   alpaca_broker.py    deterministic checks  orders.py (WAL,     trades · orders · signals ·
   synthetic.py*    simulated.py*       + sizing              idempotent ids,     equity · audit_log · state ·
                                                              reconciliation)     daily_reports
   * clearly labeled MOCK integrations       dashboard/ (FastAPI + static page, reads DB; kill-switch POST)
```

| Package | Responsibility |
|---|---|
| `config.py` | Pydantic-validated YAML config + env-only secrets; paper/live credentials never fall back |
| `safety.py` | Persistent kill switch; live-trading authorization gate |
| `health.py` | Broker connectivity monitor (errors/latency). Unhealthy means no new orders |
| `data/` | `MarketDataProvider`: Alpaca stocks+options (real) and `SyntheticMarketData` (mock) |
| `broker/` | `Broker`: Alpaca paper/live (endpoint verified) and `SimulatedBroker` (mock) |
| `analysis/` | Causal indicators, regime classifier (bullish/bearish/neutral), Black-Scholes greeks/IV |
| `strategies/` | Signal rules + instrument mapping; the same objects drive live trading and backtests |
| `options/selector.py` | Contract selection by DTE, delta band, spread %, open interest, volume; debit spreads |
| `ai/claude_engine.py` | Claude structured review (`messages.parse` → `AIReview`), fail-closed |
| `risk/engine.py` | Deterministic validator + position sizing |
| `execution/` | Write-ahead order log, duplicate protection, sync, reconciliation, exit rules |
| `backtest/` | Simulator, cost model, portfolio replay, metrics, walk-forward, acceptance gate |
| `reporting/`, `dashboard/` | Daily report; live monitoring dashboard |

### Strategies

| Name | Signal | Instrument |
|---|---|---|
| `stock_momentum` | trend continuation (ADX, aligned EMAs, ROC, MACD) | long stock |
| `stock_breakout` | 20-day Donchian breakout on relative volume | long stock |
| `stock_reversal` | RSI(2) pullback within a 200-day uptrend | long stock |
| `option_momentum` | momentum, both directions | long call / long put, Δ≈0.50, 21–45 DTE |
| `option_breakout` | breakout / breakdown | long call / put |
| `option_reversal` | short-term exhaustion | long call / put |
| `option_debit_spread` | momentum | bull call / bear put debit spread (max loss = debit) |
| `option_volatility` | Bollinger-width compression → expansion | long call / put (buy convexity when vol is cheap) |
| `option_scalp` | 5-min EMA cross vs VWAP, intraday only | 1–7 DTE long call / put, flat by 15:50 ET |

### Risk controls (all in `config/default.yaml`, enforced in `risk/engine.py`)

| Control | Default (paper) | Live (tighter, applied on top) |
|---|---|---|
| Max risk per trade | 0.5% of equity | 0.25% |
| Max daily loss (realized + unrealized vs start-of-day) | 2% → no new trades | 1% |
| Max simultaneous positions | 3 | 2 |
| Max position notional | 20% of equity | 10% |
| Averaging down / pyramiding / martingale | prohibited (config refuses `true`) | prohibited |
| Stale or missing quotes | rejected (> 60 s) | same |
| Bid-ask spread | ≤ 0.15% stock, ≤ 8% option | same |
| Option liquidity | OI ≥ 500, volume ≥ 50 (when the feed reports it), Δ 0.30–0.65 | same |
| Option risk | full premium assumed lost (gaps; stops may fill far worse) | same |
| Stock stop slippage | +0.25% of price added to per-share risk | same |
| Entry price sanity | within 0.5% (stock) / 10% (option) of live mid. Blocks hallucinated prices | same |
| Duplicate orders | deterministic `client_order_id` + DB uniqueness + signal window | same |
| Broker connectivity | 3 consecutive or 5-in-5-min errors → no new orders until recovered | same |
| Reconciliation mismatch | halts new entries | same |
| New-entry cutoff | 15:30 ET; intraday strategies flattened 15:50 ET | same |

Exits are never blocked by these gates, so capital protection keeps running when entries stop. Before selling,
an exit checks that the broker actually holds the position, so a double exit can never create a short.

---

## Installation

```bash
cd autotrader
./scripts/setup.sh                 # venv + pinned deps (requirements.lock.txt) + test suite
# or: pip install -r requirements.lock.txt && pip install --no-deps -e .
cp .env.example .env && chmod 600 .env   # then fill in keys
```

Requires Python ≥ 3.11 (built and tested on 3.13). Secrets are read **only** from environment variables or `.env`:

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude review (optional. Without it, `ai.on_ai_unavailable: skip` means no trades) |
| `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET_KEY` | paper trading and historical data for backtests |
| `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY` | live only. Must differ from the paper keys |
| `DASHBOARD_TOKEN` | required for the dashboard's emergency-stop / resume buttons |

Log output passes through a redaction filter that scrubs configured secrets and key-shaped strings.

## Operating workflow

```bash
# 0. Try everything offline (synthetic data + simulated broker; clearly labeled MOCK)
autotrader backtest --source synthetic          # pipeline demo; can NOT approve paper/live trading
AUTOTRADER_MODE=mock autotrader simulate --days 10 --no-ai
autotrader dashboard                             # http://127.0.0.1:8080

# 1. Research & validate on real Alpaca history (writes state/strategy_approvals.json + reports/)
autotrader backtest --source alpaca --years 5

# 2. Paper trade autonomously (only strategies that passed step 1 will trade)
autotrader run                                   # or: docker compose up -d
autotrader status
autotrader report                                # latest daily report (also in reports/)

# Emergency stop (persists across restarts until released)
autotrader kill --reason "manual" [--flatten]
autotrader resume
```

The bot waits for the market to open (Alpaca clock), trades between open+5 min and 15:30 ET, manages
positions continuously, and writes a daily report after the close. No individual trade needs approval
inside the paper environment.

### Backtesting methodology (`backtest/`)

* **No look-ahead:** indicators are causal (tested by comparing prefix and full-series computations). A
  strategy sees `feats.iloc[:i+1]`, and a signal on bar *i* fills at bar *i+1*'s **open**.
* **Pessimistic fills:** if a bar touches both stop and target, the stop is assumed to fill first. A gap
  through the stop fills at the open. Option stops get extra slippage.
* **Costs:** stock slippage + half-spread (bps) + commission; option half-spread % + slippage % +
  per-contract fees, on both legs of spreads.
* **Portfolio replay** uses the live sizing rules (0.5% risk, 3 positions, notional cap).
* **Out-of-sample:** a 70/30 in-sample/holdout split with default params, plus walk-forward optimisation
  (train 252 bars → test 63, rolling) over each strategy's small parameter grid.
* **Benchmark:** equal-weight buy-and-hold of the same universe over the out-of-sample span.
* **Reported:** trades, win rate, profit factor, expectancy ($ and R), avg win/loss, total return, CAGR,
  max drawdown, Sharpe, Sortino, positive-window fraction.
* **Acceptance** (`backtest.acceptance`): ≥30 OOS trades, PF ≥ 1.2, max DD ≤ 20%, expectancy ≥ 0.05R,
  Sharpe ≥ 0.5, ≥ 50% positive walk-forward windows, and OOS Sharpe not degraded > 60% vs in-sample.
  Strategies that fail are not traded.
* **Option prices in backtests are MODELED** (Black-Scholes, IV = realized vol × 1.10, constant over the
  trade), because free historical option quotes are limited. Reports say `option_pricing:
  modeled_black_scholes`. Treat option backtests as approximate until checked against real option history.

## Live trading (disabled by default)

There is no code path from paper to live:

1. Create `config/live.yaml` from `config/live.example.yaml` (`trading.mode: live`, `live.enabled: true`).
2. Set `ALPACA_LIVE_*` keys. They must differ from the paper keys, and paper mode never reads them.
3. Run `autotrader --config config/live.yaml authorize-live` interactively and type the confirmation phrase.
   This writes an authorization bound to the **SHA-256 of the full config**. It expires after
   `authorization_ttl_hours` (24 by default), and any config edit invalidates it.
4. `autotrader --config config/live.yaml run`. `AUTOTRADER_MODE` can never select live.

Live mode applies the tighter `live.risk` limits on top of the normal ones. The live broker class refuses to
construct without authorization and verifies that it points at the live endpoint, not paper. The same check
in reverse protects paper mode.

## Testing

`pytest` runs 87 offline tests covering indicators (causality), options math, config and credential
separation, the live gate, kill-switch persistence, log redaction, every risk-engine rule, the order manager
(write-ahead, duplicates, ambiguous submit recovery, bracket detection, no double exits, reconciliation,
stale entries), exit rules, the AI engine (structured parsing, fail-closed on errors/refusals), the backtest
(no look-ahead, next-bar fills, pessimistic stops, costs, portfolio limits, metrics, acceptance), end-to-end
bot cycles (position cap, AI skip, AI can't oversize or move prices, kill switch + flatten, daily-loss stop,
unhealthy broker, stale data, cutoff, EOD report), and the dashboard (token-protected kill switch).

## What is and isn't verified

| Verified here (offline) | **Not** verified (needs your credentials / market hours) |
|---|---|
| All logic above via 87 automated tests | Real Alpaca paper order placement, fills, and bracket/`mleg` behaviour |
| Full bot loop over multi-day **synthetic** sessions with a **simulated** broker | Alpaca market-data responses (field availability, e.g. option volume / greeks on the `indicative` feed) |
| Backtest/walk-forward pipeline on **synthetic** data | Any strategy's performance on **real** data. Run `backtest --source alpaca` |
| Dashboard rendering (light/dark, phone width) | Live Claude API calls (the request shape matches SDK `anthropic==1.13`; tests use a fake client) |
| Alpaca SDK request/response field names (introspected from `alpaca-py==0.44`) | Live trading of any kind |

Known limitations: market holidays and half-days come from Alpaca's clock in paper/live (mock mode assumes
regular weekday hours). The portfolio drawdown in backtests uses realized equity. Alpaca's `indicative` option
feed may omit some fields, and the selector rejects contracts whose open interest is unknown.
