"""Command-line interface: `autotrader <command>`."""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from .config import PROJECT_ROOT, Credentials, TradingMode, load_config
from .logging_setup import setup_logging

log = logging.getLogger("autotrader")


def _setup(args):
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(args.config)
    creds = Credentials()
    setup_logging(cfg.resolve_path(cfg.storage.log_dir), creds.secret_values())
    return cfg, creds


def cmd_run(args) -> int:
    cfg, creds = _setup(args)
    from .engine.bot import TradingBot
    from .engine.factory import build_components
    c = build_components(cfg, creds)
    log.info("starting bot mode=%s broker=%s data=%s ai=%s", cfg.trading.mode.value, c.broker.name,
             c.data.source, "on" if c.ai.available else "off")
    if cfg.trading.mode == TradingMode.MOCK:
        log.warning("MOCK MODE: synthetic data + simulated broker. Results are NOT real.")
    bot = TradingBot(cfg, c.broker, c.data, c.db, c.ai, c.kill, c.health, live_authorized=c.live_authorized,
                     broker_state_path=c.broker_state_path)
    bot.run_forever(max_cycles=args.cycles)
    return 0


def cmd_backtest(args) -> int:
    cfg, creds = _setup(args)
    from .backtest.validation import evaluate, record_result
    from .backtest.walkforward import research_strategy
    from .strategies.registry import STRATEGIES
    names = args.strategy or cfg.strategies.enabled
    if args.source == "alpaca":
        from .data.alpaca_data import AlpacaMarketData
        key, secret = creds.alpaca(TradingMode.PAPER)  # data access only; paper keys suffice
        data = AlpacaMarketData(key, secret, cfg.data.stock_feed, cfg.data.options_feed)
    else:
        from .data.synthetic import SyntheticMarketData
        data = SyntheticMarketData(seed=args.seed)
        print("WARNING: SYNTHETIC data — results demonstrate the pipeline only and cannot approve "
              "strategies for paper/live trading.", file=sys.stderr)
    out_dir = cfg.resolve_path(cfg.storage.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    approvals_path = cfg.resolve_path(cfg.strategies.validation_file)
    summary = []
    for name in names:
        strat = STRATEGIES[name]
        is_opt = strat.instrument.value != "stock"
        syms = args.symbols or (cfg.universe.options_underlyings if is_opt else cfg.universe.stocks)
        bars = {}
        for s in syms:
            try:
                if strat.timeframe == "5m":
                    bars[s] = data.intraday_bars(s, cfg.data.intraday_timeframe_minutes, cfg.backtest.intraday_history_days)
                elif args.source == "alpaca":
                    bars[s] = data.daily_bars(s, int(args.years * 252))
                else:
                    bars[s] = data.full_history(s).tail(int(args.years * 252))
            except Exception as e:
                log.warning("no data for %s: %s", s, e)
        report = research_strategy(strat, bars, cfg, data.source)
        ok, reasons = evaluate(report, cfg.backtest.acceptance)
        report["acceptance"] = {"approved": ok, "reasons": reasons}
        record_result(approvals_path, report, ok, reasons)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out_dir / f"backtest_{name}_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
        wf = report.get("walk_forward", {})
        summary.append((name, ok, wf.get("trades"), wf.get("win_rate"), wf.get("profit_factor"),
                        wf.get("expectancy_r"), wf.get("max_drawdown_pct"), wf.get("sharpe"),
                        report.get("benchmark_buy_hold", {}).get("sharpe"), "; ".join(reasons)[:120]))
    print(f"\nData source: {data.source}   (out-of-sample walk-forward results)")
    print(f"{'strategy':22} {'pass':5} {'trades':>6} {'win%':>6} {'PF':>6} {'expR':>7} {'maxDD%':>7} {'Sharpe':>7} {'B&H Sh':>7}  reasons")
    for n, ok, tr, wr, pf, er, dd, sh, bsh, why in summary:
        print(f"{n:22} {'YES' if ok else 'no':5} {tr or 0:>6} {100*(wr or 0):>6.1f} {pf or 0:>6.2f} {er or 0:>7.3f} "
              f"{dd or 0:>7.2f} {sh or 0:>7.2f} {bsh or 0:>7.2f}  {why}")
    return 0


def cmd_kill(args) -> int:
    cfg, _ = _setup(args)
    from .db import Database
    from .safety import KillSwitch
    KillSwitch(cfg.state_dir).engage(args.reason, actor=f"cli:{getpass.getuser()}", flatten=args.flatten)
    Database(cfg.database_url, cfg.trading.mode.value).audit("kill_switch_engaged",
                                                             {"reason": args.reason, "flatten": args.flatten, "actor": "cli"}, "WARNING")
    print("KILL SWITCH ENGAGED" + (" (flatten positions)" if args.flatten else ""))
    return 0


def cmd_resume(args) -> int:
    cfg, _ = _setup(args)
    from .db import Database
    from .safety import KillSwitch
    KillSwitch(cfg.state_dir).release(actor="cli")
    Database(cfg.database_url, cfg.trading.mode.value).audit("kill_switch_released", {"actor": "cli"}, "WARNING")
    print("kill switch released")
    return 0


def cmd_status(args) -> int:
    cfg, _ = _setup(args)
    from .backtest.validation import load_approvals
    from .db import Database
    from .safety import KillSwitch
    db = Database(cfg.database_url, cfg.trading.mode.value)
    print(json.dumps({
        "mode": cfg.trading.mode.value,
        "kill_switch": KillSwitch(cfg.state_dir).info(),
        "bot_status": db.get_state("bot_status"),
        "open_trades": [{"id": t.id, "symbol": t.trade_symbol, "status": t.status} for t in db.open_trades()],
        "strategy_approvals": {k: {"approved": v.get("approved"), "data_source": v.get("data_source")}
                               for k, v in load_approvals(cfg.resolve_path(cfg.strategies.validation_file)).items()},
    }, indent=2, default=str))
    return 0


def cmd_dashboard(args) -> int:
    cfg, creds = _setup(args)
    import uvicorn
    from .dashboard.app import create_app
    if not creds.dashboard_token:
        print("DASHBOARD_TOKEN is not set: the emergency-stop endpoint will refuse all requests.", file=sys.stderr)
    uvicorn.run(create_app(cfg, creds), host=cfg.dashboard.host, port=cfg.dashboard.port, log_level="warning")
    return 0


def cmd_authorize_live(args) -> int:
    cfg, creds = _setup(args)
    from .safety import CONFIRM_PHRASE, assert_live_authorized, write_live_authorization
    print("=" * 72)
    print("LIVE TRADING AUTHORIZATION — REAL MONEY")
    print(f"config fingerprint: {cfg.fingerprint()[:16]}…  mode={cfg.trading.mode.value}  live.enabled={cfg.live.enabled}")
    r = cfg.effective_risk()
    print(f"effective limits: risk/trade {r.max_risk_per_trade_pct}%  daily loss {r.max_daily_loss_pct}%  "
          f"max positions {r.max_open_positions}  orders/day {r.max_orders_per_day}")
    print(f"authorization expires after {cfg.live.authorization_ttl_hours}h; any config change invalidates it.")
    print("=" * 72)
    creds.alpaca(TradingMode.LIVE)  # fail early if live keys missing / equal to paper
    if not sys.stdin.isatty():
        print("refusing: authorization must be performed interactively", file=sys.stderr)
        return 2
    phrase = input(f'Type exactly "{CONFIRM_PHRASE}" to continue: ')
    path = write_live_authorization(cfg, getpass.getuser(), phrase)
    assert_live_authorized(cfg, creds)
    print(f"live trading authorized; written to {path}")
    return 0


def cmd_report(args) -> int:
    cfg, _ = _setup(args)
    from .db import Database
    db = Database(cfg.database_url, cfg.trading.mode.value)
    reps = db.reports(1)
    print(json.dumps(reps[0].report if reps else {"message": "no reports yet"}, indent=2, default=str))
    return 0


def cmd_simulate(args) -> int:
    """MOCK: run the full bot over N consecutive simulated sessions with an accelerated clock."""
    cfg = load_config(args.config, overrides={"trading": {"mode": "mock"}, "ai": {"enabled": not args.no_ai}})
    load_dotenv(PROJECT_ROOT / ".env")
    creds = Credentials()
    setup_logging(None, creds.secret_values(), level=logging.WARNING)
    from .ai.claude_engine import ClaudeDecisionEngine
    from .broker.simulated import SimulatedBroker
    from .data.synthetic import NY, SyntheticMarketData
    from .db import Database
    from .engine.bot import TradingBot
    from .health import HealthMonitor
    from .safety import KillSwitch
    day = datetime.fromisoformat(args.start).date() if args.start else (datetime.now(NY).date() - timedelta(days=7))
    state = {"now": datetime(day.year, day.month, day.day, 9, 0, tzinfo=NY).astimezone(timezone.utc)}
    clock = lambda: state["now"]  # noqa: E731
    data = SyntheticMarketData(seed=args.seed, clock=clock)
    broker = SimulatedBroker(data, cfg.backtest.initial_equity, clock=clock)
    broker_path = cfg.state_dir / "sim_broker.json"
    broker.load(broker_path)
    db = Database(cfg.database_url, "mock", clock=clock)
    bot = TradingBot(cfg, broker, data, db, ClaudeDecisionEngine(cfg.ai, creds.anthropic_api_key),
                     KillSwitch(cfg.state_dir), HealthMonitor(cfg.broker_health), clock=clock,
                     broker_state_path=broker_path)
    print("SIMULATION — synthetic data, simulated fills. Results are NOT real.")
    sessions = 0
    while sessions < args.days:
        if day.weekday() < 5:
            state["now"] = datetime(day.year, day.month, day.day, 9, 31, tzinfo=NY).astimezone(timezone.utc)
            end = datetime(day.year, day.month, day.day, 16, 5, tzinfo=NY).astimezone(timezone.utc)
            while state["now"] <= end:
                r = bot.cycle()
                if r.submitted or r.exits:
                    print(f"{state['now'].astimezone(NY):%Y-%m-%d %H:%M} entries={r.submitted} exits={r.exits}")
                state["now"] += timedelta(minutes=args.step)
            rep = db.reports(1)
            if rep and rep[0].day == day.isoformat():
                rr = rep[0].report
                print(f"  {day} equity {rr['end_equity']:.2f}  day P&L {rr['day_pnl']}  closed {rr['trades_closed']}  "
                      f"open {rr['open_positions']}  signals {rr['signals']}")
            sessions += 1
        day += timedelta(days=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autotrader")
    ap.add_argument("--config", help="overlay config file (default: config/default.yaml only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run the autonomous trading loop")
    p.add_argument("--cycles", type=int, default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("backtest", help="research + validate strategies (gates autonomous trading)")
    p.add_argument("--strategy", action="append")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--source", choices=["alpaca", "synthetic"], default="alpaca")
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_backtest)
    p = sub.add_parser("kill", help="engage the persistent emergency kill switch")
    p.add_argument("--reason", default="manual")
    p.add_argument("--flatten", action="store_true", help="also close all open positions")
    p.set_defaults(fn=cmd_kill)
    sub.add_parser("resume", help="release the kill switch").set_defaults(fn=cmd_resume)
    sub.add_parser("status", help="print bot status").set_defaults(fn=cmd_status)
    sub.add_parser("dashboard", help="serve the monitoring dashboard").set_defaults(fn=cmd_dashboard)
    sub.add_parser("authorize-live", help="explicit real-money authorization step").set_defaults(fn=cmd_authorize_live)
    sub.add_parser("report", help="print the latest daily report").set_defaults(fn=cmd_report)
    p = sub.add_parser("simulate", help="MOCK: run accelerated synthetic sessions end-to-end")
    p.add_argument("--start", help="first session date (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--step", type=int, default=5, help="minutes per cycle")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-ai", action="store_true")
    p.set_defaults(fn=cmd_simulate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
