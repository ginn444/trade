"""The autonomous trading loop.

Each cycle:
  1. honour the kill switch (cancel entries; flatten if requested)
  2. read the market clock and account; establish start-of-day equity
  3. sync orders with the broker, cancel stale entries, escalate stuck exits, reconcile
  4. manage open positions (stop / target / time / EOD / expiry exits)
  5. if every gate allows, scan the universe -> strategy candidates -> Claude review ->
     deterministic risk validation -> order submission
  6. snapshot equity and heartbeat; produce the daily report after the close
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ..ai.claude_engine import ClaudeDecisionEngine, match_decisions
from ..backtest.validation import load_approvals
from ..config import AppConfig, TradingMode
from ..data.base import DataUnavailable
from ..db import Database
from ..execution.orders import OrderManager
from ..execution.positions import evaluate_exit, unrealized_pnl
from ..health import HealthMonitor
from ..models import Candidate, InstrumentType, Quote, TradeProposal
from ..options.selector import hold_minutes, select_option_candidate
from ..reporting.daily import build_daily_report
from ..risk.engine import RiskContext, RiskEngine
from ..safety import KillSwitch
from ..strategies.base import Strategy, daily_features, intraday_features
from ..strategies.registry import STRATEGIES

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CycleResult:
    status: str
    candidates: int = 0
    submitted: int = 0
    exits: int = 0
    notes: list[str] = field(default_factory=list)


class TradingBot:
    def __init__(self, cfg: AppConfig, broker, data, db: Database, ai: ClaudeDecisionEngine,
                 kill: KillSwitch, health: HealthMonitor | None = None, clock=_utcnow,
                 live_authorized: bool = False, broker_state_path=None):
        self.cfg = cfg
        self.broker = broker
        self.data = data
        self.db = db
        self.ai = ai
        self.kill = kill
        self.health = health or HealthMonitor(cfg.broker_health)
        self.clock = clock
        self.tz = ZoneInfo(cfg.trading.timezone)
        self.risk = RiskEngine(cfg)
        self.orders = OrderManager(cfg, broker, db, self.health, clock)
        self.live_authorized = live_authorized
        self.broker_state_path = broker_state_path  # mock mode: persist simulated broker between cycles
        self._daily_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._reconcile_ok = True
        self._last_status = "init"
        if cfg.trading.mode == TradingMode.LIVE and not live_authorized:
            raise RuntimeError("TradingBot constructed in live mode without authorization")

    # ================================================================== helpers
    def _timed(self, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            out = fn(*a, **kw)
        except Exception as e:
            self.health.record_error(f"{getattr(fn, '__name__', 'call')}: {e.__class__.__name__}")
            raise
        self.health.record_success(time.monotonic() - t0)
        return out

    def _set_status(self, status: str, **extra) -> None:
        self._last_status = status
        self.db.set_state("bot_status", {"status": status, "ts": self.clock().isoformat(),
                                         "mode": self.cfg.trading.mode.value, "broker": self.broker.name,
                                         "data_source": self.data.source, "health": self.health.status(), **extra})

    def _quote(self, symbol: str) -> Quote | None:
        try:
            if len(symbol) > 15 and symbol[-9] in "CP":
                return self.data.option_quote(symbol)
            return self.data.quote(symbol)
        except Exception as e:
            log.warning("quote unavailable for %s: %s", symbol, e)
            return None

    def _trade_quotes(self, t) -> dict[str, Quote]:
        syms = [l["symbol"] for l in (t.legs or [])] or [t.underlying]
        out = {}
        for s in syms:
            q = self._quote(s)
            if q is not None:
                out[s] = q
        return out

    def approved_strategies(self) -> set[str]:
        approvals = load_approvals(self.cfg.resolve_path(self.cfg.strategies.validation_file))
        ok = set()
        for name, rec in approvals.items():
            if not rec.get("approved"):
                continue
            synthetic = str(rec.get("data_source", "")).upper().startswith("SYNTHETIC")
            if synthetic and self.cfg.trading.mode != TradingMode.MOCK:
                continue  # synthetic-data validation never authorizes paper/live trading
            ok.add(name)
        return ok

    # ================================================================== main loop
    def run_forever(self, max_cycles: int | None = None) -> None:
        n = 0
        backoff = 1
        self.db.audit("bot_start", {"mode": self.cfg.trading.mode.value, "broker": self.broker.name,
                                    "data_source": self.data.source})
        while max_cycles is None or n < max_cycles:
            n += 1
            try:
                res = self.cycle()
                backoff = 1
                sleep_s = self.cfg.trading.loop_interval_seconds if res.status != "market_closed" else 60
            except Exception as e:
                log.error("cycle failed: %s", e)
                self.db.audit("cycle_error", {"error": f"{e.__class__.__name__}: {e}"[:500],
                                              "trace": traceback.format_exc()[-2000:]}, "ERROR")
                sleep_s = min(300, 5 * backoff)
                backoff *= 2
            if max_cycles is None or n < max_cycles:
                time.sleep(sleep_s)

    def cycle(self) -> CycleResult:
        try:
            return self._cycle()
        finally:
            save = getattr(self.broker, "save", None)
            if self.broker_state_path and callable(save):
                save(self.broker_state_path)

    def _cycle(self) -> CycleResult:
        now = self.clock()
        tick = getattr(self.broker, "tick", None)
        if callable(tick):
            tick()

        # ---- 1. kill switch ------------------------------------------------
        if self.kill.is_engaged():
            info = self.kill.info() or {}
            res = CycleResult("killed", notes=[info.get("reason", "")])
            try:
                self.orders.sync()
                self.orders.cancel_all_entries()
                if info.get("flatten"):
                    res.exits = self._manage_positions(now, kill_flatten=True)
            except Exception as e:
                self.db.audit("kill_switch_action_failed", {"error": str(e)[:300]}, "ERROR")
            self._set_status("kill_switch_engaged", kill=info)
            return res

        # ---- 2. clock & account ------------------------------------------
        try:
            clock = self._timed(self.broker.get_clock)
            account = self._timed(self.broker.get_account)
        except Exception as e:
            self._set_status("broker_unreachable", error=e.__class__.__name__)
            return CycleResult("broker_unreachable", notes=[str(e)[:200]])
        today = now.astimezone(self.tz).date().isoformat()
        day_start = self.db.get_state(f"day_start_equity:{today}")
        if day_start is None and clock.is_open:
            day_start = account.last_equity if account.last_equity > 0 else account.equity
            self.db.set_state(f"day_start_equity:{today}", day_start)
            self.db.audit("day_start", {"date": today, "equity": day_start})

        # ---- 3. orders / reconciliation -----------------------------------
        try:
            self.orders.sync()
            self.orders.cancel_stale_entries()
            self.orders.escalate_stuck_exits(self._trade_quotes)
            rec = self.orders.reconcile()
            self._reconcile_ok = rec["ok"]
        except Exception as e:
            self._reconcile_ok = False
            self.db.audit("sync_failed", {"error": f"{e.__class__.__name__}: {e}"[:300]}, "ERROR")

        if not clock.is_open:
            self._after_close(now, account, day_start)
            self._snapshot(account, day_start or account.last_equity)
            self._set_status("market_closed", next_open=clock.next_open.isoformat() if clock.next_open else None)
            return CycleResult("market_closed")

        # ---- 4. positions ---------------------------------------------------
        res = CycleResult("running")
        res.exits = self._manage_positions(now)

        # ---- 5. daily loss gate -------------------------------------------
        dd_pct = (account.equity - day_start) / day_start * 100 if day_start else 0.0
        loss_hit = dd_pct <= -self.cfg.effective_risk().max_daily_loss_pct
        if loss_hit and not self.db.get_state(f"loss_limit_hit:{today}"):
            self.db.set_state(f"loss_limit_hit:{today}", True)
            self.db.audit("daily_loss_limit", {"drawdown_pct": round(dd_pct, 3)}, "WARNING")

        # ---- 6. scan & trade ------------------------------------------------
        gate = self._entry_gate(now, loss_hit)
        if gate is None:
            try:
                sub, cands = self._scan_and_trade(now, account, day_start, clock)
                res.submitted, res.candidates = sub, cands
            except Exception as e:
                self.db.audit("scan_failed", {"error": f"{e.__class__.__name__}: {e}"[:300],
                                              "trace": traceback.format_exc()[-1500:]}, "ERROR")
                res.notes.append("scan_failed")
        else:
            res.notes.append(gate)

        self._snapshot(account, day_start)
        self._set_status("running" if gate is None else f"no_new_entries:{gate}", daily_pnl_pct=round(dd_pct, 3))
        return res

    def _entry_gate(self, now: datetime, loss_hit: bool) -> str | None:
        if loss_hit:
            return "daily_loss_limit"
        if not self.health.healthy:
            return "broker_unhealthy"
        if not self._reconcile_ok and self.cfg.effective_risk().halt_on_reconcile_mismatch:
            return "reconcile_mismatch"
        local = now.astimezone(self.tz)
        if local.time() >= self.cfg.trading.cutoff_time:
            return "past_cutoff"
        open_t = local.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=self.cfg.trading.open_delay_minutes)
        if local < open_t:
            return "open_delay"
        if len(self.db.open_trades()) >= self.cfg.effective_risk().max_open_positions:
            return "max_positions"
        return None

    # ================================================================== positions
    def _manage_positions(self, now: datetime, kill_flatten: bool = False) -> int:
        exits = 0
        for t in self.db.open_trades():
            if t.status != "open":
                continue
            quotes = self._trade_quotes(t)
            d = evaluate_exit(t, quotes, now, self.cfg, kill_flatten=kill_flatten)
            if d is None:
                continue
            ok = self.orders.submit_exit(t, d.reason, quotes, aggressive=d.aggressive)
            exits += int(ok)
        return exits

    # ================================================================== scanning
    def _features(self, symbol: str, strat: Strategy, now: datetime) -> pd.DataFrame | None:
        try:
            if strat.timeframe == "1d":
                key = (symbol, now.astimezone(self.tz).date().isoformat())
                if key not in self._daily_cache:
                    bars = self.data.daily_bars(symbol, self.cfg.data.daily_lookback_days)
                    last = bars.index[-1]
                    if (now - last.to_pydatetime()).days > 5:
                        raise DataUnavailable(f"daily bars stale for {symbol} (last {last.date()})")
                    self._daily_cache = {k: v for k, v in self._daily_cache.items() if k[1] == key[1]}
                    self._daily_cache[key] = daily_features(bars)
                return self._daily_cache[key]
            m = self.cfg.data.intraday_timeframe_minutes
            bars = self.data.intraday_bars(symbol, m, 3)
            age_min = (now - bars.index[-1].to_pydatetime()).total_seconds() / 60 - m
            if age_min > self.cfg.data.max_bar_age_minutes:
                raise DataUnavailable(f"intraday bars stale for {symbol} ({age_min:.0f} min)")
            return intraday_features(bars)
        except DataUnavailable as e:
            self.db.audit("data_unavailable", {"symbol": symbol, "error": str(e)[:200]}, "WARNING")
            return None
        except Exception as e:
            self.db.audit("data_error", {"symbol": symbol, "error": f"{e.__class__.__name__}: {e}"[:200]}, "WARNING")
            return None

    def _build_candidates(self, now: datetime, approved: set[str]) -> list[Candidate]:
        out: list[Candidate] = []
        held = {t.underlying for t in self.db.open_trades()}
        risk = self.cfg.effective_risk()
        for name in self.cfg.strategies.enabled:
            strat = STRATEGIES.get(name)
            if strat is None or (self.cfg.strategies.require_validation and name not in approved):
                continue
            is_opt = strat.instrument != InstrumentType.STOCK
            if is_opt and not self.cfg.universe.options_enabled:
                continue
            symbols = self.cfg.universe.options_underlyings if is_opt else self.cfg.universe.stocks
            for sym in symbols:
                if sym in held:
                    continue
                feats = self._features(sym, strat, now)
                if feats is None:
                    continue
                daily = feats if strat.timeframe == "1d" else self._features(sym, STRATEGIES["stock_momentum"], now)
                if daily is not None:
                    dv = daily["dollar_vol20"].iloc[-1]
                    if pd.isna(dv) or dv < self.cfg.universe.min_avg_dollar_volume or daily["close"].iloc[-1] < self.cfg.universe.min_price:
                        continue
                sig = strat.evaluate(feats)
                if sig is None:
                    continue
                q = self._quote(sym)
                if q is None or not q.is_valid() or q.age_seconds(now) > self.cfg.data.max_quote_age_seconds:
                    self.db.audit("stale_quote_skip", {"symbol": sym, "strategy": name}, "WARNING")
                    continue
                spot = q.mid
                # re-anchor stop/target distances to the live price (signal was computed on the last bar)
                shift = spot - sig.entry
                sig.entry, sig.stop, sig.target = spot, sig.stop + shift, sig.target + shift
                if strat.instrument == InstrumentType.STOCK:
                    out.append(Candidate(name, sym, sig.direction, InstrumentType.STOCK, round(spot, 2),
                                         round(sig.stop, 2), round(sig.target, 2), spot, sig.regime, sig.features,
                                         intraday_only=strat.intraday_only, rationale=sig.rationale,
                                         max_hold_minutes=hold_minutes(strat, sig.max_hold_bars)))
                    continue
                try:
                    chain = self.data.option_chain(sym, strat.option_dte[0], strat.option_dte[1],
                                                   "call" if sig.direction.value == "long" else "put", now.date())
                except Exception as e:
                    self.db.audit("option_chain_unavailable", {"symbol": sym, "error": str(e)[:200]}, "WARNING")
                    continue
                sel = select_option_candidate(strat, sig, sym, spot, chain, risk, now, self.cfg.data.max_quote_age_seconds)
                if sel.candidate is None:
                    self.db.add_signal(signal_key=f"{name}:{sym}:{now.date()}", strategy=name, symbol=sym,
                                       trade_symbol=sym, direction=sig.direction.value,
                                       instrument=strat.instrument.value, regime=sig.regime.value,
                                       outcome="rejected", reasons=sel.reasons, features=sig.features)
                    continue
                out.append(sel.candidate)
        for c in out:
            c.created_at = now  # signal identity (duplicate protection) follows the bot clock
        # best reward/risk first
        out.sort(key=lambda c: -((c.target_price - c.entry_price) / max(c.entry_price - c.stop_price, 1e-9)))
        return out

    def _scan_and_trade(self, now: datetime, account, day_start, clock) -> tuple[int, int]:
        approved = self.approved_strategies()
        if self.cfg.strategies.require_validation and not approved:
            self._set_status("running", note="no validated strategies; scanning disabled")
            return 0, 0
        cands = self._build_candidates(now, approved)
        if not cands:
            return 0, 0
        risk = self.cfg.effective_risk()
        slots = risk.max_open_positions - len(self.db.open_trades())
        cands = cands[: max(0, min(self.cfg.ai.max_candidates_per_cycle, slots * 2))]
        # deterministic proposals + engine-computed max quantity
        proposals: dict[str, TradeProposal] = {}
        max_q: dict[str, int] = {}
        for i, c in enumerate(cands):
            p = TradeProposal(c, 0, c.entry_price, c.stop_price, c.target_price, 0, 0)
            sz = self.risk.size(p, account.equity, account.buying_power)
            proposals[f"c{i}"] = p
            max_q[f"c{i}"] = sz.qty
        # AI review
        decisions = {}
        ai_used = False
        if self.ai.available:
            ctx = {"timestamp": now.isoformat(), "mode": self.cfg.trading.mode.value,
                   "equity": round(account.equity, 2),
                   "daily_pnl_pct": round((account.equity - day_start) / day_start * 100, 3) if day_start else None,
                   "open_positions": [{"symbol": t.underlying, "instrument": t.instrument, "direction": t.direction}
                                      for t in self.db.open_trades()],
                   "max_risk_per_trade_usd": round(account.equity * risk.max_risk_per_trade_pct / 100, 2),
                   "data_source": self.data.source}
            result = self.ai.review([(cid, p.candidate, max_q[cid]) for cid, p in proposals.items()], ctx)
            if result.review is not None:
                decisions = match_decisions(result.review, list(proposals))
                ai_used = True
                self.db.audit("ai_review", {"latency_s": round(result.latency_s, 2), "model": result.model,
                                            "summary": result.review.market_summary[:500], "n": len(decisions)})
            else:
                self.db.audit("ai_unavailable", {"error": result.error}, "WARNING")
        if not ai_used and self.cfg.ai.enabled and self.cfg.ai.on_ai_unavailable == "skip":
            for cid, p in proposals.items():
                self._record_signal(p, "ai_skip", ["AI unavailable; policy=skip"])
            return 0, len(cands)

        submitted = 0
        for cid, p in proposals.items():
            c = p.candidate
            d = decisions.get(cid)
            if ai_used:
                if d is None:
                    self._record_signal(p, "ai_skip", ["no AI decision for candidate"])
                    continue
                if d.action != "enter":
                    self._record_signal(p, "ai_skip", [d.explanation[:300]], d)
                    continue
                p.entry_price, p.stop_price, p.target_price = d.entry_price, d.stop_loss, d.profit_target
                p.quantity = d.position_size
                p.ai_decision, p.decided_by = d, "claude"
            else:
                p.quantity = max_q[cid]
            quotes = {}
            for s in ([l.symbol for l in c.legs] or [c.symbol]):
                q = self._quote(s)
                if q is not None:
                    quotes[s] = q
            open_trades = self.db.open_trades()
            try:
                open_orders = self._timed(self.broker.get_open_orders)
            except Exception:
                open_orders = None
            rctx = RiskContext(
                now=now, account=account, day_start_equity=day_start, open_trades=open_trades,
                open_broker_orders=open_orders or [], quotes=quotes, kill_switch_engaged=self.kill.is_engaged(),
                broker_healthy=self.health.healthy and open_orders is not None, market_open=clock.is_open,
                session_open_time=now.astimezone(self.tz).replace(hour=9, minute=30, second=0, microsecond=0),
                approved_strategies=approved,
                entry_orders_today=self.db.count_entry_orders_since(
                    now.astimezone(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)),
                duplicate_signal=(self.db.trade_by_client_id(p.client_order_id) is not None
                                  or self.db.signal_traded_recently(
                                      c.signal_key(), timedelta(minutes=risk.duplicate_signal_window_minutes))),
                live_authorized=self.live_authorized, reconcile_ok=self._reconcile_ok)
            verdict = self.risk.validate(p, rctx)
            if not verdict.approved:
                self._record_signal(p, "rejected", verdict.reasons, d, verdict.checks)
                continue
            self._record_signal(p, "approved", [], d, verdict.checks)
            trade = self.orders.submit_entry(p, verdict.adjusted_quantity)
            if trade is not None:
                submitted += 1
                # refresh account (buying power) after each submission
                try:
                    account = self._timed(self.broker.get_account)
                except Exception:
                    break
        return submitted, len(cands)

    def _record_signal(self, p: TradeProposal, outcome: str, reasons: list[str], d=None, checks=None) -> None:
        c = p.candidate
        self.db.add_signal(signal_key=c.signal_key(), strategy=c.strategy, symbol=c.symbol, trade_symbol=c.trade_symbol,
                           direction=c.direction.value, instrument=c.instrument.value, regime=c.regime.value,
                           outcome=outcome, reasons=reasons,
                           ai_explanation=d.explanation if d else None,
                           ai_decision=d.model_dump() if d else None,
                           features={**c.features, "entry": p.entry_price, "stop": p.stop_price,
                                     "target": p.target_price, "qty": p.quantity, "risk_usd": p.risk_usd},
                           risk_checks=checks)

    # ================================================================== bookkeeping
    def _snapshot(self, account, day_start) -> None:
        risk_in_use = 0.0
        for t in self.db.open_trades():
            risk_in_use += t.risk_usd or 0
        ds = day_start or account.last_equity or account.equity
        self.db.add_equity_snapshot(equity=account.equity, buying_power=account.buying_power, cash=account.cash,
                                    day_start_equity=ds, daily_pnl=account.equity - ds,
                                    open_positions=len(self.db.open_trades()), risk_in_use_usd=risk_in_use)

    def _after_close(self, now: datetime, account, day_start) -> None:
        local = now.astimezone(self.tz)
        today = local.date()
        if local.hour < 16 or today.weekday() >= 5:
            return
        if self.db.get_state(f"report_done:{today.isoformat()}"):
            return
        report = build_daily_report(self.db, self.cfg, today, account, day_start)
        self.db.save_report(today, report)
        out = self.cfg.resolve_path(self.cfg.storage.reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        import json
        (out / f"daily_{self.cfg.trading.mode.value}_{today.isoformat()}.json").write_text(json.dumps(report, indent=2, default=str))
        self.db.set_state(f"report_done:{today.isoformat()}", True)
        self.db.audit("daily_report", {"date": today.isoformat(), "net_pnl": report.get("day_pnl")})
        end = getattr(self.broker, "end_of_day", None)
        if callable(end):
            end()

    def portfolio_view(self) -> list[dict]:
        rows = []
        for t in self.db.open_trades():
            q = self._trade_quotes(t)
            rows.append({"id": t.id, "symbol": t.trade_symbol, "unrealized_pnl": unrealized_pnl(t, q)})
        return rows
