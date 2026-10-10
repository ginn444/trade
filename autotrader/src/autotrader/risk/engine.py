"""Deterministic risk engine.

Every order intent — whether proposed by Claude or by a deterministic strategy — must pass
``RiskEngine.validate`` before an order is submitted. The engine:

* recomputes position size itself from equity and stop distance (AI-proposed size can only be
  reduced, never increased);
* checks every hard limit (kill switch, connectivity, daily loss, position count, duplicates,
  stale data, liquidity, structure of option orders, strategy validation status, trading window);
* fails closed: any missing input is a rejection.

It has no dependency on the AI module and never parses free text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ..config import AppConfig, RiskCfg, TradingMode
from ..models import AccountState, Direction, InstrumentType, Quote, RiskVerdict, TradeProposal


@dataclass
class RiskContext:
    now: datetime
    account: AccountState | None
    day_start_equity: float | None
    open_trades: list = field(default_factory=list)          # db.Trade rows in open statuses
    open_broker_orders: list = field(default_factory=list)   # BrokerOrder
    quotes: dict[str, Quote] = field(default_factory=dict)   # by tradable symbol (legs individually)
    kill_switch_engaged: bool = True
    broker_healthy: bool = False
    market_open: bool = False
    session_open_time: datetime | None = None
    approved_strategies: set[str] = field(default_factory=set)
    entry_orders_today: int = 0
    duplicate_signal: bool = False
    live_authorized: bool = False
    reconcile_ok: bool = True


@dataclass
class SizeResult:
    qty: int
    per_unit_risk: float
    risk_usd: float
    est_costs_usd: float
    multiplier: int
    notes: list[str]


class RiskEngine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mode = cfg.trading.mode
        self.tz = ZoneInfo(cfg.trading.timezone)

    @property
    def risk(self) -> RiskCfg:
        return self.cfg.effective_risk()

    # ------------------------------------------------------------------ sizing
    def size(self, p: TradeProposal, equity: float, buying_power: float) -> SizeResult:
        r = self.risk
        c = p.candidate
        budget = equity * r.max_risk_per_trade_pct / 100.0
        notes: list[str] = []
        if c.instrument == InstrumentType.STOCK:
            mult = 1
            slip = p.entry_price * r.stop_slippage_pct_stock / 100.0
            spread_cost = p.entry_price * self.cfg.backtest.stock_spread_bps / 10_000
            per_unit = (p.entry_price - p.stop_price) + slip + spread_cost
            unit_cost = p.entry_price
            fees_per_unit = self.cfg.backtest.stock_commission_per_share * 2
        else:
            mult = 100
            n_legs = max(len(c.legs), 1)
            fees_per_unit = (self.cfg.backtest.option_regulatory_fee_per_contract
                             + self.cfg.backtest.option_commission_per_contract) * n_legs * 2
            worst_entry = self._worst_entry_premium(p)
            if r.options_risk_full_premium or c.instrument == InstrumentType.DEBIT_SPREAD:
                per_unit = worst_entry * mult       # whole premium can be lost (gaps, stop slippage)
                notes.append("option risk = full premium")
            else:
                per_unit = (worst_entry - p.stop_price) * mult * 1.5  # 50% gap allowance
            unit_cost = worst_entry * mult
        per_unit += fees_per_unit
        if per_unit <= 0:
            return SizeResult(0, per_unit, 0, 0, mult, ["non-positive per-unit risk"])
        qty = math.floor(budget / per_unit)
        cap_notional = math.floor(equity * r.max_position_notional_pct / 100.0 / unit_cost) if unit_cost > 0 else 0
        usable_bp = buying_power * (1 - r.min_buying_power_buffer_pct / 100.0)
        cap_bp = math.floor(usable_bp / unit_cost) if unit_cost > 0 else 0
        if cap_notional < qty:
            notes.append(f"capped by notional limit ({cap_notional})")
        if cap_bp < qty:
            notes.append(f"capped by buying power ({cap_bp})")
        qty = max(0, min(qty, cap_notional, cap_bp))
        est_costs = fees_per_unit * qty
        return SizeResult(qty, per_unit, per_unit * qty, est_costs, mult, notes)

    def _worst_entry_premium(self, p: TradeProposal) -> float:
        c = p.candidate
        if c.instrument == InstrumentType.DEBIT_SPREAD and len(c.legs) == 2:
            buy = next((l for l in c.legs if l.side == "buy"), None)
            sell = next((l for l in c.legs if l.side == "sell"), None)
            if buy and sell and buy.contract and sell.contract and buy.contract.quote and sell.contract.quote:
                return max(p.entry_price, buy.contract.quote.ask - sell.contract.quote.bid)
        return p.entry_price

    # ------------------------------------------------------------------ validation
    def validate(self, p: TradeProposal, ctx: RiskContext) -> RiskVerdict:
        r = self.risk
        c = p.candidate
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        def check(name: str, ok: bool, why: str) -> None:
            checks[name] = bool(ok)
            if not ok:
                reasons.append(why)

        check("kill_switch_off", not ctx.kill_switch_engaged, "emergency kill switch is engaged")
        if self.mode == TradingMode.LIVE:
            check("live_authorized", ctx.live_authorized, "live trading not authorized")
        check("broker_healthy", ctx.broker_healthy, "brokerage connectivity unreliable")
        check("reconciled", ctx.reconcile_ok or not r.halt_on_reconcile_mismatch,
              "order/position reconciliation mismatch unresolved")
        acct = ctx.account
        check("account_available", acct is not None and acct.equity > 0, "account state unavailable")
        if acct is None or acct.equity <= 0:
            return RiskVerdict(False, reasons, 0, checks)
        check("account_not_blocked", not acct.trading_blocked, "account trading blocked")

        # ---- trading window
        check("market_open", ctx.market_open, "market closed")
        local = ctx.now.astimezone(self.tz)
        if ctx.session_open_time is not None:
            earliest = ctx.session_open_time.astimezone(self.tz).timestamp() + self.cfg.trading.open_delay_minutes * 60
            check("after_open_delay", local.timestamp() >= earliest, "inside post-open delay")
        check("before_cutoff", local.time() < self.cfg.trading.cutoff_time,
              f"past new-entry cutoff {self.cfg.trading.new_entry_cutoff}")

        # ---- loss limits / activity limits
        day_start = ctx.day_start_equity or acct.last_equity
        check("day_start_known", bool(day_start and day_start > 0), "start-of-day equity unknown")
        if day_start and day_start > 0:
            dd = (acct.equity - day_start) / day_start * 100
            check("daily_loss_ok", dd > -r.max_daily_loss_pct,
                  f"daily loss limit reached ({dd:.2f}% <= -{r.max_daily_loss_pct}%)")
        check("orders_per_day_ok", ctx.entry_orders_today < r.max_orders_per_day,
              f"max entry orders per day reached ({r.max_orders_per_day})")

        # ---- exposure
        n_open = len(ctx.open_trades)
        check("max_positions_ok", n_open < r.max_open_positions,
              f"max simultaneous positions reached ({n_open}/{r.max_open_positions})")
        same = [t for t in ctx.open_trades if t.underlying == c.symbol]
        check("no_add_to_existing", not same,
              f"already exposed to {c.symbol} (no averaging down / pyramiding)")
        pending_syms = {o.symbol for o in ctx.open_broker_orders}
        leg_syms = {l.symbol for l in c.legs} or {c.symbol}
        check("no_pending_same_symbol", not (pending_syms & (leg_syms | {c.symbol})),
              "an open order already exists for this instrument")
        check("not_duplicate", not ctx.duplicate_signal, "duplicate signal already acted on")

        # ---- strategy approval
        need_val = self.cfg.strategies.require_validation
        check("strategy_enabled", c.strategy in self.cfg.strategies.enabled, f"strategy {c.strategy} not enabled")
        if need_val:
            check("strategy_validated", c.strategy in ctx.approved_strategies,
                  f"strategy {c.strategy} has not passed backtest validation")

        # ---- instrument structure (defined risk only)
        self._check_structure(p, check)

        # ---- AI output consistency (AI may never change the instrument or loosen anything)
        if p.ai_decision is not None:
            d = p.ai_decision
            check("ai_action_enter", d.action == "enter", "AI recommended skip")
            check("ai_matches_candidate",
                  d.ticker.upper() == c.symbol.upper() and d.instrument == c.instrument.value
                  and d.direction == c.direction.value and d.strategy == c.strategy
                  and d.instrument_symbol.replace(" ", "") == c.trade_symbol,
                  "AI decision does not match the candidate instrument/direction/strategy")

        # ---- data freshness & liquidity
        symbols = [l.symbol for l in c.legs] if c.legs else [c.symbol]
        max_age = self.cfg.data.max_quote_age_seconds
        for s in symbols:
            q = ctx.quotes.get(s)
            if q is None or not q.is_valid():
                check(f"quote_{s}", False, f"missing/invalid quote for {s}")
                continue
            check(f"fresh_{s}", q.age_seconds(ctx.now) <= max_age,
                  f"stale quote for {s} ({q.age_seconds(ctx.now):.0f}s > {max_age}s)")
            lim = r.max_spread_pct_stock if c.instrument == InstrumentType.STOCK else r.max_spread_pct_option
            check(f"spread_{s}", q.spread_pct <= lim, f"{s} spread {q.spread_pct:.2f}% > {lim}%")

        # ---- price sanity: entry must be near the live market (blocks hallucinated prices)
        ref = self._reference_price(p, ctx)
        if ref is None:
            check("reference_price", False, "cannot compute reference price")
        else:
            tol = 0.005 if c.instrument == InstrumentType.STOCK else 0.10
            check("entry_near_market", abs(p.entry_price - ref) / ref <= tol,
                  f"entry {p.entry_price:.2f} too far from market {ref:.2f}")

        # ---- stop/target geometry (premium terms for options are always long-premium)
        long_like = c.instrument != InstrumentType.STOCK or c.direction == Direction.LONG
        if long_like:
            geom = 0 < p.stop_price < p.entry_price < p.target_price
        else:
            geom = False
        check("stop_target_geometry", geom, "stop/entry/target ordering invalid")
        if geom:
            rr = (p.target_price - p.entry_price) / (p.entry_price - p.stop_price)
            check("reward_risk_ok", rr >= r.min_reward_risk, f"reward/risk {rr:.2f} < {r.min_reward_risk}")

        # ---- sizing (engine's own computation is authoritative)
        sz = self.size(p, acct.equity, acct.buying_power)
        qty = min(sz.qty, p.quantity) if p.quantity > 0 else sz.qty
        check("size_positive", qty >= 1, "position size rounds to zero within risk limits")
        risk_usd = sz.per_unit_risk * qty
        budget = acct.equity * r.max_risk_per_trade_pct / 100.0
        check("risk_within_budget", risk_usd <= budget + 1e-6,
              f"risk ${risk_usd:.2f} exceeds budget ${budget:.2f}")
        p.risk_usd = round(risk_usd, 2)
        p.est_costs_usd = round(sz.est_costs_usd * (qty / sz.qty) if sz.qty else 0, 2)

        approved = all(checks.values())
        return RiskVerdict(approved, reasons, qty if approved else 0, checks)

    def _check_structure(self, p: TradeProposal, check) -> None:
        c = p.candidate
        if c.instrument == InstrumentType.STOCK:
            check("stock_long_only", c.direction == Direction.LONG,
                  "short stock prohibited (unlimited theoretical loss)")
            check("stock_no_legs", not c.legs, "stock trade must not carry option legs")
            return
        if c.instrument == InstrumentType.OPTION:
            ok = len(c.legs) == 1 and c.legs[0].side == "buy" and c.legs[0].contract is not None
            if ok:
                k = c.legs[0].contract.option_type
                ok = (k == "call") == (c.direction == Direction.LONG)
            check("single_long_option", ok, "single-option trades must be one LONG call (bullish) or put (bearish)")
            return
        if c.instrument == InstrumentType.DEBIT_SPREAD:
            ok = len(c.legs) == 2 and all(l.contract is not None for l in c.legs)
            if ok:
                buy = [l for l in c.legs if l.side == "buy"]
                sell = [l for l in c.legs if l.side == "sell"]
                ok = len(buy) == 1 and len(sell) == 1 and buy[0].ratio == sell[0].ratio
                if ok:
                    b, s = buy[0].contract, sell[0].contract
                    ok = (b.option_type == s.option_type and b.expiration == s.expiration and b.underlying == s.underlying
                          and ((b.option_type == "call" and s.strike > b.strike and c.direction == Direction.LONG)
                               or (b.option_type == "put" and s.strike < b.strike and c.direction == Direction.SHORT)))
            check("defined_risk_debit_spread", ok,
                  "debit spread must be long-nearer/short-further, same type & expiry (defined risk)")
            return
        check("known_instrument", False, f"unsupported instrument {c.instrument}")

    def _reference_price(self, p: TradeProposal, ctx: RiskContext) -> float | None:
        c = p.candidate
        if c.instrument == InstrumentType.STOCK:
            q = ctx.quotes.get(c.symbol)
            return q.mid if q else None
        total = 0.0
        for l in c.legs:
            q = ctx.quotes.get(l.symbol)
            if q is None:
                return None
            total += q.mid * l.ratio * (1 if l.side == "buy" else -1)
        return total if total > 0 else None


def session_open_from_clock(now: datetime, tz: str = "America/New_York") -> datetime:
    local = now.astimezone(ZoneInfo(tz))
    return local.replace(hour=9, minute=30, second=0, microsecond=0)


def in_flatten_window(now: datetime, cfg: AppConfig) -> bool:
    return now.astimezone(ZoneInfo(cfg.trading.timezone)).time() >= cfg.trading.flatten_time


def time_of_day(now: datetime, cfg: AppConfig) -> time:
    return now.astimezone(ZoneInfo(cfg.trading.timezone)).time()
