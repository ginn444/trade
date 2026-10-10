"""Order management: write-ahead logging, idempotent submission, sync, reconciliation and exits.

Duplicate-order protection is layered:
1. deterministic ``client_order_id`` derived from the signal (broker rejects re-use);
2. DB uniqueness on ``client_order_id`` for trades and orders;
3. the risk engine's duplicate-signal and pending-order checks.

Every order is written to the DB *before* it is sent. If submission fails ambiguously (timeout,
connection reset) the broker is queried by client_order_id to learn whether it actually landed.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from ..broker.base import BrokerError, DuplicateOrderError, LegSpec, OrderSpec
from ..config import AppConfig
from ..db import Database, Trade
from ..health import HealthMonitor
from ..models import BrokerOrder, InstrumentType, Quote, TradeProposal

log = logging.getLogger(__name__)
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day", "stopped", "suspended"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt):
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class OrderManager:
    def __init__(self, cfg: AppConfig, broker, db: Database, health: HealthMonitor, clock=_now):
        self.cfg = cfg
        self.broker = broker
        self.db = db
        self.health = health
        self.clock = clock

    # ------------------------------------------------------------------ broker call wrapper
    def _call(self, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            out = fn(*a, **kw)
        except (DuplicateOrderError,):
            self.health.record_success(time.monotonic() - t0)
            raise
        except Exception as e:
            self.health.record_error(f"{getattr(fn, '__name__', 'call')}: {e.__class__.__name__}")
            raise
        self.health.record_success(time.monotonic() - t0)
        return out

    # ------------------------------------------------------------------ entries
    def build_entry_spec(self, p: TradeProposal, qty: int) -> OrderSpec:
        c = p.candidate
        off = self.cfg.execution.limit_offset_pct / 100.0
        if c.instrument == InstrumentType.STOCK:
            limit = round(p.entry_price * (1 + off), 2)
            if self.cfg.execution.use_bracket_for_stocks:
                return OrderSpec(p.client_order_id, c.symbol, "buy", qty, "limit", limit, "day", "bracket",
                                 take_profit=round(p.target_price, 2), stop_loss=round(p.stop_price, 2))
            return OrderSpec(p.client_order_id, c.symbol, "buy", qty, "limit", limit)
        if c.instrument == InstrumentType.OPTION:
            leg = c.legs[0]
            ask = leg.contract.quote.ask if leg.contract and leg.contract.quote else p.entry_price
            limit = round(min(p.entry_price * (1 + off), ask), 2)  # never pay above the ask
            return OrderSpec(p.client_order_id, leg.symbol, "buy", qty, "limit", limit, "day",
                             position_intent="buy_to_open")
        legs = [LegSpec(l.symbol, l.side, l.ratio, "buy_to_open" if l.side == "buy" else "sell_to_open") for l in c.legs]
        return OrderSpec(p.client_order_id, "", "buy", qty, "limit", round(p.entry_price, 2), "day", "mleg", legs=legs)

    def submit_entry(self, p: TradeProposal, qty: int) -> Trade | None:
        c = p.candidate
        if self.db.trade_by_client_id(p.client_order_id) is not None:
            self.db.audit("duplicate_blocked", {"client_order_id": p.client_order_id}, "WARNING")
            return None
        spec = self.build_entry_spec(p, qty)
        spec.validate()
        trade = self.db.add_trade(
            signal_key=c.signal_key(), client_order_id=p.client_order_id, strategy=c.strategy,
            underlying=c.symbol, trade_symbol=c.trade_symbol, instrument=c.instrument.value,
            direction=c.direction.value, qty=qty, planned_entry=p.entry_price, stop_price=p.stop_price,
            target_price=p.target_price, risk_usd=p.risk_usd, est_costs_usd=p.est_costs_usd,
            multiplier=1 if c.instrument == InstrumentType.STOCK else 100,
            legs=[{"symbol": l.symbol, "side": l.side, "ratio": l.ratio,
                   "type": l.contract.option_type if l.contract else None,
                   "strike": l.contract.strike if l.contract else None,
                   "expiration": l.contract.expiration.isoformat() if l.contract else None} for l in c.legs] or None,
            status="pending_entry", max_hold_minutes=c.max_hold_minutes, intraday_only=c.intraday_only,
            decided_by=p.decided_by, ai_explanation=p.ai_decision.explanation if p.ai_decision else c.rationale)
        self.db.add_order(client_order_id=spec.client_order_id, trade_id=trade.id, purpose="entry",
                          symbol=spec.symbol or c.trade_symbol, side=spec.side, qty=qty, order_type=spec.order_type,
                          limit_price=spec.limit_price, status="intent",
                          payload={"order_class": spec.order_class, "take_profit": spec.take_profit,
                                   "stop_loss": spec.stop_loss, "legs": [l.__dict__ for l in spec.legs]})
        bo = self._submit(spec)
        if bo is None:
            self.db.update_trade(trade.id, status="cancelled", exit_reason="submit_failed", closed_at=self.clock())
            return None
        self.db.update_trade(trade.id, entry_order_id=bo.id)
        self.db.audit("entry_submitted", {"trade_id": trade.id, "client_order_id": spec.client_order_id,
                                          "symbol": c.trade_symbol, "qty": qty, "limit": spec.limit_price,
                                          "risk_usd": p.risk_usd, "broker": self.broker.name})
        self._apply_order_update(spec.client_order_id, bo)
        return self.db.get_trade(trade.id)

    def _submit(self, spec: OrderSpec) -> BrokerOrder | None:
        try:
            bo = self._call(self.broker.submit_order, spec)
            self.db.update_order(spec.client_order_id, status="submitted", broker_order_id=bo.id)
            return bo
        except DuplicateOrderError:
            existing = self._lookup(spec.client_order_id)
            self.db.audit("duplicate_client_order_id", {"client_order_id": spec.client_order_id,
                                                         "found": existing is not None}, "WARNING")
            if existing:
                self.db.update_order(spec.client_order_id, status="submitted", broker_order_id=existing.id)
            return existing
        except BrokerError as e:
            self.db.update_order(spec.client_order_id, status="submit_failed", error=str(e)[:500])
            self.db.audit("order_rejected", {"client_order_id": spec.client_order_id, "error": str(e)[:500]}, "ERROR")
            return None
        except Exception as e:  # ambiguous failure: did the order reach the broker?
            existing = self._lookup(spec.client_order_id)
            if existing is not None:
                self.db.update_order(spec.client_order_id, status="submitted", broker_order_id=existing.id)
                self.db.audit("ambiguous_submit_resolved", {"client_order_id": spec.client_order_id}, "WARNING")
                return existing
            self.db.update_order(spec.client_order_id, status="submit_failed", error=f"{e.__class__.__name__}: {e}"[:500])
            self.db.audit("order_submit_error", {"client_order_id": spec.client_order_id,
                                                 "error": e.__class__.__name__}, "ERROR")
            return None

    def _lookup(self, client_order_id: str) -> BrokerOrder | None:
        try:
            return self._call(self.broker.get_order_by_client_id, client_order_id)
        except Exception:
            return None

    # ------------------------------------------------------------------ sync
    def sync(self) -> None:
        """Pull the latest broker state for every unresolved order and update trades."""
        for rec in self.db.unresolved_orders():
            if rec.status == "intent":
                # Written but never confirmed sent (crash between write and submit): ask the broker.
                bo = self._lookup(rec.client_order_id)
                if bo is None:
                    if (self.clock() - _aware(rec.created_at)) > timedelta(minutes=2):
                        self.db.update_order(rec.client_order_id, status="submit_failed", error="never reached broker")
                        if rec.trade_id and rec.purpose == "entry":
                            self.db.update_trade(rec.trade_id, status="cancelled", exit_reason="never_submitted",
                                                 closed_at=self.clock())
                    continue
            else:
                bo = self._lookup(rec.client_order_id)
                if bo is None:
                    continue
            self._apply_order_update(rec.client_order_id, bo)
        # Broker-side bracket children (stop / take-profit) can fill at any time: re-check them every cycle.
        for t in self.db.open_trades():
            if t.instrument == "stock" and t.status == "open" and t.entry_order_id:
                bo = self._lookup(t.client_order_id)
                if bo is not None and bo.legs:
                    self._apply_order_update(t.client_order_id, bo)

    def _held_qty(self) -> dict[str, float] | None:
        try:
            return {p.symbol: p.qty for p in self._call(self.broker.get_positions)}
        except Exception:
            return None

    def _apply_order_update(self, client_order_id: str, bo: BrokerOrder) -> None:
        rec = self.db.update_order(client_order_id, status=bo.status, broker_order_id=bo.id,
                                   filled_qty=bo.filled_qty, filled_avg_price=bo.filled_avg_price)
        if rec is None or rec.trade_id is None:
            return
        t = self.db.get_trade(rec.trade_id)
        if t is None:
            return
        if rec.purpose == "entry":
            if bo.status == "filled" and t.status == "pending_entry":
                self.db.update_trade(t.id, status="open", entry_price=bo.filled_avg_price, filled_qty=bo.filled_qty,
                                     opened_at=self.clock())
                self.db.audit("entry_filled", {"trade_id": t.id, "price": bo.filled_avg_price, "qty": bo.filled_qty})
            elif bo.status in TERMINAL and bo.status != "filled" and t.status == "pending_entry":
                if bo.filled_qty and bo.filled_qty > 0:
                    self.db.update_trade(t.id, status="open", entry_price=bo.filled_avg_price,
                                         filled_qty=bo.filled_qty, qty=int(bo.filled_qty), opened_at=self.clock())
                    self.db.audit("entry_partial", {"trade_id": t.id, "filled": bo.filled_qty})
                else:
                    self.db.update_trade(t.id, status="cancelled", exit_reason=f"entry_{bo.status}",
                                         closed_at=self.clock())
                    self.db.audit("entry_unfilled", {"trade_id": t.id, "status": bo.status})
            # broker-side bracket child filled => position closed at the broker
            if t.status in ("open", "pending_entry") and bo.legs:
                for leg in bo.legs:
                    if leg.status == "filled" and leg.side == "sell" and leg.filled_avg_price:
                        reason = "bracket_stop" if (leg.order_type in ("stop", "stop_limit")) else "bracket_target"
                        self._close_trade(self.db.get_trade(t.id), leg.filled_avg_price, reason)
                        break
        elif rec.purpose == "exit":
            if bo.status == "filled" and t.status == "closing":
                self._close_trade(t, bo.filled_avg_price, t.exit_reason or "exit")
            elif bo.status in TERMINAL and bo.status != "filled" and t.status == "closing":
                # exit did not complete: back to open so the position manager retries
                self.db.update_trade(t.id, status="open", exit_order_id=None, exit_client_order_id=None)
                self.db.audit("exit_unfilled_retry", {"trade_id": t.id, "status": bo.status}, "WARNING")

    def _close_trade(self, t: Trade, exit_price: float | None, reason: str) -> None:
        if exit_price is None or t.entry_price is None:
            self.db.update_trade(t.id, status="closed", exit_reason=reason, closed_at=self.clock())
            return
        qty = t.filled_qty or t.qty
        pnl = (exit_price - t.entry_price) * qty * t.multiplier
        fees = (self.cfg.backtest.option_regulatory_fee_per_contract * len(t.legs or [1]) * 2 * qty
                if t.multiplier == 100 else 0.0)
        pnl -= fees
        self.db.update_trade(t.id, status="closed", exit_price=exit_price, realized_pnl=round(pnl, 2),
                             exit_reason=reason, closed_at=self.clock())
        self.db.audit("trade_closed", {"trade_id": t.id, "exit_price": exit_price, "pnl": round(pnl, 2), "reason": reason})

    # ------------------------------------------------------------------ stale entries
    def cancel_stale_entries(self) -> None:
        timeout = timedelta(seconds=self.cfg.execution.entry_timeout_seconds)
        for t in self.db.open_trades():
            if t.status != "pending_entry" or not t.entry_order_id:
                continue
            if self.clock() - _aware(t.created_at) > timeout:
                try:
                    self._call(self.broker.cancel_order, t.entry_order_id)
                    self.db.audit("entry_timeout_cancel", {"trade_id": t.id})
                except Exception as e:
                    self.db.audit("cancel_failed", {"trade_id": t.id, "error": str(e)[:200]}, "ERROR")

    # ------------------------------------------------------------------ exits
    def submit_exit(self, t: Trade, reason: str, quotes: dict[str, Quote], aggressive: bool = False) -> bool:
        if t.status != "open":
            return False
        # Never sell what the broker does not hold (prevents double exits creating shorts).
        held = self._held_qty()
        if held is None:
            self.db.audit("exit_deferred_no_positions", {"trade_id": t.id}, "WARNING")
            return False
        qty_needed = float(t.filled_qty or t.qty)
        need = ({t.underlying: qty_needed} if t.instrument == "stock" else
                {l["symbol"]: qty_needed * l.get("ratio", 1) * (1 if l["side"] == "buy" else -1) for l in (t.legs or [])})
        for sym, q in need.items():
            h = held.get(sym, 0.0)
            if (q > 0 and h < q - 1e-9) or (q < 0 and h > q + 1e-9):
                self.db.audit("exit_blocked_position_mismatch",
                              {"trade_id": t.id, "symbol": sym, "expected": q, "broker": h, "reason": reason}, "ERROR")
                return False
        n = int(self.db.get_state(f"exit_attempts:{t.id}", 0)) + 1
        self.db.set_state(f"exit_attempts:{t.id}", n)
        aggressive = aggressive or n > 1
        # cancel broker-side bracket children first so we never double-sell
        if t.instrument == "stock" and t.entry_order_id:
            bo = self._lookup(t.client_order_id)
            for leg in (bo.legs if bo else []):
                if leg.is_open:
                    try:
                        self._call(self.broker.cancel_order, leg.id)
                    except Exception as e:
                        self.db.audit("bracket_cancel_failed", {"trade_id": t.id, "error": str(e)[:200]}, "ERROR")
                        return False
        qty = int(t.filled_qty or t.qty)
        cid = f"{t.client_order_id}-x{n}"
        off = self.cfg.execution.limit_offset_pct / 100.0
        try:
            if t.instrument == "stock":
                q = quotes.get(t.underlying)
                spec = (OrderSpec(cid, t.underlying, "sell", qty, "market", time_in_force="day") if (aggressive or q is None)
                        else OrderSpec(cid, t.underlying, "sell", qty, "limit", round(q.bid * (1 - off), 2)))
            elif t.instrument == "option":
                sym = t.legs[0]["symbol"]
                q = quotes.get(sym)
                if q is None or not q.is_valid():
                    spec = OrderSpec(cid, sym, "sell", qty, "market", position_intent="sell_to_close")
                else:
                    px = q.bid if aggressive else max(q.bid, round(q.mid * (1 - off), 2))
                    spec = OrderSpec(cid, sym, "sell", qty, "limit", round(max(px, 0.01), 2),
                                     position_intent="sell_to_close")
            else:  # debit spread: close both legs together
                legs = [LegSpec(l["symbol"], "sell" if l["side"] == "buy" else "buy", l.get("ratio", 1),
                                "sell_to_close" if l["side"] == "buy" else "buy_to_close") for l in t.legs]
                credit = 0.0
                for l in t.legs:
                    q = quotes.get(l["symbol"])
                    if q is None:
                        credit = None
                        break
                    credit += (q.bid if l["side"] == "buy" else -q.ask) if aggressive else (q.mid if l["side"] == "buy" else -q.mid)
                if credit is None:
                    spec = OrderSpec(cid, "", "sell", qty, "market", order_class="mleg", legs=legs)
                else:
                    spec = OrderSpec(cid, "", "sell", qty, "limit", round(max(credit, 0.01), 2), order_class="mleg", legs=legs)
        except (KeyError, IndexError, TypeError) as e:
            self.db.audit("exit_build_failed", {"trade_id": t.id, "error": str(e)}, "ERROR")
            return False
        self.db.add_order(client_order_id=cid, trade_id=t.id, purpose="exit", symbol=spec.symbol or t.trade_symbol,
                          side=spec.side, qty=qty, order_type=spec.order_type, limit_price=spec.limit_price,
                          status="intent", payload={"reason": reason, "aggressive": aggressive})
        self.db.update_trade(t.id, status="closing", exit_reason=reason, exit_client_order_id=cid,
                             exit_requested_at=self.clock())
        bo = self._submit(spec)
        if bo is None:
            self.db.update_trade(t.id, status="open", exit_client_order_id=None)
            return False
        self.db.update_trade(t.id, exit_order_id=bo.id)
        self.db.audit("exit_submitted", {"trade_id": t.id, "reason": reason, "client_order_id": cid,
                                         "type": spec.order_type, "limit": spec.limit_price})
        self._apply_order_update(cid, bo)
        return True

    def escalate_stuck_exits(self, quotes_fn) -> None:
        """Exits that have not filled after N seconds are cancelled; the next cycle re-submits aggressively."""
        wait = timedelta(seconds=self.cfg.execution.option_exit_aggressive_after_seconds)
        for t in self.db.open_trades():
            if t.status == "closing" and t.exit_order_id and t.exit_requested_at \
                    and self.clock() - _aware(t.exit_requested_at) > wait:
                try:
                    self._call(self.broker.cancel_order, t.exit_order_id)
                    self.db.audit("exit_escalate", {"trade_id": t.id})
                except Exception as e:
                    self.db.audit("exit_cancel_failed", {"trade_id": t.id, "error": str(e)[:200]}, "ERROR")

    # ------------------------------------------------------------------ reconciliation
    def reconcile(self) -> dict:
        """Compare DB open trades with broker positions/orders. Returns a report; mismatches are audited."""
        positions = {p.symbol: p for p in self._call(self.broker.get_positions)}
        open_orders = self._call(self.broker.get_open_orders)
        report = {"unknown_positions": [], "missing_positions": [], "qty_mismatch": [], "ok": True}
        expected: dict[str, float] = {}
        for t in self.db.open_trades():
            if t.status == "pending_entry":
                continue
            q = float(t.filled_qty or t.qty)
            if t.instrument == "stock":
                expected[t.underlying] = expected.get(t.underlying, 0) + q
            else:
                for l in t.legs or []:
                    expected[l["symbol"]] = expected.get(l["symbol"], 0) + q * l.get("ratio", 1) * (1 if l["side"] == "buy" else -1)
        for sym, q in expected.items():
            p = positions.get(sym)
            if p is None:
                report["missing_positions"].append(sym)
            elif abs(p.qty - q) > 1e-6:
                report["qty_mismatch"].append({"symbol": sym, "db": q, "broker": p.qty})
        for sym in positions:
            if sym not in expected:
                report["unknown_positions"].append(sym)
        report["ok"] = not (report["unknown_positions"] or report["missing_positions"] or report["qty_mismatch"])
        report["open_orders"] = len(open_orders)
        if not report["ok"]:
            self.db.audit("reconcile_mismatch", report, "WARNING")
        return report

    def cancel_all_entries(self) -> None:
        for t in self.db.open_trades():
            if t.status == "pending_entry" and t.entry_order_id:
                try:
                    self._call(self.broker.cancel_order, t.entry_order_id)
                except Exception as e:
                    self.db.audit("cancel_failed", {"trade_id": t.id, "error": str(e)[:200]}, "ERROR")
