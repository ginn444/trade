"""Position monitoring and exit rules (stop-loss, take-profit, time stop, EOD flatten, expiry)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from ..config import AppConfig
from ..db import Trade
from ..models import Quote
from ..risk.engine import in_flatten_window


@dataclass
class ExitDecision:
    reason: str
    aggressive: bool
    mark: float | None


def _aware(dt):
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def executable_value(t: Trade, quotes: dict[str, Quote]) -> float | None:
    """Price at which the position could be exited NOW (bid for longs, ask for short legs)."""
    if t.instrument == "stock":
        q = quotes.get(t.underlying)
        return q.bid if q and q.is_valid() else None
    total = 0.0
    for l in t.legs or []:
        q = quotes.get(l["symbol"])
        if q is None or not q.is_valid():
            return None
        total += (q.bid if l["side"] == "buy" else -q.ask) * l.get("ratio", 1)
    return total


def nearest_expiry(t: Trade) -> date | None:
    exps = [date.fromisoformat(l["expiration"]) for l in (t.legs or []) if l.get("expiration")]
    return min(exps) if exps else None


def evaluate_exit(t: Trade, quotes: dict[str, Quote], now: datetime, cfg: AppConfig,
                  kill_flatten: bool = False) -> ExitDecision | None:
    if t.status != "open":
        return None
    mark = executable_value(t, quotes)
    if kill_flatten:
        return ExitDecision("kill_switch_flatten", True, mark)
    if t.intraday_only and in_flatten_window(now, cfg):
        return ExitDecision("eod_flatten", True, mark)
    exp = nearest_expiry(t)
    if exp is not None:
        dte = (exp - now.date()).days
        if not t.intraday_only and dte <= 1:
            return ExitDecision("expiry_risk", True, mark)
        if dte <= 0 and in_flatten_window(now, cfg):
            return ExitDecision("expiry_day_flatten", True, mark)
    if mark is not None:
        # Stocks with a broker-side bracket still get a software backstop if the stop is gapped through.
        if mark <= t.stop_price:
            return ExitDecision("stop_loss", True, mark)
        if mark >= t.target_price:
            return ExitDecision("take_profit", False, mark)
    if t.max_hold_minutes and t.opened_at:
        held = (now - _aware(t.opened_at)).total_seconds() / 60
        if held >= t.max_hold_minutes:
            return ExitDecision("time_stop", False, mark)
    return None


def unrealized_pnl(t: Trade, quotes: dict[str, Quote]) -> float | None:
    if t.status not in ("open", "closing") or t.entry_price is None:
        return None
    v = executable_value(t, quotes)
    if v is None:
        return None
    return (v - t.entry_price) * (t.filled_qty or t.qty) * t.multiplier
