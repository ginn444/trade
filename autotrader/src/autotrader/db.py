"""Persistence: positions/trades, orders, signals, equity snapshots, audit log, bot state."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, func, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; everything we store is UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """One managed position from entry intent to exit."""
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), index=True)
    signal_key: Mapped[str] = mapped_column(String(64), index=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    underlying: Mapped[str] = mapped_column(String(16), index=True)
    trade_symbol: Mapped[str] = mapped_column(String(128))
    instrument: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    qty: Mapped[int] = mapped_column(Integer)
    filled_qty: Mapped[float] = mapped_column(Float, default=0)
    planned_entry: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    risk_usd: Mapped[float] = mapped_column(Float)
    est_costs_usd: Mapped[float] = mapped_column(Float, default=0)
    multiplier: Mapped[int] = mapped_column(Integer, default=1)
    legs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending_entry|open|closing|closed|cancelled|error
    max_hold_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    intraday_only: Mapped[bool] = mapped_column(Boolean, default=False)
    entry_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    decided_by: Mapped[str] = mapped_column(String(16), default="deterministic")
    ai_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OrderRecord(Base):
    """Write-ahead order log: a row is written BEFORE submission, then updated from the broker."""
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), index=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    purpose: Mapped[str] = mapped_column(String(8))  # entry|exit|cancel
    symbol: Mapped[str] = mapped_column(String(128))
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(16))
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)  # intent|submitted|<broker status>|submit_failed
    filled_qty: Mapped[float] = mapped_column(Float, default=0)
    filled_avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class SignalRecord(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    signal_key: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(16))
    trade_symbol: Mapped[str] = mapped_column(String(128))
    direction: Mapped[str] = mapped_column(String(8))
    instrument: Mapped[str] = mapped_column(String(16))
    regime: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[str] = mapped_column(String(16))  # approved|rejected|ai_skip|error
    reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ai_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    risk_checks: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    equity: Mapped[float] = mapped_column(Float)
    buying_power: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    day_start_equity: Mapped[float] = mapped_column(Float)
    daily_pnl: Mapped[float] = mapped_column(Float)
    open_positions: Mapped[int] = mapped_column(Integer)
    risk_in_use_usd: Mapped[float] = mapped_column(Float, default=0)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    level: Mapped[str] = mapped_column(String(8))
    event: Mapped[str] = mapped_column(String(64), index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BotState(Base):
    __tablename__ = "bot_state"
    __table_args__ = (UniqueConstraint("mode", "key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8))
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class DailyReport(Base):
    __tablename__ = "daily_reports"
    __table_args__ = (UniqueConstraint("mode", "day"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8))
    day: Mapped[str] = mapped_column(String(10))
    report: Mapped[dict] = mapped_column(JSON)


OPEN_TRADE_STATUSES = ("pending_entry", "open", "closing")


class Database:
    def __init__(self, url: str, mode: str, clock=None):
        if url.startswith("sqlite:///") and ":memory:" not in url:
            Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                from sqlalchemy.pool import StaticPool
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, future=True, **kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, future=True)
        self.mode = mode
        self.clock = clock or _now

    def session(self) -> Session:
        return self.Session()

    # ---- audit ------------------------------------------------------------
    def audit(self, event: str, details: dict | None = None, level: str = "INFO") -> None:
        with self.session() as s:
            s.add(AuditLog(mode=self.mode, ts=self.clock(), event=event, level=level, details=_jsonable(details)))
            s.commit()

    def recent_audit(self, limit: int = 100) -> list[AuditLog]:
        with self.session() as s:
            return list(s.scalars(select(AuditLog).where(AuditLog.mode == self.mode)
                                  .order_by(AuditLog.id.desc()).limit(limit)))

    # ---- state ------------------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            row = s.scalar(select(BotState).where(BotState.mode == self.mode, BotState.key == key))
            return default if row is None else row.value

    def set_state(self, key: str, value: Any) -> None:
        with self.session() as s:
            row = s.scalar(select(BotState).where(BotState.mode == self.mode, BotState.key == key))
            if row is None:
                s.add(BotState(mode=self.mode, key=key, value=_jsonable(value)))
            else:
                row.value = _jsonable(value)
                row.updated_at = _now()
            s.commit()

    # ---- trades -----------------------------------------------------------
    def add_trade(self, **fields: Any) -> Trade:
        with self.session() as s:
            fields.setdefault("created_at", self.clock())
            t = Trade(mode=self.mode, **fields)
            s.add(t)
            s.commit()
            return t

    def update_trade(self, trade_id: int, **fields: Any) -> Trade:
        with self.session() as s:
            t = s.get(Trade, trade_id)
            if t is None:
                raise KeyError(trade_id)
            for k, v in fields.items():
                setattr(t, k, v)
            s.commit()
            return t

    def get_trade(self, trade_id: int) -> Trade | None:
        with self.session() as s:
            return s.get(Trade, trade_id)

    def open_trades(self) -> list[Trade]:
        with self.session() as s:
            return list(s.scalars(select(Trade).where(Trade.mode == self.mode,
                                                      Trade.status.in_(OPEN_TRADE_STATUSES))))

    def closed_trades(self, since: datetime | None = None, limit: int | None = None) -> list[Trade]:
        with self.session() as s:
            q = select(Trade).where(Trade.mode == self.mode, Trade.status == "closed")
            if since is not None:
                q = q.where(Trade.closed_at >= since)
            q = q.order_by(Trade.closed_at.desc())
            if limit:
                q = q.limit(limit)
            return list(s.scalars(q))

    def trade_by_client_id(self, client_order_id: str) -> Trade | None:
        with self.session() as s:
            return s.scalar(select(Trade).where(Trade.client_order_id == client_order_id))

    def signal_traded_recently(self, signal_key: str, within: timedelta) -> bool:
        cutoff = self.clock() - within
        with self.session() as s:
            n = s.scalar(select(func.count(Trade.id)).where(
                Trade.mode == self.mode, Trade.signal_key == signal_key, Trade.created_at >= cutoff))
            return bool(n)

    def realized_pnl_since(self, since: datetime) -> float:
        with self.session() as s:
            v = s.scalar(select(func.sum(Trade.realized_pnl)).where(
                Trade.mode == self.mode, Trade.status == "closed", Trade.closed_at >= since))
            return float(v or 0.0)

    # ---- orders -----------------------------------------------------------
    def add_order(self, **fields: Any) -> OrderRecord:
        with self.session() as s:
            fields.setdefault("created_at", self.clock())
            o = OrderRecord(mode=self.mode, **fields)
            s.add(o)
            s.commit()
            return o

    def update_order(self, client_order_id: str, **fields: Any) -> OrderRecord | None:
        with self.session() as s:
            o = s.scalar(select(OrderRecord).where(OrderRecord.client_order_id == client_order_id))
            if o is None:
                return None
            for k, v in fields.items():
                setattr(o, k, v)
            o.updated_at = _now()
            s.commit()
            return o

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        with self.session() as s:
            return s.scalar(select(OrderRecord).where(OrderRecord.client_order_id == client_order_id))

    def orders_since(self, since: datetime) -> list[OrderRecord]:
        with self.session() as s:
            return list(s.scalars(select(OrderRecord).where(
                OrderRecord.mode == self.mode, OrderRecord.created_at >= since)))

    def count_entry_orders_since(self, since: datetime) -> int:
        with self.session() as s:
            return int(s.scalar(select(func.count(OrderRecord.id)).where(
                OrderRecord.mode == self.mode, OrderRecord.purpose == "entry",
                OrderRecord.created_at >= since)) or 0)

    def unresolved_orders(self) -> list[OrderRecord]:
        terminal = ("filled", "canceled", "expired", "rejected", "submit_failed", "replaced", "done_for_day")
        with self.session() as s:
            return list(s.scalars(select(OrderRecord).where(
                OrderRecord.mode == self.mode, OrderRecord.status.not_in(terminal))))

    # ---- signals / equity / reports ---------------------------------------
    def add_signal(self, **fields: Any) -> None:
        with self.session() as s:
            s.add(SignalRecord(mode=self.mode, ts=self.clock(), **{k: _jsonable(v) for k, v in fields.items()}))
            s.commit()

    def recent_signals(self, limit: int = 50) -> list[SignalRecord]:
        with self.session() as s:
            return list(s.scalars(select(SignalRecord).where(SignalRecord.mode == self.mode)
                                  .order_by(SignalRecord.id.desc()).limit(limit)))

    def add_equity_snapshot(self, **fields: Any) -> None:
        with self.session() as s:
            fields.setdefault("ts", self.clock())
            s.add(EquitySnapshot(mode=self.mode, **fields))
            s.commit()

    def equity_history(self, limit: int = 2000) -> list[EquitySnapshot]:
        with self.session() as s:
            rows = list(s.scalars(select(EquitySnapshot).where(EquitySnapshot.mode == self.mode)
                                  .order_by(EquitySnapshot.id.desc()).limit(limit)))
            return list(reversed(rows))

    def save_report(self, day: date, report: dict) -> None:
        with self.session() as s:
            row = s.scalar(select(DailyReport).where(DailyReport.mode == self.mode,
                                                     DailyReport.day == day.isoformat()))
            if row is None:
                s.add(DailyReport(mode=self.mode, day=day.isoformat(), report=_jsonable(report)))
            else:
                row.report = _jsonable(report)
            s.commit()

    def reports(self, limit: int = 30) -> list[DailyReport]:
        with self.session() as s:
            return list(s.scalars(select(DailyReport).where(DailyReport.mode == self.mode)
                                  .order_by(DailyReport.day.desc()).limit(limit)))


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    return json.loads(json.dumps(v, default=str))


def row_to_dict(row: Any, fields: Iterable[str] | None = None) -> dict:
    cols = fields or [c.name for c in row.__table__.columns]
    out = {}
    for c in cols:
        v = getattr(row, c)
        if isinstance(v, datetime):
            v = _aware(v).isoformat()
        out[c] = v
    return out
