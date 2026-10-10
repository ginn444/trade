"""Core domain types shared by every layer."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Direction(str, Enum):
    LONG = "long"     # profits when underlying rises (long stock, long call, bull call spread)
    SHORT = "short"   # profits when underlying falls (long put, bear put spread). Never short options.


class InstrumentType(str, Enum):
    STOCK = "stock"
    OPTION = "option"            # single long call or long put
    DEBIT_SPREAD = "debit_spread"  # long + short same-type, same-expiry; defined risk


class Regime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    bid_size: float = 0
    ask_size: float = 0
    source: str = "unknown"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return float("inf") if m <= 0 else (self.ask - self.bid) / m * 100

    def is_valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or utcnow()
        return (now - self.timestamp).total_seconds()


@dataclass(frozen=True)
class OptionContract:
    symbol: str                 # OCC symbol
    underlying: str
    option_type: Literal["call", "put"]
    strike: float
    expiration: date
    quote: Quote | None = None
    implied_vol: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    open_interest: int | None = None
    volume: int | None = None
    multiplier: int = 100

    def dte(self, today: date) -> int:
        return (self.expiration - today).days


@dataclass
class OptionLeg:
    symbol: str
    side: Literal["buy", "sell"]
    ratio: int = 1
    contract: OptionContract | None = None


@dataclass
class Candidate:
    """A deterministic strategy signal, before AI review and risk validation."""
    strategy: str
    symbol: str                       # underlying ticker
    direction: Direction
    instrument: InstrumentType
    entry_price: float                # per-share for stock; per-contract premium (x1) for options/spreads
    stop_price: float                 # same units as entry_price (for options: premium level)
    target_price: float
    underlying_price: float
    regime: Regime
    features: dict = field(default_factory=dict)
    legs: list[OptionLeg] = field(default_factory=list)
    max_hold_minutes: int | None = None
    intraday_only: bool = False
    created_at: datetime = field(default_factory=utcnow)
    rationale: str = ""

    @property
    def trade_symbol(self) -> str:
        if self.instrument == InstrumentType.STOCK:
            return self.symbol
        if self.instrument == InstrumentType.OPTION:
            return self.legs[0].symbol
        return "|".join(f"{l.side}:{l.symbol}" for l in self.legs)

    def signal_key(self) -> str:
        """Identity used for duplicate-order protection (strategy+instrument+direction+day)."""
        day = self.created_at.astimezone(timezone.utc).date().isoformat()
        raw = f"{self.strategy}|{self.trade_symbol}|{self.direction.value}|{day}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


class AIDecision(BaseModel):
    """Structured output Claude must produce for each candidate (validated by pydantic + risk engine)."""
    candidate_id: str = Field(description="The candidate_id this decision refers to, copied exactly")
    ticker: str = Field(description="Underlying ticker symbol")
    instrument: Literal["stock", "option", "debit_spread"]
    instrument_symbol: str = Field(description="Exact tradable symbol: ticker for stock, OCC symbol for option, legs joined for spreads")
    action: Literal["enter", "skip"]
    direction: Literal["long", "short"]
    strategy: str
    entry_price: float = Field(description="Proposed limit entry price (per share or per-contract premium)")
    stop_loss: float
    profit_target: float
    position_size: int = Field(description="Proposed quantity (shares or contracts). The risk engine may reduce it.")
    confidence: float = Field(ge=0, le=1)
    estimated_risk_usd: float = Field(description="Estimated dollar loss if stopped out, including slippage")
    estimated_costs_usd: float = Field(description="Estimated commissions, fees and spread cost, round trip")
    explanation: str = Field(description="Concise reasoning grounded ONLY in the supplied data")
    key_risks: list[str] = Field(default_factory=list)


@dataclass
class TradeProposal:
    """A fully specified order intent that has passed (or is about to face) the risk engine."""
    candidate: Candidate
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    risk_usd: float
    est_costs_usd: float
    ai_decision: AIDecision | None = None
    decided_by: str = "deterministic"  # "claude" | "deterministic"

    @property
    def client_order_id(self) -> str:
        return f"at-{self.candidate.signal_key()}"


@dataclass
class RiskVerdict:
    approved: bool
    reasons: list[str]
    adjusted_quantity: int = 0
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class AccountState:
    equity: float
    buying_power: float
    cash: float
    last_equity: float          # prior close equity (start-of-day reference)
    options_buying_power: float | None = None
    trading_blocked: bool = False
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    asset_class: str  # "us_equity" | "us_option"


@dataclass
class BrokerOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None
    filled_avg_price: float | None = None
    created_at: datetime | None = None
    order_class: str = "simple"
    legs: list["BrokerOrder"] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status in {"new", "accepted", "pending_new", "partially_filled", "held",
                               "accepted_for_bidding", "pending_replace", "pending_cancel", "calculated"}


class AIReview(BaseModel):
    """Claude's full response for one scan cycle: exactly one decision per candidate."""
    market_summary: str = Field(description="One or two sentences on overall conditions, from the supplied data only")
    decisions: list[AIDecision]
