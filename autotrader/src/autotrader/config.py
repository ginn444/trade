"""Configuration loading.

Configuration comes from YAML files (non-secret settings) and environment variables (secrets).
Secrets are never stored on the config object's serializable fields; see ``Credentials``.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class TradingMode(str, Enum):
    MOCK = "mock"
    PAPER = "paper"
    LIVE = "live"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradingCfg(_Strict):
    mode: TradingMode = TradingMode.PAPER
    loop_interval_seconds: int = 30
    open_delay_minutes: int = 5
    new_entry_cutoff: str = "15:30"
    flatten_intraday_at: str = "15:50"
    timezone: str = "America/New_York"

    @field_validator("new_entry_cutoff", "flatten_intraday_at")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        time.fromisoformat(v)
        return v

    @property
    def cutoff_time(self) -> time:
        return time.fromisoformat(self.new_entry_cutoff)

    @property
    def flatten_time(self) -> time:
        return time.fromisoformat(self.flatten_intraday_at)


class UniverseCfg(_Strict):
    stocks: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    min_avg_dollar_volume: float = 50_000_000
    min_price: float = 10.0
    options_enabled: bool = True
    options_underlyings: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])


class DataCfg(_Strict):
    stock_feed: str = "iex"
    options_feed: str = "indicative"
    max_quote_age_seconds: int = 60
    max_bar_age_minutes: int = 30
    daily_lookback_days: int = 300
    intraday_timeframe_minutes: int = 5


class RiskCfg(_Strict):
    max_risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 3
    max_position_notional_pct: float = 20.0
    min_reward_risk: float = 1.5
    allow_averaging_down: bool = False
    max_spread_pct_stock: float = 0.15
    max_spread_pct_option: float = 8.0
    min_option_open_interest: int = 500
    min_option_volume: int = 50
    option_dte_min: int = 7
    option_dte_max: int = 45
    scalp_dte_min: int = 1
    scalp_dte_max: int = 7
    option_delta_min: float = 0.30
    option_delta_max: float = 0.65
    options_risk_full_premium: bool = True
    stop_slippage_pct_stock: float = 0.25
    duplicate_signal_window_minutes: int = 60
    max_orders_per_day: int = 20
    halt_on_reconcile_mismatch: bool = True
    min_buying_power_buffer_pct: float = 10.0

    @field_validator("allow_averaging_down")
    @classmethod
    def _no_avg_down(cls, v: bool) -> bool:
        if v:
            raise ValueError("allow_averaging_down cannot be enabled: averaging down is prohibited")
        return v

    @field_validator("max_risk_per_trade_pct", "max_daily_loss_pct")
    @classmethod
    def _pct_sane(cls, v: float) -> float:
        if not 0 < v <= 5:
            raise ValueError("risk percentages must be in (0, 5]")
        return v


class LiveRiskCfg(_Strict):
    max_risk_per_trade_pct: float = 0.25
    max_daily_loss_pct: float = 1.0
    max_open_positions: int = 2
    max_position_notional_pct: float = 10.0
    max_orders_per_day: int = 10


class BrokerHealthCfg(_Strict):
    max_consecutive_errors: int = 3
    error_window_seconds: int = 300
    max_errors_in_window: int = 5
    max_latency_seconds: float = 5.0
    recovery_successes_required: int = 3


class ExecutionCfg(_Strict):
    entry_order_type: str = "limit"
    limit_offset_pct: float = 0.05
    entry_timeout_seconds: int = 120
    option_exit_aggressive_after_seconds: int = 30
    use_bracket_for_stocks: bool = True


class AICfg(_Strict):
    enabled: bool = True
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    timeout_seconds: float = 90
    use_server_fallbacks: bool = True
    max_candidates_per_cycle: int = 5
    on_ai_unavailable: str = "skip"

    @field_validator("on_ai_unavailable")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in {"skip", "deterministic"}:
            raise ValueError("on_ai_unavailable must be 'skip' or 'deterministic'")
        return v


class StrategiesCfg(_Strict):
    enabled: list[str] = Field(default_factory=list)
    require_validation: bool = True
    validation_file: str = "state/strategy_approvals.json"


class AcceptanceCfg(_Strict):
    min_trades: int = 30
    min_profit_factor: float = 1.2
    max_drawdown_pct: float = 20.0
    min_expectancy_r: float = 0.05
    min_sharpe: float = 0.5
    min_positive_wf_windows_frac: float = 0.5
    require_beat_benchmark_sharpe: bool = False
    max_oos_degradation: float = 0.6


class BacktestCfg(_Strict):
    initial_equity: float = 100_000
    stock_commission_per_share: float = 0.0
    stock_slippage_bps: float = 5
    stock_spread_bps: float = 3
    option_commission_per_contract: float = 0.0
    option_regulatory_fee_per_contract: float = 0.05
    option_half_spread_pct: float = 4.0
    option_slippage_pct: float = 1.0
    option_iv_premium: float = 1.10
    walk_forward_train_bars: int = 252
    walk_forward_test_bars: int = 63
    walk_forward_train_bars_intraday: int = 1560   # 20 sessions of 5-minute bars
    walk_forward_test_bars_intraday: int = 390     # 5 sessions
    intraday_history_days: int = 120
    acceptance: AcceptanceCfg = Field(default_factory=AcceptanceCfg)


class StorageCfg(_Strict):
    database_url: str = "sqlite:///state/autotrader.db"
    state_dir: str = "state"
    reports_dir: str = "reports"
    log_dir: str = "state/logs"


class DashboardCfg(_Strict):
    host: str = "127.0.0.1"
    port: int = 8080


class LiveCfg(_Strict):
    enabled: bool = False
    authorization_file: str = "state/live_authorization.json"
    authorization_ttl_hours: int = 24
    risk: LiveRiskCfg = Field(default_factory=LiveRiskCfg)


class AppConfig(_Strict):
    trading: TradingCfg = Field(default_factory=TradingCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    broker_health: BrokerHealthCfg = Field(default_factory=BrokerHealthCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    ai: AICfg = Field(default_factory=AICfg)
    strategies: StrategiesCfg = Field(default_factory=StrategiesCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    dashboard: DashboardCfg = Field(default_factory=DashboardCfg)
    live: LiveCfg = Field(default_factory=LiveCfg)

    # ---- derived helpers -------------------------------------------------
    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def state_dir(self) -> Path:
        return self.resolve_path(self.storage.state_dir)

    @property
    def database_url(self) -> str:
        url = self.storage.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith(prefix + "/") and ":memory:" not in url:
            return prefix + str(self.resolve_path(url[len(prefix):]))
        return url

    def effective_risk(self) -> RiskCfg:
        """Risk limits in force for the current mode. In live mode the tighter of the two applies."""
        if self.trading.mode != TradingMode.LIVE:
            return self.risk
        lr = self.live.risk
        merged = self.risk.model_copy(
            update={
                "max_risk_per_trade_pct": min(self.risk.max_risk_per_trade_pct, lr.max_risk_per_trade_pct),
                "max_daily_loss_pct": min(self.risk.max_daily_loss_pct, lr.max_daily_loss_pct),
                "max_open_positions": min(self.risk.max_open_positions, lr.max_open_positions),
                "max_position_notional_pct": min(self.risk.max_position_notional_pct, lr.max_position_notional_pct),
                "max_orders_per_day": min(self.risk.max_orders_per_day, lr.max_orders_per_day),
            }
        )
        return merged

    def fingerprint(self) -> str:
        """Stable SHA-256 of the full config (used to bind live authorizations to a config)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load default.yaml, then an optional overlay file, then explicit overrides.

    ``AUTOTRADER_CONFIG`` selects the overlay when ``path`` is not given. ``AUTOTRADER_MODE`` may
    switch between mock and paper only — it can never select live mode (live must be set in a file
    that is then explicitly authorized).
    """
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    overlay_path = path or os.environ.get("AUTOTRADER_CONFIG")
    if overlay_path:
        p = Path(overlay_path)
        if not p.is_absolute():
            p = (Path.cwd() / p) if (Path.cwd() / p).exists() else PROJECT_ROOT / p
        if p.resolve() != DEFAULT_CONFIG_PATH.resolve():
            data = _deep_merge(data, yaml.safe_load(p.read_text()) or {})
    env_mode = os.environ.get("AUTOTRADER_MODE")
    if env_mode:
        if env_mode.lower() == TradingMode.LIVE.value:
            raise ValueError("AUTOTRADER_MODE cannot select live mode; use an authorized live config file")
        data = _deep_merge(data, {"trading": {"mode": env_mode.lower()}})
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)


class Credentials:
    """Holds secrets read from the environment. ``repr`` never reveals values."""

    def __init__(self, env: dict[str, str] | None = None):
        self._env = dict(os.environ if env is None else env)

    def _get(self, name: str) -> str | None:
        v = self._env.get(name)
        return v.strip() if v and v.strip() else None

    @property
    def anthropic_api_key(self) -> str | None:
        return self._get("ANTHROPIC_API_KEY")

    def alpaca(self, mode: TradingMode) -> tuple[str, str]:
        """Return (key, secret) strictly for the requested mode. Never falls back across modes."""
        if mode == TradingMode.PAPER:
            k, s = self._get("ALPACA_PAPER_API_KEY"), self._get("ALPACA_PAPER_SECRET_KEY")
            label = "ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY"
        elif mode == TradingMode.LIVE:
            k, s = self._get("ALPACA_LIVE_API_KEY"), self._get("ALPACA_LIVE_SECRET_KEY")
            label = "ALPACA_LIVE_API_KEY / ALPACA_LIVE_SECRET_KEY"
        else:
            raise ValueError("mock mode uses no brokerage credentials")
        if not k or not s:
            raise MissingCredentialsError(f"{label} must be set for {mode.value} mode")
        if mode == TradingMode.PAPER and k == self._get("ALPACA_LIVE_API_KEY"):
            raise MissingCredentialsError("paper and live API keys must differ")
        if mode == TradingMode.LIVE and k == self._get("ALPACA_PAPER_API_KEY"):
            raise MissingCredentialsError("paper and live API keys must differ")
        return k, s

    @property
    def dashboard_token(self) -> str | None:
        return self._get("DASHBOARD_TOKEN")

    def secret_values(self) -> list[str]:
        names = [
            "ANTHROPIC_API_KEY", "ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET_KEY",
            "ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY", "DASHBOARD_TOKEN",
        ]
        return [v for n in names if (v := self._get(n))]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Credentials(<redacted>)"


class MissingCredentialsError(RuntimeError):
    pass
