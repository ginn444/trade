"""Persistent emergency kill switch and the live-trading authorization gate."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import AppConfig, Credentials, TradingMode


class KillSwitch:
    """File-backed kill switch. Survives restarts; only an explicit `release` clears it.

    The file is the source of truth so that the dashboard, CLI and bot (separate processes) agree.
    If the file cannot be read, the switch is treated as ENGAGED (fail-safe).
    """

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "KILL_SWITCH"

    def engage(self, reason: str, actor: str = "unknown", flatten: bool = False) -> None:
        """Stop all new orders and cancel pending entries. ``flatten`` also closes open positions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"engaged_at": datetime.now(timezone.utc).isoformat(), "reason": reason, "actor": actor,
                   "flatten": bool(flatten)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)

    def release(self, actor: str = "unknown") -> None:
        if self.path.exists():
            self.path.unlink()

    def is_engaged(self) -> bool:
        try:
            return self.path.exists()
        except OSError:
            return True

    def info(self) -> dict | None:
        if not self.is_engaged():
            return None
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"reason": "unreadable kill-switch file (treated as engaged)"}


class LiveTradingNotAuthorized(RuntimeError):
    pass


CONFIRM_PHRASE = "I AUTHORIZE REAL-MONEY TRADING"


@dataclass
class LiveAuthorization:
    config_fingerprint: str
    authorized_at: datetime
    expires_at: datetime
    operator: str

    def to_json(self) -> str:
        return json.dumps({
            "config_fingerprint": self.config_fingerprint,
            "authorized_at": self.authorized_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "operator": self.operator,
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LiveAuthorization":
        d = json.loads(text)
        return cls(d["config_fingerprint"], datetime.fromisoformat(d["authorized_at"]),
                   datetime.fromisoformat(d["expires_at"]), d.get("operator", "unknown"))


def write_live_authorization(cfg: AppConfig, operator: str, confirmation: str) -> Path:
    """Create the authorization file. Requires the exact confirmation phrase and a live-enabled config."""
    if confirmation.strip() != CONFIRM_PHRASE:
        raise LiveTradingNotAuthorized("confirmation phrase mismatch; live trading NOT authorized")
    if cfg.trading.mode != TradingMode.LIVE or not cfg.live.enabled:
        raise LiveTradingNotAuthorized("config must set trading.mode=live and live.enabled=true")
    now = datetime.now(timezone.utc)
    auth = LiveAuthorization(cfg.fingerprint(), now, now + timedelta(hours=cfg.live.authorization_ttl_hours), operator)
    path = cfg.resolve_path(cfg.live.authorization_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(auth.to_json())
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover
        pass
    return path


def assert_live_authorized(cfg: AppConfig, creds: Credentials, now: datetime | None = None) -> None:
    """Every condition must hold, otherwise raise. Called before any live broker is constructed."""
    now = now or datetime.now(timezone.utc)
    if cfg.trading.mode != TradingMode.LIVE:
        raise LiveTradingNotAuthorized("trading.mode is not live")
    if not cfg.live.enabled:
        raise LiveTradingNotAuthorized("live.enabled is false")
    creds.alpaca(TradingMode.LIVE)  # raises if missing or identical to paper keys
    path = cfg.resolve_path(cfg.live.authorization_file)
    if not path.exists():
        raise LiveTradingNotAuthorized(f"no live authorization at {path}; run `autotrader authorize-live`")
    try:
        auth = LiveAuthorization.from_json(path.read_text())
    except (OSError, ValueError, KeyError) as e:
        raise LiveTradingNotAuthorized(f"unreadable live authorization: {e}") from e
    if auth.config_fingerprint != cfg.fingerprint():
        raise LiveTradingNotAuthorized("config changed since authorization; re-authorize")
    if now >= auth.expires_at:
        raise LiveTradingNotAuthorized("live authorization expired; re-authorize")
