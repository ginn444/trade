"""Build broker / data / AI components for the configured mode.

There is no code path from paper to live: each mode reads only its own credentials, and a live
broker is constructed only after ``assert_live_authorized`` succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ai.claude_engine import ClaudeDecisionEngine
from ..config import AppConfig, Credentials, TradingMode
from ..db import Database
from ..health import HealthMonitor
from ..safety import KillSwitch, assert_live_authorized


@dataclass
class Components:
    broker: object
    data: object
    db: Database
    ai: ClaudeDecisionEngine
    kill: KillSwitch
    health: HealthMonitor
    live_authorized: bool
    broker_state_path: object = None


def build_components(cfg: AppConfig, creds: Credentials, *, ai_client=None) -> Components:
    mode = cfg.trading.mode
    db = Database(cfg.database_url, mode.value)
    kill = KillSwitch(cfg.state_dir)
    health = HealthMonitor(cfg.broker_health)
    ai = ClaudeDecisionEngine(cfg.ai, creds.anthropic_api_key, client=ai_client)
    live_ok = False
    if mode == TradingMode.MOCK:
        from ..broker.simulated import SimulatedBroker
        from ..data.synthetic import SyntheticMarketData
        data = SyntheticMarketData()
        broker = SimulatedBroker(data, cfg.backtest.initial_equity)
        state_path = cfg.state_dir / "sim_broker.json"
        broker.load(state_path)
    elif mode == TradingMode.PAPER:
        from ..broker.alpaca_broker import AlpacaBroker
        from ..data.alpaca_data import AlpacaMarketData
        key, secret = creds.alpaca(TradingMode.PAPER)
        broker = AlpacaBroker(key, secret, paper=True)
        data = AlpacaMarketData(key, secret, cfg.data.stock_feed, cfg.data.options_feed, broker.client)
    elif mode == TradingMode.LIVE:
        assert_live_authorized(cfg, creds)
        from ..broker.alpaca_broker import AlpacaBroker
        from ..data.alpaca_data import AlpacaMarketData
        key, secret = creds.alpaca(TradingMode.LIVE)
        broker = AlpacaBroker(key, secret, paper=False, _live_authorized=True)
        data = AlpacaMarketData(key, secret, cfg.data.stock_feed, cfg.data.options_feed, broker.client)
        live_ok = True
        db.audit("live_mode_started", {"config_fingerprint": cfg.fingerprint()[:16]}, "WARNING")
    else:  # pragma: no cover
        raise ValueError(mode)
    return Components(broker, data, db, ai, kill, health, live_ok,
                      state_path if mode == TradingMode.MOCK else None)
