"""Brokerage connectivity health monitor.

New orders are blocked while unhealthy. Exits are still attempted (capital protection first).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .config import BrokerHealthCfg


@dataclass
class HealthMonitor:
    cfg: BrokerHealthCfg
    clock: callable = time.monotonic
    consecutive_errors: int = 0
    successes_since_failure: int = 0
    healthy: bool = True
    last_error: str | None = None
    _errors: deque = field(default_factory=deque)

    def record_success(self, latency_s: float = 0.0) -> None:
        if latency_s > self.cfg.max_latency_seconds:
            self.record_error(f"slow response {latency_s:.2f}s")
            return
        self.consecutive_errors = 0
        self.successes_since_failure += 1
        if not self.healthy and self.successes_since_failure >= self.cfg.recovery_successes_required:
            self._prune()
            if len(self._errors) < self.cfg.max_errors_in_window:
                self.healthy = True

    def record_error(self, message: str) -> None:
        now = self.clock()
        self.last_error = message
        self.consecutive_errors += 1
        self.successes_since_failure = 0
        self._errors.append(now)
        self._prune()
        if (self.consecutive_errors >= self.cfg.max_consecutive_errors
                or len(self._errors) >= self.cfg.max_errors_in_window):
            self.healthy = False

    def _prune(self) -> None:
        cutoff = self.clock() - self.cfg.error_window_seconds
        while self._errors and self._errors[0] < cutoff:
            self._errors.popleft()

    def status(self) -> dict:
        self._prune()
        return {"healthy": self.healthy, "consecutive_errors": self.consecutive_errors,
                "errors_in_window": len(self._errors), "last_error": self.last_error}
