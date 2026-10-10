"""Logging with secret redaction.

Every handler gets a ``RedactingFilter`` that scrubs known secret values and anything shaped like
an API key / bearer token from log records before they are emitted.
"""
from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(PK|AK|CK)[A-Z0-9]{14,}\b"),                       # Alpaca key ids
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|authorization|token)(\s*[=:]\s*)([^\s,'\"]+)"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
]


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s and len(s) >= 4]

    def redact(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***REDACTED***")
        for p in _PATTERNS:
            if p.groups >= 3:
                text = p.sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", text)
            else:
                text = p.sub("***REDACTED***", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        record.msg = self.redact(msg)
        record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def setup_logging(log_dir: Path | None, secrets: list[str] | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    flt = RedactingFilter(secrets)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_dir / "autotrader.log", maxBytes=10_000_000, backupCount=10))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(flt)
        root.addHandler(h)
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
