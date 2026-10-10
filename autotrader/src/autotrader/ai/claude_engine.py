"""Claude-based decision engine.

Claude receives structured market data for deterministic strategy candidates and returns a
schema-validated ``AIReview``. Its role is advisory and *restrictive*: it may skip a candidate,
tighten levels, or reduce size, and must explain itself. Everything it returns is then validated
by the deterministic RiskEngine; Claude's output can never bypass or loosen risk controls.

Fail-closed: on any API error, refusal, truncation or schema failure, no AI decision is produced
and (by default, ``ai.on_ai_unavailable: skip``) no trade is taken.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from ..config import AICfg
from ..models import AIDecision, AIReview, Candidate

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the analysis component of an automated, risk-controlled trading system.
You review trade candidates produced by deterministic, backtested strategies and decide, for each one,
whether to `enter` or `skip`.

Rules:
- Use ONLY the data supplied in the request. Do not invent prices, news, earnings dates, or statistics.
  If the data is insufficient to judge a candidate, choose `skip` and say why.
- You cannot change a candidate's ticker, instrument, instrument_symbol, direction or strategy; copy them exactly.
- entry_price must stay within the candidate's quoted bid/ask. You may tighten (never widen) the stop,
  and you may lower the profit target, but stop < entry < target must hold (prices are option premiums
  for option and spread instruments).
- position_size must not exceed the candidate's `max_quantity`. Smaller is allowed.
- Consider trend strength, momentum, volatility, volume, bid-ask spread, and for options: implied
  volatility, delta, theta decay versus expected holding period, gamma, and days to expiry.
- Prefer skipping over marginal trades. Capital preservation matters more than trade frequency.
- estimated_risk_usd = loss if stopped out including slippage (for options assume the full premium
  can be lost). estimated_costs_usd = round-trip spread cost plus fees.
- A separate deterministic risk engine will re-validate and may reject or shrink anything you approve.
Return exactly one decision per candidate."""


@dataclass
class AIResult:
    review: AIReview | None
    error: str | None = None
    latency_s: float = 0.0
    model: str | None = None


def candidate_payload(cid: str, c: Candidate, max_qty: int) -> dict:
    legs = []
    for l in c.legs:
        k = l.contract
        q = k.quote if k else None
        legs.append({
            "symbol": l.symbol, "side": l.side, "ratio": l.ratio,
            "type": k.option_type if k else None, "strike": k.strike if k else None,
            "expiration": k.expiration.isoformat() if k else None,
            "bid": q.bid if q else None, "ask": q.ask if q else None,
            "implied_vol": k.implied_vol if k else None, "delta": k.delta if k else None,
            "gamma": k.gamma if k else None, "theta_per_day": k.theta if k else None,
            "open_interest": k.open_interest if k else None,
        })
    return {
        "candidate_id": cid, "strategy": c.strategy, "ticker": c.symbol, "instrument": c.instrument.value,
        "instrument_symbol": c.trade_symbol, "direction": c.direction.value, "regime": c.regime.value,
        "underlying_price": c.underlying_price, "proposed_entry": c.entry_price, "proposed_stop": c.stop_price,
        "proposed_target": c.target_price, "max_quantity": max_qty, "max_hold_minutes": c.max_hold_minutes,
        "intraday_only": c.intraday_only, "signal_rationale": c.rationale, "features": c.features, "legs": legs,
    }


class ClaudeDecisionEngine:
    def __init__(self, cfg: AICfg, api_key: str | None, client=None):
        self.cfg = cfg
        self._client = client
        if self._client is None and api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, timeout=cfg.timeout_seconds, max_retries=2)

    @property
    def available(self) -> bool:
        return self.cfg.enabled and self._client is not None

    def review(self, candidates: list[tuple[str, Candidate, int]], context: dict) -> AIResult:
        if not self.available:
            return AIResult(None, "AI disabled or no ANTHROPIC_API_KEY")
        import anthropic
        user = {
            "context": context,
            "candidates": [candidate_payload(cid, c, mq) for cid, c, mq in candidates],
        }
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Review these trade candidates.\n\n" + json.dumps(user, default=str)}],
            output_format=AIReview,
            output_config={"effort": self.cfg.effort},
        )
        t0 = time.monotonic()
        try:
            if self.cfg.use_server_fallbacks:
                resp = self._client.beta.messages.parse(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self._client.messages.parse(**kwargs)
        except anthropic.RateLimitError as e:
            return AIResult(None, f"rate limited: {e.__class__.__name__}", time.monotonic() - t0)
        except anthropic.APIStatusError as e:
            return AIResult(None, f"API status {e.status_code}", time.monotonic() - t0)
        except anthropic.APIConnectionError as e:
            return AIResult(None, f"connection error: {e.__class__.__name__}", time.monotonic() - t0)
        except Exception as e:  # schema/validation errors from the parse helper, etc.
            return AIResult(None, f"AI call failed: {e.__class__.__name__}: {e}", time.monotonic() - t0)
        latency = time.monotonic() - t0
        if getattr(resp, "stop_reason", None) == "refusal":
            return AIResult(None, "model refused", latency, getattr(resp, "model", None))
        if getattr(resp, "stop_reason", None) == "max_tokens":
            return AIResult(None, "response truncated (max_tokens)", latency, getattr(resp, "model", None))
        review = getattr(resp, "parsed_output", None)
        if review is None:
            return AIResult(None, "no parsed output", latency, getattr(resp, "model", None))
        if not isinstance(review, AIReview):
            try:
                review = AIReview.model_validate(review if isinstance(review, dict) else review.model_dump())
            except Exception as e:
                return AIResult(None, f"schema validation failed: {e}", latency)
        return AIResult(review, None, latency, getattr(resp, "model", None))


def match_decisions(review: AIReview, ids: list[str]) -> dict[str, AIDecision]:
    """Map decisions to candidates; ignore unknown ids and duplicates (first wins)."""
    out: dict[str, AIDecision] = {}
    for d in review.decisions:
        if d.candidate_id in ids and d.candidate_id not in out:
            out[d.candidate_id] = d
    return out
