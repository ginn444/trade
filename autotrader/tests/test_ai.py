from types import SimpleNamespace

import anthropic
import httpx

from autotrader.ai.claude_engine import ClaudeDecisionEngine, match_decisions
from autotrader.config import AICfg
from autotrader.models import AIReview

from conftest import FakeClaude, stock_candidate


def _cands():
    return [("c0", stock_candidate(), 100), ("c1", stock_candidate("BBB"), 50)]


def test_structured_review_parsed():
    fake = FakeClaude()
    eng = ClaudeDecisionEngine(AICfg(), None, client=fake)
    res = eng.review(_cands(), {"equity": 1})
    assert res.error is None and len(res.review.decisions) == 2
    kw = fake.calls[0]
    assert kw["output_format"] is AIReview
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert "Use ONLY the data supplied" in kw["system"]


def test_without_fallbacks_uses_ga_parse():
    fake = FakeClaude()
    ClaudeDecisionEngine(AICfg(use_server_fallbacks=False), None, client=fake).review(_cands(), {})
    assert "fallbacks" not in fake.calls[0]


def test_fail_closed_on_errors_and_refusals():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for err in [anthropic.APIConnectionError(request=req), ValueError("bad schema")]:
        res = ClaudeDecisionEngine(AICfg(), None, client=FakeClaude(fail=err)).review(_cands(), {})
        assert res.review is None and res.error
    for sr in ["refusal", "max_tokens"]:
        res = ClaudeDecisionEngine(AICfg(), None, client=FakeClaude(stop_reason=sr)).review(_cands(), {})
        assert res.review is None


def test_unavailable_without_key_or_disabled():
    assert not ClaudeDecisionEngine(AICfg(), None).available
    assert not ClaudeDecisionEngine(AICfg(enabled=False), None, client=FakeClaude()).available


def test_match_decisions_ignores_unknown_and_duplicates():
    res = ClaudeDecisionEngine(AICfg(), None, client=FakeClaude()).review(_cands(), {})
    r = res.review
    r.decisions.append(r.decisions[0].model_copy(update={"candidate_id": "zzz"}))
    r.decisions.append(r.decisions[0].model_copy(update={"action": "skip"}))
    m = match_decisions(r, ["c0", "c1"])
    assert set(m) == {"c0", "c1"} and m["c0"].action == "enter"
