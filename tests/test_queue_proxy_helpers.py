"""Pure-function tests for the queue proxy."""

import asyncio
import hashlib

from starlette.requests import Request

from overlaat import queue_proxy as qp


def test_load_caps(tmp_path):
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        "model_list:\n"
        "  - model_name: a\n"
        "    litellm_params:\n"
        "      max_parallel_requests: 4\n"
        "  - model_name: b\n"  # no cap -> excluded
        "    litellm_params: {}\n"
    )
    assert qp.load_caps(cfg) == {"a": 4}


def test_load_caps_missing(tmp_path):
    assert qp.load_caps(tmp_path / "nope.yaml") == {}


def test_extract_tokens():
    tail = b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\ndata: [DONE]\n'
    assert qp._extract_tokens(tail) == (12, 34)
    assert qp._extract_tokens(b"no usage here") == (None, None)


def test_extract_tokens_takes_last():
    tail = b'"completion_tokens":1 ... "completion_tokens":99'
    assert qp._extract_tokens(tail)[1] == 99


def test_get_semaphore(monkeypatch):
    monkeypatch.setattr(qp, "CAPS", {"m": 2})
    monkeypatch.setattr(qp, "SEMAPHORES", {})
    sem = qp.get_semaphore("m")
    assert isinstance(sem, asyncio.Semaphore)
    assert qp.get_semaphore("m") is sem  # cached
    assert qp.get_semaphore("unknown") is None


def test_key_fp():
    req = Request({"type": "http", "headers": [(b"authorization", b"Bearer sk-test")]})
    assert qp._key_fp(req) == hashlib.sha256(b"sk-test").hexdigest()[:8]
    assert qp._key_fp(Request({"type": "http", "headers": []})) == "none"


# -- prompt-size-weighted admission cost (#18) -------------------------------


def test_estimate_prompt_tokens_chat_string_content():
    # chars / CHARS_PER_TOKEN(4), summed across messages.
    payload = {"messages": [{"content": "a" * 40}, {"content": "b" * 8}]}
    assert qp.estimate_prompt_tokens(payload) == (40 + 8) // 4


def test_estimate_prompt_tokens_multimodal_parts():
    # Only the text parts count; image parts contribute nothing.
    payload = {
        "messages": [
            {
                "content": [
                    {"type": "text", "text": "x" * 16},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            }
        ]
    }
    assert qp.estimate_prompt_tokens(payload) == 16 // 4


def test_estimate_prompt_tokens_completions_prompt():
    assert qp.estimate_prompt_tokens({"prompt": "z" * 20}) == 20 // 4
    assert qp.estimate_prompt_tokens({"prompt": ["aa", "bbbb"]}) == (2 + 4) // 4


def test_estimate_prompt_tokens_no_measurable_prompt():
    # /embeddings, /rerank etc. carry no messages/prompt -> 0 -> weight 1x.
    assert qp.estimate_prompt_tokens({"input": "irrelevant"}) == 0
    assert qp.estimate_prompt_tokens({}) == 0


def test_prompt_weight_default_tiers():
    t = qp._DEFAULT_WEIGHT_TIERS
    assert qp.prompt_weight(0, t) == 1.0
    assert qp.prompt_weight(2000, t) == 1.0  # inclusive upper bound
    assert qp.prompt_weight(2001, t) == 2.0
    assert qp.prompt_weight(8000, t) == 2.0
    assert qp.prompt_weight(8001, t) == 4.0
    assert qp.prompt_weight(33000, t) == 4.0


def test_parse_weight_tiers_override_and_fallback():
    parsed = qp._parse_weight_tiers("1000:1,5000:3,inf:6")
    assert parsed == ((1000.0, 1.0), (5000.0, 3.0), (float("inf"), 6.0))
    # Unsorted input is normalized ascending.
    assert qp._parse_weight_tiers("inf:6,1000:1")[0] == (1000.0, 1.0)
    # Garbage / a multiplier < 1 falls back to the default (never raises).
    assert qp._parse_weight_tiers("not-a-tier") == qp._DEFAULT_WEIGHT_TIERS
    assert qp._parse_weight_tiers("1000:0.5,inf:2") == qp._DEFAULT_WEIGHT_TIERS


def test_load_pool_heavy_max(tmp_path):
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        "overlaat:\n"
        "  pools:\n"
        "    default: {heavy_max: leave_room}\n"
        "    batch: {heavy_max: full_pool}\n"
        "    typo: {heavy_max: nonsense}\n"  # invalid -> ignored
        "    nobudget: {budget: 2.0}\n"  # no heavy_max -> absent (defaults later)
    )
    hm = qp.load_pool_heavy_max(cfg)
    assert hm == {"default": "leave_room", "batch": "full_pool"}
    assert qp.load_pool_heavy_max(tmp_path / "nope.yaml") == {}
