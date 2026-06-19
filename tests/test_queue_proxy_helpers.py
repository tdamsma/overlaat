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
