#!/usr/bin/env python3
"""Unit tests for the parts of the STT daemon that don't need a model: remote
engine configuration, request shape, and error handling.

Runs under pytest, or standalone (`python daemon/test_relaystt.py`) — the
standalone runner skips anything needing a fixture and says how many.

Importing whisper_daemon pulls in nothing heavy: mlx-whisper is imported lazily
inside load_model(), so these stay fast and model-free.
"""
import io
import json
import os
import struct
import sys
import urllib.error

try:
    import pytest
except ImportError:  # the fallback runner at the bottom covers this
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relaystt_daemon
from relaystt_daemon import RemoteEngine


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch, body=b'{"text":"hello there","language":"english"}'):
    """Intercept urlopen and hand back the Request the engine built."""
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        seen["method"] = req.get_method()
        return _FakeResponse(body)

    monkeypatch.setattr(relaystt_daemon.urllib.request, "urlopen", fake_urlopen)
    return seen


def _wav(tmp_path, data=b"RIFF" + b"\0" * 100):
    p = tmp_path / "clip.wav"
    p.write_bytes(data)
    return str(p)


# ── configuration ────────────────────────────────────────────────

def test_disabled_without_a_url():
    assert RemoteEngine().enabled is False


def test_enabled_by_url_alone():
    """A URL is the whole switch — there is no separate enable flag that could
    drift out of sync with it."""
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    assert e.enabled is True
    assert e.url == "http://198.51.100.10:8080/v1/audio/transcriptions"


def test_trailing_slash_does_not_double_up():
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1/", model="up/asr")
    assert e.url == "http://198.51.100.10:8080/v1/audio/transcriptions"


def test_env_overrides_arguments(monkeypatch):
    monkeypatch.setenv("RELAYSTT_REMOTE_URL", "http://host:9999/v1")
    monkeypatch.setenv("RELAYSTT_REMOTE_MODEL", "env/model")
    e = RemoteEngine(base_url="http://ignored:1/v1", model="ignored")
    assert e.url.startswith("http://host:9999/v1")
    assert e.model == "env/model"


def test_enabled_without_model_is_fatal():
    """Failing at construction is the point: a remote daemon with no model id
    would otherwise start clean and fail every transcription."""
    try:
        RemoteEngine(base_url="http://198.51.100.10:8080/v1")
    except ValueError as e:
        assert "RELAYSTT_REMOTE_MODEL" in str(e)
    else:
        raise AssertionError("expected ValueError for remote without a model")


def test_token_read_from_named_env_var_and_absent_from_label(monkeypatch):
    monkeypatch.setenv("MY_STT_TOKEN", "s3cret")
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr",
                     api_key_env="MY_STT_TOKEN")
    assert e.api_key == "s3cret"
    assert "s3cret" not in e.label


# ── request shape ────────────────────────────────────────────────

def test_multipart_carries_model_language_and_audio(tmp_path, monkeypatch):
    audio = b"RIFF" + bytes(range(256))
    path = _wav(tmp_path, audio)
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(monkeypatch)

    result = e.transcribe(path, language="en")

    assert seen["method"] == "POST"
    assert seen["url"] == "http://198.51.100.10:8080/v1/audio/transcriptions"
    ctype = next(v for k, v in seen["headers"].items() if k.lower() == "content-type")
    assert ctype.startswith("multipart/form-data; boundary=")
    body = seen["body"]
    assert b'name="model"' in body and b"up/asr" in body
    assert b'name="language"' in body and b"en" in body
    # The audio must survive byte-for-byte — a transcoding bug here would show
    # up as mysteriously bad transcription, not as an error.
    assert audio in body
    assert b'filename="clip.wav"' in body
    assert result["text"] == "hello there"


def test_language_omitted_when_not_given(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(monkeypatch)
    e.transcribe(_wav(tmp_path))
    assert b'name="language"' not in seen["body"]


def test_bearer_sent_only_when_configured(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(monkeypatch)
    e.transcribe(_wav(tmp_path))
    assert not any(k.lower() == "authorization" for k in seen["headers"])

    monkeypatch.setenv("RELAYSTT_REMOTE_API_KEY", "tok")
    e2 = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    seen2 = _capture(monkeypatch)
    e2.transcribe(_wav(tmp_path))
    assert seen2["headers"]["Authorization"] == "Bearer tok"


# ── failure modes ────────────────────────────────────────────────

def test_http_error_names_endpoint_and_reason(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model"}}'))

    monkeypatch.setattr(relaystt_daemon.urllib.request, "urlopen", boom)
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "198.51.100.10:8080" in str(err)
        assert "HTTP 400" in str(err) and "unknown model" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_unreachable_error_is_actionable(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")

    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(relaystt_daemon.urllib.request, "urlopen", boom)
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "unreachable" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_non_json_body_is_reported(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    _capture(monkeypatch, body=b"<html>proxy error</html>")
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "not JSON" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_json_without_text_field_is_reported(tmp_path, monkeypatch):
    """A 200 with the wrong shape is worse than an error — it would surface as
    silent empty transcriptions."""
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    _capture(monkeypatch, body=b'{"unexpected":"shape"}')
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "no text field" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_error_detail_unwraps_common_shapes():
    assert RemoteEngine._error_detail(b'{"error":{"message":"nope"}}') == "nope"
    assert RemoteEngine._error_detail(b'{"detail":"fastapi style"}') == "fastapi style"
    assert RemoteEngine._error_detail(b"plain text") == "plain text"


# ── daemon wiring ────────────────────────────────────────────────

def test_daemon_defaults_to_local_engine():
    assert relaystt_daemon.RelaySTTDaemon().remote.enabled is False


def test_daemon_takes_the_engine_it_is_given():
    e = RemoteEngine(base_url="http://198.51.100.10:8080/v1", model="up/asr")
    assert relaystt_daemon.RelaySTTDaemon(remote=e).remote.enabled is True


# ── framing: probe vs truncation ──────────────────────────────────

class _FakeSock:
    """Serves a scripted byte stream, then behaves like a closed peer."""

    def __init__(self, data=b""):
        self._data = data

    def recv(self, n):
        if not self._data:
            return b""
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


def test_probe_disconnect_is_not_an_error():
    """A health check opens the port and hangs up without sending. That is
    normal, and must not surface as an error — noise here buries the truncated
    request below, which is a real fault."""
    d = relaystt_daemon.RelaySTTDaemon()
    try:
        d._recv_request(_FakeSock(b""))
    except relaystt_daemon.ClientDisconnected:
        pass
    else:
        raise AssertionError("expected ClientDisconnected for a bare probe")


def test_truncated_request_is_still_an_error():
    """Closing mid-message is data loss and keeps the louder exception."""
    d = relaystt_daemon.RelaySTTDaemon()
    # A length header promising 100 bytes, followed by only 3 and a close.
    payload = struct.pack("!I", 100) + b"abc"
    try:
        d._recv_request(_FakeSock(payload))
    except relaystt_daemon.ClientDisconnected:
        raise AssertionError("a truncated request must not be treated as a probe")
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError for a truncated request")


def test_header_only_close_is_truncation_not_probe():
    """Bytes arrived, so the peer did start asking something — the close that
    follows is truncation even though the payload read saw zero bytes."""
    d = relaystt_daemon.RelaySTTDaemon()
    try:
        d._recv_request(_FakeSock(struct.pack("!I", 50)))
    except relaystt_daemon.ClientDisconnected:
        raise AssertionError("a close after the header is truncation, not a probe")
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")


if __name__ == "__main__":
    # pytest is not a declared dependency, so this file stays runnable without
    # it. With pytest present, defer to it — the fixture-based tests only run
    # that way. Without it, run what can be driven by hand and say what was
    # skipped rather than reporting a clean sweep over a third of the suite.
    try:
        import pytest as _pytest
    except ImportError:
        _pytest = None

    if _pytest is not None:
        sys.exit(_pytest.main([os.path.abspath(__file__), "-q"]))

    import inspect
    import pathlib
    import tempfile
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = skipped = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        if "monkeypatch" in params:
            print(f"  SKIP {name} (needs pytest)")
            skipped += 1
            continue
        try:
            if params:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception:
            print(f"  FAIL {name}")
            traceback.print_exc()
    total = len(fns) - skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped (install pytest to run all)")
    sys.exit(0 if passed == total else 1)
