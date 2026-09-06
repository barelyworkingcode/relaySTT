#!/usr/bin/env python3
"""Unit tests for the relaySTT daemon: remote engine configuration, pinned
transport, request shape, and error handling.

Runs under pytest, or standalone (`python daemon/test_relaystt.py`) — the
standalone runner skips anything needing a fixture and says how many.
"""
import hashlib
import http.server
import io
import json
import os
import shutil
import ssl
import struct
import subprocess
import sys
import threading
import urllib.error

import numpy as np
import soundfile as sf

try:
    import pytest
except ImportError:  # the fallback runner at the bottom covers this
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pinned_transport
import relaystt_daemon
from relaystt_daemon import RemoteEngine


class _StubEngine:
    """A minimal engine for tests that only exercise protocol framing and
    request validation, never real transcription."""

    label = "<stub>"
    model = "stub"

    def transcribe(self, wav_path, language=None):
        return {"text": "", "language": language or "unknown"}


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(engine, monkeypatch, body=b'{"text":"hello there","language":"english"}'):
    """Intercept the engine's opener and hand back the Request it built."""
    seen = {}

    def fake_open(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        seen["method"] = req.get_method()
        return _FakeResponse(body)

    monkeypatch.setattr(engine._opener, "open", fake_open)
    return seen


def _wav(tmp_path, data=b"RIFF" + b"\0" * 100):
    p = tmp_path / "clip.wav"
    p.write_bytes(data)
    return str(p)


# ── configuration ────────────────────────────────────────────────

def test_missing_url_is_fatal():
    try:
        RemoteEngine(model="up/asr")
    except ValueError as e:
        assert "RELAYSTT_REMOTE_URL" in str(e)
    else:
        raise AssertionError("expected ValueError for a remote engine without a URL")


def test_url_builds_transcriptions_endpoint():
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    assert e.url == "https://198.51.100.10:8080/v1/audio/transcriptions"


def test_trailing_slash_does_not_double_up():
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1/", model="up/asr")
    assert e.url == "https://198.51.100.10:8080/v1/audio/transcriptions"


def test_env_overrides_arguments(monkeypatch):
    monkeypatch.setenv("RELAYSTT_REMOTE_URL", "https://host:9999/v1")
    monkeypatch.setenv("RELAYSTT_REMOTE_MODEL", "env/model")
    e = RemoteEngine(base_url="https://ignored:1/v1", model="ignored")
    assert e.url.startswith("https://host:9999/v1")
    assert e.model == "env/model"


def test_enabled_without_model_is_fatal():
    """Failing at construction is the point: a remote daemon with no model id
    would otherwise start clean and fail every transcription."""
    try:
        RemoteEngine(base_url="https://198.51.100.10:8080/v1")
    except ValueError as e:
        assert "RELAYSTT_REMOTE_MODEL" in str(e)
    else:
        raise AssertionError("expected ValueError for remote without a model")


def test_token_read_from_named_env_var_and_absent_from_label(monkeypatch):
    monkeypatch.setenv("MY_STT_TOKEN", "s3cret")
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr",
                     api_key_env="MY_STT_TOKEN")
    assert e.api_key == "s3cret"
    assert "s3cret" not in e.label


# ── request shape ────────────────────────────────────────────────

def test_multipart_carries_model_language_and_audio(tmp_path, monkeypatch):
    audio = b"RIFF" + bytes(range(256))
    path = _wav(tmp_path, audio)
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(e, monkeypatch)

    result = e.transcribe(path, language="en")

    assert seen["method"] == "POST"
    assert seen["url"] == "https://198.51.100.10:8080/v1/audio/transcriptions"
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
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(e, monkeypatch)
    e.transcribe(_wav(tmp_path))
    assert b'name="language"' not in seen["body"]


def test_bearer_sent_only_when_configured(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    seen = _capture(e, monkeypatch)
    e.transcribe(_wav(tmp_path))
    assert not any(k.lower() == "authorization" for k in seen["headers"])

    monkeypatch.setenv("RELAYSTT_REMOTE_API_KEY", "tok")
    e2 = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    seen2 = _capture(e2, monkeypatch)
    e2.transcribe(_wav(tmp_path))
    assert seen2["headers"]["Authorization"] == "Bearer tok"


# ── failure modes ────────────────────────────────────────────────

def test_http_error_names_endpoint_and_reason(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model"}}'))

    monkeypatch.setattr(e._opener, "open", boom)
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "198.51.100.10:8080" in str(err)
        assert "HTTP 400" in str(err) and "unknown model" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_unreachable_error_is_actionable(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")

    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(e._opener, "open", boom)
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "unreachable" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_non_json_body_is_reported(tmp_path, monkeypatch):
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    _capture(e, monkeypatch, body=b"<html>proxy error</html>")
    try:
        e.transcribe(_wav(tmp_path))
    except RuntimeError as err:
        assert "not JSON" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_json_without_text_field_is_reported(tmp_path, monkeypatch):
    """A 200 with the wrong shape is worse than an error — it would surface as
    silent empty transcriptions."""
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    _capture(e, monkeypatch, body=b'{"unexpected":"shape"}')
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

def test_daemon_takes_the_engine_it_is_given():
    e = RemoteEngine(base_url="https://198.51.100.10:8080/v1", model="up/asr")
    assert relaystt_daemon.RelaySTTDaemon(engine=e).engine is e


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
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    try:
        d._recv_request(_FakeSock(b""))
    except relaystt_daemon.ClientDisconnected:
        pass
    else:
        raise AssertionError("expected ClientDisconnected for a bare probe")


def test_truncated_request_is_still_an_error():
    """Closing mid-message is data loss and keeps the louder exception."""
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
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
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    try:
        d._recv_request(_FakeSock(struct.pack("!I", 50)))
    except relaystt_daemon.ClientDisconnected:
        raise AssertionError("a close after the header is truncation, not a probe")
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")


# ── framing: limits and legacy path ─────────────────────────────────

def test_recv_request_rejects_oversized_frame():
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    header = struct.pack("!I", relaystt_daemon.MAX_FRAME_BYTES + 1)
    try:
        d._recv_request(_FakeSock(header))
    except ValueError as e:
        assert "too large" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for an oversized frame")


def test_recv_request_rejects_raw_json_framing():
    """Eve only ever sends a length prefix; a leading '{' means a client is
    speaking the retired raw-JSON dialect, not that this one happens to be
    small enough to parse directly."""
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    try:
        d._recv_request(_FakeSock(b'{"ac'))
    except ValueError as e:
        assert "length prefix" in str(e)
    else:
        raise AssertionError("expected ValueError for raw JSON framing")


# ── handle_client: request validation ───────────────────────────────

class _FakeSockIO(_FakeSock):
    """_FakeSock plus the sendall/close surface handle_client drives."""

    def __init__(self, data=b""):
        super().__init__(data)
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def close(self):
        pass


def _request_sock(payload):
    body = json.dumps(payload).encode("utf-8")
    return _FakeSockIO(struct.pack("!I", len(body)) + body)


def _decode_response(sock):
    length = struct.unpack("!I", sock.sent[:4])[0]
    return json.loads(sock.sent[4:4 + length].decode("utf-8"))


def test_handle_client_rejects_non_string_audio():
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    sock = _request_sock({"audio_base64": 12345})
    d.handle_client(sock, ("test", 0))
    resp = _decode_response(sock)
    assert resp["success"] is False


def test_handle_client_rejects_invalid_base64():
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    sock = _request_sock({"audio_base64": "not base64!! at all"})
    d.handle_client(sock, ("test", 0))
    resp = _decode_response(sock)
    assert resp["success"] is False
    assert "not valid base64" in resp["error"]


def test_handle_client_rejects_long_language():
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    sock = _request_sock({"audio_base64": "AAAA", "language": "x" * 17})
    d.handle_client(sock, ("test", 0))
    resp = _decode_response(sock)
    assert resp["success"] is False
    assert "language" in resp["error"]


def test_handle_client_rejects_non_string_language():
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    sock = _request_sock({"audio_base64": "AAAA", "language": 5})
    d.handle_client(sock, ("test", 0))
    resp = _decode_response(sock)
    assert resp["success"] is False


# ── transcribe: duration cap and logging ────────────────────────────

def _silence_wav_bytes(seconds=0.5, sr=16000, channels=1):
    frames = int(seconds * sr)
    shape = frames if channels == 1 else (frames, channels)
    data = np.zeros(shape, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_transcribe_returns_text(monkeypatch):
    audio_bytes = _silence_wav_bytes()
    e = RemoteEngine(base_url="http://127.0.0.1:1/v1", model="up/asr")
    monkeypatch.setattr(e, "transcribe",
                        lambda wav_path, language=None: {"text": "hello", "language": "en"})
    d = relaystt_daemon.RelaySTTDaemon(engine=e)

    result = d.transcribe(audio_bytes)

    assert result["text"] == "hello"


def test_transcribe_rejects_audio_over_max_duration(monkeypatch):
    audio_bytes = _silence_wav_bytes()
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    monkeypatch.setattr(d, "_ensure_wav", lambda input_path: input_path)

    class _Info:
        duration = relaystt_daemon.MAX_AUDIO_SECONDS + 1

    monkeypatch.setattr(sf, "info", lambda path: _Info())

    try:
        d.transcribe(audio_bytes)
    except ValueError as e:
        assert "too long" in str(e)
    else:
        raise AssertionError("expected ValueError for over-long audio")


def test_transcript_text_not_logged(monkeypatch, capsys):
    audio_bytes = _silence_wav_bytes()
    e = RemoteEngine(base_url="http://127.0.0.1:1/v1", model="up/asr")
    secret_text = "the quick brown fox jumps"
    monkeypatch.setattr(e, "transcribe",
                        lambda wav_path, language=None: {"text": secret_text, "language": "en"})
    d = relaystt_daemon.RelaySTTDaemon(engine=e)

    d.transcribe(audio_bytes)

    out = capsys.readouterr().out
    assert secret_text not in out
    assert "chars=" in out


# ── _ensure_wav: argv, real conversion, timeout cleanup ─────────────

def test_ensure_wav_argv_has_safety_flags(tmp_path, monkeypatch):
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        raise subprocess.CalledProcessError(1, argv, stderr=b"boom")

    monkeypatch.setattr(relaystt_daemon.subprocess, "run", fake_run)
    input_path = str(tmp_path / "in.raw")
    with open(input_path, "wb") as f:
        f.write(b"not audio")

    d._ensure_wav(input_path)

    argv = captured["argv"]
    assert "-nostdin" in argv
    pw_index = argv.index("-protocol_whitelist")
    assert argv[pw_index + 1] == "file"
    assert pw_index < argv.index("-i")
    assert "-t" in argv


def test_ensure_wav_real_run_converts_to_16k_mono(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    input_path = str(tmp_path / "in.wav")
    frames = int(0.5 * 22050)
    data = np.zeros((frames, 2), dtype=np.float32)
    sf.write(input_path, data, 22050, subtype="PCM_16")

    out_path = d._ensure_wav(input_path)

    info = sf.info(out_path)
    assert info.samplerate == 16000
    assert info.channels == 1


def test_ensure_wav_timeout_removes_partial_file(tmp_path, monkeypatch):
    d = relaystt_daemon.RelaySTTDaemon(engine=_StubEngine())
    input_path = str(tmp_path / "in.raw")
    with open(input_path, "wb") as f:
        f.write(b"data")
    wav_path = input_path + ".converted.wav"

    def fake_run(argv, **kwargs):
        with open(wav_path, "wb") as f:
            f.write(b"partial")
        raise subprocess.TimeoutExpired(cmd=argv, timeout=relaystt_daemon.FFMPEG_TIMEOUT_S)

    monkeypatch.setattr(relaystt_daemon.subprocess, "run", fake_run)

    result = d._ensure_wav(input_path)

    assert result == input_path
    assert not os.path.exists(wav_path)


# ── pinned_transport: parse_pins ─────────────────────────────────────

_SAMPLE_FP = "ab" * 32  # 64 lowercase hex chars


def test_parse_pins_normalises_colons_and_case():
    colonized = ":".join(_SAMPLE_FP[i:i + 2] for i in range(0, len(_SAMPLE_FP), 2)).upper()
    assert pinned_transport.parse_pins(colonized) == [_SAMPLE_FP]


def test_parse_pins_accepts_list_input():
    assert pinned_transport.parse_pins([_SAMPLE_FP.upper(), _SAMPLE_FP]) == [_SAMPLE_FP, _SAMPLE_FP]


def test_parse_pins_rejects_malformed_entry():
    try:
        pinned_transport.parse_pins("not-a-fingerprint")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a malformed pin")


def test_parse_pins_empty_input_returns_empty_list():
    assert pinned_transport.parse_pins(None) == []
    assert pinned_transport.parse_pins("") == []
    assert pinned_transport.parse_pins([]) == []


# ── pinned_transport: assert_transport_config ────────────────────────

def test_assert_transport_config_http_non_loopback_is_fatal():
    try:
        pinned_transport.assert_transport_config("http://198.51.100.10:8080/v1", None, [])
    except ValueError as e:
        assert "https" in str(e)
    else:
        raise AssertionError("expected ValueError for non-loopback http")


def test_assert_transport_config_http_loopback_is_ok():
    pinned_transport.assert_transport_config("http://127.0.0.1:8080/v1", None, [])
    pinned_transport.assert_transport_config("http://localhost:8080/v1", None, [])
    pinned_transport.assert_transport_config("http://[::1]:8080/v1", None, [])


def test_assert_transport_config_http_with_pins_is_fatal():
    try:
        pinned_transport.assert_transport_config("http://127.0.0.1:8080/v1", None, [_SAMPLE_FP])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for http with pins")


def test_assert_transport_config_http_with_ca_file_is_fatal(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("does not need to parse -- rejected before it is read")
    try:
        pinned_transport.assert_transport_config("http://127.0.0.1:8080/v1", str(ca), [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for http with a ca_file")


def test_assert_transport_config_missing_ca_file_is_fatal(tmp_path):
    missing = str(tmp_path / "does-not-exist.pem")
    try:
        pinned_transport.assert_transport_config("https://example.com/v1", missing, [])
    except ValueError as e:
        assert missing in str(e)
    else:
        raise AssertionError("expected ValueError for a missing ca_file")


def test_assert_transport_config_malformed_pin_is_fatal():
    try:
        pinned_transport.assert_transport_config("https://example.com/v1", None, ["not-hex"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a malformed pin")


def test_assert_transport_config_missing_url_is_fatal():
    try:
        pinned_transport.assert_transport_config("", None, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a missing URL")


def test_assert_transport_config_bad_scheme_is_fatal():
    try:
        pinned_transport.assert_transport_config("ftp://host/v1", None, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-http(s) scheme")


# ── TLS fixtures: real certs via the openssl CLI ─────────────────────

def _run_openssl(args):
    subprocess.run(["openssl"] + args, check=True, capture_output=True)


if pytest is not None:
    @pytest.fixture
    def tls_certs(tmp_path):
        if shutil.which("openssl") is None:
            pytest.skip("openssl not installed")

        ca_key = tmp_path / "ca.key"
        ca_crt = tmp_path / "ca.crt"
        _run_openssl(["genrsa", "-out", str(ca_key), "2048"])
        _run_openssl([
            "req", "-x509", "-new", "-nodes", "-key", str(ca_key),
            "-sha256", "-days", "2", "-out", str(ca_crt),
            "-subj", "/CN=relaySTT-test-CA",
        ])

        def _make_leaf(name):
            key = tmp_path / f"{name}.key"
            csr = tmp_path / f"{name}.csr"
            crt = tmp_path / f"{name}.crt"
            ext = tmp_path / f"{name}.ext"
            ext.write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
            _run_openssl(["genrsa", "-out", str(key), "2048"])
            _run_openssl(["req", "-new", "-key", str(key), "-out", str(csr),
                          "-subj", f"/CN={name}"])
            _run_openssl([
                "x509", "-req", "-in", str(csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
                "-CAcreateserial", "-out", str(crt), "-days", "2", "-sha256",
                "-extfile", str(ext),
            ])
            return str(key), str(crt)

        leaf1_key, leaf1_crt = _make_leaf("leaf1")
        leaf2_key, leaf2_crt = _make_leaf("leaf2")

        def _fingerprint(crt_path):
            with open(crt_path) as f:
                pem = f.read()
            return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()

        return {
            "ca_crt": str(ca_crt),
            "leaf1_key": leaf1_key, "leaf1_crt": leaf1_crt, "leaf1_fp": _fingerprint(leaf1_crt),
            "leaf2_key": leaf2_key, "leaf2_crt": leaf2_crt, "leaf2_fp": _fingerprint(leaf2_crt),
        }


class _TranscribeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"text": "pinned ok", "language": "en"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def _start_tls_server(cert_path, key_path):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _TranscribeHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _real_wav(tmp_path, seconds=0.2, sr=16000):
    frames = int(seconds * sr)
    data = np.zeros(frames, dtype=np.float32)
    path = tmp_path / "clip.wav"
    sf.write(str(path), data, sr, subtype="PCM_16")
    return str(path)


# ── TLS: in-process, real certificates ────────────────────────────────

def test_https_with_ca_no_pins_succeeds(tmp_path, tls_certs):
    httpd = _start_tls_server(tls_certs["leaf1_crt"], tls_certs["leaf1_key"])
    try:
        port = httpd.server_address[1]
        e = RemoteEngine(base_url=f"https://127.0.0.1:{port}/v1", model="up/asr",
                         ca_file=tls_certs["ca_crt"])
        result = e.transcribe(_real_wav(tmp_path))
        assert result["text"] == "pinned ok"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_https_without_ca_uses_system_roots_and_fails(tmp_path, tls_certs):
    httpd = _start_tls_server(tls_certs["leaf1_crt"], tls_certs["leaf1_key"])
    try:
        port = httpd.server_address[1]
        e = RemoteEngine(base_url=f"https://127.0.0.1:{port}/v1", model="up/asr")
        try:
            e.transcribe(_real_wav(tmp_path))
        except RuntimeError as err:
            assert f"127.0.0.1:{port}" in str(err)
        else:
            raise AssertionError("expected RuntimeError for an untrusted self-signed cert")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_https_with_pin_of_matching_leaf_succeeds(tmp_path, tls_certs):
    httpd = _start_tls_server(tls_certs["leaf1_crt"], tls_certs["leaf1_key"])
    try:
        port = httpd.server_address[1]
        e = RemoteEngine(base_url=f"https://127.0.0.1:{port}/v1", model="up/asr",
                         ca_file=tls_certs["ca_crt"], pins=tls_certs["leaf1_fp"])
        result = e.transcribe(_real_wav(tmp_path))
        assert result["text"] == "pinned ok"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_https_with_pin_mismatch_blocks_valid_cert_from_same_ca(tmp_path, tls_certs):
    """The MITM-with-a-valid-cert case: leaf2 is signed by the same trusted CA
    as leaf1, so chain validation alone would accept it. The pin must still
    reject it."""
    httpd = _start_tls_server(tls_certs["leaf2_crt"], tls_certs["leaf2_key"])
    try:
        port = httpd.server_address[1]
        e = RemoteEngine(base_url=f"https://127.0.0.1:{port}/v1", model="up/asr",
                         ca_file=tls_certs["ca_crt"], pins=tls_certs["leaf1_fp"])
        try:
            e.transcribe(_real_wav(tmp_path))
        except RuntimeError as err:
            assert "not pinned" in str(err)
        else:
            raise AssertionError("expected RuntimeError for a pin mismatch")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_env_overrides_cli_for_ca_and_pins(tls_certs, monkeypatch):
    monkeypatch.setenv("RELAYSTT_REMOTE_CA", tls_certs["leaf1_crt"])
    monkeypatch.setenv("RELAYSTT_REMOTE_PIN_SHA256", tls_certs["leaf1_fp"])
    e = RemoteEngine(base_url="https://127.0.0.1:1/v1", model="up/asr",
                     ca_file=tls_certs["ca_crt"], pins=tls_certs["leaf2_fp"])
    assert e.ca_file == tls_certs["leaf1_crt"]
    assert e.pins == [tls_certs["leaf1_fp"]]


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
        if "monkeypatch" in params or "tls_certs" in params:
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
