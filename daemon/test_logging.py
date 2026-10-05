"""Structured-log tests: the writer (log_line.Logger) and the daemon's use of it.

Lines are checked against the shared schema fixture with a small keyword
checker local to this file, so a schema keyword nobody here understands fails
loudly instead of passing unchecked.
"""

import base64
import io
import json
import os
import re
import struct
import subprocess
import sys

import pytest
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import log_line
import relaystt_daemon
from relaystt_daemon import RemoteEngine

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "testdata", "logging-schema.json"), encoding="utf-8") as _f:
    SCHEMA = json.load(_f)

HEX32 = re.compile(r"^[0-9a-f]{32}$")
_IGNORED = {"$schema", "$id", "title", "description"}


# ── test-local schema checker ────────────────────────────────────

def check(value, schema, path="$"):
    """Return a list of violations; raise on any keyword it does not know."""
    errs = []
    for kw, arg in schema.items():
        if kw in _IGNORED:
            continue
        if kw == "type":
            ok = {
                "object": isinstance(value, dict),
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
            }[arg]
            if not ok:
                errs.append(f"{path}: not {arg}")
        elif kw == "required":
            errs += [f"{path}: missing {k}" for k in arg if k not in value]
        elif kw == "enum":
            if value not in arg:
                errs.append(f"{path}: {value!r} not in enum")
        elif kw == "pattern":
            if isinstance(value, str) and not re.search(arg, value):
                errs.append(f"{path}: {value!r} does not match {arg}")
        elif kw == "maxLength":
            if isinstance(value, str) and len(value) > arg:
                errs.append(f"{path}: longer than {arg}")
        elif kw == "minLength":
            if isinstance(value, str) and len(value) < arg:
                errs.append(f"{path}: shorter than {arg}")
        elif kw == "minimum":
            if isinstance(value, int) and value < arg:
                errs.append(f"{path}: below {arg}")
        elif kw == "additionalProperties":
            if arg is not True and arg is not False:
                raise ValueError("additionalProperties schema form unsupported")
            if arg is False and isinstance(value, dict):
                extra = set(value) - set(schema.get("properties", {}))
                errs += [f"{path}: extra {k}" for k in extra]
        elif kw == "properties":
            if isinstance(value, dict):
                for k, sub in arg.items():
                    if k in value:
                        errs += check(value[k], sub, f"{path}.{k}")
        else:
            raise ValueError(f"unsupported schema keyword: {kw}")
    return errs


def parse_lines(text):
    """Every line must be a JSON object that validates; return them."""
    out = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        assert isinstance(obj, dict), raw
        assert check(obj, SCHEMA) == [], raw
        out.append(obj)
    return out


def _logger(level=None, clock=None, service=None):
    buf = io.StringIO()
    env = {}
    if level:
        env["RELAY_LOG_LEVEL"] = level
    if service:
        env["RELAY_SERVICE_ID"] = service
    kw = {"stream": buf, "env": env, "wall": lambda: 1_700_000_000.123}
    if clock:
        kw["monotonic"] = clock
    return log_line.Logger(**kw), buf


# ── the checker itself ───────────────────────────────────────────

def test_checker_rejects_unknown_keyword_and_bad_line():
    with pytest.raises(ValueError):
        check({}, {"type": "object", "oneOf": []})
    assert check({"level": "loud"}, SCHEMA) != []


# ── writer ───────────────────────────────────────────────────────

def test_every_level_line_validates_against_schema():
    lg, buf = _logger("debug")
    lg.log("error", "e", error="boom")
    lg.log("warn", "w")
    lg.log("info", "i", op="stt.transcribe", duration_ms=12)
    lg.log("debug", "d")
    lines = parse_lines(buf.getvalue())
    assert [l["level"] for l in lines] == ["error", "warn", "info", "debug"]
    assert lines[0]["service"] == "relaystt-daemon"
    assert lines[0]["ts"].endswith("Z")


def test_writer_defaults_ownership_and_truncation():
    lg, buf = _logger(service="acme-svc")
    lg.log("warn", "x" * 900, status=503, error="e" * 900,
           **{"ts": "spoof", "service": "spoof", "trace_id": "evil-trace-id",
              "level": "spoofl", "msg": "spoofm"})
    lg.log("info", "ok line")
    warn, info = parse_lines(buf.getvalue())
    assert warn["level"] == "warn" and warn["service"] == "acme-svc"
    assert warn["attr_ts"] == "spoof" and warn["attr_service"] == "spoof"
    assert warn["attr_trace_id"] == "evil-trace-id" and warn["trace_id"] == ""
    assert warn["attr_level"] == "spoofl" and warn["attr_msg"] == "spoofm"
    assert warn["ts"].endswith("Z") and warn["msg"] == "x" * 500
    assert warn["http_status"] == 503 and warn["status"] == "error"
    assert len(warn["msg"]) == 500 and len(warn["error"]) == 500
    assert (info["op"], info["status"], info["duration_ms"], info["error"]) == ("log", "ok", 0, "")


def test_trace_id_comes_only_from_trace_scope():
    lg, buf = _logger()
    with log_line.trace_scope("scope-id-1"):
        lg.log("info", "m", **{"trace_id": "evil-trace-id"})
    (line,) = parse_lines(buf.getvalue())
    assert line["trace_id"] == "scope-id-1"
    assert line["attr_trace_id"] == "evil-trace-id"


def test_level_env_controls_debug():
    lg, buf = _logger()
    lg.log("debug", "hidden")
    lg.log("info", "shown")
    assert [l["msg"] for l in parse_lines(buf.getvalue())] == ["shown"]
    lg, buf = _logger("debug")
    lg.log("debug", "visible")
    assert [l["msg"] for l in parse_lines(buf.getvalue())] == ["visible"]


def test_debug_window_closes_after_30_minutes_with_one_warn():
    now = [0.0]
    lg, buf = _logger("debug", clock=lambda: now[0])
    lg.log("debug", "early")
    now[0] = 1801.0
    lg.log("debug", "late-debug")
    lg.log("info", "late-info")
    lg.log("debug", "later-debug")
    lines = parse_lines(buf.getvalue())
    assert [(l["level"], l["msg"]) for l in lines][0] == ("debug", "early")
    rest = lines[1:]
    assert [l["level"] for l in rest] == ["warn", "info"]
    assert rest[1]["msg"] == "late-info"


@pytest.mark.parametrize("s,ok", [
    ("abcd1234", True), ("a" * 64, True), ("A_b-C_d-9", True),
    ("short", False), ("a" * 65, False), ("bad chars!!", False),
    ("", False), ("abcdefgh\n", False), (12345678, False),
])
def test_valid_trace_id(s, ok):
    assert log_line.valid_trace_id(s) is ok


def test_new_trace_id_shape_and_scope():
    assert HEX32.match(log_line.new_trace_id())
    with log_line.trace_scope("scope-id-1"):
        assert log_line.current_trace_id() == "scope-id-1"
    assert log_line.current_trace_id() != "scope-id-1"


# ── daemon ───────────────────────────────────────────────────────

class _Sock:
    def __init__(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self._data = struct.pack("!I", len(body)) + body
        self.sent = b""

    def recv(self, n):
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def sendall(self, data):
        self.sent += data

    def close(self):
        pass

    def response(self):
        n = struct.unpack("!I", self.sent[:4])[0]
        return json.loads(self.sent[4:4 + n])


class _Engine:
    label = "<stub>"
    model = "stub"

    def __init__(self, text=""):
        self.text = text

    def transcribe(self, wav_path, language=None):
        return {"text": self.text, "language": language or "en"}


def _wav_b64(seconds=0.5):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype="int16"), 16000,
             format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _run(daemon, **frame):
    frame.setdefault("audio_base64", _wav_b64())
    sock = _Sock(frame)
    daemon.handle_client(sock, ("testbox", 0))
    return sock.response()


def test_daemon_stderr_is_all_schema_valid_json_with_transcribe_op(capsys):
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine("hello"))
    assert _run(d)["success"] is True
    lines = parse_lines(capsys.readouterr().err)
    ops = [l for l in lines if l["op"] == "stt.transcribe"]
    assert len(ops) == 1
    assert ops[0]["level"] == "info" and ops[0]["status"] == "ok"
    assert isinstance(ops[0]["duration_ms"], int)


def test_valid_trace_id_is_kept_in_daemon_lines(capsys):
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    _run(d, trace_id="req-trace-0001")
    lines = parse_lines(capsys.readouterr().err)
    assert lines and {l["trace_id"] for l in lines} == {"req-trace-0001"}


@pytest.mark.parametrize("bad", ["short", "bad chars!!!", "", "abcdefgh\n", None])
def test_invalid_or_missing_trace_id_is_replaced(bad, capsys):
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    frame = {} if bad is None else {"trace_id": bad}
    _run(d, **frame)
    err = capsys.readouterr().err
    lines = parse_lines(err)
    ids = {l["trace_id"] for l in lines}
    assert lines and len(ids) == 1 and HEX32.match(ids.pop())
    if bad:
        assert bad.strip() not in err


@pytest.mark.parametrize("url,expect_header", [
    ("unix:/tmp/rstt-model.sock", True),
    ("http://127.0.0.1:8080/v1", True),
    ("https://198.51.100.10:8080/v1", False),
])
def test_engine_request_trace_header(url, expect_header, monkeypatch):
    engine = RemoteEngine(base_url=url, model="up/asr")
    seen = {}

    class _Resp:
        def read(self):
            return b'{"text":"t","language":"en"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp()

    monkeypatch.setattr(engine._opener, "open", fake_open)
    d = relaystt_daemon.RelaySTTDaemon(engine=engine)
    assert _run(d, trace_id="req-trace-0002")["success"] is True
    if expect_header:
        assert seen["headers"]["x-trace-id"] == "req-trace-0002"
    else:
        assert "x-trace-id" not in seen["headers"]


def test_ffmpeg_failure_logs_exit_code_not_ffmpeg_text(monkeypatch, capsys):
    def boom(cmd, *a, **kw):
        raise subprocess.CalledProcessError(
            7, cmd + ["CANARY-CMD-ARG"], stderr=b"SECRET-FFMPEG-TEXT")

    monkeypatch.setattr(relaystt_daemon.subprocess, "run", boom)
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    _run(d, audio_base64=base64.b64encode(b"not audio at all " * 8).decode())
    err = capsys.readouterr().err
    lines = parse_lines(err)
    assert any("exit code 7" in l["error"] for l in lines if l["error"])
    assert "SECRET-FFMPEG-TEXT" not in err and "CANARY-CMD-ARG" not in err


def test_canary_secrets_never_reach_stderr(monkeypatch, capsys):
    key = "CANARY-KEY-" + "k" * 30
    text = "CANARY-TRANSCRIPT-" + "t" * 20
    monkeypatch.setenv("RELAYSTT_REMOTE_API_KEY", key)
    audio = _wav_b64()
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine(text))
    _run(d, audio_base64=audio, trace_id="req-trace-0003")
    err = capsys.readouterr().err
    assert err.strip()
    for secret in (key, text, audio[:40]):
        assert secret not in err


# ── one stt.transcribe line per handled request ──────────────────

def _transcribe_lines(err):
    return [l for l in parse_lines(err) if l["op"] == "stt.transcribe"]


def test_zero_duration_wav_logs_one_info_ok_line(capsys):
    hdr = io.BytesIO()
    sf.write(hdr, np.zeros(0, dtype="int16"), 16000, format="WAV", subtype="PCM_16")
    assert len(hdr.getvalue()) == 44
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    _run(d, audio_base64=base64.b64encode(hdr.getvalue()).decode())
    ops = _transcribe_lines(capsys.readouterr().err)
    assert len(ops) == 1
    assert ops[0]["level"] == "info" and ops[0]["status"] == "ok"
    assert isinstance(ops[0]["duration_ms"], int)


@pytest.mark.parametrize("payload", [
    {},
    {"audio_base64": "!!!not base64!!!"},
    {"audio_base64": "AAAA", "language": 123},
], ids=["missing-audio", "invalid-base64", "non-string-language"])
def test_rejected_request_logs_one_warn_denied_line(payload, capsys):
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    sock = _Sock(payload)
    d.handle_client(sock, ("testbox", 0))
    assert sock.response()["success"] is False
    ops = _transcribe_lines(capsys.readouterr().err)
    assert len(ops) == 1
    assert ops[0]["level"] == "warn" and ops[0]["status"] == "denied"
    assert ops[0]["error"] != "" and isinstance(ops[0]["duration_ms"], int)


def test_ping_logs_no_transcribe_line(capsys):
    d = relaystt_daemon.RelaySTTDaemon(engine=_Engine())
    sock = _Sock({"action": "ping"})
    d.handle_client(sock, ("testbox", 0))
    assert _transcribe_lines(capsys.readouterr().err) == []


# ── error text never carries a remote body ───────────────────────

def _serve_500(canary):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            body = json.dumps({"error": {"message": canary}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_remote_http_500_body_never_reaches_stderr(capsys):
    canary = "CANARY-BODY-" + "b" * 20
    httpd = _serve_500(canary)
    try:
        engine = RemoteEngine(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", model="m")
        d = relaystt_daemon.RelaySTTDaemon(engine=engine)
        assert _run(d)["success"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
    err = capsys.readouterr().err
    assert canary not in err
    ops = _transcribe_lines(err)
    assert len(ops) == 1 and ops[0]["status"] == "error"


def test_engine_runtime_error_text_never_reaches_stderr(capsys):
    canary = "CANARY-RTERR-" + "r" * 20
    engine = _Engine()

    def boom(p, language=None):
        raise RuntimeError(canary)

    engine.transcribe = boom
    d = relaystt_daemon.RelaySTTDaemon(engine=engine)
    assert _run(d)["success"] is False
    err = capsys.readouterr().err
    assert canary not in err
    ops = _transcribe_lines(err)
    assert len(ops) == 1 and ops[0]["status"] == "error"
