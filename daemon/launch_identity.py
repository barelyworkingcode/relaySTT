#!/usr/bin/env python3
"""relaySTT <-> Relay launch identity (the Hello handshake).

When relay launches the daemon it sets RELAY_BRIDGE_SOCKET / RELAY_SERVICE_ID /
RELAY_LAUNCH_FD and hands a one-shot launch secret over an inherited pipe (fd
3). No relay credential is ever in the environment. At startup the daemon
drains that pipe and presents the secret in a `Hello` on the bridge socket;
relay binds this process's kernel audit token as the service's identity, and
every later bridge request from this process carries no token at all.

relaySTT registers no manifest and has no bridge requests beyond Hello today
— this module only establishes launch identity, so relay knows the process
it launched came up honestly. Without RELAY_LAUNCH_FD (a standalone run or a
unit test) `establish_launch_identity` is a no-op.

Ported from relayTTS's daemon/relay_bridge.py (relay/docs/launch-identity.md
is the protocol both implement).
"""
import json
import os
import re
import socket

# Env-var ABI relay sets at launch. None of these is secret.
ENV_BRIDGE_SOCKET = "RELAY_BRIDGE_SOCKET"
ENV_SERVICE_ID = "RELAY_SERVICE_ID"
ENV_LAUNCH_FD = "RELAY_LAUNCH_FD"

_MAX_LINE = 10 * 1024 * 1024  # mirrors bridge.MaxMessageSize
_LAUNCH_SECRET_RE = re.compile(r"[0-9a-f]{64}")
# A well-formed secret is 64 bytes; anything past this is malformed, so the
# read stops rather than buffering whatever an unexpected fd produces.
_LAUNCH_SECRET_READ_CAP = 4096
_HELLO_TIMEOUT_S = 5.0


class LaunchIdentityError(RuntimeError):
    """Relay launched us but the launch identity could not be established.
    Messages never contain the secret."""


def _read_line(c: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf and len(buf) < _MAX_LINE:
        chunk = c.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", "replace")


def read_launch_secret(fd: int) -> str:
    """Drain `fd` to EOF, close it, and return the 64-lowercase-hex secret."""
    data = b""
    try:
        while len(data) <= _LAUNCH_SECRET_READ_CAP:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            data += chunk
    except OSError as e:
        raise LaunchIdentityError(f"cannot read launch fd {fd}: {e.strerror}") from None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        secret = data.decode("ascii")
    except UnicodeDecodeError:
        secret = ""
    if not _LAUNCH_SECRET_RE.fullmatch(secret):
        raise LaunchIdentityError(
            f"launch secret malformed ({len(data)} bytes; want 64 lowercase hex)")
    return secret


def build_hello_payload(service_id: str, secret: str) -> bytes:
    return json.dumps({"type": "Hello", "name": service_id,
                       "token": secret}).encode("utf-8") + b"\n"


def send_hello(bridge_sock: str, service_id: str, secret: str,
               timeout: float = _HELLO_TIMEOUT_S) -> dict:
    """Present the launch secret; return the OK frame's `data` on success."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(timeout)
            c.connect(bridge_sock)
            c.sendall(build_hello_payload(service_id, secret))
            line = _read_line(c)
    except OSError as e:
        raise LaunchIdentityError(f"Hello transport failure: {e.__class__.__name__}") from None
    if not line:
        raise LaunchIdentityError("bridge closed without answering Hello")
    try:
        resp = json.loads(line)
    except ValueError:
        resp = None
    if not isinstance(resp, dict):
        raise LaunchIdentityError("malformed Hello response")
    if resp.get("type") == "Error":
        raise LaunchIdentityError(
            f"Hello refused: code {resp.get('code')}: {resp.get('message')}")
    data = resp.get("data")
    if (resp.get("type") != "OK" or not isinstance(data, dict)
            or data.get("service_id") != service_id
            or not isinstance(data.get("relay_pid"), int)
            or isinstance(data.get("relay_pid"), bool)
            or data.get("relay_pid") <= 0):
        raise LaunchIdentityError("malformed Hello response")
    return data


def establish_launch_identity(environ=os.environ) -> bool:
    """Run the launch handshake if relay launched us.

    Returns False when RELAY_LAUNCH_FD is unset (standalone), True once relay
    has bound this process's identity, and raises LaunchIdentityError on any
    failure in between. Must run before the daemon spawns any child.
    """
    raw_fd = environ.get(ENV_LAUNCH_FD)
    if raw_fd is None:
        return False
    # Deliberate: removed before anything else can fail, so no child spawned
    # later (ffmpeg) inherits a pointer to a launch pipe.
    del environ[ENV_LAUNCH_FD]
    try:
        fd = int(raw_fd)
    except ValueError:
        fd = -1
    if fd < 0:
        raise LaunchIdentityError(f"{ENV_LAUNCH_FD} is not a file descriptor")
    secret = read_launch_secret(fd)
    bridge_sock = environ.get(ENV_BRIDGE_SOCKET, "")
    service_id = environ.get(ENV_SERVICE_ID, "")
    if not bridge_sock or not service_id:
        raise LaunchIdentityError(
            f"{ENV_LAUNCH_FD} is set but {ENV_BRIDGE_SOCKET} or {ENV_SERVICE_ID} is empty")
    send_hello(bridge_sock, service_id, secret)
    return True
