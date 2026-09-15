"""TLS transport for the remote STT engine, plus the `unix:<path>` transport
used when the remote is relay's own model.sock.

Certificate verification is always on; there is no skip-verify / insecure
flag anywhere in this module, and none may be added — a handshake that can't
be verified must fail the request, not silently downgrade to plaintext-grade
trust. The unix transport has no analogous knob either — there is nothing to
pin, since relay identifies this service by its launch identity, not a
certificate (see `UnixSocketOpener`).

Shares its shape with the equivalent module in the sibling relayTTS repo
(same rules, same function names) but is a standalone copy — nothing here
imports across repos.
"""
import hashlib
import http.client
import io
import socket
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_HEX_DIGITS = set("0123456789abcdef")
_UNIX_PREFIX = "unix:"


def _normalize_pin(raw):
    pin = raw.strip().replace(":", "").lower()
    if len(pin) != 64 or any(c not in _HEX_DIGITS for c in pin):
        raise ValueError(
            f"'{raw}' is not a valid SHA-256 fingerprint: expected 64 hex "
            f"characters (colons optional)")
    return pin


def parse_pins(text_or_list):
    """Normalise a comma-separated string or a list of pins into lowercase
    hex fingerprints with colons stripped. Raises ValueError on anything that
    isn't 64 hex chars once normalised."""
    if not text_or_list:
        return []
    items = text_or_list.split(",") if isinstance(text_or_list, str) else list(text_or_list)
    return [_normalize_pin(item) for item in items if item and item.strip()]


def is_unix_url(url) -> bool:
    """True for the `unix:<abs path>` form of a remote base_url — HTTP over
    AF_UNIX rather than TCP, with no bearer header ever sent (relay
    identifies the caller by its launch identity)."""
    return bool(url) and url.startswith(_UNIX_PREFIX)


def parse_unix_socket_path(url: str) -> str:
    """Extract and validate the socket path from a `unix:<path>` base_url.

    The path is handed straight to socket.connect(), never joined against a
    directory, so a relative path would silently resolve against whatever
    the daemon's CWD happens to be at connect time rather than the path the
    operator wrote — refused here instead, at startup.
    """
    path = url[len(_UNIX_PREFIX):]
    if not path.startswith("/"):
        raise ValueError(
            f"remote STT base_url {url!r} names a unix socket with a relative "
            "path; use an absolute path, e.g. unix:/path/to/model.sock")
    return path


def assert_transport_config(url, ca_file, pins):
    """Validate a remote-transport configuration before any connection is
    attempted. Raises ValueError describing what to change."""
    if not url:
        raise ValueError(
            "remote STT needs a URL: pass --remote-url or set RELAYSTT_REMOTE_URL")

    if is_unix_url(url):
        parse_unix_socket_path(url)
        pins = [_normalize_pin(p) for p in (pins or [])]
        if ca_file or pins:
            raise ValueError(
                f"--remote-ca / --remote-pin-sha256 have no effect on a unix "
                f"socket base_url ({url!r}); relay identifies this service by "
                "its launch identity, not TLS — drop them")
        return

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"remote STT URL must be http or https, got '{parsed.scheme or url}'")

    is_loopback = parsed.hostname in _LOOPBACK_HOSTS
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError(
            f"remote STT at {parsed.hostname} is plain HTTP over the network; "
            f"use https:// and pin its CA with --remote-ca / RELAYSTT_REMOTE_CA")

    if parsed.scheme == "http" and (ca_file or pins):
        raise ValueError(
            "--remote-ca / --remote-pin-sha256 apply to https only; the "
            "configured URL is http")

    if ca_file:
        try:
            ssl.create_default_context(cafile=ca_file)
        except (OSError, ssl.SSLError) as e:
            raise ValueError(f"--remote-ca {ca_file} is not a readable PEM bundle: {e}") from None

    for pin in (pins or []):
        _normalize_pin(pin)


def build_opener(url, ca_file, pins):
    """Build a urllib opener enforcing the given trust configuration. Never
    calls install_opener — the caller holds the returned opener itself, so a
    misconfigured engine can't change global request behavior.

    `unix:<path>` gets a `UnixSocketOpener` — HTTP over AF_UNIX, no TLS and
    no bearer header (see that class).
    """
    if is_unix_url(url):
        return UnixSocketOpener(parse_unix_socket_path(url))

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        return urllib.request.build_opener()

    ctx = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    if not pins:
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    pin_set = set(pins)

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            der = self.sock.getpeercert(binary_form=True)
            fingerprint = hashlib.sha256(der).hexdigest()
            if fingerprint not in pin_set:
                self.sock.close()
                raise ssl.SSLCertVerificationError(
                    f"remote certificate fingerprint {fingerprint} is not pinned")

    class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_PinnedHTTPSConnection, req, context=ctx)

    return urllib.request.build_opener(_PinnedHTTPSHandler())


# ── unix:<path> transport: HTTP over AF_UNIX, no TLS, no bearer header ──
#
# Used when the remote is relay's own model.sock: relay identifies the caller
# by the kernel's audit token on the connection, not by anything in the
# request, so this transport never has a credential to attach in the first
# place.

class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client.HTTPConnection dialed over AF_UNIX instead of AF_INET.

    `host` is a fixed placeholder — relay's model.sock does not
    hostname-route, but HTTP/1.1 still wants a Host header, and http.client
    derives one from `self.host`.
    """

    def __init__(self, sock_path: str, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        super().__init__("model.sock", timeout=timeout)
        self._sock_path = sock_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Deliberate: the family is hardcoded above, so this can never fire
        # on the code as written today. It stays as a hard stop, checked
        # before connect() so it fires on the socket's own family rather
        # than on whatever OSError a mismatched connect() target happens to
        # produce — against the one change that would matter: a future edit
        # that lets `sock` come from somewhere else (a passed-in factory, a
        # retry path) and quietly reintroduces a TCP fallback for a URL an
        # operator wrote as `unix:`.
        if sock.family != socket.AF_UNIX:
            sock.close()
            raise AssertionError(
                f"unix transport dialed a {sock.family!r} socket, not AF_UNIX")
        if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(self.timeout)
        sock.connect(self._sock_path)
        self.sock = sock


class _BufferedResponse:
    """A fully-read response, matching the subset of urllib's response
    interface RemoteEngine uses (`read()`, used as a context manager). The
    body is read out under UnixSocketOpener's lock before this is handed
    back, so the caller can take as long as it likes with it without holding
    the connection open for anyone else — see UnixSocketOpener.open."""

    def __init__(self, status: int, body: bytes):
        self.status = self.code = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class UnixSocketOpener:
    """RemoteEngine's transport for a `unix:<path>` base_url: HTTP/1.1 over
    AF_UNIX, one kept-alive connection reused across requests rather than one
    dialed and torn down per call — the file-descriptor-leak failure mode a
    per-call transport has if nothing ever closes it.

    A single connection, not a pool: `_lock` serializes every request,
    because RelaySTTDaemon dispatches one thread per TCP client and
    http.client connections handle one in-flight request at a time.
    Transcription calls are already effectively serialized against a single
    remote model, so this costs nothing a pool would have avoided. A
    connection the peer has since idle-closed — relay is an ordinary
    HTTP/1.1 server about that — is detected and replaced once before
    giving up.
    """

    def __init__(self, sock_path: str):
        self._sock_path = sock_path
        self._lock = threading.Lock()
        self._conn: "_UnixHTTPConnection | None" = None

    def _drop_connection(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def open(self, req, timeout=None):
        """Send `req` (a urllib.request.Request built against a `unix://`
        URL) and return a `_BufferedResponse`, or raise `urllib.error.
        HTTPError` / `URLError` — the same exceptions RemoteEngine already
        handles for the http/https transports."""
        with self._lock:
            headers = dict(req.header_items())
            selector = req.selector
            method = req.get_method()
            body = req.data

            last_exc = None
            status = reason = resp_headers = will_close = data = None
            for attempt in range(2):
                if self._conn is None:
                    self._conn = _UnixHTTPConnection(self._sock_path, timeout=timeout)
                conn = self._conn
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                try:
                    conn.request(method, selector, body=body, headers=headers)
                    resp = conn.getresponse()
                    data = resp.read()
                    status, reason = resp.status, resp.reason
                    resp_headers, will_close = resp.headers, resp.will_close
                    break
                except (http.client.BadStatusLine, http.client.RemoteDisconnected,
                        ConnectionError, OSError) as e:
                    self._drop_connection()
                    last_exc = e
            else:
                raise urllib.error.URLError(last_exc)

            if will_close:
                self._drop_connection()

        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, reason, resp_headers,
                                         io.BytesIO(data))
        return _BufferedResponse(status, data)
