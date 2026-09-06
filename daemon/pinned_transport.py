"""TLS transport for the remote STT engine.

Certificate verification is always on; there is no skip-verify / insecure
flag anywhere in this module, and none may be added — a handshake that can't
be verified must fail the request, not silently downgrade to plaintext-grade
trust.
"""
import hashlib
import http.client
import ssl
import urllib.parse
import urllib.request

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_HEX_DIGITS = set("0123456789abcdef")


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


def assert_transport_config(url, ca_file, pins):
    """Validate a remote-transport configuration before any connection is
    attempted. Raises ValueError describing what to change."""
    if not url:
        raise ValueError(
            "remote STT needs a URL: pass --remote-url or set RELAYSTT_REMOTE_URL")

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
    misconfigured engine can't change global request behavior."""
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
