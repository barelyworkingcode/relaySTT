#!/usr/bin/env python3
"""
relaySTT Daemon Server
A thin protocol shell over a remote, OpenAI-compatible transcription server.
Accepts base64-encoded audio and returns transcribed text in JSON responses.
"""
import argparse
import base64
import binascii
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import warnings

import pinned_transport

warnings.filterwarnings("ignore")

# Per-request and per-connection ceilings. Without these a single client can
# pin a thread or a temp file indefinitely, or feed the model something that
# takes minutes to transcribe.
MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_AUDIO_SECONDS = 600.0
CLIENT_TIMEOUT_S = 60.0
LISTEN_BACKLOG = 64
FFMPEG_TIMEOUT_S = 30.0


class ClientDisconnected(Exception):
    """The peer closed before sending any part of a request.

    That is what a bare TCP health check looks like — connect, observe the port
    is open, hang up. It is not an error, and logging it as one buries the
    truncated-request case that is.
    """


class RemoteEngine:
    """Transcription over HTTPS (or loopback-only HTTP) against an
    OpenAI-compatible /v1/audio/transcriptions server.

    The daemon loads no model of its own — no weights, no local inference —
    so it can run on a machine too small to hold them (a VM, a spare box)
    while a host with a GPU does the work. It keeps the payload guards, the
    ffmpeg conversion to 16 kHz mono, the duration probe, and the
    byte-identical TCP protocol on 9998; only the model call is remote.

    If a router fronts the inference server, point base_url at the router: it
    can hold the upstream credential, so no secret has to live beside the
    daemon. api_key_env names the variable to read when the endpoint does
    authenticate — the token itself is never a command-line argument, where it
    would be visible in the process list.
    """

    def __init__(self, base_url=None, model=None, api_key_env=None, timeout=120.0,
                 ca_file=None, pins=None):
        self.base_url = (os.environ.get("RELAYSTT_REMOTE_URL") or base_url or "").rstrip("/")
        self.model = os.environ.get("RELAYSTT_REMOTE_MODEL") or model or ""
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env or "RELAYSTT_REMOTE_API_KEY") or None
        self.ca_file = os.environ.get("RELAYSTT_REMOTE_CA") or ca_file or None
        pin_source = os.environ.get("RELAYSTT_REMOTE_PIN_SHA256") or pins

        if not self.base_url:
            raise ValueError(
                "remote STT needs an endpoint: pass --remote-url or set "
                "RELAYSTT_REMOTE_URL to an OpenAI-compatible base URL")
        if not self.model:
            raise ValueError(
                "remote STT needs a model id: pass --remote-model or set "
                "RELAYSTT_REMOTE_MODEL to the id the remote server exposes")

        self.pins = pinned_transport.parse_pins(pin_source)
        pinned_transport.assert_transport_config(self.base_url, self.ca_file, self.pins)
        self._opener = pinned_transport.build_opener(self.base_url, self.ca_file, self.pins)

    @property
    def url(self):
        return f"{self.base_url}/audio/transcriptions"

    @property
    def label(self):
        """Endpoint identity for logs and errors — never includes the token."""
        return self.base_url or "<unset>"

    @staticmethod
    def _error_detail(body):
        """Pull a human message out of an error body, whatever shape it is."""
        text = (body or b"")[:400].decode("utf-8", "replace").strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            return text
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if err:
                return str(err)
            if parsed.get("detail"):
                return str(parsed["detail"])
        return text

    @staticmethod
    def _encode_multipart(fields, filename, audio):
        """Build a multipart/form-data body. Written out by hand because the
        standard library has no client-side encoder and the alternative is
        taking a dependency purely to format a few headers."""
        boundary = uuid.uuid4().hex
        out = bytearray()
        for name, value in fields.items():
            if value is None:
                continue
            out += f"--{boundary}\r\n".encode()
            out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            out += f"{value}\r\n".encode()
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n').encode()
        out += b"Content-Type: audio/wav\r\n\r\n"
        out += audio
        out += f"\r\n--{boundary}--\r\n".encode()
        return bytes(out), f"multipart/form-data; boundary={boundary}"

    def transcribe(self, wav_path, language=None):
        """Send one clip and return {text, language}. Raises RuntimeError with
        the endpoint and reason on any failure, so the caller can answer the
        client with something actionable instead of a stack trace."""
        with open(wav_path, "rb") as f:
            audio = f.read()

        fields = {"model": self.model}
        if language:
            fields["language"] = language
        body, content_type = self._encode_multipart(
            fields, os.path.basename(wav_path), audio)

        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": content_type})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"remote STT {self.label} returned HTTP {e.code}: "
                f"{self._error_detail(e.read())}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"remote STT {self.label} unreachable: {e.reason}") from None

        try:
            result = json.loads(payload)
        except ValueError:
            raise RuntimeError(
                f"remote STT {self.label} returned {len(payload)} bytes that "
                f"are not JSON") from None
        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError(
                f"remote STT {self.label} response has no text field: "
                f"{str(result)[:200]}")
        return result


class RelaySTTDaemon:
    def __init__(self, host="localhost", port=9998, idle_timeout=900, engine=None):
        self.host = host
        self.port = port
        self.engine = engine or RemoteEngine()
        self.running = False
        self.sock = None
        self.idle_timeout = idle_timeout
        self.last_activity = None
        self.activity_lock = threading.Lock()

    def update_activity(self):
        with self.activity_lock:
            self.last_activity = time.time()

    def check_idle_timeout(self):
        if self.idle_timeout <= 0:
            return False
        with self.activity_lock:
            if self.last_activity is None:
                return False
            return (time.time() - self.last_activity) > self.idle_timeout

    def idle_monitor(self):
        if self.idle_timeout <= 0:
            return
        while self.running:
            if self.check_idle_timeout():
                print(f"Daemon idle for {self.idle_timeout // 60} minutes, shutting down...")
                self.running = False
                break
            time.sleep(30)

    def _empty_result(self, language):
        return {"text": "", "language": language or "unknown", "duration": 0, "transcription_time": 0}

    def transcribe(self, audio_bytes, language=None):
        """Transcribe audio bytes and return text + metadata."""
        # Guard the remote call from obviously-invalid payloads. A
        # malformed/empty buffer (e.g. a truncated or zero-length capture) is
        # not worth a network round trip, so bail with an empty result instead.
        # A valid WAV header alone is 44 bytes.
        if not audio_bytes or len(audio_bytes) < 44:
            print(f"Rejecting tiny audio payload ({len(audio_bytes) if audio_bytes else 0} bytes)")
            return self._empty_result(language)

        # Write audio to a temp file (the remote engine sends a file path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            # If the audio is not WAV, convert with ffmpeg
            tmp.write(audio_bytes)
            tmp.close()

            # Probe format and convert to 16kHz mono WAV if needed
            wav_path = self._ensure_wav(tmp.name)

            # Confirm the (possibly converted) file is decodable audio before
            # sending it to the remote engine — an unreadable file would
            # otherwise ship as a useless upload and waste the round trip.
            try:
                import soundfile as sf
                duration = sf.info(wav_path).duration
            except Exception as e:
                print(f"Unreadable audio, skipping transcription: {e}")
                return self._empty_result(language)
            if duration <= 0:
                return self._empty_result(language)
            if duration > MAX_AUDIO_SECONDS:
                raise ValueError(f"audio too long ({duration:.0f}s > {MAX_AUDIO_SECONDS:.0f}s)")

            t0 = time.time()
            result = self.engine.transcribe(wav_path, language)
            transcription_time = time.time() - t0

            text = result.get("text", "").strip()
            detected_lang = result.get("language", language or "unknown")

            print(f"Transcribed: {duration:.1f}s audio in {transcription_time:.2f}s lang={detected_lang} chars={len(text)}")

            return {
                "text": text,
                "language": detected_lang,
                "duration": round(duration, 3),
                "transcription_time": round(transcription_time, 3),
            }
        finally:
            os.unlink(tmp.name)
            # Clean up converted file if different
            if 'wav_path' in locals() and wav_path != tmp.name and os.path.exists(wav_path):
                os.unlink(wav_path)

    def _ensure_wav(self, input_path):
        """Convert input audio to 16kHz mono WAV if needed. Returns path to WAV file."""
        # Check if it's already a valid WAV
        try:
            import soundfile as sf
            info = sf.info(input_path)
            if info.samplerate == 16000 and info.channels == 1:
                return input_path
        except Exception:
            pass

        # Convert with ffmpeg
        wav_path = input_path + ".converted.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                 "-protocol_whitelist", "file", "-i", input_path,
                 # Bounds the output on disk; a tiny highly-compressed input
                 # cannot expand past this before the duration check rejects it.
                 "-t", str(MAX_AUDIO_SECONDS + 1),
                 "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, check=True, timeout=FFMPEG_TIMEOUT_S,
            )
            return wav_path
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"")[:200].decode("utf-8", "replace")
            print(f"ffmpeg conversion failed: {e}: {stderr}, trying raw input...")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"ffmpeg conversion failed: {e}, trying raw input...")
        # A partial file here would otherwise sit in TMPDIR forever: the caller
        # is about to fall back to input_path, so nothing else ever cleans it up.
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        return input_path

    # -- TCP protocol (identical to Kokoro daemon) --

    def _recv_all(self, sock, length, *, allow_empty=False):
        """Read exactly `length` bytes. With allow_empty, a close before the
        first byte raises ClientDisconnected rather than ConnectionError — the
        caller passes it only for the header read, so a close *after* some bytes
        arrived is still the truncation error it has always been."""
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(length - received, 65536))
            if not chunk:
                if allow_empty and received == 0:
                    raise ClientDisconnected()
                raise ConnectionError("Connection closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_request(self, sock):
        """Receive a length-prefixed JSON request (4-byte big-endian header)."""
        header = self._recv_all(sock, 4, allow_empty=True)
        payload_len = struct.unpack("!I", header)[0]

        if header[0] in (0x7B, 0x5B):
            raise ValueError(
                "raw JSON framing is not supported; send a 4-byte big-endian length prefix")

        if payload_len > MAX_FRAME_BYTES:
            raise ValueError(f"Payload too large: {payload_len}")
        payload = self._recv_all(sock, payload_len)
        return json.loads(payload.decode("utf-8"))

    def _send_response(self, sock, response_dict):
        """Send a length-prefixed JSON response."""
        data = json.dumps(response_dict).encode("utf-8")
        sock.sendall(struct.pack("!I", len(data)) + data)

    # -- Client handling --

    def handle_client(self, client_socket, addr):
        try:
            self.update_activity()
            request = self._recv_request(client_socket)

            # Health check
            if request.get("action") == "ping":
                self._send_response(client_socket, {"success": True, "status": "ready"})
                return

            # Transcription request
            audio_b64 = request.get("audio_base64", "")
            if not isinstance(audio_b64, str) or not audio_b64:
                self._send_response(client_socket, {"success": False, "error": "No audio_base64 provided"})
                return

            language = request.get("language")
            if language is not None and (not isinstance(language, str) or len(language) > 16):
                self._send_response(client_socket, {"success": False, "error": "language must be a short string"})
                return

            try:
                audio_bytes = base64.b64decode(audio_b64, validate=True)
            except binascii.Error:
                self._send_response(client_socket, {"success": False, "error": "audio_base64 is not valid base64"})
                return

            result = self.transcribe(audio_bytes, language)

            self._send_response(client_socket, {
                "success": True,
                **result,
            })

        except ClientDisconnected:
            # A port probe, or a client that hung up before asking anything.
            # Nothing to answer and nothing worth saying.
            return
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            try:
                self._send_response(client_socket, {"success": False, "error": str(e)})
            except:
                pass
        finally:
            client_socket.close()

    # -- Server lifecycle --

    def start(self):
        # Nothing to load: no local model, no weights. A bad endpoint surfaces
        # per-request rather than blocking startup, so the daemon still comes
        # up if the remote host is booting behind it.
        print(f"Remote engine: {self.engine.label} "
              f"(model={self.engine.model}) — no local model is loaded")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
        self.sock.bind((self.host, self.port))
        self.sock.listen(LISTEN_BACKLOG)
        self.running = True

        print(f"relaySTT Daemon started on {self.host}:{self.port}")
        if self.idle_timeout > 0:
            print(f"Auto-shutdown after {self.idle_timeout // 60} minutes idle")
        else:
            print("Idle timeout disabled")

        self.update_activity()

        idle_thread = threading.Thread(target=self.idle_monitor, daemon=True)
        idle_thread.start()

        try:
            while self.running:
                try:
                    client_sock, addr = self.sock.accept()
                    # CPython forces an accepted socket back to blocking mode even
                    # though the listener has a timeout — without this, a peer that
                    # stalls mid-request holds its handler thread forever.
                    client_sock.settimeout(CLIENT_TIMEOUT_S)
                    t = threading.Thread(target=self.handle_client, args=(client_sock, addr), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except socket.error:
                    if self.running:
                        print("Socket error")
                    break
        except KeyboardInterrupt:
            print("\nStopping daemon...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("Daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="relaySTT Daemon")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--remote-url", default=None,
                        help="OpenAI-compatible base URL, e.g. http://host:8080/v1 "
                             "(or RELAYSTT_REMOTE_URL). Required")
    parser.add_argument("--remote-model", default=None,
                        help="Model id the REMOTE server exposes (or RELAYSTT_REMOTE_MODEL). "
                             "A router may prefix or alias its upstreams. Required")
    parser.add_argument("--remote-api-key-env", default=None,
                        help="Name of the env var holding a bearer token for the remote "
                             "endpoint (default RELAYSTT_REMOTE_API_KEY). The token is "
                             "never passed on the command line")
    parser.add_argument("--remote-ca", default=None,
                        help="PEM CA bundle to trust for the remote endpoint (or "
                             "RELAYSTT_REMOTE_CA). When set it is the ONLY trust anchor "
                             "consulted; system roots are not")
    parser.add_argument("--remote-pin-sha256", default=None,
                        help="Comma-separated SHA-256 fingerprints of the remote "
                             "endpoint's leaf certificate (or RELAYSTT_REMOTE_PIN_SHA256). "
                             "The connection fails unless the presented certificate matches")
    parser.add_argument("--idle-timeout", type=int, default=900,
                        help="Auto-shutdown after idle seconds (0 = disabled)")
    args = parser.parse_args()

    engine = RemoteEngine(base_url=args.remote_url, model=args.remote_model,
                          api_key_env=args.remote_api_key_env,
                          ca_file=args.remote_ca, pins=args.remote_pin_sha256)
    daemon = RelaySTTDaemon(host=args.host, port=args.port, idle_timeout=args.idle_timeout,
                           engine=engine)
    daemon.start()


if __name__ == "__main__":
    main()
