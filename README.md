# relaySTT

A speech-to-text daemon. Send base64-encoded audio over TCP, get transcribed text back as JSON.

The daemon loads no model itself. It is a thin protocol shell that calls an OpenAI-compatible `/v1/audio/transcriptions` server — see [Engine](#engine).

The protocol sibling of [relayTTS](https://github.com/barelyworkingcode/relayTTS): same length-prefixed JSON framing, adjacent port, same wrapper shape.

MIT licensed.

## Built on

This daemon is a thin protocol shell. The hard part — speech recognition — belongs to whatever server it points at:

| | |
|---|---|
| [Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) — Alibaba (Apache-2.0) | what the remote engine points at in practice |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Prince Canuma (MIT) | STT/TTS model support that remote-side servers build on |
| [mlx-community](https://huggingface.co/mlx-community) | the MLX conversions of the model above |
| [soundfile](https://github.com/bastibe/python-soundfile) (BSD-3) · [NumPy](https://numpy.org) (BSD-3) · [FFmpeg](https://ffmpeg.org) | decoding, arrays, and format conversion |

What this repo adds is the daemon: a length-prefixed TCP protocol, a warm process, crash supervision, and a fail-closed TLS transport to wherever inference happens.

## Requirements

- macOS with Apple Silicon
- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Miniforge)
- [Homebrew](https://brew.sh/) (for ffmpeg)

## Setup

```bash
RELAYSTT_REMOTE_URL=http://<router>:<port>/v1 RELAYSTT_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
```

This will:
1. Create a `relaystt` conda environment with Python 3.11 and dependencies (`soundfile`, `numpy`)
2. Install `ffmpeg` via Homebrew if not present
3. Register the daemon with [Relay](https://relay.dev) for auto-start (if Relay is installed)

`RELAYSTT_REMOTE_MODEL` is required the first time a machine registers the service — the daemon loads no model of its own, so without one it would come up and answer `ping` while failing every transcription. `RELAYSTT_REMOTE_URL` is required too *unless* Relay is installed, in which case `build.sh` defaults it to Relay's own model socket (see "Remote via relay's model socket" below).

## Usage

### Start the daemon

```bash
RELAYSTT_REMOTE_URL=http://<router>:<port>/v1 RELAYSTT_REMOTE_MODEL=<id-that-server-exposes> \
  ./daemon/daemon_wrapper.sh
```

The daemon listens on `localhost:9998` by default. Options:

```bash
./daemon/daemon_wrapper.sh --port 9998 --remote-url http://<router>:<port>/v1 \
  --remote-model <id-that-server-exposes> --idle-timeout 900
```

The daemon auto-shuts down after 15 minutes of inactivity by default (set `--idle-timeout 0` to disable).

### Transcribe audio

```bash
./test_transcribe.sh audio.wav          # transcribe a file
./test_transcribe.sh audio.wav en       # with language hint
./test_transcribe.sh --ping             # health check
```

### TCP protocol

Clients connect over TCP and exchange length-prefixed JSON messages (4-byte big-endian header); the header must carry a byte count, not a JSON body — legacy raw-JSON framing is no longer accepted. A request is capped at 32 MB and audio at 600 seconds (a `duration`/size over either limit comes back as an error response, not a stack trace), and an idle connection is dropped after 60 seconds.

**Request:**
```json
{
  "audio_base64": "<base64-encoded audio>",
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "text": "transcribed text",
  "language": "en",
  "duration": 3.5,
  "transcription_time": 0.42
}
```

**Health check:**
```json
{"action": "ping"}
```

Audio is automatically converted to 16kHz mono WAV via ffmpeg if needed.

## Engine

The daemon calls an OpenAI-compatible `POST {base_url}/audio/transcriptions` server; it never loads model weights itself. Measured footprint: **28 MB** resident.

This is for running the daemon on a machine too small to hold the weights — a VM, a spare box — while something with a GPU does the work. Nothing else changes: the payload guards, the ffmpeg conversion to 16 kHz mono, the duration probe and the TCP protocol on 9998 are all identical, so clients cannot tell what is behind the daemon.

```bash
python daemon/relaystt_daemon.py \
  --remote-url http://<router>:<port>/v1 \
  --remote-model <id-that-server-exposes>
```

Notes:

- Both `--remote-url` and `--remote-model` (or `RELAYSTT_REMOTE_URL` / `RELAYSTT_REMOTE_MODEL`) are required; the daemon refuses to start without them.
- `--remote-model` is the id the **remote** server exposes. A router may prefix or alias its upstreams.
- If a router fronts the inference server, point at the router: it can hold the upstream credential, so no token has to live beside the daemon. `--remote-api-key-env` names the variable to read if the endpoint does authenticate; the token is never a command-line argument, where it would show up in the process list.
- A bad endpoint surfaces per request, not at startup, so the daemon still comes up and answers `ping` while a remote host is booting.

## Transport security

There is no skip-verify or insecure flag, and none may be added — a connection that can't be verified fails the request rather than degrading to plaintext-grade trust.

- `--remote-url` / `RELAYSTT_REMOTE_URL` must be `http` or `https`. `http` is allowed only when the host is loopback (`127.0.0.1`, `::1`, `localhost`); any other host over `http` is refused — use `https` and pin its CA instead.
- `--remote-ca` / `RELAYSTT_REMOTE_CA` names a PEM bundle. When set it is the **only** trust anchor consulted — system roots are not. Relay's local CA at `~/Library/Application Support/Relay/ca.crt` is the natural value for relay-issued certificates.
- `--remote-pin-sha256` / `RELAYSTT_REMOTE_PIN_SHA256` is a comma-separated list of SHA-256 fingerprints of the endpoint's leaf certificate (hex, colons optional, case-insensitive). When set, the connection fails unless the presented leaf matches one of them — this catches a certificate that is otherwise validly signed by a trusted CA.
- `--remote-ca` / `--remote-pin-sha256` apply to `https` only; setting either alongside an `http` URL is a startup error.
- Environment variables win over the equivalent CLI flag, same as `RELAYSTT_REMOTE_URL` / `RELAYSTT_REMOTE_MODEL`.

Get a certificate's fingerprint with:

```bash
openssl s_client -connect host:port </dev/null 2>/dev/null | openssl x509 -fingerprint -sha256 -noout
```

### Remote via relay's model socket

`RELAYSTT_REMOTE_URL` also accepts `unix:<absolute path>` — HTTP over
`AF_UNIX` against relay's own model endpoint instead of a
directly-configured server. `build.sh` defaults to this
(`unix:$HOME/Library/Application Support/relay/model.sock`) whenever Relay
is installed and no `RELAYSTT_REMOTE_URL` is given.

- The base path is always `/v1` — the socket path names the socket file, not
  a URL prefix.
- **No `Authorization` or `x-api-key` header is ever sent on this path.**
  Relay identifies the daemon by its launch identity (the kernel's audit
  token on the connection), not by a header — a header would be judged as a
  bearer credential instead and can only make the call worse. A configured
  `RELAYSTT_REMOTE_API_KEY` (or whatever `--remote-api-key-env` names) is
  ignored, with a one-line startup warning naming the variable, never its
  value.
- `RELAYSTT_REMOTE_CA` / `RELAYSTT_REMOTE_PIN_SHA256` don't apply here —
  there is no TLS layer on `AF_UNIX` — and `assert_transport_config` refuses
  startup if either is set alongside a `unix:` URL, the same "would be a
  false sense of safety" rule plain `http` already gets.
- The path after `unix:` must be absolute; a relative one is refused at
  startup with a clear message.
- One connection is opened per daemon process and reused (HTTP keep-alive)
  across requests rather than one dialed and torn down per transcription.
- The `model` field still travels as a multipart part (same as any other
  transport here) — relay's model endpoint extracts `model` from either a
  JSON key or a multipart part to authorize the call.
- Relay's model endpoint returns 401 if this service doesn't hold the
  `models` capability, and the same 404 for a model that's unknown as for
  one that isn't in `--allowed-model`; 429 (admission timeout) and 503 (no
  model host registered) are retried with backoff.
- **Registering the `models` capability is presence-gated** — `build.sh`'s
  `relay service register` call raises a real macOS confirmation dialog and
  must be run at the console, not over SSH. If the service is already
  registered, `build.sh` prints the exact `unregister`-then-`register`
  command needed to pick up the capability change; there is no in-place
  update. `build.sh` only requests `models` when `RELAYSTT_REMOTE_URL` is
  actually `unix:*` — a plain `https://` remote never calls relay's model
  socket, so granting the capability there would sit on the service record
  unused.

## Relay launch identity

When [Relay](https://relay.dev) launches the daemon, `daemon/launch_identity.py` drains the one-shot secret Relay hands over on fd 3 and presents it in a `Hello` on `RELAY_BRIDGE_SOCKET`, before anything else runs — no argument parsing, no config loading, no thread, no subprocess. Relay binds this process's kernel audit token as the daemon's identity; no relay credential is ever held in the environment. A failed handshake exits 78 without logging the secret, and `daemon/daemon_wrapper.sh` treats exit 78 as final (the launch secret is single-use, so a respawn could never succeed) rather than restarting. Running the daemon standalone (`RELAY_LAUNCH_FD` unset) skips the handshake entirely.

Full protocol: `relay/docs/launch-identity.md`.

## Round-trip test

Test the full TTS-to-STT pipeline with a [Kokoro TTS](https://github.com/barelyworkingcode/kokoro) daemon:

```bash
./test_roundtrip.sh                          # default phrase
./test_roundtrip.sh "Custom test phrase"     # custom phrase
```

Requires a Kokoro daemon running on port 9997.
