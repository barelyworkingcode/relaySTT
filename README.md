# Whisper STT Daemon

A local speech-to-text daemon that keeps a Whisper model loaded in memory with Apple Silicon MLX acceleration. Send base64-encoded audio over TCP and get back transcribed text as JSON.

Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast inference on Apple Silicon. Default model: `mlx-community/whisper-large-v3-turbo`.

## Requirements

- macOS with Apple Silicon
- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Miniforge)
- [Homebrew](https://brew.sh/) (for ffmpeg)

## Setup

```bash
./build.sh
```

This will:
1. Create a `whisper` conda environment with Python 3.11 and dependencies (`mlx-whisper`, `soundfile`, `numpy`)
2. Install `ffmpeg` via Homebrew if not present
3. Register the daemon with [Relay](https://relay.dev) for auto-start (if Relay is installed)

## Usage

### Start the daemon

```bash
./daemon/daemon_wrapper.sh
```

The daemon listens on `localhost:9998` by default. Options:

```bash
./daemon/daemon_wrapper.sh --port 9998 --model mlx-community/whisper-large-v3-turbo --idle-timeout 900
```

The daemon auto-shuts down after 15 minutes of inactivity by default (set `--idle-timeout 0` to disable).

### Transcribe audio

```bash
./test_transcribe.sh audio.wav          # transcribe a file
./test_transcribe.sh audio.wav en       # with language hint
./test_transcribe.sh --ping             # health check
```

### TCP protocol

Clients connect over TCP and exchange length-prefixed JSON messages (4-byte big-endian header).

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

## Round-trip test

Test the full TTS-to-STT pipeline with a [Kokoro TTS](https://github.com/barelyworkingcode/kokoro) daemon:

```bash
./test_roundtrip.sh                          # default phrase
./test_roundtrip.sh "Custom test phrase"     # custom phrase
```

Requires a Kokoro daemon running on port 9997.
