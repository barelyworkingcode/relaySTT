#!/bin/bash
# Quick test: send audio file to Whisper daemon and print transcription
#
# Usage:
#   ./test_transcribe.sh audio.wav                    # transcribe file
#   ./test_transcribe.sh audio.wav en                  # with language hint
#   ./test_transcribe.sh --ping                        # health check
set -euo pipefail

PORT="${WHISPER_PORT:-9998}"
HOST="${WHISPER_HOST:-localhost}"

if [[ "${1:-}" == "--ping" ]]; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/bin/activate" relaystt 2>/dev/null || true

    python3 -c "
import socket, struct, json, sys

host, port = '$HOST', $PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((host, port))
except ConnectionRefusedError:
    print(f'Cannot connect to Whisper daemon at {host}:{port}')
    sys.exit(1)

payload = json.dumps({'action': 'ping'}).encode('utf-8')
sock.sendall(struct.pack('!I', len(payload)) + payload)

header = b''
while len(header) < 4:
    header += sock.recv(4 - len(header))
resp_len = struct.unpack('!I', header)[0]

data = b''
while len(data) < resp_len:
    data += sock.recv(min(resp_len - len(data), 65536))
sock.close()

response = json.loads(data.decode('utf-8'))
print(json.dumps(response, indent=2))
"
    exit 0
fi

AUDIO_FILE="${1:?Usage: $0 <audio_file> [language]}"
LANGUAGE="${2:-}"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "File not found: $AUDIO_FILE"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" relaystt 2>/dev/null || true

python3 -c "
import socket, struct, json, base64, sys

host, port = '$HOST', $PORT
language = '$LANGUAGE' or None

# Read and encode audio file
with open('$AUDIO_FILE', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('ascii')

request = {'audio_base64': audio_b64}
if language:
    request['language'] = language

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((host, port))
except ConnectionRefusedError:
    print(f'Cannot connect to Whisper daemon at {host}:{port}')
    print('Start it with: ./build.sh  (or via Relay)')
    sys.exit(1)

payload = json.dumps(request).encode('utf-8')
sock.sendall(struct.pack('!I', len(payload)) + payload)

header = b''
while len(header) < 4:
    header += sock.recv(4 - len(header))
resp_len = struct.unpack('!I', header)[0]

data = b''
while len(data) < resp_len:
    data += sock.recv(min(resp_len - len(data), 65536))
sock.close()

response = json.loads(data.decode('utf-8'))

if not response.get('success'):
    print(f'Error: {response.get(\"error\", \"unknown\")}')
    sys.exit(1)

text = response.get('text', '')
lang = response.get('language', '?')
duration = response.get('duration', '?')
t_time = response.get('transcription_time', '?')
print(f'Duration: {duration}s | Transcription time: {t_time}s | Language: {lang}')
print(f'Text: {text}')
"
