#!/bin/bash
# Round-trip test: TTS -> WAV file -> relaySTT -> verify text matches
#
# Usage:
#   ./test_roundtrip.sh                               # default test phrase
#   ./test_roundtrip.sh "Custom test phrase"           # custom phrase
set -euo pipefail

KOKORO_PORT="${KOKORO_PORT:-9997}"
KOKORO_HOST="${KOKORO_HOST:-localhost}"
WHISPER_PORT="${WHISPER_PORT:-9998}"
WHISPER_HOST="${WHISPER_HOST:-localhost}"

KNOWN_TEXT="${1:-The quick brown fox jumps over the lazy dog}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" relaystt 2>/dev/null || true

python3 -c "
import socket, struct, json, base64, sys, tempfile, os

KOKORO_HOST = '$KOKORO_HOST'
KOKORO_PORT = $KOKORO_PORT
WHISPER_HOST = '$WHISPER_HOST'
WHISPER_PORT = $WHISPER_PORT
KNOWN_TEXT = '''$KNOWN_TEXT'''

def tcp_request(host, port, request_dict):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect((host, port))
    payload = json.dumps(request_dict).encode('utf-8')
    sock.sendall(struct.pack('!I', len(payload)) + payload)
    header = b''
    while len(header) < 4:
        header += sock.recv(4 - len(header))
    resp_len = struct.unpack('!I', header)[0]
    data = b''
    while len(data) < resp_len:
        data += sock.recv(min(resp_len - len(data), 65536))
    sock.close()
    return json.loads(data.decode('utf-8'))

# Step 1: Generate audio via Kokoro
print(f'[1/3] Generating audio via Kokoro for: \"{KNOWN_TEXT}\"')
try:
    tts_response = tcp_request(KOKORO_HOST, KOKORO_PORT, {
        'text': KNOWN_TEXT,
        'voice': 'af_heart',
        'speed': 1.0,
    })
except ConnectionRefusedError:
    print(f'ERROR: Cannot connect to Kokoro daemon at {KOKORO_HOST}:{KOKORO_PORT}')
    sys.exit(1)

if not tts_response.get('success'):
    print(f'ERROR: Kokoro TTS failed: {tts_response.get(\"error\")}')
    sys.exit(1)

audio_b64 = tts_response['audio_base64']
duration = tts_response.get('duration', '?')
print(f'   Generated {duration}s audio')

# Step 2: Transcribe via Whisper
print(f'[2/3] Transcribing via Whisper...')
try:
    stt_response = tcp_request(WHISPER_HOST, WHISPER_PORT, {
        'audio_base64': audio_b64,
        'language': 'en',
    })
except ConnectionRefusedError:
    print(f'ERROR: Cannot connect to Whisper daemon at {WHISPER_HOST}:{WHISPER_PORT}')
    sys.exit(1)

if not stt_response.get('success'):
    print(f'ERROR: Whisper STT failed: {stt_response.get(\"error\")}')
    sys.exit(1)

transcribed = stt_response.get('text', '').strip()
t_time = stt_response.get('transcription_time', '?')
print(f'   Transcribed in {t_time}s')

# Step 3: Compare
print(f'[3/3] Comparing results...')
print(f'   Expected: {KNOWN_TEXT}')
print(f'   Got:      {transcribed}')

# Fuzzy comparison: lowercase, strip punctuation
import re
def normalize(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

expected_norm = normalize(KNOWN_TEXT)
got_norm = normalize(transcribed)

if expected_norm == got_norm:
    print('   PASS: Exact match!')
elif expected_norm in got_norm or got_norm in expected_norm:
    print('   PASS: Substring match (close enough)')
else:
    # Word-level comparison
    expected_words = set(expected_norm.split())
    got_words = set(got_norm.split())
    overlap = expected_words & got_words
    total = expected_words | got_words
    similarity = len(overlap) / len(total) if total else 0
    if similarity >= 0.7:
        print(f'   PASS: {similarity:.0%} word overlap')
    else:
        print(f'   FAIL: Only {similarity:.0%} word overlap')
        sys.exit(1)
"
