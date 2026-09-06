#!/bin/bash
# Wrapper that activates the conda env and supervises the relaySTT daemon.

# Activate conda environment
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" relaystt

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Line-buffer the daemon's stdout/stderr so operational logs (engine mode, model
# load, errors) reach Relay's logfile promptly instead of sitting in a block
# buffer for the life of a long-running process. Mirrors relayTTS's wrapper.
export PYTHONUNBUFFERED=1

# Restart-on-crash supervision (mirrors the Kokoro TTS daemon).
#
# The daemon holds no native model libs — it is a protocol shell over a
# remote server — but ffmpeg still runs as a subprocess per request and the
# process can still die (a malformed-audio fault, OOM). If it does, we
# restart it so STT self-heals instead of staying dead until the next Relay
# launch — the daemon is registered --autostart only, with no
# restart-on-crash from Relay.
#
# Relay runs this wrapper as a process-group leader and stops the service by
# SIGTERM-ing the whole group (1s grace, then SIGKILL). So: run python in the
# background, wait on it, and trap TERM/INT to forward the signal, stop the loop,
# and exit promptly — well inside Relay's grace window. Only an unexpected exit
# (crash) triggers a respawn; a clean exit or a stop signal ends the loop.
term=0
child=""
shutdown() { term=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; }
trap shutdown TERM INT

while true; do
    python "$SCRIPT_DIR/relaystt_daemon.py" --idle-timeout 0 "$@" &
    child=$!
    wait "$child"
    code=$?
    if [ "$term" -eq 1 ] || [ "$code" -eq 0 ]; then
        break
    fi
    echo "relaystt daemon exited (code $code) — restarting in 2s" >&2
    sleep 2
done
