#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda environment exists
if ! conda info --envs 2>/dev/null | grep -q "^whisper "; then
    echo "Setting up conda environment..."
    "$SCRIPT_DIR/setup_env.sh"
fi

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    if "$RELAY" service list 2>/dev/null | grep -q "whisper-daemon"; then
        echo "Already registered with Relay. Daemon will use updated scripts."
    else
        "$RELAY" service register \
            --name whisper-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart
        echo "Registered whisper-daemon service with Relay"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
