#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda environment exists
if ! conda info --envs 2>/dev/null | grep -q "^relaystt "; then
    echo "Setting up conda environment..."
    "$SCRIPT_DIR/setup_env.sh"
fi

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    # Remote inference is a deployment choice, not a source change: export
    # these before running this and they are baked into the service
    # registration, so the daemon comes up in remote mode and loads no model.
    #
    #   RELAYSTT_REMOTE_URL=http://<router>:<port>/v1 \
    #   RELAYSTT_REMOTE_MODEL=<id-that-server-exposes> ./build.sh
    #
    REGISTER_ENV=()
    if [ -n "${RELAYSTT_REMOTE_URL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_URL=$RELAYSTT_REMOTE_URL")
    fi
    if [ -n "${RELAYSTT_REMOTE_MODEL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_MODEL=$RELAYSTT_REMOTE_MODEL")
    fi

    if "$RELAY" service list 2>/dev/null | grep -q "relaystt-daemon"; then
        echo "Already registered with Relay. Daemon will use updated scripts."
        # `service register` is the only way to set env; there is no update
        # verb. So say so rather than silently ignoring a changed URL.
        if [ ${#REGISTER_ENV[@]} -gt 0 ]; then
            echo "NOTE: WHISPER_REMOTE_* is set but the service is already" \
                 "registered. To change it, unregister first:"
            echo "      $RELAY service unregister --name relaystt-daemon && ./build.sh"
        fi
    else
        "$RELAY" service register \
            --name relaystt-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            "${REGISTER_ENV[@]+"${REGISTER_ENV[@]}"}"
        if [ ${#REGISTER_ENV[@]} -gt 0 ]; then
            echo "Registered relaystt-daemon service with Relay (remote: $RELAYSTT_REMOTE_URL)"
        else
            echo "Registered relaystt-daemon service with Relay (local model)"
        fi
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
