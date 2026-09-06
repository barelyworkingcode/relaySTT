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
    ALREADY_REGISTERED=0
    if "$RELAY" service list 2>/dev/null | grep -q "relaystt-daemon"; then
        ALREADY_REGISTERED=1
    fi

    # The daemon loads no model of its own, so the endpoint it calls is not
    # optional: a first registration without it would come up and answer
    # ping while failing every transcription.
    if [ "$ALREADY_REGISTERED" -eq 0 ]; then
        MISSING=()
        [ -z "${RELAYSTT_REMOTE_URL:-}" ] && MISSING+=(RELAYSTT_REMOTE_URL)
        [ -z "${RELAYSTT_REMOTE_MODEL:-}" ] && MISSING+=(RELAYSTT_REMOTE_MODEL)
        if [ ${#MISSING[@]} -gt 0 ]; then
            echo "Missing required environment variable(s) for registration: ${MISSING[*]}" >&2
            echo "Set them and re-run, e.g.:" >&2
            echo "  RELAYSTT_REMOTE_URL=http://<router>:<port>/v1 \\" >&2
            echo "  RELAYSTT_REMOTE_MODEL=<id-that-server-exposes> ./build.sh" >&2
            exit 1
        fi
    fi

    REGISTER_ENV=()
    if [ -n "${RELAYSTT_REMOTE_URL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_URL=$RELAYSTT_REMOTE_URL")
    fi
    if [ -n "${RELAYSTT_REMOTE_MODEL:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_MODEL=$RELAYSTT_REMOTE_MODEL")
    fi
    if [ -n "${RELAYSTT_REMOTE_CA:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_CA=$RELAYSTT_REMOTE_CA")
    fi
    if [ -n "${RELAYSTT_REMOTE_PIN_SHA256:-}" ]; then
        REGISTER_ENV+=(--env "RELAYSTT_REMOTE_PIN_SHA256=$RELAYSTT_REMOTE_PIN_SHA256")
    fi

    if [ "$ALREADY_REGISTERED" -eq 1 ]; then
        echo "Already registered with Relay. Daemon will use updated scripts."
        # `service register` is the only way to set env; there is no update
        # verb. So say so rather than silently ignoring a changed URL.
        if [ ${#REGISTER_ENV[@]} -gt 0 ]; then
            echo "NOTE: RELAYSTT_REMOTE_* is set but the service is already" \
                 "registered. To change it, unregister first:"
            echo "      $RELAY service unregister --name relaystt-daemon && ./build.sh"
        fi
    else
        "$RELAY" service register \
            --name relaystt-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            "${REGISTER_ENV[@]}"
        echo "Registered relaystt-daemon service with Relay (remote: $RELAYSTT_REMOTE_URL)"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
