#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda environment exists
CONDA_ENVS="$(conda info --envs 2>/dev/null || true)"
if ! grep -q "^relaystt " <<< "$CONDA_ENVS"; then
    echo "Setting up conda environment..."
    "$SCRIPT_DIR/setup_env.sh"
fi

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    # Capture before grep: `service list | grep -q` under pipefail exits 141
    # (SIGPIPE) when grep exits on a match before the list finishes writing,
    # which would make an already-registered daemon look unregistered.
    SERVICE_LIST="$("$RELAY" service list 2>/dev/null || true)"
    ALREADY_REGISTERED=0
    if grep -q "relaystt-daemon" <<< "$SERVICE_LIST"; then
        ALREADY_REGISTERED=1
    fi

    # With relay present, its own model.sock is the default remote endpoint:
    # the daemon reaches its transcription model through relay's
    # launch-identity auth (no bearer header — see
    # daemon/pinned_transport.py's unix transport) instead of a
    # directly-configured HTTPS endpoint. An explicit RELAYSTT_REMOTE_URL
    # (e.g. talking to a server directly, or in dev) always wins.
    if [ -z "${RELAYSTT_REMOTE_URL:-}" ]; then
        RELAYSTT_REMOTE_URL="unix:$HOME/Library/Application Support/relay/model.sock"
    fi

    # The daemon loads no model of its own, so the endpoint it calls is not
    # optional: a first registration without it would come up and answer
    # ping while failing every transcription. RELAYSTT_REMOTE_MODEL is
    # required on both the https and unix paths — relay's model endpoint
    # 404s a request naming no model just as an upstream router would.
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
        # `service register` is the only way to set env or capabilities;
        # there is no update verb — it upserts and restates the whole
        # record, so re-running with new values here has no effect on an
        # already-registered service. Say so rather than silently ignoring a
        # changed URL, model, or the models capability / allowed-model grant
        # this build.sh now registers.
        echo "NOTE: relay service register has no update verb. To pick up a" \
             "changed RELAYSTT_REMOTE_* variable, or the models capability /" \
             "allowed-model grant this build.sh now registers, unregister" \
             "first:"
        echo "      $RELAY service unregister --name relaystt-daemon && ./build.sh"
    else
        # `models` scopes this service to exactly the model named below
        # (empty allowed_models means NO models for a service, the opposite
        # of a project's default) — STT is meant to reach its one remote
        # model, not every model relay's broker can reach. Only grant it
        # when the daemon actually talks to relay's model.sock: a plain
        # https:// remote never calls relay's model endpoint, so granting
        # `models` there would be an unused, unrevoked capability sitting on
        # the service record.
        REGISTER_CAPS=()
        case "$RELAYSTT_REMOTE_URL" in
            unix:*)
                REGISTER_CAPS+=(--capability models --allowed-model "$RELAYSTT_REMOTE_MODEL")
                ;;
        esac

        # REGISTER_CAPS is empty on the https path (no capability at all is
        # granted there) — "${arr[@]}" on a genuinely empty array is an
        # unbound-variable error under `set -u` on the bash 3.2 that ships
        # as macOS's /bin/bash, so this expands defensively instead.
        "$RELAY" service register \
            --name relaystt-daemon \
            --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" \
            --autostart \
            "${REGISTER_CAPS[@]+"${REGISTER_CAPS[@]}"}" \
            "${REGISTER_ENV[@]+"${REGISTER_ENV[@]}"}"
        echo "Registered relaystt-daemon service with Relay (remote: $RELAYSTT_REMOTE_URL)"
    fi
else
    echo "Relay not found at $RELAY, skipping registration"
fi
