#!/bin/bash
# Setup conda environment for the relaySTT daemon
set -euo pipefail

#
# Usage:
#   ./setup_env.sh              # local engine: the daemon loads Whisper itself
#   ./setup_env.sh --remote     # remote engine: no mlx-whisper, no weights
#
# --remote omits mlx-whisper deliberately. A remote daemon never loads a model,
# and a box that *could* load one can silently fall back to doing so — which is
# the memory pressure remote mode exists to avoid. HTTP is urllib from the
# standard library, so there is no client dependency either.
#
# Both modes build the same env name, since daemon_wrapper.sh activates it by
# name; switching modes means rebuilding the env.
ENV_NAME="relaystt"
CONDA_BASE="$(conda info --base)"

REMOTE=0
for arg in "$@"; do
    case "$arg" in
        --remote) REMOTE=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

if [ "$REMOTE" -eq 1 ]; then
    DEPS="soundfile numpy"
    MODE_DESC="remote engine (no mlx-whisper)"
else
    DEPS="mlx-whisper soundfile numpy"
    MODE_DESC="local engine (mlx-whisper)"
fi
echo "Setting up '${ENV_NAME}' for the ${MODE_DESC}."

# Ensure ffmpeg is installed via brew (required for audio format conversion)
if ! brew list ffmpeg &>/dev/null; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
fi

# Check if environment already exists
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Updating packages..."
    source "${CONDA_BASE}/bin/activate" "$ENV_NAME"
    pip install --upgrade $DEPS
else
    echo "Creating conda environment '${ENV_NAME}'..."
    # --override-channels is load-bearing: without it conda still consults the
    # implicit "defaults" channels, and a modern miniconda refuses to proceed
    # until Anaconda's commercial Terms of Service are accepted —
    #   CondaToSNonInteractiveError: Terms of Service have not been accepted
    # which fails env creation outright on any fresh machine. conda-forge is
    # what this line already asked for; this makes that actual.
    conda create -n "$ENV_NAME" python=3.11 -y --override-channels -c conda-forge
    source "${CONDA_BASE}/bin/activate" "$ENV_NAME"
    pip install $DEPS
fi

echo ""
echo "Environment '${ENV_NAME}' is ready — ${MODE_DESC}."
echo "Activate with: conda activate ${ENV_NAME}"
