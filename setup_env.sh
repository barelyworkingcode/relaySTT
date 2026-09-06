#!/bin/bash
# Setup conda environment for the relaySTT daemon
set -euo pipefail

#
# Usage:
#   ./setup_env.sh
#
# The daemon is a thin protocol shell over a remote transcription server: it
# never loads a model itself, so the env is soundfile + numpy only. HTTP is
# urllib from the standard library, so there is no client dependency either.
ENV_NAME="relaystt"
CONDA_BASE="$(conda info --base)"

for arg in "$@"; do
    case "$arg" in
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

DEPS="soundfile numpy"
echo "Setting up '${ENV_NAME}'."

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
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with: conda activate ${ENV_NAME}"
