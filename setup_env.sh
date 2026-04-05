#!/bin/bash
# Setup conda environment for Whisper STT daemon
set -euo pipefail

ENV_NAME="whisper"
CONDA_BASE="$(conda info --base)"

# Ensure ffmpeg is installed via brew (required for audio format conversion)
if ! brew list ffmpeg &>/dev/null; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
fi

# Check if environment already exists
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Updating packages..."
    source "${CONDA_BASE}/bin/activate" "$ENV_NAME"
    pip install --upgrade mlx-whisper soundfile numpy
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "$ENV_NAME" python=3.11 -y -c conda-forge
    source "${CONDA_BASE}/bin/activate" "$ENV_NAME"
    pip install mlx-whisper soundfile numpy
fi

echo ""
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with: conda activate ${ENV_NAME}"
