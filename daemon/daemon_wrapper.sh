#!/bin/bash
# Wrapper script to activate conda environment before running Whisper STT daemon

# Activate conda environment
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/bin/activate" whisper

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the daemon with the conda environment's Python
exec python "$SCRIPT_DIR/whisper_daemon.py" --idle-timeout 0 "$@"
