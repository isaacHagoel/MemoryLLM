#!/bin/bash
# Template for the environment setup script
# The actual script will be created at /workspace/setup_env.sh during setup
# Source this script to set up the environment after container restart

# Define paths as variables
WORKSPACE_DIR="/workspace"
CACHE_DIR="${WORKSPACE_DIR}/.cache"
PIP_CACHE_DIR="${CACHE_DIR}/pip"
HF_CACHE_DIR="${CACHE_DIR}/huggingface"
PROJECT_DIR="${WORKSPACE_DIR}/MemoryLLM"
VENV_DIR="${PROJECT_DIR}/venv"

export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

# Activate virtual environment if it exists
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at ${VENV_DIR}/"
fi

echo "Environment setup complete!"
echo "Note: Run './install_system_packages.sh' if vim/less are missing" 