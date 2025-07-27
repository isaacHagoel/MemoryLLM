#!/bin/bash
# Template for the environment setup script
# The actual script will be created at /workspace/setup_env.sh during setup
# Source this script to set up the environment after container restart

export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip

# Activate virtual environment if it exists
if [ -f "/workspace/MemoryLLM/venv/bin/activate" ]; then
    source /workspace/MemoryLLM/venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at /workspace/MemoryLLM/venv/"
fi

echo "Environment setup complete!"
echo "Note: Run './install_system_packages.sh' if vim/less are missing" 