#!/bin/bash
# Template for the environment setup script
# The actual script will be created at /workspace/setup_env.sh during setup
# Source this script to set up the environment after container restart

export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip

# Add persistent tools to PATH
export PATH="/workspace/bin:$PATH"

# Activate virtual environment if it exists
if [ -f "/workspace/MemoryLLM/venv/bin/activate" ]; then
    source /workspace/MemoryLLM/venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at /workspace/MemoryLLM/venv/"
fi

# Verify persistent tools are available
if command -v vim >/dev/null 2>&1; then
    echo "Vim available at: $(which vim)"
else
    echo "Warning: Vim not found in PATH"
fi

if command -v less >/dev/null 2>&1; then
    echo "Less available at: $(which less)"
else
    echo "Warning: Less not found in PATH"
fi 