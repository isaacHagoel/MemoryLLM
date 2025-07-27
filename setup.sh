#!/bin/bash

# Setup script for MemoryLLM on RunPod.io
# This script should be run from /workspace/MemoryLLM/
set -e  # Exit on any error

echo "Starting MemoryLLM setup..."

# Define paths as variables
WORKSPACE_DIR="/workspace"
CACHE_DIR="${WORKSPACE_DIR}/.cache"
PIP_CACHE_DIR="${CACHE_DIR}/pip"
HF_CACHE_DIR="${CACHE_DIR}/huggingface"
PROJECT_DIR="${WORKSPACE_DIR}/MemoryLLM"
VENV_DIR="${PROJECT_DIR}/venv"
ENV_SETUP_SCRIPT="${WORKSPACE_DIR}/setup_env.sh"

# Ensure we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: requirements.txt not found. Please run this script from ${PROJECT_DIR}/"
    exit 1
fi

# Install system packages using separate script
echo "Installing system packages..."
chmod +x install_system_packages.sh
./install_system_packages.sh

# Set up persistent directories
echo "Setting up persistent directories..."
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

# Set up environment variables for persistent storage
export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

# Upgrade pip with persistent cache
echo "Upgrading pip..."
pip install --upgrade pip --cache-dir "${PIP_CACHE_DIR}"

# Create virtual environment in the persistent workspace
echo "Setting up virtual environment in ${VENV_DIR}..."
python -m venv venv
source venv/bin/activate

# Install Python dependencies with persistent cache
echo "Installing Python dependencies..."
pip install -r requirements.txt --cache-dir "${PIP_CACHE_DIR}"

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --cache-dir "${PIP_CACHE_DIR}"

# Install wheel (required for flash-attn)
echo "Installing wheel (required for flash-attn)..."
pip install wheel --cache-dir "${PIP_CACHE_DIR}"

# Install flash attention
echo "Installing flash attention..."
pip install flash-attn==2.5.8 --no-build-isolation --cache-dir "${PIP_CACHE_DIR}"

# Install transformers
echo "Installing transformers..."
pip install transformers==4.46.0 --cache-dir "${PIP_CACHE_DIR}"

# Install Flask (force reinstall to avoid dependency conflicts)
echo "Installing Flask..."
pip install flask --force-reinstall --no-deps --cache-dir "${PIP_CACHE_DIR}"

# Create a persistent environment setup script
echo "Creating persistent environment setup script..."
cat > "${ENV_SETUP_SCRIPT}" << EOF
#!/bin/bash
# Source this script to set up the environment after container restart
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
EOF

chmod +x "${ENV_SETUP_SCRIPT}"

echo "Setup completed successfully!"
echo ""
echo "IMPORTANT NOTES:"
echo "1. All Python packages and caches are stored in ${WORKSPACE_DIR} (persistent)"
echo "2. System packages (vim, less) will be lost on container restart"
echo "3. After container restart:"
echo "   a) Run: source ${ENV_SETUP_SCRIPT}"
echo "   b) Run: ./install_system_packages.sh (to reinstall vim, less, etc.)"
echo "4. To activate the virtual environment: source ${VENV_DIR}/bin/activate" 