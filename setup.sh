#!/bin/bash

# Setup script for MemoryLLM on RunPod.io
# This script should be run from /workspace/MemoryLLM/
set -e  # Exit on any error

echo "Starting MemoryLLM setup..."

# Ensure we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: requirements.txt not found. Please run this script from /workspace/MemoryLLM/"
    exit 1
fi

# Install system packages using separate script
echo "Installing system packages..."
chmod +x install_system_packages.sh
./install_system_packages.sh

# Set up persistent directories
echo "Setting up persistent directories..."
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/pip

# Set up environment variables for persistent storage
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip

# Upgrade pip with persistent cache
echo "Upgrading pip..."
pip install --upgrade pip --cache-dir /workspace/.cache/pip

# Create virtual environment in the persistent workspace
echo "Setting up virtual environment in /workspace/MemoryLLM/venv..."
python -m venv venv
source venv/bin/activate

# Install Python dependencies with persistent cache
echo "Installing Python dependencies..."
pip install -r requirements.txt --cache-dir /workspace/.cache/pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --cache-dir /workspace/.cache/pip

# Install flash attention
echo "Installing flash attention..."
pip install flash-attn==2.5.8 --no-build-isolation --cache-dir /workspace/.cache/pip

# Install transformers
echo "Installing transformers..."
pip install transformers==4.46.0 --cache-dir /workspace/.cache/pip

# Install Flask
echo "Installing Flask..."
pip install flask --cache-dir /workspace/.cache/pip

# Create a persistent environment setup script
echo "Creating persistent environment setup script..."
cat > /workspace/setup_env.sh << 'EOF'
#!/bin/bash
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
EOF

chmod +x /workspace/setup_env.sh

echo "Setup completed successfully!"
echo ""
echo "IMPORTANT NOTES:"
echo "1. All Python packages and caches are stored in /workspace (persistent)"
echo "2. System packages (vim, less) will be lost on container restart"
echo "3. After container restart:"
echo "   a) Run: source /workspace/setup_env.sh"
echo "   b) Run: ./install_system_packages.sh (to reinstall vim, less, etc.)"
echo "4. To activate the virtual environment: source /workspace/MemoryLLM/venv/bin/activate" 