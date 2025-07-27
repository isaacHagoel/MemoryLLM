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

# Update system packages (note: these will be lost on container restart)
echo "Updating system packages..."
apt update && apt upgrade -y

# Install essential build tools (temporary, for downloading static binaries)
echo "Installing temporary build tools..."
apt install wget curl -y

# Set up persistent directories
echo "Setting up persistent directories..."
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/pip
mkdir -p /workspace/bin

# Install persistent vim (static binary)
echo "Installing persistent vim..."
if [[ ! -f "/workspace/bin/vim" ]]; then
    cd /workspace/bin
    # Download static vim binary (AppImage)
    wget -O vim.appimage https://github.com/vim/vim-appimage/releases/latest/download/Vim-x86_64.AppImage
    chmod +x vim.appimage
    ln -sf vim.appimage vim
    cd - > /dev/null
    echo "Vim installed to /workspace/bin/vim"
else
    echo "Vim already installed in /workspace/bin/"
fi

# Install persistent less (static binary)
echo "Installing persistent less..."
if [[ ! -f "/workspace/bin/less" ]]; then
    cd /workspace/bin
    # Download and compile less from source to get a static version
    wget http://www.greenwoodsoftware.com/less/less-590.tar.gz
    tar -xzf less-590.tar.gz
    cd less-590
    ./configure --prefix=/workspace
    make
    make install
    cd ..
    rm -rf less-590 less-590.tar.gz
    cd - > /dev/null
    echo "Less installed to /workspace/bin/less"
else
    echo "Less already installed in /workspace/bin/"
fi

# Set up environment variables for persistent storage
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip
export PATH="/workspace/bin:$PATH"

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
EOF

chmod +x /workspace/setup_env.sh

echo "Setup completed successfully!"
echo ""
echo "IMPORTANT NOTES:"
echo "1. All Python packages and caches are stored in /workspace (persistent)"
echo "2. Vim and less are now installed persistently in /workspace/bin/"
echo "3. After container restart, run: source /workspace/setup_env.sh"
echo "4. To activate the virtual environment: source /workspace/MemoryLLM/venv/bin/activate"
echo "5. Persistent tools available: vim, less" 