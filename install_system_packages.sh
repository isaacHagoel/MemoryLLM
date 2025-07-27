#!/bin/bash

# Script to install system packages that get lost on container restart
# Run this during initial setup and after each container restart

echo "Installing system packages..."

# Update package lists
apt update

# Install essential tools
echo "Installing vim..."
apt install vim -y

echo "Installing less..."
apt install less -y

echo "Installing curl (for API testing)..."
apt install curl -y

echo "Installing jq (for JSON processing)..."
apt install jq -y

echo "System packages installed successfully!"
echo "Installed: vim, less, curl, jq" 