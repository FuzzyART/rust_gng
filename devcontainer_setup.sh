#!/bin/bash

# Devcontainer setup script
# This script helps set up the development environment for use with devcontainers
# Works with both VSCode and Zed editors

set -e  # Exit on any error

echo "Setting up devcontainer environment..."

# Check if we're in a devcontainer
if [ -f "/.devcontainer" ]; then
    echo "Already in devcontainer environment"
else
    echo "This script is meant to be run inside a devcontainer"
    echo "For VSCode: Open in container from .devcontainer/devcontainer.json"
    echo "For Zed: Open project in devcontainer"
    exit 1
fi

# Install project-specific dependencies
echo "Installing project dependencies..."

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Python packages needed for development
echo "Installing Python packages..."
pip install --upgrade pip
pip install maturin jupyterlab ipykernel numpy pandas matplotlib scikit-learn umap-learn

# Create necessary directories
mkdir -p /workspace/target/wheels
mkdir -p /workspace/dist

echo "Devcontainer setup complete!"
echo "You can now:"
echo "  1. Build the project: ./build_in_devcontainer.sh"
echo "  2. Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "  3. Work in the devcontainer with VSCode or Zed"
