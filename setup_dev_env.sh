#!/bin/bash

# Complete development environment setup script
# This script sets up everything needed for development with Jupyter and VSCode
# Usage: ./setup_dev_env.sh [dev|release]

set -e  # Exit on any error

BUILD_TYPE=${1:-"dev"}
echo "Setting up development environment with $BUILD_TYPE build..."

# Step 1: Build the Rust library with maturin using Nix
echo "Building Rust library with maturin..."
nix develop --command bash -c "cd gng && \
                               cargo clean && \
                               cargo build --${BUILD_TYPE} && \
                               maturin build --${BUILD_TYPE}"

# Step 2: Build the Docker container
echo "Building Docker container..."
./scripts/container/1-buildContainer.sh

# Step 3: Start the container
echo "Starting container..."
./scripts/container/2-startContainer.sh

# Step 4: Install the built package in the container
echo "Installing package in container..."
./scripts/container/3-install_gng.sh

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "You can now:"
echo "  1. Attach to the container with VSCode:"
echo "     docker exec -it pytorch_project_cont fish"
echo "  2. Access Jupyter at: http://localhost:8888"
echo "  3. Work on your code in the container"
echo ""
echo "To stop the container when done:"
echo "  docker stop pytorch_project_cont"
echo ""
echo "For development in Nix shell (alternative approach):"
echo "  ./nix_dev.sh shell"
echo "  ./nix_dev.sh build"
