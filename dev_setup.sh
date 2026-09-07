#!/bin/bash

# Development setup script for rust-gng-py project
# This script builds the library only, without container setup
# Usage: ./dev_setup.sh [dev|release]

set -e  # Exit on any error

BUILD_TYPE=${1:-"dev"}

echo "Starting development setup..."
echo "Build type: $BUILD_TYPE"

# Step 1: Build the Rust library with maturin using Nix
echo "Step 1: Building Rust library with maturin..."
nix develop --command bash -c "cd gng && \
                               cargo clean && \
                               cargo build --${BUILD_TYPE} && \
                               maturin build --${BUILD_TYPE}"

echo "Development setup complete!"
echo "Library built successfully. You can now:"
echo "  1. Use 'make up' to start the container with Jupyter support"
echo "  2. Or use 'make build_lib' to rebuild the library"
echo "  3. Or work directly in the Nix development shell with 'nix develop'"
