#!/bin/bash

# Streamlined build script for rust-gng-py project
# This script combines the build steps and container setup
# Usage: ./build_and_setup.sh [dev|release]

set -e  # Exit on any error

BUILD_TYPE=${1:-"dev"}

echo "Starting streamlined build process..."
echo "Build type: $BUILD_TYPE"

# Step 1: Build the Rust library with maturin using Nix
echo "Step 1: Building Rust library with maturin..."
nix develop --command bash -c "cd gng && \
                               cargo clean && \
                               cargo build --${BUILD_TYPE} && \
                               maturin build --${BUILD_TYPE}"

# Step 2: Build the Docker container
echo "Step 2: Building Docker container..."
./scripts/container/1-buildContainer.sh

# Step 3: Start the container
echo "Step 3: Starting container..."
./scripts/container/2-startContainer.sh

# Step 4: Install the built package in the container
echo "Step 4: Installing package in container..."
./scripts/container/3-install_gng.sh

echo "Build and setup complete!"
echo "You can now attach to the container with VSCode:"
echo "  docker exec -it pytorch_project_cont fish"
echo "Jupyter is available in the container at:"
echo "  http://localhost:8888"
