#!/bin/bash

# Build Rust library only (no container)
# This script builds just the Rust components using Nix
# Usage: ./build_lib_only.sh [dev|release]

set -e  # Exit on any error

BUILD_TYPE=${1:-"dev"}
echo "Building Rust library with $BUILD_TYPE build..."

# Build the Rust library with maturin using Nix
nix develop --command bash -c "cd gng && \
                               cargo clean && \
                               cargo build --${BUILD_TYPE} && \
                               maturin build --${BUILD_TYPE}"

echo "Rust library built successfully!"
echo "The wheel is available in gng/target/wheels/"
