#!/bin/bash

# Nix-based development script
# This script helps with working in the Nix development environment
# Usage: ./nix_dev.sh [command]
# Commands:
#   shell     - Enter the Nix development shell
#   build     - Build the project in Nix environment
#   install   - Install the built package in current environment
#   clean     - Clean the build artifacts
#   help      - Show this help

set -e

COMMAND=${1:-"shell"}

case $COMMAND in
    shell)
        echo "Entering Nix development shell..."
        nix develop
        ;;
    build)
        echo "Building project in Nix environment..."
        nix develop --command bash -c "cd gng && cargo build --release && maturin build --release"
        echo "Build complete! The wheel is in gng/target/wheels/"
        ;;
    install)
        echo "Installing package in current environment..."
        if [ -f "gng/target/wheels/*.whl" ]; then
            pip install gng/target/wheels/*.whl
            echo "Package installed successfully!"
        else
            echo "No wheel found. Please run 'build' first."
            exit 1
        fi
        ;;
    clean)
        echo "Cleaning build artifacts..."
        nix develop --command bash -c "cd gng && cargo clean"
        rm -rf gng/target/wheels/
        echo "Clean complete!"
        ;;
    help)
        echo "Nix-based development script for rust-gng-py project"
        echo ""
        echo "Usage: ./nix_dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  shell     - Enter the Nix development shell with all dependencies"
        echo "  build     - Build the project in Nix environment"
        echo "  install   - Install the built package in current environment"
        echo "  clean     - Clean the build artifacts"
        echo "  help      - Show this help"
        echo ""
        echo "Note: The Nix shell provides Jupyter, VSCode, and all development tools"
        echo "      without requiring Docker containers for basic development."
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Use 'help' to see available commands"
        exit 1
        ;;
esac
