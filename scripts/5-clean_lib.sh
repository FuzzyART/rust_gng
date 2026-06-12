#!/bin/bash
cwd=$PWD

# Build Rust source

nix develop --command bash -c "cd gng && \
                               cargo clean"

