#!/bin/bash
cwd=$PWD

# Build Rust source
#cd ${cwd}/../
nix develop --command bash -c "cd gng_py && \
                 cargo clean && \
                 cargo build --release"

