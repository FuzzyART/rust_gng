#!/bin/bash
cwd=$PWD

# Build Rust source
#cd ${cwd}/../
nix develop --command bash -c "cd gng_py && \
                 cargo clean && \
                 cargo build --release"

# Run Codium
#cd ${cwd}/../
nix develop --command bash -c  "cd gng_py && maturin build --release
cd .. &&
source .venv/bin/activate &&
pip install gng_py/target/wheels/gng_py-0.1.1-cp312-cp312-linux_x86_64.whl"

