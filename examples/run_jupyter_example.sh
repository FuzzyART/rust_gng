
#!/bin/bash
cwd=$PWD

# Build Rust source
cd ${cwd}/../
nix-shell --run "cd gng_py && \
                 cargo clean && \
                 cargo build --release"

# Run Codium
cd ${cwd}/../
nix-shell --run "cd gng_py && maturin develop --release && \
            cd .. && codium"