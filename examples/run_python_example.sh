
#!/bin/bash
cwd=$PWD

# Build Rust source
cd ${cwd}/../
nix-shell --run "cd gng_py && \
                 cargo clean && \
                 cargo build --release"

# Run test app, after running maturin
cd ${cwd}/../
nix-shell --run "cd gng_py &&  \
                 maturin develop --release && \
                 cd ../examples/test_app_py && \
                 python test_app.py"