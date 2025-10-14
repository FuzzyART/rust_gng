#!/bin/bash
cwd=$PWD
cd ${cwd}/../

nix-shell --run "cd examples/test_app_rust && \
                 cargo clean && \
                 cargo build --release && \
                 cargo run  --release"