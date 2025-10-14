#!/bin/bash
cwd=$PWD

# Create Circles Dataset
cd ${cwd}/../dataset_creator/
nix-shell --run "python dataset_creator_circles.py \
                 --filename /tmp/circles.csv \
                 --num_samples 300 \
                 --noise 0.05 \
                 --factor 0.3 \
                 --rng_seed 1234"

# Create Blobs Dataset
nix-shell --run "python dataset_creator_blobs.py \
                 --filename /tmp/blobs.csv \
                 --num_samples 100 \
                 --num_centers 4 \
                 --std_dev 0.5 \
                 --rng_seed 1234"

# Build Rust source
cd ${cwd}/../
nix-shell --run "cd gng_py && \
                 cargo clean && \
                 cargo build --release"

# Run test app
cd ${cwd}/../
nix-shell --run "cd examples/test_app_rust && \
./target/release/gng_test_app \
  --config ../config.json \
  --data /tmp/circles.csv \
  --output /tmp/res.json"

# display images
cd ${cwd}/../neural_gas_plotter
nix-shell --run "python input_set_plotter.py \
                 -f /tmp/circles.csv" &
nix-shell --run "python neural_gas_plotter.py \
                 /tmp/res.json"
