#!/bin/bash
cwd=$PWD

cd ${cwd}/../dataset_creator/
nix-shell --run "python dataset_creator_circles.py \
  --filename /tmp/circles.csv \
  --num_samples 300 \
  --noise 0.05 \
  --factor 0.3 \
  --rng_seed 1234"

nix-shell --run "python dataset_creator_blobs.py \
  --filename /tmp/blobs.csv \
  --num_samples 100 \
  --num_centers 4 \
  --std_dev 0.5 \
  --rng_seed 1234"

cd ${cwd}/../gng_lib
nix-shell --run "cargo build --release"

cd ${cwd}/../gng_test_app
nix-shell --run "cargo build --release"


cd ${cwd}/../gng_test_app
nix-shell --run "./target/release/gng_test_app \
  --config input.json \
  --data /tmp/circles.csv \
  --output /tmp/res.json"

cd ${cwd}/../neural_gas_plotter
nix-shell --run "python input_set_plotter.py \
  -f /tmp/circles.csv" &
nix-shell --run "python neural_gas_plotter.py \
  /tmp/res.json"
