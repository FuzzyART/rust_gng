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

