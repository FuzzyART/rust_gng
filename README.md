# rust_gng — Experiments in Rust & Python
This repo is a personal playground for experimenting with the Growing Neural Gas (GNG) algorithm and related tooling. It’s a work-in-progress portfolio project combining Rust, Python, and Nix, designed to explore cross-language workflows and reproducible environments.

## Project Structure
The repository contains three loosely coupled subprojects:

gng-core (Rust) A Rust implementation of the Growing Neural Gas algorithm. Currently under development.

dataset-maker (Python) A simple wrapper around sklearn.datasets to generate synthetic data for GNG training. Outputs to CSV.

gas-visualizer (Python) A visualization tool for inspecting GNG behavior and dataset structure. Uses matplotlib.

Each subproject includes its own nix-shell for reproducible development environments.

## Communication
Data is exchanged via CSV and JSON files.

Future plans include direct IPC or shared memory once the Rust core stabilizes.

## Goals
- Build a clean, idiomatic GNG implementation in Rust

- Create a flexible dataset generator for experimentation

- Visualize GNG evolution over time

- Showcase cross-language tooling and reproducibility with Nix

## Status
- Portfolio Project & Work in Progress This repo is part of my Rust portfolio. I created it to show recruiters what I’m working on—even if it’s not finished yet. All the LLMs agreed: publishing early is a great idea.

## Next Steps
- Add everything needed for a demo on Linux systems

- Write documentation

- Build a Python wrapper to integrate with Keras pipelines

## Ultimate Goal
- Develop real-world workflows and use cases

- Launch a Wasm-based website with an interactive demo
