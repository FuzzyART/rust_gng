🧠 GNG – Experimental ML Core (Rust + Python)

⚠️ Work in Progress / Portfolio Project
This repository is actively evolving and not even remotely finished.
It serves as an experimental playground for ML systems, Rust ↔ Python integration, and tooling — and as a portfolio project to demonstrate real-world engineering skills.

Overview

This repository contains an experimental machine learning core written in Rust, with Python bindings and a growing set of tools, experiments, and examples around it.

Everything currently lives in one repository to keep development fast and flexible. The structure and boundaries will evolve as the project matures.


## Usage

### Compilation





```bash
sh scripts/1-build_lib.sh
sh scripts/2-build_py_lib.sh

sh scripts/container/1-buildContainer.sh
sh scripts/container/2-startContainer.sh
sh scripts/container/3-install_gng.sh

```
in VSCode/Codium, attach to remote container




## Current components

🦀 Rust GNG library

Core algorithms and performance-critical logic

🐍 Python bindings

Built using maturin

Exposed as a Python package

🧪 Experiments & prototypes

Jupyter notebooks for ML exploration

🛠 Quick-and-dirty tools

Data generators

Visualization helpers

📚 Documentation & setup guides

Including notes on GPU / ROCm / TensorFlow setups

🚧 Planned

A main application that combines everything into a usable system

Project Status

✔ Early architecture in place

✔ Rust ↔ Python integration working

✔ Nix-based dev environment

❌ APIs not stable

❌ No backwards compatibility guarantees

❌ Not production-ready

This project prioritizes learning, experimentation, and correctness over polish — for now.

Development Environment

The project uses Nix to provide a reproducible development environment.

Make sure you have:

Nix (with flakes enabled)

Rust toolchain

Python ≥ 3.12

## Build Instructions

- Build the Rust library
```bash
nix develop --command bash -c "cd gng && cargo build"
```
- Build the Python package

    - This creates Python wheels using maturin.

```bash
nix develop --command bash -c "cd gng && maturin build"
```

- Install the Python package in a virtual environment

```bashy
python3.12 -m venv venv && \
source venv/bin/activate && \
pip install gng/target/wheels/*
```

## Repository Structure (High-Level)
```
├── app
├── experiments
├── gng
├── scripts
├── tools
├── venv
├── flake.lock
├── flake.nix
├── LICENSE
├── README.md

```


Structure is expected to change as the project grows.

## Goals

- Explore ML system design in Rust

- Learn and demonstrate FFI & Python bindings

- Experiment with GPU acceleration
- Build a foundation for a real application
- Serve as a portfolio project to demonstrate:
- Systems programming
- ML experimentation
- Tooling & developer experience
- Pragmatic engineering decisions
## Non-Goals (for now)
- Production stability
- Clean public API
- Backwards compatibility
- Polished UX
- These will come later — if and when the project stabilizes.

## Contributions

This is primarily a personal project, but:

- Issues, feedback, and discussion are welcome

- Code style and structure may change rapidly

## Disclaimer

This repository reflects active learning and experimentation.
Some solutions are intentionally exploratory rather than “final”.

If you’re reviewing this as part of a hiring process:
this repo is meant to show how I think, build, and iterate — not just polished end results.
