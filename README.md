# NeuroGas — Experimental ML Core

> An experimental Rust implementation of a modular clustering algorithm, initially grounded in Growing Neural Gas.

## Overview

NeuroGas is an experimental machine learning core written in Rust.

The project starts from **Growing Neural Gas (GNG)** as a known mathematical baseline. The implementation is being structured around independently replaceable processing steps, allowing individual parts of the algorithm to be investigated, measured, and eventually replaced with alternative approaches.

The long-term goal is to explore a clustering approach that is better suited to parallel processing and whose behavior can be visualized as a system of interacting particles rather than a conventional graph-based clustering algorithm.

The algorithm itself is still under development.

## Current Approach

The implementation is being decomposed into modular processing steps, including:

* Weight calculation
* Distance calculation
* Neighbor search
* Node adaptation
* Neighborhood updates
* Node insertion and removal
* Clustering behavior

The first objective is **mathematical equivalence with GNG**.

Once that baseline is established, individual processing steps can be replaced or redesigned independently. This provides a controlled way to experiment with alternative approaches while retaining a known reference implementation.

## Project Status

🚧 **Highly experimental**

The project is currently focused on establishing the core algorithm and its mathematical behavior.

At this stage:

* GNG serves as the initial reference algorithm
* The implementation is written entirely in Rust
* Core processing steps are being modularized
* Tests are used to verify algorithmic behavior
* APIs and internal architecture may change substantially
* No production guarantees are made

The architecture is expected to evolve alongside the algorithm.

## Building

Build the project with Cargo:

```bash
cargo build
```

For an optimized release build:

```bash
cargo build --release
```

## Testing

Run the test suite with:

```bash
cargo test
```

To see test output:

```bash
cargo test -- --nocapture
```

## Development Philosophy

### Establish a known baseline first

GNG provides a well-understood reference point. Before introducing fundamentally different behavior, the implementation should reproduce the relevant mathematical properties of GNG.

### Keep processing steps replaceable

Operations such as weighting, distance evaluation, and neighbor search should remain sufficiently isolated that alternative implementations can be investigated independently.

### Prefer experimentation over premature abstraction

The architecture exists to support experimentation with the algorithm. Abstractions should emerge from the requirements of the implementation rather than being introduced for architectural completeness alone.

### Treat parallelism as a first-class concern

A major motivation for the project is exploring whether clustering behavior can be expressed through operations that are naturally amenable to parallel execution.

## Long-Term Direction

The intended direction is an algorithm that differs substantially from GNG while retaining some of the useful properties of competitive and topology-forming clustering algorithms.

One particularly interesting target is a system whose state and evolution can be visualized as a **particle swarm**: a collection of interacting points that move, adapt, and organize themselves according to local information.

The exact algorithm is intentionally not defined yet.

The current implementation is the experimental foundation for discovering it.

## Scope

This crate contains the machine learning core and its tests.

Applications, visualization, data generation, and other supporting tooling are intentionally kept separate from the core implementation. This allows the algorithm and its tests to evolve independently from the systems used to consume and visualize its output.

## Contributing

This is primarily an experimental research project.

The architecture, algorithms, and APIs may change significantly as new ideas are tested. Technical feedback, discussion, and issues are welcome.

## Disclaimer

This project is experimental research software.

It should not currently be considered a finished algorithm, stable ML framework, or production-ready library. Its purpose is to explore, test, and iteratively develop ideas around clustering, modular algorithm design, and parallel machine learning.

