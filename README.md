GNG — Rust Machine Learning Library

Experimental machine learning library written in Rust.

Overview

GNG is an experimental machine learning library focused on implementing and exploring ML algorithms in Rust.

The project is primarily concerned with the core library and its implementation, rather than applications, visualization, or data-processing tooling.

The repository is intentionally kept small and focused. Applications and supporting tooling that consume GNG are maintained separately.

Project Status

This is an experimental and evolving project.

- 🦀 Core implementation written in Rust
- 🧪 Tests included with the library
- 🚧 APIs may change
- 🚧 Not production-ready

The current priority is building a solid understanding of the algorithms and their implementation rather than maintaining a stable public API.

## Building

Build the library with Cargo:

```bash
cd gng
cargo build
```

For an optimized release build:

```bash
cd gng
cargo build --release
```

### Testing

Run the test suite with:

```bash
cd gng
cargo test
```

To run tests with output:

```bash
cd gng
cargo test -- --nocapture
```

### Development

The project uses the standard Rust toolchain and Cargo for development and dependency management.

A typical development workflow is:

cd gng
cargo build
cargo test

## Repository Scope

This repository contains the GNG library itself.

Applications, experiments, visualization, data generation, and other supporting tools are intentionally kept outside this repository to keep the core library focused.

## Goals
- Explore machine learning algorithms in Rust
- Build a maintainable and testable ML core
- Learn and experiment with systems-level ML implementation
- Keep the core library independent from application-specific tooling

## Non-Goals

At this stage, the project does not aim to provide:

- A stable public API
- Production guarantees
- Backwards compatibility
- A complete ML framework
- Application-level tooling or visualization

These may change as the project develops.

## Contributing

This is primarily a personal experimental project, but issues, feedback, and discussion are welcome.

The codebase and APIs may change significantly as development continues.

## Disclaimer

This project is experimental and actively evolving. Some implementations and design decisions are exploratory and may change as the project develops.
