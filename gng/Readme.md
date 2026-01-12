markdown
# Neurogas

**Status:** 🚧 In development  
**Purpose:** Learning project & portfolio showcase  
**Note:** The API is experimental and subject to change.

---

## Overview

Neurogas is a Rust implementation of **Growing Neural Gas (GNG)**, an unsupervised learning algorithm for topology discovery and clustering.  
This crate is primarily developed for **educational purposes** and as part of my **portfolio**. While it is functional, the API is evolving and may change significantly between versions.

---

## Features

- 🧠 Growing Neural Gas implementation
- 🔄 Adaptive topology learning
- ✍️ Built as a demonstration of Rust and open-source contribution

---

## Installation

Cargo.toml
```toml

[package]
name = "gng_test_app"
version = "0.1.0"
edition = "2021"

[dependencies]
neurogas = {version = "0.0.2"}
```

```rust
use::neurogas::Gng;

fn main() {
    const DATA_FILE: &str = "../blobs.csv";
    const OUTPUT_FILE: &str = "/tmp/output.json";
    let mut gng = Gng::new();

    gng.set_parameters(
        2,      // input_width
        -1.0,   // weight_rng_min
        1.0,    // weight_rng_max
        50,     // edge_removal_age
        200,    // neuron_creation_interval
        10000,  // max_train_iterations
        0.096,  // target_error
        0.1,    // epsilon_w
        0.006,  // epsilon_n
        0.5,    // alpha
        0.995,  // beta
    );

    gng.init_dataset(DATA_FILE);
    gng.fit();
    gng.save_model_json(OUTPUT_FILE);
}


```



## Disclaimer
Not production-ready. Built for learning, experimentation, and portfolio use only.

This project is not maintained and may be abandoned permanently, especially if I move into a non-coding or non-IT role.

No support, no updates, no guarantees — fork it if you need it.