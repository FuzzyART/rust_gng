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

        let input_width = 2;
        let weight_rng_min = -1.0;
        let weight_rng_max = 1.0;
        let edge_removal_age = 50;
        let neuron_creation_interval = 200;
        let max_epochs = 30;
        let max_neurons = 50;
        let target_error = 0.096;
        let epsilon_w = 0.1;
        let epsilon_n = 0.006;
        let alpha = 0.5;
        let beta = 0.995;

        ctx.set_parameters(
            input_width,
            weight_rng_min,
            weight_rng_max,
            edge_removal_age,
            neuron_creation_interval,
            max_epochs,
            max_neurons,
            target_error,
            epsilon_w,
            epsilon_n,
            alpha,
            beta,
        );

    gng.init_dataset(DATA_FILE);


   //----------------------------------------
   // for complete training 
    gng.fit();
    
    //----------------------------------------
    // for training n epochs
    ctx.init_step();
    ctx.fit_step();
    ctx.get_neurons();
    ctx.get_edges();
    // do something else
    ctx.fit_step();
    ctx.get_neurons();
    ctx.get_edges();
    // ...

    
    gng.save_model_json(OUTPUT_FILE);
}


```



## Disclaimer
Not production-ready. Built for learning, experimentation, and portfolio use only.

This project is not maintained and may be abandoned permanently, especially if I move into a non-coding or non-IT role.

No support, no updates, no guarantees — fork it if you need it.
