# rust_gng — Growing Neural Gas in Rust & Python

This is a personal playground for experimenting with the Growing Neural Gas (GNG) algorithm using Rust, Python, and Nix. It's a work-in-progress portfolio project focused on cross-language workflows and reproducible development environments.

🧱 Project Structure

gng_py/ (Rust) — A working implementation of the GNG algorithm

dataset_creator/ (Python) — Generates synthetic data via sklearn, outputs to CSV

neural_gas_plotter/ (Python) — Visualizes GNG behavior using matplotlib

Each component includes its own nix-shell for consistent, reproducible dev environments.

📁 Examples

See the examples/ directory for runnable usage demos:

run_rust_example.sh

run_python_example.sh

run_jupyter_example.sh

simple_usage.sh

test_app_py

All scripts are run via nix-shell for setup-free execution.

🔄 Data Flow

Data is currently exchanged using CSV and JSON.
Future work will explore shared memory or IPC for tighter integration.

✅ Project Goals

Build a clean, idiomatic Rust implementation of GNG

Provide flexible dataset generation & visualization tools

Showcase Nix-based reproducibility and cross-language workflows

Enable Keras integration and interactive WebAssembly demos

🚧 Status

GNG core is working

Examples are live

Still in early development — follow along as it evolves

