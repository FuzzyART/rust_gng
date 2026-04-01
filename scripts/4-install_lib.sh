#!/bin/bash
# Run Codium
#cd ${cwd}/../
nix develop --command bash -c  "cd gng && maturin build --release
cd .. &&
source venv/bin/activate &&
pip install gng/target/wheels/gng_py-0.1.1-cp312-cp312-linux_x86_64.whl"


