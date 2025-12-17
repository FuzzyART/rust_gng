#!/bin/bash
nix develop --command bash -c  "python3.12 -m venv venv &&
source venv/bin/activate &&
pip install numpy pandas matplotlib ipykernel"


