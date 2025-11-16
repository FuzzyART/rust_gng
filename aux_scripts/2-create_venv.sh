##!/usr/bin/env bash
#set -e
#
## Create and activate virtual environment
#python -m venv .venv
#source .venv/bin/activate
#
## Upgrade pip and install dependencies
#pip install --upgrade pip
#pip install numpy pandas matplotlib ipykernel
#
## Install your wheel
#pip install dist/*.whl
#
## Register kernel for Jupyter
#python -m ipykernel install --user --name=gng_py --display-name "Python (gng_py)"




#!/bin/bash
nix develop --command bash -c  "python -m venv .venv &&
source .venv/bin/activate &&
pip install numpy pandas matplotlib ipykernel "


