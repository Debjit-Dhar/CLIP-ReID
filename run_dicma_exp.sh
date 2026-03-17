#!/usr/bin/env bash

# Example: run a DiCMA training experiment on Market-1501 with ViT backbone.
# Adjust DATA_ROOT and OUTPUT_DIR as needed.

DATA_ROOT="/path/to/datasets"
OUTPUT_DIR="./runs/dicma"

python train.py \
  --config_file configs/person/vit_dicma.yml \
  --opts DATASETS.ROOT_DIR "${DATA_ROOT}" OUTPUT_DIR "${OUTPUT_DIR}"
