#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root inside the torch conda environment:
#   conda run -n torch bash scripts/train/run_residual_ablations.sh

COMMON_ARGS=(
  --pos_weight 500
  --lr 1e-3
  --max_epochs 20
  --batch_size 64
)

python scripts/train/train_residual_bce.py \
  "${COMMON_ARGS[@]}" \
  --activation silu \
  --no-use_strand_features \
  --bottleneck_dilations none

python scripts/train/train_residual_bce.py \
  "${COMMON_ARGS[@]}" \
  --activation silu \
  --bottleneck_dilations none

python scripts/train/train_residual_bce.py \
  "${COMMON_ARGS[@]}" \
  --activation silu \
  --bottleneck_dilations 1,2,4,8

python scripts/train/train_residual_bce.py \
  "${COMMON_ARGS[@]}" \
  --activation gelu \
  --bottleneck_dilations 1,2,4,8

python scripts/train/train_residual_bce.py \
  "${COMMON_ARGS[@]}" \
  --activation relu \
  --bottleneck_dilations 1,2,4,8
