#!/usr/bin/env bash
# Single-node multi-GPU launch on the H100 box.
#   ./scripts/launch.sh configs/stage1_align.yaml          # use all visible GPUs
#   NUM_GPUS=4 ./scripts/launch.sh configs/stage2_finetune.yaml
set -euo pipefail

CONFIG="${1:?usage: launch.sh <config.yaml>}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

echo "Launching ${CONFIG} on ${NUM_GPUS} GPU(s)"
accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --mixed_precision bf16 \
  -m vlm.train --config "${CONFIG}"
