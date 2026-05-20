#!/bin/bash
# RL Full Campaign v2 -- pkl-based start sampling, 30 iters, 20 eval/object
set -e

cd /data/erwinpi/SINGER

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FiGS

# Set ACADOS
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/data/erwinpi/FiGS-Standalone/acados/lib"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=== RL Full Campaign v2 (pkl-based starts, 30 iters) ==="
echo "Start: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

stdbuf -oL -eL python ssv_muilti3dgs_campaign.py rl-finetune \
  --config-file configs/experiment/ssv_rl_finetune_v9_full2.yml

echo "=== Done: $(date) ==="
