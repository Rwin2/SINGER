#!/bin/bash
set -e
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/data/erwinpi/FiGS-Standalone/acados/lib"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd /data/erwinpi/SINGER
eval "$(conda shell.bash hook)"
conda activate FiGS
exec stdbuf -oL -eL python -u scripts/eval_rl_benchmark.py
