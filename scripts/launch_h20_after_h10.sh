#!/bin/bash
# Wait for H=10 training (PID 2238700) to finish, then start H=20
set -e
cd /data/erwinpi/SINGER

echo "[$(date)] Waiting for H=10 training (PID 2238700) to finish..."

while kill -0 2238700 2>/dev/null; do
    sleep 60
done

echo "[$(date)] H=10 training finished! Starting H=20 K=5 training..."

CUDA_VISIBLE_DEVICES=0 \
ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
conda run --no-capture-output -n FiGS \
    python scripts/train_chunked_horizons.py --horizons 20 --epochs 30

echo "[$(date)] H=20 training complete!"
