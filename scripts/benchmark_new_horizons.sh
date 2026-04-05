#!/bin/bash
# Run benchmark for H=10 and H=20 models after main benchmark finishes
# Waits for the main benchmark (PID 2409053) to complete first
set -e
cd /data/erwinpi/SINGER

echo "[$(date)] Waiting for main benchmark (PID 2409053) to finish..."

while kill -0 2409053 2>/dev/null; do
    sleep 120
done

echo "[$(date)] Main benchmark finished! Starting H=10/H=20 benchmark..."

# Also wait for H=20 training if still running
H20_TRAINING=$(pgrep -f "train_chunked_horizons.py.*--horizons 20" || true)
if [ -n "$H20_TRAINING" ]; then
    echo "[$(date)] H=20 training still running (PID $H20_TRAINING), waiting..."
    while kill -0 $H20_TRAINING 2>/dev/null; do
        sleep 120
    done
    echo "[$(date)] H=20 training finished!"
fi

# Run benchmark for just the new horizon models
CUDA_VISIBLE_DEVICES=0 \
ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
conda run --no-capture-output -n FiGS \
    python -u scripts/eval_fair_benchmark.py \
    --models chunked_h10,chunked_h20 \
    --max-traj 50 \
    2>&1 | tee horizon_benchmark.log

echo "[$(date)] Horizon benchmark complete!"
