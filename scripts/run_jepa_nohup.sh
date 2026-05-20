#!/usr/bin/env bash
# Launch JEPA dynamics training in nohup, GPU 1, unbuffered logging.
set -u
cd /data/erwinpi/SINGER
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/data/erwinpi/FiGS-Standalone/acados/lib
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1

LOG=/data/erwinpi/SINGER/jepa_training.log
PIDF=/data/erwinpi/SINGER/jepa_training.pid

nohup conda run --no-capture-output -n FiGS \
    python -u scripts/train_jepa_dynamics.py \
    > "$LOG" 2>&1 &

echo $! > "$PIDF"
disown
echo "Launched JEPA: PID=$(cat $PIDF)  GPU=$CUDA_VISIBLE_DEVICES  log=$LOG"
