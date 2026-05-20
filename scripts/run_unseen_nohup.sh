#!/usr/bin/env bash
# Launch eval_unseen_benchmark.py in nohup, GPU 1, with unbuffered logging.
set -u
cd /data/erwinpi/SINGER
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/data/erwinpi/FiGS-Standalone/acados/lib
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

LOG=/data/erwinpi/SINGER/unseen_benchmark_v2.log
PIDF=/data/erwinpi/SINGER/unseen_benchmark_v2.pid
OUT=/data/erwinpi/SINGER/cohorts/unseen_benchmark_results_v2.json

# chunked first so we can verify the fix in the first ~5 min
MODELS="chunked_h10,chunked_h5,chunked_h20,dynreg_l01,dynreg_l05,baseline,v9_dagger,fm_only,fm_chunked"

nohup conda run --no-capture-output -n FiGS \
    python -u scripts/eval_unseen_benchmark.py \
        --models "$MODELS" \
        --output "$OUT" \
    > "$LOG" 2>&1 &

echo $! > "$PIDF"
disown
echo "Launched: $(cat $PIDF)  log=$LOG"
