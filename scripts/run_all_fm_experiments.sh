#!/bin/bash
# Run all flow matching experiments sequentially
# Usage: bash scripts/run_all_fm_experiments.sh

set -e
cd /data/erwinpi/SINGER

export CUDA_VISIBLE_DEVICES=0
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib

echo "=== Phase 2a: Flow Matching (fm_only, 4D) ==="
conda run -n FiGS python scripts/test_flow_matching.py --mode fm_only

echo ""
echo "=== Phase 2b: Flow Matching + Chunking (fm_chunked, 20D) ==="
conda run -n FiGS python scripts/test_flow_matching.py --mode fm_chunked

echo ""
echo "=== Benchmarking All Models ==="
conda run -n FiGS python scripts/eval_all_experiments.py --models baseline,chunked,fm_only,fm_chunked --max-traj 20

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
