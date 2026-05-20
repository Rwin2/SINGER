#!/bin/bash
# Launch comprehensive DAgger on GPU 0
set -e
cd /data/erwinpi/SINGER

echo "=== Launching COMPREHENSIVE DAgger campaign on GPU 0 ==="
echo "  $(date)"
echo ""

export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib
export CUDA_VISIBLE_DEVICES=0

conda run --no-capture-output -n FiGS python -u scripts/dagger_hw1_comprehensive.py \
    2>&1 | tee /tmp/hw1_comprehensive_$(date +%Y%m%d_%H%M%S).log
