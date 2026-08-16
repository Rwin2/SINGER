#!/bin/bash
# GT V3 local training on GPU 3
# Launch: nohup bash run_gt_v3_training.sh > gt_v3_training.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=/data/erwinpi/FiGS-Standalone/acados/lib:$LD_LIBRARY_PATH

cd /data/erwinpi/SINGER

echo "=== GT V3 Train History (100 epochs) — $(date) ==="
conda run -n FiGS python -u ssv_muilti3dgs_campaign.py train-history \
    --config-file configs/experiment/ssv_bc_gt_centroid.yml

if [ $? -ne 0 ]; then
    echo "FAILED: train-history at $(date)"
    exit 1
fi
echo "=== Train History DONE — $(date) ==="

echo "=== GT V3 Train Command (150 epochs) — $(date) ==="
conda run -n FiGS python -u ssv_muilti3dgs_campaign.py train-command \
    --config-file configs/experiment/ssv_bc_gt_centroid.yml

if [ $? -ne 0 ]; then
    echo "FAILED: train-command at $(date)"
    exit 1
fi
echo "=== Train Command DONE — $(date) ==="
echo "=== ALL TRAINING COMPLETE — $(date) ==="
