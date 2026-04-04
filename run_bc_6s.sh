#!/bin/bash
# Full BC training pipeline with 6s duration, 20Hz rate
# Expected: ~160K samples, ~12-20h total on L40S
# Model saved to: cohorts/ssv_BC_6S/roster/InstinctJester/model.pth

set -e  # exit on error

cd /data/erwinpi/SINGER
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib
source /home/erwinpi/miniconda3/etc/profile.d/conda.sh
conda activate FiGS

# Use GPU 1 (GPU 0 may have Jupyter kernels)
export CUDA_VISIBLE_DEVICES=1

CONFIG="configs/experiment/ssv_bc_6s.yml"
LOG="logs/bc_6s_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "========================================" | tee -a "$LOG"
echo "BC 6s Pipeline — $(date)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Step 1: Generate rollouts (MPC simulation + video rendering)
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 1/5: Generate rollouts..." | tee -a "$LOG"
python -u ssv_muilti3dgs_campaign.py generate-rollouts --config-file "$CONFIG" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 1 done." | tee -a "$LOG"

# Step 2: Generate validation rollouts
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 2/5: Generate validation rollouts..." | tee -a "$LOG"
python -u ssv_muilti3dgs_campaign.py generate-rollouts --config-file "$CONFIG" --validation-mode 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 2 done." | tee -a "$LOG"

# Step 3: Generate observations (pilot.OODA on each frame)
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 3/5: Generate observations..." | tee -a "$LOG"
python -u ssv_muilti3dgs_campaign.py generate-observations --config-file "$CONFIG" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 3 done." | tee -a "$LOG"

# Step 4: Train history encoder
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 4/5: Train history encoder..." | tee -a "$LOG"
python -u ssv_muilti3dgs_campaign.py train-history --config-file "$CONFIG" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 4 done." | tee -a "$LOG"

# Step 5: Train commander
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 5/5: Train commander (150 epochs)..." | tee -a "$LOG"
python -u ssv_muilti3dgs_campaign.py train-command --config-file "$CONFIG" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Step 5 done." | tee -a "$LOG"

# Safety: backup the final model
MODEL_DIR="cohorts/ssv_BC_6S/roster/InstinctJester"
BACKUP_DIR="cohorts/ssv_BC_6S/model_backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "$MODEL_DIR/model.pth" ]; then
    cp "$MODEL_DIR/model.pth" "$BACKUP_DIR/model_${TIMESTAMP}.pth"
    cp "$MODEL_DIR/model.pth" "$BACKUP_DIR/model_final.pth"
    cp "$MODEL_DIR/last_model.pth" "$BACKUP_DIR/last_model_${TIMESTAMP}.pth" 2>/dev/null || true
    cp "$MODEL_DIR/config.json" "$BACKUP_DIR/config_${TIMESTAMP}.json" 2>/dev/null || true
    cp "$MODEL_DIR/losses_Commander.pt" "$BACKUP_DIR/losses_Commander_${TIMESTAMP}.pt" 2>/dev/null || true
    echo "" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    echo "MODEL SAVED AND BACKED UP" | tee -a "$LOG"
    echo "  Primary: $MODEL_DIR/model.pth" | tee -a "$LOG"
    echo "  Backup:  $BACKUP_DIR/model_final.pth" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
else
    echo "WARNING: model.pth not found after training!" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "Pipeline complete at $(date)" | tee -a "$LOG"
