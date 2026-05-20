#!/bin/bash
cd /data/erwinpi/FiGS-Standalone/3dgs/workspace
export CUDA_VISIBLE_DEVICES=0

echo "=== Training backroom gemsplat ===" 
# Already running, skip if output exists
if [ -d "outputs/backroom/gemsplat/2026-04-17_174902" ]; then
  echo "Backroom already training/done, checking..."
  # Wait for it to finish if still running
  while pgrep -f "ns-train.*backroom" > /dev/null; do
    sleep 30
    echo "Waiting for backroom to finish..."
  done
  echo "Backroom done!"
fi

echo "=== Training mid_gate gemsplat ==="
conda run --no-capture-output -n FiGS ns-train gemsplat \
  --data mid_gate \
  --viewer.quit-on-train-completion True \
  --output-dir outputs \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased \
  --max-num-iterations 30000 \
  nerfstudio-data \
  --orientation-method none \
  --center-method none 2>&1 | tee /data/erwinpi/SINGER/gemsplat_mid_gate.log

echo "=== Training src_open gemsplat ==="
conda run --no-capture-output -n FiGS ns-train gemsplat \
  --data src_open \
  --viewer.quit-on-train-completion True \
  --output-dir outputs \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased \
  --max-num-iterations 30000 \
  nerfstudio-data \
  --orientation-method none \
  --center-method none 2>&1 | tee /data/erwinpi/SINGER/gemsplat_src_open.log

echo "=== ALL DONE ==="
