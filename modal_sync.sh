#!/bin/bash
# Upload coruscant data to Modal Volume for backup + remote execution
# Usage: bash modal_sync.sh
#
# Creates volume "singer-data" and uploads:
#   /SINGER/          - source code, configs, cohort results, pilot weights
#   /FiGS-Standalone/ - source code, 3dgs scene data, splat checkpoints, acados

set -e
MODAL="modal"

echo "=== Creating Modal volume ==="
$MODAL volume create singer-data 2>/dev/null || echo "Volume 'singer-data' already exists"

echo ""
echo "=== Uploading SINGER source code + configs ==="
$MODAL volume put singer-data /data/erwinpi/SINGER/src/ /SINGER/src/
$MODAL volume put singer-data /data/erwinpi/SINGER/configs/ /SINGER/configs/
$MODAL volume put singer-data /data/erwinpi/SINGER/notebooks/ /SINGER/notebooks/
$MODAL volume put singer-data /data/erwinpi/SINGER/scripts/ /SINGER/scripts/
$MODAL volume put singer-data /data/erwinpi/SINGER/Cs224R/ /SINGER/Cs224R/
$MODAL volume put singer-data /data/erwinpi/SINGER/ssv_muilti3dgs_campaign.py /SINGER/
$MODAL volume put singer-data /data/erwinpi/SINGER/setup.py /SINGER/ 2>/dev/null || true
$MODAL volume put singer-data /data/erwinpi/SINGER/pyproject.toml /SINGER/ 2>/dev/null || true
$MODAL volume put singer-data /data/erwinpi/SINGER/README.md /SINGER/ 2>/dev/null || true

echo ""
echo "=== Uploading pilot model weights (all cohorts) ==="
for cohort_dir in /data/erwinpi/SINGER/cohorts/*/; do
    cohort=$(basename "$cohort_dir")
    # Upload roster (model.pth, best_model.pth, config.json)
    if [ -d "$cohort_dir/roster" ]; then
        echo "  Uploading $cohort/roster/"
        $MODAL volume put singer-data "$cohort_dir/roster/" "/SINGER/cohorts/$cohort/roster/"
    fi
    # Upload any benchmark results JSONs
    for json in "$cohort_dir"/*.json; do
        [ -f "$json" ] && $MODAL volume put singer-data "$json" "/SINGER/cohorts/$cohort/"
    done
done

echo ""
echo "=== Uploading FiGS-Standalone source code ==="
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/src/ /FiGS-Standalone/src/
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/setup.py /FiGS-Standalone/ 2>/dev/null || true
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/pyproject.toml /FiGS-Standalone/ 2>/dev/null || true

echo ""
echo "=== Uploading gemsplat ==="
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/gemsplat/ /FiGS-Standalone/gemsplat/

echo ""
echo "=== Uploading acados (pre-built) ==="
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/acados/ /FiGS-Standalone/acados/

echo ""
echo "=== Uploading Hierarchical-Localization ==="
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/Hierarchical-Localization/ /FiGS-Standalone/Hierarchical-Localization/

echo ""
echo "=== Uploading 3dgs scene data + splat checkpoints (~12GB) ==="
$MODAL volume put singer-data /data/erwinpi/FiGS-Standalone/3dgs/ /FiGS-Standalone/3dgs/

echo ""
echo "=== Uploading requirements.txt ==="
pip freeze > /tmp/figs_requirements.txt 2>/dev/null
$MODAL volume put singer-data /tmp/figs_requirements.txt /requirements.txt

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
echo ""
echo "Verify with:  modal volume ls singer-data /"
echo "Shell into:   modal shell --volume singer-data"
echo ""
