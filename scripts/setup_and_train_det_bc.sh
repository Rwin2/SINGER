#!/bin/bash
# Setup a new deterministic BC cohort from source and train Commander
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/setup_and_train_det_bc.sh <config_file>
set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash scripts/setup_and_train_det_bc.sh <config_file>"
    exit 1
fi

# Parse cohort and source_cohort from yaml
COHORT=$(grep '^cohort:' "$CONFIG_FILE" | sed 's/cohort: *"\(.*\)"/\1/')
SOURCE=$(grep '^source_cohort:' "$CONFIG_FILE" | sed 's/source_cohort: *"\(.*\)"/\1/')
PILOT="InstinctJester"

echo "Setting up cohort: $COHORT from source: $SOURCE"

COHORT_DIR="cohorts/$COHORT"
ROSTER_DIR="$COHORT_DIR/roster/$PILOT"
SOURCE_ROSTER="cohorts/$SOURCE/roster/$PILOT"
SOURCE_OBS="cohorts/$SOURCE/observation_data/$PILOT"
DEST_OBS="$COHORT_DIR/observation_data/$PILOT"

# Create directories
mkdir -p "$ROSTER_DIR"
mkdir -p "$DEST_OBS"

# Copy config and model from source
cp "$SOURCE_ROSTER/config.json" "$ROSTER_DIR/config.json"
cp "$SOURCE_ROSTER/model.pth" "$ROSTER_DIR/model.pth"
echo "Copied config.json and model.pth from $SOURCE"

# Symlink observation data
for course_dir in $SOURCE_OBS/*/; do
    course=$(basename "$course_dir")
    if [ ! -e "$DEST_OBS/$course" ]; then
        ln -s "$(realpath "$course_dir")" "$DEST_OBS/$course"
        echo "Symlinked $course"
    fi
done

echo "Setup done. Training Commander..."

# Train
python ssv_muilti3dgs_campaign.py train-command \
    --config-file "$CONFIG_FILE" \
    --use-wandb --wandb-project singer
