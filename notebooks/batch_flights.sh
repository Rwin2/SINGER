#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/<username>/SINGER/configs/experiment/ssv_multi3dgs.yml"
TMP="/home/<username>/SINGER/configs/experiment/ssv_multi3dgs.tmp.yml"

# reverse order
FLIGHTS=("packardpark" "flightroom_ssv_exp")

for flight in "${FLIGHTS[@]}"; do
  echo "=== Running flight = $flight ==="

  perl -pe '
    if (/^[[:space:]]*-\s*\[/)      { s/^/#/; }
    if (/^([[:space:]]*)#\s*-\s*\[.*'"$flight"'.*\]/) { s/^([[:space:]]*)#\s*/\1/; }
  ' "$CONFIG" > "$TMP"

  # build the args in an array
  args=( generate-rollouts --config-file "$TMP" )
  
  args+=( --validation-mode )

  # args+=( --use-wandb --wandb-project <wandb-project-name> )
  args+=( --use-wandb --wandb-project <wandb-project-name>  \
          --wandb-run-id <wandb-run-id> --wandb-resume allow )

  python ssv_multi3dgs_campaign_coruscant.py "${args[@]}"
done

rm -f "$TMP"
