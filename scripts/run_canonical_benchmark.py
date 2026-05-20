#!/usr/bin/env python3
"""
Run canonical BC vs DAgger benchmark using run_cross_cohort_benchmark().
Same seed/branches as bc_vs_dagger_comparison.py for consistency check.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n FiGS python -u scripts/run_canonical_benchmark.py
"""
import os, sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import run_cross_cohort_benchmark

SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_CENTROID_V9"
PILOT_NAME = "InstinctJester"

SEED = 42
MAX_BRANCHES = 50

models = [
    {
        "label": "BC_Centroid_V9",
        "cohort": BC_COHORT,
        "pilot_name": PILOT_NAME,
        "model_path": os.path.join(
            WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "model.pth"),
    },
    {
        "label": "DAgger_Centroid_V9",
        "cohort": DAGGER_COHORT,
        "pilot_name": PILOT_NAME,
        "model_path": os.path.join(
            WORKSPACE, "cohorts", DAGGER_COHORT, "roster", PILOT_NAME, "model.pth"),
    },
]

flights = [[SCENE, None]]
output_path = os.path.join(
    WORKSPACE, "cohorts", DAGGER_COHORT, "visualizations",
    "canonical_benchmark_results.json")

results = run_cross_cohort_benchmark(
    models=models,
    flights=flights,
    scenes_cfg_dir=SCENES_CFG,
    benchmark_seed=SEED,
    max_trajectories=MAX_BRANCHES,
    output_path=output_path,
    bc_cohort=BC_COHORT,
)

print(f"\nResults saved to {output_path}")
