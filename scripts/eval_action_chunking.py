#!/usr/bin/env python3
"""
Evaluate action chunking BC model using SINGER's benchmark.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/eval_action_chunking.py
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sousvide.instruct.train_dagger import run_cross_cohort_benchmark

# Paths
WORKSPACE = os.path.dirname(os.path.abspath(__file__)).replace('/scripts', '')
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
FLIGHTS = [["flightroom_ssv_exp", "flightroom_ssv_exp"]]

# Models to compare
models = [
    {
        "label": "V9_BC_baseline",
        "cohort": "ssv_BC_CENTROID_V9",
        "pilot_name": "InstinctJester",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CENTROID_V9",
                                    "roster", "InstinctJester", "model.pth"),
    },
    {
        "label": "Chunked_BC",
        "cohort": "ssv_BC_CHUNKED_TEST",
        "pilot_name": "InstinctJester_chunked",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CHUNKED_TEST",
                                    "roster", "InstinctJester_chunked", "model.pth"),
    },
]

# Quick benchmark: 20 trajectories per object (vs 50 for full benchmark)
results = run_cross_cohort_benchmark(
    models=models,
    flights=FLIGHTS,
    scenes_cfg_dir=SCENES_CFG,
    benchmark_seed=42,
    max_trajectories=20,
    output_path=os.path.join(WORKSPACE, "cohorts", "ssv_BC_CHUNKED_TEST",
                              "benchmark_results.json"),
)

# Print summary
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
for label, obj_results in results.items():
    print(f"\n{label}:")
    total_success = 0
    total_collision = 0
    total_runs = 0
    for obj_name, metrics in obj_results.items():
        sr = metrics.get("success_rate", 0) * 100
        cr = metrics.get("collision_rate", 0) * 100
        gd = metrics.get("goal_dist", float('inf'))
        print(f"  {obj_name}: success={sr:.0f}%, collision={cr:.0f}%, goal_dist={gd:.2f}m")
        total_success += sr
        total_collision += cr
        total_runs += 1
    if total_runs > 0:
        print(f"  AVERAGE: success={total_success/total_runs:.1f}%, "
              f"collision={total_collision/total_runs:.1f}%")

print("\nResults saved to cohorts/ssv_BC_CHUNKED_TEST/benchmark_results.json")
