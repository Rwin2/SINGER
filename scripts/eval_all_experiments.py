#!/usr/bin/env python3
"""
Benchmark all experiment variants against V9 baseline.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/eval_all_experiments.py [--models baseline,chunked,fm_only,fm_chunked]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sousvide.instruct.train_dagger import run_cross_cohort_benchmark

WORKSPACE = os.path.dirname(os.path.abspath(__file__)).replace('/scripts', '')
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
FLIGHTS = [["flightroom_ssv_exp", "flightroom_ssv_exp"]]

# All available models
ALL_MODELS = {
    "baseline": {
        "label": "V9_BC_baseline",
        "cohort": "ssv_BC_CENTROID_V9",
        "pilot_name": "InstinctJester",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CENTROID_V9",
                                    "roster", "InstinctJester", "model.pth"),
    },
    "chunked": {
        "label": "Chunked_BC_H5K2",
        "cohort": "ssv_BC_CHUNKED_TEST",
        "pilot_name": "InstinctJester_chunked",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CHUNKED_TEST",
                                    "roster", "InstinctJester_chunked", "model.pth"),
    },
    "fm_only": {
        "label": "FlowMatching_4D",
        "cohort": "ssv_BC_FM_FM_ONLY",
        "pilot_name": "InstinctJester",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_FM_FM_ONLY",
                                    "roster", "InstinctJester", "model.pth"),
    },
    "fm_chunked": {
        "label": "FlowMatching_Chunked_20D",
        "cohort": "ssv_BC_FM_FM_CHUNKED",
        "pilot_name": "InstinctJester_chunked",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_FM_FM_CHUNKED",
                                    "roster", "InstinctJester_chunked", "model.pth"),
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="baseline,chunked",
                        help="Comma-separated model keys to benchmark")
    parser.add_argument("--max-traj", type=int, default=50,
                        help="Trajectories per object (20=quick, 50=full)")
    parser.add_argument("--full-range", action="store_true", default=True,
                        help="Use full trajectory range (not just 2nd half)")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]
    models = []
    for k in model_keys:
        if k not in ALL_MODELS:
            print(f"Warning: unknown model key '{k}', skipping")
            continue
        cfg = ALL_MODELS[k]
        if not os.path.exists(cfg["model_path"]):
            print(f"Warning: model not found for '{k}': {cfg['model_path']}, skipping")
            continue
        models.append(cfg)

    if not models:
        print("No valid models to benchmark!")
        sys.exit(1)

    print(f"Benchmarking {len(models)} models: {[m['label'] for m in models]}")

    results = run_cross_cohort_benchmark(
        models=models,
        flights=FLIGHTS,
        scenes_cfg_dir=SCENES_CFG,
        benchmark_seed=42,
        max_trajectories=args.max_traj,
        output_path=os.path.join(WORKSPACE, "cohorts", "experiment_benchmark_results.json"),
        full_range=args.full_range,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("EXPERIMENT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'Success%':>10} {'Collision%':>12} {'GoalDist':>10}")
    print("-" * 70)
    for label, obj_results in results.items():
        total_sr, total_cr, total_gd, n = 0, 0, 0, 0
        for obj_name, metrics in obj_results.items():
            total_sr += metrics.get("success_rate", 0) * 100
            total_cr += metrics.get("collision_rate", 0) * 100
            total_gd += metrics.get("goal_dist", float('inf'))
            n += 1
        if n > 0:
            print(f"{label:<30} {total_sr/n:>9.1f}% {total_cr/n:>11.1f}% {total_gd/n:>9.2f}m")
    print("=" * 70)
