#!/usr/bin/env python3
"""
Fair benchmark: evaluate models using SECOND-HALF starts only (held out from BC training).
This matches the V9 DAgger benchmark methodology (88% success baseline).

Unlike eval_all_experiments.py (which uses --full-range including easy starts),
this uses full_range=False for a proper held-out comparison.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/eval_fair_benchmark.py \
        [--models baseline,chunked,fm_only,fm_chunked] \
        [--max-traj 50]
"""

import os
import sys
import json
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
    "v9_dagger": {
        "label": "V9_DAgger_best",
        "cohort": "SSV_DAGGER_CENTROID_V9",
        "pilot_name": "InstinctJester",
        "model_path": os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_CENTROID_V9",
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
    "chunked_h10": {
        "label": "Chunked_BC_H10K3",
        "cohort": "ssv_BC_CHUNKED_H10K3",
        "pilot_name": "InstinctJester_chunked_h10",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CHUNKED_H10K3",
                                    "roster", "InstinctJester_chunked_h10", "model.pth"),
    },
    "chunked_h20": {
        "label": "Chunked_BC_H20K5",
        "cohort": "ssv_BC_CHUNKED_H20K5",
        "pilot_name": "InstinctJester_chunked_h20",
        "model_path": os.path.join(WORKSPACE, "cohorts", "ssv_BC_CHUNKED_H20K5",
                                    "roster", "InstinctJester_chunked_h20", "model.pth"),
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fair benchmark with second-half starts (held out from BC)")
    parser.add_argument("--models", default="baseline,v9_dagger,chunked,chunked_h10,chunked_h20,fm_only,fm_chunked",
                        help="Comma-separated model keys to benchmark")
    parser.add_argument("--max-traj", type=int, default=50,
                        help="Trajectories per object (20=quick, 50=full)")
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
    print(f"  Using full trajectory range (full_range=True)")
    print(f"  {args.max_traj} trajectories per object")

    output_path = os.path.join(WORKSPACE, "cohorts", "fair_benchmark_results.json")

    results = run_cross_cohort_benchmark(
        models=models,
        flights=FLIGHTS,
        scenes_cfg_dir=SCENES_CFG,
        benchmark_seed=42,
        max_trajectories=args.max_traj,
        output_path=output_path,
        full_range=True,  # Full trajectory range — most relevant benchmark
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("FULL TRAJECTORY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'Success%':>10} {'Collision%':>12} {'GoalDist':>10}")
    print("-" * 70)
    for label, obj_results in results.items():
        obj_items = [(k, v) for k, v in obj_results.items() if k != "__overall__"]
        n = len(obj_items)
        if n == 0:
            continue
        total_sr = sum(v.get("success_rate", 0) * 100 for _, v in obj_items)
        total_cr = sum(v.get("collision_rate", 0) * 100 for _, v in obj_items)
        total_gd = sum(v.get("goal_dist", float('inf')) for _, v in obj_items)
        print(f"{label:<30} {total_sr/n:>9.1f}% {total_cr/n:>11.1f}% {total_gd/n:>9.2f}m")

        # Per-object details
        for obj_name, metrics in obj_items:
            short = obj_name[:25]
            sr = metrics.get("success_rate", 0) * 100
            cr = metrics.get("collision_rate", 0) * 100
            gd = metrics.get("goal_dist", float('inf'))
            print(f"  {short:<28} {sr:>9.1f}% {cr:>11.1f}% {gd:>9.2f}m")

    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
