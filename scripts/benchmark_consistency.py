#!/usr/bin/env python3
"""
Consistency verification — run the V9 DAgger benchmark with multiple seeds
to confirm that 88% success rate is robust and not an artifact of seed=42.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
      LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
      conda run --no-capture-output -n FiGS python -u scripts/benchmark_consistency.py
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────────────────────
COHORT_NAME     = "SSV_DAGGER_CENTROID_V9"
BC_COHORT_NAME  = "ssv_BC_CENTROID_V9"
METHOD_NAME     = "rrt_6s"
PILOT_NAME      = "InstinctJester"
MAX_TRAJECTORIES = 50          # same as original benchmark
SEEDS           = [42, 123, 456, 789, 2026]  # 42 is the original

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
os.chdir(WORKSPACE)

# ── Imports (after path setup) ───────────────────────────────────────────────
from sousvide.instruct.train_dagger import (
    _run_benchmark_pilot,
    _get_scene,
    _preload_bc_trajectories,
    _preload_all_pkls,
    Pilot,
)

def main():
    # Paths
    cohort_path    = os.path.join(WORKSPACE, "cohorts", COHORT_NAME)
    scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")
    method_path    = os.path.join(WORKSPACE, "configs", "method", METHOD_NAME + ".json")
    flights        = [("flightroom_ssv_exp", "flightroom_ssv_exp")]

    # Load method config for frame/rollout names
    with open(method_path) as f:
        mcfg = json.load(f)
    frame_name   = mcfg.get("sample_set", {}).get("frame", "carl")
    rollout_name = mcfg.get("sample_set", {}).get("rollout", "baseline")

    # Scene configs
    objective_configs   = {}
    collision_detectors = {}
    scene_names         = []
    for scene_name, _ in flights:
        if scene_name not in objective_configs:
            with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
                objective_configs[scene_name] = yaml.safe_load(f)
            scene_names.append(scene_name)

    # Pre-load scene + trajectories (once)
    print("[ConsistencyBench] Loading scene...")
    for sn in scene_names:
        _get_scene(sn, scenes_cfg_dir, frame_name, rollout_name)

    n_bc = _preload_bc_trajectories(BC_COHORT_NAME, flights, scenes_cfg_dir)
    print(f"[ConsistencyBench] {n_bc} BC trajectories loaded\n")

    # Load pilot with DAgger best model
    model_path = os.path.join(cohort_path, "roster", PILOT_NAME, "model.pth")
    print(f"[ConsistencyBench] Model: {model_path}")
    assert os.path.isfile(model_path), f"Model not found: {model_path}"

    pilot = Pilot(COHORT_NAME, PILOT_NAME)
    pilot.set_mode("deploy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot.model.to(device)

    # Benchmark paths (reuse existing rrt backup)
    sim_base   = os.path.join(cohort_path, "simulation_data", "dagger")
    rrt_backup = os.path.join(cohort_path, "dagger_data", PILOT_NAME, "_benchmark_rrt_backup")

    print(f"\n{'='*70}")
    print(f"[ConsistencyBench] Running {len(SEEDS)} seeds × {MAX_TRAJECTORIES} traj/obj × 3 objects")
    print(f"[ConsistencyBench] Seeds: {SEEDS}")
    print(f"{'='*70}\n")

    results = {}
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n{'─'*50}")
        print(f"  SEED = {seed}")
        print(f"{'─'*50}")

        metrics = _run_benchmark_pilot(
            pilot=pilot,
            model_path=model_path,
            label=f"seed_{seed}",
            workspace_path=WORKSPACE,
            cohort_name=COHORT_NAME,
            cohort_path=cohort_path,
            method_name=METHOD_NAME,
            flights=flights,
            scenes_cfg_dir=scenes_cfg_dir,
            objective_configs=objective_configs,
            collision_detectors=collision_detectors,
            scene_names=scene_names,
            sim_base=sim_base,
            rrt_backup=rrt_backup,
            benchmark_seed=seed,
            max_trajectories=MAX_TRAJECTORIES,
        )

        elapsed = time.time() - t0
        sr = float(metrics.get("success_rate", np.array([0])).mean()) * 100
        cr = float(metrics.get("collision_rate", np.array([0])).mean()) * 100
        gd = float(metrics.get("goal_dist", np.array([0])).mean())
        results[seed] = {"success": sr, "collision": cr, "goal_dist": gd, "time_s": elapsed}
        print(f"  → seed={seed}: success={sr:.1f}%  collision={cr:.1f}%  goal_dist={gd:.2f}m  ({elapsed:.0f}s)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("[ConsistencyBench] RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Seed':>6}  {'Success':>8}  {'Collision':>10}  {'Goal Dist':>10}")
    print(f"{'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}")

    successes = []
    for seed in SEEDS:
        r = results[seed]
        print(f"{seed:>6}  {r['success']:>7.1f}%  {r['collision']:>9.1f}%  {r['goal_dist']:>9.2f}m")
        successes.append(r['success'])

    mean_sr = np.mean(successes)
    std_sr  = np.std(successes)
    print(f"{'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}")
    print(f"{'Mean':>6}  {mean_sr:>7.1f}%  ±{std_sr:.1f}%")
    print(f"\n[ConsistencyBench] Original claim: 88.0% (seed=42)")
    print(f"[ConsistencyBench] Cross-seed mean: {mean_sr:.1f}% ± {std_sr:.1f}%")

    if std_sr < 5.0 and mean_sr > 80.0:
        print("[ConsistencyBench] ✅ ROBUST — low variance, consistently high success")
    elif std_sr < 10.0:
        print("[ConsistencyBench] ⚠️ MODERATE variance — results depend on start positions")
    else:
        print("[ConsistencyBench] ❌ HIGH variance — 88% may not be representative")

    # Save JSON
    out_path = os.path.join(cohort_path, "benchmark_consistency.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[ConsistencyBench] Results saved → {out_path}")


if __name__ == "__main__":
    main()
