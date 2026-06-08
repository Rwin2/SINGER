#!/usr/bin/env python3
"""
Benchmark DAgger vs IQL on the 17 specific R8 failed branches.

These are the branches where the DAgger Comprehensive model failed during R8,
which were used to collect IQL training data.

Usage:
    source activate FiGS
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_iql_r8.py
"""
import os
import sys
import json
import time
import math
from datetime import datetime

import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_all_branches, _preload_bc_trajectories,
    _swap_model, _make_terminal_fn, _evaluate_run, _reset_pilot,
    DEVICE,
)
from sousvide.control.pilot import Pilot

# The 17 failed R8 branches (from collect_rl_transitions.py)
FAILED_R8 = {
    "green clock": [5, 50, 71, 87, 88],
    "green and pink leafblower": [14, 29, 50, 52, 53, 68, 90, 91, 96, 101, 109],
    "yellow handheld cordless drill on two boxes": [7],
}

BC_COHORT = "ssv_BC_CENTROID_V9"
SCENE = "flightroom_ssv_exp"
SCENES_CFG_DIR = os.path.join(WORKSPACE, "configs", "scenes")
FLIGHTS = [[SCENE, SCENE]]

MODELS = [
    {
        "label": "DAgger",
        "cohort": "SSV_DAGGER_COMPREHENSIVE",
        "pilot_name": "InstinctJester",
        "model_file": "model_best.pth",
    },
    {
        "label": "IQL_V2",
        "cohort": "SSV_IQL_V2",
        "pilot_name": "InstinctJester",
        "model_file": "model.pth",
    },
    {
        "label": "IQL_V3_Frozen",
        "cohort": "SSV_IQL_V3_FROZEN",
        "pilot_name": "InstinctJester",
        "model_file": "model.pth",
    },
]


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "cohorts", "SSV_IQL_V2", "benchmark_r8", ts)
    os.makedirs(out_dir, exist_ok=True)

    # Load scene
    print("Loading scene...", flush=True)
    scene_data = _get_scene(SCENE, SCENES_CFG_DIR)
    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    simulator = scene_data["simulator"]

    # Load all branches and filter to R8 failures
    print("Loading branches...", flush=True)
    _preload_bc_trajectories(BC_COHORT, FLIGHTS, SCENES_CFG_DIR)

    branches_by_obj = {}
    for obj_name in queries:
        all_br = _load_all_branches(SCENE, obj_name, "", SCENES_CFG_DIR, None)
        if not all_br:
            print(f"  WARNING: No branches for {obj_name}")
            continue
        r8_ids = FAILED_R8.get(obj_name, [])
        selected = [(bid, all_br[bid]) for bid in r8_ids if bid < len(all_br)]
        branches_by_obj[obj_name] = selected
        print(f"  {obj_name}: {len(selected)}/{len(r8_ids)} R8 branches found")

    from scipy.spatial import cKDTree
    pc_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(pc_arr) if pc_arr.shape[0] > 0 else None

    # Run benchmark for each model
    results = {}

    for spec in MODELS:
        label = spec["label"]
        model_path = os.path.join(
            WORKSPACE, "cohorts", spec["cohort"], "roster",
            spec["pilot_name"], spec["model_file"]
        )
        print(f"\n{'='*70}")
        print(f"[{label}] {model_path}")
        print(f"{'='*70}", flush=True)

        pilot = Pilot(spec["cohort"], spec["pilot_name"])
        pilot.set_mode("deploy")
        pilot.model.to(DEVICE)
        pilot = _swap_model(pilot, model_path)

        model_results = {}

        for obj_idx, obj_name in enumerate(queries):
            if obj_name not in branches_by_obj:
                continue
            branch_list = branches_by_obj[obj_name]
            obj_target = obj_targets[obj_idx]

            successes, collisions, goal_dists = [], [], []

            for run_i, (branch_id, tXUi) in enumerate(branch_list):
                x0 = tXUi[1:11, 0].copy()
                t_start = float(tXUi[0, 0])
                t_end = float(tXUi[0, -1])

                terminal_fn, term_info = _make_terminal_fn(
                    obj_target, pc_tree,
                    env_min=scene_data.get("env_min"),
                    env_max=scene_data.get("env_max"),
                )

                # Reset pilot state
                _reset_pilot(pilot)

                t0 = time.time()
                result = simulator.simulate(
                    policy=pilot, t0=t_start, tf=t_end, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=None, verbose=False,
                    early_stop_fn=terminal_fn,
                )
                Tro, Xro, Uro, Iro = result[0], result[1], result[2], result[3]
                sim_s = time.time() - t0

                ev = _evaluate_run(Xro, obj_target, pc_arr,
                                   env_min=scene_data.get("env_min"),
                                   env_max=scene_data.get("env_max"),
                                   tXUi=tXUi, idx0=0)

                s = float(ev["success"])
                c = float(ev["collision"])
                gd = float(ev["goal_dist"])
                successes.append(s)
                collisions.append(c)
                goal_dists.append(gd)

                status = "OK" if s else ("COLL" if c else "MISS")
                init_d = float(np.linalg.norm(x0[:3] - np.asarray(obj_target).flatten()[:3]))
                fov_p = ev.get("fov_pct", float("nan"))
                fov_str = f"{fov_p:.0%}" if not math.isnan(fov_p) else "N/A"

                print(f"  [{label}] {run_i+1}/{len(branch_list)}  {status:4s}  "
                      f"br{branch_id}  d0={init_d:5.1f}m  goal={gd:5.2f}m  "
                      f"steps={Xro.shape[1]:>3d}  fov={fov_str:>4s}  ({sim_s:.0f}s)",
                      flush=True)

            sr = np.mean(successes) * 100
            cr = np.mean(collisions) * 100
            avg_gd = np.mean(goal_dists)
            print(f"  [{label}] --- '{obj_name}'  success={sr:.0f}% ({int(sum(successes))}/{len(successes)})  "
                  f"collision={cr:.0f}% ({int(sum(collisions))})  avg_goal={avg_gd:.2f}m",
                  flush=True)

            model_results[obj_name] = {
                "success_rate": sr,
                "collision_rate": cr,
                "avg_goal_dist": avg_gd,
                "n_runs": len(branch_list),
                "n_success": int(sum(successes)),
                "n_collision": int(sum(collisions)),
                "branch_ids": [bid for bid, _ in branch_list],
            }

        # Overall
        all_s = sum(r["n_success"] for r in model_results.values())
        all_c = sum(r["n_collision"] for r in model_results.values())
        all_n = sum(r["n_runs"] for r in model_results.values())
        overall_sr = all_s / all_n * 100 if all_n > 0 else 0
        overall_cr = all_c / all_n * 100 if all_n > 0 else 0

        print(f"\n  [{label}] OVERALL: {overall_sr:.1f}% success ({all_s}/{all_n}), "
              f"{overall_cr:.1f}% collision ({all_c}/{all_n})", flush=True)

        model_results["_overall"] = {
            "success_rate": overall_sr,
            "collision_rate": overall_cr,
            "n_runs": all_n,
            "n_success": all_s,
            "n_collision": all_c,
        }
        results[label] = model_results

        del pilot
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(out_dir, "r8_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE — R8 Failed Branches (IQL Training Data)")
    print(f"{'='*70}")
    print(f"{'Object':<45s} {'DAgger SR':>10s} {'IQL SR':>10s} {'Δ':>8s}")
    print("-" * 73)
    for obj_name in queries:
        if obj_name in results.get("DAgger", {}) and obj_name in results.get("IQL", {}):
            d_sr = results["DAgger"][obj_name]["success_rate"]
            i_sr = results["IQL"][obj_name]["success_rate"]
            delta = i_sr - d_sr
            sign = "+" if delta > 0 else ""
            print(f"  {obj_name:<43s} {d_sr:>8.1f}%  {i_sr:>8.1f}%  {sign}{delta:>6.1f}pp")
    if "_overall" in results.get("DAgger", {}) and "_overall" in results.get("IQL", {}):
        d_sr = results["DAgger"]["_overall"]["success_rate"]
        i_sr = results["IQL"]["_overall"]["success_rate"]
        delta = i_sr - d_sr
        sign = "+" if delta > 0 else ""
        print("-" * 73)
        print(f"  {'OVERALL':<43s} {d_sr:>8.1f}%  {i_sr:>8.1f}%  {sign}{delta:>6.1f}pp")


if __name__ == "__main__":
    main()
