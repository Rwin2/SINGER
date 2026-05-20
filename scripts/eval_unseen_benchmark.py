#!/usr/bin/env python3
"""
UNSEEN-trajectory benchmark for ALL model variants.

Uses the 33 validation trajectories (trajectories_val*.pt) regenerated on
2026-04-06 as held-out RRT branches that no model has trained on. For each
configured model, replays the val trajectory's initial state through the sim
with the model's policy and scores success / collision / goal distance using
the canonical _evaluate_run() helper.

Key differences from eval_fair_benchmark.py:
  - Uses validation rollouts (UNSEEN), not configs/scenes/*.pkl (SEEN)
  - Resets chunk_buf between runs (so chunked policies don't carry state across runs)
  - Monkey-patches FlowMatchingCommanderWrapper.extract_inputs after load
    (older saved checkpoints lack this attr; class fix only helps new training)

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/eval_unseen_benchmark.py [--models a,b,c]
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.control.pilot import Pilot
from sousvide.instruct.train_dagger import (
    DEVICE,
    SUCCESS_RADIUS,
    _evaluate_run,
    _get_scene,
)

SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT_FOR_VAL_DATA = "ssv_BC_CENTROID_V9"   # holds the trajectories_val*.pt
SCENE = "flightroom_ssv_exp"

ALL_MODELS = {
    "baseline": {
        "label": "V9_BC_baseline",
        "cohort": "ssv_BC_CENTROID_V9",
        "pilot_name": "InstinctJester",
    },
    "v9_dagger": {
        "label": "V9_DAgger_best",
        "cohort": "SSV_DAGGER_CENTROID_V9",
        "pilot_name": "InstinctJester",
    },
    "chunked_h5": {
        "label": "Chunked_BC_H5K2",
        "cohort": "ssv_BC_CHUNKED_TEST",
        "pilot_name": "InstinctJester_chunked",
    },
    "chunked_h10": {
        "label": "Chunked_BC_H10K3",
        "cohort": "ssv_BC_CHUNKED_H10K3",
        "pilot_name": "InstinctJester_chunked_h10",
    },
    "chunked_h20": {
        "label": "Chunked_BC_H20K6",
        "cohort": "ssv_BC_CHUNKED_H20K6",
        "pilot_name": "InstinctJester_chunked_h20",
    },
    "dynreg_l01": {
        "label": "DynReg_H10K3_L0.1",
        "cohort": "ssv_BC_DYNREG_H10K3_L0.1",
        "pilot_name": "InstinctJester_dynreg_h10",
    },
    "dynreg_l05": {
        "label": "DynReg_H10K3_L0.5",
        "cohort": "ssv_BC_DYNREG_H10K3_L0.5",
        "pilot_name": "InstinctJester_dynreg_h10",
    },
    "fm_only": {
        "label": "FlowMatching_4D",
        "cohort": "ssv_BC_FM_FM_ONLY",
        "pilot_name": "InstinctJester",
    },
    "fm_chunked": {
        "label": "FlowMatching_Chunked_20D",
        "cohort": "ssv_BC_FM_FM_CHUNKED",
        "pilot_name": "InstinctJester_chunked",
    },
    "jepa_h10": {
        "label": "JEPA_H10K3",
        "cohort": "ssv_BC_JEPA_H10K3",
        "pilot_name": "InstinctJester_chunked_h10",
    },
}


def _resolve_model_path(cohort: str, pilot_name: str) -> str:
    return os.path.join(WORKSPACE, "cohorts", cohort, "roster", pilot_name, "model.pth")


def _swap_and_patch(pilot: Pilot, model_path: str) -> Pilot:
    """Load model into pilot and patch FM wrapper extract_inputs if needed."""
    pilot.model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    pilot.model.to(DEVICE)
    pilot.model.eval()
    # Older FlowMatchingCommanderWrapper checkpoints lack extract_inputs
    if not hasattr(pilot.model, "extract_inputs") and hasattr(pilot.model, "svnet"):
        pilot.model.extract_inputs = pilot.model.svnet.extract_inputs
    return pilot


def _build_fresh_pilot(cohort: str, pilot_name: str, model_path: str) -> Pilot:
    """Instantiate a brand-new Pilot per model so its profile (chunk_horizon,
    centroid_version, etc.) matches the cohort. Reusing one Pilot across
    cohorts silently kept the first model's profile and made chunked policies
    return raw 20D/40D/80D actions to the simulator (crash:
    'could not broadcast input array from shape (40,) into shape (4,)')."""
    pilot = Pilot(cohort, pilot_name)
    pilot.set_mode("deploy")
    pilot.model.to(DEVICE)
    pilot = _swap_and_patch(pilot, model_path)
    return pilot


def _reset_pilot(pilot: Pilot) -> None:
    pilot.hy_flag = False
    pilot.hy_idx = 0
    pilot.DxU.zero_()
    if hasattr(pilot, "Znn") and isinstance(pilot.Znn, torch.Tensor):
        pilot.Znn.zero_()
    if hasattr(pilot, "chunk_buf"):
        pilot.chunk_buf = None
        pilot.chunk_step = 0


def _load_val_trajectories_per_object(scene_name: str, obj_targets, queries):
    """Group trajectories_val*.pt by nearest object goal location."""
    rollout_dir = os.path.join(
        WORKSPACE, "cohorts", BC_COHORT_FOR_VAL_DATA, "rollout_data", scene_name
    )
    val_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))
    print(f"  [val] {len(val_files)} validation trajectory files in {rollout_dir}")

    obj_locs = np.array([np.squeeze(t) for t in obj_targets])
    per_obj = {q: [] for q in queries}

    for tf in val_files:
        try:
            data = torch.load(tf, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [val] skip {os.path.basename(tf)}: {e}")
            continue
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best = int(np.argmin(dists))
        if float(dists[best]) < 5.0:
            per_obj[queries[best]].append(tXUd)

    for q in queries:
        print(f"    '{q}': {len(per_obj[q])} val trajectories")
    return per_obj


def benchmark_model(
    pilot: Pilot,
    model_path: str,
    label: str,
    val_per_obj: dict,
    scene_data: dict,
) -> dict:
    print(f"\n[Unseen] ▶ {label}  ({os.path.basename(model_path)})  full trajectory from t=0")
    print(f"  chunk_horizon={getattr(pilot, 'chunk_horizon', None)}  "
          f"chunk_action_dim={getattr(pilot, 'chunk_action_dim', None)}")

    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    env_min = scene_data.get("env_min")
    env_max = scene_data.get("env_max")

    label_results = {}
    all_diag = []

    for obj_idx, obj_name in enumerate(queries):
        trajs = val_per_obj.get(obj_name, [])
        if not trajs:
            print(f"  [{label}] '{obj_name}' — no val trajectories, skip")
            continue
        obj_target = (
            obj_targets[obj_idx] if obj_idx < len(obj_targets) else trajs[0][1:4, -1]
        )
        goal_dists, successes, collisions = [], [], []
        # Build KD-tree once per object
        from scipy.spatial import cKDTree as _cKDTree
        _pc_tree = _cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None
        from sousvide.instruct.train_dagger import _make_terminal_fn
        for run_i, tXUi in enumerate(trajs):
            _reset_pilot(pilot)
            t_start = float(tXUi[0, 0])
            t_end = float(tXUi[0, -1])
            x0 = tXUi[1:11, 0].copy()

            # Terminal state: stop at success (goal zone) OR collision
            _terminal_fn, _term_info = _make_terminal_fn(
                obj_target, _pc_tree, env_min, env_max
            )

            t_sim = time.time()
            try:
                result = simulator.simulate(
                    policy=pilot,
                    t0=t_start,
                    tf=t_end,
                    x0=x0,
                    obj=np.zeros((18, 1)),
                    query=obj_name,
                    vision_processor=None,
                    verbose=False,
                    early_stop_fn=_terminal_fn,
                )
            except Exception as e:
                print(f"  [{label}] 💀 '{obj_name[:20]}' run {run_i+1}/{len(trajs)} sim crash: {e}")
                successes.append(0.0)
                collisions.append(1.0)
                goal_dists.append(float("inf"))
                continue
            Xro = result[1]
            ev = _evaluate_run(
                Xro, obj_target, epcds_arr,
                env_min=env_min, env_max=env_max,
                tXUi=tXUi, idx0=0,
            )
            success = float(ev["success"])
            collided = float(ev["collision"])
            gd = float(ev["goal_dist"])
            successes.append(success)
            collisions.append(collided)
            goal_dists.append(gd)
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            sim_s = time.time() - t_sim
            status = "✓" if success else ("💥" if collided else "✗")

            # Comprehensive per-run log
            parts = [
                f"  [{label}] {status}  '{obj_name[:20]}'  val={run_i}  run {run_i+1}/{len(trajs)}",
                f"d0={d0:.1f}m  goal_dist={gd:.2f}m  min_goal={ev['min_goal_dist']:.2f}m@step{ev['min_goal_step']}",
                f"steps={ev['n_steps']}  ({sim_s:.0f}s)",
            ]
            if collided:
                cpos = ev.get("collision_position")
                cpos_str = f"({cpos[0]:.1f},{cpos[1]:.1f},{cpos[2]:.1f})" if cpos else "?"
                cs = ev["collision_step"]
                expert_pos_cs = tXUi[1:4, min(cs, tXUi.shape[1] - 1)]
                ep_str = f"({expert_pos_cs[0]:.1f},{expert_pos_cs[1]:.1f},{expert_pos_cs[2]:.1f})"
                model_pos_cs = result[1][:3, min(cs, result[1].shape[1] - 1)]
                dev = float(np.linalg.norm(model_pos_cs - expert_pos_cs))
                parts.append(
                    f"COLL@step{cs}  type={ev['collision_type']}  "
                    f"pos={cpos_str}  expert={ep_str}  dev={dev:.2f}m  "
                    f"goal@coll={ev['collision_dist_to_goal']:.2f}m"
                )
            parts.append(
                f"clearance={ev['min_clearance']:.2f}m@step{ev['min_clearance_step']}  "
                f"fov={ev['fov_pct']:.0%}  pos_dev={ev['mean_pos_dev']:.2f}m  "
                f"ori_dev={ev['mean_orient_dev_deg']:.1f}°"
            )
            print("  ".join(parts))

            # Collect diagnostics for JSON
            all_diag.append({
                "object": obj_name, "val_idx": run_i, "d0": d0,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "collision_type": ev["collision_type"],
                "collision_step": ev["collision_step"],
                "collision_position": ev.get("collision_position"),
                "collision_dist_to_goal": ev["collision_dist_to_goal"],
                "goal_dist": gd, "min_goal_dist": ev["min_goal_dist"],
                "min_goal_step": ev["min_goal_step"],
                "first_entry_step": ev["first_entry_step"],
                "n_steps": ev["n_steps"], "sim_time": sim_s,
                "min_clearance": ev["min_clearance"],
                "min_clearance_step": ev["min_clearance_step"],
                "fov_pct": ev["fov_pct"],
                "mean_pos_dev": ev["mean_pos_dev"],
                "mean_orient_dev_deg": ev["mean_orient_dev_deg"],
                "out_of_bounds": bool(ev["out_of_bounds"]),
            })

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_mean = float(np.mean(goal_dists)) if goal_dists else float("nan")
        gd_std = float(np.std(goal_dists)) if goal_dists else float("nan")
        print(
            f"  [{label}] ── '{obj_name[:25]}'  success={sr:.0%}  "
            f"collision={cr:.0%}  goal_dist={gd_mean:.2f}±{gd_std:.2f}m  n={len(trajs)}"
        )
        label_results[obj_name] = {
            "success_rate": sr,
            "collision_rate": cr,
            "goal_dist": gd_mean,
            "goal_dist_std": gd_std,
            "n_eval": len(trajs),
        }

    if label_results:
        all_sr = [v["success_rate"] for v in label_results.values()]
        all_cr = [v["collision_rate"] for v in label_results.values()]
        all_gd = [v["goal_dist"] for v in label_results.values()]
        label_results["__overall__"] = {
            "success_rate": float(np.mean(all_sr)),
            "collision_rate": float(np.mean(all_cr)),
            "goal_dist": float(np.mean(all_gd)),
        }

    # Save per-run diagnostics JSON
    if all_diag:
        import math
        diag_path = os.path.join(WORKSPACE, "cohorts", f"unseen_diag_{label}.json")
        def _sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        sanitized = [{k: _sanitize(v) for k, v in d.items()} for d in all_diag]
        with open(diag_path, "w") as f:
            json.dump({"label": label, "runs": sanitized}, f, indent=2)
        print(f"  [diag] saved {len(all_diag)} run diagnostics → {diag_path}")

    return label_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS.keys()),
        help="Comma-separated model keys",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(WORKSPACE, "cohorts", "unseen_benchmark_results.json"),
    )
    args = parser.parse_args()

    keys = [k.strip() for k in args.models.split(",")]
    selected = []
    for k in keys:
        if k not in ALL_MODELS:
            print(f"Warning: unknown model key '{k}'")
            continue
        cfg = dict(ALL_MODELS[k])
        cfg["model_path"] = _resolve_model_path(cfg["cohort"], cfg["pilot_name"])
        if not os.path.exists(cfg["model_path"]):
            print(f"Warning: model not found for '{k}': {cfg['model_path']}")
            continue
        selected.append(cfg)

    if not selected:
        print("No models to benchmark")
        sys.exit(1)

    print(f"\nUnseen benchmark on {len(selected)} models:")
    for m in selected:
        print(f"  - {m['label']}  ({m['cohort']}/{m['pilot_name']})")

    print("\nLoading scene (gsplat)...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]

    val_per_obj = _load_val_trajectories_per_object(SCENE, obj_targets, queries)

    all_results = {}
    t0_total = time.time()
    for cfg in selected:
        pilot = None
        try:
            # Fresh Pilot per model so profile (chunk_horizon, centroid_version,
            # FM wrapper) matches the model being benchmarked.
            pilot = _build_fresh_pilot(cfg["cohort"], cfg["pilot_name"], cfg["model_path"])
            res = benchmark_model(
                pilot,
                cfg["model_path"],
                cfg["label"],
                val_per_obj,
                scene_data,
            )
            all_results[cfg["label"]] = res
        except Exception as e:
            import traceback
            print(f"  [{cfg['label']}] FAILED: {e}")
            traceback.print_exc()
            all_results[cfg["label"]] = {"__error__": str(e)}
        # Persist after each model in case of crash
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        del pilot
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("UNSEEN BENCHMARK SUMMARY")
    print(f"  total time: {(time.time()-t0_total)/3600:.2f}h")
    print("=" * 70)
    print(f"{'Model':<30} {'Success%':>10} {'Collision%':>12} {'GoalDist':>10}")
    print("-" * 70)
    for label, obj_results in all_results.items():
        if "__error__" in obj_results:
            print(f"{label:<30} {'ERROR':>10}")
            continue
        ov = obj_results.get("__overall__", {})
        sr = ov.get("success_rate", 0) * 100
        cr = ov.get("collision_rate", 0) * 100
        gd = ov.get("goal_dist", float("inf"))
        print(f"{label:<30} {sr:>9.1f}% {cr:>11.1f}% {gd:>9.2f}m")
        for obj_name, m in obj_results.items():
            if obj_name == "__overall__":
                continue
            print(
                f"  {obj_name[:28]:<28} {m['success_rate']*100:>9.1f}% "
                f"{m['collision_rate']*100:>11.1f}% {m['goal_dist']:>9.2f}m"
                f"  n={m['n_eval']}"
            )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
