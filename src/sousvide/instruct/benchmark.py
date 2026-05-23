"""
Unified benchmark for SINGER models.

Replaces: benchmark_seen_unseen.py, eval_unseen_benchmark.py, eval_fair_benchmark.py,
benchmark_consistency.py, run_canonical_benchmark.py, reproduce_benchmark.py,
eval_expert_15branches.py, eval_expert_branch64.py, run_hw1_cross_cohort_overlay.py.

Usage (via CLI):
    $RUN benchmark --config-file configs/experiment/ssv_dagger_centroid_v9.yml \
        --models "BC:ssv_BC_CENTROID_V9/InstinctJester, DAgger:SSV_DAGGER_CENTROID_V9/InstinctJester" \
        --branches seen --max-trajectories 50 --seed 42 --save-plots --save-analysis

Or directly:
    from sousvide.instruct.benchmark import run_unified_benchmark
"""
import os
import sys
import json
import glob
import time
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch

WORKSPACE = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _get_pkl, _load_all_branches, _preload_all_pkls,
    _preload_bc_trajectories, _swap_model, _make_terminal_fn,
    _evaluate_run, _save_benchmark_plotly,
    _SCENE_CACHE, _BC_TRAJ_CACHE, _PKL_CACHE,
    DEVICE,
)
from sousvide.control.pilot import Pilot
from sousvide.visualize.analyze_simulated_experiments import (
    analyze_trajectory_performance, compute_aggregate_statistics,
)


def evaluate_branches(pilot, model_path, branches_dict, scene_data, label="Eval",
                       output_dir=None, save_plots=False, is_expert=False,
                       collect_annotations=False):
    """Evaluate a pilot (or MPC expert) on explicit branches.

    When collect_annotations=True, wraps the pilot in a RecordingPilot to capture
    (tcr, xcr, xnn) at each step, then relabels offline with MPC expert. This
    combines eval + DAgger collection in a single simulation pass.

    Args:
        pilot: Pilot instance (will be swapped to model_path). Ignored if is_expert=True.
        model_path: path to model.pth to evaluate. Ignored if is_expert=True.
        branches_dict: {obj_name: [(branch_id, tXUi), ...]}
        scene_data: from _get_scene()
        label: label for logging
        output_dir: if set, save plotly + analysis
        save_plots: save plotly HTMLs
        is_expert: if True, use VehicleRateMPC instead of pilot
        collect_annotations: if True, record pilot inputs and relabel with expert offline

    Returns:
        (per_object, diagnostics, overall_sr, overall_cr)
        If collect_annotations=True, returns (per_object, diagnostics, overall_sr, overall_cr, annotations)
    """
    from scipy.spatial import cKDTree
    import yaml

    from figs.control.vehicle_rate_mpc import VehicleRateMPC
    from sousvide.instruct.train_dagger import RecordingPilot, _offline_relabel
    if not is_expert:
        pilot = _swap_model(pilot, model_path)
    simulator = scene_data["simulator"]
    all_annotations = []
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    # Get scene radii for performance analysis
    scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")
    scene_name = None
    for k, v in _SCENE_CACHE.items():
        if v["simulator"] is simulator:
            scene_name = k
            break
    if scene_name:
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            scene_cfg = yaml.safe_load(f)
        scene_radii = scene_cfg.get("radii", [[2.0, 0.4]] * len(queries))
    else:
        scene_radii = [[2.0, 0.4]] * len(queries)

    per_object = {}
    all_diag = []

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        branches = branches_dict[obj_name]
        exclusion_r = float(scene_radii[min(obj_idx, len(scene_radii)-1)][0])
        collision_r = float(scene_radii[min(obj_idx, len(scene_radii)-1)][1])
        successes, collisions, goal_dists = [], [], []
        obj_runs_for_plot = []
        perf_analyses = []

        print(f"\n  [{label}] === {obj_name} ({len(branches)} branches) ===")

        for bi, (br_id, tXUi) in enumerate(branches):
            if is_expert:
                policy = VehicleRateMPC(tXUi, "vrmpc_rrt", "carl", "InstinctJester")
            elif collect_annotations:
                _reset_pilot(pilot)
                recording = RecordingPilot(pilot)
                policy = recording
            else:
                _reset_pilot(pilot)
                policy = pilot
            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )

            t0_sim = time.time()
            result = simulator.simulate(
                policy=policy, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
                x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=obj_name,
                vision_processor=None, verbose=False,
                early_stop_fn=terminal_fn,
            )
            Xro = result[1]
            sim_s = time.time() - t0_sim

            ev = _evaluate_run(Xro, obj_target, epcds_arr,
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
            stop = term_info.get("reason") or "timeout"
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))

            print(f"  [{label}] {bi+1:3d}/{len(branches)}  {status:4s}  br{br_id}"
                  f"  d0={d0:5.1f}m  goal={gd:5.2f}m  steps={ev['n_steps']:3d}"
                  f"  stop={stop:<12s}  ({sim_s:.0f}s)")

            all_diag.append({
                "object": obj_name, "branch_id": br_id,
                "d0": d0, "success": bool(ev["success"]),
                "collision": bool(ev["collision"]),
                "goal_dist": gd, "min_goal_dist": float(ev["min_goal_dist"]),
                "stop_reason": stop, "n_steps": ev["n_steps"],
                "sim_time_s": round(sim_s, 1),
            })

            obj_short = obj_name.replace(" ", "_")[:30]
            perf = analyze_trajectory_performance(
                Xro, np.asarray(obj_target).flatten()[:3],
                epcds_arr, exclusion_r, collision_r,
                trajectory_name=f"{label}_{obj_short}_br{br_id}",
            )
            perf_analyses.append(perf)

            if save_plots:
                obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy(), br_id))

            # Offline expert relabeling (no re-simulation needed)
            if collect_annotations and not is_expert:
                anns = _offline_relabel(recording.records, tXUi)
                all_annotations.extend(anns)
                print(f"    [relabel] {len(anns)} annotations from br{br_id}", flush=True)

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_m = float(np.mean(goal_dists)) if goal_dists else float("nan")
        n_ok = int(sum(successes))
        agg = compute_aggregate_statistics(perf_analyses) if perf_analyses else {}

        print(f"  [{label}] --- '{obj_name[:30]}'  success={sr:.0%} ({n_ok}/{len(branches)})"
              f"  collision={cr:.0%}  avg_goal={gd_m:.2f}m")
        if agg:
            print(f"  [{label}]     norm_dist={agg.get('mean_normalized_distance',0):.3f}"
                  f"  yaw_err={agg.get('mean_yaw_error_degrees',0):.1f}deg"
                  f"  fov={agg.get('mean_fov_pct',0):.0%}"
                  f"  clearance={agg.get('mean_min_clearance',0):.2f}m"
                  f"  steps={agg.get('mean_n_steps',0):.0f}"
                  f"  traj_len={agg.get('mean_trajectory_length',0):.1f}m")

        per_object[obj_name] = {
            "success_rate": sr, "collision_rate": cr, "goal_dist": gd_m,
            "n_eval": len(branches), "n_success": n_ok,
            "performance_analysis": agg,
        }

        if save_plots and obj_runs_for_plot and output_dir:
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            obj_short = obj_name.replace(" ", "_")[:30]
            html_path = os.path.join(plots_dir, f"{label}_{obj_short}.html")
            _save_benchmark_plotly(
                obj_runs=obj_runs_for_plot, obj_target=obj_target,
                simulator=simulator,
                obj_name=f"{label} — {obj_name} ({sr:.0%} success)",
                save_path=html_path,
            )

    overall_sr = float(np.mean([v["success_rate"] for v in per_object.values()])) if per_object else 0.0
    overall_cr = float(np.mean([v["collision_rate"] for v in per_object.values()])) if per_object else 0.0
    if collect_annotations:
        return per_object, all_diag, overall_sr, overall_cr, all_annotations
    return per_object, all_diag, overall_sr, overall_cr


def _reset_pilot(pilot):
    """Reset pilot state between benchmark runs."""
    pilot.hy_flag = False
    pilot.hy_idx = 0
    pilot.DxU.zero_()
    if hasattr(pilot, 'Znn') and isinstance(pilot.Znn, torch.Tensor):
        pilot.Znn.zero_()
    if hasattr(pilot, 'chunk_buf'):
        pilot.chunk_buf = None
        pilot.chunk_step = 0


def _load_validation_branches(bc_cohort, scene_name, scenes_cfg_dir):
    """Load validation-only branches (trajectories_val*.pt)."""
    scene_data = _SCENE_CACHE.get(scene_name)
    if scene_data is None:
        scene_data = _get_scene(scene_name, scenes_cfg_dir)

    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    obj_locs = np.array([np.squeeze(t) for t in obj_targets])

    rollout_dir = os.path.join(WORKSPACE, "cohorts", bc_cohort, "rollout_data", scene_name)
    val_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))

    if not val_files:
        print(f"  [WARN] No validation files in {rollout_dir}")
        return {}

    per_obj = {i: [] for i in range(len(queries))}
    for vf in val_files:
        data = torch.load(vf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or not hasattr(tXUd, "shape") or tXUd.shape[0] != 18:
            continue
        goal = np.array(tXUd[1:4, -1])
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best = int(np.argmin(dists))
        if float(dists[best]) < 5.0:
            per_obj[best].append(tXUd)

    result = {}
    for obj_idx, obj_name in enumerate(queries):
        trajs = per_obj[obj_idx]
        if trajs:
            result[f"{scene_name}_{obj_name}"] = trajs
            print(f"  [Val] '{obj_name}': {len(trajs)} validation branches")
    return result


def _resolve_branches(flights, scenes_cfg_dir, bc_cohort, cohort_path,
                      mode, seed, max_trajectories):
    """
    Resolve which branches to benchmark on.

    mode: "seen" | "unseen" | "both"
    Returns: dict[key, list[np.ndarray]]  where key = "{scene}_{obj}"
    """
    print(f"  [resolve_branches] mode={mode}  seed={seed}  max_traj={max_trajectories}", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    branches = {}

    for scene_name, _ in flights:
        print(f"  [resolve_branches] Loading scene '{scene_name}'...", flush=True)
        scene_data = _get_scene(scene_name, scenes_cfg_dir)
        queries = scene_data["queries"]
        print(f"  [resolve_branches] Scene loaded. Queries: {queries}", flush=True)

        if mode in ("seen", "both"):
            print(f"  [resolve_branches] Loading BC trajectories from '{bc_cohort}'...", flush=True)
            _preload_bc_trajectories(bc_cohort, flights, scenes_cfg_dir)
            print(f"  [resolve_branches] BC trajectories loaded.", flush=True)
            for obj_name in queries:
                key = f"{scene_name}_{obj_name}"
                # Load all branches directly from rollout_data (no pkl needed)
                print(f"  [resolve_branches] Loading branches for '{obj_name}'...", flush=True)
                all_br = _load_all_branches(
                    scene_name, obj_name, cohort_path or "",
                    scenes_cfg_dir, None,
                )
                if not all_br:
                    print(f"  [resolve_branches] WARNING: No branches for '{obj_name}'", flush=True)
                    continue
                n_sample = min(max_trajectories, len(all_br))
                idxs = np.random.choice(len(all_br), size=n_sample, replace=False)
                branches[key] = [(int(i), all_br[i]) for i in idxs]
                print(f"  [Seen] '{obj_name}': {n_sample}/{len(all_br)} branches"
                      f"  ids={list(idxs)}", flush=True)

        if mode in ("unseen", "both"):
            print(f"  [resolve_branches] Loading validation branches...", flush=True)
            val_branches = _load_validation_branches(bc_cohort, scene_name, scenes_cfg_dir)
            for key, trajs in val_branches.items():
                # Tag with val_ prefix for branch IDs
                tagged = [(f"val{i}", t) for i, t in enumerate(trajs)]
                if mode == "both" and key in branches:
                    branches[key + "_unseen"] = tagged
                else:
                    branches[key] = tagged

    print(f"  [resolve_branches] Done. {len(branches)} branch sets resolved.", flush=True)
    return branches


def _parse_model_specs(models_str):
    """
    Parse model specs from CLI string.
    Format: "Label:cohort/pilot_name" or "Label:cohort/pilot_name:model.pth"

    Returns list of dicts: [{label, cohort, pilot_name, model_path}]
    """
    specs = []
    for entry in models_str.split(","):
        entry = entry.strip()
        parts = entry.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid model spec '{entry}'. Use 'Label:cohort/pilot_name'")

        label = parts[0].strip()
        cohort_pilot = parts[1].strip()

        if "/" not in cohort_pilot:
            raise ValueError(f"Model spec '{entry}' missing pilot name. Use 'Label:cohort/pilot_name'")
        cohort, pilot_name = cohort_pilot.rsplit("/", 1)

        if len(parts) >= 3:
            model_file = parts[2].strip()
        else:
            model_file = "model.pth"

        model_path = os.path.join(WORKSPACE, "cohorts", cohort, "roster", pilot_name, model_file)
        if not os.path.exists(model_path):
            print(f"  [WARN] Model not found: {model_path}")

        specs.append({
            "label": label,
            "cohort": cohort,
            "pilot_name": pilot_name,
            "model_path": model_path,
        })
    return specs


def _save_overlay_plotly(model_trajectories, obj_target, simulator, obj_name,
                         scene_name, scenes_cfg_dir, save_path):
    """
    Multi-model overlay plotly: each model's trajectory in a different color.

    model_trajectories: list of {label, positions (Nx3), ev, color}
    """
    import plotly.graph_objects as go
    import figs.tsampling.build_rrt_dataset as bd
    import yaml

    goal_loc = np.asarray(obj_target).flatten()[:3]
    scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")

    # Use cached scene data to avoid redundant bd.get_objectives calls
    from sousvide.instruct.train_dagger import SUCCESS_RADIUS
    cached = _SCENE_CACHE.get(scene_name)
    if cached and "epcds_list" in cached:
        epcds_list = cached["epcds_list"]
    else:
        with open(scene_cfg_file) as f:
            sc = yaml.safe_load(f)
        _, _, epcds_list, _ = bd.get_objectives(
            simulator.gsplat, sc.get("queries", []), sc.get("similarities", None), False)
    radius_info = {"r1": sc.get("radii", [[2.0, 0.4]])[0][0], "r2": SUCCESS_RADIUS}
    fig = bd.visualize_multiple_trajectories([], epcds_list, goal_loc, radius_info, scene_cfg_file, simulator)

    # Success zone circle
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=goal_loc[0] + SUCCESS_RADIUS * np.cos(theta),
        y=goal_loc[1] + SUCCESS_RADIUS * np.sin(theta),
        z=np.full(200, goal_loc[2]),
        mode="lines", line=dict(color="limegreen", width=4, dash="dash"),
        name=f"Success zone (r={SUCCESS_RADIUS}m)"))

    colors = ["red", "orange", "blue", "purple", "cyan", "magenta", "green"]
    for i, mt in enumerate(model_trajectories):
        pos = mt["positions"]
        ev = mt["ev"]
        label = mt["label"]
        color = mt.get("color", colors[i % len(colors)])
        status = "COLL" if ev["collision"] else ("OK" if ev["success"] else "MISS")

        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="lines", line=dict(color=color, width=6),
            name=f"{label} ({status})"))
        end_sym = "circle-open" if ev["success"] else "x"
        fig.add_trace(go.Scatter3d(
            x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
            mode="markers", marker=dict(size=12, color=color, symbol=end_sym),
            showlegend=False))

    # Start marker (same for all)
    if model_trajectories:
        p0 = model_trajectories[0]["positions"][0]
        fig.add_trace(go.Scatter3d(
            x=[p0[0]], y=[p0[1]], z=[p0[2]],
            mode="markers", marker=dict(size=12, color="white", symbol="circle",
                                         line=dict(color="black", width=2)),
            name="Start"))

    fig.update_layout(
        title=f"<b>{obj_name}</b><br>"
              f"<span style='font-size:14px'>"
              + " | ".join(f"{mt['label']}" for mt in model_trajectories)
              + "</span>",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12)))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"  [overlay] -> {save_path}")


def run_unified_benchmark(
    flights: list,
    scenes_cfg_dir: str,
    model_specs: list,
    bc_cohort: str,
    branches_mode: str = "seen",
    max_trajectories: int = 50,
    seeds: list = None,
    save_plots: bool = False,
    save_videos: bool = False,
    save_analysis: bool = True,
    include_expert: bool = False,
    overlay: bool = False,
    output_dir: str = None,
) -> dict:
    """
    Unified benchmark entry point.

    Parameters
    ----------
    flights : list of [scene_name, course_name]
    scenes_cfg_dir : path to configs/scenes/
    model_specs : list of {label, cohort, pilot_name, model_path}
    bc_cohort : BC cohort for loading branches
    branches_mode : "seen" | "unseen" | "both"
    max_trajectories : number of branches per object per model
    seeds : list of seeds (multi-seed consistency mode)
    save_plots : save per-object plotly HTMLs
    save_videos : save MP4 simulation videos
    save_analysis : save JSON results + diagnostics
    include_expert : add VehicleRateMPC as reference model
    overlay : multi-model overlay plotly (requires 2+ models)
    output_dir : output directory (auto-generated if None)
    """
    if seeds is None:
        seeds = [42]

    # Determine output directory
    if output_dir is None:
        primary_cohort = model_specs[0]["cohort"] if model_specs else "benchmark"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(WORKSPACE, "cohorts", primary_cohort, "benchmark_results", ts)
    os.makedirs(output_dir, exist_ok=True)
    if save_plots:
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    if save_videos:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    cohort_path = os.path.join(WORKSPACE, "cohorts", bc_cohort)

    print("=" * 70)
    print("[Benchmark] Unified SINGER Benchmark")
    print(f"  Models:     {[s['label'] for s in model_specs]}")
    print(f"  Branches:   {branches_mode}")
    print(f"  Max traj:   {max_trajectories}/object")
    print(f"  Seeds:      {seeds}")
    print(f"  Expert:     {include_expert}")
    print(f"  Plots:      {save_plots}  Videos: {save_videos}  Analysis: {save_analysis}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    all_seed_results = {}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"[Benchmark] Seed={seed}")
        print(f"{'='*70}")

        # Clear caches for fresh seed
        _BC_TRAJ_CACHE.clear()

        # Resolve branches
        shared_branches = _resolve_branches(
            flights, scenes_cfg_dir, bc_cohort, cohort_path,
            branches_mode, seed, max_trajectories,
        )

        if not shared_branches:
            print("  [WARN] No branches resolved. Check bc_cohort and branches_mode.")
            continue

        # Add expert if requested
        effective_specs = list(model_specs)
        if include_expert:
            effective_specs.insert(0, {
                "label": "Expert",
                "cohort": bc_cohort,
                "pilot_name": "InstinctJester",
                "model_path": None,
                "is_expert": True,
            })

        seed_results = {}
        all_diag = []

        for spec in effective_specs:
            label = spec["label"]
            is_expert = spec.get("is_expert", False)

            print(f"\n[Benchmark] >>> {label}")
            if not is_expert:
                print(f"  cohort={spec['cohort']}  pilot={spec['pilot_name']}  weights={spec['model_path']}")
                pilot = Pilot(spec["cohort"], spec["pilot_name"])
                pilot.set_mode("deploy")
                pilot.model.to(DEVICE)
                pilot = _swap_model(pilot, spec["model_path"])
            else:
                pilot = None
                print(f"  MPC Expert (VehicleRateMPC)")

            label_results = {}

            for scene_name, _ in flights:
                scene_data = _get_scene(scene_name, scenes_cfg_dir)
                simulator = scene_data["simulator"]
                obj_targets = scene_data["obj_targets"]
                queries = scene_data["queries"]

                # Load scene config for radii (needed by performance analysis)
                import yaml
                scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
                with open(scene_cfg_file) as f:
                    scene_cfg = yaml.safe_load(f)
                scene_radii = scene_cfg.get("radii", [[2.0, 0.4]] * len(queries))

                for obj_idx, obj_name in enumerate(queries):
                    key = f"{scene_name}_{obj_name}"
                    # Check both seen and unseen keys
                    branch_key = key if key in shared_branches else f"{key}_unseen"
                    if branch_key not in shared_branches:
                        continue

                    obj_target = obj_targets[obj_idx] if obj_idx < len(obj_targets) else np.zeros(3)
                    branch_list = shared_branches[branch_key]
                    n_runs = len(branch_list)
                    exclusion_radius = float(scene_radii[min(obj_idx, len(scene_radii)-1)][0])
                    collision_radius = float(scene_radii[min(obj_idx, len(scene_radii)-1)][1])

                    from scipy.spatial import cKDTree
                    pc_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
                    pc_tree = cKDTree(pc_arr) if pc_arr.shape[0] > 0 else None

                    successes, collisions, goal_dists = [], [], []
                    obj_runs_for_plot = []
                    overlay_data = []
                    perf_analyses = []

                    for run_i, (branch_id, tXUi) in enumerate(branch_list):
                        x0 = tXUi[1:11, 0].copy()
                        t_start = float(tXUi[0, 0])
                        t_end = float(tXUi[0, -1])

                        terminal_fn, term_info = _make_terminal_fn(
                            obj_target, pc_tree,
                            env_min=scene_data.get("env_min"),
                            env_max=scene_data.get("env_max"),
                        )

                        if is_expert:
                            from figs.control.vehicle_rate_mpc import VehicleRateMPC
                            policy = VehicleRateMPC(tXUi, "vrmpc_rrt", "carl", "expert")
                        else:
                            _reset_pilot(pilot)
                            policy = pilot

                        t0_sim = time.time()
                        result = simulator.simulate(
                            policy=policy, t0=t_start, tf=t_end, x0=x0,
                            obj=np.zeros((18, 1)), query=obj_name,
                            vision_processor=None, verbose=False,
                            early_stop_fn=terminal_fn,
                        )
                        Tro, Xro, Uro, Iro = result[0], result[1], result[2], result[3]
                        sim_s = time.time() - t0_sim

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
                        stop = term_info.get("reason", "timeout")
                        init_d = float(np.linalg.norm(x0[:3] - np.asarray(obj_target).flatten()[:3]))
                        fov_p = ev.get("fov_pct", float("nan"))
                        fov_str = f"{fov_p:.0%}" if not math.isnan(fov_p) else "N/A"

                        traj_id = f"br{branch_id}"
                        print(f"  [{label}] {run_i+1:3d}/{n_runs}  {status:4s}  {traj_id}"
                              f"  d0={init_d:5.1f}m  goal={gd:5.2f}m"
                              f"  steps={ev['n_steps']:3d}  fov={fov_str:>4s}"
                              f"  stop={stop:<12s}  ({sim_s:.0f}s)")

                        all_diag.append({
                            "seed": seed, "model": label, "object": obj_name,
                            "branch_id": branch_id, "traj_id": traj_id,
                            "branch_set": "unseen" if "_unseen" in branch_key else "seen",
                            "run": run_i, "success": bool(ev["success"]),
                            "collision": bool(ev["collision"]),
                            "goal_dist": gd, "min_goal_dist": float(ev["min_goal_dist"]),
                            "n_steps": ev["n_steps"], "sim_time_s": round(sim_s, 1),
                            "fov_pct": float(fov_p) if not math.isnan(fov_p) else None,
                            "d0": init_d, "stop_reason": stop,
                        })

                        if save_plots:
                            obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy(), branch_id))
                        if overlay:
                            overlay_data.append({
                                "label": label,
                                "positions": Xro[:3, :].T,
                                "ev": ev,
                                "branch_id": branch_id,
                            })

                        # Detailed performance analysis
                        obj_short = obj_name.replace(" ", "_")[:30]
                        traj_name = f"{label}_{obj_short}_{traj_id}"
                        perf = analyze_trajectory_performance(
                            Xro, np.asarray(obj_target).flatten()[:3],
                            pc_arr, exclusion_radius, collision_radius,
                            trajectory_name=traj_name,
                        )
                        perf_analyses.append(perf)

                        # Save simulation video
                        if save_videos and Iro is not None:
                            import imageio
                            from torchvision.transforms import Resize
                            transform = Resize((720, 1280), antialias=True)
                            obj_short = obj_name.replace(" ", "_")[:30]
                            for ch_name in ("semantic", "rgb"):
                                if ch_name not in Iro:
                                    continue
                                frames_raw = Iro[ch_name]
                                frames_out = torch.zeros((frames_raw.shape[0], 720, 1280, 3))
                                imgs_t = torch.from_numpy(frames_raw)
                                for fi in range(imgs_t.shape[0] - 1):
                                    img = imgs_t[fi].permute(2, 0, 1)
                                    frames_out[fi] = transform(img).permute(1, 2, 0)
                                vid_path = os.path.join(
                                    output_dir, "videos",
                                    f"{label}_{obj_short}_{traj_id}_{ch_name}.mp4")
                                imageio.mimwrite(vid_path, frames_out.numpy().astype("uint8"), fps=20)
                                print(f"    [video] -> {vid_path}", flush=True)

                    sr = float(np.mean(successes)) if successes else 0.0
                    cr = float(np.mean(collisions)) if collisions else 0.0
                    gd_m = float(np.mean(goal_dists)) if goal_dists else float("nan")
                    n_ok = int(sum(successes))
                    n_coll = int(sum(collisions))

                    print(f"  [{label}] --- '{obj_name[:30]}'  success={sr:.0%} ({n_ok}/{n_runs})"
                          f"  collision={cr:.0%} ({n_coll})  avg_goal={gd_m:.2f}m")

                    # Compute aggregate performance analysis
                    agg_stats = compute_aggregate_statistics(perf_analyses) if perf_analyses else {}
                    if agg_stats:
                        norm_d = agg_stats.get("mean_normalized_distance", float("nan"))
                        yaw_d = agg_stats.get("mean_yaw_error_degrees", float("nan"))
                        fov_sr = agg_stats.get("camera_fov_success_rate", float("nan"))
                        traj_len = agg_stats.get("mean_trajectory_length", float("nan"))
                        print(f"  [{label}]     norm_dist={norm_d:.3f}  yaw_err={yaw_d:.1f}deg"
                              f"  fov_rate={fov_sr:.0%}  traj_len={traj_len:.1f}m", flush=True)

                    label_results[obj_name] = {
                        "success_rate": sr, "collision_rate": cr,
                        "goal_dist": gd_m, "n_eval": n_runs,
                        "n_success": n_ok, "n_collision": n_coll,
                        "performance_analysis": agg_stats,
                    }

                    # Save detailed per-run analysis JSON
                    if save_analysis and perf_analyses:
                        analysis_dir = os.path.join(output_dir, "analysis")
                        os.makedirs(analysis_dir, exist_ok=True)
                        obj_short = obj_name.replace(" ", "_")[:30]
                        analysis_file = os.path.join(analysis_dir, f"{label}_{obj_short}_seed{seed}.json")
                        # Convert numpy arrays to lists for JSON serialization
                        serializable = []
                        for pa in perf_analyses:
                            pa_copy = {}
                            for k, v in pa.items():
                                if isinstance(v, np.ndarray):
                                    pa_copy[k] = v.tolist()
                                elif isinstance(v, (np.floating, np.integer)):
                                    pa_copy[k] = float(v)
                                elif isinstance(v, np.bool_):
                                    pa_copy[k] = bool(v)
                                else:
                                    pa_copy[k] = v
                            serializable.append(pa_copy)
                        with open(analysis_file, "w") as f:
                            json.dump({
                                "model": label, "object": obj_name, "seed": seed,
                                "aggregate": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                              for k, v in agg_stats.items()},
                                "per_run": serializable,
                            }, f, indent=2, default=float)
                        print(f"    [analysis] -> {analysis_file}", flush=True)

                    # Save per-object plotly
                    if save_plots and obj_runs_for_plot:
                        obj_short = obj_name.replace(" ", "_")[:30]
                        html_path = os.path.join(output_dir, "plots", f"{label}_{obj_short}.html")
                        _save_benchmark_plotly(
                            obj_runs=obj_runs_for_plot, obj_target=obj_target,
                            simulator=simulator,
                            obj_name=f"{label} — {obj_name} ({sr:.0%} success)",
                            save_path=html_path,
                        )

            # Aggregate across objects
            all_sr = [v["success_rate"] for v in label_results.values()]
            all_cr = [v["collision_rate"] for v in label_results.values()]
            label_results["__overall__"] = {
                "success_rate": float(np.mean(all_sr)) if all_sr else 0.0,
                "collision_rate": float(np.mean(all_cr)) if all_cr else 0.0,
            }
            seed_results[label] = label_results

            if not is_expert:
                del pilot
                torch.cuda.empty_cache()

        # Multi-model overlay plotly (one per object, all models overlaid)
        if overlay and len(effective_specs) > 1 and save_plots:
            print("\n  [Generating overlay plots...]")
            for scene_name, _ in flights:
                scene_data = _get_scene(scene_name, scenes_cfg_dir)
                simulator = scene_data["simulator"]
                queries = scene_data["queries"]
                obj_targets = scene_data["obj_targets"]

                for obj_idx, obj_name in enumerate(queries):
                    # Collect one representative run per model for overlay
                    overlay_runs = []
                    for d in all_diag:
                        if d["object"] == obj_name and d["run"] == 0 and d["seed"] == seed:
                            overlay_runs.append(d)

                    if len(overlay_runs) >= 2:
                        # Re-run one branch per model for overlay (uses first branch)
                        key = f"{scene_name}_{obj_name}"
                        branch_key = key if key in shared_branches else f"{key}_unseen"
                        if branch_key not in shared_branches:
                            continue
                        tXUi = shared_branches[branch_key][0]
                        obj_target = obj_targets[obj_idx]

                        overlay_trajs = []
                        for spec in effective_specs:
                            is_exp = spec.get("is_expert", False)
                            if is_exp:
                                from figs.control.vehicle_rate_mpc import VehicleRateMPC
                                pol = VehicleRateMPC(tXUi, "vrmpc_rrt", "carl", "expert")
                            else:
                                pol = Pilot(spec["cohort"], spec["pilot_name"])
                                pol.set_mode("deploy")
                                pol.model.to(DEVICE)
                                pol = _swap_model(pol, spec["model_path"])
                                _reset_pilot(pol)

                            pc_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
                            pc_tree_o = cKDTree(pc_arr) if pc_arr.shape[0] > 0 else None
                            tfn, _ = _make_terminal_fn(
                                obj_target, pc_tree_o,
                                env_min=scene_data.get("env_min"),
                                env_max=scene_data.get("env_max"))
                            res = simulator.simulate(
                                policy=pol, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
                                x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)),
                                query=obj_name, vision_processor=None, verbose=False,
                                early_stop_fn=tfn)
                            Xro_o = res[1]
                            ev_o = _evaluate_run(Xro_o, obj_target, pc_arr,
                                                  env_min=scene_data.get("env_min"),
                                                  env_max=scene_data.get("env_max"),
                                                  tXUi=tXUi, idx0=0)
                            overlay_trajs.append({
                                "label": spec["label"],
                                "positions": Xro_o[:3, :].T,
                                "ev": ev_o,
                            })
                            if not is_exp:
                                del pol

                        obj_short = obj_name.replace(" ", "_")[:30]
                        html_path = os.path.join(output_dir, "plots", f"overlay_{obj_short}.html")
                        _save_overlay_plotly(
                            overlay_trajs, obj_target, simulator, obj_name,
                            scene_name, scenes_cfg_dir, html_path)

        # Summary table
        print(f"\n{'='*70}")
        print(f"[Benchmark] Summary (seed={seed})")
        print(f"{'='*70}")
        header = f"  {'Model':25} {'SR':>6} {'CR':>6} {'N':>4}"
        print(header)
        print(f"  {'-'*45}")
        for lbl, res in seed_results.items():
            ov = res.get("__overall__", {})
            sr = ov.get("success_rate", 0)
            cr = ov.get("collision_rate", 0)
            n = sum(v.get("n_eval", 0) for k, v in res.items() if k != "__overall__")
            print(f"  {lbl:25} {sr:>5.0%} {cr:>5.0%} {n:>4}")

        # Per-object breakdown
        print(f"\n  Per-object success rates:")
        obj_names = [k for k in list(seed_results.values())[0].keys() if k != "__overall__"]
        header_models = " ".join(f"{lbl[:12]:>12}" for lbl in seed_results)
        print(f"  {'Object':35} {header_models}")
        for obj_name in obj_names:
            cells = []
            for lbl, res in seed_results.items():
                sr = res.get(obj_name, {}).get("success_rate", float("nan"))
                cells.append(f"{sr:>11.0%}")
            print(f"  {obj_name[:35]:35} " + " ".join(cells))

        all_seed_results[seed] = seed_results

        # Save results
        if save_analysis:
            results_file = os.path.join(output_dir, f"benchmark_results_seed{seed}.json")
            with open(results_file, "w") as f:
                json.dump({
                    "seed": seed,
                    "branches_mode": branches_mode,
                    "max_trajectories": max_trajectories,
                    "models": [s["label"] for s in model_specs],
                    "include_expert": include_expert,
                    "results": seed_results,
                }, f, indent=2, default=float)
            print(f"  Results -> {results_file}")

            diag_file = os.path.join(output_dir, f"diagnostics_seed{seed}.json")
            with open(diag_file, "w") as f:
                json.dump(all_diag, f, indent=2, default=float)
            print(f"  Diagnostics -> {diag_file}")

    # Multi-seed summary
    if len(seeds) > 1 and save_analysis:
        print(f"\n{'='*70}")
        print(f"[Benchmark] Multi-seed summary")
        print(f"{'='*70}")
        for lbl in model_specs:
            label = lbl["label"]
            srs = [all_seed_results[s].get(label, {}).get("__overall__", {}).get("success_rate", 0) for s in seeds]
            print(f"  {label:25} SR: {np.mean(srs):.1%} +/- {np.std(srs):.1%}  ({[f'{s:.0%}' for s in srs]})")

        summary_file = os.path.join(output_dir, "multi_seed_summary.json")
        with open(summary_file, "w") as f:
            json.dump({"seeds": seeds, "results": all_seed_results}, f, indent=2, default=float)
        print(f"  Summary -> {summary_file}")

    print(f"\n{'='*70}")
    print(f"[Benchmark] Done. Output: {output_dir}")
    print(f"{'='*70}")

    return all_seed_results
