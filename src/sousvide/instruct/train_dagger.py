"""
DAgger (Dataset Aggregation) — version optimisée.
"""

import os
import gc
import glob
import json
import pickle
import shutil
import yaml
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.tsampling.build_rrt_dataset as bd
from figs.utilities.trajectory_helper import process_branch

from sousvide.control.pilot import Pilot
import sousvide.instruct.train_policy as tp
from sousvide.flight.deploy_ssv import simulate_rollouts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# PERF : caches globaux — survivent entre pilots ET entre appels benchmark
# ──────────────────────────────────────────────────────────────────────────────

# {scene_name: {"simulator": Simulator, "obj_targets": [...], "epcds_arr": ndarray}}
_SCENE_CACHE: Dict[str, dict] = {}

# {f"{scene_name}_{obj_name}": {"tXUi": ndarray, ...}}
_PKL_CACHE: Dict[str, dict] = {}

# {f"{scene_name}_{obj_name}": [list of (18, N_i) tXUd arrays]}  — from BC rollout data
_BC_TRAJ_CACHE: Dict[str, List[np.ndarray]] = {}

# {f"{scene_name}_{obj_name}": [list of (18, N_i) tXUd arrays]}  — resolved branches
_BRANCHES_CACHE: Dict[str, List[np.ndarray]] = {}


DUMMY_NEVER_REACHED = "DUMMY_NEVER_REACHED"

def _get_scene(
    scene_name: str,
    scenes_cfg_dir: str,
    frame_name: str = "carl",
    rollout_name: str = "baseline",
) -> dict:
    """
    Retourne (depuis cache ou en créant) :
      simulator, obj_targets, epcds_arr
    Le gsplat n'est chargé QU'UNE SEULE FOIS par scène pour tout le run.
    """
    if scene_name in _SCENE_CACHE:
        print(f"  [SceneCache] ♻️  '{scene_name}' depuis cache")
        return _SCENE_CACHE[scene_name]

    print(f"  [SceneCache] 🔄 Chargement gsplat '{scene_name}' rollout='{rollout_name}' frame='{frame_name}'...")
    simulator = Simulator(scene_name, rollout_name)
    simulator.load_frame(frame_name)

    with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
        sc = yaml.safe_load(f)
    objectives_list = sc.get("queries", [])
    similarities    = sc.get("similarities", None)

    obj_targets, _, epcds_list, epcds_arr = bd.get_objectives(
        simulator.gsplat, objectives_list, similarities, False
    )
    env_min = np.array(sc.get("minbound", [-1e6, -1e6, -1e6]), dtype=float)
    env_max = np.array(sc.get("maxbound", [ 1e6,  1e6,  1e6]), dtype=float)
    _SCENE_CACHE[scene_name] = dict(
        simulator=simulator,
        obj_targets=obj_targets,
        epcds_list=epcds_list,
        epcds_arr=epcds_arr,
        queries=objectives_list,
        env_min=env_min,
        env_max=env_max,
    )
    print(f"  [SceneCache] ✅ '{scene_name}' en cache — {len(obj_targets)} targets")
    return _SCENE_CACHE[scene_name]


def _get_pkl(scene_name: str, obj_name: str, scenes_cfg_dir: str) -> Optional[dict]:
    """
    Retourne (depuis cache mémoire ou disque) le pkl RRT pour un objectif.
    Ne relit jamais le fichier deux fois.
    """
    key = f"{scene_name}_{obj_name}"
    if key in _PKL_CACHE:
        return _PKL_CACHE[key]

    # Essayer avec espaces ou underscores
    for name_variant in [obj_name, obj_name.replace("_", " ")]:
        pkl_path = os.path.join(scenes_cfg_dir, f"{scene_name}_{name_variant}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if "tXUi" not in data:
                print(f"  [PKLCache] ⚠️  tXUi absent dans {pkl_path}")
                return None
            _PKL_CACHE[key] = data
            print(f"  [PKLCache] ✅ {key}  tXUi={data['tXUi'].shape}")
            return data

    print(f"  [PKLCache] ⚠️  Introuvable : {scene_name}_{obj_name}.pkl")
    return None


def _preload_all_pkls(flights: List[Tuple[str, str]], scenes_cfg_dir: str) -> int:
    """Précharge tous les pkl en mémoire. Retourne le nombre chargés."""
    n = 0
    for scene_name, _ in flights:
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            sc = yaml.safe_load(f)
        for obj_name in sc.get("queries", []):
            data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if data is not None:
                n += 1
    return n


def _preload_bc_trajectories(
    bc_cohort_name: str,
    flights: List[Tuple[str, str]],
    scenes_cfg_dir: str,
) -> int:
    """
    Load ALL tXUd from BC rollout data and assign to objects by goal proximity.
    Populates _BC_TRAJ_CACHE and _PKL_CACHE (for backwards compat with _get_pkl).
    Returns number of trajectories loaded.
    """
    workspace_path = str(Path(__file__).resolve().parents[3])
    total = 0

    for scene_name, _ in flights:
        # Get object targets from scene cache (must be preloaded)
        scene_data = _SCENE_CACHE.get(scene_name)
        if scene_data is None:
            print(f"  [BC-Traj] ⚠️ Scene '{scene_name}' not in cache — skip")
            continue
        obj_targets = scene_data["obj_targets"]  # list of (3,) arrays
        queries = scene_data["queries"]

        if not queries or not obj_targets:
            continue

        # Load all trajectory files
        rollout_dir = os.path.join(
            workspace_path, "cohorts", bc_cohort_name,
            "rollout_data", scene_name,
        )
        traj_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories*.pt")))
        if not traj_files:
            print(f"  [BC-Traj] ⚠️ No trajectory files in {rollout_dir}")
            continue

        # Collect all tXUd and match to nearest object
        # obj_targets may be (1,3) or (3,) — squeeze to (3,) each
        obj_locs = np.array([np.squeeze(t) for t in obj_targets])  # (n_obj, 3)
        per_obj: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(queries))}

        for tf in traj_files:
            data = torch.load(tf, map_location="cpu", weights_only=False)
            tXUd = data.get("tXUd")
            if tXUd is None or not hasattr(tXUd, "shape") or tXUd.shape[0] != 18:
                continue
            # Goal = last position in trajectory
            goal = tXUd[1:4, -1]
            dists = np.linalg.norm(obj_locs - goal, axis=1)
            best_obj = int(np.argmin(dists))
            if float(dists[best_obj]) < 5.0:  # within 5m (objects are far apart; goal ≠ centroid)
                per_obj[best_obj].append(tXUd)

        for obj_idx, obj_name in enumerate(queries):
            key = f"{scene_name}_{obj_name}"
            trajs = per_obj[obj_idx]
            if not trajs:
                continue
            _BC_TRAJ_CACHE[key] = trajs
            total += len(trajs)

            # Also populate _PKL_CACHE so _get_pkl() works without pkl files
            if key not in _PKL_CACHE:
                _PKL_CACHE[key] = {
                    "tXUi": trajs[0],
                    "obj_loc": trajs[0][1:4, -1],
                }

            print(f"  [BC-Traj] ✅ '{obj_name}': {len(trajs)} branches from BC rollout data")

    return total


# ── Consistent success evaluation (matches BC analysis pipeline) ───────────

EXCLUSION_RADIUS = 2.0       # r1: exclusion zone radius from scene config
COLLISION_RADIUS = 0.15      # r2: collision detection radius
# Max's soft_success_radius = r1 + 2*r2 = 2.0 + 0.3 = 2.3m
SUCCESS_RADIUS   = EXCLUSION_RADIUS + 2 * COLLISION_RADIUS  # 2.3m — matches Max's analysis
HORIZONTAL_FOV   = np.radians(85)  # 85° camera FOV (logged, not enforced)


def _make_terminal_fn(goal, pc_tree, env_min=None, env_max=None):
    """Build an early-stop callback for the simulator.

    Stops simulation when the drone:
      (a) enters the goal zone (≤ SUCCESS_RADIUS)  → terminal success
      (b) collides with point cloud (clearance < COLLISION_RADIUS) → terminal collision
      (c) goes out of bounds → terminal collision

    Args:
        goal: 3D goal position (array-like)
        pc_tree: scipy.spatial.cKDTree of scene point cloud (or None)
        env_min/max: scene boundaries (or None)

    Returns:
        early_stop_fn(xcr, k) → bool
        terminal_info dict (mutated in-place by callback with terminal reason)
    """
    goal_flat = np.asarray(goal).flatten()[:3]
    env_min_arr = np.asarray(env_min).flatten()[:3] if env_min is not None else None
    env_max_arr = np.asarray(env_max).flatten()[:3] if env_max is not None else None
    info = {"reason": None, "step": -1}

    def _terminal_check(xcr, k):
        pos = xcr[:3]
        # Success: goal zone AND goal in FOV
        goal_dist = float(np.linalg.norm(pos - goal_flat))
        if goal_dist <= SUCCESS_RADIUS:
            # Check FOV: is goal within camera horizontal FOV?
            dx = goal_flat[0] - pos[0]
            dy = goal_flat[1] - pos[1]
            req_yaw = np.arctan2(dy, dx)
            qx, qy, qz, qw = xcr[6], xcr[7], xcr[8], xcr[9]
            act_yaw = np.arctan2(2 * (qw * qz + qx * qy),
                                 1 - 2 * (qy**2 + qz**2))
            yaw_err = abs(act_yaw - req_yaw)
            if yaw_err > np.pi:
                yaw_err = 2 * np.pi - yaw_err
            if yaw_err <= (HORIZONTAL_FOV / 2):
                info["reason"] = "success"
                info["step"] = k
                return True
        if pc_tree is not None:
            d, _ = pc_tree.query(pos, k=1)
            if d < COLLISION_RADIUS:
                info["reason"] = "collision_pc"
                info["step"] = k
                return True
        if env_min_arr is not None and env_max_arr is not None:
            if np.any(pos < env_min_arr) or np.any(pos > env_max_arr):
                info["reason"] = "collision_oob"
                info["step"] = k
                return True
        return False

    return _terminal_check, info


def _evaluate_run(
    Xro: np.ndarray,
    obj_target: np.ndarray,
    pc_bench: np.ndarray,
    success_radius: float = SUCCESS_RADIUS,
    collision_radius: float = COLLISION_RADIUS,
    check_fov: bool = True,
    env_min: np.ndarray = None,
    env_max: np.ndarray = None,
    tXUi: np.ndarray = None,
    idx0: int = 0,
) -> dict:
    """
    Evaluate a single simulation run.

    Success criteria (per Max's analyze_simulated_experiments.py):
      1. Drone enters within success_radius (r1+2*r2=2.3m) of obj_target (centroid)
      2. No collision BEFORE first entry into the goal zone
      FOV is computed and logged but NOT part of success (matches Max's code).

    Collision is detected via:
      a. Point-cloud proximity (< collision_radius from any Gaussian center)
      b. Out-of-bounds (drone exits env_min/env_max scene boundaries)

    Returns dict with: success, collision, collided_before_goal, goal_dist,
                       min_goal_dist, first_entry_step, goal_in_fov, out_of_bounds
    """
    n_steps = Xro.shape[1]
    positions = Xro[:3, :].T  # (N, 3)

    # Ensure obj_target is a flat (3,) array
    obj_target = np.asarray(obj_target).flatten()[:3]

    # Distance to goal at each timestep
    goal_dists = np.linalg.norm(positions - obj_target, axis=1)
    min_goal_dist = float(goal_dists.min())
    min_goal_step = int(np.argmin(goal_dists))
    final_goal_dist = float(goal_dists[-1])

    # First entry into goal zone (any timestep)
    in_zone = goal_dists <= success_radius
    reached_goal = bool(in_zone.any())
    first_entry_step = int(np.argmax(in_zone)) if reached_goal else -1

    # ── Collision detection ──

    # (a) Point-cloud proximity + clearance
    collided_pc = False
    collision_step = n_steps
    min_clearance = float('inf')
    min_clearance_step = -1
    clearance_array = None  # per-step clearance for forensics
    if pc_bench.shape[0] > 0:
        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
        pc_t  = torch.from_numpy(pc_bench).float().to(DEVICE)
        dists_to_pc = torch.cdist(Xro_t, pc_t)  # (N, M)
        # Min clearance per timestep (distance to nearest obstacle)
        per_step_clearance = dists_to_pc.min(dim=1).values  # (N,)
        clearance_array = per_step_clearance.cpu().numpy()
        min_clearance = float(per_step_clearance.min())
        min_clearance_step = int(per_step_clearance.argmin())
        collision_mask = (dists_to_pc < collision_radius).any(dim=1)  # (N,)
        if collision_mask.any():
            collided_pc = True
            collision_step = int(collision_mask.nonzero()[0][0])
        del Xro_t, pc_t, dists_to_pc

    # (b) Out-of-bounds (drone exits scene boundaries)
    out_of_bounds = False
    oob_step = n_steps
    if env_min is not None and env_max is not None:
        env_min_arr = np.asarray(env_min).flatten()[:3]
        env_max_arr = np.asarray(env_max).flatten()[:3]
        below = (positions < env_min_arr).any(axis=1)
        above = (positions > env_max_arr).any(axis=1)
        oob_mask = below | above
        if oob_mask.any():
            out_of_bounds = True
            oob_step = int(np.argmax(oob_mask))

    # Combined collision: point-cloud OR out-of-bounds
    collided = collided_pc or out_of_bounds
    if collided:
        collision_step = min(
            collision_step if collided_pc else n_steps,
            oob_step if out_of_bounds else n_steps,
        )

    # Collision BEFORE reaching goal zone (what matters for success)
    collided_before_goal = collided and (
        not reached_goal or collision_step < first_entry_step
    )

    # FOV check at the moment of first entry to goal zone
    goal_in_fov = False
    if reached_goal and not collided_before_goal and check_fov:
        check_step = first_entry_step
        cam_pos = Xro[:3, check_step]
        # Xro state layout: [x,y,z, vx,vy,vz, qx,qy,qz,qw, ...] = 10+ rows
        if Xro.shape[0] >= 10:
            quat = Xro[6:10, check_step]  # [qx, qy, qz, qw]
            dx = obj_target[0] - cam_pos[0]
            dy = obj_target[1] - cam_pos[1]
            required_yaw = np.arctan2(dy, dx)
            # Drone yaw from quaternion — same formula as BC analysis pipeline
            qx, qy, qz, qw = quat
            actual_yaw = np.arctan2(2 * (qw * qz + qx * qy),
                                    1 - 2 * (qy**2 + qz**2))
            yaw_error = abs(actual_yaw - required_yaw)
            if yaw_error > np.pi:
                yaw_error = 2 * np.pi - yaw_error
            goal_in_fov = yaw_error <= (HORIZONTAL_FOV / 2)
        else:
            goal_in_fov = True
    elif reached_goal and not collided_before_goal:
        goal_in_fov = True  # skip FOV check if disabled

    # Success = first entry into goal zone + no collision before entry
    # (FOV is logged but not enforced, matching Max's analyze_simulated_experiments.py)
    success = reached_goal and not collided_before_goal

    # ── Deviation analysis (cheap post-processing if tXUi provided) ──
    # Only measure deviation UNTIL first entry into success zone.
    # After the drone reaches the goal, divergence from reference is expected.
    mean_pos_dev = float('nan')
    mean_orient_dev_deg = float('nan')
    fov_pct = float('nan')
    if tXUi is not None and Xro.shape[0] >= 10 and tXUi.shape[0] >= 11:
        T_ref = min(n_steps, tXUi.shape[1] - idx0)
        # Cutoff at first entry into success zone (deviation after is irrelevant)
        T_dev = first_entry_step if (reached_goal and first_entry_step > 0) else T_ref
        T_dev = min(T_dev, T_ref)
        if T_dev > 0:
            # Position deviation: ||Xro[0:3,t] - tXUi[1:4, idx0+t]||
            ref_pos = tXUi[1:4, idx0:idx0 + T_dev]  # (3, T_dev)
            act_pos = Xro[:3, :T_dev]                # (3, T_dev)
            pos_devs = np.linalg.norm(act_pos - ref_pos, axis=0)  # (T_dev,)
            mean_pos_dev = float(np.mean(pos_devs))

            # Orientation deviation: quaternion angle between Xro[6:10] and tXUi[7:11]
            ref_quat = tXUi[7:11, idx0:idx0 + T_dev]  # (4, T_dev)
            act_quat = Xro[6:10, :T_dev]               # (4, T_dev)
            dots = np.clip(np.abs(np.sum(act_quat * ref_quat, axis=0)), 0.0, 1.0)
            orient_devs_deg = np.degrees(2.0 * np.arccos(dots))
            mean_orient_dev_deg = float(np.mean(orient_devs_deg))

            # FOV percentage: what fraction of timesteps have goal in camera FOV
            fov_count = 0
            for t in range(T_dev):
                q = Xro[6:10, t]
                qx, qy, qz, qw = q
                cam_pos_t = Xro[:3, t]
                dx_t = obj_target[0] - cam_pos_t[0]
                dy_t = obj_target[1] - cam_pos_t[1]
                req_yaw = np.arctan2(dy_t, dx_t)
                act_yaw = np.arctan2(2 * (qw * qz + qx * qy),
                                     1 - 2 * (qy**2 + qz**2))
                yaw_err = abs(act_yaw - req_yaw)
                if yaw_err > np.pi:
                    yaw_err = 2 * np.pi - yaw_err
                if yaw_err <= (HORIZONTAL_FOV / 2):
                    fov_count += 1
            fov_pct = fov_count / max(T_dev, 1)

    # ── Collision diagnostics ──
    collision_type = "none"
    collision_position = None
    collision_dist_to_goal = float('nan')
    actual_collision_step = -1
    if collided:
        actual_collision_step = collision_step if collision_step < n_steps else -1
        if collided_pc and (not out_of_bounds or collision_step <= oob_step):
            collision_type = "pc"
        else:
            collision_type = "oob"
        if actual_collision_step >= 0 and actual_collision_step < n_steps:
            collision_position = Xro[:3, actual_collision_step].tolist()
            collision_dist_to_goal = float(goal_dists[actual_collision_step])

    return {
        "success":              success,
        "collision":            collided_before_goal,
        "collided_before_goal": collided_before_goal,
        "goal_dist":            final_goal_dist,
        "min_goal_dist":        min_goal_dist,
        "min_goal_step":        min_goal_step,
        "first_entry_step":     first_entry_step,
        "goal_in_fov":          goal_in_fov,
        "out_of_bounds":        out_of_bounds,
        "total_reward":         -final_goal_dist,
        "mean_pos_dev":         mean_pos_dev,
        "mean_orient_dev_deg":  mean_orient_dev_deg,
        "fov_pct":              fov_pct,
        "n_steps":              n_steps,
        "collision_step":       actual_collision_step,
        "collision_type":       collision_type,
        "collision_position":   collision_position,
        "collision_dist_to_goal": collision_dist_to_goal,
        "min_clearance":        min_clearance,
        "min_clearance_step":   min_clearance_step,
        "clearance_array":      clearance_array,  # numpy array (N,) or None
    }


# ── Multi-branch cache: all parameterized branches per object ──────────────
_BRANCHES_CACHE: Dict[str, List[np.ndarray]] = {}


def _load_all_branches(
    scene_name: str,
    obj_name: str,
    cohort_path: str,
    scenes_cfg_dir: str,
    fallback_tXUi: np.ndarray,
) -> List[np.ndarray]:
    """
    Load all parameterized branches (18×N tXUi arrays) for an object.
    Sources, in priority order:
      1. BC rollout trajectories (_BC_TRAJ_CACHE) — authoritative source
      2. Filtered branch pkl from simulation_data/*/rrt_planning/ (raw waypoints → parameterize)
      3. Fallback: just the single branch from configs/scenes/ pkl

    Returns a list of (18, N_i) arrays.  Results are cached in _BRANCHES_CACHE.
    """
    key = f"{scene_name}_{obj_name}"
    if key in _BRANCHES_CACHE:
        return _BRANCHES_CACHE[key]

    # Priority 1: BC rollout trajectories (authoritative — same data BC trained on)
    if key in _BC_TRAJ_CACHE:
        branches_tXUi = _BC_TRAJ_CACHE[key]
        print(f"  [Branches] ✅ {len(branches_tXUi)} branches from BC rollout data for '{obj_name}'")
        _BRANCHES_CACHE[key] = branches_tXUi
        return branches_tXUi

    # Priority 2: filtered pkl with all branches (raw waypoints)
    branches_tXUi = []
    sim_data_root = os.path.join(cohort_path, "simulation_data")
    if os.path.isdir(sim_data_root):
        # Find most recent rrt_planning dir
        rrt_dirs = sorted(glob.glob(os.path.join(sim_data_root, "*/rrt_planning")))
        for rrt_dir in reversed(rrt_dirs):
            filtered_pkl = os.path.join(rrt_dir, f"{scene_name}_filtered_{obj_name}.pkl")
            if os.path.isfile(filtered_pkl):
                with open(filtered_pkl, "rb") as f:
                    raw_branches = pickle.load(f)
                if isinstance(raw_branches, list) and len(raw_branches) > 0:
                    # Load scene config for parameterization
                    scene_cfg_path = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
                    with open(scene_cfg_path) as f:
                        scene_cfg = yaml.safe_load(f)
                    queries = scene_cfg.get("queries", [])
                    obj_idx = queries.index(obj_name) if obj_name in queries else 0
                    altitudes = scene_cfg.get("altitudes", [-1.0])
                    alt = float(altitudes[min(obj_idx, len(altitudes) - 1)])
                    obj_loc = fallback_tXUi[1:4, -1]  # goal from reference

                    print(f"  [Branches] Parameterizing {len(raw_branches)} branches for '{obj_name}'...")
                    for br_idx, waypoints in enumerate(raw_branches):
                        wps = np.array(waypoints)
                        if wps.ndim != 2 or wps.shape[1] < 2:
                            continue
                        # waypoints are (N, 3) — already 3D with altitude set
                        try:
                            tXUi_br, _, _ = process_branch(
                                branch_id=br_idx,
                                positions=wps.tolist(),
                                dt=1.0 / 20,
                                constant_velocity=1.5,
                                obj_loc=obj_loc,
                                pad_t=2,
                                viz=False,
                                threshold_distance=1.5,
                            )
                            if tXUi_br is not None:
                                # Copy motor values from reference
                                tXUi_br[14:18, :] = fallback_tXUi[14:18, 0:1]
                                branches_tXUi.append(tXUi_br)
                        except Exception:
                            continue
                    if branches_tXUi:
                        print(f"  [Branches] ✅ {len(branches_tXUi)}/{len(raw_branches)} branches parameterized for '{obj_name}'")
                break

    if not branches_tXUi:
        # Fallback: single branch from configs/scenes pkl
        branches_tXUi = [fallback_tXUi]
        print(f"  [Branches] ⚠️ Using single fallback branch for '{obj_name}'")

    _BRANCHES_CACHE[key] = branches_tXUi
    return branches_tXUi


def _save_benchmark_plotly(
    obj_runs: list,
    obj_target: np.ndarray,
    simulator,
    obj_name: str,
    save_path: str,
    reference_branches: list = None,
) -> None:
    """Save interactive Plotly HTML using canonical bd.visualize_multiple_trajectories style.

    Args:
        obj_runs: list of (Xro, ev_dict, tXUi) tuples for this object.
        obj_target: 3D target position.
        simulator: FiGS simulator (for point cloud / scene context).
        obj_name: object name for title.
        save_path: output .html path.
        reference_branches: unused (kept for API compat).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        import plotly.graph_objects as go
        from scipy.spatial import cKDTree

        goal_loc = np.asarray(obj_target).flatten()[:3]

        # Use cached scene data (avoids redundant bd.get_objectives calls)
        workspace = str(Path(__file__).resolve().parents[3])
        scenes_cfg_dir = os.path.join(workspace, "configs", "scenes")
        scene_name = None
        for key, sd in _SCENE_CACHE.items():
            if sd["simulator"] is simulator:
                scene_name = key
                break
        if scene_name is None:
            scene_name = "flightroom_ssv_exp"

        scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
        cached = _SCENE_CACHE.get(scene_name)
        if cached and "epcds_list" in cached:
            epcds_list = cached["epcds_list"]
        else:
            with open(scene_cfg_file) as _f:
                sc = yaml.safe_load(_f)
            _, _, epcds_list, _ = bd.get_objectives(
                simulator.gsplat, sc.get("queries", []), sc.get("similarities", None), False
            )
        radius_info = {"r1": 2.0, "r2": SUCCESS_RADIUS}

        fig = bd.visualize_multiple_trajectories(
            [], epcds_list, goal_loc, radius_info, scene_cfg_file, simulator
        )

        # Success zone
        theta = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter3d(
            x=goal_loc[0] + SUCCESS_RADIUS * np.cos(theta),
            y=goal_loc[1] + SUCCESS_RADIUS * np.sin(theta),
            z=np.full(200, goal_loc[2]),
            mode="lines", line=dict(color="green", width=3, dash="dash"),
            name=f"Success Zone (r={SUCCESS_RADIUS}m)",
        ))

        colors = ["red", "orange", "purple", "magenta", "brown",
                  "cyan", "pink", "olive", "teal", "navy"]
        for run_i, run_data in enumerate(obj_runs):
            # Support both (Xro, ev, tXUi) and (Xro, ev, tXUi, branch_id)
            if len(run_data) == 4:
                Xro, ev, tXUi, br_id = run_data
            else:
                Xro, ev, tXUi = run_data
                br_id = run_i
            ref_pos = tXUi[1:4, :].T
            pilot_pos = Xro[:3, :].T
            clr = colors[run_i % len(colors)]
            status = "COLL" if ev["collision"] else ("OK" if ev["success"] else "MISS")
            dev = ev.get("mean_pos_dev", float("nan"))
            traj_label = f"br{br_id}"

            # Expert reference (blue, thin)
            fig.add_trace(go.Scatter3d(
                x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
                mode="lines", line=dict(color="blue", width=3),
                name=f"Expert {traj_label}" if run_i == 0 else None,
                showlegend=(run_i == 0),
                legendgroup=f"expert_{br_id}",
                visible=True,
            ))
            # Pilot trajectory (toggleable per branch)
            fig.add_trace(go.Scatter3d(
                x=pilot_pos[:, 0], y=pilot_pos[:, 1], z=pilot_pos[:, 2],
                mode="lines", line=dict(color=clr, width=4),
                name=f"{traj_label} ({status}, dev={dev:.2f}m)",
                legendgroup=f"pilot_{br_id}",
                visible=True,
            ))
            # End marker
            fig.add_trace(go.Scatter3d(
                x=[pilot_pos[-1, 0]], y=[pilot_pos[-1, 1]], z=[pilot_pos[-1, 2]],
                mode="markers",
                marker=dict(size=10, color="red" if ev["collision"] else clr,
                            symbol="x" if ev["collision"] else "circle-open"),
                showlegend=False,
                legendgroup=f"pilot_{br_id}",
            ))

        n_ok = sum(1 for rd in obj_runs if (rd[1] if len(rd) == 3 else rd[1])["success"])
        fig.update_layout(
            title=f"<b>{obj_name}</b><br>"
                  f"<span style='font-size:14px'>"
                  f"{n_ok}/{len(obj_runs)} success</span>",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                        font=dict(size=11)),
        )
        fig.write_html(save_path)
        print(f"  [plotly] -> {save_path}")
    except Exception as e:
        print(f"  [plotly] WARNING: could not save plot: {e}")


def _save_traj_plot(Tro: np.ndarray, Xro: np.ndarray, Uro, save_path: str, title: str = "") -> None:
    """Save spatial + time trajectory plots to disk. Non-blocking, closes figures after saving."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        # Align Tro and Xro to shortest (fencepost: Tro may be Nctl+1, Xro Nctl)
        T = min(Tro.shape[0], Xro.shape[1])
        Tro = Tro[:T]
        Xro = Xro[:, :T]

        fig = plt.figure(figsize=(14, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot(Xro[0], Xro[1], Xro[2], "b-", linewidth=1.2)
        ax3d.scatter(*Xro[:3, 0], c="g", s=40, zorder=5, label="start")
        ax3d.scatter(*Xro[:3, -1], c="r", s=40, zorder=5, label="end")
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title(title or save_path)
        ax3d.legend(fontsize=7)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(Tro, Xro[0], label="x"); ax2.plot(Tro, Xro[1], label="y"); ax2.plot(Tro, Xro[2], label="z")
        ax2.set_xlabel("t (s)"); ax2.set_ylabel("position (m)")
        ax2.legend(fontsize=7); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=90)
        plt.close(fig)
        print(f"  [plot] → {save_path}")
    except Exception as e:
        print(f"  [plot] WARNING: could not save plot: {e}")
        plt.close("all")



def _swap_model(pilot: Pilot, model_path: str) -> Pilot:
    pilot.model = torch.load(model_path, map_location=DEVICE)
    pilot.model.to(DEVICE)
    pilot.model.eval()
    print(f"  [swap] {Path(model_path).name} → {DEVICE}")
    return pilot


def _save_model_checkpoint(pilot: Pilot, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    model_cpu = pilot.model.cpu()
    torch.save(model_cpu, dst_path)
    pilot.model.to(DEVICE)
    print(f"  [ckpt] Sauvegardé → {dst_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PERF : _generate_rrt_backup — utilise _PKL_CACHE, 0 gsplat reload
# ──────────────────────────────────────────────────────────────────────────────


def _aggregate_dagger_dataset(
    all_annotations: List[dict], existing_file: Optional[str],
    aggregate: bool = True,
) -> List[dict]:
    """
    If aggregate=True (classic DAgger): accumulate all past annotations.
    If aggregate=False (online DAgger): use ONLY current iteration's annotations.
    Online mode prevents catastrophic forgetting when all starting states are fixed,
    since accumulation just adds more copies of the same 8 reference states.
    """
    if aggregate and existing_file and os.path.exists(existing_file):
        return torch.load(existing_file) + all_annotations
    return all_annotations


def _retrain_commander(
    cohort_name: str, pilot_name: str,
    aggregated_file: str, Nep: int, lim_sv: int,
    default_mass: float = 0.3, default_fn: float = 0.3,
    lr: float = 1e-4,
    bc_cohort_name: str = None,
    dagger_only: bool = False,
    oversample: int = 1,
    freeze_vision: bool = True,
) -> None:
    """
    Convert DAgger annotations to BC observation format and fine-tune the Commander.

    Modes:
      - dagger_only=True:  Train ONLY on DAgger annotations (fast, focused corrections)
      - dagger_only=False: Train on BC + DAgger mixed data (slow, preserves BC distribution)

    oversample: Duplicate DAgger annotations N times to increase their weight.

    freeze_vision: If True (default), freeze VisionMLP during retraining — only update
      CommanderSV weights. This preserves the semantic object discrimination learned
      during BC training. Without this, DAgger data overwhelms BC data and the model
      "forgets" which object to navigate to (catastrophic forgetting of visual features).

    The DAgger aggregated file is a flat list of step-level dicts:
        {"xnn": {...}, "x": ndarray, "u": ndarray, "t": float, "query": ndarray}

    The BC observation format expected by generate_dataset / extract_data is:
        {"data": [{"Xnn": [...], "Ynn": [...], "Ndata": int, ...}], "set": "", ...}
    """
    workspace_path = str(Path(__file__).resolve().parents[3])
    annotations = torch.load(aggregated_file, weights_only=False)

    if not annotations:
        print("  [retrain] No annotations — skipping retraining.")
        return

    Xnn, Ynn = [], []
    default_mfn = np.array([default_mass, default_fn], dtype=np.float32)

    for ann in annotations:
        xnn = ann.get("xnn")
        if not xnn:
            continue
        ynn = {
            "unn": np.array(ann["u"], dtype=np.float32),
            "mfn": default_mfn.copy(),
            "onn": np.array(ann["x"], dtype=np.float32),
        }
        Xnn.append(xnn)
        Ynn.append(ynn)

    if not Xnn:
        print("  [retrain] No valid xnn entries in annotations — skipping.")
        return

    # Oversample DAgger annotations to increase their influence
    if oversample > 1:
        Xnn_orig, Ynn_orig = Xnn, Ynn
        Xnn = Xnn_orig * oversample
        Ynn = Ynn_orig * oversample
        print(f"  [retrain] {len(Xnn_orig)} annotations × {oversample} = {len(Xnn)} samples")
    else:
        print(f"  [retrain] {len(Xnn)} annotation samples")

    obs_data = {
        "data": [{
            "Xnn": Xnn, "Ynn": Ynn, "Ndata": len(Xnn),
            "rollout_id": 0, "course": "dagger",
            "frame": {"mass": default_mass, "force_normalized": default_fn},
        }],
        "set": "", "Nobs": len(Xnn), "course": "dagger",
    }

    # Save inside a "dagger" course dir so get_data_paths() picks it up
    course_dir = os.path.join(
        workspace_path, "cohorts", cohort_name,
        "observation_data", pilot_name, "dagger",
    )
    os.makedirs(course_dir, exist_ok=True)
    dst = os.path.join(course_dir, "observations_dagger.pt")
    torch.save(obs_data, dst)
    print(f"  [retrain] {len(Xnn)} DAgger samples → {dst}")

    if dagger_only:
        course = "dagger"
        print(f"  [retrain] DAgger-only mode: {len(Xnn)} samples, {Nep} epochs, lr={lr}")
    else:
        course = None   # all courses (BC + DAgger mixed)
        # Symlink BC observation course dirs into DAgger cohort so that
        # course_name=None picks up both BC data and DAgger data for mixed training.
        if bc_cohort_name:
            bc_obs_base = os.path.join(
                workspace_path, "cohorts", bc_cohort_name,
                "observation_data", pilot_name,
            )
            dag_obs_base = os.path.join(
                workspace_path, "cohorts", cohort_name,
                "observation_data", pilot_name,
            )
            if os.path.isdir(bc_obs_base):
                for entry in os.scandir(bc_obs_base):
                    if not entry.is_dir():
                        continue
                    if entry.name == "dagger":
                        continue  # skip BC's own dagger dir — use ours
                    link_path = os.path.join(dag_obs_base, entry.name)
                    if not os.path.exists(link_path):
                        os.symlink(entry.path, link_path)
                        print(f"  [retrain] symlink BC data: {entry.name} → {entry.path}")
                    else:
                        print(f"  [retrain] BC symlink already exists: {link_path}")
            else:
                print(f"  [retrain] WARNING: bc_cohort_name={bc_cohort_name} not found at {bc_obs_base}")
        print(f"  [retrain] Mixed BC+DAgger mode: {Nep} epochs, lr={lr}")

    # ── Freeze VisionMLP during DAgger retraining ──
    # By default, Commander unlock list includes both CommanderSV and VisionMLP
    # (svnet.py line 96). DAgger data overwhelms BC data, causing the VisionMLP
    # to forget semantic object discrimination → drone navigates to wrong objects.
    # Fix: create Pilot directly, patch its unlock list, call train_student.
    from sousvide.control.pilot import Pilot as _Pilot
    student = _Pilot(cohort_name, pilot_name)
    student.set_mode('train')

    if freeze_vision and hasattr(student.model, 'get_network') and "Commander" in student.model.get_network:
        # Patch unlock list: only CommanderSV, NOT VisionMLP
        original_unlock = student.model.get_network["Commander"]["Unlock"]
        import torch.nn as nn
        student.model.get_network["Commander"]["Unlock"] = nn.ModuleList([student.model.network["CommanderSV"]])
        n_frozen = sum(p.numel() for p in student.model.network["VisionMLP"].parameters())
        n_unlocked = sum(p.numel() for p in student.model.network["CommanderSV"].parameters())
        print(f"  [retrain] VisionMLP FROZEN ({n_frozen:,} params) — only CommanderSV ({n_unlocked:,} params) updated")

    tp.train_student(cohort_name, student, "Commander", Nep,
                     lim_sv=lim_sv, lr=lr, batch_size=64)







def _wandb_log_round(pilot_name, round_i, coll_sr, coll_cr, eval_sr, eval_cr,
                      n_annotations, eval_per_object):
    """Log DAgger round metrics to wandb."""
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log({
            f"dagger/{pilot_name}/round": round_i,
            f"dagger/{pilot_name}/collection_sr": coll_sr,
            f"dagger/{pilot_name}/collection_cr": coll_cr,
            f"dagger/{pilot_name}/eval_sr": eval_sr,
            f"dagger/{pilot_name}/eval_cr": eval_cr,
            f"dagger/{pilot_name}/n_annotations": n_annotations,
        })
        for obj_name, stats in eval_per_object.items():
            obj_short = obj_name.replace(" ", "_")[:20]
            wandb.log({
                f"dagger/{pilot_name}/{obj_short}/sr": stats["success_rate"],
                f"dagger/{pilot_name}/{obj_short}/cr": stats["collision_rate"],
                f"dagger/{pilot_name}/{obj_short}/gd": stats["goal_dist"],
            })
    except Exception as e:
        print(f"  [WARN] wandb: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# DAgger Policy — pure learner rollout + expert relabeling
# ──────────────────────────────────────────────────────────────────────────────

class DAggerPolicy:
    """Pure learner rollout with expert relabeling at each timestep.

    The learner (pilot) controls the drone. At each step, the expert (MPC)
    computes what it would have commanded. The expert command is recorded
    as the annotation label, paired with the pilot's neural network input (xnn).
    """
    def __init__(self, expert: VehicleRateMPC, pilot: Pilot):
        self.expert = expert
        self.pilot = pilot
        self.hz = pilot.hz
        self.nzcr = pilot.nzcr
        self.annotations = []
        self._u_pilot_prev = np.zeros(4)

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        u_expert, _, _, _ = self.expert.control(tcr, xcr, upr, obj, icr, zcr)
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(
            self._u_pilot_prev, tcr, xcr, obj, icr, zcr)
        self._u_pilot_prev = u_pilot.copy()
        xnn_cpu = {k: v.detach().cpu() for k, v in xnn.items()} if xnn else {}
        self.annotations.append({
            "xnn": xnn_cpu, "x": xcr.copy(), "u": u_expert.copy(),
            "t": tcr, "query": obj,
        })
        return u_pilot, znn, adv, tsol

    def reset(self):
        self.annotations = []
        self._u_pilot_prev = np.zeros(4)


def _reset_pilot(pilot):
    """Reset pilot state between runs."""
    pilot.hy_flag = False
    pilot.hy_idx = 0
    pilot.DxU.zero_()
    if hasattr(pilot, 'Znn') and isinstance(pilot.Znn, torch.Tensor):
        pilot.Znn.zero_()
    if hasattr(pilot, 'chunk_buf'):
        pilot.chunk_buf = None
        pilot.chunk_step = 0


def _load_training_branches(bc_cohort, scene_name, scenes_cfg_dir):
    """Load ONLY training BC trajectories (not validation).

    Returns {obj_name: {branch_id: tXUi_array}}.
    """
    workspace = str(Path(__file__).resolve().parents[3])
    scene_data = _SCENE_CACHE.get(scene_name)
    if scene_data is None:
        scene_data = _get_scene(scene_name, scenes_cfg_dir)

    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    obj_locs = np.array([np.squeeze(t) for t in obj_targets])

    rollout_dir = os.path.join(workspace, "cohorts", bc_cohort, "rollout_data", scene_name)
    train_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories[0-9]*.pt")))

    if not train_files:
        raise FileNotFoundError(f"No training trajectories in {rollout_dir}")

    per_obj = {i: [] for i in range(len(queries))}
    for tf in train_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
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
            result[obj_name] = {i: tr for i, tr in enumerate(trajs)}
            print(f"  [Train] '{obj_name}': {len(trajs)} training branches")
    return result


def _select_branches_for_round(all_branches, prior_failures, rng, round_i,
                                max_success_per_obj=5):
    """Round 1: ALL branches. Round 2+: ALL failures + random successes."""
    selected = {}
    for obj_name, branches in all_branches.items():
        all_ids = list(branches.keys())
        if round_i == 1:
            selected[obj_name] = [(bid, branches[bid]) for bid in all_ids]
        else:
            fail_ids = prior_failures.get(obj_name, [])
            success_ids = [bid for bid in all_ids if bid not in fail_ids]
            n_success = min(max_success_per_obj, len(success_ids))
            sampled = rng.sample(success_ids, n_success) if success_ids else []
            round_ids = sorted(set(fail_ids + sampled))
            selected[obj_name] = [(bid, branches[bid]) for bid in round_ids]

        n_sel = len(selected[obj_name])
        n_fail = len(prior_failures.get(obj_name, [])) if round_i > 1 else 0
        print(f"  '{obj_name[:30]}': {n_sel} branches for R{round_i}"
              f"  ({len(all_ids)} total" +
              (f", {n_fail} fail" if round_i > 1 else "") + ")")
    return selected


def _collect_and_evaluate(pilot, model_path, branches_dict, scene_data, round_i,
                           output_dir=None, is_dagger=True):
    """Run pilot on branches, optionally collecting DAgger annotations.

    When is_dagger=True: uses DAggerPolicy (pure learner + expert relabel).
    When is_dagger=False: uses pilot directly (evaluation only).
    """
    from scipy.spatial import cKDTree
    pilot = _swap_model(pilot, model_path)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    all_annotations = []
    per_object = {}
    all_diag = []
    label = f"R{round_i}" if is_dagger else f"Eval_R{round_i}"
    mode = "DAgger" if is_dagger else "Eval"

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        branches = branches_dict[obj_name]
        successes, collisions, goal_dists = [], [], []
        obj_runs_for_plot = []

        print(f"\n  [{mode}] === {obj_name} ({len(branches)} branches) ===")

        for bi, (br_id, tXUi) in enumerate(branches):
            _reset_pilot(pilot)
            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )

            if is_dagger:
                expert = VehicleRateMPC(tXUi, "vrmpc_rrt", "carl", "InstinctJester")
                dagger_pol = DAggerPolicy(expert, pilot)
                policy = dagger_pol
            else:
                policy = pilot

            _t = time.time()
            result = simulator.simulate(
                policy=policy, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
                x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=obj_name,
                vision_processor=None, verbose=False,
                early_stop_fn=terminal_fn,
            )
            Xro = result[1]
            sim_s = time.time() - _t

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
            stop = term_info.get("reason", "timeout")
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            n_ann = 0
            if is_dagger:
                n_ann = len(dagger_pol.annotations)
                all_annotations.extend(dagger_pol.annotations)

            print(f"  [{label}] {bi+1:3d}/{len(branches)}  {status:4s}  br{br_id}"
                  f"  d0={d0:5.1f}m  goal={gd:5.2f}m  steps={ev['n_steps']:3d}"
                  f"  stop={stop:<12s}" +
                  (f"  ann={n_ann:4d}" if is_dagger else "") +
                  f"  ({sim_s:.0f}s)")

            all_diag.append({
                "object": obj_name, "branch_id": br_id,
                "d0": d0, "success": bool(ev["success"]),
                "collision": bool(ev["collision"]),
                "goal_dist": gd, "min_goal_dist": float(ev["min_goal_dist"]),
                "stop_reason": stop, "n_steps": ev["n_steps"],
                "sim_time_s": round(sim_s, 1),
            })
            if output_dir:
                obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy(), br_id))

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_m = float(np.mean(goal_dists)) if goal_dists else float("nan")
        n_ok = int(sum(successes))
        print(f"  [{mode}] --- '{obj_name[:30]}'  success={sr:.0%} ({n_ok}/{len(branches)})"
              f"  collision={cr:.0%}  avg_goal={gd_m:.2f}m")
        per_object[obj_name] = {
            "success_rate": sr, "collision_rate": cr, "goal_dist": gd_m,
            "n_eval": len(branches), "n_success": n_ok,
        }

        if output_dir and obj_runs_for_plot:
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
    return all_annotations, per_object, all_diag, overall_sr, overall_cr


# ──────────────────────────────────────────────────────────────────────────────
# Main DAgger training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_dagger_policy(
    cohort_name: str,
    method_name: str,
    roster: List[str],
    flights: List[Tuple[str, str]],
    bc_cohort_name: str,
    n_rounds: int = 10,
    n_epochs_per_round: int = 30,
    lr: float = 2e-5,
    seed: int = 42,
    max_success_per_obj: int = 5,
    **kwargs,
) -> dict:
    """
    DAgger training: pure learner rollout + expert relabeling.

    Algorithm:
      Round 1: Fly ALL training branches with learner, expert relabels.
      Round 2+: Fly ALL failure branches + random successes, expert relabels.
      Each round: reset to BC weights, retrain on BC + ALL aggregated DAgger data.
      Each round: evaluate retrained model on ALL training branches.

    Crash-resilient via per-round checkpoints.
    """
    import random

    workspace = str(Path(__file__).resolve().parents[3])
    scenes_cfg_dir = os.path.join(workspace, "configs", "scenes")
    cohort_path = os.path.join(workspace, "cohorts", cohort_name)
    bc_model_path = os.path.join(workspace, "cohorts", bc_cohort_name, "roster",
                                  roster[0], "model.pth")
    pilot_name = roster[0]

    rng = random.Random(seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cohort_path, "visualizations", f"dagger_{ts}")
    ckpt_dir = os.path.join(cohort_path, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("[DAgger] Training — pure learner + expert relabeling")
    print(f"  BC model      : {bc_model_path}")
    print(f"  Rounds        : {n_rounds}")
    print(f"  Epochs/round  : {n_epochs_per_round}")
    print(f"  LR            : {lr}")
    print(f"  Cohort        : {cohort_name}")
    print(f"  Output        : {out_dir}")
    print("=" * 70)

    # Load scene + training branches
    scene_name = flights[0][0]
    scene_data = _get_scene(scene_name, scenes_cfg_dir)
    all_branch_data = _load_training_branches(bc_cohort_name, scene_name, scenes_cfg_dir)

    n_total = sum(len(v) for v in all_branch_data.values())
    print(f"\n  Total training branches: {n_total}")
    for obj, brs in all_branch_data.items():
        print(f"    {obj[:40]}: {len(brs)}")

    # Check for resume
    def _ckpt_path(r, step):
        return os.path.join(ckpt_dir, f"round_{r}_{step}.pt")

    def _load_ckpt(r, step):
        p = _ckpt_path(r, step)
        return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None

    def _save_ckpt(r, step, data):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(data, _ckpt_path(r, step))
        print(f"  [ckpt] Saved {step} -> {_ckpt_path(r, step)}")

    last_completed = 0
    all_annotations = []
    for r in range(1, n_rounds + 1):
        coll = _load_ckpt(r, "collected")
        if coll is None:
            break
        all_annotations.extend(coll["annotations"])
        ev = _load_ckpt(r, "evaluated")
        if ev is None:
            break
        last_completed = r

    if last_completed > 0:
        print(f"\n  [resume] Resuming from round {last_completed + 1}"
              f" ({len(all_annotations)} prior annotations)")

    # Init pilot
    pilot = Pilot(bc_cohort_name, pilot_name)
    pilot.set_mode("deploy")

    model_path = bc_model_path
    model_dir = os.path.join(cohort_path, "roster", pilot_name)
    os.makedirs(model_dir, exist_ok=True)

    if last_completed > 0:
        round_model = os.path.join(model_dir, f"model_round_{last_completed}.pth")
        if os.path.exists(round_model):
            model_path = round_model
            print(f"  [resume] Using model from round {last_completed}")
        elif os.path.exists(os.path.join(model_dir, "model.pth")):
            model_path = os.path.join(model_dir, "model.pth")

    round_results = []

    for r in range(last_completed + 1, n_rounds + 1):
        print(f"\n{'='*70}")
        print(f"[DAgger Round {r}/{n_rounds}]  ({len(all_annotations)} annotations so far)")
        print(f"{'='*70}")

        # Select branches
        if r == 1:
            prior_failures = {}
        else:
            ev_ckpt = _load_ckpt(r - 1, "evaluated")
            prior_failures = {}
            if ev_ckpt:
                for d in ev_ckpt.get("diagnostics", []):
                    if not d["success"]:
                        obj = d["object"]
                        prior_failures.setdefault(obj, []).append(d["branch_id"])

        selected = _select_branches_for_round(
            all_branch_data, prior_failures, rng, r, max_success_per_obj)

        # Collect DAgger data
        coll_ckpt = _load_ckpt(r, "collected")
        if coll_ckpt is not None:
            new_anns = coll_ckpt["annotations"]
            coll_sr = coll_ckpt["overall_sr"]
            coll_cr = coll_ckpt["overall_cr"]
            print(f"  [resume] Round {r} collection loaded ({len(new_anns)} ann, {coll_sr:.0%} SR)")
        else:
            new_anns, coll_results, coll_diag, coll_sr, coll_cr = \
                _collect_and_evaluate(pilot, model_path, selected,
                                       scene_data, round_i=r, output_dir=out_dir,
                                       is_dagger=True)
            _save_ckpt(r, "collected", {
                "annotations": new_anns, "results": coll_results,
                "diagnostics": coll_diag,
                "overall_sr": coll_sr, "overall_cr": coll_cr,
            })

        all_annotations.extend(new_anns)
        print(f"  Total annotations: {len(all_annotations)}")

        # Retrain: reset to BC weights, train on BC + all DAgger
        print(f"\n  Retraining Commander ({n_epochs_per_round} epochs, lr={lr})...")
        agg_dir = os.path.join(cohort_path, "dagger_data", pilot_name)
        os.makedirs(agg_dir, exist_ok=True)
        agg_file = os.path.join(agg_dir, "aggregated_annotations.pt")
        torch.save(all_annotations, agg_file)

        dst_model = os.path.join(model_dir, "model.pth")
        shutil.copy2(bc_model_path, dst_model)

        _retrain_commander(
            cohort_name=cohort_name, pilot_name=pilot_name,
            aggregated_file=agg_file, Nep=n_epochs_per_round,
            lim_sv=10, lr=lr,
            bc_cohort_name=bc_cohort_name,
            dagger_only=False, freeze_vision=True,
        )
        model_path = dst_model
        round_model = os.path.join(model_dir, f"model_round_{r}.pth")
        shutil.copy2(dst_model, round_model)
        print(f"  Model saved -> {round_model}")

        # Evaluate on ALL training branches
        print(f"\n  Evaluating on ALL {n_total} training branches...")
        eval_branches = {obj: [(bid, br) for bid, br in brs.items()]
                         for obj, brs in all_branch_data.items()}
        _, eval_results, eval_diag, eval_sr, eval_cr = \
            _collect_and_evaluate(pilot, model_path, eval_branches,
                                   scene_data, round_i=r, output_dir=out_dir,
                                   is_dagger=False)
        _save_ckpt(r, "evaluated", {
            "results": eval_results, "diagnostics": eval_diag,
            "overall_sr": eval_sr, "overall_cr": eval_cr,
        })

        # Log to wandb
        _wandb_log_round(pilot_name, r, coll_sr, coll_cr, eval_sr, eval_cr,
                          len(all_annotations), eval_results)

        round_results.append({
            "round": r,
            "n_new_annotations": len(new_anns),
            "n_total_annotations": len(all_annotations),
            "collection": {"sr": coll_sr, "cr": coll_cr},
            "eval": {"sr": eval_sr, "cr": eval_cr},
            "eval_per_object": eval_results,
        })

        # Progress table
        print(f"\n{'='*70}")
        print(f"[Progress] After Round {r}/{n_rounds}")
        print(f"  {'Round':8} {'Coll SR':>8} {'Eval SR':>8} {'Eval CR':>8} {'Annotations':>12}")
        print(f"  {'-'*50}")
        for rr in round_results:
            print(f"  R{rr['round']:<7} {rr['collection']['sr']:>7.0%}"
                  f" {rr['eval']['sr']:>7.0%} {rr['eval']['cr']:>7.0%}"
                  f" {rr['n_total_annotations']:>12}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"[DAgger] COMPLETE — {n_rounds} rounds")
    print(f"{'='*70}")
    summary = {
        "timestamp": ts, "cohort": cohort_name,
        "config": {
            "n_rounds": n_rounds, "n_epochs_per_round": n_epochs_per_round,
            "lr": lr, "seed": seed, "bc_cohort": bc_cohort_name,
            "n_total_training_branches": n_total,
        },
        "rounds": round_results,
    }
    out_path = os.path.join(out_dir, "dagger_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Results -> {out_path}")

    return {pilot_name: round_results}
