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

from sousvide.instruct.expert_controllers import PotentialFieldExpert, OnlineRRTExpert

from sousvide.control.pilot import Pilot
import sousvide.instruct.train_policy as tp
from scipy.spatial.transform import Rotation
from sousvide.flight.deploy_ssv import simulate_rollouts
from sousvide.flight.vision_processor_base import create_vision_processor
from sousvide.rl import load_simulation_results, prepare_batch_data


# ──────────────────────────────────────────────────────────────────────────────
# EWC (Elastic Weight Consolidation) — prevents catastrophic forgetting
# ──────────────────────────────────────────────────────────────────────────────

def compute_fisher_information(model, dataloader, device, mode="Commander"):
    """
    Compute diagonal Fisher Information Matrix for EWC.
    Uses the model's predictions (not labels) to compute empirical Fisher.

    Args:
        model: SVNet model
        dataloader: DataLoader with BC or DAgger training data
        device: torch device

    Returns:
        fisher_dict: {param_name: Fisher diagonal tensor}
        optpar_dict: {param_name: optimal parameter snapshot}
    """
    fisher_dict = {}
    optpar_dict = {}

    # Only compute Fisher for Commander parameters (the ones we retrain)
    commander = model.network["CommanderSV"]
    for name, param in commander.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)
        optpar_dict[name] = param.data.clone()

    model.eval()
    n_samples = 0
    for inputs, labels in dataloader:
        inputs = tuple(t.to(device) for t in inputs)
        labels = labels.to(device)

        model.zero_grad()
        output, _ = model(*inputs)
        # Use squared prediction as proxy for log-likelihood
        loss = (output ** 2).sum()
        loss.backward()

        for name, param in commander.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        n_samples += labels.shape[0]

    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= max(n_samples, 1)

    model.train()
    return fisher_dict, optpar_dict


def ewc_penalty(model, fisher_dict, optpar_dict):
    """
    Compute EWC penalty: sum_i F_i * (theta_i - theta*_i)^2

    Args:
        model: SVNet model
        fisher_dict: Fisher diagonal from compute_fisher_information
        optpar_dict: Optimal parameter snapshot

    Returns:
        penalty: scalar tensor
    """
    penalty = torch.tensor(0.0, device=DEVICE)
    commander = model.network["CommanderSV"]
    for name, param in commander.named_parameters():
        if name in fisher_dict:
            penalty += (fisher_dict[name] * (param - optpar_dict[name]) ** 2).sum()
    return penalty


def _retrain_commander_ewc(
    cohort_name: str, pilot_name: str,
    aggregated_file: str, Nep: int, lim_sv: int,
    default_mass: float = 0.3, default_fn: float = 0.3,
    lr: float = 1e-4,
    bc_cohort_name: str = None,
    dagger_only: bool = False,
    oversample: int = 1,
    freeze_vision: bool = True,
    ewc_lambda: float = 0.0,
    fisher_dict: dict = None,
    optpar_dict: dict = None,
    lr_schedule: str = None,
    weight_decay: float = 0.0,
    collision_weight_alpha: float = 0.0,
    collision_weight_threshold: float = 0.5,
    max_dagger_samples: int = 0,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Enhanced version of _retrain_commander with EWC, LR scheduling,
    and collision-weighted loss.

    Returns (fisher_dict, optpar_dict) for use in next iteration.
    If ewc_lambda=0, no scheduling, and no collision weighting,
    behaves identically to _retrain_commander.

    collision_weight_alpha: if > 0, annotations near obstacles (clearance < threshold)
        get upweighted by alpha * (1 - clearance/threshold). This teaches the Commander
        to be extra careful near obstacles, addressing the collision plateau.
    collision_weight_threshold: distance (m) below which collision weighting kicks in.
    max_dagger_samples: if > 0, subsample DAgger annotations to this max BEFORE oversample.
        Prevents DAgger:BC ratio drift in cumulative aggregation mode. Set to ~BC_size
        to maintain ~1:1 ratio after oversample=1, or BC_size/oversample for other ratios.
    """
    workspace_path = str(Path(__file__).resolve().parents[3])
    annotations = torch.load(aggregated_file, weights_only=False)

    if not annotations:
        print("  [retrain_ewc] No annotations — skipping.")
        return fisher_dict, optpar_dict

    # Cap DAgger samples to prevent ratio drift in cumulative aggregation.
    # Without this, DAgger:BC ratio grows from ~15% (iter 0) to ~69% (iter 11),
    # causing the model to progressively overfit to corrective DAgger data and
    # drift further from the BC distribution (manifests as increasing expert
    # command magnitude over iterations).
    if max_dagger_samples > 0 and len(annotations) > max_dagger_samples:
        print(f"  [retrain_ewc] Capping DAgger annotations: {len(annotations)} → {max_dagger_samples} "
              f"(prevents ratio drift)")
        indices = np.random.choice(len(annotations), max_dagger_samples, replace=False)
        annotations = [annotations[i] for i in indices]

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
        print("  [retrain_ewc] No valid xnn entries — skipping.")
        return fisher_dict, optpar_dict

    # Compute collision weights from per-annotation clearance
    sample_weights = None
    if collision_weight_alpha > 0:
        raw_weights = []
        n_with_clearance = 0
        for ann in annotations:
            cl = ann.get("clearance", None)
            if cl is not None:
                n_with_clearance += 1
                # Higher weight for annotations near obstacles
                proximity = max(0.0, collision_weight_threshold - cl) / collision_weight_threshold
                w = 1.0 + collision_weight_alpha * proximity
            else:
                w = 1.0
            raw_weights.append(w)
        if n_with_clearance > 0:
            sample_weights = raw_weights
            n_upweighted = sum(1 for w in raw_weights if w > 1.01)
            avg_w = np.mean(raw_weights)
            print(f"  [retrain_ewc] Collision weighting: {n_upweighted}/{n_with_clearance} "
                  f"near-obstacle samples (alpha={collision_weight_alpha}, "
                  f"threshold={collision_weight_threshold}m, avg_w={avg_w:.2f})")
        else:
            print(f"  [retrain_ewc] No clearance data in annotations — collision weighting disabled")

    if oversample > 1:
        Xnn = Xnn * oversample
        Ynn = Ynn * oversample
        if sample_weights is not None:
            sample_weights = sample_weights * oversample
        print(f"  [retrain_ewc] {len(Xnn)//oversample} annotations x {oversample} = {len(Xnn)} samples")
    else:
        print(f"  [retrain_ewc] {len(Xnn)} annotation samples")

    obs_data = {
        "data": [{
            "Xnn": Xnn, "Ynn": Ynn, "Ndata": len(Xnn),
            "rollout_id": 0, "course": "dagger",
            "frame": {"mass": default_mass, "force_normalized": default_fn},
        }],
        "set": "", "Nobs": len(Xnn), "course": "dagger",
    }

    # Save observation file
    course_dir = os.path.join(
        workspace_path, "cohorts", cohort_name,
        "observation_data", pilot_name, "dagger",
    )
    os.makedirs(course_dir, exist_ok=True)
    dst = os.path.join(course_dir, "observations_dagger.pt")
    torch.save(obs_data, dst)

    if dagger_only:
        course = "dagger"
    else:
        course = None
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
                    if not entry.is_dir() or entry.name == "dagger":
                        continue
                    link_path = os.path.join(dag_obs_base, entry.name)
                    if not os.path.exists(link_path):
                        os.symlink(entry.path, link_path)

    # Load pilot and freeze vision
    from sousvide.control.pilot import Pilot as _Pilot
    student = _Pilot(cohort_name, pilot_name)
    student.set_mode('train')
    student.model.to(DEVICE)

    if freeze_vision and hasattr(student.model, 'get_network') and "Commander" in student.model.get_network:
        student.model.get_network["Commander"]["Unlock"] = nn.ModuleList(
            [student.model.network["CommanderSV"]]
        )
        n_frozen = sum(p.numel() for p in student.model.network["VisionMLP"].parameters())
        n_unlocked = sum(p.numel() for p in student.model.network["CommanderSV"].parameters())
        print(f"  [retrain_ewc] VisionMLP FROZEN ({n_frozen:,} params) — only CommanderSV ({n_unlocked:,} params) updated")

    # Custom training loop with EWC
    from sousvide.instruct.synthesized_data import generate_dataset, get_data_paths
    from torch.utils.data import DataLoader

    use_collision_weights = collision_weight_alpha > 0 and sample_weights is not None
    criterion = nn.MSELoss(reduction='none') if use_collision_weights else nn.MSELoss(reduction='mean')
    model = student.model.get_network["Commander"]["Train"]

    # Unlock Commander only
    for param in student.model.parameters():
        param.requires_grad = False
    for param in student.model.get_network["Commander"]["Unlock"].parameters():
        param.requires_grad = True

    # Setup optimizer with optional weight decay
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Nep, eta_min=lr * 0.01)
    elif lr_schedule == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(Nep // 3, 1), gamma=0.5)

    # Get data
    od_train_files, od_test_files, _, _ = get_data_paths(cohort_name, student.name, course_name=course)
    if not od_train_files:
        print("  [retrain_ewc] No training data found — skipping.")
        return fisher_dict, optpar_dict

    # Build collision weight tensor for the DAgger data (first N samples)
    # The DAgger observations are saved first, so they correspond to the first portion
    # of the dataset. BC observations (if any) follow and get weight=1.0.
    weight_tensor = None
    if use_collision_weights:
        weight_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=DEVICE)
        print(f"  [retrain_ewc] Collision weight tensor: {weight_tensor.shape[0]} samples, "
              f"range=[{weight_tensor.min():.2f}, {weight_tensor.max():.2f}]")

    print(f"  [retrain_ewc] Training {Nep} epochs, lr={lr}, ewc_lambda={ewc_lambda}, "
          f"schedule={lr_schedule or 'none'}, wd={weight_decay}"
          f"{f', collision_alpha={collision_weight_alpha}' if use_collision_weights else ''}")

    best_loss = float('inf')
    for ep in range(Nep):
        epoch_loss = 0.0
        n_samples = 0
        sample_idx = 0  # track position for weight indexing

        for od_file in od_train_files:
            dataset = generate_dataset(od_file, student, "Commander", DEVICE)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

            for inputs, labels in dataloader:
                inputs = tuple(t.to(DEVICE) for t in inputs)
                labels = labels.to(DEVICE)

                prediction, _ = model(*inputs)

                if use_collision_weights:
                    # Per-element loss: (batch, output_dim)
                    per_sample_loss = criterion(prediction, labels).mean(dim=-1)  # (batch,)
                    # Apply collision weights for DAgger samples
                    batch_size = labels.shape[0]
                    if weight_tensor is not None and sample_idx + batch_size <= weight_tensor.shape[0]:
                        # Note: shuffle breaks exact weight-to-sample mapping.
                        # Since DAgger data is separate from BC data, and the weight
                        # distribution is consistent across the DAgger dataset,
                        # we use a statistical approximation: randomly sample weights
                        # from the weight tensor for each batch. This preserves
                        # the overall distribution of collision-aware emphasis.
                        w_idx = torch.randint(0, weight_tensor.shape[0], (batch_size,))
                        batch_weights = weight_tensor[w_idx]
                        loss = (per_sample_loss * batch_weights).mean()
                    else:
                        loss = per_sample_loss.mean()
                else:
                    loss = criterion(prediction, labels)

                # Add EWC penalty
                if ewc_lambda > 0 and fisher_dict is not None:
                    ewc_loss = ewc_penalty(student.model, fisher_dict, optpar_dict)
                    loss = loss + ewc_lambda * ewc_loss

                loss.backward()
                # Gradient clipping for stability (especially with collision weights + EWC)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item() * labels.shape[0]
                n_samples += labels.shape[0]
                sample_idx += labels.shape[0]

        avg_loss = epoch_loss / max(n_samples, 1)
        if scheduler:
            scheduler.step()

        if (ep + 1) % max(Nep // 5, 1) == 0 or ep == Nep - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            ewc_str = f" ewc={ewc_lambda * ewc_penalty(student.model, fisher_dict, optpar_dict).item():.4f}" if ewc_lambda > 0 and fisher_dict else ""
            print(f"    [ep {ep+1}/{Nep}] loss={avg_loss:.6f} lr={cur_lr:.2e}{ewc_str}")

    # Save model
    model_path = os.path.join(student.path, "model.pth")
    for param in student.model.parameters():
        param.requires_grad = False
    torch.save(student.model, model_path)
    print(f"  [retrain_ewc] Model saved → {model_path}")

    # Compute new Fisher information for next iteration's EWC
    new_fisher, new_optpar = None, None
    if ewc_lambda > 0:
        print("  [retrain_ewc] Computing Fisher information for next iteration...")
        all_datasets = []
        for od_file in od_train_files:
            ds = generate_dataset(od_file, student, "Commander", DEVICE)
            all_datasets.append(ds)
        if all_datasets:
            from torch.utils.data import ConcatDataset
            combined = ConcatDataset(all_datasets)
            combined_loader = DataLoader(combined, batch_size=64, shuffle=False)
            new_fisher, new_optpar = compute_fisher_information(
                student.model, combined_loader, DEVICE
            )
            print(f"  [retrain_ewc] Fisher computed for {sum(f.numel() for f in new_fisher.values())} params")

    return new_fisher, new_optpar

# ──────────────────────────────────────────────────────────────────────────────
# Device + cuDNN
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

# ──────────────────────────────────────────────────────────────────────────────
# PERF : caches globaux — survivent entre pilots ET entre appels benchmark
# ──────────────────────────────────────────────────────────────────────────────

# {scene_name: {"simulator": Simulator, "obj_targets": [...], "epcds_arr": ndarray}}
_SCENE_CACHE: Dict[str, dict] = {}

# {f"{scene_name}_{obj_name}": {"tXUi": ndarray, ...}}
_PKL_CACHE: Dict[str, dict] = {}

# {f"{scene_name}_{obj_name}": [list of (18, N_i) tXUd arrays]}  — from BC rollout data
_BC_TRAJ_CACHE: Dict[str, List[np.ndarray]] = {}


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
    final_goal_dist = float(goal_dists[-1])

    # First entry into goal zone (any timestep)
    in_zone = goal_dists <= success_radius
    reached_goal = bool(in_zone.any())
    first_entry_step = int(np.argmax(in_zone)) if reached_goal else -1

    # ── Collision detection ──

    # (a) Point-cloud proximity
    collided_pc = False
    collision_step = n_steps
    if pc_bench.shape[0] > 0:
        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
        pc_t  = torch.from_numpy(pc_bench).float().to(DEVICE)
        dists_to_pc = torch.cdist(Xro_t, pc_t)  # (N, M)
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

    return {
        "success":              success,
        "collision":            collided_before_goal,
        "collided_before_goal": collided_before_goal,
        "goal_dist":            final_goal_dist,
        "min_goal_dist":        min_goal_dist,
        "first_entry_step":     first_entry_step,
        "goal_in_fov":          goal_in_fov,
        "out_of_bounds":        out_of_bounds,
        "total_reward":         -final_goal_dist,
        "mean_pos_dev":         mean_pos_dev,
        "mean_orient_dev_deg":  mean_orient_dev_deg,
        "fov_pct":              fov_pct,
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
    """Save interactive Plotly HTML reusing create_comparison_figure from compare_trajectories_3d.

    Args:
        obj_runs: list of (Xro, ev_dict, tXUi) tuples for this object.
        obj_target: 3D target position.
        simulator: FiGS simulator (for point cloud extraction).
        obj_name: object name for title.
        save_path: output .html path.
        reference_branches: optional list of tXUi arrays (background, not used if per-run refs exist).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        import sys as _sys
        # __file__ = src/sousvide/instruct/train_dagger.py → 4 levels up to repo root
        _scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        from compare_trajectories_3d import create_comparison_figure, get_point_cloud

        pts, cols = get_point_cloud(simulator)

        # Convert (Xro, ev, tXUi) tuples to the format expected by create_comparison_figure
        formatted_runs = []
        per_run_refs = []
        for entry in obj_runs:
            Xro, ev, tXUi = entry[0], entry[1], entry[2]
            Tro = np.arange(Xro.shape[1], dtype=float) / 20.0
            formatted_runs.append((Xro, Tro, ev, []))
            per_run_refs.append(tXUi)

        # Use per-run reference branches (the actual RRT branch each run followed)
        all_refs = per_run_refs if per_run_refs else (reference_branches or [])

        fig = create_comparison_figure(
            pts, cols, obj_target, obj_name,
            expert_runs=[],
            bc_runs=[],
            dagger_runs=formatted_runs,
            reference_branches=all_refs,
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


def _clear_caches():
    global _SCENE_CACHE, _PKL_CACHE
    for v in _SCENE_CACHE.values():
        del v["simulator"]
    _SCENE_CACHE.clear()
    _PKL_CACHE.clear()
    torch.cuda.empty_cache()
    gc.collect()
    print("[Cache] Caches libérés")


# ──────────────────────────────────────────────────────────────────────────────
# Politique mixte β * expert + (1-β) * pilot
# ──────────────────────────────────────────────────────────────────────────────

class MixedPolicy:
    """
    Politique mixte compatible avec l'interface Pilot.control() attendue par Simulator.simulate().
    Interface : control(tcr, xcr, upr, obj, icr, zcr) → (unn, znn, adv, tsol)
    """
    def __init__(self, expert: VehicleRateMPC, pilot: Pilot, beta: float):
        self.expert      = expert
        self.pilot       = pilot
        self.beta        = beta
        self.hz          = pilot.hz          # requis par Simulator (n_sim2ctl)
        self.nzcr        = pilot.nzcr        # requis par Simulator (zcr init)
        self.annotations: List[dict] = []
        self._u_exp_prev = np.zeros(4)       # expert-only prev action for clean history

    def control(
        self,
        tcr: float,
        xcr: np.ndarray,
        upr: np.ndarray,
        obj: np.ndarray,
        icr,
        zcr,
    ):
        """Interface identique à Pilot.control() — appelée par Simulator.simulate()."""
        # Expert MPC
        u_expert, _, _, _ = self.expert.control(tcr, xcr, upr, obj, icr, zcr)

        # Pilot neural — pass u_exp_prev (pure expert history) to match BC distribution.
        # BC training always uses upr = ucr (expert action), never a mixed action.
        # Passing the mixed u_out_prev would corrupt dxu_par → HistoryEncoder breakdown.
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(self._u_exp_prev, tcr, xcr, obj, icr, zcr)

        # Update expert-only history for next step (mirrors BC: upr = ucr each step)
        self._u_exp_prev = u_expert.copy()

        # Annotation : expert command + pilot observation (xnn) for retraining
        # Detach xnn tensors to CPU so they can be saved/aggregated safely
        xnn_cpu = {k: v.detach().cpu() for k, v in xnn.items()} if xnn else {}
        self.annotations.append({
            "xnn":   xnn_cpu,
            "x":     xcr.copy(),
            "u":     u_expert.copy(),
            "t":     tcr,
            "query": obj,
        })

        u_out = u_expert if np.random.rand() < self.beta else u_pilot
        return u_out, znn, adv, tsol

    def reset_annotations(self):
        self.annotations = []
        self._u_exp_prev = np.zeros(4)   # mirrors BC: upr = np.zeros(4) at rollout start


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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

def _generate_rrt_backup(
    pilot: Pilot,
    model_path: str,
    workspace_path: str,
    cohort_name: str,
    cohort_path: str,
    method_name: str,
    flights: List[Tuple[str, str]],
    scenes_cfg_dir: str,
    objective_configs: dict,
    sim_base: str,
    rrt_backup: str,
    benchmark_seed: int,
    max_trajectories: int,
    existing_rrt_dir: Optional[str] = None,
) -> None:
    # Si _PKL_CACHE déjà peuplé → copie directe, 0 simulate_rollouts
    if _PKL_CACHE:
        print(f"[RRT-Backup] ✅ {len(_PKL_CACHE)} pkl en cache mémoire — 0 gsplat reload")
        os.makedirs(rrt_backup, exist_ok=True)
        for key in _PKL_CACHE:
            src = os.path.join(scenes_cfg_dir, f"{key}.pkl")
            # Fallback avec espaces
            if not os.path.exists(src):
                src = os.path.join(
                    scenes_cfg_dir,
                    f"{key.replace('_', ' ', 1)}.pkl"
                )
            dst = os.path.join(rrt_backup, f"{key}.pkl")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        return

    # Chercher sur disque dans configs/scenes/
    scene_pkl_dir = os.path.join(workspace_path, "configs", "scenes")
    existing_pkls = []
    for scene_name, _ in flights:
        existing_pkls.extend(glob.glob(
            os.path.join(scene_pkl_dir, f"{scene_name}*.pkl")
        ))

    if existing_pkls:
        print(f"[RRT-Backup] {len(existing_pkls)} .pkl sur disque — copie directe")
        os.makedirs(rrt_backup, exist_ok=True)
        for f in existing_pkls:
            shutil.copy2(f, rrt_backup)
        # Peupler _PKL_CACHE depuis ce qu'on vient de trouver
        _preload_all_pkls(flights, scenes_cfg_dir)
        return

    # Fallback : 1 seul appel simulate_rollouts
    print("[RRT-Backup] Aucun pkl — génération RRT (1 appel)...")
    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)
    pilot = _swap_model(pilot, model_path)

    simulate_rollouts(
        workspace_path=workspace_path,
        cohort_name=cohort_name,
        cohort_path=cohort_path,
        method_name=method_name,
        pilot=pilot,
        flights=flights,
        scenes_cfg_dir=scenes_cfg_dir,
        objectives_all=objective_configs,
        max_trajectories=max_trajectories,
        review=False,
        disable_visualization=True,
        show_progress=True,
    )
    plt.close("all")

    sim_parent = os.path.dirname(sim_base)
    latest_ts  = max(
        d for d in os.listdir(sim_parent)
        if os.path.isdir(os.path.join(sim_parent, d)) and d != "dagger"
    )
    rrt_source = os.path.join(sim_parent, latest_ts, "rrt_planning")
    if os.path.exists(rrt_backup):
        shutil.rmtree(rrt_backup)
    shutil.copytree(rrt_source, rrt_backup)
    print(f"[RRT-Backup] {len(os.listdir(rrt_backup))} pkl → {rrt_backup}")
    _preload_all_pkls(flights, scenes_cfg_dir)


# ──────────────────────────────────────────────────────────────────────────────
# PERF : benchmark — réutilise _SCENE_CACHE, 0 rechargement gsplat
# ──────────────────────────────────────────────────────────────────────────────

def _run_benchmark_pilot(
    pilot,
    model_path: str,
    label: str,
    workspace_path: str,
    cohort_name: str,
    cohort_path: str,
    method_name: str,
    flights,
    scenes_cfg_dir: str,
    objective_configs: dict,
    collision_detectors: dict,
    scene_names,
    sim_base: str,
    rrt_backup: str,
    benchmark_seed: int,
    max_trajectories: int,
    save_plots: bool = False,
    save_videos: bool = False,
    output_dir: str = None,
) -> dict:
    """
    Benchmark the pilot on max_trajectories start positions per object.
    Uses the same uniform-sampling approach as _eval_full_trajectories so
    all evaluation phases (before, per-iter, after) are comparable.
    Benchmark seed ensures before/after use identical start positions.
    """
    pilot = _swap_model(pilot, model_path)

    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

    all_analyses = []
    all_Tro, all_Xro, all_Uro = [], [], []
    per_object = {}  # per-object metrics for wandb

    for scene_name, _ in flights:
        scene_data  = _get_scene(scene_name, scenes_cfg_dir)
        simulator   = scene_data["simulator"]
        obj_targets = scene_data["obj_targets"]
        queries     = scene_data["queries"]

        for obj_idx, obj_name in enumerate(queries):
            pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if pkl_data is None:
                continue

            tXUi_default = pkl_data["tXUi"]
            obj_target = (
                obj_targets[obj_idx] if obj_idx < len(obj_targets)
                else pkl_data.get("obj_loc", np.zeros(3))
            )

            # Load all branches for multi-branch benchmarking
            all_branches = _load_all_branches(
                scene_name, obj_name, cohort_path,
                scenes_cfg_dir, tXUi_default,
            )

            # Sample max_trajectories branches (each run = different route)
            n_available = len(all_branches)
            n_sample = min(max_trajectories, n_available)
            branch_idxs = np.random.choice(n_available, size=n_sample, replace=False)
            # If we need more runs than branches, allow repeats with offset starts
            if max_trajectories > n_available:
                extra = np.random.choice(n_available, size=max_trajectories - n_available, replace=True)
                branch_idxs = np.concatenate([branch_idxs, extra])

            print(f"  [{label}] '{obj_name}'  {max_trajectories} runs across "
                  f"{n_available} branches  seed={benchmark_seed}")
            obj_analyses = []
            obj_runs_for_plot = []
            for run_i, br_idx in enumerate(branch_idxs):
                # Reset pilot history between benchmark runs
                pilot.hy_flag = False
                pilot.hy_idx = 0
                pilot.DxU.zero_()
                if hasattr(pilot, 'Znn'):
                    pilot.Znn.zero_()

                tXUi = all_branches[br_idx]
                t0 = float(tXUi[0, 0])
                tf = float(tXUi[0, -1])
                T  = tf - t0
                x0      = tXUi[1:11, 0].copy()
                t_start = t0
                t_end   = tf

                _t_sim = time.time()
                result  = simulator.simulate(
                    policy=pilot, t0=t_start, tf=t_end, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=None, verbose=False,
                )
                Tro, Xro = result[0], result[1]
                Uro = result[2] if len(result) > 2 else None
                Iro = result[3] if len(result) > 3 else None

                pc_bench = scene_data.get("epcds_arr", np.zeros((0,3)))
                ev = _evaluate_run(Xro, obj_target, pc_bench,
                                   env_min=scene_data.get("env_min"),
                                   env_max=scene_data.get("env_max"),
                                   tXUi=tXUi, idx0=0)
                goal_dist = ev["goal_dist"]
                collided  = ev["collision"]
                success   = ev["success"]
                fov_ok    = ev["goal_in_fov"]
                oob       = ev.get("out_of_bounds", False)
                pos_dev = ev.get("mean_pos_dev", float('nan'))
                ori_dev = ev.get("mean_orient_dev_deg", float('nan'))
                fov_p   = ev.get("fov_pct", float('nan'))
                status    = "✓" if success else ("🚧" if oob else ("💥" if collided else ("👁" if not fov_ok else "✗")))
                fov_str   = "fov=✓" if fov_ok else "fov=✗"
                min_gd    = ev["min_goal_dist"]
                oob_str   = "  OOB!" if oob else ""
                dev_str = f"  pos_dev={pos_dev:.2f}m  ori_dev={ori_dev:.1f}°  fov={fov_p:.0%}" if not np.isnan(pos_dev) else ""
                print(f"  [{label}] {status}  '{obj_name[:20]}'  run {run_i+1}/{max_trajectories}"
                      f"  goal_dist={goal_dist:.2f}m  min={min_gd:.2f}m  coll={collided}  {fov_str}{oob_str}{dev_str}  ({time.time()-_t_sim:.1f}s)")

                # Collect run data for Plotly visualization
                if save_plots and output_dir is not None:
                    obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy()))

                # Save video from rendered frames (same pattern as deploy_ssv.py)
                if save_videos and output_dir is not None and Iro is not None:
                    import imageio
                    from torchvision.transforms import Resize
                    vid_dir = os.path.join(output_dir, "videos")
                    os.makedirs(vid_dir, exist_ok=True)
                    for ch_name in Iro:
                        if ch_name == "depth_raw":
                            continue
                        frames_np = Iro[ch_name]
                        if frames_np.shape[0] == 0:
                            continue
                        # Resize to 720x1280 like deploy_ssv.py
                        resize = Resize((720, 1280), antialias=True)
                        frames_t = torch.from_numpy(frames_np)
                        out_frames = []
                        for fi in range(frames_t.shape[0]):
                            img = frames_t[fi]
                            if img.ndim == 3 and img.shape[-1] == 3:
                                img = img.permute(2, 0, 1)  # HWC -> CHW
                            img = resize(img)
                            out_frames.append(img.permute(1, 2, 0).numpy().astype(np.uint8))
                        # Full object name like deploy_ssv.py:
                        # sim_video_{scene}_{object}_{label}_run{N}_{channel}.mp4
                        vid_path = os.path.join(vid_dir,
                                                f"sim_video_{scene_name}_{obj_name}_{label}_run{run_i:03d}_{ch_name}.mp4")
                        imageio.mimwrite(vid_path, out_frames, fps=20)
                        print(f"  [video] -> {vid_path}")

                analysis = {
                    "collision":               collided,
                    "success":                 success,
                    "clearance_series":        None,
                    "goal_in_camera_fov_series": None,
                    "goal_in_camera_fov":      fov_ok,
                    "total_reward":            -goal_dist,
                    "goal_dist":               goal_dist,
                    "min_goal_dist":           min_gd,
                    "min_clearance":           None,
                    "mean_pos_dev":            pos_dev,
                    "mean_orient_dev_deg":     ori_dev,
                    "fov_pct":                 fov_p,
                }
                all_analyses.append(analysis)
                obj_analyses.append(analysis)
                all_Tro.append(Tro)
                all_Xro.append(Xro)
                if Uro is not None:
                    all_Uro.append(Uro)

            sr = sum(1 for a in obj_analyses if a["success"]) / max(len(obj_analyses), 1)
            cr = sum(1 for a in obj_analyses if a["collision"]) / max(len(obj_analyses), 1)
            gd = float(np.mean([-a["total_reward"] for a in obj_analyses]))
            avg_pd = float(np.nanmean([a.get("mean_pos_dev", float('nan')) for a in obj_analyses]))
            avg_od = float(np.nanmean([a.get("mean_orient_dev_deg", float('nan')) for a in obj_analyses]))
            avg_fp = float(np.nanmean([a.get("fov_pct", float('nan')) for a in obj_analyses]))
            dev_summary = f"  pos_dev={avg_pd:.2f}m  ori_dev={avg_od:.1f}°  fov={avg_fp:.0%}" if not np.isnan(avg_pd) else ""
            print(f"  [{label}] ── '{obj_name[:25]}'  success={sr:.0%}  mean_goal_dist={gd:.2f}m{dev_summary}")
            per_object[obj_name] = {"success_rate": sr, "collision_rate": cr, "goal_dist": gd,
                                    "mean_pos_dev": avg_pd, "mean_orient_dev_deg": avg_od, "fov_pct": avg_fp}

            # Save interactive Plotly HTML per object
            if save_plots and output_dir is not None and obj_runs_for_plot:
                obj_short = obj_name.replace(" ", "_")[:30]
                html_path = os.path.join(output_dir, "plots",
                                         f"{label}_{obj_short}.html")
                _save_benchmark_plotly(
                    obj_runs=obj_runs_for_plot,
                    obj_target=obj_target,
                    simulator=simulator,
                    obj_name=f"{label} — {obj_name} ({sr:.0%} success)",
                    save_path=html_path,
                    reference_branches=all_branches[:min(10, len(all_branches))],
                )
                del obj_runs_for_plot

        torch.cuda.empty_cache()

    print(f"  [{label}] {len(all_analyses)} total trajectories evaluated")
    metrics = _extract_metrics_from_analyses(all_analyses, scene_data["epcds_arr"])
    metrics["label"] = label
    metrics["per_object"] = per_object

    del all_Tro, all_Xro, all_Uro
    gc.collect()
    return metrics


def _extract_metrics_from_analyses(analyses: list, point_cloud) -> dict:
    """Calcule les métriques directement depuis les analyses (sans load_simulation_results)."""
    collision_rates, clearances_mean, fov_rates, returns_ = [], [], [], []
    successes_, goal_dists_ = [], []

    pc = point_cloud
    if isinstance(pc, list):
        pc = np.concatenate(pc, axis=0) if pc else np.zeros((0, 3))

    for a in analyses:
        collision_rates.append(float(a.get("collision", False)))
        clr = a.get("clearance_series") or a.get("min_clearance")
        if clr is not None:
            arr = np.asarray(clr).reshape(-1)
            clearances_mean.append(float(arr.mean()) if arr.size else np.nan)
        else:
            clearances_mean.append(np.nan)

        fov = a.get("goal_in_camera_fov_series") or a.get("goal_in_camera_fov")
        if fov is not None:
            arr = np.asarray(fov).reshape(-1)
            fov_rates.append(float(np.mean(arr > 0.5)) if arr.size else np.nan)
        else:
            fov_rates.append(np.nan)

        ret = a.get("total_reward") or a.get("return_sum")
        returns_.append(float(ret) if ret is not None else np.nan)

        successes_.append(float(a.get("success", False)))
        # total_reward = -goal_dist
        gd = a.get("goal_dist")
        if gd is None and ret is not None:
            gd = -float(ret)
        goal_dists_.append(float(gd) if gd is not None else np.nan)

    return {
        "collision_rate": np.array(collision_rates),
        "clearance_mean": np.array(clearances_mean),
        "fov_rate":       np.array(fov_rates),
        "return_sum":     np.array(returns_),
        "traj_length":    np.array([1.0] * len(analyses)),
        "success_rate":   np.array(successes_),
        "goal_dist":      np.array(goal_dists_),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cross-cohort benchmark — compare multiple InstinctJester variants on the
# exact same held-out start conditions (seed ≠ DAgger benchmark_seed).
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_cohort_benchmark(
    models: list,
    flights: list,
    scenes_cfg_dir: str,
    benchmark_seed: int = 123,
    max_trajectories: int = 50,
    output_path: Optional[str] = None,
    full_range: bool = False,
) -> dict:
    """
    Evaluate multiple InstinctJester model variants on the **same** held-out
    benchmark conditions and produce a clean comparison table.

    Parameters
    ----------
    models : list of dicts, each with keys:
        - "label"       : display name (e.g. "before_dagger", "after_potential", "after_rrt")
        - "cohort"      : cohort name used to instantiate Pilot (e.g. "ssv_CLIPSEG_NORMAL")
        - "pilot_name"  : roster name (e.g. "InstinctJester")
        - "model_path"  : path to .pth weights to load into the pilot
    flights       : same list-of-[scene, obj_query] as DAgger training
    scenes_cfg_dir: path to configs/scenes/
    benchmark_seed: integer seed — use a value DIFFERENT from the DAgger
                    benchmark_seed (42) so these conditions are unseen.
    max_trajectories: trajectories per object per model (sampled from the
                    SECOND half of tXUi — held out from BC training).
    output_path   : optional path to write a JSON summary.

    Returns
    -------
    dict: {label: {obj_name: {goal_dist, success_rate, collision_rate}}}
    """
    print("\n" + "=" * 70)
    print("[CrossBenchmark] Held-out cross-cohort comparison")
    print(f"  seed={benchmark_seed}  n={max_trajectories} traj/obj  "
          f"models={[m['label'] for m in models]}")
    print("=" * 70 + "\n")

    # Pre-load scenes (uses global _SCENE_CACHE — may already be warm)
    for scene_name, _ in flights:
        _get_scene(scene_name, scenes_cfg_dir)
    _preload_all_pkls(flights, scenes_cfg_dir)

    # Precompute shared start_idxs per (scene, obj) so EVERY model gets
    # identical starting conditions.
    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

    shared_starts: Dict[str, np.ndarray] = {}  # key: f"{scene}_{obj}"
    for scene_name, _ in flights:
        scene_data = _get_scene(scene_name, scenes_cfg_dir)
        queries    = scene_data["queries"]
        for obj_name in queries:
            key      = f"{scene_name}_{obj_name}"
            pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if pkl_data is None:
                continue
            tXUi   = pkl_data["tXUi"]
            n_cols = tXUi.shape[1]
            start_idx = 1 if full_range else max(1, n_cols // 2)
            shared_starts[key] = np.linspace(
                start_idx, n_cols - 1, max_trajectories, dtype=int
            )

    all_results: dict = {}

    for model_cfg in models:
        label       = model_cfg["label"]
        cohort      = model_cfg["cohort"]
        pilot_name  = model_cfg["pilot_name"]
        model_path  = model_cfg["model_path"]

        print(f"\n[CrossBenchmark] ▶ {label}")
        print(f"  cohort={cohort}  pilot={pilot_name}  weights={model_path}")

        pilot = Pilot(cohort, pilot_name)
        pilot.set_mode("deploy")
        pilot.model.to(DEVICE)
        pilot = _swap_model(pilot, model_path)

        label_results: dict = {}

        for scene_name, _ in flights:
            scene_data  = _get_scene(scene_name, scenes_cfg_dir)
            simulator   = scene_data["simulator"]
            obj_targets = scene_data["obj_targets"]
            queries     = scene_data["queries"]

            for obj_idx, obj_name in enumerate(queries):
                key      = f"{scene_name}_{obj_name}"
                pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
                if pkl_data is None or key not in shared_starts:
                    continue

                tXUi       = pkl_data["tXUi"]
                obj_target = (
                    obj_targets[obj_idx] if obj_idx < len(obj_targets)
                    else pkl_data.get("obj_loc", np.zeros(3))
                )
                t0 = float(tXUi[0,  0])
                tf = float(tXUi[0, -1])
                T  = tf - t0
                start_idxs = shared_starts[key]

                goal_dists, successes, collisions = [], [], []
                for run_i, s_idx in enumerate(start_idxs):
                    # Reset pilot history between runs
                    pilot.hy_flag = False
                    pilot.hy_idx = 0
                    pilot.DxU.zero_()
                    if hasattr(pilot, 'Znn'):
                        pilot.Znn.zero_()
                    if hasattr(pilot, 'chunk_buf'):
                        pilot.chunk_buf = None
                        pilot.chunk_step = 0

                    x0      = tXUi[1:11, s_idx].copy()
                    t_start = float(tXUi[0, s_idx])
                    t_end   = tf

                    _t = time.time()
                    result    = simulator.simulate(
                        policy=pilot, t0=t_start, tf=t_end, x0=x0,
                        obj=np.zeros((18, 1)), query=obj_name,
                        vision_processor=None, verbose=False,
                    )
                    Xro       = result[1]
                    goal_dist = float(np.linalg.norm(Xro[:3, -1] - obj_target))

                    pc_ev = scene_data.get("epcds_arr", np.zeros((0, 3)))
                    ev = _evaluate_run(Xro, obj_target, pc_ev,
                                       env_min=scene_data.get("env_min"),
                                       env_max=scene_data.get("env_max"),
                                       tXUi=tXUi, idx0=int(s_idx))
                    success  = ev["success"]
                    collided = ev["collision"]
                    fov_ok   = ev["goal_in_fov"]
                    pos_dev = ev.get("mean_pos_dev", float('nan'))
                    ori_dev = ev.get("mean_orient_dev_deg", float('nan'))
                    fov_p   = ev.get("fov_pct", float('nan'))
                    goal_dists.append(goal_dist)
                    successes.append(success)
                    collisions.append(collided)
                    status = "✓" if success else ("💥" if collided else ("👁" if not fov_ok else "✗"))
                    dev_str = f"  pos_dev={pos_dev:.2f}m  ori_dev={ori_dev:.1f}°  fov={fov_p:.0%}" if not np.isnan(pos_dev) else ""
                    print(f"  [{label}] {status}  '{obj_name[:20]}'  "
                          f"run {run_i+1}/{max_trajectories}"
                          f"  goal_dist={goal_dist:.2f}m{dev_str}  ({time.time()-_t:.1f}s)")

                sr   = float(np.mean(successes))
                cr   = float(np.mean(collisions))
                gd   = float(np.mean(goal_dists))
                gd_s = float(np.std(goal_dists))
                print(f"  [{label}] ── '{obj_name[:25]}'  "
                      f"success={sr:.0%}  collision={cr:.0%}  "
                      f"goal_dist={gd:.2f}±{gd_s:.2f}m")
                label_results[obj_name] = {
                    "goal_dist":      gd,
                    "goal_dist_std":  gd_s,
                    "goal_dist_min":  float(np.min(goal_dists)),
                    "success_rate":   sr,
                    "collision_rate": cr,
                    "n_eval":         max_trajectories,
                }

            torch.cuda.empty_cache()

        # Aggregate across objects
        all_gd = [v["goal_dist"] for v in label_results.values()]
        all_sr = [v["success_rate"] for v in label_results.values()]
        all_cr = [v["collision_rate"] for v in label_results.values()]
        label_results["__overall__"] = {
            "goal_dist":      float(np.mean(all_gd)) if all_gd else np.nan,
            "success_rate":   float(np.mean(all_sr)) if all_sr else np.nan,
            "collision_rate": float(np.mean(all_cr)) if all_cr else np.nan,
        }
        all_results[label] = label_results

        del pilot
        gc.collect()
        torch.cuda.empty_cache()

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[CrossBenchmark] COMPARISON TABLE")
    print(f"  seed={benchmark_seed}  n={max_trajectories}/obj  "
          f"(second half of tXUi, held out from BC training)")
    print("=" * 70)

    # Collect all object names (excluding __overall__)
    all_objs = []
    for label_res in all_results.values():
        for k in label_res:
            if k != "__overall__" and k not in all_objs:
                all_objs.append(k)

    col_w = 20
    header = f"{'Object':<{col_w}}" + "".join(
        f"  {m['label'][:18]:>18}" for m in models
    )
    print(header)
    print("-" * len(header))

    for obj_name in all_objs + ["__overall__"]:
        display = "OVERALL" if obj_name == "__overall__" else obj_name[:col_w]
        row = f"{display:<{col_w}}"
        for m in models:
            r = all_results.get(m["label"], {}).get(obj_name)
            if r is None:
                row += f"  {'N/A':>18}"
            else:
                cell = f"{r['goal_dist']:.2f}m {r['success_rate']:.0%}"
                row += f"  {cell:>18}"
        print(row)
    print("  (format: mean_goal_dist  success_rate)")
    print()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[CrossBenchmark] Results saved → {output_path}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Per-iteration full-trajectory evaluation (no model swap, no rrt_backup)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_full_trajectories(
    pilot,
    flights: list,
    scenes_cfg_dir: str,
    label: str = "eval",
    vision_processor=None,
    n_eval: int = 1,
    eval_seed: int = None,
    cohort_path: str = None,
) -> dict:
    """
    Run n_eval full-trajectory simulations per object, starting from uniformly
    sampled positions along tXUi.  Returns per-object success_rate + goal_dist
    stats for honest per-iteration progress tracking.
    """
    # Fix Bug 1: use fixed seed for per-iter eval so results are comparable
    # across iterations AND match the final benchmark distribution.
    if eval_seed is not None:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)

    results = {}
    for scene_name, _ in flights:
        scene_data  = _get_scene(scene_name, scenes_cfg_dir)
        simulator   = scene_data["simulator"]
        obj_targets = scene_data["obj_targets"]
        queries     = scene_data["queries"]

        for obj_idx, obj_name in enumerate(queries):
            pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if pkl_data is None:
                continue
            tXUi_default = pkl_data["tXUi"]
            obj_target = (
                obj_targets[obj_idx] if obj_idx < len(obj_targets)
                else pkl_data.get("obj_loc", np.zeros(3))
            )

            # Multi-branch eval: each run tests a different route from t=0
            all_branches = _load_all_branches(
                scene_name, obj_name, cohort_path,
                scenes_cfg_dir, tXUi_default,
            )
            n_available = len(all_branches)
            n_sample = min(n_eval, n_available)
            branch_idxs = np.random.choice(n_available, size=n_sample, replace=False)
            if n_eval > n_available:
                extra = np.random.choice(n_available, size=n_eval - n_available, replace=True)
                branch_idxs = np.concatenate([branch_idxs, extra])

            goal_dists, successes, collisions = [], [], []
            pos_devs_all, ori_devs_all, fov_pcts_all = [], [], []
            for run_i, br_idx in enumerate(branch_idxs):
                # Reset pilot history between eval runs
                pilot.hy_flag = False
                pilot.hy_idx = 0
                pilot.DxU.zero_()
                if hasattr(pilot, 'Znn'):
                    pilot.Znn.zero_()

                tXUi = all_branches[br_idx]
                t_start = float(tXUi[0, 0])
                t_end   = float(tXUi[0, -1])
                x0      = tXUi[1:11, 0].copy()
                _t = time.time()
                result    = simulator.simulate(
                    policy=pilot, t0=t_start, tf=t_end, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=vision_processor, verbose=False,
                )
                Xro       = result[1]
                pc_ev = scene_data.get("epcds_arr", np.zeros((0,3)))
                ev = _evaluate_run(Xro, obj_target, pc_ev,
                                   env_min=scene_data.get("env_min"),
                                   env_max=scene_data.get("env_max"),
                                   tXUi=tXUi, idx0=0)
                goal_dist = ev["goal_dist"]
                collided_ev = ev["collision"]
                success   = ev["success"]
                fov_ok    = ev["goal_in_fov"]
                goal_dists.append(goal_dist)
                successes.append(success)
                collisions.append(collided_ev)
                oob = ev.get("out_of_bounds", False)
                pos_dev = ev.get("mean_pos_dev", float('nan'))
                ori_dev = ev.get("mean_orient_dev_deg", float('nan'))
                fov_p   = ev.get("fov_pct", float('nan'))
                pos_devs_all.append(pos_dev)
                ori_devs_all.append(ori_dev)
                fov_pcts_all.append(fov_p)
                status = "✓" if success else ("🚧" if oob else ("💥" if collided_ev else ("👁" if not fov_ok else "✗")))
                oob_str = "  OOB!" if oob else ""
                dev_str = f"  pos_dev={pos_dev:.2f}m  ori_dev={ori_dev:.1f}°  fov={fov_p:.0%}" if not np.isnan(pos_dev) else ""
                print(f"  [{label}] {status}  '{obj_name[:20]}'  "
                      f"run {run_i+1}/{n_eval}  goal_dist={goal_dist:.2f}m  "
                      f"min={ev['min_goal_dist']:.2f}m  coll={collided_ev}  "
                      f"fov={'✓' if fov_ok else '✗'}{oob_str}{dev_str}  ({time.time()-_t:.1f}s)")

            sr        = sum(successes) / len(successes)
            cr        = sum(collisions) / len(collisions)
            mean_dist = float(np.mean(goal_dists))
            std_dist  = float(np.std(goal_dists))
            avg_pd = float(np.nanmean(pos_devs_all))
            avg_od = float(np.nanmean(ori_devs_all))
            avg_fp = float(np.nanmean(fov_pcts_all))
            dev_summary = f"  pos_dev={avg_pd:.2f}m  ori_dev={avg_od:.1f}°  fov={avg_fp:.0%}" if not np.isnan(avg_pd) else ""
            print(f"  [{label}] ── '{obj_name[:25]}'  "
                  f"success={sr:.0%}  collision={cr:.0%}  "
                  f"goal_dist={mean_dist:.2f}±{std_dist:.2f}m  "
                  f"best={float(np.min(goal_dists)):.2f}m{dev_summary}")
            results[obj_name] = {
                "goal_dist":      mean_dist,
                "goal_dist_std":  std_dist,
                "goal_dist_min":  float(np.min(goal_dists)),
                "success":        sr >= 0.5,
                "success_rate":   sr,
                "collision_rate": cr,
                "n_eval":         n_eval,
                "mean_pos_dev":   avg_pd,
                "mean_orient_dev_deg": avg_od,
                "fov_pct":        avg_fp,
            }

        torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Collecte rollout DAgger
# ──────────────────────────────────────────────────────────────────────────────

def _collect_dagger_rollout(
    simulator,          # Simulator (depuis _SCENE_CACHE)
    mixed_policy,       # MixedPolicy
    perturbation: dict,
    tXUi: np.ndarray,
    obj_name: str,
    point_cloud,
    obj_target: np.ndarray,
    collision_threshold: float,
    drift_threshold: float,
    vision_processor,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> dict:
    mixed_policy.reset_annotations()

    # Allow caller to specify a sub-window of the trajectory (e.g. 2s segments).
    # If not given, fall back to the full trajectory.
    if t_start is None:
        t_start = float(tXUi[0, 0])
    if t_end is None:
        t_end = float(tXUi[0, -1])

    # Extract x0: use perturbation if provided (may include position noise),
    # otherwise fall back to tXUi reference.
    idx0 = int(np.searchsorted(tXUi[0, :], t_start))
    idx0 = min(idx0, tXUi.shape[1] - 1)
    if perturbation is not None and "x0" in perturbation:
        x0 = np.array(perturbation["x0"], dtype=float)[:10].copy()
    else:
        x0 = tXUi[1:11, idx0].copy()  # state is rows 1-10 (nx=10)

    t0 = t_start
    tf = t_end

    print(f"  [rollout] ▶ '{obj_name}'  t=[{t0:.2f},{tf:.2f}]s  x0_pos={x0[:3]}  β={mixed_policy.beta:.3f}")
    _t_rollout = time.time()

    # FIX: Simulator.simulate() retourne (Tro, Xro, Uro, ...)
    # signature: simulate(policy, t0, tf, x0, obj=None, query=None, vision_processor=None, ...)
    result = simulator.simulate(
        policy=mixed_policy,
        t0=t0,
        tf=tf,
        x0=x0,
        obj=np.zeros((18, 1)),
        query=obj_name,
        vision_processor=vision_processor,
        verbose=False,
    )
    # simulate() retourne un tuple (Tro, Xro, Uro, ...) selon simulator.py
    Tro, Xro = result[0], result[1]
    Uro = result[2] if len(result) > 2 else None
    print(f"  [rollout] ✅ simulate done in {time.time()-_t_rollout:.1f}s  Tro={Tro.shape}  annotations so far={len(mixed_policy.annotations)}")

    pc = (np.concatenate(point_cloud, axis=0)
          if isinstance(point_cloud, list) and len(point_cloud) > 0
          else point_cloud if isinstance(point_cloud, np.ndarray)
          else np.zeros((0, 3)))

    collision_steps, drift_steps = [], []

    if pc.shape[0] > 0:
        pc_t  = torch.from_numpy(pc).float().to(DEVICE)
        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
        T     = Xro_t.shape[0]

        dists           = torch.cdist(Xro_t, pc_t)
        collision_steps = (dists < collision_threshold).any(dim=1)\
                          .nonzero(as_tuple=True)[0].cpu().tolist()

        # FIX: align reference slice to window start (idx0), not t=0.
        # Without this, windows starting mid-trajectory compare the drone's
        # positions against the reference at t=0, which is always wrong.
        ref_end = min(idx0 + T, tXUi.shape[1])
        ref_len = ref_end - idx0
        ref_t   = torch.from_numpy(tXUi[1:4, idx0:ref_end].T).float().to(DEVICE)
        T_drift = min(T, ref_len)
        drift_steps = (
            torch.norm(Xro_t[:T_drift] - ref_t[:T_drift], dim=1) > drift_threshold
        ).nonzero(as_tuple=True)[0].cpu().tolist()

        del pc_t, Xro_t, dists, ref_t
        torch.cuda.empty_cache()
    else:
        for i, x in enumerate(Xro.T):
            # FIX: offset reference index by idx0 for window alignment
            ref = tXUi[1:4, min(idx0 + i, tXUi.shape[1] - 1)]
            if np.linalg.norm(x[:3] - ref) > drift_threshold:
                drift_steps.append(i)

    goal_dist_final = float(np.linalg.norm(Xro[:3, -1] - obj_target))
    analysis = {
        "collision": bool(collision_steps),
        "success": goal_dist_final < 2.0 and not bool(collision_steps),
        "clearance_series": None,
        "goal_in_camera_fov_series": None,
        "total_reward": -goal_dist_final,
        "min_clearance": None,
    }

    # Compute per-annotation clearance (distance to nearest obstacle)
    # for collision-weighted retraining
    annotations_out = mixed_policy.annotations.copy()
    if pc.shape[0] > 0 and annotations_out:
        from scipy.spatial import cKDTree
        kdtree = cKDTree(pc)
        for ann in annotations_out:
            pos = np.array(ann.get("x", np.zeros(10)))[:3]
            dist, _ = kdtree.query(pos, k=1)
            ann["clearance"] = float(dist)

    return {
        "Tro":             Tro,
        "Xro":             Xro,
        "Uro":             Uro,
        "annotations":     annotations_out,
        "collision_steps": collision_steps,
        "drift_steps":     drift_steps,
        "analysis":        analysis,
        "obj_name":        obj_name,
    }


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle in degrees between two quaternions (scalar-first: w,x,y,z)."""
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _filter_deviation_annotations(
    annotations: List[dict],
    Xro: np.ndarray,
    tXUi: np.ndarray,
    obj_target: np.ndarray,
    idx0: int,
    deviation_threshold: float = 0.3,
    close_approach_dist: float = 5.0,
    collision_steps: Optional[List[int]] = None,
    max_goal_dist: float = float('inf'),
    max_deviation_dist: float = float('inf'),
    orientation_deviation_deg: Optional[float] = None,
    max_orientation_dev_deg: float = 180.0,
) -> List[dict]:
    """
    Filter full-trajectory DAgger annotations to keep only useful ones:
      - States where the pilot deviated from the reference tXUi (> deviation_threshold).
      - States where the pilot's orientation deviated from reference (> orientation_deviation_deg).
      - States where the drone was within close_approach_dist of the goal.
      - Hard cutoff: discard all annotations at or after the first collision step,
        since post-crash drone physics diverge and those states are garbage.
      - max_goal_dist: discard annotations where drone is more than this distance
        from the goal (catches runaway trajectories where drone flies away from target).
      - max_deviation_dist: discard annotations where drone deviated MORE than this
        from the reference trajectory (catches extreme altitude/position excursions
        that are physically outside the scene and would corrupt Commander training).
      - max_orientation_dev_deg: discard annotations where orientation deviated MORE
        than this (catches flipped/tumbling states).

    This discards the majority of "fly straight from 25m away" timesteps that
    would otherwise corrupt BC fine-grained approach behaviour.
    """
    T = len(annotations)
    if T == 0:
        return annotations

    # Hard cutoff at first collision — post-crash positions are physically invalid.
    cutoff = T
    if collision_steps:
        first_coll = int(collision_steps[0])
        cutoff = min(first_coll, T)

    # Trajectory-level runaway detection: if no collision and the drone's final
    # position is beyond max_goal_dist, this is a runaway trajectory.
    # For runaway trajectories we only keep near-goal states (< close_approach_dist),
    # NOT deviation-based states (those just teach the model to fly away).
    is_runaway = (not collision_steps) and max_goal_dist < float('inf') and (
        Xro.shape[1] > 0 and
        np.linalg.norm(Xro[:3, min(cutoff - 1, Xro.shape[1] - 1)] - obj_target) > max_goal_dist
    )

    # Hard cutoff at exclusion zone entry — once the drone reaches the goal,
    # the mission is accomplished. Post-success divergence from reference is
    # expected and should NOT generate annotations.
    goal_entry_cutoff = cutoff
    for i in range(min(T, cutoff)):
        if i >= Xro.shape[1]:
            break
        if np.linalg.norm(Xro[:3, i] - obj_target) <= SUCCESS_RADIUS:
            goal_entry_cutoff = i
            break

    keep: set = set()
    for i in range(min(T, cutoff, goal_entry_cutoff)):
        if i >= Xro.shape[1]:
            break
        pos = Xro[:3, i]

        # Compute reference position and deviation distance (used in multiple checks below)
        ref_idx = min(idx0 + i, tXUi.shape[1] - 1)
        ref_pos = tXUi[1:4, ref_idx]
        dev_dist = float(np.linalg.norm(pos - ref_pos))

        # Compute orientation deviation (quaternion angle)
        # Xro layout: pos=0:3, vel=3:6, quat=6:10 (scalar first)
        # tXUi layout: time=0, pos=1:4, vel=4:7, quat=7:11 (scalar first)
        orient_dev = 0.0
        if orientation_deviation_deg is not None or max_orientation_dev_deg < 180.0:
            if Xro.shape[0] >= 10 and tXUi.shape[0] >= 11:
                quat_actual = Xro[6:10, i]
                quat_ref = tXUi[7:11, ref_idx]
                orient_dev = _quat_angle_deg(quat_actual, quat_ref)

        # Discard if drone has gone completely off-course (extreme altitude or position
        # excursion). These states are outside the scene bounds and train the Commander
        # to produce destabilising commands.
        if dev_dist > max_deviation_dist:
            continue

        # Discard if orientation is extreme (flipped/tumbling)
        if orient_dev > max_orientation_dev_deg:
            continue

        # Discard if drone is too far from goal (per-step runaway filter)
        goal_dist = np.linalg.norm(pos - obj_target)
        if goal_dist > max_goal_dist:
            continue

        # Keep near-goal states (approach phase, but not inside exclusion zone)
        if goal_dist < close_approach_dist:
            keep.add(i)
            continue

        # For runaway trajectories: skip deviation-based annotations
        if is_runaway:
            continue

        # Keep if drone deviated from reference trajectory at this timestep
        # (position OR orientation deviation triggers keep)
        if dev_dist > deviation_threshold:
            keep.add(i)
        elif orientation_deviation_deg is not None and orient_dev > orientation_deviation_deg:
            keep.add(i)

    # Build filtered list
    filtered = []
    for i in sorted(keep):
        filtered.append(annotations[i])

    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Métriques / Agrégation / Re-entraînement / W&B
# ──────────────────────────────────────────────────────────────────────────────

def _compute_dagger_metrics(
    rollouts: List[dict], iteration: int, beta: float, n_annotations: int = 0,
) -> dict:
    total      = len(rollouts)
    goal_dists = [-r["analysis"].get("total_reward", 0.0) for r in rollouts]
    return {
        "iteration":          iteration,
        "beta":               beta,
        "total_rollouts":     total,
        "collision_rate":     sum(1 for r in rollouts if r["collision_steps"]) / max(total, 1),
        "success_rate":       sum(1 for r in rollouts if r["analysis"].get("success", False)) / max(total, 1),
        "window_goal_dist":   float(np.mean(goal_dists)) if goal_dists else 0.0,
        "n_annotations":      n_annotations,
    }


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
                     lim_sv=lim_sv, lr=lr, batch_size=64, course_name=course)


def _wandb_log_iteration(pilot_name: str, m: dict, iteration: int) -> None:
    try:
        import wandb
        if wandb.run is not None: wandb.log({
            f"dagger/{pilot_name}/beta":             m["beta"],
            f"dagger/{pilot_name}/collision_rate":   m["collision_rate"],
            f"dagger/{pilot_name}/success_rate":     m["success_rate"],
            f"dagger/{pilot_name}/window_goal_dist": m.get("window_goal_dist", 0.0),
            f"dagger/{pilot_name}/n_annotations":    m.get("n_annotations", 0),
            f"dagger/{pilot_name}/total_rollouts":   m["total_rollouts"],
            "dagger/iteration":                       m["iteration"],
        })
    except Exception as e:
        print(f"  [WARN] wandb: {e}")


def _wandb_log_benchmark(pilot_name: str, before: dict, after: dict) -> None:
    try:
        import wandb
        fin = lambda x: x[np.isfinite(x)]
        if wandb.run is None:
            return

        b_sr = before.get("success_rate", np.zeros(1))
        a_sr = after.get("success_rate", np.zeros(1))
        b_gd = before.get("goal_dist", np.zeros(1))
        a_gd = after.get("goal_dist", np.zeros(1))

        wandb.log({
            f"benchmark/{pilot_name}/before/collision_rate": before["collision_rate"].mean(),
            f"benchmark/{pilot_name}/before/success_rate":   b_sr.mean(),
            f"benchmark/{pilot_name}/before/goal_dist_mean": np.nanmean(b_gd),
            f"benchmark/{pilot_name}/before/return_mean":    fin(before["return_sum"]).mean(),
            f"benchmark/{pilot_name}/after/collision_rate":  after["collision_rate"].mean(),
            f"benchmark/{pilot_name}/after/success_rate":    a_sr.mean(),
            f"benchmark/{pilot_name}/after/goal_dist_mean":  np.nanmean(a_gd),
            f"benchmark/{pilot_name}/after/return_mean":     fin(after["return_sum"]).mean(),
            f"benchmark/{pilot_name}/delta/success_rate":    a_sr.mean() - b_sr.mean(),
            f"benchmark/{pilot_name}/delta/collision_rate":
                after["collision_rate"].mean() - before["collision_rate"].mean(),
            f"benchmark/{pilot_name}/delta/goal_dist":
                np.nanmean(a_gd) - np.nanmean(b_gd),
        })

        # Summary table for benchmark comparison
        table = wandb.Table(columns=["metric", "before", "after", "delta"])
        table.add_data("success_rate", f"{b_sr.mean():.1%}", f"{a_sr.mean():.1%}",
                       f"{(a_sr.mean()-b_sr.mean())*100:+.1f}pp")
        table.add_data("collision_rate", f"{before['collision_rate'].mean():.1%}",
                       f"{after['collision_rate'].mean():.1%}",
                       f"{(after['collision_rate'].mean()-before['collision_rate'].mean())*100:+.1f}pp")
        table.add_data("goal_dist", f"{np.nanmean(b_gd):.2f}m", f"{np.nanmean(a_gd):.2f}m",
                       f"{np.nanmean(a_gd)-np.nanmean(b_gd):+.2f}m")
        wandb.log({f"benchmark/{pilot_name}/summary": table})

        # Per-object benchmark table
        b_obj = before.get("per_object", {})
        a_obj = after.get("per_object", {})
        if b_obj and a_obj:
            obj_table = wandb.Table(columns=["object", "before_sr", "after_sr", "delta_sr",
                                              "before_cr", "after_cr", "before_gd", "after_gd"])
            for obj_name in b_obj:
                if obj_name in a_obj:
                    short = obj_name.split()[-1] if len(obj_name) > 15 else obj_name
                    b, a = b_obj[obj_name], a_obj[obj_name]
                    obj_table.add_data(
                        short,
                        f"{b['success_rate']:.0%}", f"{a['success_rate']:.0%}",
                        f"{(a['success_rate']-b['success_rate'])*100:+.1f}pp",
                        f"{b['collision_rate']:.0%}", f"{a['collision_rate']:.0%}",
                        f"{b['goal_dist']:.2f}m", f"{a['goal_dist']:.2f}m",
                    )
            wandb.log({f"benchmark/{pilot_name}/per_object": obj_table})

    except Exception as e:
        print(f"  [WARN] wandb benchmark: {e}")


def _print_benchmark_comparison(before: dict, after: dict, pilot_name: str) -> None:
    fin = lambda x: x[np.isfinite(x)]
    def _s(m):
        cr  = m["collision_rate"]
        sr  = m.get("success_rate",  np.zeros_like(cr))
        gd  = m.get("goal_dist",     np.full_like(cr, np.nan))
        clr = fin(m["clearance_mean"])
        fov = fin(m["fov_rate"])
        print(f"\n  ── {m['label']} ({len(cr)} rollouts) ──")
        print(f"    collision_rate : {cr.mean()*100:.1f}%  ({int(cr.sum())}/{len(cr)})")
        print(f"    success_rate   : {sr.mean()*100:.1f}%  ({int(sr.sum())}/{len(sr)})")
        print(f"    goal_dist_mean : {np.nanmean(gd):.2f} m")
        print(f"    return mean    : {fin(m['return_sum']).mean():.1f}")
        if clr.size:
            print(f"    clearance_mean : {clr.mean():.3f} m")
    print("\n" + "=" * 62)
    print(f"  BENCHMARK DAgger — {pilot_name}")
    print("=" * 62)
    _s(before); _s(after)
    b_gd = np.nanmean(before.get("goal_dist", np.array([np.nan])))
    a_gd = np.nanmean(after.get("goal_dist",  np.array([np.nan])))
    for name, delta, better_if in [
        ("collision_rate", (after["collision_rate"].mean()  - before["collision_rate"].mean())*100,  "<"),
        ("success_rate",   (after.get("success_rate", np.zeros(1)).mean() -
                            before.get("success_rate", np.zeros(1)).mean())*100,                     ">"),
        ("goal_dist_mean", a_gd - b_gd,                                                              "<"),
        ("return_mean",    fin(after["return_sum"]).mean() - fin(before["return_sum"]).mean(),        ">"),
    ]:
        ok = (delta < 0) if better_if == "<" else (delta > 0)
        unit = "pp" if "rate" in name else ("m" if "dist" in name else "")
        print(f"  Δ {name:<15}: {delta:+.2f}{unit}  {'✓ better' if ok else '✗ worse'}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Expert factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_expert(
    expert_type: str,
    tXUi:        np.ndarray,
    obj_target:  np.ndarray,
    obj_idx:     int,
    scene_data:  dict,
    scene_cfg:   dict,
    policy_name: str,
    frame_name:  str,
    pilot_name:  str,
    x0_start:    np.ndarray = None,  # perturbed start pos (3,) — used by rrt_mpc
):
    """
    Return the expert controller for this DAgger iteration segment.
      "mpc"      – VehicleRateMPC tracking the BC reference trajectory (recovery-to-ref)
      "potential" – PotentialFieldExpert (goal-seeking + obstacle avoidance)
      "rrt"      – OnlineRRTExpert (RRT* replanning + pure-pursuit geometric controller)
    """
    if expert_type == "potential":
        return PotentialFieldExpert(
            goal=obj_target,
            point_cloud=scene_data["epcds_arr"],
        )
    elif expert_type == "rrt":
        return OnlineRRTExpert(
            goal=obj_target,
            point_cloud=scene_data["epcds_arr"],
            scene_cfg=scene_cfg,
            obj_idx=obj_idx,
            replan_interval=2.0,
        )
    else:   # "mpc" — default, original behaviour
        return VehicleRateMPC(tXUi, policy_name, frame_name, pilot_name)


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale DAgger
# ──────────────────────────────────────────────────────────────────────────────

def train_dagger_policy(
    cohort_name: str,
    method_name: str,
    roster: List[str],
    flights: List[Tuple[str, str]],
    n_iterations: int          = 10,
    beta_start: float          = 0.7,
    beta_decay: float          = 0.85,
    collision_threshold: float = 0.15,
    drift_threshold: float     = 2.0,
    Nep_per_iter: int          = 50,
    lim_sv: int                = 10,
    max_trajectories: int      = 10,
    n_eval_per_iter: int       = 10,
    benchmark_seed: int        = 42,
    use_wandb: bool            = False,
    wandb_project: str         = "singer-dagger",
    wandb_run_name: str        = "dagger",
    expert_type: str           = "mpc",
    aggregate_dagger: bool     = False,
    start_pos_noise: float     = 0.3,
    n_rollouts_per_object: int = 5,
    deviation_filter_dist: float = 0.3,
    close_approach_dist: float   = 5.0,
    max_annotation_goal_dist: float = 50.0,
    max_deviation_dist: float  = float('inf'),
    orientation_deviation_deg: float = None,
    max_orientation_dev_deg: float = 180.0,
    dagger_lr: float           = 1e-5,
    bc_cohort_name: str        = None,
    eval_seed: int             = None,
    reset_to_best: bool        = False,
    patience: int              = 2,
    dagger_only: bool          = False,
    dagger_oversample: int     = 1,
    # ── V10 enhancements (backward compatible — defaults disable them) ──
    ewc_lambda: float          = 0.0,
    lr_schedule: str           = None,
    lr_decay_per_iter: float   = 1.0,
    weight_decay: float        = 0.0,
    # ── Collision-weighted loss (V10+) ──
    collision_weight_alpha: float = 0.0,
    collision_weight_threshold: float = 0.5,
    # ── Data ratio control (V11+) ──
    max_dagger_samples: int = 0,
) -> dict:

    use_ewc = ewc_lambda > 0.0
    if use_ewc:
        print(f"[DAgger] EWC enabled: lambda={ewc_lambda}")
    if lr_schedule:
        print(f"[DAgger] LR schedule: {lr_schedule}")
    if lr_decay_per_iter != 1.0:
        print(f"[DAgger] LR decay per iter: {lr_decay_per_iter}")

    print(f"[DAgger] Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"[DAgger] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[DAgger] VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    workspace_path = str(Path(__file__).resolve().parents[3])
    cohort_path    = os.path.join(workspace_path, "cohorts", cohort_name)
    method_path    = os.path.join(workspace_path, "configs", "method", method_name + ".json")
    scenes_cfg_dir = os.path.join(workspace_path, "configs", "scenes")

    sim_base = os.path.join(cohort_path, "simulation_data", "dagger")
    os.makedirs(sim_base, exist_ok=True)
    print(f"[DAgger] sim_base isolé → {sim_base}")

    all_metrics: dict = {name: [] for name in roster}

    # ── Configs scènes ────────────────────────────────────────────────────────
    objective_configs   = {}
    collision_detectors = {}
    scene_names         = []
    for scene_name, _ in flights:
        if scene_name in objective_configs:
            continue
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            objective_configs[scene_name] = yaml.safe_load(f)
        scene_names.append(scene_name)

    with open(method_path) as _mf:
        _mcfg = json.load(_mf)
    _vp_type = (
        _mcfg.get("sample_set", {}).get("vision_processor_type")
        or _mcfg.get("vision_processor_type")
    )
    # FIX : policy et frame sont dans sample_set
    _base_policy_name = (
        _mcfg.get("sample_set", {}).get("policy")
        or _mcfg.get("policy", "vrmpc_rrt")
    )
    _base_frame_name = (
        _mcfg.get("sample_set", {}).get("frame")
        or _mcfg.get("frame", "carl")
    )
    _base_rollout_name = (
        _mcfg.get("sample_set", {}).get("rollout")
        or _mcfg.get("rollout", "baseline")
    )
    _Tdt_ro = _mcfg.get("sample_set", {}).get("duration", 2.0)
    print(f"[DAgger] VehicleRateMPC policy='{_base_policy_name}' frame='{_base_frame_name}' rollout='{_base_rollout_name}'")
    print(f"[DAgger] Mode            : Full-trajectory + deviation filter (Option B)")
    print(f"[DAgger] Rollouts/object : {n_rollouts_per_object} branches per iteration")
    print(f"[DAgger] Per-iter eval   : {n_eval_per_iter} runs/object  |  Benchmark: {max_trajectories} runs/object")
    print(f"[DAgger] Aggregation     : {'cumulative' if aggregate_dagger else 'online (per-iter only)'}")
    print(f"[DAgger] Start-pos noise : ±{start_pos_noise}m")
    _orient_str = f" OR orient_dev>{orientation_deviation_deg}°" if orientation_deviation_deg else ""
    print(f"[DAgger] Ann filter      : keep if drift>{deviation_filter_dist}m{_orient_str} OR goal_dist<{close_approach_dist}m  |  discard if goal_dist>{max_annotation_goal_dist}m or orient>{max_orientation_dev_deg}°")

    vision_processor = create_vision_processor(_vp_type)
    if vision_processor is not None and hasattr(vision_processor, "to"):
        vision_processor = vision_processor.to(DEVICE)
        print(f"[DAgger] vision_processor → {DEVICE}")

    # ── PERF : précharger gsplat + trajectoires UNE SEULE FOIS ──────────────
    print("\n[DAgger] ⏳ Préchargement scènes + trajectoires (1 seule fois pour tout le run)...")
    for scene_name in scene_names:
        _get_scene(scene_name, scenes_cfg_dir, _base_frame_name, _base_rollout_name)

    # Load trajectories from BC rollout data (authoritative source)
    n_bc_traj = 0
    if bc_cohort_name:
        n_bc_traj = _preload_bc_trajectories(bc_cohort_name, flights, scenes_cfg_dir)
        print(f"[DAgger] ✅ {n_bc_traj} BC trajectories loaded as reference branches")
    else:
        # Fallback to pkl files if no BC cohort specified
        n_pkls = _preload_all_pkls(flights, scenes_cfg_dir)
        print(f"[DAgger] ✅ {n_pkls} pkl en cache (no bc_cohort specified)")
    print(f"[DAgger] ✅ {len(_SCENE_CACHE)} scène(s) loaded\n")

    # ── Boucle par pilot ─────────────────────────────────────────────────────
    for pilot_name in roster:
        print(f"\n{'='*62}")
        print(f"[DAgger] Pilot : {pilot_name}  ({DEVICE})")
        print(f"{'='*62}")
        print(f"  Iterations   : {n_iterations}")
        print(f"  β_start      : {beta_start}  decay={beta_decay}")
        print(f"  Nep/iter     : {Nep_per_iter}")
        print(f"  collision_th : {collision_threshold} m")
        print(f"  drift_th     : {drift_threshold} m")
        print(f"  expert_type  : {expert_type}")
        if bc_cohort_name:
            print(f"  bc_cohort    : {bc_cohort_name}")

        # ── Init DAgger cohort roster from BC cohort if model not present ──
        if bc_cohort_name:
            dag_roster_dir = os.path.join(cohort_path, "roster", pilot_name)
            dag_model_path = os.path.join(dag_roster_dir, "model.pth")
            bc_roster_dir  = os.path.join(workspace_path, "cohorts", bc_cohort_name, "roster", pilot_name)
            bc_model_path  = os.path.join(bc_roster_dir, "model.pth")
            if not os.path.isfile(dag_model_path) and os.path.isfile(bc_model_path):
                os.makedirs(dag_roster_dir, exist_ok=True)
                for fname in ("model.pth", "last_model.pth", "config.json", "losses_Commander.pt"):
                    src = os.path.join(bc_roster_dir, fname)
                    dst = os.path.join(dag_roster_dir, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        print(f"[DAgger] Copied BC pilot file: {fname}")
                print(f"[DAgger] ✅ DAgger cohort initialized from BC cohort '{bc_cohort_name}'")
            elif os.path.isfile(dag_model_path):
                print(f"[DAgger] DAgger cohort roster already exists — skipping BC copy")
            else:
                print(f"[DAgger] WARNING: bc_cohort model not found at {bc_model_path}")

        pilot = Pilot(cohort_name, pilot_name)
        pilot.set_mode("deploy")
        pilot.model.to(DEVICE)

        dagger_dir  = os.path.join(cohort_path, "dagger_data", pilot_name)
        run_ts      = time.strftime("%Y%m%d_%H%M%S")
        bench_dir   = os.path.join(cohort_path, "training_benchmarks", run_ts)
        rrt_backup  = os.path.join(dagger_dir, "_benchmark_rrt_backup")
        os.makedirs(bench_dir,  exist_ok=True)
        os.makedirs(rrt_backup, exist_ok=True)

        # ── Model management: model.pth is always the "current best" ──
        # Save the pre-DAgger model as model_before_dagger.pth (in roster dir)
        roster_dir = os.path.join(cohort_path, "roster", pilot_name)
        roster_model_path = os.path.join(roster_dir, "model.pth")
        model_before_path = os.path.join(roster_dir, "model_before_dagger.pth")
        if os.path.isfile(roster_model_path):
            shutil.copy2(roster_model_path, model_before_path)
            print(f"[DAgger] Saved pre-DAgger model → {model_before_path}")
        # Also keep a copy in benchmark dir for archival
        _save_model_checkpoint(pilot, os.path.join(bench_dir, "model_before_dagger.pth"))

        # 0 gsplat reload — utilise _PKL_CACHE
        _generate_rrt_backup(
            pilot=pilot, model_path=model_before_path,
            workspace_path=workspace_path, cohort_name=cohort_name,
            cohort_path=cohort_path, method_name=method_name,
            flights=flights, scenes_cfg_dir=scenes_cfg_dir,
            objective_configs=objective_configs,
            sim_base=sim_base, rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed, max_trajectories=max_trajectories,
        )

        # 0 gsplat reload — utilise _SCENE_CACHE
        metrics_before = _run_benchmark_pilot(
            pilot=pilot, model_path=model_before_path, label="before_dagger",
            workspace_path=workspace_path, cohort_name=cohort_name,
            cohort_path=cohort_path, method_name=method_name, flights=flights,
            scenes_cfg_dir=scenes_cfg_dir, objective_configs=objective_configs,
            collision_detectors=collision_detectors, scene_names=scene_names,
            sim_base=sim_base, rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed, max_trajectories=max_trajectories,
        )

        # ── Expert (MPC) baseline: establish gold standard deviation ──────
        print("\n[DAgger] Running expert (MPC) evaluation — gold standard deviation baseline...")
        np.random.seed(benchmark_seed)
        torch.manual_seed(benchmark_seed)
        n_expert_eval = min(20, max_trajectories)  # quick eval, 20 runs/object
        for scene_name, _ in flights:
            scene_data  = _get_scene(scene_name, scenes_cfg_dir)
            simulator   = scene_data["simulator"]
            obj_targets = scene_data["obj_targets"]
            queries     = scene_data["queries"]
            for obj_idx, obj_name in enumerate(queries):
                pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
                if pkl_data is None:
                    continue
                tXUi_default = pkl_data["tXUi"]
                obj_target = (
                    obj_targets[obj_idx] if obj_idx < len(obj_targets)
                    else pkl_data.get("obj_loc", np.zeros(3))
                )
                all_branches = _load_all_branches(
                    scene_name, obj_name, cohort_path,
                    scenes_cfg_dir, tXUi_default,
                )
                n_available = len(all_branches)
                branch_idxs = np.random.choice(n_available, size=min(n_expert_eval, n_available), replace=False)
                exp_sr, exp_pd, exp_od, exp_fp = [], [], [], []
                for br_idx in branch_idxs:
                    tXUi_br = all_branches[br_idx]
                    expert_policy = _make_expert(
                        expert_type, tXUi_br, obj_target, obj_idx,
                        scene_data, objective_configs[scene_name],
                        _base_policy_name, _base_frame_name, pilot_name,
                        x0_start=tXUi_br[1:4, 0],
                    )
                    result = simulator.simulate(
                        policy=expert_policy, t0=float(tXUi_br[0, 0]),
                        tf=float(tXUi_br[0, -1]), x0=tXUi_br[1:11, 0].copy(),
                        obj=np.zeros((18, 1)), query=obj_name,
                        vision_processor=None, verbose=False,
                    )
                    Xro_exp = result[1]
                    pc_ev = scene_data.get("epcds_arr", np.zeros((0, 3)))
                    ev_exp = _evaluate_run(Xro_exp, obj_target, pc_ev,
                                           env_min=scene_data.get("env_min"),
                                           env_max=scene_data.get("env_max"),
                                           tXUi=tXUi_br, idx0=0)
                    exp_sr.append(ev_exp["success"])
                    exp_pd.append(ev_exp.get("mean_pos_dev", float('nan')))
                    exp_od.append(ev_exp.get("mean_orient_dev_deg", float('nan')))
                    exp_fp.append(ev_exp.get("fov_pct", float('nan')))
                avg_sr = float(np.mean(exp_sr))
                avg_pd = float(np.nanmean(exp_pd))
                avg_od = float(np.nanmean(exp_od))
                avg_fp = float(np.nanmean(exp_fp))
                print(f"  [expert_mpc] ── '{obj_name[:25]}'  "
                      f"success={avg_sr:.0%}  pos_dev={avg_pd:.2f}m  ori_dev={avg_od:.1f}°  fov={avg_fp:.0%}")
            torch.cuda.empty_cache()
        print("[DAgger] Expert baseline complete — deviation values above are the ideal target\n")

        aggregated_file = os.path.join(dagger_dir, "dagger_aggregated.pt")
        # Fresh campaign: back up any aggregated file from a previous run so it
        # is not re-loaded by _aggregate_dagger_dataset (cross-campaign contamination).
        if os.path.isfile(aggregated_file):
            backup_agg = aggregated_file.replace(".pt", f"_backup_{run_ts}.pt")
            shutil.move(aggregated_file, backup_agg)
            print(f"[DAgger] Backed up previous aggregated file → {os.path.basename(backup_agg)}")

        beta, global_step = beta_start, 0
        current_lr = dagger_lr  # will be decayed by lr_decay_per_iter each iteration
        # EWC state: Fisher info + parameter snapshot (computed after first retrain)
        _fisher_dict = None
        _optpar_dict = None
        # Fix Bug 2: track best model by mean success_rate (higher = better),
        # with mean_goal_dist as tiebreaker. This prevents hard objects (drill)
        # from dominating the metric when easy objects (clock, leafblower) improve.
        best_success_rate = -1.0
        best_goal_dist    = float('inf')
        # Best model checkpoint (staging copy — model.pth is the deploy path)
        best_model_staging = os.path.join(roster_dir, "model_best_staging.pth")
        consecutive_drops = 0  # early stopping: count consecutive iters without improvement
        _patience = max(patience, 2)  # minimum 2

        # ── Boucle DAgger ─────────────────────────────────────────────────
        for iteration in range(n_iterations):
            _t_iter = time.time()
            print(f"\n[DAgger] ── Itération {iteration}/{n_iterations-1}  β={beta:.3f}  ({time.strftime('%H:%M:%S')})")

            # Fix Bug 3: reset to best model before rollout collection to prevent
            # catastrophic cascade. Each iteration collects data from the current best
            # model (not the latest potentially-degraded one). New annotations still
            # provide diversity because start_pos_noise randomises starting positions.
            if reset_to_best and iteration > 0 and os.path.isfile(best_model_staging):
                print(f"[DAgger] ↻ Resetting to best model (sr={best_success_rate:.0%} gd={best_goal_dist:.2f}m) before rollout")
                pilot = _swap_model(pilot, best_model_staging)
                pilot.set_mode("deploy")

            all_rollouts, all_annotations = [], []

            for scene_name, obj_query in flights:
                # 0 gsplat reload — _SCENE_CACHE
                scene_data  = _get_scene(scene_name, scenes_cfg_dir)
                simulator   = scene_data["simulator"]
                obj_targets = scene_data["obj_targets"]
                queries     = scene_data["queries"]

                for obj_idx, obj_name in enumerate(queries):
                    # 0 I/O disque — _PKL_CACHE
                    pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
                    if pkl_data is None:
                        continue

                    tXUi_default = pkl_data["tXUi"]
                    obj_target = (
                        obj_targets[obj_idx]
                        if obj_idx < len(obj_targets)
                        else pkl_data.get("obj_loc", np.zeros(3))
                    )

                    # Load all available branches for this object
                    all_branches = _load_all_branches(
                        scene_name, obj_name, cohort_path,
                        scenes_cfg_dir, tXUi_default,
                    )

                    # Sample n_rollouts_per_object branches (with replacement if needed)
                    n_sample = min(n_rollouts_per_object, len(all_branches))
                    branch_idxs = np.random.choice(len(all_branches), size=n_sample, replace=False)
                    print(f"  [iter{iteration:03d}] '{obj_name}': {n_sample} branches sampled from {len(all_branches)} available")

                    for br_i, br_idx in enumerate(branch_idxs):
                        tXUi = all_branches[br_idx]

                        # Reset pilot history between rollouts to prevent cross-object
                        # history contamination. Without this, the pilot's DxU buffer
                        # carries state from the previous rollout (possibly a different
                        # object), which corrupts the first few steps of annotations.
                        pilot.hy_flag = False
                        pilot.hy_idx = 0
                        pilot.DxU.zero_()
                        if hasattr(pilot, 'Znn'):
                            pilot.Znn.zero_()

                        # Option B: run the FULL trajectory (not 2s windows) so the
                        # mixed policy encounters actual navigation states, then keep
                        # only annotations at deviation / near-goal timesteps.
                        t_traj_start = float(tXUi[0, 0])
                        t_traj_end   = float(tXUi[0, -1])

                        # Perturb initial position
                        ref_idx0 = min(
                            int(np.searchsorted(tXUi[0, :], t_traj_start)),
                            tXUi.shape[1] - 1,
                        )
                        x0_ref = tXUi[1:, ref_idx0].copy()
                        if start_pos_noise > 0.0:
                            # Position noise only — vel/quat perturbations cause MPC divergence
                            env_min = scene_data.get("env_min", np.array([-1e6, -1e6, -1e6]))
                            env_max = scene_data.get("env_max", np.array([ 1e6,  1e6,  1e6]))
                            x0_ref[:3] += np.random.uniform(-start_pos_noise, start_pos_noise, size=3)
                            x0_ref[:3]  = np.clip(x0_ref[:3], env_min, env_max)
                        perturbation = {"t0": t_traj_start, "x0": x0_ref}

                        # Build expert for this branch
                        expert       = _make_expert(
                            expert_type, tXUi, obj_target, obj_idx,
                            scene_data, objective_configs[scene_name],
                            _base_policy_name, _base_frame_name, pilot_name,
                            x0_start=x0_ref[:3],
                        )
                        mixed_policy = MixedPolicy(expert, pilot, beta)

                        rollout = _collect_dagger_rollout(
                            simulator=simulator,
                            mixed_policy=mixed_policy,
                            perturbation=perturbation,
                            tXUi=tXUi,
                            obj_name=obj_name,
                            point_cloud=scene_data["epcds_arr"],
                            obj_target=obj_target,
                            collision_threshold=collision_threshold,
                            drift_threshold=drift_threshold,
                            vision_processor=vision_processor,
                            t_start=t_traj_start,
                            t_end=t_traj_end,
                        )

                        # Filter: keep only deviation + near-goal annotations,
                        # discarding all timesteps at or after the first collision,
                        # and any states where drone deviated beyond max_deviation_dist
                        # (extreme altitude / out-of-scene excursions).
                        filtered_ann = _filter_deviation_annotations(
                            annotations=rollout["annotations"],
                            Xro=rollout["Xro"],
                            tXUi=tXUi,
                            obj_target=obj_target,
                            idx0=ref_idx0,
                            deviation_threshold=deviation_filter_dist,
                            close_approach_dist=close_approach_dist,
                            collision_steps=rollout["collision_steps"],
                            max_goal_dist=max_annotation_goal_dist,
                            max_deviation_dist=max_deviation_dist,
                            orientation_deviation_deg=orientation_deviation_deg,
                            max_orientation_dev_deg=max_orientation_dev_deg,
                        )

                        # Expert-rescue: if crash happened too early (< 10 pre-crash
                        # annotations), run a β=1 expert-only backup rollout to get
                        # clean demonstrations the model never saw during BC training.
                        RESCUE_THRESHOLD = 10
                        if len(filtered_ann) < RESCUE_THRESHOLD and rollout["collision_steps"]:
                            # Fresh expert to avoid stale waypoint state from main rollout
                            rescue_expert = _make_expert(
                                expert_type, tXUi, obj_target, obj_idx,
                                scene_data, objective_configs[scene_name],
                                _base_policy_name, _base_frame_name, pilot_name,
                                x0_start=x0_ref[:3],
                            )
                            expert_only = MixedPolicy(rescue_expert, pilot, beta=1.0)
                            backup_rollout = _collect_dagger_rollout(
                                simulator=simulator,
                                mixed_policy=expert_only,
                                perturbation=perturbation,
                                tXUi=tXUi,
                                obj_name=obj_name,
                                point_cloud=scene_data["epcds_arr"],
                                obj_target=obj_target,
                                collision_threshold=collision_threshold,
                                drift_threshold=drift_threshold,
                                vision_processor=vision_processor,
                                t_start=t_traj_start,
                                t_end=t_traj_end,
                            )
                            # For expert-rescue rollouts: keep ALL annotations
                            # (the expert's path is useful to imitate entirely).
                            # Only apply collision cutoff + max_deviation_dist sanity guard.
                            backup_filtered = _filter_deviation_annotations(
                                annotations=backup_rollout["annotations"],
                                Xro=backup_rollout["Xro"],
                                tXUi=tXUi,
                                obj_target=obj_target,
                            idx0=ref_idx0,
                            deviation_threshold=0.0,   # keep all (no deviation filter)
                            close_approach_dist=1e9,   # keep all (no distance filter)
                            collision_steps=backup_rollout["collision_steps"],
                            max_deviation_dist=max_deviation_dist,
                        )
                            RESCUE_MAX_ANN = 20  # cap rescue to prevent dominating the dataset
                            backup_filtered = backup_filtered[:RESCUE_MAX_ANN]
                            print(f"  [rescue] β=1.0: {len(backup_filtered)} expert annotations added for '{obj_name[:20]}'")
                            filtered_ann = filtered_ann + backup_filtered

                        all_rollouts.append(rollout)
                        all_annotations.extend(filtered_ann)
                        _save_traj_plot(
                            rollout["Tro"], rollout["Xro"], rollout["Uro"],
                            save_path=os.path.join(
                                dagger_dir, "plots",
                                f"iter{iteration:03d}_{obj_name.replace(' ','_')}_br{br_idx:03d}.png",
                            ),
                            title=f"iter={iteration} β={beta:.2f} | {obj_name} br{br_idx} {t_traj_start:.1f}→{t_traj_end:.1f}s",
                        )

                        used  = torch.cuda.memory_allocated() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        goal_dist = -rollout["analysis"].get("total_reward", 0.0)
                        print(
                            f"  [{obj_name[:20]}"
                            f" br{br_idx} t=[{t_traj_start:.1f},{t_traj_end:.1f}]s]"
                            f"  coll={len(rollout['collision_steps'])}"
                            f"  goal_dist={goal_dist:.2f}m"
                            f"  ann_raw={len(rollout['annotations'])} → kept={len(filtered_ann)}"
                            f"  GPU={used:.1f}/{total:.0f}GB"
                        )

            # Agrégation
            mode_str = "cumulative" if aggregate_dagger else "online (replacing)"
            print(f"\n[DAgger] {mode_str} aggregation  {len(all_annotations)} new annotations...")
            _t_agg = time.time()
            agg_data = _aggregate_dagger_dataset(
                all_annotations, aggregated_file, aggregate=aggregate_dagger
            )
            torch.save(agg_data, aggregated_file)
            torch.save(all_annotations,
                       os.path.join(dagger_dir, f"dagger_iter_{iteration:03d}.pt"))
            print(f"  [agg] {len(agg_data)} samples for retraining ({mode_str})  ({time.time()-_t_agg:.1f}s)")

            # Restore best model to model.pth before retraining so fine-tune
            # starts from the best checkpoint (retrain will overwrite model.pth).
            if reset_to_best and os.path.isfile(best_model_staging):
                shutil.copy2(best_model_staging, roster_model_path)
                print(f"[DAgger] ↻ Restored best model to model.pth before retrain")

            # Re-entraînement
            print(f"[DAgger] Retraining Commander  Nep={Nep_per_iter} lim_sv={lim_sv} lr={current_lr:.2e}...")
            _t_retrain = time.time()
            if use_ewc or lr_schedule or weight_decay > 0 or collision_weight_alpha > 0 or max_dagger_samples > 0:
                _fisher_dict, _optpar_dict = _retrain_commander_ewc(
                    cohort_name, pilot_name, aggregated_file, Nep_per_iter, lim_sv,
                    lr=current_lr, bc_cohort_name=bc_cohort_name, dagger_only=dagger_only,
                    oversample=dagger_oversample, ewc_lambda=ewc_lambda,
                    fisher_dict=_fisher_dict, optpar_dict=_optpar_dict,
                    lr_schedule=lr_schedule, weight_decay=weight_decay,
                    collision_weight_alpha=collision_weight_alpha,
                    collision_weight_threshold=collision_weight_threshold,
                    max_dagger_samples=max_dagger_samples,
                )
            else:
                _retrain_commander(cohort_name, pilot_name, aggregated_file, Nep_per_iter, lim_sv, lr=current_lr,
                                   bc_cohort_name=bc_cohort_name, dagger_only=dagger_only,
                                   oversample=dagger_oversample)
            print(f"[DAgger] Retraining done in {time.time()-_t_retrain:.1f}s")
            # Decay LR for next iteration
            current_lr *= lr_decay_per_iter

            # Recharger pilot avec nouveaux poids
            pilot = Pilot(cohort_name, pilot_name)
            pilot.set_mode("deploy")
            pilot.model.to(DEVICE)

            # ── Per-segment metrics (data-collection windows) ─────────────────
            m = _compute_dagger_metrics(all_rollouts, iteration, beta,
                                        n_annotations=len(all_annotations))
            all_metrics[pilot_name].append(m)

            # ── Full-trajectory evaluation after retrain ──────────────────────
            # Runs one complete t0→tf sim per object — honest progress metric,
            # not contaminated by window-alignment artefacts.
            print(f"\n[DAgger] Full-traj eval after iter {iteration} retrain...")
            _t_eval = time.time()
            iter_eval = _eval_full_trajectories(
                pilot, flights, scenes_cfg_dir,
                label=f"iter{iteration:03d}",
                vision_processor=vision_processor,
                n_eval=n_eval_per_iter,
                eval_seed=eval_seed,
                cohort_path=cohort_path,
            )
            n_ok  = sum(1 for v in iter_eval.values() if v["success"])
            n_tot = len(iter_eval)
            print(f"[DAgger] Full-traj eval done in {time.time()-_t_eval:.1f}s  "
                  f"{n_ok}/{n_tot} objects reached")
            m["full_traj_eval"]    = iter_eval
            m["full_traj_success"] = n_ok / max(n_tot, 1)

            # Fix Bug 2: use mean success_rate as primary metric, goal_dist as tiebreaker.
            # This prevents one hard object (drill at 9.40m) from blocking a checkpoint
            # when easy objects improved (leafblower 0%→100%).
            iter_sr = float(np.mean([v["success_rate"] for v in iter_eval.values()]))
            iter_gd = float(np.mean([v["goal_dist"] for v in iter_eval.values()]))
            is_better = (
                iter_sr > best_success_rate or
                (iter_sr == best_success_rate and iter_gd < best_goal_dist)
            )
            if is_better:
                best_success_rate = iter_sr
                best_goal_dist = iter_gd
                _save_model_checkpoint(pilot, best_model_staging)  # staging backup
                _save_model_checkpoint(pilot, os.path.join(bench_dir, "model_best_dagger.pth"))  # archival
                # Also update model.pth so it's always the current best
                shutil.copy2(best_model_staging, roster_model_path)
                print(f"[DAgger] ★ New best model at iter {iteration}  sr={best_success_rate:.0%}  gd={best_goal_dist:.2f}m  → model.pth")
                consecutive_drops = 0
            else:
                consecutive_drops += 1
                # Restore best model to model.pth (retrain may have degraded it)
                if os.path.isfile(best_model_staging):
                    shutil.copy2(best_model_staging, roster_model_path)
                print(f"[DAgger] ⚠ No improvement (iter sr={iter_sr:.0%} vs best={best_success_rate:.0%})  "
                      f"consecutive_drops={consecutive_drops}/{_patience}  → restored best to model.pth")

            print(f"[DAgger] Itération {iteration} done in {time.time()-_t_iter:.1f}s")
            print(f"  Segment metrics : collision={m['collision_rate']:.1%}"
                  f"  win_goal={m['window_goal_dist']:.2f}m"
                  f"  ann_new={m['n_annotations']}  agg_total={len(agg_data)}")
            print(f"  Full-traj eval ({n_ok}/{n_tot} objects, {n_eval_per_iter} runs/obj):", end="")
            for obj_name, r in iter_eval.items():
                s = "✓" if r["success"] else "✗"
                print(f"  {s} {obj_name.split()[-1]}"
                      f"({r['goal_dist']:.1f}±{r['goal_dist_std']:.1f}m"
                      f" sr={r['success_rate']:.0%} cr={r['collision_rate']:.0%})", end="")
            print()

            if use_wandb:
                _wandb_log_iteration(pilot_name, m, iteration)
                try:
                    import wandb
                    if wandb.run is not None:
                        # Per-iteration aggregate metrics
                        wandb.log({
                            f"dagger/{pilot_name}/full_traj_success":
                                m["full_traj_success"],
                            f"dagger/{pilot_name}/full_traj_goal_dist_mean":
                                float(np.mean([v["goal_dist"] for v in iter_eval.values()])),
                            f"dagger/{pilot_name}/best_success_rate": best_success_rate,
                            f"dagger/{pilot_name}/best_goal_dist": best_goal_dist,
                            **{f"dagger/{pilot_name}/per_object/{k.replace(' ','_')}/goal_dist":
                               v["goal_dist"] for k, v in iter_eval.items()},
                            **{f"dagger/{pilot_name}/per_object/{k.replace(' ','_')}/success_rate":
                               v["success_rate"] for k, v in iter_eval.items()},
                            **{f"dagger/{pilot_name}/per_object/{k.replace(' ','_')}/collision_rate":
                               v["collision_rate"] for k, v in iter_eval.items()},
                        })

                        # Per-object summary table for this iteration
                        obj_table = wandb.Table(columns=["object", "success_rate", "collision_rate", "goal_dist", "goal_dist_std"])
                        for k, v in iter_eval.items():
                            short_name = k.split()[-1] if len(k) > 15 else k
                            obj_table.add_data(short_name, f"{v['success_rate']:.0%}",
                                              f"{v['collision_rate']:.0%}",
                                              f"{v['goal_dist']:.2f}m", f"{v['goal_dist_std']:.2f}m")
                        wandb.log({f"dagger/{pilot_name}/iter_{iteration}_objects": obj_table})
                except Exception:
                    pass

            global_step += 1
            beta *= beta_decay

            # Early stopping: if N consecutive iterations failed to improve,
            # the model has likely converged — stop to avoid wasting compute.
            if consecutive_drops >= _patience:
                print(f"\n[DAgger] ⛔ Early stopping at iter {iteration}: "
                      f"{_patience} consecutive iterations without improvement "
                      f"(best sr={best_success_rate:.0%} gd={best_goal_dist:.2f}m)")
                break

        # Ensure best model is at model.pth and loaded in pilot
        model_after_path = os.path.join(bench_dir, "model_after_dagger.pth")
        if os.path.isfile(best_model_staging) and best_success_rate >= 0.0:
            shutil.copy2(best_model_staging, roster_model_path)
            pilot = _swap_model(pilot, roster_model_path)
            pilot.set_mode("deploy")
            print(f"[DAgger] Best model (sr={best_success_rate:.0%} gd={best_goal_dist:.2f}m) → model.pth")
        else:
            print(f"[DAgger] No best model checkpoint found — using final iter model")
        _save_model_checkpoint(pilot, model_after_path)  # archival copy
        # Clean up staging file
        if os.path.isfile(best_model_staging):
            os.remove(best_model_staging)

        # 0 gsplat reload — utilise _SCENE_CACHE
        metrics_after = _run_benchmark_pilot(
            pilot=pilot, model_path=model_after_path, label="after_dagger",
            workspace_path=workspace_path, cohort_name=cohort_name,
            cohort_path=cohort_path, method_name=method_name, flights=flights,
            scenes_cfg_dir=scenes_cfg_dir, objective_configs=objective_configs,
            collision_detectors=collision_detectors, scene_names=scene_names,
            sim_base=sim_base, rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed, max_trajectories=max_trajectories,
        )

        _print_benchmark_comparison(metrics_before, metrics_after, pilot_name)
        if use_wandb:
            _wandb_log_benchmark(pilot_name, metrics_before, metrics_after)

        bench_json = os.path.join(bench_dir, "benchmark_results.json")
        def _safe_json(m):
            out = {}
            for k, v in m.items():
                if isinstance(v, np.ndarray):
                    out[k] = [float(x) if np.isfinite(x) else None for x in v.tolist()]
                else:
                    out[k] = v
            return out
        with open(bench_json, "w") as f:
            json.dump({
                "before": _safe_json(metrics_before),
                "after":  _safe_json(metrics_after),
            }, f, indent=2)
        print(f"[DAgger] Benchmark → {bench_json}")

    # Nettoyage final
    if vision_processor is not None:
        del vision_processor
    _clear_caches()

    return all_metrics
