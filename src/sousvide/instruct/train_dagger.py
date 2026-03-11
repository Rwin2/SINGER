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
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.tsampling.build_rrt_dataset as bd

from sousvide.control.pilot import Pilot
import sousvide.instruct.train_policy as tp
from sousvide.flight.deploy_ssv import simulate_rollouts
from sousvide.flight.vision_processor_base import create_vision_processor
from sousvide.rl import load_simulation_results, prepare_batch_data
from sousvide.visualize.analyze_simulated_experiments import (
    analyze_trajectory_performance,
)

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
    _SCENE_CACHE[scene_name] = dict(
        simulator=simulator,
        obj_targets=obj_targets,
        epcds_arr=epcds_arr,
        queries=objectives_list,
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

        # Pilot neural — use OODA directly to capture xnn for Commander retraining
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(upr, tcr, xcr, obj, icr, zcr)

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
) -> dict:
    pilot = _swap_model(pilot, model_path)

    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

    all_Tro, all_Xro, all_Uro = [], [], []
    all_analyses = []

    for scene_name, _ in flights:
        scene_data  = _get_scene(scene_name, scenes_cfg_dir)   # depuis cache
        simulator   = scene_data["simulator"]
        obj_targets = scene_data["obj_targets"]
        queries     = scene_data["queries"]

        for obj_idx, obj_name in enumerate(queries):
            pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if pkl_data is None:
                continue

            tXUi = pkl_data["tXUi"]
            obj_target = (
                obj_targets[obj_idx]
                if obj_idx < len(obj_targets)
                else pkl_data.get("obj_loc", np.zeros(3))
            )

            t0 = float(tXUi[0, 0])
            tf = float(tXUi[0, -1])
            x0 = tXUi[1:11, 0].copy()  # state is rows 1-10 (nx=10), not all 17 rows

            print(f"  [{label}] simulate '{obj_name}'  tXUi={tXUi.shape}")

            # FIX: signature correcte de Simulator.simulate()
            result = simulator.simulate(
                policy=pilot,
                t0=t0,
                tf=tf,
                x0=x0,
                obj=np.zeros((18, 1)),
                query=obj_name,
                vision_processor=None,
                verbose=False,
            )
            Tro, Xro = result[0], result[1]
            Uro = result[2] if len(result) > 2 else None

            pc = scene_data["epcds_arr"]
            if isinstance(pc, list):
                pc = np.concatenate(pc, axis=0) if pc else np.zeros((0, 3))

            analysis = analyze_trajectory_performance(
                Xro=Xro, goal_location=obj_target, point_cloud=pc,
                exclusion_radius=2.0, collision_radius=0.15, verbose=False,
            )
            all_Tro.append(Tro)
            all_Xro.append(Xro)
            if Uro is not None:
                all_Uro.append(Uro)
            all_analyses.append(analysis)

        torch.cuda.empty_cache()

    print(f"  [{label}] {len(all_analyses)} trajectories simulated")
    metrics = _extract_metrics_from_analyses(all_analyses, scene_data["epcds_arr"])
    metrics["label"] = label

    del all_Tro, all_Xro, all_Uro
    gc.collect()
    return metrics


def _extract_metrics_from_analyses(analyses: list, point_cloud) -> dict:
    """Calcule les métriques directement depuis les analyses (sans load_simulation_results)."""
    collision_rates, clearances_mean, fov_rates, returns_ = [], [], [], []

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

    return {
        "collision_rate": np.array(collision_rates),
        "clearance_mean": np.array(clearances_mean),
        "fov_rate":       np.array(fov_rates),
        "return_sum":     np.array(returns_),
        "traj_length":    np.array([1.0] * len(analyses)),
    }


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

    # Extract x0 from tXUi at the time step nearest to t_start
    idx0 = int(np.searchsorted(tXUi[0, :], t_start))
    idx0 = min(idx0, tXUi.shape[1] - 1)
    x0 = tXUi[1:11, idx0].copy()  # state is rows 1-10 (nx=10)

    t0 = t_start
    tf = t_end

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

    pc = (np.concatenate(point_cloud, axis=0)
          if isinstance(point_cloud, list) and len(point_cloud) > 0
          else point_cloud if isinstance(point_cloud, np.ndarray)
          else np.zeros((0, 3)))

    collision_steps, drift_steps = [], []

    if pc.shape[0] > 0:
        pc_t  = torch.from_numpy(pc).float().to(DEVICE)
        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
        T     = min(Xro_t.shape[0], tXUi.shape[1])

        dists           = torch.cdist(Xro_t, pc_t)
        collision_steps = (dists < collision_threshold).any(dim=1)\
                          .nonzero(as_tuple=True)[0].cpu().tolist()

        ref_t       = torch.from_numpy(tXUi[1:4, :T].T).float().to(DEVICE)
        drift_steps = (
            torch.norm(Xro_t[:T] - ref_t[:T], dim=1) > drift_threshold
        ).nonzero(as_tuple=True)[0].cpu().tolist()

        del pc_t, Xro_t, dists, ref_t
        torch.cuda.empty_cache()
    else:
        for i, x in enumerate(Xro.T):
            ref = tXUi[1:4, min(i, tXUi.shape[1] - 1)]
            if np.linalg.norm(x[:3] - ref) > drift_threshold:
                drift_steps.append(i)

    analysis = analyze_trajectory_performance(
        Xro=Xro, goal_location=obj_target, point_cloud=pc,
        exclusion_radius=2.0, collision_radius=collision_threshold, verbose=False,
    )

    return {
        "Tro":             Tro,
        "Xro":             Xro,
        "Uro":             Uro,
        "annotations":     mixed_policy.annotations.copy(),
        "collision_steps": collision_steps,
        "drift_steps":     drift_steps,
        "analysis":        analysis,
        "obj_name":        obj_name,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Métriques / Agrégation / Re-entraînement / W&B
# ──────────────────────────────────────────────────────────────────────────────

def _compute_dagger_metrics(rollouts: List[dict], iteration: int, beta: float) -> dict:
    total = len(rollouts)
    return {
        "iteration":      iteration,
        "beta":           beta,
        "total_rollouts": total,
        "collision_rate": sum(1 for r in rollouts if r["collision_steps"]) / max(total, 1),
        "drift_rate":     sum(1 for r in rollouts if r["drift_steps"])     / max(total, 1),
        "success_rate":   sum(1 for r in rollouts if r["analysis"].get("success", False)) / max(total, 1),
    }


def _aggregate_dagger_dataset(
    all_annotations: List[dict], existing_file: Optional[str],
) -> List[dict]:
    if existing_file and os.path.exists(existing_file):
        return torch.load(existing_file) + all_annotations
    return all_annotations


def _retrain_commander(
    cohort_name: str, pilot_name: str,
    aggregated_file: str, Nep: int, lim_sv: int,
    default_mass: float = 0.3, default_fn: float = 0.3,
) -> None:
    """
    Convert DAgger annotations to BC observation format and retrain the Commander.

    The DAgger aggregated file is a flat list of step-level dicts:
        {"xnn": {...}, "x": ndarray, "u": ndarray, "t": float, "query": ndarray}

    The BC observation format expected by generate_dataset / extract_data is:
        {"data": [{"Xnn": [...], "Ynn": [...], "Ndata": int, ...}], "set": "", ...}

    We save the converted file into a dedicated "dagger" course subdirectory so
    that get_data_paths() finds it alongside the existing BC observations.
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
        Xnn.append(xnn)
        Ynn.append({
            "unn": np.array(ann["u"], dtype=np.float32),
            "mfn": default_mfn.copy(),
            "onn": np.array(ann["x"], dtype=np.float32),
        })

    if not Xnn:
        print("  [retrain] No valid xnn entries in annotations — skipping.")
        return

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
    print(f"  [retrain] {len(Xnn)} samples → {dst}")

    tp.train_roster(cohort_name, [pilot_name], "Commander", Nep, lim_sv=lim_sv)


def _wandb_log_iteration(pilot_name: str, m: dict, step: int) -> None:
    try:
        import wandb
        if wandb.run is not None: wandb.log({
            f"dagger/{pilot_name}/beta":           m["beta"],
            f"dagger/{pilot_name}/collision_rate": m["collision_rate"],
            f"dagger/{pilot_name}/drift_rate":     m["drift_rate"],
            f"dagger/{pilot_name}/success_rate":   m["success_rate"],
            f"dagger/{pilot_name}/total_rollouts": m["total_rollouts"],
            "dagger/iteration":                     m["iteration"],
        }, step=step)
    except Exception as e:
        print(f"  [WARN] wandb: {e}")


def _wandb_log_benchmark(pilot_name: str, before: dict, after: dict) -> None:
    try:
        import wandb
        fin = lambda x: x[np.isfinite(x)]
        if wandb.run is not None: wandb.log({
            f"benchmark/{pilot_name}/before/collision_rate": before["collision_rate"].mean(),
            f"benchmark/{pilot_name}/before/clearance_mean": fin(before["clearance_mean"]).mean(),
            f"benchmark/{pilot_name}/before/fov_rate":       np.nanmean(before["fov_rate"]),
            f"benchmark/{pilot_name}/before/return_mean":    fin(before["return_sum"]).mean(),
            f"benchmark/{pilot_name}/after/collision_rate":  after["collision_rate"].mean(),
            f"benchmark/{pilot_name}/after/clearance_mean":  fin(after["clearance_mean"]).mean(),
            f"benchmark/{pilot_name}/after/fov_rate":        np.nanmean(after["fov_rate"]),
            f"benchmark/{pilot_name}/after/return_mean":     fin(after["return_sum"]).mean(),
            f"benchmark/{pilot_name}/delta/collision_rate":
                after["collision_rate"].mean() - before["collision_rate"].mean(),
        })
    except Exception as e:
        print(f"  [WARN] wandb benchmark: {e}")


def _print_benchmark_comparison(before: dict, after: dict, pilot_name: str) -> None:
    fin = lambda x: x[np.isfinite(x)]
    def _s(m):
        cr = m["collision_rate"]; clr = fin(m["clearance_mean"]); fov = fin(m["fov_rate"])
        print(f"\n  ── {m['label']} ({len(cr)} rollouts) ──")
        print(f"    collision_rate : {cr.mean()*100:.1f}%  ({int(cr.sum())}/{len(cr)})")
        print(f"    clearance_mean : {clr.mean():.3f} m")
        print(f"    fov_rate       : {np.nanmean(fov)*100:.1f}%")
        print(f"    return mean    : {fin(m['return_sum']).mean():.1f}")
    print("\n" + "=" * 62)
    print(f"  BENCHMARK DAgger — {pilot_name}")
    print("=" * 62)
    _s(before); _s(after)
    for name, delta, better_if in [
        ("collision_rate", (after["collision_rate"].mean()  - before["collision_rate"].mean())*100,  "<"),
        ("clearance_mean", fin(after["clearance_mean"]).mean() - fin(before["clearance_mean"]).mean(), ">"),
        ("fov_rate",       (np.nanmean(after["fov_rate"]) - np.nanmean(before["fov_rate"]))*100,     ">"),
        ("return_mean",    fin(after["return_sum"]).mean() - fin(before["return_sum"]).mean(),        ">"),
    ]:
        ok = (delta < 0) if better_if == "<" else (delta > 0)
        unit = "pp" if "rate" in name or "fov" in name else ""
        print(f"  Δ {name:<15}: {delta:+.1f}{unit}  {'✓ better' if ok else '✗ worse'}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale DAgger
# ──────────────────────────────────────────────────────────────────────────────

def train_dagger_policy(
    cohort_name: str,
    method_name: str,
    roster: List[str],
    flights: List[Tuple[str, str]],
    n_iterations: int          = 5,
    beta_start: float          = 1.0,
    beta_decay: float          = 0.5,
    collision_threshold: float = 0.15,
    drift_threshold: float     = 2.0,
    Nep_per_iter: int          = 50,
    lim_sv: int                = 10,
    max_trajectories: int      = 10,
    benchmark_seed: int        = 42,
    use_wandb: bool            = False,
    wandb_project: str         = "singer-dagger",
    wandb_run_name: str        = "dagger",
) -> dict:

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
    print(f"[DAgger] Trajectory window  : {_Tdt_ro}s per segment (matches BC training)")

    vision_processor = create_vision_processor(_vp_type)
    if vision_processor is not None and hasattr(vision_processor, "to"):
        vision_processor = vision_processor.to(DEVICE)
        print(f"[DAgger] vision_processor → {DEVICE}")

    # ── PERF : précharger gsplat + pkl UNE SEULE FOIS ────────────────────────
    print("\n[DAgger] ⏳ Préchargement scènes + pkl (1 seule fois pour tout le run)...")
    for scene_name in scene_names:
        _get_scene(scene_name, scenes_cfg_dir, _base_frame_name, _base_rollout_name)
    n_pkls = _preload_all_pkls(flights, scenes_cfg_dir)
    print(f"[DAgger] ✅ {len(_SCENE_CACHE)} scène(s), {n_pkls} pkl en cache\n")

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

        pilot = Pilot(cohort_name, pilot_name)
        pilot.set_mode("deploy")
        pilot.model.to(DEVICE)

        dagger_dir = os.path.join(cohort_path, "dagger_data", pilot_name)
        bench_dir  = os.path.join(dagger_dir, "benchmark")
        rrt_backup = os.path.join(dagger_dir, "_benchmark_rrt_backup")
        os.makedirs(bench_dir,  exist_ok=True)
        os.makedirs(rrt_backup, exist_ok=True)

        model_before_path = os.path.join(bench_dir, "model_before_dagger.pth")
        _save_model_checkpoint(pilot, model_before_path)

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

        aggregated_file = os.path.join(dagger_dir, "dagger_aggregated.pt")
        beta, global_step = beta_start, 0

        # ── Boucle DAgger ─────────────────────────────────────────────────
        for iteration in range(n_iterations):
            print(f"\n[DAgger] Itération {iteration}/{n_iterations-1}  β={beta:.3f}")

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

                    tXUi = pkl_data["tXUi"]
                    obj_target = (
                        obj_targets[obj_idx]
                        if obj_idx < len(obj_targets)
                        else pkl_data.get("obj_loc", np.zeros(3))
                    )

                    # Build expert MPC once per object (ACADOS setup is expensive).
                    # The same MPC is reused across all 2s windows of this trajectory.
                    expert_mpc   = VehicleRateMPC(tXUi, _base_policy_name, _base_frame_name, pilot_name)
                    mixed_policy = MixedPolicy(expert_mpc, pilot, beta)

                    # Split trajectory into 2s windows — mirrors BC training (compute_batches).
                    t_traj_start = float(tXUi[0, 0])
                    t_traj_end   = float(tXUi[0, -1])
                    n_windows = max(1, int((t_traj_end - t_traj_start) // _Tdt_ro))

                    for seg_idx in range(n_windows):
                        t_seg_start = t_traj_start + seg_idx * _Tdt_ro
                        t_seg_end   = min(t_seg_start + _Tdt_ro, t_traj_end)
                        perturbation = {"t0": t_seg_start, "x0": tXUi[1:, int(np.searchsorted(tXUi[0, :], t_seg_start))].copy()}

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
                            t_start=t_seg_start,
                            t_end=t_seg_end,
                        )
                        all_rollouts.append(rollout)
                        all_annotations.extend(rollout["annotations"])

                        used  = torch.cuda.memory_allocated() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(
                            f"  [{obj_name} seg {seg_idx}/{n_windows-1}"
                            f" t=[{t_seg_start:.1f},{t_seg_end:.1f}]s]"
                            f"  coll={len(rollout['collision_steps'])}"
                            f"  drift={len(rollout['drift_steps'])}"
                            f"  ok={rollout['analysis'].get('success', False)}"
                            f"  GPU={used:.1f}/{total:.0f}GB"
                        )

            # Agrégation
            agg_data = _aggregate_dagger_dataset(all_annotations, aggregated_file)
            torch.save(agg_data, aggregated_file)
            torch.save(all_annotations,
                       os.path.join(dagger_dir, f"dagger_iter_{iteration:03d}.pt"))
            print(f"  [agg] {len(agg_data)} samples → {aggregated_file}")

            # Re-entraînement
            _retrain_commander(cohort_name, pilot_name, aggregated_file, Nep_per_iter, lim_sv)

            # Recharger pilot avec nouveaux poids
            pilot = Pilot(cohort_name, pilot_name)
            pilot.set_mode("deploy")
            pilot.model.to(DEVICE)

            m = _compute_dagger_metrics(all_rollouts, iteration, beta)
            all_metrics[pilot_name].append(m)
            print(f"  collision={m['collision_rate']:.1%}"
                  f"  drift={m['drift_rate']:.1%}"
                  f"  success={m['success_rate']:.1%}")

            if use_wandb:
                _wandb_log_iteration(pilot_name, m, global_step)

            global_step += 1
            beta *= beta_decay

        # Checkpoint final + benchmark after
        model_after_path = os.path.join(bench_dir, "model_after_dagger.pth")
        _save_model_checkpoint(pilot, model_after_path)

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
        with open(bench_json, "w") as f:
            json.dump({
                "before": {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in metrics_before.items()},
                "after":  {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in metrics_after.items()},
            }, f, indent=2)
        print(f"[DAgger] Benchmark → {bench_json}")

    # Nettoyage final
    if vision_processor is not None:
        del vision_processor
    _clear_caches()

    return all_metrics
