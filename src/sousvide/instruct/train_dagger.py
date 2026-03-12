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

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.tsampling.build_rrt_dataset as bd

from sousvide.instruct.expert_controllers import PotentialFieldExpert, OnlineRRTExpert

from sousvide.control.pilot import Pilot
import sousvide.instruct.train_policy as tp
from sousvide.flight.deploy_ssv import simulate_rollouts
from sousvide.flight.vision_processor_base import create_vision_processor
from sousvide.rl import load_simulation_results, prepare_batch_data

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

    for scene_name, _ in flights:
        scene_data  = _get_scene(scene_name, scenes_cfg_dir)
        simulator   = scene_data["simulator"]
        obj_targets = scene_data["obj_targets"]
        queries     = scene_data["queries"]

        for obj_idx, obj_name in enumerate(queries):
            pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
            if pkl_data is None:
                continue

            tXUi       = pkl_data["tXUi"]
            obj_target = (
                obj_targets[obj_idx] if obj_idx < len(obj_targets)
                else pkl_data.get("obj_loc", np.zeros(3))
            )

            t0 = float(tXUi[0,  0])
            tf = float(tXUi[0, -1])
            T  = tf - t0

            # Sample max_trajectories start positions from the SECOND half of
            # tXUi — unseen during BC training (per-iter eval uses first half).
            # before/after both sample the same indices → fully comparable.
            n_cols     = tXUi.shape[1]
            half       = max(1, n_cols // 2)
            start_idxs = np.linspace(half, n_cols - 1, max_trajectories, dtype=int)

            print(f"  [{label}] '{obj_name}'  {max_trajectories} runs  "
                  f"t_dur={T:.1f}s  seed={benchmark_seed}")
            for run_i, s_idx in enumerate(start_idxs):
                x0      = tXUi[1:11, s_idx].copy()
                t_start = float(tXUi[0, s_idx])
                t_end   = t_start + T

                _t_sim = time.time()
                result  = simulator.simulate(
                    policy=pilot, t0=t_start, tf=t_end, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=None, verbose=False,
                )
                Tro, Xro = result[0], result[1]
                Uro = result[2] if len(result) > 2 else None

                goal_dist = float(np.linalg.norm(Xro[:3, -1] - obj_target))
                pc_bench = scene_data.get("epcds_arr", np.zeros((0,3)))
                collided  = False
                if pc_bench.shape[0] > 0:
                    Xro_t  = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
                    pc_t   = torch.from_numpy(pc_bench).float().to(DEVICE)
                    collided = bool((torch.cdist(Xro_t, pc_t) < 0.15).any().item())
                    del Xro_t, pc_t
                success   = goal_dist < 2.0 and not collided
                status    = "✓" if success else ("💥" if collided else "✗")
                print(f"  [{label}] {status}  '{obj_name[:20]}'  run {run_i+1}/{max_trajectories}"
                      f"  goal_dist={goal_dist:.2f}m  coll={collided}  ({time.time()-_t_sim:.1f}s)")
                analysis = {
                    "collision":               collided,
                    "success":                 success,
                    "clearance_series":        None,
                    "goal_in_camera_fov_series": None,
                    "total_reward":            -goal_dist,
                    "min_clearance":           None,
                }
                all_analyses.append(analysis)
                all_Tro.append(Tro)
                all_Xro.append(Xro)
                if Uro is not None:
                    all_Uro.append(Uro)

            sr = sum(1 for a in all_analyses[-max_trajectories:] if a["success"]) / max_trajectories
            gd = float(np.mean([-a["total_reward"] for a in all_analyses[-max_trajectories:]]))
            print(f"  [{label}] ── '{obj_name[:25]}'  success={sr:.0%}  mean_goal_dist={gd:.2f}m")

        torch.cuda.empty_cache()

    print(f"  [{label}] {len(all_analyses)} total trajectories evaluated")
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
            half   = max(1, n_cols // 2)
            shared_starts[key] = np.linspace(
                half, n_cols - 1, max_trajectories, dtype=int
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
                    x0      = tXUi[1:11, s_idx].copy()
                    t_start = float(tXUi[0, s_idx])
                    t_end   = t_start + T

                    _t = time.time()
                    result    = simulator.simulate(
                        policy=pilot, t0=t_start, tf=t_end, x0=x0,
                        obj=np.zeros((18, 1)), query=obj_name,
                        vision_processor=None, verbose=False,
                    )
                    Xro       = result[1]
                    goal_dist = float(np.linalg.norm(Xro[:3, -1] - obj_target))

                    pc_ev = scene_data.get("epcds_arr", np.zeros((0, 3)))
                    collided = False
                    if pc_ev.shape[0] > 0:
                        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
                        pc_t  = torch.from_numpy(pc_ev).float().to(DEVICE)
                        collided = bool((torch.cdist(Xro_t, pc_t) < 0.15).any().item())
                        del Xro_t, pc_t

                    success = goal_dist < 2.0 and not collided
                    goal_dists.append(goal_dist)
                    successes.append(success)
                    collisions.append(collided)
                    status = "✓" if success else ("💥" if collided else "✗")
                    print(f"  [{label}] {status}  '{obj_name[:20]}'  "
                          f"run {run_i+1}/{max_trajectories}"
                          f"  goal_dist={goal_dist:.2f}m  ({time.time()-_t:.1f}s)")

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
) -> dict:
    """
    Run n_eval full-trajectory simulations per object, starting from uniformly
    sampled positions along tXUi.  Returns per-object success_rate + goal_dist
    stats for honest per-iteration progress tracking.
    """
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
            tXUi       = pkl_data["tXUi"]
            obj_target = (
                obj_targets[obj_idx] if obj_idx < len(obj_targets)
                else pkl_data.get("obj_loc", np.zeros(3))
            )

            t0 = float(tXUi[0,  0])
            tf = float(tXUi[0, -1])
            T  = tf - t0          # full trajectory duration

            # Sample n_eval start indices uniformly along the FULL tXUi range
            # (both halves — this eval tracks in-distribution progress during training)
            n_cols     = tXUi.shape[1]
            start_idxs = np.linspace(0, n_cols - 1, n_eval, dtype=int)

            goal_dists, successes = [], []
            for run_i, s_idx in enumerate(start_idxs):
                x0      = tXUi[1:11, s_idx].copy()
                t_start = float(tXUi[0, s_idx])
                t_end   = t_start + T

                _t = time.time()
                result    = simulator.simulate(
                    policy=pilot, t0=t_start, tf=t_end, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=vision_processor, verbose=False,
                )
                Xro       = result[1]
                goal_dist = float(np.linalg.norm(Xro[:3, -1] - obj_target))
                pc_ev = scene_data.get("epcds_arr", np.zeros((0,3)))
                collided_ev = False
                if pc_ev.shape[0] > 0:
                    Xro_t  = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
                    pc_t   = torch.from_numpy(pc_ev).float().to(DEVICE)
                    collided_ev = bool((torch.cdist(Xro_t, pc_t) < 0.15).any().item())
                    del Xro_t, pc_t
                success   = goal_dist < 2.0 and not collided_ev
                goal_dists.append(goal_dist)
                successes.append(success)
                status = "✓" if success else ("💥" if collided_ev else "✗")
                print(f"  [{label}] {status}  '{obj_name[:20]}'  "
                      f"run {run_i+1}/{n_eval}  goal_dist={goal_dist:.2f}m  "
                      f"coll={collided_ev}  ({time.time()-_t:.1f}s)")

            sr        = sum(successes) / len(successes)
            mean_dist = float(np.mean(goal_dists))
            std_dist  = float(np.std(goal_dists))
            print(f"  [{label}] ── '{obj_name[:25]}'  "
                  f"success={sr:.0%}  "
                  f"goal_dist={mean_dist:.2f}±{std_dist:.2f}m  "
                  f"best={float(np.min(goal_dists)):.2f}m")
            results[obj_name] = {
                "goal_dist":     mean_dist,
                "goal_dist_std": std_dist,
                "goal_dist_min": float(np.min(goal_dists)),
                "success":       sr >= 0.5,
                "success_rate":  sr,
                "n_eval":        n_eval,
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


def _filter_deviation_annotations(
    annotations: List[dict],
    Xro: np.ndarray,
    tXUi: np.ndarray,
    obj_target: np.ndarray,
    idx0: int,
    deviation_threshold: float = 0.3,
    close_approach_dist: float = 5.0,
) -> List[dict]:
    """
    Filter full-trajectory DAgger annotations to keep only useful ones:
      - States where the pilot deviated from the reference tXUi (> deviation_threshold).
      - States where the drone was within close_approach_dist of the goal.

    This discards the majority of "fly straight from 25m away" timesteps that
    would otherwise corrupt BC fine-grained approach behaviour.
    """
    T = len(annotations)
    if T == 0:
        return annotations

    keep: set = set()
    for i in range(T):
        if i >= Xro.shape[1]:
            break
        pos = Xro[:3, i]

        # Always keep near-goal states (final approach / arrival)
        if np.linalg.norm(pos - obj_target) < close_approach_dist:
            keep.add(i)
            continue

        # Keep if drone deviated from reference trajectory at this timestep
        ref_idx = min(idx0 + i, tXUi.shape[1] - 1)
        ref_pos = tXUi[1:4, ref_idx]
        if np.linalg.norm(pos - ref_pos) > deviation_threshold:
            keep.add(i)

    return [annotations[i] for i in sorted(keep)]


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
            f"dagger/{pilot_name}/beta":             m["beta"],
            f"dagger/{pilot_name}/collision_rate":   m["collision_rate"],
            f"dagger/{pilot_name}/success_rate":     m["success_rate"],
            f"dagger/{pilot_name}/window_goal_dist": m.get("window_goal_dist", 0.0),
            f"dagger/{pilot_name}/n_annotations":    m.get("n_annotations", 0),
            f"dagger/{pilot_name}/total_rollouts":   m["total_rollouts"],
            "dagger/iteration":                       m["iteration"],
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
):
    """
    Return the expert controller for this DAgger iteration segment.
      "mpc"       – original VehicleRateMPC (recovery-to-reference)
      "potential" – PotentialFieldExpert   (goal-seeking + obstacle avoidance)
      "rrt"       – OnlineRRTExpert        (RRT* replanning + pure-pursuit)
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
    deviation_filter_dist: float = 0.3,
    close_approach_dist: float   = 5.0,
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
    print(f"[DAgger] Mode            : Full-trajectory + deviation filter (Option B)")
    print(f"[DAgger] Per-iter eval   : {n_eval_per_iter} runs/object  |  Benchmark: {max_trajectories} runs/object")
    print(f"[DAgger] Aggregation     : {'cumulative' if aggregate_dagger else 'online (per-iter only)'}")
    print(f"[DAgger] Start-pos noise : ±{start_pos_noise}m")
    print(f"[DAgger] Ann filter      : keep if drift>{deviation_filter_dist}m OR goal_dist<{close_approach_dist}m")

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
        print(f"  expert_type  : {expert_type}")

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
            _t_iter = time.time()
            print(f"\n[DAgger] ── Itération {iteration}/{n_iterations-1}  β={beta:.3f}  ({time.strftime('%H:%M:%S')})")

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

                    # Build expert once per object.
                    # For "mpc": ACADOS setup is expensive → reused across all 2s windows.
                    # For "potential"/"rrt": lightweight, also reused across windows.
                    expert       = _make_expert(
                        expert_type, tXUi, obj_target, obj_idx,
                        scene_data, objective_configs[scene_name],
                        _base_policy_name, _base_frame_name, pilot_name,
                    )
                    mixed_policy = MixedPolicy(expert, pilot, beta)

                    # Option B: run the FULL trajectory (not 2s windows) so the
                    # mixed policy encounters actual navigation states, then keep
                    # only annotations at deviation / near-goal timesteps.
                    t_traj_start = float(tXUi[0, 0])
                    t_traj_end   = float(tXUi[0, -1])

                    # Perturb initial position so each iteration starts slightly
                    # differently → diverse states not seen during BC training.
                    ref_idx0 = min(
                        int(np.searchsorted(tXUi[0, :], t_traj_start)),
                        tXUi.shape[1] - 1,
                    )
                    x0_ref = tXUi[1:, ref_idx0].copy()
                    if start_pos_noise > 0.0:
                        x0_ref[:3] += np.random.uniform(
                            -start_pos_noise, start_pos_noise, size=3
                        )
                    perturbation = {"t0": t_traj_start, "x0": x0_ref}

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

                    # Filter: keep only deviation + near-goal annotations.
                    filtered_ann = _filter_deviation_annotations(
                        annotations=rollout["annotations"],
                        Xro=rollout["Xro"],
                        tXUi=tXUi,
                        obj_target=obj_target,
                        idx0=ref_idx0,
                        deviation_threshold=deviation_filter_dist,
                        close_approach_dist=close_approach_dist,
                    )

                    all_rollouts.append(rollout)
                    all_annotations.extend(filtered_ann)
                    _save_traj_plot(
                        rollout["Tro"], rollout["Xro"], rollout["Uro"],
                        save_path=os.path.join(
                            dagger_dir, "plots",
                            f"iter{iteration:03d}_{obj_name.replace(' ','_')}_fulltraj.png",
                        ),
                        title=f"iter={iteration} β={beta:.2f} | {obj_name} {t_traj_start:.1f}→{t_traj_end:.1f}s",
                    )

                    used  = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    goal_dist = -rollout["analysis"].get("total_reward", 0.0)
                    print(
                        f"  [{obj_name[:20]}"
                        f" t=[{t_traj_start:.1f},{t_traj_end:.1f}]s]"
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

            # Re-entraînement
            print(f"[DAgger] Retraining Commander  Nep={Nep_per_iter} lim_sv={lim_sv}...")
            _t_retrain = time.time()
            _retrain_commander(cohort_name, pilot_name, aggregated_file, Nep_per_iter, lim_sv)
            print(f"[DAgger] Retraining done in {time.time()-_t_retrain:.1f}s")

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
            )
            n_ok  = sum(1 for v in iter_eval.values() if v["success"])
            n_tot = len(iter_eval)
            print(f"[DAgger] Full-traj eval done in {time.time()-_t_eval:.1f}s  "
                  f"{n_ok}/{n_tot} objects reached")
            m["full_traj_eval"]    = iter_eval
            m["full_traj_success"] = n_ok / max(n_tot, 1)

            print(f"[DAgger] Itération {iteration} done in {time.time()-_t_iter:.1f}s")
            print(f"  Segment metrics : collision={m['collision_rate']:.1%}"
                  f"  win_goal={m['window_goal_dist']:.2f}m"
                  f"  ann_new={m['n_annotations']}  agg_total={len(agg_data)}")
            print(f"  Full-traj eval ({n_ok}/{n_tot} objects, {10} runs/obj):", end="")
            for obj_name, r in iter_eval.items():
                s = "✓" if r["success"] else "✗"
                print(f"  {s} {obj_name.split()[-1]}"
                      f"({r['goal_dist']:.1f}±{r['goal_dist_std']:.1f}m"
                      f" {r['success_rate']:.0%})", end="")
            print()

            if use_wandb:
                _wandb_log_iteration(pilot_name, m, global_step)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            f"dagger/{pilot_name}/full_traj_success":
                                m["full_traj_success"],
                            f"dagger/{pilot_name}/full_traj_goal_dist_mean":
                                float(np.mean([v["goal_dist"] for v in iter_eval.values()])),
                            **{f"dagger/{pilot_name}/full_traj/{k.replace(' ','_')}_dist":
                               v["goal_dist"] for k, v in iter_eval.items()},
                            **{f"dagger/{pilot_name}/full_traj/{k.replace(' ','_')}_sr":
                               v["success_rate"] for k, v in iter_eval.items()},
                        }, step=global_step)
                except Exception:
                    pass

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
