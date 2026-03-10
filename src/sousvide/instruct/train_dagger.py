"""
DAgger (Dataset Aggregation) pour réduire les collisions et dérives du Pilot.

Référence:
    Ross et al., "A Reduction of Imitation Learning and Structured Prediction
    to No-Regret Online Learning", AISTATS 2011.

Principe:
    Itération k :
      1. Politique mixte  π_k = β_k * expert + (1-β_k) * pilot
      2. Collecte des états réellement visités par π_k
      3. Expert annote chaque état  →  (obs, u_expert)
      4. Agrégation  D ← D ∪ D_k
      5. Re-entraînement du Commander sur D via tp.train_roster
      6. β_{k+1} = β_k * decay   (pilot prend progressivement la main)
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
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.utilities.trajectory_helper as th
import figs.tsampling.build_rrt_dataset as bd
import figs.scene_editing.scene_editing_utils as scdt

from sousvide.control.pilot import Pilot
import sousvide.synthesize.rollout_generator as gd
import sousvide.instruct.train_policy as tp
from sousvide.flight.deploy_ssv import simulate_rollouts
from sousvide.flight.vision_processor_base import create_vision_processor
from sousvide.rl import load_simulation_results, prepare_batch_data
from sousvide.rl.collision_detector import CollisionDetector
from sousvide.visualize.analyze_simulated_experiments import (
    analyze_trajectory_performance,
)


# ──────────────────────────────────────────────────────────────────────────────
# Politique mixte β * expert + (1-β) * pilot
# ──────────────────────────────────────────────────────────────────────────────

class MixedPolicy:
    """
    À chaque appel, choisit stochastiquement expert ou pilot selon β.
    Enregistre l'observation RÉELLE du Commander (image + x + query)
    ainsi que la commande expert pour l'annotation DAgger.
    """

    def __init__(self, expert: VehicleRateMPC, pilot: Pilot, beta: float):
        self.expert      = expert
        self.pilot       = pilot
        self.beta        = beta
        self.annotations: List[dict] = []

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        query: str,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        u_expert = self.expert(t, x)
        u_pilot  = self.pilot(t, x, image, query) if image is not None else u_expert

        obs_entry = {
            "x":     x.copy(),
            "u":     u_expert.copy(),   # clé "u" = format attendu par generate_dataset()
            "t":     t,
            "query": query,
        }
        if image is not None:
            obs_entry["image"] = image.copy()

        self.annotations.append(obs_entry)
        return u_expert if np.random.rand() < self.beta else u_pilot

    def reset_annotations(self):
        self.annotations = []


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers — pattern identique au notebook simple_rl.ipynb
# ──────────────────────────────────────────────────────────────────────────────

def _swap_model(pilot: Pilot, model_path: str) -> Pilot:
    """
    Remplace les poids du pilot EN-PLACE (même objet).
    Pattern identique au notebook : pas de copytree, pas de swap disque.
    """
    pilot.model = torch.load(model_path, map_location="cpu")
    pilot.model.eval()
    print(f"  [swap] {Path(model_path).name}")
    return pilot


def _save_model_checkpoint(pilot: Pilot, dst_path: str) -> None:
    """Sauvegarde pilot.model vers un fichier .pth pour pouvoir le recharger."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(pilot.model, dst_path)
    print(f"  [ckpt] Sauvegardé → {dst_path}")


def _latest_sim_dir_by_mtime(sim_data_base: str) -> str:
    """Retourne le sous-répertoire le plus récent par mtime (pas lexicographique)."""
    subdirs = [
        os.path.join(sim_data_base, d)
        for d in os.listdir(sim_data_base)
        if os.path.isdir(os.path.join(sim_data_base, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"[DAgger] Aucun sous-répertoire dans {sim_data_base}")
    return max(subdirs, key=os.path.getmtime)


def _extract_metrics(
    trajectories: list,
    metadata: dict,
    collision_detectors: dict,
    scene_names: List[str],
) -> dict:
    """
    Calcule collision / clearance / fov / return.
    Même logique que le notebook simple_rl.ipynb → résultats comparables.
    """
    collision_rates = []
    clearances_mean = []
    fov_rates       = []
    returns         = []
    traj_lengths    = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bs in range(0, len(trajectories), 32):
        be         = min(bs + 32, len(trajectories))
        meta_slice = {
            k: (v[bs:be] if isinstance(v, list) else v)
            for k, v in metadata.items()
        }

        batch = prepare_batch_data(
            trajectories[bs:be], meta_slice,
            collision_detectors[scene_names[0]],
            device,
        )

        for i, analysis in enumerate(batch["analyses"]):
            collision_rates.append(float(analysis["collision"]))

            clr = np.array(batch["clearances"][i])
            clearances_mean.append(float(clr.mean()) if clr.size else np.nan)
            traj_lengths.append(int(len(clr)))

            fov_series = analysis.get("goal_in_camera_fov_series")
            if fov_series is None:
                fov_series = analysis.get("goal_in_camera_fov")
            if fov_series is not None:
                arr = np.asarray(fov_series).reshape(-1)
                fov_rates.append(float(np.mean(arr > 0.5)) if arr.size else np.nan)
            else:
                fov_rates.append(np.nan)

            if "rewards" in batch and i < len(batch["rewards"]):
                rew = np.array(batch["rewards"][i])
                returns.append(float(rew.sum()) if rew.size else np.nan)
            else:
                returns.append(np.nan)

        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "collision_rate": np.array(collision_rates),
        "clearance_mean": np.array(clearances_mean),
        "traj_length":    np.array(traj_lengths, dtype=float),
        "fov_rate":       np.array(fov_rates),
        "return_sum":     np.array(returns),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark avant / après — pattern review=True → backup → review=False
# ──────────────────────────────────────────────────────────────────────────────

def _run_benchmark_pilot(
    pilot: Pilot,
    model_path: str,
    label: str,
    workspace_path: str,
    cohort_name: str,
    cohort_path: str,
    method_name: str,
    flights: List[Tuple[str, str]],
    scenes_cfg_dir: str,
    objective_configs: dict,
    collision_detectors: dict,
    scene_names: List[str],
    sim_base: str,
    rrt_backup: str,
    benchmark_seed: int,
    max_trajectories: int,
) -> dict:
    """
    Rejoue les MÊMES .pkl RRT avec le model_path donné.
    Pattern identique à _run_pilot() du notebook simple_rl.ipynb.
    """
    pilot = _swap_model(pilot, model_path)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_dir = os.path.join(sim_base, ts)
    new_rrt = os.path.join(new_dir, "rrt_planning")
    os.makedirs(new_rrt, exist_ok=True)

    # Injecter les mêmes .pkl → starts/goals identiques
    for f in glob.glob(os.path.join(rrt_backup, "*.pkl")):
        shutil.copy2(f, new_rrt)
    print(f"\n[{label}] Injected {len(os.listdir(new_rrt))} .pkl → {ts}")

    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

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
        review=False,              # ← lit les .pkl existants, ne re-planifie PAS
        disable_visualization=True,
        show_progress=True,
    )
    plt.close("all")

    # Sanity check timestamp
    actual_latest = max(
        d for d in os.listdir(sim_base)
        if os.path.isdir(os.path.join(sim_base, d))
    )
    assert actual_latest == ts, \
        f"[DAgger-Bench] Conflit timestamp : {actual_latest} != {ts}"

    trajectories, metadata, raw_data = load_simulation_results(
        cohort_path, pilot_name=pilot.name
    )
    print(f"  {len(trajectories)} trajectories loaded")

    metrics = _extract_metrics(trajectories, metadata, collision_detectors, scene_names)
    metrics["label"] = label

    del trajectories, metadata, raw_data
    gc.collect()
    return metrics


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
) -> None:
    """
    Génère les .pkl RRT UNE SEULE FOIS avec review=True (seed fixée)
    et les sauvegarde dans rrt_backup.
    Pattern identique au STEP 3 du notebook simple_rl.ipynb.
    """
    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

    pilot = _swap_model(pilot, model_path)

    print("[RRT-Backup] Generating paths once (review=True, seed fixée)...")
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
        review=True,               # ← planifie les routes RRT
        disable_visualization=True,
        show_progress=True,
    )
    plt.close("all")

    latest_ts  = max(
        d for d in os.listdir(sim_base)
        if os.path.isdir(os.path.join(sim_base, d))
    )
    rrt_source = os.path.join(sim_base, latest_ts, "rrt_planning")

    if os.path.exists(rrt_backup):
        shutil.rmtree(rrt_backup)
    shutil.copytree(rrt_source, rrt_backup)
    print(f"[RRT-Backup] {len(os.listdir(rrt_backup))} .pkl sauvegardés → {rrt_backup}")


def _print_benchmark_comparison(before: dict, after: dict, pilot_name: str) -> None:
    fin = lambda x: x[np.isfinite(x)]

    def _summary(m: dict) -> None:
        cr  = m["collision_rate"]
        clr = fin(m["clearance_mean"])
        fov = fin(m["fov_rate"])
        ret = fin(m["return_sum"])
        print(f"\n  ── {m['label']} ({len(cr)} rollouts) ──")
        print(f"    collision_rate : {cr.mean()*100:.1f}%  ({int(cr.sum())}/{len(cr)})")
        print(f"    clearance_mean : {clr.mean():.3f} m  (min={clr.min():.3f})")
        print(f"    fov_rate       : {np.nanmean(fov)*100:.1f}%")
        print(f"    return mean    : {fin(m['return_sum']).mean():.1f}")

    print("\n" + "=" * 62)
    print(f"  BENCHMARK DAgger — {pilot_name}")
    print("=" * 62)
    _summary(before)
    _summary(after)

    delta_cr  = after["collision_rate"].mean() - before["collision_rate"].mean()
    delta_clr = fin(after["clearance_mean"]).mean() - fin(before["clearance_mean"]).mean()
    delta_fov = np.nanmean(after["fov_rate"])  - np.nanmean(before["fov_rate"])
    delta_ret = fin(after["return_sum"]).mean()  - fin(before["return_sum"]).mean()

    print(f"\n  Δ collision_rate : {delta_cr*100:+.1f}pp  "
          f"{'✓ better' if delta_cr  < 0 else '✗ worse'}")
    print(f"  Δ clearance_mean : {delta_clr:+.3f} m   "
          f"{'✓ better' if delta_clr > 0 else '✗ worse'}")
    print(f"  Δ fov_rate       : {delta_fov*100:+.1f}pp  "
          f"{'✓ better' if delta_fov > 0 else '✗ worse'}")
    print(f"  Δ return         : {delta_ret:+.1f}       "
          f"{'✓ better' if delta_ret > 0 else '✗ worse'}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Agrégation dataset DAgger
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_dagger_dataset(
    cohort_path: str,
    pilot_name: str,
    new_rollouts: List[dict],
    iteration: int,
) -> str:
    dagger_dir = os.path.join(cohort_path, "dagger_data", pilot_name)
    os.makedirs(dagger_dir, exist_ok=True)

    obs_shards = []
    for rollout in new_rollouts:
        for annot in rollout.get("annotations", []):
            shard = {
                "x":     annot["x"],
                "u":     annot["u"],       # clé "u" = format attendu par generate_dataset()
                "t":     annot["t"],
                "query": annot["query"],
            }
            if "image" in annot:
                shard["image"] = annot["image"]
            obs_shards.append(shard)

    iter_file = os.path.join(dagger_dir, f"dagger_iter_{iteration:03d}.pt")
    torch.save(obs_shards, iter_file)
    print(f"[DAgger] Itération {iteration} : {len(obs_shards)} obs → {iter_file}")

    all_shards = []
    for fname in sorted(os.listdir(dagger_dir)):
        if fname.startswith("dagger_iter_") and fname.endswith(".pt"):
            all_shards.extend(torch.load(os.path.join(dagger_dir, fname)))

    aggregated_file = os.path.join(dagger_dir, "dagger_aggregated.pt")
    torch.save(all_shards, aggregated_file)
    print(f"[DAgger] Dataset agrégé : {len(all_shards)} observations au total")
    return aggregated_file


# ──────────────────────────────────────────────────────────────────────────────
# Re-entraînement Commander via tp.train_roster
# ──────────────────────────────────────────────────────────────────────────────

def _retrain_commander(
    cohort_name: str,
    pilot_name: str,
    aggregated_file: str,
    Nep: int,
    lim_sv: int,
) -> None:
    pilot      = Pilot(cohort_name, pilot_name)
    pilot_path = pilot.path

    obs_dir = os.path.join(pilot_path, "observation_data")
    os.makedirs(obs_dir, exist_ok=True)

    dagger_shard_dst = os.path.join(obs_dir, "observations_dagger.pt")
    shutil.copy2(aggregated_file, dagger_shard_dst)
    print(f"[DAgger-Train] Shard injecté → {dagger_shard_dst}")

    tp.train_roster(
        cohort_name=cohort_name,
        roster=[pilot_name],
        mode="Commander",
        Neps=Nep,
        lim_sv=lim_sv,
    )
    print(f"[DAgger-Train] {pilot_name} : Commander mis à jour ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Métriques DAgger internes (pour le suivi par itération)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_dagger_metrics(
    rollouts: List[dict],
    iteration: int,
    beta: float,
) -> dict:
    total        = len(rollouts)
    n_collisions = sum(1 for r in rollouts if len(r["collision_steps"]) > 0)
    n_drifts     = sum(1 for r in rollouts if len(r["drift_steps"])     > 0)
    n_success    = sum(
        1 for r in rollouts if r["analysis"].get("reached_goal", False)
    )
    return {
        "iteration":      iteration,
        "beta":           beta,
        "total_rollouts": total,
        "collision_rate": n_collisions / max(total, 1),
        "drift_rate":     n_drifts     / max(total, 1),
        "success_rate":   n_success    / max(total, 1),
    }


def _print_iter_metrics(m: dict, pilot_name: str) -> None:
    print(
        f"[DAgger] iter={m['iteration']}  pilot={pilot_name}"
        f"  β={m['beta']:.3f}"
        f"  collisions={m['collision_rate']:.1%}"
        f"  dérives={m['drift_rate']:.1%}"
        f"  succès={m['success_rate']:.1%}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collecte d'un rollout DAgger (interne, pendant l'entraînement)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_collision(x, point_cloud, threshold):
    return bool(np.any(np.linalg.norm(point_cloud - x[:3], axis=1) < threshold))


def _detect_drift(x, x_ref, threshold):
    return bool(np.linalg.norm(x[:3] - x_ref[:3]) > threshold)


def _collect_dagger_rollout(
    simulator, mixed_policy, perturbation, tXUi,
    obj_name, point_cloud, goal_location,
    collision_threshold, drift_threshold, vision_processor,
) -> dict:
    mixed_policy.reset_annotations()

    Tro, Xro, Uro, Iro, Tsol, Adv = simulator.simulate(
        mixed_policy,
        perturbation["t0"], tXUi[0, -1], perturbation["x0"],
        np.zeros((18, 1)),
        query=obj_name,
        vision_processor=vision_processor,
        verbose=False,
    )

    collisions, drifts = [], []
    for k in range(Xro.shape[1]):
        xk    = Xro[:, k]
        x_ref = tXUi[1:11, min(k, tXUi.shape[1] - 1)]
        if _detect_collision(xk, point_cloud, collision_threshold):
            collisions.append(k)
        if _detect_drift(xk, x_ref, drift_threshold):
            drifts.append(k)

    analysis = analyze_trajectory_performance(
        Xro, goal_location, point_cloud,
        collision_threshold, collision_threshold * 0.6,
        trajectory_name=f"dagger_{obj_name}",
    )

    return {
        "Tro": Tro, "Xro": Xro, "Uro": Uro, "Iro": Iro,
        "Tsol": Tsol, "Adv": Adv, "tXUi": tXUi,
        "annotations":     mixed_policy.annotations,
        "collision_steps": collisions,
        "drift_steps":     drifts,
        "analysis":        analysis,
        "obj_name":        obj_name,
        "beta":            mixed_policy.beta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Boucle DAgger principale
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
) -> dict:
    """
    Boucle DAgger complète avec benchmark avant/après sur exactement
    les mêmes starts/goals (pattern review=True → backup → review=False).
    """

    # ── Chemins ──────────────────────────────────────────────────────────────
    workspace_path = str(Path(__file__).resolve().parents[3])
    cohort_path    = os.path.join(workspace_path, "cohorts", cohort_name)
    method_path    = os.path.join(workspace_path, "configs", "method",
                                  method_name + ".json")
    scenes_cfg_dir = os.path.join(workspace_path, "configs", "scenes")
    perception_dir = os.path.join(workspace_path, "configs", "perception")
    sim_base       = os.path.join(cohort_path, "simulation_data")
    os.makedirs(sim_base, exist_ok=True)

    # ── Chargement config ─────────────────────────────────────────────────────
    with open(method_path) as f:
        method_config = json.load(f)

    sample_cfg         = method_config["sample_set"]
    trajectory_set_cfg = method_config["trajectory_set"]
    frame_set_cfg      = method_config["frame_set"]
    test_cfg           = method_config["test_set"]

    base_policy_name = sample_cfg["policy"]
    base_frame_name  = sample_cfg["frame"]
    rollout_name     = sample_cfg["rollout"]
    vision_proc_type = sample_cfg.get("vision_processor", "none")
    Nrep             = test_cfg["reps"]
    Trep             = np.zeros(Nrep)

    with open(os.path.join(workspace_path, "configs", "frame",
                           base_frame_name + ".json")) as f:
        base_frame_config = json.load(f)

    with open(os.path.join(perception_dir,
                           "onnx_benchmark_config.json")) as f:
        perception_config = json.load(f)

    onnx_path = perception_config.get("onnx_model_path", None)

    # ── Scenes + collision detectors (une seule fois) ─────────────────────────
    # Pattern identique au STEP 1 du notebook simple_rl.ipynb
    print("\n[SETUP] Loading scenes and point clouds...")
    objective_configs   = {}
    collision_detectors = {}
    scene_names = list(dict.fromkeys(s for s, _ in flights))

    for scene_name in scene_names:
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            scene_cfg = yaml.safe_load(f)
        objective_configs[scene_name] = scene_cfg["queries"]

        sim = Simulator(scene_name, rollout_name)
        _, env_pcd, _, _, _, _ = scdt.rescale_point_cloud(
            sim.gsplat, viz=False, cull=False, verbose=False
        )
        collision_detectors[scene_name] = CollisionDetector(
            point_cloud=env_pcd, collision_radius=collision_threshold
        )
        print(f"  [{scene_name}] ✓  {env_pcd.shape[0]:,} pts")

    # ── Vision processor ──────────────────────────────────────────────────────
    if vision_proc_type.lower() != "none":
        proc_kwargs = {}
        if vision_proc_type.lower() == "clipseg" and onnx_path:
            proc_kwargs["onnx_model_path"] = onnx_path
        vision_processor = create_vision_processor(vision_proc_type, **proc_kwargs)
    else:
        vision_processor = None

    # ── Résultats globaux ─────────────────────────────────────────────────────
    all_metrics: dict      = {name: [] for name in roster}
    benchmark_results: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Boucle par pilot
    # ──────────────────────────────────────────────────────────────────────────
    for pilot_name in roster:
        print("=" * 70)
        print(f"[DAgger] Pilot : {pilot_name}  —  {n_iterations} itérations")
        print("=" * 70)

        pilot      = Pilot(cohort_name, pilot_name)
        pilot_path = pilot.path
        pilot.set_mode("deploy")

        # Chemins des checkpoints .pth (modèle en mémoire, pas copytree)
        bench_dir      = os.path.join(cohort_path, "dagger_data", pilot_name, "benchmark")
        before_pth     = os.path.join(bench_dir, "model_before_dagger.pth")
        after_pth      = os.path.join(bench_dir, "model_after_dagger.pth")
        rrt_backup     = os.path.join(cohort_path, "dagger_data", pilot_name,
                                      "_benchmark_rrt_backup")
        os.makedirs(bench_dir, exist_ok=True)

        # ── Sauvegarder le modèle AVANT DAgger ───────────────────────────────
        _save_model_checkpoint(pilot, before_pth)

        # ── Générer les .pkl RRT UNE SEULE FOIS (seed fixée) ─────────────────
        # Pattern STEP 3 du notebook : review=True → backup
        _generate_rrt_backup(
            pilot=pilot,
            model_path=before_pth,
            workspace_path=workspace_path,
            cohort_name=cohort_name,
            cohort_path=cohort_path,
            method_name=method_name,
            flights=flights,
            scenes_cfg_dir=scenes_cfg_dir,
            objective_configs=objective_configs,
            sim_base=sim_base,
            rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed,
            max_trajectories=max_trajectories,
        )

        # ── Benchmark AVANT DAgger (review=False, mêmes .pkl) ────────────────
        bench_before = _run_benchmark_pilot(
            pilot=pilot,
            model_path=before_pth,
            label="before_dagger",
            workspace_path=workspace_path,
            cohort_name=cohort_name,
            cohort_path=cohort_path,
            method_name=method_name,
            flights=flights,
            scenes_cfg_dir=scenes_cfg_dir,
            objective_configs=objective_configs,
            collision_detectors=collision_detectors,
            scene_names=scene_names,
            sim_base=sim_base,
            rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed,
            max_trajectories=max_trajectories,
        )
        print(f"[DAgger-Bench] AVANT → collision={bench_before['collision_rate'].mean()*100:.1f}%")

        # ─────────────────────────────────────────────────────────────────────
        # Boucle DAgger
        # ─────────────────────────────────────────────────────────────────────
        beta = beta_start

        for iteration in range(n_iterations):
            print(f"\n── Itération {iteration}/{n_iterations - 1}  β={beta:.3f} ──")

            new_rollouts: List[dict] = []

            for scene_name, course_name in flights:
                with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
                    scene_cfg = yaml.safe_load(f)

                objectives   = scene_cfg["queries"]
                radii        = scene_cfg["radii"]
                altitudes    = scene_cfg["altitudes"]
                similarities = scene_cfg.get("similarities", None)

                simulator = Simulator(scene_name, rollout_name)
                obj_targets, _, _, epcds_arr = bd.get_objectives(
                    simulator.gsplat, objectives, similarities, False
                )
                _, obj_centroids = th.process_RRT_objectives(
                    obj_targets, epcds_arr, {}, radii, altitudes
                )
                Frames = gd.generate_frames(Trep, base_frame_config, frame_set_cfg)

                for obj_idx, obj_name in enumerate(objectives):
                    latest_dir    = _latest_sim_dir_by_mtime(sim_base)
                    filtered_file = os.path.join(
                        latest_dir, "rrt_planning",
                        f"{scene_name}_filtered_{obj_name}.pkl"
                    )
                    with open(filtered_file, "rb") as f:
                        filtered_trajectories = pickle.load(f)

                    traj_list, _, _ = th.parameterize_RRT_trajectories(
                        filtered_trajectories, obj_centroids[obj_idx],
                        1.0, 20, randint=0
                    )
                    tXUi = traj_list[0]

                    Perturbations = gd.generate_perturbations(
                        Tsps=Trep, tXUi=tXUi,
                        trajectory_set_config=trajectory_set_cfg
                    )

                    for frame, perturbation in zip(Frames, Perturbations):
                        simulator.load_frame(frame)
                        expert = VehicleRateMPC(
                            tXUi, base_policy_name, base_frame_name, "expert"
                        )
                        mixed = MixedPolicy(expert, pilot, beta)

                        rollout = _collect_dagger_rollout(
                            simulator, mixed, perturbation, tXUi,
                            obj_name, epcds_arr, obj_targets[obj_idx],
                            collision_threshold, drift_threshold, vision_processor,
                        )
                        new_rollouts.append(rollout)

                        print(
                            f"  [{obj_name}]  coll={len(rollout['collision_steps'])}"
                            f"  drift={len(rollout['drift_steps'])}"
                            f"  ok={rollout['analysis'].get('reached_goal', False)}"
                        )

            # ── Agrégation + re-entraînement ──────────────────────────────────
            aggregated_file = aggregate_dagger_dataset(
                cohort_path, pilot_name, new_rollouts, iteration
            )
            _retrain_commander(cohort_name, pilot_name, aggregated_file, Nep_per_iter, lim_sv)

            # Recharger le pilot avec les nouveaux poids
            pilot = Pilot(cohort_name, pilot_name)
            pilot.set_mode("deploy")

            m = _compute_dagger_metrics(new_rollouts, iteration, beta)
            all_metrics[pilot_name].append(m)
            _print_iter_metrics(m, pilot_name)

            beta = max(0.0, beta * beta_decay)

        # ── Sauvegarder le modèle APRÈS DAgger ───────────────────────────────
        _save_model_checkpoint(pilot, after_pth)

        # ── Benchmark APRÈS DAgger (exactement mêmes .pkl) ───────────────────
        bench_after = _run_benchmark_pilot(
            pilot=pilot,
            model_path=after_pth,
            label="after_dagger",
            workspace_path=workspace_path,
            cohort_name=cohort_name,
            cohort_path=cohort_path,
            method_name=method_name,
            flights=flights,
            scenes_cfg_dir=scenes_cfg_dir,
            objective_configs=objective_configs,
            collision_detectors=collision_detectors,
            scene_names=scene_names,
            sim_base=sim_base,
            rrt_backup=rrt_backup,
            benchmark_seed=benchmark_seed,
            max_trajectories=max_trajectories,
        )

        _print_benchmark_comparison(bench_before, bench_after, pilot_name)

        # Sauvegarder JSON
        bench_json = os.path.join(bench_dir, "benchmark_results.json")
        with open(bench_json, "w") as f:
            json.dump({
                "before": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in bench_before.items()},
                "after":  {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in bench_after.items()},
            }, f, indent=2)
        print(f"[DAgger-Bench] JSON sauvegardé → {bench_json}")

        benchmark_results[pilot_name] = {
            "before": bench_before,
            "after":  bench_after,
        }

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    if vision_processor is not None:
        del vision_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Retourne métriques DAgger + résultats benchmark
    return {"metrics": all_metrics, "benchmark": benchmark_results}