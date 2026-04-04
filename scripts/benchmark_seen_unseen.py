#!/usr/bin/env python3
"""
Overnight benchmark: SEEN vs UNSEEN branches for V9 DAgger.

Split BC branches into:
  - SEEN: 50 branches sampled from the 110 training branches per object
  - UNSEEN: 11 validation branches per object (from generate-rollouts --validation-mode)

Runs both BC (before_dagger) and DAgger (after_dagger) models on each set.
Total: 4 benchmark passes × 3 objects = ~12 × N_runs simulations.

Results saved to: cohorts/SSV_DAGGER_CENTROID_V9/benchmark_seen_unseen/
"""
import os, sys, json, yaml, time, glob
import numpy as np
import torch
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
os.chdir(WORKSPACE)

from sousvide.instruct.train_dagger import (
    _run_benchmark_pilot, _get_scene, _generate_rrt_backup,
    _BC_TRAJ_CACHE, _BRANCHES_CACHE, _PKL_CACHE,
    Pilot, DEVICE,
)

# ── Config ──────────────────────────────────────────────────────────────────
config_path = os.path.join(WORKSPACE, "configs", "experiment", "ssv_dagger_centroid_v9.yml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cohort_name    = cfg["cohort"]          # SSV_DAGGER_CENTROID_V9
method_name    = cfg["method"]          # rrt_6s
bc_cohort_name = cfg["bc_cohort"]       # ssv_BC_CENTROID_V9
roster         = cfg["roster"]
flights        = [tuple(x) for x in cfg["flights"]]
BENCHMARK_SEED = 42
SAVE_PLOTS  = True     # Save trajectory plots (3D + time series) per run
SAVE_VIDEOS = False    # Save MP4 videos per run (WARNING: ~50MB each, 600 runs = ~30GB)

cohort_path    = os.path.join(WORKSPACE, "cohorts", cohort_name)
method_path    = os.path.join(WORKSPACE, "configs", "method", method_name + ".json")
scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")

with open(method_path) as f:
    mcfg = json.load(f)
frame_name   = mcfg["sample_set"]["frame"]
rollout_name = mcfg["sample_set"]["rollout"]

# ── Load scene ──────────────────────────────────────────────────────────────
print("Loading scene (gsplat)...")
scene_names = []
objective_configs = {}
for scene_name, _ in flights:
    if scene_name not in objective_configs:
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            objective_configs[scene_name] = yaml.safe_load(f)
        scene_names.append(scene_name)
    _get_scene(scene_name, scenes_cfg_dir, frame_name, rollout_name)

# ── Split BC trajectories into SEEN (train) and UNSEEN (val) ───────────────
print("\nSplitting BC trajectories into SEEN/UNSEEN...")

seen_trajs = {}   # {scene_obj: [tXUd, ...]}
unseen_trajs = {}

for scene_name, _ in flights:
    scene_data = objective_configs[scene_name]
    queries = scene_data["queries"]

    rollout_dir = os.path.join(
        WORKSPACE, "cohorts", bc_cohort_name, "rollout_data", scene_name
    )

    # Get object target locations from scene cache
    from sousvide.instruct.train_dagger import _SCENE_CACHE
    obj_targets = _SCENE_CACHE[scene_name]["obj_targets"]
    obj_locs = np.array([np.squeeze(t) for t in obj_targets])

    # Load training trajectories (trajectories00000.pt, etc.)
    train_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories[0-9]*.pt")))
    # Load validation trajectories (trajectories_val*.pt)
    val_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))

    print(f"  {scene_name}: {len(train_files)} train files, {len(val_files)} val files")

    # Assign to objects by goal proximity
    train_per_obj = {i: [] for i in range(len(queries))}
    val_per_obj = {i: [] for i in range(len(queries))}

    for tf in train_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best_obj = int(np.argmin(dists))
        if float(dists[best_obj]) < 5.0:
            train_per_obj[best_obj].append(tXUd)

    for tf in val_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best_obj = int(np.argmin(dists))
        if float(dists[best_obj]) < 5.0:
            val_per_obj[best_obj].append(tXUd)

    for obj_idx, obj_name in enumerate(queries):
        key = f"{scene_name}_{obj_name}"
        seen_trajs[key] = train_per_obj[obj_idx]
        unseen_trajs[key] = val_per_obj[obj_idx]
        print(f"    '{obj_name}': {len(train_per_obj[obj_idx])} seen, {len(val_per_obj[obj_idx])} unseen")

# ── Create pilot ────────────────────────────────────────────────────────────
pilot_name = roster[0]
pilot = Pilot(cohort_name, pilot_name)
pilot.set_mode("deploy")
pilot.model.to(DEVICE)

# ── Model paths ─────────────────────────────────────────────────────────────
roster_dir = os.path.join(cohort_path, "roster", pilot_name)
model_before_path = os.path.join(roster_dir, "model_before_dagger.pth")
model_after_path = os.path.join(roster_dir, "model.pth")

print(f"\nBC model: {model_before_path} (exists: {os.path.isfile(model_before_path)})")
print(f"DAgger model: {model_after_path} (exists: {os.path.isfile(model_after_path)})")

# ── Setup output ────────────────────────────────────────────────────────────
dagger_dir = os.path.join(cohort_path, "dagger_data", pilot_name)
sim_base   = os.path.join(cohort_path, "simulation_data", "dagger")
rrt_backup = os.path.join(dagger_dir, "_benchmark_rrt_backup")
os.makedirs(sim_base, exist_ok=True)
os.makedirs(rrt_backup, exist_ok=True)

output_dir = os.path.join(cohort_path, "post_training_benchmarks",
                          datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(output_dir, exist_ok=True)

collision_detectors = {}

# ── Helper: inject custom branches into caches ──────────────────────────────
def set_branch_cache(branch_dict):
    """Replace _BC_TRAJ_CACHE and clear _BRANCHES_CACHE so _load_all_branches reloads."""
    _BC_TRAJ_CACHE.clear()
    _BRANCHES_CACHE.clear()
    for key, trajs in branch_dict.items():
        _BC_TRAJ_CACHE[key] = trajs
        # Also set _PKL_CACHE for backwards compat
        if trajs and key not in _PKL_CACHE:
            _PKL_CACHE[key] = {
                "tXUi": trajs[0],
                "obj_loc": trajs[0][1:4, -1],
            }

# ── Run all 4 benchmarks ───────────────────────────────────────────────────
results = {}
total_start = time.time()

benchmarks = [
    ("SEEN_BC",       seen_trajs,   model_before_path, 50),
    ("SEEN_DAGGER",   seen_trajs,   model_after_path,  50),
    ("UNSEEN_BC",     unseen_trajs, model_before_path, 11),
    ("UNSEEN_DAGGER", unseen_trajs, model_after_path,  11),
]

for bench_label, branch_set, model_path, max_traj in benchmarks:
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {bench_label}  (max_traj={max_traj})")
    print(f"{'='*70}")

    # Inject the appropriate branch set
    set_branch_cache(branch_set)

    t0 = time.time()
    try:
        metrics = _run_benchmark_pilot(
            pilot=pilot, model_path=model_path, label=bench_label,
            workspace_path=WORKSPACE, cohort_name=cohort_name,
            cohort_path=cohort_path, method_name=method_name, flights=flights,
            scenes_cfg_dir=scenes_cfg_dir, objective_configs=objective_configs,
            collision_detectors=collision_detectors, scene_names=scene_names,
            sim_base=sim_base, rrt_backup=rrt_backup,
            benchmark_seed=BENCHMARK_SEED, max_trajectories=max_traj,
            save_plots=SAVE_PLOTS, save_videos=SAVE_VIDEOS,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0

        sr = float(metrics.get("success_rate", np.array([0])).mean()) * 100
        cr = float(metrics.get("collision_rate", np.array([0])).mean()) * 100
        results[bench_label] = {"success": sr, "collision": cr, "time": elapsed, "metrics": metrics}
        print(f"\n  >>> {bench_label}: success={sr:.1f}%  collision={cr:.1f}%  ({elapsed:.0f}s)")

    except Exception as e:
        print(f"\n  >>> {bench_label} FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[bench_label] = {"success": -1, "collision": -1, "error": str(e)}

total_elapsed = time.time() - total_start

# ── Final summary ───────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"FINAL RESULTS — SEEN vs UNSEEN BENCHMARK")
print(f"{'='*70}")
print(f"Total time: {total_elapsed/3600:.1f} hours")
print(f"")
print(f"{'Set':<18} {'Model':<10} {'Success':>8} {'Collision':>10} {'N/obj':>6}")
print(f"{'-'*52}")
for label, r in results.items():
    parts = label.split("_", 1)
    set_name = parts[0]
    model_name = parts[1] if len(parts) > 1 else ""
    n = 50 if "SEEN" in label else 11
    if r["success"] >= 0:
        print(f"{set_name:<18} {model_name:<10} {r['success']:>7.1f}% {r['collision']:>9.1f}% {n:>6}")
    else:
        print(f"{set_name:<18} {model_name:<10} {'FAILED':>8} {'':>10} {n:>6}")

print(f"\nReference (original V9 training log, on 50 seen branches):")
print(f"  BC baseline:  80.7% success, 13.3% collision")
print(f"  After DAgger: 88.0% success,  8.0% collision")

# ── Save results ────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(output_dir, f"results_{timestamp}.json")
save_data = {}
for label, r in results.items():
    save_entry = {k: v for k, v in r.items() if k != "metrics"}
    if "metrics" in r and isinstance(r["metrics"], dict):
        # Convert numpy arrays to lists for JSON
        for mk, mv in r["metrics"].items():
            if hasattr(mv, "tolist"):
                save_entry[f"metric_{mk}"] = mv.tolist()
    save_data[label] = save_entry

with open(result_file, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to: {result_file}")

# Also write a human-readable summary
summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
with open(summary_file, "w") as f:
    f.write(f"V9 DAgger — Seen vs Unseen Benchmark\n")
    f.write(f"Date: {datetime.now().isoformat()}\n")
    f.write(f"Total time: {total_elapsed/3600:.1f} hours\n")
    f.write(f"Seed: {BENCHMARK_SEED}\n\n")
    f.write(f"{'Set':<18} {'Model':<10} {'Success':>8} {'Collision':>10} {'N/obj':>6}\n")
    f.write(f"{'-'*52}\n")
    for label, r in results.items():
        parts = label.split("_", 1)
        set_name = parts[0]
        model_name = parts[1] if len(parts) > 1 else ""
        n = 50 if "SEEN" in label else 11
        if r["success"] >= 0:
            f.write(f"{set_name:<18} {model_name:<10} {r['success']:>7.1f}% {r['collision']:>9.1f}% {n:>6}\n")
        else:
            f.write(f"{set_name:<18} {model_name:<10} {'FAILED':>8} {'':>10} {n:>6}\n")
    f.write(f"\nReference (original V9 training log):\n")
    f.write(f"  BC:     80.7% success, 13.3% collision (50/obj)\n")
    f.write(f"  DAgger: 88.0% success,  8.0% collision (50/obj)\n")

print(f"Summary saved to: {summary_file}")
print(f"\nDONE.")
