#!/usr/bin/env python3
"""
Reproduce the EXACT benchmark from V9 DAgger training.
Calls the same _run_benchmark_pilot with the same setup as train_dagger_policy.
"""
import os, sys, json, yaml, time
import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
os.chdir(WORKSPACE)

from sousvide.instruct.train_dagger import (
    _run_benchmark_pilot, _get_scene, _preload_bc_trajectories,
    _generate_rrt_backup, Pilot, DEVICE,
)
from sousvide.flight.vision_processor_base import create_vision_processor

# ── Load V9 config (same as training) ────────────────────────────────────────
config_path = os.path.join(WORKSPACE, "configs", "experiment", "ssv_dagger_centroid_v9.yml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cohort_name    = cfg["cohort"]          # SSV_DAGGER_CENTROID_V9
method_name    = cfg["method"]          # rrt_6s
bc_cohort_name = cfg["bc_cohort"]       # ssv_BC_CENTROID_V9
roster         = cfg["roster"]          # ["InstinctJester"]
flights        = [tuple(x) for x in cfg["flights"]]
benchmark_seed = 42                     # hardcoded in training
max_trajectories = cfg.get("n_benchmark", 50)

print(f"Cohort: {cohort_name}")
print(f"BC cohort: {bc_cohort_name}")
print(f"Method: {method_name}")
print(f"Benchmark seed: {benchmark_seed}")
print(f"Max trajectories: {max_trajectories}")

# ── Exact same setup as train_dagger_policy ──────────────────────────────────
cohort_path    = os.path.join(WORKSPACE, "cohorts", cohort_name)
method_path    = os.path.join(WORKSPACE, "configs", "method", method_name + ".json")
scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")

objective_configs   = {}
collision_detectors = {}
scene_names         = []
for scene_name, _ in flights:
    if scene_name not in objective_configs:
        with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
            objective_configs[scene_name] = yaml.safe_load(f)
        scene_names.append(scene_name)

with open(method_path) as f:
    mcfg = json.load(f)
frame_name   = mcfg.get("sample_set", {}).get("frame", "carl")
rollout_name = mcfg.get("sample_set", {}).get("rollout", "baseline")

# Pre-load scene (same as training)
print(f"\nLoading scene...")
for sn in scene_names:
    _get_scene(sn, scenes_cfg_dir, frame_name, rollout_name)

# Load BC trajectories (same as training)
n_bc = _preload_bc_trajectories(bc_cohort_name, flights, scenes_cfg_dir)
print(f"BC trajectories loaded: {n_bc}")

# ── Create pilot (same as line 2374 in training) ────────────────────────────
pilot_name = roster[0]
pilot = Pilot(cohort_name, pilot_name)
pilot.set_mode("deploy")
pilot.model.to(DEVICE)

# ── Setup paths (same as training) ──────────────────────────────────────────
dagger_dir = os.path.join(cohort_path, "dagger_data", pilot_name)
sim_base   = os.path.join(cohort_path, "simulation_data", "dagger")
rrt_backup = os.path.join(dagger_dir, "_benchmark_rrt_backup")
os.makedirs(sim_base, exist_ok=True)
os.makedirs(rrt_backup, exist_ok=True)

# ── Model paths ─────────────────────────────────────────────────────────────
roster_dir = os.path.join(cohort_path, "roster", pilot_name)
# The "before" model = BC model (same as model_before_dagger.pth)
model_before_path = os.path.join(roster_dir, "model_before_dagger.pth")
# The "after" model = DAgger best (model.pth in roster = model_after_dagger.pth)
model_after_path = os.path.join(
    dagger_dir, "benchmark_20260321_004347", "model_after_dagger.pth")

print(f"\nBefore model: {model_before_path} (exists: {os.path.isfile(model_before_path)})")
print(f"After model: {model_after_path} (exists: {os.path.isfile(model_after_path)})")

# ── Run BEFORE benchmark (BC model) ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"BEFORE DAGGER BENCHMARK (BC model)")
print(f"{'='*70}")
t0 = time.time()
metrics_before = _run_benchmark_pilot(
    pilot=pilot, model_path=model_before_path, label="before_dagger",
    workspace_path=WORKSPACE, cohort_name=cohort_name,
    cohort_path=cohort_path, method_name=method_name, flights=flights,
    scenes_cfg_dir=scenes_cfg_dir, objective_configs=objective_configs,
    collision_detectors=collision_detectors, scene_names=scene_names,
    sim_base=sim_base, rrt_backup=rrt_backup,
    benchmark_seed=benchmark_seed, max_trajectories=max_trajectories,
)
print(f"Before benchmark done in {time.time()-t0:.0f}s")

# ── Run AFTER benchmark (DAgger best model) ─────────────────────────────────
print(f"\n{'='*70}")
print(f"AFTER DAGGER BENCHMARK (DAgger best model)")
print(f"{'='*70}")
t0 = time.time()
metrics_after = _run_benchmark_pilot(
    pilot=pilot, model_path=model_after_path, label="after_dagger",
    workspace_path=WORKSPACE, cohort_name=cohort_name,
    cohort_path=cohort_path, method_name=method_name, flights=flights,
    scenes_cfg_dir=scenes_cfg_dir, objective_configs=objective_configs,
    collision_detectors=collision_detectors, scene_names=scene_names,
    sim_base=sim_base, rrt_backup=rrt_backup,
    benchmark_seed=benchmark_seed, max_trajectories=max_trajectories,
)
print(f"After benchmark done in {time.time()-t0:.0f}s")

# ── Compare with original ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"COMPARISON WITH ORIGINAL RESULTS")
print(f"{'='*70}")

for label, metrics in [("before", metrics_before), ("after", metrics_after)]:
    sr = float(metrics.get("success_rate", np.array([0])).mean()) * 100
    cr = float(metrics.get("collision_rate", np.array([0])).mean()) * 100
    print(f"  {label}: success={sr:.1f}%  collision={cr:.1f}%")

print(f"\nOriginal results (from benchmark_results.json):")
print(f"  before: success=80.7%  collision=13.3%")
print(f"  after:  success=88.0%  collision=8.0%")
