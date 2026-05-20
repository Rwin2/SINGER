#!/usr/bin/env python3
"""Quick standalone test of run_unseen_benchmark() from train_rl_ppo.py.

Loads the DAgger V9 model and runs the unseen benchmark on just 1 run per object
(3 total) to verify no quaternion crashes. If this passes, PPO v2 benchmark is safe.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_unseen_benchmark.py
"""
import os, sys, time, json, copy
import numpy as np
import torch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

from src.sousvide.instruct.train_rl_ppo import (
    _load_val_trajectories_per_object,
    run_unseen_benchmark,
    _get_scene,
)

# Config
SOURCE_COHORT = "SSV_DAGGER_CENTROID_V9"
BC_COHORT = "ssv_BC_CENTROID_V9"
SCENE = "flightroom_ssv_exp"

print("=== Unseen Benchmark Smoke Test ===")
print(f"Source model: {SOURCE_COHORT}")
print(f"BC cohort: {BC_COHORT}")
print(f"Scene: {SCENE}")

# 1. Load scene
scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")
scene_data = _get_scene(SCENE, scenes_cfg_dir)
simulator = scene_data["simulator"]
obj_targets = scene_data["obj_targets"]
queries = scene_data["queries"]
epcds_arr = scene_data.get("epcds_arr", None)
env_min = scene_data.get("env_min", None)
env_max = scene_data.get("env_max", None)

print(f"\nObjects: {queries}")
print(f"obj_targets shapes: {[t.shape for t in obj_targets]}")

# 2. Load validation trajectories
val_per_obj = _load_val_trajectories_per_object(
    WORKSPACE, BC_COHORT, SCENE, obj_targets, queries
)
for name, trajs in val_per_obj.items():
    print(f"  {name}: {len(trajs)} val trajectories")

# 3. Build pilot (same way train_rl_ppo.py does it)
from sousvide.control.pilot import Pilot

# Copy model to a temp cohort to avoid modifying the original
TEST_COHORT = "SSV_RL_PPO_V2_TEST"
dst_roster = os.path.join(WORKSPACE, "cohorts", TEST_COHORT, "roster", "InstinctJester")
os.makedirs(dst_roster, exist_ok=True)
import shutil
src_roster = os.path.join(WORKSPACE, "cohorts", SOURCE_COHORT, "roster", "InstinctJester")
for fname in ("model.pth", "config.json", "losses_Commander.pt"):
    src = os.path.join(src_roster, fname)
    dst = os.path.join(dst_roster, fname)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

pilot = Pilot(TEST_COHORT, "InstinctJester")
pilot.set_mode("deploy")
pilot.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = pilot.model
model_path = os.path.join(dst_roster, "model.pth")

# 5. Use ALL validation trajectories (11 per object) for proper benchmark
# 6. Run benchmark
print(f"\n>>> Running unseen benchmark (full trajectory from t=0, all val trajectories)...")
t0 = time.time()
result = run_unseen_benchmark(
    pilot=pilot,
    model=model,
    model_path=model_path,
    label="dagger_v9_baseline",
    val_per_obj=val_per_obj,
    simulator=simulator,
    obj_targets=obj_targets,
    queries=queries,
    epcds_arr=epcds_arr,
    env_min=env_min,
    env_max=env_max,
)
elapsed = time.time() - t0

print(f"\n=== RESULT ({elapsed:.0f}s) ===")
print(f"Overall: {result}")
print("\n>>> BENCHMARK TEST PASSED - no quaternion crashes!")
