#!/usr/bin/env python3
"""
Diagnostic: compare old (75th percentile) vs new (median) centroid methods
on actual rendered images to see if they produce different features.
"""
import os, sys, yaml, json
import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
os.chdir(WORKSPACE)

from sousvide.instruct.train_dagger import _get_scene, _preload_bc_trajectories, _get_pkl, Pilot, DEVICE
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup
config_path = os.path.join(WORKSPACE, "configs", "experiment", "ssv_dagger_centroid_v9.yml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

flights = [tuple(x) for x in cfg["flights"]]
scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")
method_path = os.path.join(WORKSPACE, "configs", "method", cfg["method"] + ".json")

with open(method_path) as f:
    mcfg = json.load(f)
frame_name = mcfg.get("sample_set", {}).get("frame", "carl")
rollout_name = mcfg.get("sample_set", {}).get("rollout", "baseline")

# Load scene
scene_data = _get_scene("flightroom_ssv_exp", scenes_cfg_dir, frame_name, rollout_name)
simulator = scene_data["simulator"]
queries = scene_data["queries"]

# Load BC trajs for branch data
_preload_bc_trajectories(cfg["bc_cohort"], flights, scenes_cfg_dir)

# Get a branch to simulate
from sousvide.instruct.train_dagger import _load_all_branches
pkl_data = _get_pkl("flightroom_ssv_exp", queries[0], scenes_cfg_dir)
all_branches = _load_all_branches(
    "flightroom_ssv_exp", queries[0],
    os.path.join(WORKSPACE, "cohorts", cfg["cohort"]),
    scenes_cfg_dir, pkl_data["tXUi"],
)

# Use seed=42, first branch selection (same as benchmark)
np.random.seed(42)
branch_idxs = np.random.choice(len(all_branches), size=50, replace=False)
tXUi = all_branches[branch_idxs[0]]

# Render one image
x0 = tXUi[1:11, 0].copy()
t0 = float(tXUi[0, 0])

import figs.utilities.trajectory_helper as th
cam_cfg = simulator.conFiG["drone"]["camera"]
T_c2b = simulator.conFiG["drone"]["T_c2b"]
camera = simulator.gsplat.generate_output_camera(cam_cfg)

Tb2w = th.xv_to_T(x0)
T_c2w = Tb2w @ T_c2b
extra_ch = getattr(simulator.gsplat, "extra_channels", [])
image_dict = simulator.gsplat.render_rgb(camera, T_c2w, queries[0], extra_channels=extra_ch)
icr_raw = image_dict["semantic"]  # turbo-colormapped uint8

print(f"Raw image shape: {icr_raw.shape}, dtype: {icr_raw.dtype}, range: [{icr_raw.min()}, {icr_raw.max()}]")

# Apply same transform as Pilot.process_image
transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
icr_normalized = transform(image=icr_raw)["image"]  # (3, 224, 224) float tensor
print(f"Normalized image shape: {icr_normalized.shape}, range: [{icr_normalized.min():.3f}, {icr_normalized.max():.3f}]")

# Compute heat map (same as _compute_centroid)
img = icr_normalized.numpy()
heat = img.mean(axis=0)  # (224, 224)
print(f"Heat map range: [{heat.min():.4f}, {heat.max():.4f}]")

if heat.max() > 1.0:
    heat_divided = heat / 255.0
    print(f"After /255: range [{heat_divided.min():.6f}, {heat_divided.max():.6f}]")
    print(f"  → This is WRONG for ImageNet-normalized data! Values are tiny.")
else:
    heat_divided = heat
    print(f"No /255 applied (values already in [-1, 1])")

# Compare: OLD method (75th percentile)
heat_work = heat_divided  # use what the code actually does
p75 = np.percentile(heat_work, 75)
mask_old = heat_work > p75
if mask_old.sum() > 0:
    ys_old, xs_old = np.where(mask_old)
    w_old = heat_work[mask_old]
    cx_old = np.average(xs_old, weights=w_old) / 224
    cy_old = np.average(ys_old, weights=w_old) / 224
    bearing_old = 2.0 * cx_old - 1.0
    elevation_old = 2.0 * cy_old - 1.0
    print(f"\nOLD (75th percentile):")
    print(f"  threshold = {p75:.6f}")
    print(f"  pixels above = {mask_old.sum()} ({mask_old.sum()/(224*224)*100:.1f}%)")
    print(f"  bearing = {bearing_old:.4f}, elevation = {elevation_old:.4f}")

# Compare: NEW method (median)
median_val = np.median(heat_work)
mask_new = heat_work > median_val
if mask_new.sum() > 0:
    ys_new, xs_new = np.where(mask_new)
    w_new = heat_work[mask_new]
    cx_new = np.average(xs_new, weights=w_new) / 224
    cy_new = np.average(ys_new, weights=w_new) / 224
    bearing_new = 2.0 * cx_new - 1.0
    elevation_new = 2.0 * cy_new - 1.0
    print(f"\nNEW (median):")
    print(f"  threshold = {median_val:.6f}")
    print(f"  pixels above = {mask_new.sum()} ({mask_new.sum()/(224*224)*100:.1f}%)")
    print(f"  bearing = {bearing_new:.4f}, elevation = {elevation_new:.4f}")

# Difference
if mask_old.sum() > 0 and mask_new.sum() > 0:
    print(f"\nDIFFERENCE:")
    print(f"  bearing delta  = {bearing_new - bearing_old:+.4f}")
    print(f"  elevation delta = {elevation_new - elevation_old:+.4f}")
    if abs(bearing_new - bearing_old) > 0.05 or abs(elevation_new - elevation_old) > 0.05:
        print(f"  ⚠️ SIGNIFICANT difference! This would change model behavior.")
    else:
        print(f"  Small difference — probably not the root cause.")

# Also check: what if we DON'T divide by 255?
print(f"\n{'='*60}")
print(f"COMPARISON WITHOUT /255 NORMALIZATION")
heat_raw = heat  # no /255
p75_r = np.percentile(heat_raw, 75)
mask_old_r = heat_raw > p75_r
median_r = np.median(heat_raw)
mask_new_r = heat_raw > median_r
if mask_old_r.sum() > 0 and mask_new_r.sum() > 0:
    ys_o, xs_o = np.where(mask_old_r)
    w_o = heat_raw[mask_old_r]
    cx_o, cy_o = np.average(xs_o, weights=w_o)/224, np.average(ys_o, weights=w_o)/224
    ys_n, xs_n = np.where(mask_new_r)
    w_n = heat_raw[mask_new_r]
    cx_n, cy_n = np.average(xs_n, weights=w_n)/224, np.average(ys_n, weights=w_n)/224
    print(f"  OLD: bearing={2*cx_o-1:.4f}, elevation={2*cy_o-1:.4f} (p75={p75_r:.4f})")
    print(f"  NEW: bearing={2*cx_n-1:.4f}, elevation={2*cy_n-1:.4f} (median={median_r:.4f})")
    print(f"  Delta: bearing={2*(cx_n-cx_o):+.4f}, elevation={2*(cy_n-cy_o):+.4f}")

# Also test: what did the ORIGINAL V9 code ACTUALLY look like?
# From the handoff: "used 75th percentile as threshold"
# The original was likely: threshold = np.percentile(heat, 75)
# where heat was computed from the RAW turbo image (before ImageNet normalization)
print(f"\n{'='*60}")
print(f"WHAT IF OLD CODE USED RAW (uint8) IMAGE?")
if icr_raw.ndim == 3:
    heat_uint8 = icr_raw.mean(axis=2).astype(np.float32)
else:
    heat_uint8 = icr_raw.astype(np.float32)
print(f"  Raw heat range: [{heat_uint8.min():.1f}, {heat_uint8.max():.1f}]")
p75_raw = np.percentile(heat_uint8, 75)
mask_raw = heat_uint8 > p75_raw
if mask_raw.sum() > 0:
    ys_r, xs_r = np.where(mask_raw)
    w_r = heat_uint8[mask_raw]
    cx_r, cy_r = np.average(xs_r, weights=w_r)/icr_raw.shape[1], np.average(ys_r, weights=w_r)/icr_raw.shape[0]
    print(f"  p75={p75_raw:.1f}, pixels={mask_raw.sum()}")
    print(f"  bearing={2*cx_r-1:.4f}, elevation={2*cy_r-1:.4f}")

print(f"\nDone.")
