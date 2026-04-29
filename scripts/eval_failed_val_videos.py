#!/usr/bin/env python3
"""
Re-run failed (+ 1 success per object) validation trajectories and save
RGB / semantic / depth videos using the same format as deploy_ssv.py.

On the RGB frames, overlays:
  • CLIPSeg centroid (RED)  — what the model actually uses as goal bearing
  • GT centroid    (GREEN) — ground-truth target projected to image (for analysis only)

Uses the existing pipeline code: _get_scene, _make_terminal_fn, _evaluate_run,
Resize(720,1280), imageio.mimwrite at 20fps.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/eval_failed_val_videos.py
"""
import os, sys, time, glob, json
from datetime import datetime

import numpy as np
import torch
import cv2
import imageio
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from torchvision.transforms import Resize

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.control.pilot import Pilot
from sousvide.instruct.train_dagger import (
    DEVICE, SUCCESS_RADIUS, _evaluate_run, _get_scene, _make_terminal_fn,
)
import figs.tsampling.build_rrt_dataset as bd

# ── Config ──────────────────────────────────────────────────────────────
SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_CENTROID_V9"
PILOT_NAME = "InstinctJester"
FPS = 20

# Failed val indices per object + 1 success each
CASES = {
    "green clock":                                     {"failed": [4, 5, 6], "success": [0]},
    "yellow handheld cordless drill on two boxes":     {"failed": [6, 7],    "success": [0]},
    "green and pink leafblower":                       {"failed": [1],       "success": [0]},
}

# Camera (from carl.json) for GT projection
FX, FY = 462.956, 463.002
CX, CY = 323.076, 181.184
T_C2B = np.array([
    [ 0.0,  0.0, -1.0,  0.10],
    [ 1.0,  0.0,  0.0, -0.03],
    [ 0.0, -1.0,  0.0, -0.01],
    [ 0.0,  0.0,  0.0,  1.00],
])
# deploy_ssv.py video resize
transform = Resize((720, 1280), antialias=True)


# ── Helpers ─────────────────────────────────────────────────────────────

def _xv_to_T(xv):
    """State vector → 4×4 body-to-world transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(xv[6:10]).as_matrix()
    T[:3, 3] = xv[:3]
    return T


def _project_gt(obj_target, xcr):
    """Project ground-truth 3D target to pixel coords.
    Returns (u, v) or None if behind camera.
    Camera convention: nerfstudio/OpenGL (forward = -Z, Y up)."""
    T_c2w = _xv_to_T(xcr) @ T_C2B
    T_w2c = np.linalg.inv(T_c2w)
    pt = T_w2c @ np.array([*np.squeeze(obj_target), 1.0])
    if pt[2] >= -0.01:          # behind camera
        return None
    depth = -pt[2]
    u = FX * pt[0] / depth + CX
    v = FY * (-pt[1]) / depth + CY   # flip Y: OpenGL Y-up → pixel Y-down
    return int(round(u)), int(round(v))


# ── Confidence gate + dead-reckoning ──

CONF_GATE = 0.90       # minimum V12 confidence to trust centroid
MAX_PIXELS = 15000     # max blob size — larger = diffuse, unreliable
IMG_W, IMG_H = 640, 360


def _deadreckon_bearing(last_bearing, last_elevation, last_xcr, current_xcr):
    """Predict goal bearing in current frame from last confident observation.

    Uses exact camera model: transform the goal ray from the last camera frame
    to the current camera frame using the drone's rotation change.
    No approximations — accounts for yaw, pitch, and roll.

    Returns (bearing, elevation, u_px, v_px) or None if goal is behind camera.
    """
    # Reconstruct the 3D ray direction in the last camera frame
    # bearing ∈ [-1,1] → pixel u;  elevation ∈ [-1,1] → pixel v
    u_last = (last_bearing + 1.0) * IMG_W / 2.0
    v_last = (last_elevation + 1.0) * IMG_H / 2.0

    # Pixel → ray in camera frame (OpenGL: forward=-Z, right=+X, up=+Y)
    ray_cam = np.array([
        (u_last - CX) / FX,
        -(v_last - CY) / FY,      # flip Y: pixel Y-down → OpenGL Y-up
        -1.0,
    ])
    ray_cam /= np.linalg.norm(ray_cam)

    # Camera-to-world at last step → ray in world frame
    T_c2w_last = _xv_to_T(last_xcr) @ T_C2B
    ray_world = T_c2w_last[:3, :3] @ ray_cam

    # World-to-camera at current step → ray in current camera frame
    T_c2w_now = _xv_to_T(current_xcr) @ T_C2B
    R_w2c_now = T_c2w_now[:3, :3].T
    ray_now = R_w2c_now @ ray_world

    # Check if goal is behind current camera (OpenGL: forward = -Z)
    if ray_now[2] >= -1e-6:
        return None   # goal behind camera

    # Project to pixel
    depth = -ray_now[2]
    u_new = FX * ray_now[0] / depth + CX
    v_new = FY * (-ray_now[1]) / depth + CY

    bearing_new = 2.0 * (u_new / IMG_W) - 1.0
    elevation_new = 2.0 * (v_new / IMG_H) - 1.0

    return bearing_new, elevation_new, int(round(u_new)), int(round(v_new))


def _centroid_from_semantic(sem_frame):
    """Replicate Pilot._compute_centroid V9 on a semantic frame.
    Returns dict with cx_px, cy_px, bearing, elevation, confidence, compactness
    or None if degenerate."""
    img = np.array(sem_frame, dtype=np.float32)
    if img.ndim == 3:
        heat = img.mean(axis=2) if img.shape[2] in (1, 3) else img.mean(axis=0)
    elif img.ndim == 2:
        heat = img.copy()
    else:
        return None
    if heat.max() > 1.0:
        heat = heat / 255.0

    # V9: percentile-75 threshold — NO min-max normalization
    raw_conf = float(heat.max())
    H, W = heat.shape
    threshold = np.percentile(heat, 75)
    mask = heat > threshold
    if mask.sum() < 5:
        return None
    ys, xs = np.where(mask)
    w = heat[mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))
    bearing   = 2.0 * (cx / W) - 1.0
    elevation = 2.0 * (cy / H) - 1.0
    if len(xs) > 1:
        sp = np.sqrt(np.std(xs / W)**2 + np.std(ys / H)**2)
        compactness = float(np.clip(1.0 - 2.0 * sp, 0, 1))
    else:
        compactness = 1.0
    return dict(cx_px=int(round(cx)), cy_px=int(round(cy)),
                bearing=bearing, elevation=elevation,
                confidence=raw_conf, compactness=compactness)


def _centroid_v11(sem_frame):
    """GT-inspired improved centroid from 2D semantic frame.

    Key differences from V9:
      1. Absolute threshold (clip similarity - 0.5, then > 0.5) instead of percentile
      2. Largest connected component only (spatial clustering)
      3. Confidence = fraction of image above threshold (natural visibility indicator)

    Returns dict with cx_px, cy_px, bearing, elevation, confidence, visible
    or None if target not detected."""
    img = np.array(sem_frame, dtype=np.float32)
    if img.ndim == 3:
        heat = img.mean(axis=2) if img.shape[2] in (1, 3) else img.mean(axis=0)
    elif img.ndim == 2:
        heat = img.copy()
    else:
        return None
    if heat.max() > 1.0:
        heat = heat / 255.0

    H, W = heat.shape

    # Step 1: Absolute threshold — inspired by GT's (similarity - 0.5) > 0.90
    # In rendered 2D frames, raw similarity values are lower, so we use a tuned threshold
    # Scale like GT: shift down and normalize
    shifted = np.clip(heat - 0.5, 0, 1)
    smax = shifted.max()
    if smax < 1e-6:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=0.0, visible=False)
    scaled = shifted / smax

    # Absolute threshold on the scaled similarity (like GT's 0.90)
    abs_threshold = 0.7
    mask = scaled > abs_threshold
    n_above = int(mask.sum())

    # Confidence: how much of the image lights up (high = clear target present)
    confidence = float(n_above) / (H * W)

    # Visibility gate: if too few pixels survive, target likely not in frame
    if n_above < 20:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=confidence, visible=False)

    # Step 2: Largest connected component — spatial clustering like GT's outlier removal
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, centroids_cv = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if n_labels <= 1:  # only background
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=confidence, visible=False)

    # Find largest component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label

    # Step 3: Weighted centroid within the largest blob
    ys, xs = np.where(blob_mask)
    w = heat[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))

    bearing = 2.0 * (cx / W) - 1.0
    elevation = 2.0 * (cy / H) - 1.0

    return dict(cx_px=int(round(cx)), cy_px=int(round(cy)),
                bearing=bearing, elevation=elevation,
                confidence=confidence, visible=True)


def _centroid_v12(sim_raw):
    """Centroid from RAW similarity (single-channel, [0,1] after render_rescale).

    Unlike V9/V11 which operate on a turbo-colormapped RGB image, this uses
    the actual per-pixel CLIP similarity values. Same approach as GT:
      1. Absolute threshold on raw similarity
      2. Largest connected component
      3. Weighted centroid within blob
      4. Confidence = peak similarity (meaningful now, not clamped by colormap)

    Args:
        sim_raw: np.ndarray (H, W) float, raw similarity in [0, 1]

    Returns dict or None.
    """
    if sim_raw is None:
        return None
    heat = np.array(sim_raw, dtype=np.float32)
    if heat.ndim != 2:
        if heat.ndim == 3 and heat.shape[2] == 1:
            heat = heat[:, :, 0]
        else:
            return None

    H, W = heat.shape
    peak = float(heat.max())

    # Absolute threshold: keep pixels > 70% of peak similarity
    # This is analogous to GT's 0.90 threshold on scaled similarity
    threshold = 0.7 * peak
    mask = heat > threshold
    n_above = int(mask.sum())

    # Confidence: peak raw similarity (now meaningful — high = strong CLIP match)
    confidence = peak

    if n_above < 10:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=confidence, visible=False, n_pixels=0)

    # Largest connected component
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=confidence, visible=False, n_pixels=0)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label
    blob_area = int(areas[best_label - 1])

    # Weighted centroid within the largest blob
    ys, xs = np.where(blob_mask)
    w = heat[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))

    bearing = 2.0 * (cx / W) - 1.0
    elevation = 2.0 * (cy / H) - 1.0

    return dict(cx_px=int(round(cx)), cy_px=int(round(cy)),
                bearing=bearing, elevation=elevation,
                confidence=confidence, visible=True,
                n_pixels=blob_area)


def _centroid_from_clipseg(rgb_frame, query, clipseg_model):
    """Run CLIPSeg on an RGB frame and compute centroid from raw logits.

    Uses the same approach as V12 (absolute threshold + largest connected
    component) but on CLIPSeg logits instead of gsplat similarity.

    Args:
        rgb_frame: (H, W, 3) uint8 RGB
        query: text prompt (e.g. "green clock")
        clipseg_model: CLIPSegHFModel instance

    Returns dict with cx_px, cy_px, bearing, elevation, confidence, visible, n_pixels
    """
    from PIL import Image as PILImage
    img = PILImage.fromarray(rgb_frame)
    _, scaled = clipseg_model.clipseg_hf_inference(
        image=img, prompt=query, resize_output_to_input=True,
    )
    # scaled: (H, W) float in [0,1] after running min-max rescale
    heat = np.array(scaled, dtype=np.float32)
    if heat.ndim != 2:
        return None
    H, W = heat.shape
    peak = float(heat.max())

    # Same method as V12: absolute threshold + largest connected component
    threshold = 0.7 * peak
    mask = heat > threshold
    n_above = int(mask.sum())

    if n_above < 10:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=peak, visible=False, n_pixels=0)

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return dict(cx_px=W // 2, cy_px=H // 2, bearing=0.0, elevation=0.0,
                    confidence=peak, visible=False, n_pixels=0)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label
    blob_area = int(areas[best_label - 1])

    ys, xs = np.where(blob_mask)
    w = heat[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))

    bearing = 2.0 * (cx / W) - 1.0
    elevation = 2.0 * (cy / H) - 1.0

    return dict(cx_px=int(round(cx)), cy_px=int(round(cy)),
                bearing=bearing, elevation=elevation,
                confidence=peak, visible=True,
                n_pixels=blob_area)


def _draw_marker(frame, x, y, color, label, sz=14, th=2):
    """Draw crosshair + label. Handles off-screen gracefully."""
    h, w = frame.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, sz, th)
        cv2.putText(frame, label, (x + sz, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _save_video(frames_np, path):
    """Save video exactly like deploy_ssv.py: resize to 720×1280, 20fps."""
    n = frames_np.shape[0]
    out = torch.zeros((n, 720, 1280, 3))
    t = torch.from_numpy(frames_np)
    for i in range(n):
        img = t[i].permute(2, 0, 1)       # (H,W,3) → (3,H,W)
        img = transform(img)                # → (3,720,1280)
        out[i] = img.permute(1, 2, 0)      # → (720,1280,3)
    imageio.mimwrite(path, out.numpy().astype("uint8"), fps=FPS)


COLLISION_RADIUS = 0.15
HORIZONTAL_FOV = np.radians(85)


def _build_plotly(tXUi, Xro, ev, obj_target, epcds_list, pc_tree,
                  radius_info, scene_cfg_file, simulator, val_idx, obj_name):
    """Build canonical Plotly 3D trajectory figure (expert + pilot)."""
    goal_loc = np.asarray(obj_target).flatten()[:3]

    # Base figure with point cloud from canonical bd helper
    fig = bd.visualize_multiple_trajectories(
        [], epcds_list, goal_loc, radius_info, scene_cfg_file, simulator
    )

    # Success zone circle
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=goal_loc[0] + SUCCESS_RADIUS * np.cos(theta),
        y=goal_loc[1] + SUCCESS_RADIUS * np.sin(theta),
        z=np.full(200, goal_loc[2]),
        mode="lines", line=dict(color="green", width=3, dash="dash"),
        name=f"Success Zone (r={SUCCESS_RADIUS}m)",
    ))

    ref_pos = tXUi[1:4, :].T
    pilot_pos = Xro[:3, :].T

    expert_clr = np.array([pc_tree.query(ref_pos[s], k=1)[0] for s in range(len(ref_pos))])
    pilot_clr = np.array([pc_tree.query(pilot_pos[s], k=1)[0] for s in range(len(pilot_pos))])
    clr_max = max(expert_clr.max(), pilot_clr.max(), 0.5)
    clr_scale = [[0, "red"], [COLLISION_RADIUS / clr_max, "orangered"],
                 [0.3 / clr_max, "orange"], [0.5 / clr_max, "yellow"], [1, "green"]]

    # Expert trajectory
    fig.add_trace(go.Scatter3d(
        x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
        mode="lines+markers", line=dict(color="blue", width=5),
        marker=dict(size=3, color=expert_clr, colorscale=clr_scale, cmin=0, cmax=clr_max,
                    colorbar=dict(title="Expert clr", x=1.02, len=0.35, y=0.8, thickness=15)),
        name="Expert (RRT)",
        text=[f"step={s} clr={expert_clr[s]:.3f}m" for s in range(len(ref_pos))],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[ref_pos[0, 0]], y=[ref_pos[0, 1]], z=[ref_pos[0, 2]],
        mode="markers", marker=dict(size=8, color="blue", symbol="diamond"),
        name="Start",
    ))

    # Pilot trajectory
    status = "COLL" if ev["collision"] else ("SUCCESS" if ev["success"] else "MISS")
    status_color = "red" if ev["collision"] else "orange"
    fig.add_trace(go.Scatter3d(
        x=pilot_pos[:, 0], y=pilot_pos[:, 1], z=pilot_pos[:, 2],
        mode="lines+markers", line=dict(color=status_color, width=5),
        marker=dict(size=3, color=pilot_clr, colorscale=clr_scale, cmin=0, cmax=clr_max,
                    colorbar=dict(title="Pilot clr", x=1.02, len=0.35, y=0.35, thickness=15)),
        name=f"DAgger V9 ({status})",
        text=[f"step={s} clr={pilot_clr[s]:.3f}m" for s in range(len(pilot_pos))],
        hovertemplate="%{text}<extra></extra>",
    ))

    # End marker
    end_pos = pilot_pos[-1]
    if ev["collision"]:
        fig.add_trace(go.Scatter3d(
            x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
            mode="markers", marker=dict(size=12, color="red", symbol="x"),
            name=f"Collision @ step {len(pilot_pos)-1}",
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
            mode="markers", marker=dict(size=8, color="orange", symbol="circle-open",
                                        line=dict(width=3, color="orange")),
            name=f"End @ step {len(pilot_pos)-1}",
        ))

    # Deviation lines every 10 steps
    n_common = min(len(pilot_pos), len(ref_pos))
    for s in range(0, n_common, 10):
        fig.add_trace(go.Scatter3d(
            x=[pilot_pos[s, 0], ref_pos[s, 0]],
            y=[pilot_pos[s, 1], ref_pos[s, 1]],
            z=[pilot_pos[s, 2], ref_pos[s, 2]],
            mode="lines", line=dict(color="gray", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    d0 = float(np.linalg.norm(tXUi[1:4, 0] - goal_loc))
    fig.update_layout(
        title=f"<b>'{obj_name}' val={val_idx}</b> — {status}<br>"
              f"<span style='font-size:14px'>d0={d0:.1f}m  goal={ev['goal_dist']:.2f}m  "
              f"min_goal={ev['min_goal_dist']:.2f}m  steps={ev['n_steps']}</span>",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=12)),
    )
    return fig


# ── Main ────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(
        WORKSPACE, "cohorts", DAGGER_COHORT, "visualizations",
        "failed_val", timestamp
    )
    os.makedirs(out_root, exist_ok=True)
    print(f"Output → {out_root}\n")

    # ── Load scene ──
    print("Loading scene (gsplat)...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator  = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries     = scene_data["queries"]
    epcds_arr   = scene_data.get("epcds_arr", np.zeros((0, 3)))
    env_min     = scene_data.get("env_min")
    env_max     = scene_data.get("env_max")

    # For Plotly: get epcds_list + scene config file (not in _get_scene cache)
    import yaml
    scene_cfg_file = os.path.join(SCENES_CFG, f"{SCENE}.yml")
    with open(scene_cfg_file) as f:
        sc = yaml.safe_load(f)
    _, _, epcds_list, _ = bd.get_objectives(
        simulator.gsplat, sc.get("queries", []), sc.get("similarities", None), False
    )
    radius_info = {"r1": sc.get("r1", 1.0), "r2": sc.get("r2", SUCCESS_RADIUS)}

    # ── Load val trajectories ──
    rollout_dir = os.path.join(WORKSPACE, "cohorts", BC_COHORT, "rollout_data", SCENE)
    val_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))
    obj_locs = np.array([np.squeeze(t) for t in obj_targets])
    per_obj = {q: [] for q in queries}
    for tf in val_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best = int(np.argmin(dists))
        if float(dists[best]) < 5.0:
            per_obj[queries[best]].append(tXUd)
    for q in queries:
        print(f"  '{q}': {len(per_obj[q])} val trajectories")

    # ── Build CLIPSeg model ──
    print("Loading CLIPSeg model...")
    from sousvide.flight.vision_preprocess_alternate import CLIPSegHFModel
    clipseg = CLIPSegHFModel(device="cuda")
    print("  CLIPSeg ready.")

    # ── Build pilot ──
    model_path = os.path.join(
        WORKSPACE, "cohorts", DAGGER_COHORT, "roster", PILOT_NAME, "model.pth"
    )
    print(f"\nLoading pilot from {model_path}")
    pilot = Pilot(DAGGER_COHORT, PILOT_NAME)
    pilot.set_mode("deploy")
    pilot.model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    pilot.model.to(DEVICE)
    pilot.model.eval()

    summary = []

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in CASES:
            continue
        obj_target = obj_targets[obj_idx]
        trajs = per_obj.get(obj_name, [])
        if not trajs:
            continue
        pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None
        case_cfg = CASES[obj_name]
        val_indices = case_cfg["failed"] + case_cfg["success"]

        for val_i in val_indices:
            if val_i >= len(trajs):
                print(f"  val={val_i} out of range for '{obj_name}', skip")
                continue
            is_fail = val_i in case_cfg["failed"]
            tXUi = trajs[val_i]
            tag = "FAILED" if is_fail else "SUCCESS"
            short = obj_name.replace(" ", "_")[:30]

            print(f"\n{'='*60}")
            print(f"  [{tag}] '{obj_name[:35]}' val={val_i}")
            print(f"{'='*60}")

            # Reset CLIPSeg running normalization between cases
            clipseg.running_min = float('inf')
            clipseg.running_max = float('-inf')

            # Reset pilot
            pilot.hy_flag = False
            pilot.hy_idx = 0
            pilot.DxU.zero_()
            if hasattr(pilot, "Znn") and isinstance(pilot.Znn, torch.Tensor):
                pilot.Znn.zero_()
            if hasattr(pilot, "chunk_buf"):
                pilot.chunk_buf = None
                pilot.chunk_step = 0

            terminal_fn, _ = _make_terminal_fn(obj_target, pc_tree, env_min, env_max)
            x0 = tXUi[1:11, 0].copy()

            t0_sim = time.time()
            result = simulator.simulate(
                policy=pilot,
                t0=float(tXUi[0, 0]),
                tf=float(tXUi[0, -1]),
                x0=x0,
                obj=np.zeros((18, 1)),
                query=obj_name,
                vision_processor=None,
                verbose=False,
                early_stop_fn=terminal_fn,
            )
            sim_s = time.time() - t0_sim
            Tro, Xro, Uro, Iro = result[0], result[1], result[2], result[3]

            ev = _evaluate_run(
                Xro, obj_target, epcds_arr,
                env_min=env_min, env_max=env_max, tXUi=tXUi, idx0=0,
            )
            status = "✓" if ev["success"] else ("💥" if ev["collision"] else "✗")
            print(f"  {status}  goal={ev['goal_dist']:.2f}m  "
                  f"min_goal={ev['min_goal_dist']:.2f}m@step{ev['min_goal_step']}  "
                  f"steps={ev['n_steps']}  ({sim_s:.1f}s)  "
                  f"fov={ev['fov_pct']:.0%}  coll={ev['collision']}  "
                  f"clearance={ev['min_clearance']:.2f}m")

            channels = list(Iro.keys())
            n_frames = Iro[channels[0]].shape[0]
            print(f"  Iro channels: {channels}  frames={n_frames}")

            # ── Build annotated RGB + centroid log ──
            has_rgb = "rgb" in Iro
            has_sem = "semantic" in Iro
            has_raw = "similarity_raw" in Iro
            centroid_log = []

            # Dead-reckoning state
            last_good = None   # (bearing, elevation, xcr, step)

            if has_rgb:
                rgb_ann = Iro["rgb"].copy()  # (N, H, W, 3)
                for step in range(n_frames):
                    xcr = Xro[:, step + 1]
                    frame = rgb_ann[step]     # (H, W, 3) — will be uint8
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                        rgb_ann[step] = frame

                    # CLIPSeg centroid V9 (current) from colormapped semantic
                    cinfo = None
                    if has_sem:
                        cinfo = _centroid_from_semantic(Iro["semantic"][step])
                    if cinfo:
                        _draw_marker(frame, cinfo["cx_px"], cinfo["cy_px"],
                                     (255, 50, 50), "V9", sz=14, th=2)

                    # V12: centroid from RAW similarity (before colormap)
                    v12 = None
                    if has_raw:
                        v12 = _centroid_v12(Iro["similarity_raw"][step])
                    if v12 and v12.get("visible"):
                        _draw_marker(frame, v12["cx_px"], v12["cy_px"],
                                     (50, 150, 255), "V12", sz=14, th=2)

                    # CLIPSeg centroid (MAGENTA) — sim-to-real transferable
                    cseg = _centroid_from_clipseg(frame.copy(), obj_name, clipseg)
                    if cseg and cseg.get("visible"):
                        _draw_marker(frame, cseg["cx_px"], cseg["cy_px"],
                                     (255, 0, 255), "CSEG", sz=14, th=2)

                    # ── Gated V12 + Dead-Reckoning ("FUSED") ──
                    # Confidence gate: trust V12 only when high conf + compact blob
                    v12_trusted = (v12 is not None
                                   and v12.get("visible")
                                   and v12["confidence"] >= CONF_GATE
                                   and v12.get("n_pixels", 99999) < MAX_PIXELS)

                    fused_u, fused_v = None, None
                    fused_src = "none"  # "v12" or "dr" (dead-reckoning)

                    if v12_trusted:
                        # Direct measurement — high confidence
                        fused_u, fused_v = v12["cx_px"], v12["cy_px"]
                        fused_src = "v12"
                        last_good = (v12["bearing"], v12["elevation"], xcr.copy(), step)
                    elif last_good is not None:
                        # Dead-reckon from last confident observation
                        dr = _deadreckon_bearing(
                            last_good[0], last_good[1], last_good[2], xcr
                        )
                        if dr is not None:
                            _, _, fused_u, fused_v = dr
                            fused_src = f"dr(Δ{step - last_good[3]})"

                    # Draw FUSED marker (YELLOW)
                    if fused_u is not None and fused_v is not None:
                        color = (0, 255, 255) if fused_src == "v12" else (0, 200, 200)
                        label = "FUS" if fused_src == "v12" else "DR"
                        _draw_marker(frame, fused_u, fused_v, color, label, sz=16, th=2)

                    # GT projection
                    gt_px = _project_gt(obj_target, xcr)
                    gt_in = (gt_px is not None
                             and 0 <= gt_px[0] < frame.shape[1]
                             and 0 <= gt_px[1] < frame.shape[0])
                    if gt_px is not None:
                        _draw_marker(frame, gt_px[0], gt_px[1],
                                     (50, 255, 50), "GT", sz=14, th=2)

                    # Compute pixel distances to GT
                    v9_gt_dist = None
                    v12_gt_dist = None
                    fused_gt_dist = None
                    cseg_gt_dist = None
                    if gt_in and gt_px is not None:
                        if cinfo:
                            v9_gt_dist = float(np.hypot(
                                cinfo["cx_px"] - gt_px[0],
                                cinfo["cy_px"] - gt_px[1]))
                        if v12 and v12.get("visible"):
                            v12_gt_dist = float(np.hypot(
                                v12["cx_px"] - gt_px[0],
                                v12["cy_px"] - gt_px[1]))
                        if fused_u is not None:
                            fused_gt_dist = float(np.hypot(
                                fused_u - gt_px[0],
                                fused_v - gt_px[1]))
                        if cseg and cseg.get("visible"):
                            cseg_gt_dist = float(np.hypot(
                                cseg["cx_px"] - gt_px[0],
                                cseg["cy_px"] - gt_px[1]))

                    # HUD text (3 lines)
                    gd_now = float(np.linalg.norm(xcr[:3] - np.squeeze(obj_target)))
                    txt1 = f"step {step}/{n_frames}  goal={gd_now:.1f}m"
                    if v12:
                        txt1 += f"  conf={v12['confidence']:.3f}  npx={v12.get('n_pixels',0)}"
                    txt2 = f"fused={fused_src}"
                    if v9_gt_dist is not None:
                        txt2 += f"  v9={v9_gt_dist:.0f}px"
                    if v12_gt_dist is not None:
                        txt2 += f"  v12={v12_gt_dist:.0f}px"
                    if fused_gt_dist is not None:
                        txt2 += f"  fus={fused_gt_dist:.0f}px"
                    if cseg_gt_dist is not None:
                        txt2 += f"  cseg={cseg_gt_dist:.0f}px"
                    for y_pos, txt in [(18, txt1), (34, txt2)]:
                        if txt:
                            cv2.putText(frame, txt, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2, cv2.LINE_AA)
                            cv2.putText(frame, txt, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

                    centroid_log.append({
                        "step": step,
                        **(cinfo if cinfo else {}),
                        "v12_cx": v12["cx_px"] if v12 else None,
                        "v12_cy": v12["cy_px"] if v12 else None,
                        "v12_visible": v12["visible"] if v12 else None,
                        "v12_confidence": v12["confidence"] if v12 else None,
                        "v12_n_pixels": v12.get("n_pixels") if v12 else None,
                        "fused_u": fused_u, "fused_v": fused_v,
                        "fused_src": fused_src,
                        "cseg_cx": cseg["cx_px"] if cseg else None,
                        "cseg_cy": cseg["cy_px"] if cseg else None,
                        "cseg_visible": cseg["visible"] if cseg else None,
                        "cseg_confidence": cseg["confidence"] if cseg else None,
                        "cseg_n_pixels": cseg.get("n_pixels") if cseg else None,
                        "v9_gt_dist_px": v9_gt_dist,
                        "v12_gt_dist_px": v12_gt_dist,
                        "fused_gt_dist_px": fused_gt_dist,
                        "cseg_gt_dist_px": cseg_gt_dist,
                        "gt_u": gt_px[0] if gt_px else None,
                        "gt_v": gt_px[1] if gt_px else None,
                        "gt_in_frame": gt_in,
                    })

            # ── Save videos into per-case subfolder ──
            case_dir = os.path.join(out_root, f"{short}_val{val_i}_{tag.lower()}")
            os.makedirs(case_dir, exist_ok=True)
            prefix = f"sim_video_{SCENE}_{short}_val{val_i}"
            for ch_name in channels:
                if ch_name == "similarity_raw":
                    continue  # raw float data, not a video
                ch_data = Iro[ch_name]
                # Ensure uint8
                if ch_data.dtype != np.uint8:
                    if ch_data.max() <= 1.0:
                        ch_data = (ch_data * 255).astype(np.uint8)
                    else:
                        ch_data = ch_data.astype(np.uint8)
                vid_path = os.path.join(case_dir, f"{prefix}_{ch_name}.mp4")
                _save_video(ch_data, vid_path)
                print(f"    {vid_path.split('/')[-1]}")

            # Annotated RGB (with crosshairs)
            if has_rgb:
                vid_path = os.path.join(case_dir, f"{prefix}_rgb_annotated.mp4")
                _save_video(rgb_ann, vid_path)
                print(f"    {vid_path.split('/')[-1]}  (with CLIPSeg + GT overlay)")

            # ── Save centroid log ──
            log_path = os.path.join(case_dir, f"{prefix}_centroid_log.json")
            def _jsonable(v):
                import math
                if v is None:
                    return None
                if isinstance(v, dict):
                    return {k: _jsonable(val) for k, val in v.items()}
                if isinstance(v, (list, tuple)):
                    return [_jsonable(x) for x in v]
                if isinstance(v, (np.bool_,)):
                    return bool(v)
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    if math.isnan(v) or math.isinf(v):
                        return None
                    return float(v)
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        return None
                    return v
                if isinstance(v, np.ndarray):
                    return v.tolist()
                return v
            payload = _jsonable({
                "object": obj_name, "val_idx": val_i, "tag": tag,
                "eval": ev,
                "centroids": centroid_log,
            })
            with open(log_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"    {os.path.basename(log_path)}")

            # ── Plotly 3D trajectory ──
            try:
                fig = _build_plotly(
                    tXUi, Xro, ev, obj_target, epcds_list, pc_tree,
                    radius_info, scene_cfg_file, simulator, val_i, obj_name,
                )
                html_path = os.path.join(case_dir, f"{short}_val{val_i}_trajectory.html")
                fig.write_html(html_path, auto_open=False)
                print(f"    {os.path.basename(html_path)}")
            except Exception as e:
                print(f"    ⚠️  Plotly failed: {e}")

            # ── Compute V9 vs V12 vs Fused comparison stats ──
            v9_dists = [c["v9_gt_dist_px"] for c in centroid_log
                        if c.get("v9_gt_dist_px") is not None]
            v12_dists = [c["v12_gt_dist_px"] for c in centroid_log
                         if c.get("v12_gt_dist_px") is not None]
            fused_dists = [c["fused_gt_dist_px"] for c in centroid_log
                           if c.get("fused_gt_dist_px") is not None]
            cseg_dists = [c["cseg_gt_dist_px"] for c in centroid_log
                          if c.get("cseg_gt_dist_px") is not None]
            fused_v12_ct = sum(1 for c in centroid_log if c.get("fused_src") == "v12")
            fused_dr_ct = sum(1 for c in centroid_log if c.get("fused_src", "").startswith("dr"))
            fused_none_ct = sum(1 for c in centroid_log if c.get("fused_src") == "none")
            v9_mean = float(np.mean(v9_dists)) if v9_dists else None
            v12_mean = float(np.mean(v12_dists)) if v12_dists else None
            fused_mean = float(np.mean(fused_dists)) if fused_dists else None
            cseg_mean = float(np.mean(cseg_dists)) if cseg_dists else None

            parts = ["  Accuracy (GT visible): "]
            if v9_mean is not None: parts.append(f"V9={v9_mean:.0f}px")
            if v12_mean is not None: parts.append(f"V12={v12_mean:.0f}px")
            if fused_mean is not None: parts.append(f"Fused={fused_mean:.0f}px")
            if cseg_mean is not None: parts.append(f"CLIPSeg={cseg_mean:.0f}px")
            print("  ".join(parts))
            print(f"  Fused source:  v12={fused_v12_ct}  DR={fused_dr_ct}  none={fused_none_ct}  "
                  f"(GT in frame: {sum(1 for c in centroid_log if c.get('gt_in_frame'))}/{len(centroid_log)})")

            summary.append({
                "object": obj_name, "val_idx": val_i, "tag": tag,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "goal_dist": float(ev["goal_dist"]),
                "gt_visible_pct": (sum(1 for c in centroid_log if c.get("gt_in_frame"))
                                   / max(len(centroid_log), 1)),
                "v9_mean_err_px": v9_mean,
                "v12_mean_err_px": v12_mean,
                "fused_mean_err_px": fused_mean,
                "fused_v12_pct": fused_v12_ct / max(len(centroid_log), 1),
                "fused_dr_pct": fused_dr_ct / max(len(centroid_log), 1),
                "cseg_mean_err_px": cseg_mean,
            })

    # ── Summary ──
    sum_path = os.path.join(out_root, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE — {len(summary)} cases")
    print(f"Output: {out_root}")
    print(f"{'─'*100}")
    print(f"  {'Object':25s} val  tag      success  gt_vis  V9 err  V12 err  Fused err  CSEG err  v12%  DR%")
    print(f"{'─'*100}")
    for s in summary:
        v9e = f"{s['v9_mean_err_px']:.0f}px" if s.get('v9_mean_err_px') else "  -  "
        v12e = f"{s['v12_mean_err_px']:.0f}px" if s.get('v12_mean_err_px') else "  -  "
        fe = f"{s['fused_mean_err_px']:.0f}px" if s.get('fused_mean_err_px') else "  -  "
        ce = f"{s['cseg_mean_err_px']:.0f}px" if s.get('cseg_mean_err_px') else "  -  "
        print(f"  {s['object'][:25]:25s} {s['val_idx']:3d}  {s['tag']:7s}  "
              f"{str(s['success']):5s}    {s['gt_visible_pct']:4.0%}  "
              f"{v9e:>6s}   {v12e:>6s}    {fe:>6s}     {ce:>6s}   "
              f"{s.get('fused_v12_pct', 0):3.0%}  {s.get('fused_dr_pct', 0):3.0%}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
