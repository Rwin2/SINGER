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


# ── Confidence gate + Kalman filter ──

CONF_GATE = 0.90       # minimum V12 confidence to trust centroid
CSEG_CONF_GATE = 0.60  # CLIPSeg sigmoid confidence gate
MAX_PIXELS = 15000     # max blob size — larger = diffuse, unreliable
IMG_W, IMG_H = 640, 360


def _relative_cam_transform(prev_xcr, cur_xcr):
    """Compute relative camera transform: prev_cam → current_cam.

    Uses only:
      - Quaternion (IMU attitude) at both steps
      - Position change between steps (IMU velocity integration)
    Both available on a real drone. No GT object knowledge.

    Returns 4×4 matrix T_rel such that P_cur_cam = T_rel @ P_prev_cam.
    """
    T_c2w_prev = _xv_to_T(prev_xcr) @ T_C2B
    T_c2w_cur = _xv_to_T(cur_xcr) @ T_C2B
    T_rel = np.linalg.inv(T_c2w_cur) @ T_c2w_prev
    return T_rel


def _predict_state(state, prev_xcr, cur_xcr):
    """Predict new [bearing, elevation, inv_depth] after camera motion.

    The object is fixed in world. The camera moves. We:
      1. Unproject state to 3D point in prev camera frame (using inv_depth)
      2. Apply relative camera transform (from IMU)
      3. Reproject to bearing/elevation/inv_depth in current camera frame

    This naturally handles both rotation AND translation parallax.
    When inv_depth ≈ 0 (object very far), translation has no effect
    and this reduces to rotation-only — which is correct.

    Args:
        state: [bearing, elevation, inv_depth] — inv_depth = 1/depth in cam frame
        prev_xcr: drone state at previous step (from IMU)
        cur_xcr: drone state at current step (from IMU)

    Returns: new [bearing, elevation, inv_depth] or None if behind camera.
    """
    b, e, rho = state

    # bearing/elevation → pixel
    u = (b + 1.0) * IMG_W / 2.0
    v = (e + 1.0) * IMG_H / 2.0

    # Pixel → ray in camera frame (OpenGL: forward=-Z, right=+X, up=+Y)
    ray_cam = np.array([
        (u - CX) / FX,
        -(v - CY) / FY,
        -1.0,
    ])

    # Reconstruct 3D point: P = ray * depth / |ray_z|
    # depth = 1/rho,  |ray_z| = 1.0 (by construction, ray_z = -1)
    depth = 1.0 / max(rho, 1e-4)
    pt_prev = ray_cam * depth  # 3D point in prev camera frame

    # Transform to current camera frame
    T_rel = _relative_cam_transform(prev_xcr, cur_xcr)
    pt_cur_h = T_rel @ np.array([*pt_prev, 1.0])
    pt_cur = pt_cur_h[:3]

    # Check if behind camera (OpenGL: forward = -Z)
    if pt_cur[2] >= -1e-6:
        return None

    # Reproject
    depth_new = -pt_cur[2]
    u_new = FX * pt_cur[0] / depth_new + CX
    v_new = FY * (-pt_cur[1]) / depth_new + CY

    b_new = 2.0 * (u_new / IMG_W) - 1.0
    e_new = 2.0 * (v_new / IMG_H) - 1.0
    rho_new = 1.0 / depth_new

    return np.array([b_new, e_new, rho_new])


def _numerical_jacobian(state, prev_xcr, cur_xcr, eps=1e-5):
    """Jacobian of _predict_state w.r.t. state, via finite differences.

    Returns 3×3 matrix F = ∂f/∂x.
    """
    n = len(state)
    f0 = _predict_state(state, prev_xcr, cur_xcr)
    if f0 is None:
        return np.eye(n)  # fallback: identity
    F = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fp = _predict_state(state + dx, prev_xcr, cur_xcr)
        fm = _predict_state(state - dx, prev_xcr, cur_xcr)
        if fp is not None and fm is not None:
            F[:, i] = (fp - fm) / (2 * eps)
        elif fp is not None:
            F[:, i] = (fp - f0) / eps
        else:
            F[i, i] = 1.0  # fallback
    return F


class CentroidKalmanFilter:
    """Bearing-only tracker with inverse-depth estimation.

    State: x = [bearing, elevation, ρ]
      - bearing, elevation: normalized image coords ∈ [-1, 1]
      - ρ = 1/depth: inverse depth in camera frame (meters^-1)

    The object is FIXED in the world. Its apparent position changes
    because the drone moves. We use:
      - CLIPSeg detections → bearing/elevation measurement (when visible)
      - Drone IMU → relative camera motion between frames (always available)

    Inverse-depth parameterization: ρ = 1/depth.
      - Depth unknown at init → large σ_ρ (covers 1m to ∞)
      - As the drone translates and observes from different positions,
        the bearing innovation constrains ρ (triangulation)
      - Once ρ is well-estimated, parallax correction kicks in

    NO GROUND-TRUTH KNOWLEDGE: no obj_target, no goal_dist.
    Only uses CLIPSeg (RGB camera) + IMU (quaternion + velocity).
    Sim-to-real transferable.
    """

    def __init__(self, process_noise_be=0.005, process_noise_rho=0.001,
                 meas_noise=0.03, init_rho=0.3, init_sigma_rho=0.3):
        """
        Args:
            process_noise_be: bearing/elevation process noise per step (~1.6px)
            process_noise_rho: inverse depth process noise (object is static,
                               so ρ changes only due to drone translation —
                               already captured by dynamics model, small Q)
            meas_noise: CLIPSeg centroid measurement noise (~10px at 640px)
            init_rho: initial inverse depth guess. 0.3 ≈ 3.3m.
            init_sigma_rho: initial inverse depth uncertainty.
                            0.3 covers roughly 1.5m to ∞.
        """
        self.x = None          # [bearing, elevation, inv_depth]
        self.P = None          # 3×3 covariance
        self.prev_xcr = None
        self.initialized = False
        self.steps_coasting = 0

        self.Q = np.diag([process_noise_be**2, process_noise_be**2,
                          process_noise_rho**2])
        self.R = np.eye(2) * meas_noise**2
        self.H = np.array([[1, 0, 0], [0, 1, 0]])  # observe bearing, elevation
        self.init_rho = init_rho
        self.init_sigma_rho = init_sigma_rho

    def step(self, xcr, cseg_bearing=None, cseg_elevation=None):
        """Run one EKF step.

        Uses ONLY:
          - xcr: drone IMU state (quaternion + velocity/position for
            relative motion between frames — available on real drone)
          - CLIPSeg centroid bearing/elevation when detected

        The depth (1/ρ) is ESTIMATED by the filter through triangulation:
        observing the object from different positions constrains depth
        via the bearing innovation.
        """
        has_measurement = cseg_bearing is not None

        if has_measurement and not self.initialized:
            self.x = np.array([cseg_bearing, cseg_elevation, self.init_rho])
            self.P = np.diag([self.R[0, 0], self.R[1, 1],
                              self.init_sigma_rho**2])
            self.initialized = True
            self.prev_xcr = xcr.copy()
            self.steps_coasting = 0
            return

        if not self.initialized:
            self.prev_xcr = xcr.copy()
            return

        # ── Predict ──
        if self.prev_xcr is not None:
            x_pred = _predict_state(self.x, self.prev_xcr, xcr)
            if x_pred is not None:
                F = _numerical_jacobian(self.x, self.prev_xcr, xcr)
                self.x = x_pred
                self.P = F @ self.P @ F.T + self.Q
            else:
                # Object behind camera — inflate uncertainty, keep state
                self.P += 10.0 * self.Q
        self.prev_xcr = xcr.copy()

        # Ensure ρ stays positive (physical constraint)
        self.x[2] = max(self.x[2], 1e-4)

        # ── Update ──
        if has_measurement:
            z = np.array([cseg_bearing, cseg_elevation])
            y = z - self.H @ self.x                        # innovation
            S = self.H @ self.P @ self.H.T + self.R        # innovation cov
            K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
            self.x = self.x + K @ y
            self.P = (np.eye(3) - K @ self.H) @ self.P
            self.x[2] = max(self.x[2], 1e-4)               # keep ρ > 0
            self.steps_coasting = 0
        else:
            self.steps_coasting += 1

    def get_estimate(self):
        """Return (bearing, elevation, u_px, v_px, sigma, est_depth) or None."""
        if not self.initialized:
            return None
        u = (self.x[0] + 1.0) * IMG_W / 2.0
        v = (self.x[1] + 1.0) * IMG_H / 2.0
        sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))  # bearing+elev uncertainty
        est_depth = 1.0 / max(self.x[2], 1e-4)
        return self.x[0], self.x[1], int(round(u)), int(round(v)), sigma, est_depth

    def is_confident(self, max_sigma=0.15):
        if not self.initialized:
            return False
        sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
        return sigma < max_sigma


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


def _centroid_from_clipseg(rgb_frame, query, clipseg_model,
                           sigmoid_threshold=0.5, min_blob_pixels=30):
    """Run CLIPSeg on an RGB frame → proper sigmoid-based detection + centroid.

    Unlike the old version which used min-max rescaled output (confidence
    always ~1.0, useless for gating), this works on **raw logits → sigmoid**:

      sigmoid(logit) > 0.5  means "CLIPSeg thinks this pixel is the target"

    Detection pipeline:
      1. Run CLIPSeg → raw logits (H×W float, can be negative)
      2. sigmoid(logits) → per-pixel probability [0, 1]
      3. confidence = max(sigmoid) — how sure is CLIPSeg about its best pixel?
      4. Threshold at `sigmoid_threshold` (default 0.5) → binary mask
      5. Largest connected component → blob
      6. Weighted centroid within blob (weights = sigmoid probabilities)

    The confidence is now **calibrated**: a frame with no target might have
    max(sigmoid) ≈ 0.2, while a clear view gives 0.8+. This lets us gate
    properly: "I'd rather not detect than detect wrong."

    Args:
        rgb_frame: (H, W, 3) uint8 RGB
        query: text prompt (e.g. "green clock")
        clipseg_model: CLIPSegHFModel instance
        sigmoid_threshold: pixels with sigmoid > this are "detected" (default 0.5)
        min_blob_pixels: minimum blob size to count as detection (default 30)

    Returns dict with cx_px, cy_px, bearing, elevation, confidence, n_pixels
            or dict with confidence only (no detection) if target not found.
    """
    from PIL import Image as PILImage
    img = PILImage.fromarray(rgb_frame)
    H_in, W_in = rgb_frame.shape[:2]

    # Run CLIPSeg → raw logits (bypass _rescale_global)
    torch_inputs = clipseg_model.processor(
        images=img, text=query, return_tensors="pt"
    )
    torch_inputs = {k: v.to(clipseg_model.device) for k, v in torch_inputs.items()}
    with torch.no_grad():
        logits = clipseg_model.model(**torch_inputs).logits  # (1, H_out, W_out)
    logits_np = logits.cpu().squeeze().numpy().astype(np.float32)

    # Resize logits to input resolution (CLIPSeg outputs at ~352×352)
    if logits_np.shape != (H_in, W_in):
        logits_np = cv2.resize(logits_np, (W_in, H_in),
                               interpolation=cv2.INTER_LINEAR)

    # Sigmoid → calibrated probability
    prob = 1.0 / (1.0 + np.exp(-logits_np))  # (H, W) in [0, 1]
    confidence = float(prob.max())

    no_detect = dict(cx_px=W_in // 2, cy_px=H_in // 2,
                     bearing=0.0, elevation=0.0,
                     confidence=confidence, n_pixels=0)

    # Threshold on sigmoid probability
    mask = prob > sigmoid_threshold
    n_above = int(mask.sum())
    if n_above < min_blob_pixels:
        return no_detect

    # Largest connected component
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)
    if n_labels <= 1:
        return no_detect

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label
    blob_area = int(areas[best_label - 1])

    if blob_area < min_blob_pixels:
        return no_detect

    # Weighted centroid (weights = sigmoid probability within blob)
    ys, xs = np.where(blob_mask)
    w = prob[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))

    bearing = 2.0 * (cx / W_in) - 1.0
    elevation = 2.0 * (cy / H_in) - 1.0

    return dict(cx_px=int(round(cx)), cy_px=int(round(cy)),
                bearing=bearing, elevation=elevation,
                confidence=confidence, n_pixels=blob_area)


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

            # Kalman filter: CLIPSeg sensor + drone dynamics
            kf = CentroidKalmanFilter()

            # Dynamic GT validation: propagate GT bearing with our dynamics.
            # Uses same _predict_state as KF (with inv_depth).
            # Init from real GT at first visible frame + true depth, then
            # propagate with NO further GT knowledge.
            # If DGT tracks GT → dynamics + inv_depth model is correct.
            dgt_state = None     # [bearing, elevation, inv_depth]
            dgt_prev_xcr = None

            if has_rgb:
                rgb_ann = Iro["rgb"].copy()  # (N, H, W, 3)
                for step in range(n_frames):
                    xcr = Xro[:, step + 1]
                    frame = rgb_ann[step]     # (H, W, 3) — will be uint8
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                        rgb_ann[step] = frame

                    # goal_dist for HUD only (NOT used in KF or dynamics —
                    # obj_target is GT knowledge, not available on real drone)
                    goal_dist = float(np.linalg.norm(
                        xcr[:3] - np.squeeze(obj_target)))

                    # ── CLIPSeg: the only real-world sensor ──
                    cseg = _centroid_from_clipseg(frame.copy(), obj_name, clipseg)
                    cseg_visible = (cseg is not None
                                    and cseg["n_pixels"] > 0
                                    and cseg["confidence"] >= CSEG_CONF_GATE)

                    # ── Kalman filter step (NO GT knowledge) ──
                    if cseg_visible:
                        kf.step(xcr, cseg["bearing"], cseg["elevation"])
                    else:
                        kf.step(xcr)  # predict only

                    kf_est = kf.get_estimate()
                    kf_ok = kf.is_confident()
                    kf_depth = kf_est[5] if kf_est is not None else None

                    # ── GT projection (for evaluation/display only) ──
                    gt_px = _project_gt(obj_target, xcr)
                    gt_in = (gt_px is not None
                             and 0 <= gt_px[0] < frame.shape[1]
                             and 0 <= gt_px[1] < frame.shape[0])

                    # ── Dynamic GT: validate dynamics model ──
                    # Init from GT bearing + true depth (GT used ONLY at init).
                    # Then propagate with _predict_state (same as KF).
                    # DGT error = pure dynamics error (no sensor correction).
                    dgt_px = None
                    if dgt_state is None and gt_px is not None and gt_in:
                        # Compute true depth at init (GT-only, for validation)
                        T_c2w = _xv_to_T(xcr) @ T_C2B
                        T_w2c = np.linalg.inv(T_c2w)
                        pt = T_w2c @ np.array([*np.squeeze(obj_target), 1.0])
                        true_depth = -pt[2]
                        dgt_state = np.array([
                            2.0 * (gt_px[0] / IMG_W) - 1.0,
                            2.0 * (gt_px[1] / IMG_H) - 1.0,
                            1.0 / max(true_depth, 0.1),
                        ])
                        dgt_prev_xcr = xcr.copy()
                        dgt_px = gt_px
                    elif dgt_state is not None and dgt_prev_xcr is not None:
                        s_new = _predict_state(dgt_state, dgt_prev_xcr, xcr)
                        if s_new is not None:
                            dgt_state = s_new
                            u_dgt = (dgt_state[0] + 1.0) * IMG_W / 2.0
                            v_dgt = (dgt_state[1] + 1.0) * IMG_H / 2.0
                            dgt_px = (int(round(u_dgt)), int(round(v_dgt)))
                        dgt_prev_xcr = xcr.copy()

                    # ── Draw: GT (green, always) ──
                    if gt_px is not None:
                        _draw_marker(frame, gt_px[0], gt_px[1],
                                     (50, 255, 50), "GT", sz=14, th=2)

                    # ── Draw: DGT (cyan, dynamics validation) ──
                    dgt_gt_dist = None
                    if dgt_px is not None:
                        _draw_marker(frame, dgt_px[0], dgt_px[1],
                                     (255, 200, 0), "DGT", sz=10, th=1)
                        if gt_in and gt_px is not None:
                            dgt_gt_dist = float(np.hypot(
                                dgt_px[0] - gt_px[0],
                                dgt_px[1] - gt_px[1]))

                    # ── Draw: CSEG (magenta, only when visible) ──
                    if cseg_visible:
                        _draw_marker(frame, cseg["cx_px"], cseg["cy_px"],
                                     (255, 0, 255), "CSEG", sz=14, th=2)

                    # ── Draw: KF (yellow, from first sighting onward) ──
                    fused_u, fused_v = None, None
                    kf_sigma = None
                    if kf_est is not None and kf_ok:
                        _, _, fused_u, fused_v, kf_sigma, _ = kf_est
                        if cseg_visible:
                            _draw_marker(frame, fused_u, fused_v,
                                         (0, 255, 255), "KF", sz=16, th=2)
                        else:
                            _draw_marker(frame, fused_u, fused_v,
                                         (0, 180, 180),
                                         f"KF~{kf.steps_coasting}", sz=12, th=1)
                    elif kf_est is not None and not kf_ok:
                        _, _, _, _, kf_sigma, _ = kf_est
                        cv2.putText(frame, f"KF lost (s={kf_sigma:.3f})",
                                    (frame.shape[1] - 180, 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 130, 130), 1, cv2.LINE_AA)

                    # ── Pixel errors vs GT ──
                    cseg_gt_dist = None
                    kf_gt_dist = None
                    if gt_in and gt_px is not None:
                        if cseg_visible:
                            cseg_gt_dist = float(np.hypot(
                                cseg["cx_px"] - gt_px[0],
                                cseg["cy_px"] - gt_px[1]))
                        if fused_u is not None:
                            kf_gt_dist = float(np.hypot(
                                fused_u - gt_px[0], fused_v - gt_px[1]))

                    # ── HUD ──
                    txt1 = f"step {step}/{n_frames}  goal={goal_dist:.1f}m"
                    if cseg:
                        txt1 += f"  cseg_conf={cseg['confidence']:.2f}"
                    txt1 += f"  {'VISIBLE' if cseg_visible else 'occluded'}"

                    txt2 = ""
                    if kf_est is not None:
                        mode = "meas" if cseg_visible and kf.initialized else f"pred(Δ{kf.steps_coasting})"
                        depth_str = f"  d̂={kf_depth:.1f}m" if kf_depth is not None else ""
                        txt2 = f"KF:{mode}  σ={kf_sigma:.4f}{depth_str}"
                    if dgt_gt_dist is not None:
                        txt2 += f"  DGT_err={dgt_gt_dist:.0f}px"
                    parts = []
                    if cseg_gt_dist is not None: parts.append(f"cseg={cseg_gt_dist:.0f}px")
                    if kf_gt_dist is not None: parts.append(f"kf={kf_gt_dist:.0f}px")
                    txt3 = "err: " + "  ".join(parts) if parts else ""

                    for y_pos, txt in [(18, txt1), (34, txt2), (50, txt3)]:
                        if txt:
                            cv2.putText(frame, txt, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                        (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, txt, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                        (255, 255, 255), 1, cv2.LINE_AA)

                    # ── Log ──
                    centroid_log.append({
                        "step": step,
                        "cseg_cx": cseg["cx_px"] if cseg else None,
                        "cseg_cy": cseg["cy_px"] if cseg else None,
                        "cseg_confidence": cseg["confidence"] if cseg else None,
                        "cseg_visible": cseg_visible,
                        "kf_u": fused_u, "kf_v": fused_v,
                        "kf_sigma": kf_sigma,
                        "kf_confident": kf_ok,
                        "kf_coasting": kf.steps_coasting if kf.initialized else None,
                        "cseg_gt_dist_px": cseg_gt_dist,
                        "kf_gt_dist_px": kf_gt_dist,
                        "dgt_gt_dist_px": dgt_gt_dist,
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

            # ── Compute CSEG vs KF vs DGT comparison stats ──
            cseg_dists = [c["cseg_gt_dist_px"] for c in centroid_log
                          if c.get("cseg_gt_dist_px") is not None]
            kf_dists = [c["kf_gt_dist_px"] for c in centroid_log
                        if c.get("kf_gt_dist_px") is not None]
            dgt_dists = [c["dgt_gt_dist_px"] for c in centroid_log
                         if c.get("dgt_gt_dist_px") is not None]
            kf_meas_ct = sum(1 for c in centroid_log if c.get("cseg_visible"))
            kf_pred_ct = sum(1 for c in centroid_log
                             if not c.get("cseg_visible") and c.get("kf_confident"))
            kf_unc_ct = sum(1 for c in centroid_log
                            if not c.get("kf_confident", False)
                            and c.get("kf_coasting") is not None)
            cseg_mean = float(np.mean(cseg_dists)) if cseg_dists else None
            kf_mean = float(np.mean(kf_dists)) if kf_dists else None
            dgt_mean = float(np.mean(dgt_dists)) if dgt_dists else None
            gt_in_ct = sum(1 for c in centroid_log if c.get("gt_in_frame"))

            parts = ["  Accuracy (GT visible): "]
            if cseg_mean is not None: parts.append(f"CSEG={cseg_mean:.0f}px")
            if kf_mean is not None: parts.append(f"KF={kf_mean:.0f}px")
            if dgt_mean is not None: parts.append(f"DGT={dgt_mean:.0f}px")
            print("  ".join(parts))
            print(f"  KF:  measured={kf_meas_ct}  predicted={kf_pred_ct}  "
                  f"uncertain={kf_unc_ct}  "
                  f"(GT in frame: {gt_in_ct}/{len(centroid_log)})")

            summary.append({
                "object": obj_name, "val_idx": val_i, "tag": tag,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "goal_dist": float(ev["goal_dist"]),
                "gt_visible_pct": gt_in_ct / max(len(centroid_log), 1),
                "cseg_mean_err_px": cseg_mean,
                "kf_mean_err_px": kf_mean,
                "dgt_mean_err_px": dgt_mean,
                "kf_meas_pct": kf_meas_ct / max(len(centroid_log), 1),
                "kf_pred_pct": kf_pred_ct / max(len(centroid_log), 1),
            })

    # ── Summary ──
    sum_path = os.path.join(out_root, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE — {len(summary)} cases")
    print(f"Output: {out_root}")
    print(f"{'─'*100}")
    print(f"  {'Object':25s} val  tag      success  gt_vis  CSEG err  KF err  DGT err  kf:meas  kf:pred")
    print(f"{'─'*100}")
    for s in summary:
        ce = f"{s['cseg_mean_err_px']:.0f}px" if s.get('cseg_mean_err_px') else "  -  "
        ke = f"{s['kf_mean_err_px']:.0f}px" if s.get('kf_mean_err_px') else "  -  "
        de = f"{s['dgt_mean_err_px']:.0f}px" if s.get('dgt_mean_err_px') else "  -  "
        print(f"  {s['object'][:25]:25s} {s['val_idx']:3d}  {s['tag']:7s}  "
              f"{str(s['success']):5s}    {s['gt_visible_pct']:4.0%}  "
              f"{ce:>7s}   {ke:>6s}   {de:>6s}   "
              f"{s.get('kf_meas_pct', 0):5.0%}   {s.get('kf_pred_pct', 0):5.0%}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
