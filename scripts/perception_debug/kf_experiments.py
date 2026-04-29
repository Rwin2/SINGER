#!/usr/bin/env python3
"""
Offline KF experiments on saved centroid logs.

Reads the centroid_log.json files from a previous eval run,
then replays different KF configurations WITHOUT re-running CLIPSeg
or the simulator. This allows rapid iteration (~1 second per config).

The centroid logs contain per-frame:
  - cseg_cx, cseg_cy, cseg_confidence (CLIPSeg raw sigmoid output)
  - gt_u, gt_v, gt_in_frame (ground truth — for evaluation only)
  - drone state is NOT in logs → we need to load Xro from trajectory data

Strategy: load trajectory data (Xro) + centroid logs, replay KF variants,
compare to GT.

Usage:
    cd /data/erwinpi/SINGER
    conda run -n FiGS python scripts/perception_debug/kf_experiments.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

# Camera intrinsics (from carl.json)
FX, FY = 462.956, 463.002
CX, CY = 323.076, 181.184
IMG_W, IMG_H = 640, 360

T_C2B = np.array([
    [ 0.0,  0.0, -1.0,  0.10],
    [ 1.0,  0.0,  0.0, -0.03],
    [ 0.0, -1.0,  0.0, -0.01],
    [ 0.0,  0.0,  0.0,  1.00],
])


# ── Geometry helpers ──

def _xv_to_T(xv):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(xv[6:10]).as_matrix()
    T[:3, 3] = xv[:3]
    return T


def _relative_cam_transform(prev_xcr, cur_xcr):
    T_c2w_prev = _xv_to_T(prev_xcr) @ T_C2B
    T_c2w_cur = _xv_to_T(cur_xcr) @ T_C2B
    return np.linalg.inv(T_c2w_cur) @ T_c2w_prev


def predict_state(state, prev_xcr, cur_xcr):
    """Predict [bearing, elevation, inv_depth] after camera motion."""
    b, e, rho = state
    u = (b + 1.0) * IMG_W / 2.0
    v = (e + 1.0) * IMG_H / 2.0
    ray_cam = np.array([(u - CX) / FX, -(v - CY) / FY, -1.0])
    depth = 1.0 / max(rho, 1e-4)
    pt_prev = ray_cam * depth
    T_rel = _relative_cam_transform(prev_xcr, cur_xcr)
    pt_cur_h = T_rel @ np.array([*pt_prev, 1.0])
    pt_cur = pt_cur_h[:3]
    if pt_cur[2] >= -1e-6:
        return None
    depth_new = -pt_cur[2]
    u_new = FX * pt_cur[0] / depth_new + CX
    v_new = FY * (-pt_cur[1]) / depth_new + CY
    b_new = 2.0 * (u_new / IMG_W) - 1.0
    e_new = 2.0 * (v_new / IMG_H) - 1.0
    rho_new = 1.0 / depth_new
    return np.array([b_new, e_new, rho_new])


def numerical_jacobian(state, prev_xcr, cur_xcr, eps=1e-5):
    n = len(state)
    f0 = predict_state(state, prev_xcr, cur_xcr)
    if f0 is None:
        return np.eye(n)
    F = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fp = predict_state(state + dx, prev_xcr, cur_xcr)
        fm = predict_state(state - dx, prev_xcr, cur_xcr)
        if fp is not None and fm is not None:
            F[:, i] = (fp - fm) / (2 * eps)
        elif fp is not None:
            F[:, i] = (fp - f0) / eps
        else:
            F[i, i] = 1.0
    return F


# ── KF Variants ──

class InvDepthEKF:
    """Baseline inverse-depth EKF (same as current eval script)."""

    def __init__(self, conf_gate=0.60, process_noise_be=0.005,
                 process_noise_rho=0.001, meas_noise=0.03,
                 init_rho=0.3, init_sigma_rho=0.3,
                 innovation_gate=None, adaptive_R=False,
                 temporal_decay_gate=False):
        self.conf_gate = conf_gate
        self.x = None
        self.P = None
        self.prev_xcr = None
        self.initialized = False
        self.steps_coasting = 0
        self.Q = np.diag([process_noise_be**2, process_noise_be**2,
                          process_noise_rho**2])
        self.R_base = np.eye(2) * meas_noise**2
        self.H = np.array([[1, 0, 0], [0, 1, 0]])
        self.init_rho = init_rho
        self.init_sigma_rho = init_sigma_rho
        self.innovation_gate = innovation_gate  # Mahalanobis threshold (e.g. 3.0)
        self.adaptive_R = adaptive_R  # scale R by 1/confidence
        self.temporal_decay_gate = temporal_decay_gate  # lower threshold after first detection
        self._ever_seen = False
        self.name = "InvDepthEKF"

    def _get_conf_gate(self):
        """Possibly lower the gate after first confident detection."""
        if self.temporal_decay_gate and self._ever_seen:
            return max(self.conf_gate - 0.15, 0.3)  # lower by 0.15 after first sighting
        return self.conf_gate

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
        """Process one frame. Returns (u_px, v_px, sigma) or None."""
        gate = self._get_conf_gate()
        has_meas = (cseg_n_pixels > 0 and cseg_confidence >= gate)

        if has_meas and not self.initialized:
            self.x = np.array([cseg_bearing, cseg_elevation, self.init_rho])
            self.P = np.diag([self.R_base[0, 0], self.R_base[1, 1],
                              self.init_sigma_rho**2])
            self.initialized = True
            self.prev_xcr = xcr.copy()
            self.steps_coasting = 0
            self._ever_seen = True
            u = (self.x[0] + 1.0) * IMG_W / 2.0
            v = (self.x[1] + 1.0) * IMG_H / 2.0
            sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
            return int(round(u)), int(round(v)), sigma

        if not self.initialized:
            self.prev_xcr = xcr.copy()
            return None

        # Predict
        if self.prev_xcr is not None:
            x_pred = predict_state(self.x, self.prev_xcr, xcr)
            if x_pred is not None:
                F = numerical_jacobian(self.x, self.prev_xcr, xcr)
                self.x = x_pred
                self.P = F @ self.P @ F.T + self.Q
            else:
                self.P += 10.0 * self.Q
        self.prev_xcr = xcr.copy()
        self.x[2] = max(self.x[2], 1e-4)

        # Update
        if has_meas:
            z = np.array([cseg_bearing, cseg_elevation])
            y = z - self.H @ self.x  # innovation

            # Innovation gating: reject if Mahalanobis distance too large
            if self.innovation_gate is not None:
                S = self.H @ self.P @ self.H.T + self.R_base
                maha = float(y @ np.linalg.inv(S) @ y)
                if maha > self.innovation_gate**2:
                    has_meas = False  # reject this measurement

        if has_meas:
            # Adaptive R: scale measurement noise by inverse confidence
            if self.adaptive_R and cseg_confidence > 0.01:
                # Higher confidence → lower R. At conf=1.0: R=R_base. At conf=0.5: R=2*R_base
                scale = 1.0 / max(cseg_confidence, 0.3)
                R = self.R_base * scale
            else:
                R = self.R_base

            z = np.array([cseg_bearing, cseg_elevation])
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(3) - K @ self.H) @ self.P
            self.x[2] = max(self.x[2], 1e-4)
            self.steps_coasting = 0
            self._ever_seen = True
        else:
            self.steps_coasting += 1

        u = (self.x[0] + 1.0) * IMG_W / 2.0
        v = (self.x[1] + 1.0) * IMG_H / 2.0
        sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
        return int(round(u)), int(round(v)), sigma


# ── Data Loading ──

def load_case_data(eval_dir, case_dir_name):
    """Load centroid log + trajectory data for one case."""
    case_path = os.path.join(eval_dir, case_dir_name)
    log_files = glob.glob(os.path.join(case_path, "*_centroid_log.json"))
    if not log_files:
        return None
    with open(log_files[0]) as f:
        log_data = json.load(f)
    return log_data


def load_trajectory_data(cohort_dir, scene, obj_name, val_idx):
    """Load Xro (drone states) from trajectory data."""
    sim_dirs = sorted(glob.glob(os.path.join(
        cohort_dir, "simulation_data", "*")))
    for sim_dir in reversed(sim_dirs):  # most recent first
        traj_files = glob.glob(os.path.join(sim_dir, "trajectories_*.pt"))
        for tf in traj_files:
            try:
                data = torch.load(tf, map_location="cpu", weights_only=False)
                for entry in data:
                    if (entry.get("object") == obj_name and
                        entry.get("val_idx") == val_idx):
                        return entry
            except:
                continue
    return None


# ── Main Experiment Runner ──

def run_experiment(centroid_log, Xro, kf_instance, obj_target=None):
    """Run a KF variant on one trajectory's centroid log + drone states.

    Returns dict with per-frame errors and summary stats.
    """
    frames = centroid_log
    n = len(frames)
    results = []

    for i, c in enumerate(frames):
        xcr = Xro[:, i + 1]

        # CLIPSeg data
        cseg_conf = c.get("cseg_confidence", 0.0) or 0.0
        cseg_n_px = c.get("cseg_n_pixels", 0)
        if cseg_n_px is None:
            cseg_n_px = 0

        # Compute bearing/elevation from CSEG pixel coords
        cseg_cx = c.get("cseg_cx")
        cseg_cy = c.get("cseg_cy")
        if cseg_cx is not None and cseg_cy is not None and cseg_n_px > 0:
            cseg_bearing = 2.0 * (cseg_cx / IMG_W) - 1.0
            cseg_elevation = 2.0 * (cseg_cy / IMG_H) - 1.0
        else:
            cseg_bearing = 0.0
            cseg_elevation = 0.0
            cseg_n_px = 0

        # Run KF
        est = kf_instance.step(xcr, cseg_bearing, cseg_elevation,
                               cseg_conf, cseg_n_px)

        # GT data
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # Compute errors
        kf_err = None
        cseg_err = None
        if gt_in and gt_u is not None and gt_v is not None:
            if est is not None:
                kf_err = float(np.hypot(est[0] - gt_u, est[1] - gt_v))
            if cseg_n_px > 0 and cseg_cx is not None:
                cseg_err = float(np.hypot(cseg_cx - gt_u, cseg_cy - gt_v))

        results.append({
            "step": i,
            "kf_u": est[0] if est else None,
            "kf_v": est[1] if est else None,
            "kf_sigma": est[2] if est else None,
            "kf_err_px": kf_err,
            "cseg_err_px": cseg_err,
            "cseg_conf": cseg_conf,
            "cseg_visible": c.get("cseg_visible", False),
            "gt_in": gt_in,
            "coasting": kf_instance.steps_coasting,
        })

    # Summary
    kf_errs = [r["kf_err_px"] for r in results if r["kf_err_px"] is not None]
    cseg_errs = [r["cseg_err_px"] for r in results if r["cseg_err_px"] is not None]
    n_meas = sum(1 for r in results if r.get("cseg_visible"))

    summary = {
        "n_frames": n,
        "n_gt_visible": sum(1 for r in results if r["gt_in"]),
        "n_cseg_detected": n_meas,
        "n_kf_active": sum(1 for r in results if r["kf_u"] is not None),
        "kf_mean": float(np.mean(kf_errs)) if kf_errs else None,
        "kf_median": float(np.median(kf_errs)) if kf_errs else None,
        "kf_p75": float(np.percentile(kf_errs, 75)) if kf_errs else None,
        "kf_p90": float(np.percentile(kf_errs, 90)) if kf_errs else None,
        "kf_max": float(np.max(kf_errs)) if kf_errs else None,
        "cseg_mean": float(np.mean(cseg_errs)) if cseg_errs else None,
        "cseg_median": float(np.median(cseg_errs)) if cseg_errs else None,
    }
    return {"frames": results, "summary": summary}


def find_latest_eval_dir():
    """Find the most recent eval output directory."""
    base = os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_CENTROID_V9",
                        "visualizations", "failed_val")
    dirs = sorted(glob.glob(os.path.join(base, "20*")))
    return dirs[-1] if dirs else None


def main():
    import torch

    eval_dir = find_latest_eval_dir()
    if eval_dir is None:
        print("ERROR: No eval directory found")
        return
    print(f"Using eval data from: {eval_dir}")

    # Find all case directories
    case_dirs = sorted([d for d in os.listdir(eval_dir)
                        if os.path.isdir(os.path.join(eval_dir, d))])
    print(f"Found {len(case_dirs)} cases: {case_dirs}")

    # Load trajectory data for drone states
    cohort_dir = os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_CENTROID_V9")

    # We need Xro for each case. Load from the simulation that produced the eval.
    # The eval script saves Xro in the trajectory, let's check the centroid log
    # for whether it has enough info, otherwise we need to re-load.

    # First, check if centroid logs have cseg_bearing/elevation or just pixels
    sample_log = load_case_data(eval_dir, case_dirs[0])
    if sample_log is None:
        print("ERROR: Could not load sample log")
        return

    # Check if we have bearing info
    sample_frame = sample_log["centroids"][0]
    has_bearing = "cseg_bearing" in sample_frame
    print(f"Log has bearing data: {has_bearing}")
    print(f"Sample frame keys: {list(sample_frame.keys())}")

    # We need Xro (drone states). These are NOT in the centroid logs.
    # We need to load them from the trajectory files.
    # The eval script loads them via _get_scene + simulate.
    # For offline replay, we need the saved trajectory data.

    # Check if trajectory data is saved alongside the eval
    traj_files = glob.glob(os.path.join(eval_dir, "**", "*.pt"), recursive=True)
    print(f"Found {len(traj_files)} .pt files in eval dir")

    # The trajectory data is in the simulation_data dir
    sim_dirs = sorted(glob.glob(os.path.join(cohort_dir, "simulation_data", "*")))
    print(f"Found {len(sim_dirs)} simulation dirs")

    # Actually, the eval script simulates trajectories live — the Xro data
    # is only in memory during the run. We need to either:
    # 1. Save Xro alongside centroid logs (modify eval script)
    # 2. Re-simulate to get Xro (slow)
    # 3. Use validation trajectory data from the cohort

    # Let's check validation data
    val_dirs = glob.glob(os.path.join(cohort_dir, "rollout_data", "*",
                                       "validation", "trajectories_*.pt"))
    print(f"Found {len(val_dirs)} validation trajectory files")

    # For now, we need to modify the eval script to ALSO save Xro.
    # Let me check if we can extract what we need differently.

    print("\n" + "="*60)
    print("NEED TO SAVE Xro IN EVAL SCRIPT FIRST")
    print("The centroid logs don't contain drone state (xcr).")
    print("We need to modify eval to save Xro alongside centroid logs,")
    print("then we can do rapid offline KF experiments.")
    print("="*60)


if __name__ == "__main__":
    main()
