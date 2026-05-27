#!/usr/bin/env python3
"""
Offline KF experiments on saved centroid logs + Xro trajectory data.

Reads centroid_log.json + trajectory_data.npz from eval runs,
replays different KF configurations WITHOUT re-running CLIPSeg.
~1 second per config per case.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/kf_experiments.py
"""
import os, sys, json, glob, time, itertools
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
    """Inverse-depth EKF with configurable options."""

    def __init__(self, conf_gate=0.60, process_noise_be=0.005,
                 process_noise_rho=0.001, meas_noise=0.03,
                 init_rho=0.3, init_sigma_rho=0.3,
                 innovation_gate=None, adaptive_R=False,
                 temporal_decay_gate=False, name=None):
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
        self.innovation_gate = innovation_gate
        self.adaptive_R = adaptive_R
        self.temporal_decay_gate = temporal_decay_gate
        self._ever_seen = False
        self.n_rejected = 0
        self.n_measured = 0
        self.name = name or "InvDepthEKF"

    def reset(self):
        self.x = None
        self.P = None
        self.prev_xcr = None
        self.initialized = False
        self.steps_coasting = 0
        self._ever_seen = False
        self.n_rejected = 0
        self.n_measured = 0

    def _get_conf_gate(self):
        if self.temporal_decay_gate and self._ever_seen:
            return max(self.conf_gate - 0.15, 0.3)
        return self.conf_gate

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
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
            self.n_measured += 1
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
            y = z - self.H @ self.x

            if self.innovation_gate is not None:
                S = self.H @ self.P @ self.H.T + self.R_base
                maha = float(y @ np.linalg.inv(S) @ y)
                if maha > self.innovation_gate**2:
                    has_meas = False
                    self.n_rejected += 1

        if has_meas:
            if self.adaptive_R and cseg_confidence > 0.01:
                scale = 1.0 / max(cseg_confidence, 0.3)**2
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
            self.n_measured += 1
        else:
            self.steps_coasting += 1

        u = (self.x[0] + 1.0) * IMG_W / 2.0
        v = (self.x[1] + 1.0) * IMG_H / 2.0
        sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
        return int(round(u)), int(round(v)), sigma


class ISEKF(InvDepthEKF):
    """Innovation-Saturated EKF: soft-clip innovation instead of hard reject.

    Reference: Aravkin et al., "Robust and scalable Bayes via a median of
    subset posteriors" (adapted concept). Instead of hard gating, scale
    the innovation down when Mahalanobis distance exceeds threshold.
    """

    def __init__(self, saturation_threshold=2.5, **kwargs):
        kwargs.pop("innovation_gate", None)  # don't use hard gate
        super().__init__(innovation_gate=None, **kwargs)
        self.saturation_threshold = saturation_threshold
        self.name = kwargs.get("name", f"ISEKF(sat={saturation_threshold})")

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
        gate = self._get_conf_gate()
        has_meas = (cseg_n_pixels > 0 and cseg_confidence >= gate)

        if has_meas and not self.initialized:
            return super().step(xcr, cseg_bearing, cseg_elevation,
                                cseg_confidence, cseg_n_pixels)

        if not self.initialized:
            self.prev_xcr = xcr.copy()
            return None

        # Predict (same as parent)
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

        if has_meas:
            if self.adaptive_R and cseg_confidence > 0.01:
                scale = 1.0 / max(cseg_confidence, 0.3)**2
                R = self.R_base * scale
            else:
                R = self.R_base

            z = np.array([cseg_bearing, cseg_elevation])
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + R

            # Compute Mahalanobis distance
            maha = float(np.sqrt(max(y @ np.linalg.inv(S) @ y, 0)))

            # Soft saturation: scale innovation down if maha > threshold
            if maha > self.saturation_threshold and maha > 1e-6:
                scale_factor = self.saturation_threshold / maha
                y = y * scale_factor
                # Also inflate R to reduce Kalman gain
                R = R / (scale_factor**2)
                S = self.H @ self.P @ self.H.T + R
                self.n_rejected += 1  # count as "soft-rejected"

            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(3) - K @ self.H) @ self.P
            self.x[2] = max(self.x[2], 1e-4)
            self.steps_coasting = 0
            self._ever_seen = True
            self.n_measured += 1
        else:
            self.steps_coasting += 1

        u = (self.x[0] + 1.0) * IMG_W / 2.0
        v = (self.x[1] + 1.0) * IMG_H / 2.0
        sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
        return int(round(u)), int(round(v)), sigma


class EMASmoothedEKF(InvDepthEKF):
    """EKF that first applies EMA smoothing to CLIPSeg measurements.

    Smooths bearing/elevation across frames before feeding to EKF.
    This reduces jitter from noisy CLIPSeg detections.
    """

    def __init__(self, ema_alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.ema_alpha = ema_alpha
        self.ema_bearing = None
        self.ema_elevation = None
        self.ema_conf = None
        self.name = kwargs.get("name", f"EMA(α={ema_alpha})")

    def reset(self):
        super().reset()
        self.ema_bearing = None
        self.ema_elevation = None
        self.ema_conf = None

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
        gate = self._get_conf_gate()
        raw_has_meas = (cseg_n_pixels > 0 and cseg_confidence >= gate)

        if raw_has_meas:
            if self.ema_bearing is None:
                self.ema_bearing = cseg_bearing
                self.ema_elevation = cseg_elevation
                self.ema_conf = cseg_confidence
            else:
                a = self.ema_alpha
                self.ema_bearing = a * cseg_bearing + (1 - a) * self.ema_bearing
                self.ema_elevation = a * cseg_elevation + (1 - a) * self.ema_elevation
                self.ema_conf = a * cseg_confidence + (1 - a) * self.ema_conf
            # Feed smoothed values to parent
            return super().step(xcr, self.ema_bearing, self.ema_elevation,
                                self.ema_conf, cseg_n_pixels)
        else:
            # No detection — just coast (pass through to parent with no detection)
            return super().step(xcr, 0.0, 0.0, 0.0, 0)


class CoastingLimitEKF(InvDepthEKF):
    """EKF that resets after too many coasting frames.

    If the target has been lost for too long, reset and wait for
    re-detection rather than drifting far from reality.
    """

    def __init__(self, max_coast_frames=30, **kwargs):
        super().__init__(**kwargs)
        self.max_coast_frames = max_coast_frames
        self.name = kwargs.get("name", f"CoastLimit({max_coast_frames})")

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
        if self.initialized and self.steps_coasting >= self.max_coast_frames:
            # Reset — we've lost the target for too long
            self.initialized = False
            self.x = None
            self.P = None
            self.steps_coasting = 0
        return super().step(xcr, cseg_bearing, cseg_elevation,
                            cseg_confidence, cseg_n_pixels)


class ConsensusInitEKF(InvDepthEKF):
    """EKF that requires N consistent detections before initializing.

    Prevents spurious first detections (like drill_val7 where CLIPSeg
    confidently detects the wrong object). Requires that the first N
    detections are within `init_radius_px` of each other.
    """

    def __init__(self, n_init=3, init_radius_px=50.0,
                 max_coast_frames=20, **kwargs):
        super().__init__(**kwargs)
        self.n_init = n_init
        self.init_radius_px = init_radius_px
        self.max_coast_frames = max_coast_frames
        self._init_buffer = []  # [(bearing, elevation)]
        self.name = kwargs.get("name",
                               f"Consensus(n={n_init},r={init_radius_px:.0f})")

    def reset(self):
        super().reset()
        self._init_buffer = []

    def step(self, xcr, cseg_bearing, cseg_elevation, cseg_confidence, cseg_n_pixels):
        # Coasting limit: reset if lost for too long
        if self.initialized and self.steps_coasting >= self.max_coast_frames:
            self.initialized = False
            self.x = None
            self.P = None
            self.steps_coasting = 0
            self._init_buffer = []

        gate = self._get_conf_gate()
        has_meas = (cseg_n_pixels > 0 and cseg_confidence >= gate)

        if has_meas and not self.initialized:
            # Add to init buffer
            self._init_buffer.append((cseg_bearing, cseg_elevation))

            # Check if we have enough consistent detections
            if len(self._init_buffer) >= self.n_init:
                # Check consistency: all detections within radius
                bearings = [b for b, e in self._init_buffer[-self.n_init:]]
                elevations = [e for b, e in self._init_buffer[-self.n_init:]]
                b_range = (max(bearings) - min(bearings)) * IMG_W / 2.0
                e_range = (max(elevations) - min(elevations)) * IMG_H / 2.0
                spread = np.hypot(b_range, e_range)

                if spread <= self.init_radius_px:
                    # Consistent! Initialize with average
                    avg_b = np.mean(bearings)
                    avg_e = np.mean(elevations)
                    self.x = np.array([avg_b, avg_e, self.init_rho])
                    self.P = np.diag([self.R_base[0, 0], self.R_base[1, 1],
                                      self.init_sigma_rho**2])
                    self.initialized = True
                    self.prev_xcr = xcr.copy()
                    self.steps_coasting = 0
                    self._ever_seen = True
                    self.n_measured += 1
                    u = (self.x[0] + 1.0) * IMG_W / 2.0
                    v = (self.x[1] + 1.0) * IMG_H / 2.0
                    sigma = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
                    return int(round(u)), int(round(v)), sigma
                else:
                    # Inconsistent: drop oldest detection
                    self._init_buffer = self._init_buffer[-(self.n_init - 1):]

            self.prev_xcr = xcr.copy()
            return None

        if not self.initialized:
            # Decay init buffer if no detection
            if not has_meas and self._init_buffer:
                # Forget oldest after some frames without detection
                pass  # keep buffer, it'll get checked on next detection
            self.prev_xcr = xcr.copy()
            return None

        # Normal EKF predict + update (from parent)
        return super().step(xcr, cseg_bearing, cseg_elevation,
                            cseg_confidence, cseg_n_pixels)


# ── Data Loading ──

def find_eval_dir_with_npz():
    """Find the most recent eval dir that has trajectory_data.npz files."""
    base = os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_CENTROID_V9",
                        "visualizations", "failed_val")
    dirs = sorted(glob.glob(os.path.join(base, "20*")))
    for d in reversed(dirs):
        npz_files = glob.glob(os.path.join(d, "**", "*_trajectory_data.npz"),
                              recursive=True)
        if npz_files:
            return d
    return None


def load_all_cases(eval_dir):
    """Load all cases from an eval directory.

    Returns list of dicts with keys:
      - name: case directory name (e.g. "green_clock_val4_failed")
      - centroids: list of per-frame centroid data
      - Xro: (10, N) drone state array
      - obj_target: (3,) GT object position (for validation only)
      - object: object name
      - tag: SUCCESS/FAILED
    """
    cases = []
    case_dirs = sorted([d for d in os.listdir(eval_dir)
                        if os.path.isdir(os.path.join(eval_dir, d))])

    for case_name in case_dirs:
        case_path = os.path.join(eval_dir, case_name)

        # Find centroid log
        log_files = glob.glob(os.path.join(case_path, "*_centroid_log.json"))
        if not log_files:
            continue

        # Find npz
        npz_files = glob.glob(os.path.join(case_path, "*_trajectory_data.npz"))
        if not npz_files:
            continue

        with open(log_files[0]) as f:
            log_data = json.load(f)

        npz_data = np.load(npz_files[0])
        Xro = npz_data["Xro"]
        obj_target = npz_data["obj_target"]

        cases.append({
            "name": case_name,
            "centroids": log_data["centroids"],
            "Xro": Xro,
            "obj_target": obj_target,
            "object": log_data.get("object", case_name),
            "tag": log_data.get("tag", ""),
        })

    return cases


# ── Experiment Runner ──

def run_experiment(centroids, Xro, kf):
    """Run a KF variant on one trajectory.

    Returns dict with per-frame errors and summary stats.
    NO ground truth is used by the KF — only for evaluation.
    """
    kf.reset()
    n = len(centroids)
    results = []

    for i, c in enumerate(centroids):
        # Xro[:, 0] = initial state, centroid[0] = step after first action
        step_idx = c.get("step", i)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        cseg_conf = c.get("cseg_confidence", 0.0) or 0.0
        cseg_n_px = c.get("cseg_n_pixels", 0) or 0

        # Use saved bearing/elevation if available, else compute from pixels
        if "cseg_bearing" in c and c["cseg_bearing"] is not None and cseg_n_px > 0:
            cseg_bearing = c["cseg_bearing"]
            cseg_elevation = c["cseg_elevation"]
        elif cseg_n_px > 0:
            cseg_cx = c.get("cseg_cx", IMG_W // 2)
            cseg_cy = c.get("cseg_cy", IMG_H // 2)
            cseg_bearing = 2.0 * (cseg_cx / IMG_W) - 1.0
            cseg_elevation = 2.0 * (cseg_cy / IMG_H) - 1.0
        else:
            cseg_bearing = 0.0
            cseg_elevation = 0.0

        est = kf.step(xcr, cseg_bearing, cseg_elevation, cseg_conf, cseg_n_px)

        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        kf_err = None
        cseg_err = None
        if gt_in and gt_u is not None and gt_v is not None:
            if est is not None:
                kf_err = float(np.hypot(est[0] - gt_u, est[1] - gt_v))
            cseg_cx = c.get("cseg_cx")
            cseg_cy = c.get("cseg_cy")
            if cseg_n_px > 0 and cseg_cx is not None:
                cseg_err = float(np.hypot(cseg_cx - gt_u, cseg_cy - gt_v))

        results.append({
            "step": i,
            "kf_err_px": kf_err,
            "cseg_err_px": cseg_err,
            "cseg_conf": cseg_conf,
            "gt_in": gt_in,
            "kf_active": est is not None,
            "coasting": kf.steps_coasting,
        })

    # Summary stats
    kf_errs = [r["kf_err_px"] for r in results if r["kf_err_px"] is not None]
    cseg_errs = [r["cseg_err_px"] for r in results if r["cseg_err_px"] is not None]
    kf_active_gt = [r for r in results if r["gt_in"] and r["kf_active"]]
    kf_coasting_gt = [r for r in results if r["gt_in"] and r["kf_active"]
                      and r["coasting"] > 0]

    def _stats(arr):
        if not arr:
            return {"n": 0}
        arr = np.array(arr)
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    # Split KF errors into measured vs coasting
    kf_meas_errs = [r["kf_err_px"] for r in results
                     if r["kf_err_px"] is not None and r["coasting"] == 0]
    kf_coast_errs = [r["kf_err_px"] for r in results
                      if r["kf_err_px"] is not None and r["coasting"] > 0]

    return {
        "frames": results,
        "kf_all": _stats(kf_errs),
        "kf_measured": _stats(kf_meas_errs),
        "kf_coasting": _stats(kf_coast_errs),
        "cseg": _stats(cseg_errs),
        "n_frames": n,
        "n_gt_visible": sum(1 for r in results if r["gt_in"]),
        "n_kf_active": sum(1 for r in results if r["kf_active"]),
        "n_measured": kf.n_measured,
        "n_rejected": kf.n_rejected,
    }


def print_summary_table(all_results):
    """Print a formatted summary table across all configs and cases."""
    print("\n" + "=" * 120)
    print(f"{'Config':<45} {'Case':<35} {'KF mean':>8} {'KF med':>8} "
          f"{'KF p90':>8} {'CSEG med':>8} {'meas':>5} {'coast':>5} {'rej':>4}")
    print("-" * 120)

    for config_name, case_results in all_results.items():
        for case_name, res in case_results.items():
            kf = res["kf_all"]
            cseg = res["cseg"]
            kf_mean = f"{kf['mean']:.1f}" if kf.get("mean") is not None else "-"
            kf_med = f"{kf['median']:.1f}" if kf.get("median") is not None else "-"
            kf_p90 = f"{kf['p90']:.1f}" if kf.get("p90") is not None else "-"
            cseg_med = f"{cseg['median']:.1f}" if cseg.get("median") is not None else "-"
            n_meas = res.get("n_measured", 0)
            n_coast = res["kf_all"].get("n", 0) - n_meas if kf.get("n") else 0
            n_rej = res.get("n_rejected", 0)
            print(f"{config_name:<45} {case_name:<35} {kf_mean:>8} {kf_med:>8} "
                  f"{kf_p90:>8} {cseg_med:>8} {n_meas:>5} {n_coast:>5} {n_rej:>4}")

    print("=" * 120)


def print_aggregate_table(all_results):
    """Print aggregate stats (mean across cases) per config."""
    print("\n" + "=" * 100)
    print(f"{'Config':<50} {'KF mean':>8} {'KF med':>8} {'KF p90':>8} "
          f"{'CSEG med':>8} {'detect%':>7} {'rej%':>6}")
    print("-" * 100)

    config_agg = {}
    for config_name, case_results in all_results.items():
        kf_means = []
        kf_meds = []
        kf_p90s = []
        cseg_meds = []
        detect_pcts = []
        rej_pcts = []

        for case_name, res in case_results.items():
            kf = res["kf_all"]
            cseg = res["cseg"]
            if kf.get("mean") is not None:
                kf_means.append(kf["mean"])
            if kf.get("median") is not None:
                kf_meds.append(kf["median"])
            if kf.get("p90") is not None:
                kf_p90s.append(kf["p90"])
            if cseg.get("median") is not None:
                cseg_meds.append(cseg["median"])
            n_gt = res["n_gt_visible"]
            n_meas = res.get("n_measured", 0)
            if n_gt > 0:
                detect_pcts.append(100.0 * n_meas / n_gt)
            total_above_gate = n_meas + res.get("n_rejected", 0)
            if total_above_gate > 0:
                rej_pcts.append(100.0 * res.get("n_rejected", 0) / total_above_gate)

        agg_mean = np.mean(kf_means) if kf_means else None
        agg_med = np.mean(kf_meds) if kf_meds else None
        agg_p90 = np.mean(kf_p90s) if kf_p90s else None
        agg_cseg = np.mean(cseg_meds) if cseg_meds else None
        agg_det = np.mean(detect_pcts) if detect_pcts else None
        agg_rej = np.mean(rej_pcts) if rej_pcts else None

        config_agg[config_name] = {
            "kf_mean": agg_mean, "kf_med": agg_med, "kf_p90": agg_p90,
            "cseg_med": agg_cseg, "detect_pct": agg_det, "rej_pct": agg_rej
        }

        m = f"{agg_mean:.1f}" if agg_mean is not None else "-"
        d = f"{agg_med:.1f}" if agg_med is not None else "-"
        p = f"{agg_p90:.1f}" if agg_p90 is not None else "-"
        c = f"{agg_cseg:.1f}" if agg_cseg is not None else "-"
        det = f"{agg_det:.1f}" if agg_det is not None else "-"
        rej = f"{agg_rej:.1f}" if agg_rej is not None else "-"
        print(f"{config_name:<50} {m:>8} {d:>8} {p:>8} {c:>8} {det:>7} {rej:>6}")

    print("=" * 100)
    return config_agg


# ── Build Experiment Configs ──

def build_experiment_grid():
    """Build a grid of KF configurations to test."""
    configs = {}

    # 1. Baseline: sweep conf_gate
    for cg in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        name = f"baseline_cg{cg:.2f}"
        configs[name] = InvDepthEKF(conf_gate=cg, name=name)

    # 2. Adaptive R: sweep conf_gate
    for cg in [0.40, 0.50, 0.60, 0.70]:
        name = f"adaptiveR_cg{cg:.2f}"
        configs[name] = InvDepthEKF(conf_gate=cg, adaptive_R=True, name=name)

    # 3. Innovation gating (chi-sq 2DOF: 5.99 at 95%, 9.21 at 99%)
    for ig in [2.0, 2.5, 3.0]:
        for cg in [0.50, 0.60]:
            name = f"innovGate{ig:.1f}_cg{cg:.2f}"
            configs[name] = InvDepthEKF(conf_gate=cg, innovation_gate=ig, name=name)

    # 4. Adaptive R + innovation gating
    for ig in [2.5, 3.0]:
        for cg in [0.50, 0.60]:
            name = f"adaptR+innovG{ig:.1f}_cg{cg:.2f}"
            configs[name] = InvDepthEKF(conf_gate=cg, adaptive_R=True,
                                         innovation_gate=ig, name=name)

    # 5. Temporal decay gate
    for cg in [0.60, 0.70]:
        name = f"tempDecay_cg{cg:.2f}"
        configs[name] = InvDepthEKF(conf_gate=cg, temporal_decay_gate=True, name=name)

    # 6. IS-EKF (innovation saturation)
    for sat in [1.5, 2.0, 2.5, 3.0]:
        for cg in [0.50, 0.60]:
            name = f"ISEKF_sat{sat:.1f}_cg{cg:.2f}"
            configs[name] = ISEKF(saturation_threshold=sat, conf_gate=cg,
                                   name=name)

    # 7. IS-EKF + adaptive R
    for sat in [2.0, 2.5]:
        for cg in [0.50, 0.60]:
            name = f"ISEKF+adaptR_sat{sat:.1f}_cg{cg:.2f}"
            configs[name] = ISEKF(saturation_threshold=sat, conf_gate=cg,
                                   adaptive_R=True, name=name)

    # 8. EMA smoothing
    for alpha in [0.2, 0.3, 0.5, 0.7]:
        for cg in [0.50, 0.60]:
            name = f"EMA_a{alpha:.1f}_cg{cg:.2f}"
            configs[name] = EMASmoothedEKF(ema_alpha=alpha, conf_gate=cg, name=name)

    # 9. Coasting limit
    for max_coast in [15, 30, 50]:
        for cg in [0.50, 0.60]:
            name = f"coastLim{max_coast}_cg{cg:.2f}"
            configs[name] = CoastingLimitEKF(max_coast_frames=max_coast,
                                              conf_gate=cg, name=name)

    # 10. Process noise sweep (with best conf_gate)
    for pn in [0.002, 0.005, 0.010, 0.020]:
        name = f"procNoise{pn:.3f}_cg0.60"
        configs[name] = InvDepthEKF(conf_gate=0.60, process_noise_be=pn, name=name)

    # 11. Measurement noise sweep
    for mn in [0.01, 0.03, 0.05, 0.10]:
        name = f"measNoise{mn:.2f}_cg0.60"
        configs[name] = InvDepthEKF(conf_gate=0.60, meas_noise=mn, name=name)

    # 12. Consensus init (require N consistent detections before starting)
    for n_init in [2, 3, 5]:
        for cg in [0.50, 0.60, 0.65]:
            name = f"consensus_n{n_init}_cg{cg:.2f}"
            configs[name] = ConsensusInitEKF(
                n_init=n_init, init_radius_px=50.0, max_coast_frames=20,
                conf_gate=cg, name=name)

    # 13. Consensus init + innovation gating + adaptive R (combined best)
    for n_init in [2, 3]:
        for cg in [0.55, 0.60, 0.65]:
            name = f"combined_n{n_init}_ig2.5_adaptR_cg{cg:.2f}"
            configs[name] = ConsensusInitEKF(
                n_init=n_init, init_radius_px=50.0, max_coast_frames=20,
                conf_gate=cg, innovation_gate=2.5, adaptive_R=True, name=name)

    # 14. Consensus init + temporal decay
    for n_init in [2, 3]:
        for cg in [0.65, 0.70]:
            name = f"consensus_n{n_init}_tempDecay_cg{cg:.2f}"
            configs[name] = ConsensusInitEKF(
                n_init=n_init, init_radius_px=50.0, max_coast_frames=20,
                conf_gate=cg, temporal_decay_gate=True, name=name)

    return configs


def main():
    t0 = time.time()
    eval_dir = find_eval_dir_with_npz()
    if eval_dir is None:
        print("ERROR: No eval directory with trajectory_data.npz files found")
        print("Run eval_failed_val_videos.py first to generate data")
        return

    print(f"Loading data from: {eval_dir}")
    cases = load_all_cases(eval_dir)
    print(f"Loaded {len(cases)} cases:")
    for c in cases:
        print(f"  {c['name']}: {len(c['centroids'])} frames, "
              f"Xro shape={c['Xro'].shape}, obj={c['object']}")

    if not cases:
        print("ERROR: No cases with both centroid logs and trajectory data")
        return

    # Build experiment grid
    configs = build_experiment_grid()
    print(f"\nRunning {len(configs)} KF configurations x {len(cases)} cases "
          f"= {len(configs) * len(cases)} experiments...")

    # Run all experiments
    all_results = {}
    for config_name, kf in configs.items():
        case_results = {}
        for case in cases:
            res = run_experiment(case["centroids"], case["Xro"], kf)
            case_results[case["name"]] = res
        all_results[config_name] = case_results

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Print results
    print_summary_table(all_results)
    config_agg = print_aggregate_table(all_results)

    # Find best config by aggregate KF median error
    best_name = None
    best_med = float("inf")
    for name, agg in config_agg.items():
        if agg["kf_med"] is not None and agg["kf_med"] < best_med:
            best_med = agg["kf_med"]
            best_name = name

    if best_name:
        print(f"\n*** BEST CONFIG: {best_name} — KF median={best_med:.1f}px ***")

    # Also find best by mean
    best_mean_name = None
    best_mean = float("inf")
    for name, agg in config_agg.items():
        if agg["kf_mean"] is not None and agg["kf_mean"] < best_mean:
            best_mean = agg["kf_mean"]
            best_mean_name = name

    if best_mean_name:
        print(f"*** BEST BY MEAN: {best_mean_name} — KF mean={best_mean:.1f}px ***")

    # Save results to JSON
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"kf_grid_{ts}.json")

    # Serialize (skip frames to keep file small)
    save_data = {}
    for config_name, case_results in all_results.items():
        save_data[config_name] = {}
        for case_name, res in case_results.items():
            save_data[config_name][case_name] = {
                k: v for k, v in res.items() if k != "frames"
            }
    save_data["_meta"] = {
        "eval_dir": eval_dir,
        "n_cases": len(cases),
        "n_configs": len(configs),
        "elapsed_s": elapsed,
        "timestamp": ts,
        "best_by_median": best_name,
        "best_by_mean": best_mean_name,
        "config_aggregate": config_agg,
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Print detailed breakdown for best config
    if best_name and best_name in all_results:
        print(f"\n{'='*80}")
        print(f"DETAILED: {best_name}")
        print(f"{'='*80}")
        for case_name, res in all_results[best_name].items():
            kf = res["kf_all"]
            kf_m = res["kf_measured"]
            kf_c = res["kf_coasting"]
            print(f"\n  {case_name}:")
            if kf.get("n", 0) > 0:
                print(f"    KF overall:  mean={kf['mean']:.1f}  med={kf['median']:.1f}  "
                      f"p90={kf['p90']:.1f}  max={kf['max']:.1f}  (n={kf['n']})")
            if kf_m.get("n", 0) > 0:
                print(f"    KF measured:  mean={kf_m['mean']:.1f}  med={kf_m['median']:.1f}  "
                      f"p90={kf_m['p90']:.1f}  (n={kf_m['n']})")
            if kf_c.get("n", 0) > 0:
                print(f"    KF coasting: mean={kf_c['mean']:.1f}  med={kf_c['median']:.1f}  "
                      f"p90={kf_c['p90']:.1f}  (n={kf_c['n']})")
            print(f"    Measured: {res['n_measured']}  Rejected: {res['n_rejected']}  "
                  f"GT visible: {res['n_gt_visible']}")


if __name__ == "__main__":
    main()
