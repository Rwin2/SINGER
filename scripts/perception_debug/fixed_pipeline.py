#!/usr/bin/env python3
"""
Fixed perception pipeline experiment.

Key findings from step_analysis.py:
1. temporal_decay_gate BUG: with conf_gate=0.05, after first detection
   the gate goes UP to 0.30 (max(0.05-0.15, 0.3)), rejecting all
   OWL-ViT measurements (typically 0.05-0.10).
2. Consensus init is too strict — filters 60% of valid detections.
3. Raw detector median error is 9.8px — better than any KF variant.

This script tests:
A. Raw detector (no filter) — baseline
B. Simple EMA on pixel coordinates
C. ConsensusInitEKF with temporal_decay_gate=False (bug fix)
D. ConsensusInitEKF with n_init=2 (less strict)
E. ConsensusInitEKF with n_init=1 (just coasting limit)
F. Pixel-space Kalman filter (2D, no inverse-depth)

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/fixed_pipeline.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

from perception_debug.kf_experiments import (
    InvDepthEKF, ConsensusInitEKF, IMG_W, IMG_H,
)


# ── Simple Filters ──

class RawDetector:
    """No filter — just use raw detections."""
    def __init__(self):
        self.steps_coasting = 0
        self.name = "RawDetector"

    def reset(self):
        self.steps_coasting = 0

    def step(self, xcr, b, e, conf, area):
        if area > 0 and conf >= 0.02:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0
            self.steps_coasting = 0
            return int(round(u)), int(round(v)), 0.0
        self.steps_coasting += 1
        return None


class EMAFilter:
    """Simple exponential moving average on pixel coordinates."""
    def __init__(self, alpha=0.5, conf_gate=0.02, max_coast=20):
        self.alpha = alpha
        self.conf_gate = conf_gate
        self.max_coast = max_coast
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.name = f"EMA(α={alpha})"

    def reset(self):
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0

    def step(self, xcr, b, e, conf, area):
        if area > 0 and conf >= self.conf_gate:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0
            if self.u_ema is None:
                self.u_ema = u
                self.v_ema = v
            else:
                self.u_ema = self.alpha * u + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * v + (1 - self.alpha) * self.v_ema
            self.steps_coasting = 0
            return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
        else:
            self.steps_coasting += 1
            if self.u_ema is not None and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
            if self.steps_coasting > self.max_coast:
                self.u_ema = None
                self.v_ema = None
            return None


class PixelKF:
    """Simple 2D pixel-space Kalman filter with constant-velocity model."""
    def __init__(self, conf_gate=0.02, process_noise=5.0, meas_noise=10.0,
                 max_coast=20):
        self.conf_gate = conf_gate
        self.process_noise = process_noise
        self.meas_noise = meas_noise
        self.max_coast = max_coast
        # State: [u, v, du, dv] (position + velocity in pixels)
        self.x = None
        self.P = None
        self.initialized = False
        self.steps_coasting = 0
        self.name = f"PixelKF(q={process_noise},r={meas_noise})"

    def reset(self):
        self.x = None
        self.P = None
        self.initialized = False
        self.steps_coasting = 0

    def step(self, xcr, b, e, conf, area):
        has_meas = (area > 0 and conf >= self.conf_gate)

        if has_meas:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0
            z = np.array([u, v])

            if not self.initialized:
                self.x = np.array([u, v, 0.0, 0.0])
                self.P = np.diag([self.meas_noise**2, self.meas_noise**2,
                                  self.process_noise**2, self.process_noise**2])
                self.initialized = True
                self.steps_coasting = 0
                return int(round(u)), int(round(v)), 0.0

        if not self.initialized:
            self.steps_coasting += 1
            return None

        # Predict
        dt = 1.0
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        q = self.process_noise
        Q = np.diag([q**2 * dt**2, q**2 * dt**2, q**2, q**2])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Update
        if has_meas:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            R = np.eye(2) * self.meas_noise**2
            z = np.array([u, v])
            y = z - H @ self.x
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ H) @ self.P
            self.steps_coasting = 0
        else:
            self.steps_coasting += 1
            if self.steps_coasting > self.max_coast:
                self.initialized = False
                self.x = None
                self.P = None
                return None

        return int(round(self.x[0])), int(round(self.x[1])), 0.0


class ConsensusEMAFilter:
    """Consensus init (n=2 or 3) + EMA tracking. No EKF."""
    def __init__(self, n_init=2, init_radius_px=50.0, alpha=0.6,
                 conf_gate=0.02, max_coast=20):
        self.n_init = n_init
        self.init_radius_px = init_radius_px
        self.alpha = alpha
        self.conf_gate = conf_gate
        self.max_coast = max_coast
        self._init_buffer = []
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.initialized = False
        self.name = f"ConsEMA(n={n_init},α={alpha})"

    def reset(self):
        self._init_buffer = []
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.initialized = False

    def step(self, xcr, b, e, conf, area):
        has_meas = (area > 0 and conf >= self.conf_gate)

        # Coasting limit
        if self.initialized and self.steps_coasting >= self.max_coast:
            self.initialized = False
            self.u_ema = None
            self.v_ema = None
            self._init_buffer = []

        if has_meas:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0

            if not self.initialized:
                self._init_buffer.append((u, v))
                if len(self._init_buffer) >= self.n_init:
                    us = [p[0] for p in self._init_buffer[-self.n_init:]]
                    vs = [p[1] for p in self._init_buffer[-self.n_init:]]
                    spread = np.hypot(max(us)-min(us), max(vs)-min(vs))
                    if spread <= self.init_radius_px:
                        self.u_ema = np.mean(us)
                        self.v_ema = np.mean(vs)
                        self.initialized = True
                        self.steps_coasting = 0
                        return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
                return None
            else:
                self.u_ema = self.alpha * u + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * v + (1 - self.alpha) * self.v_ema
                self.steps_coasting = 0
                return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
        else:
            self.steps_coasting += 1
            if self.initialized and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
            return None


# ── OWL-ViT ──

def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


RICH_PROMPTS = {
    "green clock": [
        "green clock", "a green wall clock", "green analog clock",
        "round green clock on table", "green colored clock",
    ],
    "leafblower": [
        "leafblower", "a leaf blower", "leaf blower tool",
        "garden leaf blower", "handheld blower",
    ],
    "drill": [
        "drill", "a power drill", "electric drill",
        "cordless drill", "hand drill tool",
    ],
}


def _get_prompts(obj_name, use_rich=True):
    if use_rich:
        for key, prompts in RICH_PROMPTS.items():
            if key in obj_name.lower():
                return prompts
    return [obj_name, f"a {obj_name}"]


USE_RICH_PROMPTS = True


def detect_owlvit(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
    img = Image.fromarray(frame_rgb)
    queries = [_get_prompts(obj_name, USE_RICH_PROMPTS)]
    inputs = proc(text=queries, images=img, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mdl(**inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to(dev)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold)
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    if len(boxes) == 0:
        return None, None, 0.0, 0
    best = scores.argmax()
    box = boxes[best]
    score = float(scores[best])
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = int((box[2] - box[0]) * (box[3] - box[1]))
    return cx, cy, score, area


# ── Data ──

def find_eval_dir():
    base = os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_CENTROID_V9",
                        "visualizations", "failed_val")
    dirs = sorted(glob.glob(os.path.join(base, "20*")))
    for d in reversed(dirs):
        npz_files = glob.glob(os.path.join(d, "**", "*_trajectory_data.npz"),
                              recursive=True)
        if npz_files:
            return d
    return None


def load_cases(eval_dir):
    cases = []
    for case_dir in sorted(os.listdir(eval_dir)):
        case_path = os.path.join(eval_dir, case_dir)
        if not os.path.isdir(case_path):
            continue
        rgb_vids = glob.glob(os.path.join(case_path, "*_rgb.mp4"))
        log_files = glob.glob(os.path.join(case_path, "*_centroid_log.json"))
        npz_files = glob.glob(os.path.join(case_path, "*_trajectory_data.npz"))
        if not (rgb_vids and log_files and npz_files):
            continue
        with open(log_files[0]) as f:
            log_data = json.load(f)
        npz = np.load(npz_files[0])
        cases.append({
            "name": case_dir,
            "video_path": rgb_vids[0],
            "centroids": log_data["centroids"],
            "object": log_data.get("object", case_dir),
            "Xro": npz["Xro"],
            "obj_target": npz["obj_target"],
        })
    return cases


# ── Test Runner ──

def run_test(case, filt, proc, mdl, dev):
    filt.reset()
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    Xro = case["Xro"]
    obj_name = case["object"]
    n_frames = len(centroids)

    kf_errs = []
    det_errs = []
    n_det = 0
    n_tp = 0
    n_fp = 0
    n_gt_vis = 0
    n_est = 0  # frames with any estimate (det or coast)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area = detect_owlvit(
            frame_rgb, obj_name, proc, mdl, dev, threshold=0.02)
        detected = (area > 0 and conf >= 0.02)

        step_idx = centroids[frame_idx].get("step", frame_idx)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            est = filt.step(xcr, b, e, conf, area)
        else:
            est = filt.step(xcr, 0.0, 0.0, 0.0, 0)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        if gt_in:
            n_gt_vis += 1
            if detected:
                n_tp += 1
        if detected and not gt_in:
            n_fp += 1
        if detected:
            n_det += 1

        if gt_in and gt_u is not None:
            if est is not None:
                kf_errs.append(float(np.hypot(est[0] - gt_u, est[1] - gt_v)))
                n_est += 1
            if detected and cx is not None:
                det_errs.append(float(np.hypot(cx - gt_u, cy - gt_v)))

        frame_idx += 1

    cap.release()

    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    return {
        "filt": _s(kf_errs),
        "det": _s(det_errs),
        "n_frames": frame_idx,
        "n_gt_vis": n_gt_vis,
        "n_det": n_det,
        "n_est": n_est,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "det_rate": n_tp / max(n_gt_vis, 1),
        "coverage": n_est / max(n_gt_vis, 1),
    }


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return

    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases")

    print("Loading OWL-ViT...")
    proc, mdl, dev = load_owlvit()

    # Filters to test
    filters = {
        # A. No filter
        "raw": RawDetector(),
        # B. EMA variants
        "ema_0.3": EMAFilter(alpha=0.3, max_coast=20),
        "ema_0.5": EMAFilter(alpha=0.5, max_coast=20),
        "ema_0.7": EMAFilter(alpha=0.7, max_coast=20),
        "ema_0.9": EMAFilter(alpha=0.9, max_coast=20),
        "ema_0.5_coast30": EMAFilter(alpha=0.5, max_coast=30),
        # C. Fixed ConsensusInitEKF (no temporal decay)
        "ekf_cons3_fixed": ConsensusInitEKF(
            n_init=3, init_radius_px=50.0, max_coast_frames=20,
            conf_gate=0.05, temporal_decay_gate=False),
        # D. ConsensusInitEKF n=2
        "ekf_cons2_fixed": ConsensusInitEKF(
            n_init=2, init_radius_px=50.0, max_coast_frames=20,
            conf_gate=0.05, temporal_decay_gate=False),
        # E. ConsensusInitEKF n=1 (just coasting limit)
        "ekf_cons1_fixed": ConsensusInitEKF(
            n_init=1, init_radius_px=50.0, max_coast_frames=20,
            conf_gate=0.05, temporal_decay_gate=False),
        # F. Pixel-space Kalman
        "pixel_kf_q5_r10": PixelKF(process_noise=5.0, meas_noise=10.0, max_coast=20),
        "pixel_kf_q2_r5": PixelKF(process_noise=2.0, meas_noise=5.0, max_coast=20),
        "pixel_kf_q10_r20": PixelKF(process_noise=10.0, meas_noise=20.0, max_coast=20),
        # G. Consensus + EMA
        "cons2_ema_0.6": ConsensusEMAFilter(n_init=2, alpha=0.6, max_coast=20),
        "cons2_ema_0.8": ConsensusEMAFilter(n_init=2, alpha=0.8, max_coast=20),
        "cons3_ema_0.6": ConsensusEMAFilter(n_init=3, alpha=0.6, max_coast=20),
    }

    print(f"Testing {len(filters)} filters x {len(cases)} cases = {len(filters)*len(cases)} runs")

    all_results = {}
    for filt_name, filt in filters.items():
        print(f"\n{'='*80}")
        print(f"FILTER: {filt_name}")
        print(f"{'='*80}")
        case_results = {}
        for case in cases:
            t1 = time.time()
            res = run_test(case, filt, proc, mdl, dev)
            dt = time.time() - t1

            f = res["filt"]
            d = res["det"]
            f_m = f"{f['mean']:.1f}" if f.get("mean") else "-"
            f_d = f"{f['median']:.1f}" if f.get("median") else "-"
            d_d = f"{d['median']:.1f}" if d.get("median") else "-"
            dr = f"{res['det_rate']*100:.1f}%"
            cov = f"{res['coverage']*100:.1f}%"
            print(f"  {case['name']:<45}  filt={f_m:>7}/{f_d:>7}  "
                  f"det={d_d:>7}  det%={dr:>6}  cov={cov:>6}  FP={res['n_fp']}  ({dt:.1f}s)")
            case_results[case["name"]] = res
        all_results[filt_name] = case_results

    elapsed = time.time() - t0

    # ── Aggregate Table ──
    print(f"\n\n{'#'*110}")
    print(f"{'Filter':<25} {'filt_mean':>10} {'filt_med':>10} {'filt_p90':>10} "
          f"{'det_med':>10} {'det%':>7} {'coverage':>10} {'FP':>4}")
    print(f"{'-'*110}")

    for filt_name, case_results in all_results.items():
        filt_means, filt_meds, filt_p90s, det_ms = [], [], [], []
        det_rs, covs, fps = [], [], []
        for cname, res in case_results.items():
            f = res["filt"]
            if f.get("mean"): filt_means.append(f["mean"])
            if f.get("median"): filt_meds.append(f["median"])
            if f.get("p90"): filt_p90s.append(f["p90"])
            d = res["det"]
            if d.get("median"): det_ms.append(d["median"])
            det_rs.append(res["det_rate"] * 100)
            covs.append(res["coverage"] * 100)
            fps.append(res["n_fp"])

        m = f"{np.mean(filt_means):.1f}" if filt_means else "-"
        d = f"{np.mean(filt_meds):.1f}" if filt_meds else "-"
        p = f"{np.mean(filt_p90s):.1f}" if filt_p90s else "-"
        dm = f"{np.mean(det_ms):.1f}" if det_ms else "-"
        det = f"{np.mean(det_rs):.1f}%"
        cov = f"{np.mean(covs):.1f}%"
        fp = f"{sum(fps)}"
        print(f"{filt_name:<25} {m:>10} {d:>10} {p:>10} {dm:>10} {det:>7} {cov:>10} {fp:>4}")

    print(f"{'#'*110}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"fixed_pipeline_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
