#!/usr/bin/env python3
"""
FP suppression experiments.

Problem: Rich prompts give 76.3% detection but 58 FPs (all when GT not in frame).
  - leafblower_val0: 25 FP (object leaves FOV, detector picks up random stuff)
  - drill_val7: 29 FP (same issue)
  - clock_val5: 4 FP (no GT ever visible)

Strategies:
1. Higher confidence threshold (0.05, 0.08, 0.10 vs current 0.02)
2. Spatial gating: reject detections > N px from running estimate
3. Score-weighted EMA: weight by confidence
4. Combined: spatial gate + higher threshold + EMA

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/fp_suppression.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

from perception_debug.kf_experiments import IMG_W, IMG_H


# ── Filters ──

class SpatialGateEMA:
    """EMA with spatial gating: reject detections that jump too far from track."""
    def __init__(self, alpha=0.5, conf_gate=0.02, max_coast=20,
                 gate_radius_px=100.0, n_init=2, init_radius_px=60.0):
        self.alpha = alpha
        self.conf_gate = conf_gate
        self.max_coast = max_coast
        self.gate_radius_px = gate_radius_px
        self.n_init = n_init
        self.init_radius_px = init_radius_px
        self._init_buffer = []
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.initialized = False
        self.name = f"SpatGateEMA(α={alpha},gate={gate_radius_px}px,thr={conf_gate})"
        # Track rejected/accepted for diagnostics
        self.n_rejected = 0
        self.n_accepted = 0

    def reset(self):
        self._init_buffer = []
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.initialized = False
        self.n_rejected = 0
        self.n_accepted = 0

    def step(self, xcr, b, e, conf, area):
        has_meas = (area > 0 and conf >= self.conf_gate)

        # Coast timeout
        if self.initialized and self.steps_coasting >= self.max_coast:
            self.initialized = False
            self.u_ema = None
            self.v_ema = None
            self._init_buffer = []

        if has_meas:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0

            if not self.initialized:
                # Consensus init
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
                        self.n_accepted += 1
                        return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
                return None
            else:
                # Spatial gate: reject if too far from current estimate
                dist = np.hypot(u - self.u_ema, v - self.v_ema)
                if dist > self.gate_radius_px:
                    self.n_rejected += 1
                    # Treat as no detection (coast)
                    self.steps_coasting += 1
                    if self.steps_coasting <= self.max_coast:
                        return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
                    return None

                # Accept and update EMA
                self.u_ema = self.alpha * u + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * v + (1 - self.alpha) * self.v_ema
                self.steps_coasting = 0
                self.n_accepted += 1
                return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
        else:
            self.steps_coasting += 1
            if self.initialized and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema)), 0.0
            return None


class ScoreWeightedEMA:
    """EMA that weights updates by detection confidence."""
    def __init__(self, base_alpha=0.5, conf_gate=0.02, max_coast=20):
        self.base_alpha = base_alpha
        self.conf_gate = conf_gate
        self.max_coast = max_coast
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.name = f"ScoreEMA(α={base_alpha},thr={conf_gate})"

    def reset(self):
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0

    def step(self, xcr, b, e, conf, area):
        if area > 0 and conf >= self.conf_gate:
            u = (b + 1.0) * IMG_W / 2.0
            v = (e + 1.0) * IMG_H / 2.0
            # Scale alpha by confidence (higher conf → more responsive)
            alpha = self.base_alpha * min(conf / 0.10, 1.0)
            alpha = max(alpha, 0.1)  # floor
            if self.u_ema is None:
                self.u_ema = u
                self.v_ema = v
            else:
                self.u_ema = alpha * u + (1 - alpha) * self.u_ema
                self.v_ema = alpha * v + (1 - alpha) * self.v_ema
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


class SimpleThresholdEMA:
    """Plain EMA with configurable confidence threshold."""
    def __init__(self, alpha=0.5, conf_gate=0.05, max_coast=20):
        self.alpha = alpha
        self.conf_gate = conf_gate
        self.max_coast = max_coast
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.name = f"EMA(α={alpha},thr={conf_gate})"

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


def _get_prompts(obj_name):
    for key, prompts in RICH_PROMPTS.items():
        if key in obj_name.lower():
            return prompts
    return [obj_name, f"a {obj_name}"]


def detect_owlvit(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
    """Returns (cx, cy, conf, area, all_boxes, all_scores) for FP analysis."""
    img = Image.fromarray(frame_rgb)
    queries = [_get_prompts(obj_name)]
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
        return None, None, 0.0, 0, boxes, scores
    best = scores.argmax()
    box = boxes[best]
    score = float(scores[best])
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = int((box[2] - box[0]) * (box[3] - box[1]))
    return cx, cy, score, area, boxes, scores


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
    n_est = 0
    # FP analysis
    fp_confs = []
    fp_dists_from_track = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area, all_boxes, all_scores = detect_owlvit(
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
            fp_confs.append(conf)
            # Track distance from last estimate (if any)
            if hasattr(filt, 'u_ema') and filt.u_ema is not None:
                fp_dists_from_track.append(
                    float(np.hypot(cx - filt.u_ema, cy - filt.v_ema)))
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

    res = {
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
        "fp_confs": fp_confs,
    }
    if hasattr(filt, 'n_rejected'):
        res["n_rejected"] = filt.n_rejected
        res["n_accepted"] = filt.n_accepted
    return res


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

    # Filters to test — focused on FP suppression
    filters = {
        # Baseline: raw with rich prompts, threshold=0.02
        "baseline_thr0.02": SimpleThresholdEMA(alpha=1.0, conf_gate=0.02, max_coast=0),

        # A. Higher confidence thresholds (simple)
        "thr0.05_ema0.5": SimpleThresholdEMA(alpha=0.5, conf_gate=0.05, max_coast=20),
        "thr0.08_ema0.5": SimpleThresholdEMA(alpha=0.5, conf_gate=0.08, max_coast=20),
        "thr0.10_ema0.5": SimpleThresholdEMA(alpha=0.5, conf_gate=0.10, max_coast=20),
        "thr0.15_ema0.5": SimpleThresholdEMA(alpha=0.5, conf_gate=0.15, max_coast=20),

        # B. Spatial gating (reject jumps > N px from track)
        "spat50_thr0.02": SpatialGateEMA(alpha=0.5, conf_gate=0.02, gate_radius_px=50.0,
                                          n_init=2, max_coast=20),
        "spat80_thr0.02": SpatialGateEMA(alpha=0.5, conf_gate=0.02, gate_radius_px=80.0,
                                          n_init=2, max_coast=20),
        "spat120_thr0.02": SpatialGateEMA(alpha=0.5, conf_gate=0.02, gate_radius_px=120.0,
                                           n_init=2, max_coast=20),

        # C. Combined: spatial gate + higher threshold
        "spat80_thr0.05": SpatialGateEMA(alpha=0.5, conf_gate=0.05, gate_radius_px=80.0,
                                          n_init=2, max_coast=20),
        "spat80_thr0.08": SpatialGateEMA(alpha=0.5, conf_gate=0.08, gate_radius_px=80.0,
                                          n_init=2, max_coast=20),
        "spat100_thr0.05": SpatialGateEMA(alpha=0.5, conf_gate=0.05, gate_radius_px=100.0,
                                           n_init=2, max_coast=20),

        # D. Score-weighted EMA
        "score_ema_thr0.02": ScoreWeightedEMA(base_alpha=0.5, conf_gate=0.02, max_coast=20),
        "score_ema_thr0.05": ScoreWeightedEMA(base_alpha=0.5, conf_gate=0.05, max_coast=20),

        # E. Best combo candidates
        "spat80_thr0.05_coast30": SpatialGateEMA(alpha=0.5, conf_gate=0.05, gate_radius_px=80.0,
                                                   n_init=2, max_coast=30),
        "spat100_thr0.05_a0.6": SpatialGateEMA(alpha=0.6, conf_gate=0.05, gate_radius_px=100.0,
                                                 n_init=2, max_coast=20),
    }

    print(f"Testing {len(filters)} filters x {len(cases)} cases = {len(filters)*len(cases)} runs\n")

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
            f_m = f"{f['median']:.1f}" if f.get("median") else "-"
            dr = f"{res['det_rate']*100:.1f}%"
            cov = f"{res['coverage']*100:.1f}%"
            rej = f"  rej={res.get('n_rejected','-')}" if 'n_rejected' in res else ""
            print(f"  {case['name']:<45}  med={f_m:>7}px  det={dr:>6}  cov={cov:>6}  FP={res['n_fp']}{rej}  ({dt:.1f}s)")

            # Remove non-serializable data for JSON
            res_clean = {k: v for k, v in res.items()}
            case_results[case["name"]] = res_clean
        all_results[filt_name] = case_results

    elapsed = time.time() - t0

    # ── Aggregate Table ──
    print(f"\n\n{'#'*100}")
    print(f"{'Filter':<30s}  {'det%':>6}  {'cov%':>6}  {'filt_med':>9}  {'FP':>4}  {'rejected':>8}")
    print(f"{'-'*100}")

    for filt_name, case_results in all_results.items():
        all_det = []
        all_cov = []
        all_med = []
        total_fp = 0
        total_rej = 0
        for case_name, v in case_results.items():
            n_gt = v.get("n_gt_vis", 0)
            if n_gt > 0:
                all_det.append(v.get("det_rate", 0))
                all_cov.append(v.get("coverage", 0))
            filt_med = v.get("filt", {}).get("median")
            if filt_med is not None:
                all_med.append(filt_med)
            total_fp += v.get("n_fp", 0)
            total_rej += v.get("n_rejected", 0)

        avg_det = sum(all_det)/len(all_det)*100 if all_det else 0
        avg_cov = sum(all_cov)/len(all_cov)*100 if all_cov else 0
        avg_med = sum(all_med)/len(all_med) if all_med else float('inf')
        print(f"{filt_name:<30s}  {avg_det:5.1f}%  {avg_cov:5.1f}%  {avg_med:8.1f}px  {total_fp:4d}  {total_rej:8d}")

    print(f"{'#'*100}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── FP Confidence Analysis ──
    print(f"\n\n{'='*80}")
    print("FP CONFIDENCE ANALYSIS (baseline_thr0.02)")
    print(f"{'='*80}")
    baseline = all_results.get("baseline_thr0.02", {})
    for case_name, v in baseline.items():
        fp_confs = v.get("fp_confs", [])
        if fp_confs:
            confs = np.array(fp_confs)
            print(f"  {case_name:<45}  FP={len(fp_confs):3d}  "
                  f"conf: min={confs.min():.3f}  max={confs.max():.3f}  "
                  f"med={np.median(confs):.3f}  mean={confs.mean():.3f}")

    # Save results
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"fp_suppression_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
