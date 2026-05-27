#!/usr/bin/env python3
"""
Temporal voting experiment.

Strategy: Require OWL-ViT detection in N of last M frames before accepting.
This exploits the fact that TPs are temporally consistent (object stays in frame
for many consecutive frames) while FPs are sporadic.

Also tests: confidence accumulation (running average of detection confidence,
accept only when accumulated confidence exceeds threshold).

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/temporal_voting.py
"""
import os, sys, json, glob, time
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import torch
from PIL import Image

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

from perception_debug.kf_experiments import IMG_W, IMG_H


RICH_PROMPTS = {
    "green clock": ["green clock", "a green wall clock", "green analog clock",
                    "round green clock on table", "green colored clock"],
    "leafblower": ["leafblower", "a leaf blower", "leaf blower tool",
                   "garden leaf blower", "handheld blower"],
    "drill": ["drill", "a power drill", "electric drill",
              "cordless drill", "hand drill tool"],
}

def _get_prompts(obj_name):
    for key, prompts in RICH_PROMPTS.items():
        if key in obj_name.lower():
            return prompts
    return [obj_name, f"a {obj_name}"]


class TemporalVoteEMA:
    """Accept detection only if detected in N of last M frames."""
    def __init__(self, n_vote=3, m_window=5, alpha=0.5, max_coast=30):
        self.n_vote = n_vote
        self.m_window = m_window
        self.alpha = alpha
        self.max_coast = max_coast
        self.vote_buffer = deque(maxlen=m_window)
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.tracking = False  # True once first vote passes
        self.name = f"TemporalVote({n_vote}/{m_window},alpha={alpha})"

    def reset(self):
        self.vote_buffer.clear()
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.tracking = False

    def step(self, cx, cy, conf, area):
        detected = (area > 0 and conf >= 0.02)
        self.vote_buffer.append(1 if detected else 0)

        votes = sum(self.vote_buffer)

        if detected and votes >= self.n_vote:
            if self.u_ema is None:
                self.u_ema = cx
                self.v_ema = cy
            else:
                self.u_ema = self.alpha * cx + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * cy + (1 - self.alpha) * self.v_ema
            self.steps_coasting = 0
            self.tracking = True
            return int(round(self.u_ema)), int(round(self.v_ema))
        elif detected and self.tracking:
            # Detected but not enough votes (just started losing object) — still update
            self.u_ema = self.alpha * cx + (1 - self.alpha) * self.u_ema
            self.v_ema = self.alpha * cy + (1 - self.alpha) * self.v_ema
            self.steps_coasting = 0
            return int(round(self.u_ema)), int(round(self.v_ema))
        else:
            self.steps_coasting += 1
            if self.u_ema is not None and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema))
            if self.steps_coasting > self.max_coast:
                self.u_ema = None
                self.v_ema = None
                self.tracking = False
            return None


class ConfAccumEMA:
    """Accept when accumulated confidence exceeds threshold."""
    def __init__(self, conf_thresh=0.10, decay=0.8, alpha=0.5, max_coast=30):
        self.conf_thresh = conf_thresh
        self.decay = decay
        self.alpha = alpha
        self.max_coast = max_coast
        self.accumulated_conf = 0.0
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.name = f"ConfAccum(thr={conf_thresh},decay={decay})"

    def reset(self):
        self.accumulated_conf = 0.0
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0

    def step(self, cx, cy, conf, area):
        detected = (area > 0 and conf >= 0.02)

        if detected:
            self.accumulated_conf = self.accumulated_conf * self.decay + conf
        else:
            self.accumulated_conf *= self.decay

        if detected and self.accumulated_conf >= self.conf_thresh:
            if self.u_ema is None:
                self.u_ema = cx
                self.v_ema = cy
            else:
                self.u_ema = self.alpha * cx + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * cy + (1 - self.alpha) * self.v_ema
            self.steps_coasting = 0
            return int(round(self.u_ema)), int(round(self.v_ema))
        else:
            self.steps_coasting += 1
            if self.u_ema is not None and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema))
            if self.steps_coasting > self.max_coast:
                self.u_ema = None
                self.v_ema = None
                self.accumulated_conf = 0.0
            return None


# ── Detector ──
def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit_rich(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
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
        })
    return cases


def run_test(case, filt, proc, mdl, dev):
    filt.reset()
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)

    filt_errs = []
    det_errs = []
    n_det = 0; n_tp = 0; n_fp = 0; n_gt_vis = 0; n_est = 0
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area = detect_owlvit_rich(
            frame_rgb, obj_name, proc, mdl, dev)
        detected = (area > 0 and conf >= 0.02)

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

        # Filter step
        if detected:
            est = filt.step(cx, cy, conf, area)
        else:
            est = filt.step(0, 0, 0.0, 0)

        if gt_in and gt_u is not None:
            if est is not None:
                filt_errs.append(float(np.hypot(est[0] - gt_u, est[1] - gt_v)))
                n_est += 1
            if detected:
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
        "filt": _s(filt_errs),
        "det": _s(det_errs),
        "n_frames": frame_idx, "n_gt_vis": n_gt_vis,
        "n_det": n_det, "n_est": n_est,
        "n_tp": n_tp, "n_fp": n_fp,
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

    filters = {
        # Temporal voting: N of M
        "vote_2of3": TemporalVoteEMA(n_vote=2, m_window=3, alpha=0.5, max_coast=30),
        "vote_3of5": TemporalVoteEMA(n_vote=3, m_window=5, alpha=0.5, max_coast=30),
        "vote_4of7": TemporalVoteEMA(n_vote=4, m_window=7, alpha=0.5, max_coast=30),
        "vote_2of5": TemporalVoteEMA(n_vote=2, m_window=5, alpha=0.5, max_coast=30),
        "vote_3of7": TemporalVoteEMA(n_vote=3, m_window=7, alpha=0.5, max_coast=30),
        # Confidence accumulation
        "confacc_0.05_d0.8": ConfAccumEMA(conf_thresh=0.05, decay=0.8, alpha=0.5, max_coast=30),
        "confacc_0.08_d0.8": ConfAccumEMA(conf_thresh=0.08, decay=0.8, alpha=0.5, max_coast=30),
        "confacc_0.10_d0.8": ConfAccumEMA(conf_thresh=0.10, decay=0.8, alpha=0.5, max_coast=30),
        "confacc_0.05_d0.9": ConfAccumEMA(conf_thresh=0.05, decay=0.9, alpha=0.5, max_coast=30),
        "confacc_0.10_d0.9": ConfAccumEMA(conf_thresh=0.10, decay=0.9, alpha=0.5, max_coast=30),
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
            f_med = f"{res['filt']['median']:.1f}" if res['filt'].get('median') else "-"
            cov = f"{res['coverage']*100:.1f}%"
            print(f"  {case['name']:<45}  cov={cov:>6}  med={f_med:>7}px  FP={res['n_fp']}  ({dt:.1f}s)")
            case_results[case["name"]] = res
        all_results[filt_name] = case_results

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*90}")
    print(f"{'Filter':<25s}  {'det%':>6}  {'cov%':>6}  {'filt_med':>9}  {'FP':>4}")
    print(f"{'-'*90}")
    for filt_name, case_results in all_results.items():
        all_det = []; all_cov = []; all_med = []; total_fp = 0
        for case_name, v in case_results.items():
            if v['n_gt_vis'] > 0:
                all_det.append(v['det_rate'])
                all_cov.append(v['coverage'])
            m = v.get('filt', {}).get('median')
            if m is not None:
                all_med.append(m)
            total_fp += v.get('n_fp', 0)
        avg_det = sum(all_det)/len(all_det)*100 if all_det else 0
        avg_cov = sum(all_cov)/len(all_cov)*100 if all_cov else 0
        avg_med = sum(all_med)/len(all_med) if all_med else float('inf')
        print(f"{filt_name:<25s}  {avg_det:5.1f}%  {avg_cov:5.1f}%  {avg_med:8.1f}px  {total_fp:4d}")
    print(f"{'#'*90}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"temporal_voting_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
