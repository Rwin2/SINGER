#!/usr/bin/env python3
"""
Render annotated videos with the BEST perception config:
  - OWL-ViT v1 with rich prompts (76.3% detection)
  - EMA alpha=0.5, coast=30, thr=0.02
  - Spatial gating (80px radius, consensus init n=2)

Shows THREE tracks per frame:
  - Green crosshair = GT
  - Magenta crosshair = raw detector (rich prompts)
  - Cyan crosshair = EMA-smoothed estimate (bright=measured, dim=coasting)

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/render_best_config.py
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

FPS = 20

# ── Rich Prompts ──
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


# ── EMA Filter with Spatial Gating ──
class SpatialGateEMA:
    def __init__(self, alpha=0.5, conf_gate=0.02, max_coast=30,
                 gate_radius_px=80.0, n_init=2, init_radius_px=60.0):
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
        self.last_accepted = False  # for HUD

    def reset(self):
        self._init_buffer = []
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0
        self.initialized = False
        self.last_accepted = False

    def step(self, b, e, conf, area):
        has_meas = (area > 0 and conf >= self.conf_gate)

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
                        self.last_accepted = True
                        return int(round(self.u_ema)), int(round(self.v_ema))
                self.last_accepted = False
                return None
            else:
                dist = np.hypot(u - self.u_ema, v - self.v_ema)
                if dist > self.gate_radius_px:
                    self.last_accepted = False
                    self.steps_coasting += 1
                    if self.steps_coasting <= self.max_coast:
                        return int(round(self.u_ema)), int(round(self.v_ema))
                    return None

                self.u_ema = self.alpha * u + (1 - self.alpha) * self.u_ema
                self.v_ema = self.alpha * v + (1 - self.alpha) * self.v_ema
                self.steps_coasting = 0
                self.last_accepted = True
                return int(round(self.u_ema)), int(round(self.v_ema))
        else:
            self.last_accepted = False
            self.steps_coasting += 1
            if self.initialized and self.steps_coasting <= self.max_coast:
                return int(round(self.u_ema)), int(round(self.v_ema))
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
        return None, None, 0.0, 0, None
    best = scores.argmax()
    box = boxes[best]
    score = float(scores[best])
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = int((box[2] - box[0]) * (box[3] - box[1]))
    return cx, cy, score, area, box


# ── Drawing ──
def draw_marker(frame, x, y, color, label, sz=14, th=2):
    h, w = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, sz, th)
        cv2.putText(frame, label, (x + sz, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_box(frame, box, color, th=1):
    if box is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)


def draw_hud(frame, texts):
    for y_pos, txt in texts:
        if txt:
            cv2.putText(frame, txt, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)


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
        })
    return cases


# ── Render ──
def render_case(case, proc, mdl, dev, out_dir):
    filt = SpatialGateEMA(alpha=0.5, conf_gate=0.02, max_coast=30,
                           gate_radius_px=80.0, n_init=2)
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)

    out_path = os.path.join(out_dir, f"{case['name']}_rich_ema.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (IMG_W, IMG_H))

    stats = {"n_det": 0, "n_tp": 0, "n_fp": 0, "filt_errs": [], "det_errs": []}
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area, box = detect_owlvit_rich(
            frame_rgb, obj_name, proc, mdl, dev)
        detected = (area > 0 and conf >= 0.02)

        # Filter
        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            est = filt.step(b, e, conf, area)
        else:
            est = filt.step(0.0, 0.0, 0.0, 0)

        # GT
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # Draw GT (green)
        if gt_in and gt_u is not None:
            draw_marker(frame_bgr, gt_u, gt_v, (50, 255, 50), "GT", sz=14, th=2)

        # Draw detector (magenta)
        det_err = None
        if detected:
            draw_marker(frame_bgr, cx, cy, (255, 0, 255),
                        f"det {conf:.2f}", sz=14, th=2)
            if box is not None:
                draw_box(frame_bgr, box, (255, 0, 255), th=1)
            stats["n_det"] += 1
            if gt_in:
                stats["n_tp"] += 1
                det_err = float(np.hypot(cx - gt_u, cy - gt_v))
                stats["det_errs"].append(det_err)
            else:
                stats["n_fp"] += 1

        # Draw filter estimate (cyan)
        filt_err = None
        if est is not None:
            fu, fv = est
            if filt.steps_coasting == 0:
                draw_marker(frame_bgr, fu, fv, (0, 255, 255), "EMA", sz=16, th=2)
            else:
                draw_marker(frame_bgr, fu, fv, (0, 180, 180),
                            f"EMA~{filt.steps_coasting}", sz=12, th=1)
            if gt_in and gt_u is not None:
                filt_err = float(np.hypot(fu - gt_u, fv - gt_v))
                stats["filt_errs"].append(filt_err)

        # HUD
        status = "DETECTED" if detected else "no detection"
        if detected and not filt.last_accepted and filt.initialized:
            status = "REJECTED (gate)"
        txt1 = f"step {frame_idx}/{n_frames}  conf={conf:.3f}  {status}"
        mode = ""
        if est is not None:
            if filt.steps_coasting == 0:
                mode = "meas"
            else:
                mode = f"coast({filt.steps_coasting})"
            txt2 = f"EMA: {mode}"
        else:
            txt2 = "EMA: not init"
        parts = []
        if det_err is not None:
            parts.append(f"det={det_err:.0f}px")
        if filt_err is not None:
            parts.append(f"ema={filt_err:.0f}px")
        txt3 = "err: " + "  ".join(parts) if parts else ""

        draw_hud(frame_bgr, [(18, txt1), (34, txt2), (50, txt3)])
        writer.write(frame_bgr)
        frame_idx += 1

    cap.release()
    writer.release()

    fe = stats["filt_errs"]
    de = stats["det_errs"]
    n_gt = sum(1 for c in centroids if c.get("gt_in_frame", False))
    summary = {
        "n_frames": n_frames,
        "n_gt_visible": n_gt,
        "n_detected": stats["n_det"],
        "n_tp": stats["n_tp"],
        "n_fp": stats["n_fp"],
        "det_rate": stats["n_tp"] / max(n_gt, 1),
        "filt_mean": float(np.mean(fe)) if fe else None,
        "filt_median": float(np.median(fe)) if fe else None,
        "det_mean": float(np.mean(de)) if de else None,
        "det_median": float(np.median(de)) if de else None,
    }
    return out_path, summary


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return

    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", f"best_config_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    print("Loading OWL-ViT (rich prompts)...")
    proc, mdl, dev = load_owlvit()

    all_summaries = {}
    for case in cases:
        t1 = time.time()
        out_path, summary = render_case(case, proc, mdl, dev, out_dir)
        dt = time.time() - t1
        det_r = f"{summary['det_rate']*100:.1f}%"
        f_med = f"{summary['filt_median']:.1f}" if summary['filt_median'] else "-"
        d_med = f"{summary['det_median']:.1f}" if summary['det_median'] else "-"
        print(f"  {case['name']:<45}  det={det_r:>6}  filt_med={f_med:>7}px  "
              f"det_med={d_med:>7}px  FP={summary['n_fp']}  ({dt:.1f}s)")
        all_summaries[case["name"]] = summary

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Videos: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
