#!/usr/bin/env python3
"""
Render annotated videos with the winning FP-suppressed config:
  - OWL-ViT v1 with rich prompts (76.3% detection)
  - CLIPSeg bbox_ratio verification (>= 2.0 → 95% FP reduction)
  - EMA alpha=0.5, coast=30

Shows FOUR tracks per frame:
  - Green crosshair = GT
  - Magenta crosshair = raw OWL-ViT detection
  - Red X = rejected detection (CLIPSeg verify failed)
  - Cyan crosshair = verified EMA estimate (bright=measured, dim=coasting)

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/render_verified.py
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


class EMAFilter:
    def __init__(self, alpha=0.5, max_coast=30):
        self.alpha = alpha
        self.max_coast = max_coast
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0

    def reset(self):
        self.u_ema = None
        self.v_ema = None
        self.steps_coasting = 0

    def step(self, cx, cy, accepted):
        if accepted:
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
            return None


def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def load_clipseg():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit_rich(frame_rgb, obj_name, proc, mdl, dev):
    img = Image.fromarray(frame_rgb)
    queries = [_get_prompts(obj_name)]
    inputs = proc(text=queries, images=img, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mdl(**inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to(dev)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.02)
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


def verify_clipseg(frame_rgb, obj_name, box, cseg_proc, cseg_mdl, dev, ratio_thresh=2.0):
    """Returns (passed, ratio)."""
    if box is None:
        return False, 0.0
    img = Image.fromarray(frame_rgb)
    inputs = cseg_proc(images=img, text=obj_name, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        logits = cseg_mdl(**inputs).logits
    logits_np = logits.cpu().squeeze().numpy().astype(np.float32)
    if logits_np.shape != (IMG_H, IMG_W):
        logits_np = cv2.resize(logits_np, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    prob = 1.0 / (1.0 + np.exp(-logits_np))

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_W, x2), min(IMG_H, y2)
    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    bbox_mean = prob[y1:y2, x1:x2].mean()
    outside = prob.copy()
    outside[y1:y2, x1:x2] = 0
    n_outside = IMG_W * IMG_H - (x2 - x1) * (y2 - y1)
    outside_mean = outside.sum() / max(n_outside, 1)
    if outside_mean < 0.001:
        ratio = float(bbox_mean) * 100
    else:
        ratio = float(bbox_mean / outside_mean)

    return ratio >= ratio_thresh, ratio


def draw_marker(frame, x, y, color, label, sz=14, th=2):
    h, w = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, sz, th)
        cv2.putText(frame, label, (x + sz, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_x(frame, x, y, color, label, sz=10, th=2):
    """Draw an X marker for rejected detections."""
    h, w = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_TILTED_CROSS, sz, th)
        cv2.putText(frame, label, (x + sz, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


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


def render_case(case, owl_proc, owl_mdl, cseg_proc, cseg_mdl, dev, out_dir):
    filt = EMAFilter(alpha=0.5, max_coast=30)
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)

    out_path = os.path.join(out_dir, f"{case['name']}_verified.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (IMG_W, IMG_H))

    stats = {"n_det": 0, "n_tp": 0, "n_fp": 0, "n_verified": 0,
             "verified_tp": 0, "verified_fp": 0, "filt_errs": [], "det_errs": []}
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area, box = detect_owlvit_rich(
            frame_rgb, obj_name, owl_proc, owl_mdl, dev)
        detected = (area > 0 and conf >= 0.02)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        verified = False
        ratio = 0.0
        if detected:
            stats["n_det"] += 1
            verified, ratio = verify_clipseg(
                frame_rgb, obj_name, box, cseg_proc, cseg_mdl, dev, ratio_thresh=2.0)
            if gt_in:
                stats["n_tp"] += 1
            else:
                stats["n_fp"] += 1
            if verified:
                stats["n_verified"] += 1
                if gt_in:
                    stats["verified_tp"] += 1
                else:
                    stats["verified_fp"] += 1

        # Filter with verified detections only
        est = filt.step(cx if verified else 0, cy if verified else 0, verified)

        # Draw GT (green)
        if gt_in and gt_u is not None:
            draw_marker(frame_bgr, gt_u, gt_v, (50, 255, 50), "GT", sz=14, th=2)

        # Draw detector
        det_err = None
        if detected:
            if verified:
                draw_marker(frame_bgr, cx, cy, (255, 0, 255),
                            f"det {conf:.2f} r={ratio:.1f}", sz=14, th=2)
                if box is not None:
                    draw_box(frame_bgr, box, (255, 0, 255), th=1)
            else:
                draw_x(frame_bgr, cx, cy, (0, 0, 255),
                       f"REJ r={ratio:.1f}", sz=10, th=2)
                if box is not None:
                    draw_box(frame_bgr, box, (0, 0, 255), th=1)
            if gt_in and gt_u is not None:
                det_err = float(np.hypot(cx - gt_u, cy - gt_v))
                stats["det_errs"].append(det_err)

        # Draw EMA (cyan)
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
        status = "VERIFIED" if verified else ("REJECTED" if detected else "no det")
        txt1 = f"step {frame_idx}/{n_frames}  conf={conf:.3f}  {status}"
        txt2 = f"EMA: {'meas' if filt.steps_coasting == 0 and est else 'coast('+str(filt.steps_coasting)+')' if est else 'init'}"
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
        "n_verified": stats["n_verified"],
        "verified_tp": stats["verified_tp"],
        "verified_fp": stats["verified_fp"],
        "det_rate": stats["n_tp"] / max(n_gt, 1),
        "verified_det_rate": stats["verified_tp"] / max(n_gt, 1),
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
                           "visualizations", f"verified_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    print("Loading OWL-ViT...")
    owl_proc, owl_mdl, dev = load_owlvit()
    print("Loading CLIPSeg...")
    cseg_proc, cseg_mdl, _ = load_clipseg()

    all_summaries = {}
    for case in cases:
        t1 = time.time()
        out_path, summary = render_case(
            case, owl_proc, owl_mdl, cseg_proc, cseg_mdl, dev, out_dir)
        dt = time.time() - t1
        vdet = f"{summary['verified_det_rate']*100:.1f}%"
        f_med = f"{summary['filt_median']:.1f}" if summary['filt_median'] else "-"
        print(f"  {case['name']:<45}  vdet={vdet:>6}  filt_med={f_med:>7}px  "
              f"vFP={summary['verified_fp']}  rawFP={summary['n_fp']}  ({dt:.1f}s)")
        all_summaries[case["name"]] = summary

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Videos: {out_dir}")


if __name__ == "__main__":
    main()
