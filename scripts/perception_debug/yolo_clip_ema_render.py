#!/usr/bin/env python3
"""
YOLO-World + EMA  vs  CLIPSeg + EMA  — rendered detection vs GT.

Uses baseline queries from configs/scenes/flightroom_ssv_exp.yml.
Renders annotated H.264 videos: green=GT, cyan=YOLO, magenta=CLIPSeg.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/yolo_clip_ema_render.py
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

# Baseline queries from configs/scenes/flightroom_ssv_exp.yml
BASELINE_QUERIES = [
    "green clock",
    "green and pink leafblower",
    "yellow handheld cordless drill on two boxes",
]


# ── EMA Filter ──

class EMAFilter:
    def __init__(self, alpha=0.5, max_coast=30):
        self.alpha = alpha
        self.max_coast = max_coast
        self.u = None
        self.v = None
        self.coast = 0

    def reset(self):
        self.u = None
        self.v = None
        self.coast = 0

    def step(self, cx, cy, detected):
        if detected:
            if self.u is None:
                self.u = cx
                self.v = cy
            else:
                self.u = self.alpha * cx + (1 - self.alpha) * self.u
                self.v = self.alpha * cy + (1 - self.alpha) * self.v
            self.coast = 0
            return self.u, self.v
        else:
            self.coast += 1
            if self.u is not None and self.coast <= self.max_coast:
                return self.u, self.v
            if self.coast > self.max_coast:
                self.u = None
                self.v = None
            return None, None


# ── Detectors ──

def load_yoloworld():
    from ultralytics import YOLO
    model_path = os.path.join(WORKSPACE, "yolov8l-worldv2.pt")
    if not os.path.exists(model_path):
        model = YOLO("yolov8l-worldv2.pt")
    else:
        model = YOLO(model_path)
    model.set_classes(BASELINE_QUERIES)
    return model


def detect_yoloworld(model, frame_rgb, obj_name, conf_thresh=0.05):
    results = model.predict(frame_rgb, conf=conf_thresh, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None, 0.0, 0

    boxes = results[0].boxes
    target_idx = _match_query_idx(obj_name)

    best_score = 0.0
    best_box = None
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        score = float(boxes.conf[i])
        if target_idx is not None and cls_id != target_idx:
            continue
        if score > best_score:
            best_score = score
            best_box = boxes.xyxy[i].cpu().numpy()

    if best_box is None:
        return None, None, 0.0, 0
    cx = float((best_box[0] + best_box[2]) / 2)
    cy = float((best_box[1] + best_box[3]) / 2)
    area = int((best_box[2] - best_box[0]) * (best_box[3] - best_box[1]))
    return cx, cy, best_score, area


def _match_query_idx(obj_name):
    obj_lower = obj_name.lower()
    for i, q in enumerate(BASELINE_QUERIES):
        if q in obj_lower or obj_lower in q:
            return i
        q_words = set(q.split())
        obj_words = set(obj_lower.replace("_", " ").split())
        if len(q_words & obj_words) >= 2:
            return i
    return None


def load_clipseg():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = mdl.to(device).eval()
    return proc, mdl, device


def detect_clipseg(frame_rgb, query, proc, mdl, dev,
                   sigmoid_thresh=0.5, min_blob=30):
    img = Image.fromarray(frame_rgb)
    inputs = proc(images=img, text=query, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
    logits_np = logits.cpu().squeeze().numpy().astype(np.float32)
    if logits_np.shape != (IMG_H, IMG_W):
        logits_np = cv2.resize(logits_np, (IMG_W, IMG_H),
                               interpolation=cv2.INTER_LINEAR)
    prob = 1.0 / (1.0 + np.exp(-logits_np))
    confidence = float(prob.max())
    mask = prob > sigmoid_thresh
    n_above = int(mask.sum())
    if n_above < min_blob:
        return None, None, confidence, 0

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)
    if n_labels <= 1:
        return None, None, confidence, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_area = int(areas[best_label - 1])
    if blob_area < min_blob:
        return None, None, confidence, 0

    blob_mask = labels == best_label
    ys, xs = np.where(blob_mask)
    w = prob[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))
    return cx, cy, confidence, blob_area


# ── Data ──

def get_baseline_query(obj_name):
    obj_lower = obj_name.lower()
    for q in BASELINE_QUERIES:
        if q in obj_lower or obj_lower in q:
            return q
        q_words = set(q.split())
        obj_words = set(obj_lower.replace("_", " ").split())
        if len(q_words & obj_words) >= 2:
            return q
    return obj_name


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
        cases.append({
            "name": case_dir,
            "video_path": rgb_vids[0],
            "centroids": log_data["centroids"],
            "object": log_data.get("object", case_dir),
        })
    return cases


# ── Main ──

def process_case(case, yolo_model, clip_proc, clip_mdl, clip_dev, out_dir):
    """Run both detectors + EMA on one case, render video, return metrics."""
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    query = get_baseline_query(obj_name)
    n_frames = len(centroids)

    yolo_ema = EMAFilter(alpha=0.5, max_coast=30)
    clip_ema = EMAFilter(alpha=0.5, max_coast=30)

    # Video writer
    out_path = os.path.join(out_dir, f"{case['name']}_yolo_clip.mp4")
    tmp_path = out_path.replace(".mp4", "_tmp.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (IMG_W, IMG_H))

    # Metrics
    yolo_stats = {"det_errs": [], "ema_errs": [], "n_tp": 0, "n_fp": 0,
                  "n_gt_vis": 0, "n_det": 0, "latencies": []}
    clip_stats = {"det_errs": [], "ema_errs": [], "n_tp": 0, "n_fp": 0,
                  "n_gt_vis": 0, "n_det": 0, "latencies": []}

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vis = frame_bgr.copy()

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # -- YOLO-World --
        t0 = time.time()
        ycx, ycy, yconf, yarea = detect_yoloworld(yolo_model, frame_rgb, obj_name)
        yolo_stats["latencies"].append(time.time() - t0)
        y_detected = (yarea > 0)

        # -- CLIPSeg --
        t0 = time.time()
        ccx, ccy, cconf, carea = detect_clipseg(
            frame_rgb, query, clip_proc, clip_mdl, clip_dev)
        clip_stats["latencies"].append(time.time() - t0)
        c_detected = (carea > 0)

        # -- EMA --
        yu, yv = yolo_ema.step(ycx if y_detected else 0,
                               ycy if y_detected else 0, y_detected)
        cu, cv_ = clip_ema.step(ccx if c_detected else 0,
                                ccy if c_detected else 0, c_detected)

        # -- Metrics --
        if gt_in:
            yolo_stats["n_gt_vis"] += 1
            clip_stats["n_gt_vis"] += 1
            if y_detected:
                yolo_stats["n_tp"] += 1
                if gt_u is not None:
                    yolo_stats["det_errs"].append(float(np.hypot(ycx - gt_u, ycy - gt_v)))
            if c_detected:
                clip_stats["n_tp"] += 1
                if gt_u is not None:
                    clip_stats["det_errs"].append(float(np.hypot(ccx - gt_u, ccy - gt_v)))
            if gt_u is not None:
                if yu is not None:
                    yolo_stats["ema_errs"].append(float(np.hypot(yu - gt_u, yv - gt_v)))
                if cu is not None:
                    clip_stats["ema_errs"].append(float(np.hypot(cu - gt_u, cv_ - gt_v)))
        if y_detected and not gt_in:
            yolo_stats["n_fp"] += 1
        if c_detected and not gt_in:
            clip_stats["n_fp"] += 1
        if y_detected:
            yolo_stats["n_det"] += 1
        if c_detected:
            clip_stats["n_det"] += 1

        # -- Render --
        # GT (green circle)
        if gt_in and gt_u is not None:
            cv2.circle(vis, (int(gt_u), int(gt_v)), 10, (0, 255, 0), 2)
            cv2.putText(vis, "GT", (int(gt_u)+12, int(gt_v)-2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # YOLO raw detection (cyan dot)
        if y_detected:
            cv2.circle(vis, (int(ycx), int(ycy)), 4, (255, 255, 0), -1)
        # YOLO EMA (cyan cross)
        if yu is not None:
            x, y = int(yu), int(yv)
            cv2.drawMarker(vis, (x, y), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(vis, f"Y:{yconf:.2f}" if y_detected else "Y:coast",
                       (x+10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # CLIPSeg raw detection (magenta dot)
        if c_detected:
            cv2.circle(vis, (int(ccx), int(ccy)), 4, (255, 0, 255), -1)
        # CLIPSeg EMA (magenta cross)
        if cu is not None:
            x, y = int(cu), int(cv_)
            cv2.drawMarker(vis, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(vis, f"C:{cconf:.2f}" if c_detected else "C:coast",
                       (x+10, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        # Info bar
        cv2.putText(vis, f"Q: \"{query}\"", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, f"f{frame_idx}/{n_frames}", (5, IMG_H - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Legend
        ly = 30
        cv2.circle(vis, (5, ly), 4, (0, 255, 0), -1)
        cv2.putText(vis, "GT", (15, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.drawMarker(vis, (50, ly), (255, 255, 0), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(vis, "YOLO+EMA", (60, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cv2.drawMarker(vis, (140, ly), (255, 0, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(vis, "CLIP+EMA", (150, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    # Convert to H.264
    os.system(f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p "{out_path}" 2>/dev/null')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, out_path)

    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    def _metrics(stats):
        lat = np.array(stats["latencies"])
        return {
            "det": _s(stats["det_errs"]),
            "ema": _s(stats["ema_errs"]),
            "n_gt_vis": stats["n_gt_vis"],
            "n_tp": stats["n_tp"],
            "n_fp": stats["n_fp"],
            "n_det": stats["n_det"],
            "det_rate": stats["n_tp"] / max(stats["n_gt_vis"], 1),
            "latency_ms": float(np.mean(lat) * 1000),
        }

    return {
        "query": query,
        "n_frames": frame_idx,
        "yolo_world": _metrics(yolo_stats),
        "clipseg": _metrics(clip_stats),
        "video": out_path,
    }


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return
    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases")
    print(f"Queries: {BASELINE_QUERIES}\n")

    print("Loading YOLO-World v2 (large)...")
    yolo_model = load_yoloworld()
    print("Loading CLIPSeg...")
    clip_proc, clip_mdl, clip_dev = load_clipseg()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", f"yolo_clip_ema_{ts}")
    os.makedirs(vis_dir, exist_ok=True)

    all_results = {}
    for case in cases:
        query = get_baseline_query(case["object"])
        print(f"\n{'='*80}")
        print(f"CASE: {case['name']}  (query: \"{query}\")")
        print(f"{'='*80}")

        res = process_case(case, yolo_model, clip_proc, clip_mdl, clip_dev, vis_dir)

        for det_name, key in [("YOLO+EMA", "yolo_world"), ("CLIP+EMA", "clipseg")]:
            r = res[key]
            det_pct = f"{r['det_rate']*100:.1f}%"
            d_med = f"{r['det']['median']:.1f}" if r['det'].get('median') else "-"
            e_med = f"{r['ema']['median']:.1f}" if r['ema'].get('median') else "-"
            print(f"  {det_name:<12}  det={det_pct:>6}  raw_med={d_med:>7}px  "
                  f"ema_med={e_med:>7}px  FP={r['n_fp']:>3}  "
                  f"lat={r['latency_ms']:.0f}ms")

        all_results[case["name"]] = res

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*90}")
    print(f"AGGREGATE")
    print(f"{'#'*90}")
    print(f"{'Detector':<12}  {'det%':>6}  {'raw_med':>8}  {'ema_med':>8}  {'FP':>4}  {'lat':>6}")
    print(f"{'-'*60}")

    for key, label in [("yolo_world", "YOLO+EMA"), ("clipseg", "CLIP+EMA")]:
        dets = []; raw_meds = []; ema_meds = []; fps = 0; lats = []
        for _, v in all_results.items():
            r = v[key]
            if r["n_gt_vis"] > 0:
                dets.append(r["det_rate"])
            m = r.get("det", {}).get("median")
            if m is not None:
                raw_meds.append(m)
            m = r.get("ema", {}).get("median")
            if m is not None:
                ema_meds.append(m)
            fps += r["n_fp"]
            lats.append(r["latency_ms"])

        avg_det = sum(dets)/len(dets)*100 if dets else 0
        avg_raw = sum(raw_meds)/len(raw_meds) if raw_meds else float('inf')
        avg_ema = sum(ema_meds)/len(ema_meds) if ema_meds else float('inf')
        avg_lat = sum(lats)/len(lats) if lats else 0
        print(f"{label:<12}  {avg_det:5.1f}%  {avg_raw:7.1f}px  {avg_ema:7.1f}px  "
              f"{fps:4d}  {avg_lat:5.0f}ms")

    print(f"{'#'*90}")
    print(f"\nVideos: {vis_dir}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"yolo_clip_ema_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
