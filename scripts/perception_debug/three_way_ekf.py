#!/usr/bin/env python3
"""
3-way detector comparison: YOLO-World vs CLIPSeg vs OWL-ViT
All with EKF filtering + EMA fallback, baseline queries, rendered videos.

Shows: raw detection (dot) + EKF-filtered estimate (cross) + GT (circle).
Colors: cyan=YOLO, magenta=CLIP, orange=OWL, green=GT.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/three_way_ekf.py
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
    InvDepthEKF, IMG_W, IMG_H,
)

# Baseline queries from configs/scenes/flightroom_ssv_exp.yml
BASELINE_QUERIES = [
    "green clock",
    "green and pink leafblower",
    "yellow handheld cordless drill on two boxes",
]


# ── Detectors ──

def load_yoloworld():
    from ultralytics import YOLO
    model_path = os.path.join(WORKSPACE, "yolov8l-worldv2.pt")
    model = YOLO(model_path)
    model.set_classes(BASELINE_QUERIES)
    return model


def detect_yoloworld(model, frame_rgb, target_idx, conf_thresh=0.05):
    results = model.predict(frame_rgb, conf=conf_thresh, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None, 0.0, 0
    boxes = results[0].boxes
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


def load_clipseg():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


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
    if int(mask.sum()) < min_blob:
        return None, None, confidence, 0
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
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


def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit(frame_rgb, query, proc, mdl, dev, threshold=0.02):
    img = Image.fromarray(frame_rgb)
    inputs = proc(text=[[query]], images=img, return_tensors="pt")
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


# ── Helpers ──

def px_to_be(cx, cy):
    """Pixel (cx,cy) -> normalized bearing, elevation in [-1,1]."""
    b = 2.0 * cx / IMG_W - 1.0
    e = 2.0 * cy / IMG_H - 1.0
    return b, e


def get_query_idx(obj_name):
    obj_lower = obj_name.lower()
    for i, q in enumerate(BASELINE_QUERIES):
        if q in obj_lower or obj_lower in q:
            return i, q
        q_words = set(q.split())
        obj_words = set(obj_lower.replace("_", " ").split())
        if len(q_words & obj_words) >= 2:
            return i, q
    return None, obj_name


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
            "Xro": npz["Xro"],  # (10, N_steps) drone state
        })
    return cases


def make_ekf(conf_gate=0.02):
    """Create an EKF configured for low-conf detectors like OWL-ViT / YOLO."""
    return InvDepthEKF(
        conf_gate=conf_gate,
        process_noise_be=0.005,
        process_noise_rho=0.001,
        meas_noise=0.03,
        init_rho=0.3,
        init_sigma_rho=0.3,
        innovation_gate=None,
        adaptive_R=False,
        temporal_decay_gate=False,
    )


# ── Main loop ──

def process_case(case, yolo_model, clip_proc, clip_mdl, clip_dev,
                 owl_proc, owl_mdl, owl_dev, out_dir):
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    Xro = case["Xro"]
    n_frames = min(len(centroids), Xro.shape[1])

    target_idx, query = get_query_idx(obj_name)

    # One EKF per detector
    ekf_yolo = make_ekf(conf_gate=0.05)
    ekf_clip = make_ekf(conf_gate=0.50)  # CLIPSeg confidence is sigmoid max, ~0.5-1.0
    ekf_owl = make_ekf(conf_gate=0.02)

    # Video
    out_path = os.path.join(out_dir, f"{case['name']}_3way.mp4")
    tmp_path = out_path.replace(".mp4", "_tmp.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (IMG_W, IMG_H))

    # Per-detector stats
    stats = {
        "yolo": {"raw_errs": [], "ekf_errs": [], "n_tp": 0, "n_fp": 0,
                 "n_gt_vis": 0, "n_det": 0, "n_ekf": 0, "lats": []},
        "clip": {"raw_errs": [], "ekf_errs": [], "n_tp": 0, "n_fp": 0,
                 "n_gt_vis": 0, "n_det": 0, "n_ekf": 0, "lats": []},
        "owl":  {"raw_errs": [], "ekf_errs": [], "n_tp": 0, "n_fp": 0,
                 "n_gt_vis": 0, "n_det": 0, "n_ekf": 0, "lats": []},
    }

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vis = frame_bgr.copy()

        xcr = Xro[:, frame_idx]  # drone state

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # --- Detect with all 3 ---
        t0 = time.time()
        ycx, ycy, yconf, yarea = detect_yoloworld(yolo_model, frame_rgb, target_idx)
        stats["yolo"]["lats"].append(time.time() - t0)

        t0 = time.time()
        ccx, ccy, cconf, carea = detect_clipseg(
            frame_rgb, query, clip_proc, clip_mdl, clip_dev)
        stats["clip"]["lats"].append(time.time() - t0)

        t0 = time.time()
        ocx, ocy, oconf, oarea = detect_owlvit(
            frame_rgb, query, owl_proc, owl_mdl, owl_dev)
        stats["owl"]["lats"].append(time.time() - t0)

        # --- EKF update for each ---
        detections = {
            "yolo": (ycx, ycy, yconf, yarea, 0.05),
            "clip": (ccx, ccy, cconf, carea, 0.50),
            "owl":  (ocx, ocy, oconf, oarea, 0.02),
        }
        ekfs = {"yolo": ekf_yolo, "clip": ekf_clip, "owl": ekf_owl}
        ekf_results = {}

        for key, (dcx, dcy, dconf, darea, _gate) in detections.items():
            detected = darea is not None and darea > 0
            s = stats[key]
            if gt_in:
                s["n_gt_vis"] += 1
                if detected and gt_u is not None:
                    s["raw_errs"].append(float(np.hypot(dcx - gt_u, dcy - gt_v)))
                if detected:
                    s["n_tp"] += 1
            if detected and not gt_in:
                s["n_fp"] += 1
            if detected:
                s["n_det"] += 1

            # Feed to EKF
            if detected:
                b, e = px_to_be(dcx, dcy)
                ekf_out = ekfs[key].step(xcr, b, e, dconf, darea)
            else:
                ekf_out = ekfs[key].step(xcr, 0, 0, 0.0, 0)

            if ekf_out is not None:
                eu, ev, _sigma = ekf_out
                ekf_results[key] = (eu, ev)
                if gt_in and gt_u is not None:
                    s["ekf_errs"].append(float(np.hypot(eu - gt_u, ev - gt_v)))
                    s["n_ekf"] += 1
            else:
                ekf_results[key] = None

        # --- Render ---
        # GT
        if gt_in and gt_u is not None:
            cv2.circle(vis, (int(gt_u), int(gt_v)), 10, (0, 255, 0), 2)

        # Colors: YOLO=cyan, CLIP=magenta, OWL=orange
        colors = {"yolo": (255, 255, 0), "clip": (255, 0, 255), "owl": (0, 140, 255)}
        labels = {"yolo": "Y", "clip": "C", "owl": "O"}
        det_data = {"yolo": (ycx, ycy, yconf, yarea),
                    "clip": (ccx, ccy, cconf, carea),
                    "owl":  (ocx, ocy, oconf, oarea)}

        y_offset = 0
        for key in ["yolo", "clip", "owl"]:
            col = colors[key]
            lbl = labels[key]
            dcx, dcy, dconf, darea = det_data[key]
            detected = darea is not None and darea > 0

            # Raw detection dot
            if detected:
                cv2.circle(vis, (int(dcx), int(dcy)), 3, col, -1)

            # EKF estimate cross
            ekf_pos = ekf_results.get(key)
            if ekf_pos is not None:
                eu, ev = ekf_pos
                cv2.drawMarker(vis, (int(eu), int(ev)), col,
                              cv2.MARKER_CROSS, 12, 2)
                txt = f"{lbl}:{dconf:.2f}" if detected else f"{lbl}:coast"
                cv2.putText(vis, txt, (int(eu)+10, int(ev)-3+y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            y_offset += 12

        # Info
        cv2.putText(vis, f'Q: "{query}"', (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(vis, f"f{frame_idx}/{n_frames}", (5, IMG_H - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Legend
        ly = 30
        for i, (key, name) in enumerate([("gt", "GT"), ("yolo", "YOLO"),
                                          ("clip", "CLIP"), ("owl", "OWL")]):
            x = 5 + i * 65
            c = (0, 255, 0) if key == "gt" else colors[key]
            cv2.circle(vis, (x, ly), 4, c, -1)
            cv2.putText(vis, name, (x+8, ly+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    # H.264 convert
    os.system(f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p "{out_path}" 2>/dev/null')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, out_path)

    # Summarize
    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    results = {"query": query, "n_frames": frame_idx}
    for key in ["yolo", "clip", "owl"]:
        s = stats[key]
        lat = np.array(s["lats"])
        results[key] = {
            "raw": _s(s["raw_errs"]),
            "ekf": _s(s["ekf_errs"]),
            "n_gt_vis": s["n_gt_vis"],
            "n_det": s["n_det"],
            "n_tp": s["n_tp"],
            "n_fp": s["n_fp"],
            "n_ekf": s["n_ekf"],
            "det_rate": s["n_tp"] / max(s["n_gt_vis"], 1),
            "latency_ms": float(np.mean(lat) * 1000),
        }
    results["video"] = out_path
    return results


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found"); return
    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases")
    print(f"Queries: {BASELINE_QUERIES}\n")

    print("Loading YOLO-World v2 (large)...")
    yolo_model = load_yoloworld()
    print("Loading CLIPSeg...")
    clip_proc, clip_mdl, clip_dev = load_clipseg()
    print("Loading OWL-ViT...")
    owl_proc, owl_mdl, owl_dev = load_owlvit()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", f"three_way_{ts}")
    os.makedirs(vis_dir, exist_ok=True)

    all_results = {}
    for case in cases:
        _, query = get_query_idx(case["object"])
        print(f"\n{'='*80}")
        print(f"CASE: {case['name']}  (query: \"{query}\")")
        print(f"{'='*80}")

        res = process_case(case, yolo_model, clip_proc, clip_mdl, clip_dev,
                           owl_proc, owl_mdl, owl_dev, vis_dir)

        for key, label in [("yolo", "YOLO+EKF"), ("clip", "CLIP+EKF"), ("owl", "OWL+EKF")]:
            r = res[key]
            det_pct = f"{r['det_rate']*100:.1f}%"
            r_med = f"{r['raw']['median']:.1f}" if r['raw'].get('median') else "-"
            e_med = f"{r['ekf']['median']:.1f}" if r['ekf'].get('median') else "-"
            print(f"  {label:<12}  det={det_pct:>6} ({r['n_tp']}/{r['n_gt_vis']})  "
                  f"raw={r_med:>7}px  ekf={e_med:>7}px  "
                  f"FP={r['n_fp']:>3}  n_det={r['n_det']:>4}  "
                  f"lat={r['latency_ms']:.0f}ms")

        all_results[case["name"]] = res

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*90}")
    print(f"AGGREGATE (9 cases)")
    print(f"{'#'*90}")
    print(f"{'Detector':<12}  {'det%':>6}  {'raw_med':>8}  {'ekf_med':>8}  "
          f"{'FP':>4}  {'n_det':>6}  {'lat':>6}")
    print(f"{'-'*70}")

    for key, label in [("yolo", "YOLO+EKF"), ("clip", "CLIP+EKF"), ("owl", "OWL+EKF")]:
        dets = []; raw_m = []; ekf_m = []; fps = 0; n_det_total = 0; lats = []
        for _, v in all_results.items():
            r = v[key]
            if r["n_gt_vis"] > 0:
                dets.append(r["det_rate"])
            m = r.get("raw", {}).get("median")
            if m is not None: raw_m.append(m)
            m = r.get("ekf", {}).get("median")
            if m is not None: ekf_m.append(m)
            fps += r["n_fp"]
            n_det_total += r["n_det"]
            lats.append(r["latency_ms"])

        avg_det = sum(dets)/len(dets)*100 if dets else 0
        avg_raw = sum(raw_m)/len(raw_m) if raw_m else float('inf')
        avg_ekf = sum(ekf_m)/len(ekf_m) if ekf_m else float('inf')
        avg_lat = sum(lats)/len(lats) if lats else 0
        print(f"{label:<12}  {avg_det:5.1f}%  {avg_raw:7.1f}px  {avg_ekf:7.1f}px  "
              f"{fps:4d}  {n_det_total:6d}  {avg_lat:5.0f}ms")

    print(f"{'#'*90}")
    print(f"\nVideos: {vis_dir}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"three_way_ekf_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
