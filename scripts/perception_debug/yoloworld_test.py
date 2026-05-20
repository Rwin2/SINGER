#!/usr/bin/env python3
"""
YOLO-World vs OWL-ViT comparison on failed validation cases.

Tests YOLO-World (open-vocabulary, real-time) against OWL-ViT baseline
using the actual scene queries from flightroom_ssv_exp.yml.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/yoloworld_test.py
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


def load_yoloworld(model_size="l"):
    from ultralytics import YOLO
    model_path = os.path.join(WORKSPACE, f"yolov8{model_size}-worldv2.pt")
    if not os.path.exists(model_path):
        print(f"Downloading yolov8{model_size}-worldv2...")
        model = YOLO(f"yolov8{model_size}-worldv2.pt")
        # Move to workspace
        import shutil
        if os.path.exists(f"yolov8{model_size}-worldv2.pt"):
            shutil.move(f"yolov8{model_size}-worldv2.pt", model_path)
    else:
        model = YOLO(model_path)
    model.set_classes(BASELINE_QUERIES)
    return model


def detect_yoloworld(model, frame_rgb, obj_name, conf_thresh=0.05):
    """Run YOLO-World detection, return best box matching obj_name."""
    results = model.predict(frame_rgb, conf=conf_thresh, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None, 0.0, 0

    boxes = results[0].boxes
    # Find which class index matches this object
    target_idx = None
    obj_lower = obj_name.lower()
    for i, q in enumerate(BASELINE_QUERIES):
        if q in obj_lower or obj_lower in q:
            target_idx = i
            break
        # Partial match
        for word in q.split():
            if word in obj_lower:
                target_idx = i
                break
        if target_idx is not None:
            break

    # Filter to matching class, or use all if no match found
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


def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
    """OWL-ViT detection with baseline query (single prompt, no enrichment)."""
    img = Image.fromarray(frame_rgb)
    # Use the baseline query for this object
    query = obj_name  # direct from scene config
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


def get_baseline_query(obj_name):
    """Map object name to its baseline query from scene config."""
    obj_lower = obj_name.lower()
    for q in BASELINE_QUERIES:
        if q in obj_lower or obj_lower in q:
            return q
        # Check partial overlap
        q_words = set(q.split())
        obj_words = set(obj_lower.replace("_", " ").split())
        if len(q_words & obj_words) >= 2:
            return q
    # Fallback: return the object name itself
    return obj_name


def run_detector(case, detect_fn, detector_name):
    """Run a detector on all frames, return metrics."""
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)

    det_errs = []
    all_confs = []
    n_det = 0; n_tp = 0; n_fp = 0; n_gt_vis = 0
    frame_idx = 0
    latencies = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        cx, cy, conf, area = detect_fn(frame_rgb, obj_name)
        latencies.append(time.time() - t0)

        detected = (area > 0)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        if gt_in:
            n_gt_vis += 1
            if detected:
                n_tp += 1
                if gt_u is not None:
                    det_errs.append(float(np.hypot(cx - gt_u, cy - gt_v)))
        if detected and not gt_in:
            n_fp += 1
        if detected:
            n_det += 1
            all_confs.append(conf)

        frame_idx += 1

    cap.release()

    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    lat_arr = np.array(latencies)
    return {
        "detector": detector_name,
        "det": _s(det_errs),
        "conf": _s(all_confs),
        "n_frames": frame_idx, "n_gt_vis": n_gt_vis,
        "n_det": n_det, "n_tp": n_tp, "n_fp": n_fp,
        "det_rate": n_tp / max(n_gt_vis, 1),
        "fp_rate": n_fp / max(frame_idx, 1),
        "latency_ms": {
            "mean": float(np.mean(lat_arr) * 1000),
            "median": float(np.median(lat_arr) * 1000),
            "p90": float(np.percentile(lat_arr, 90) * 1000),
        },
    }


def render_comparison_video(case, yolo_model, owlvit_proc, owlvit_mdl, owlvit_dev, out_dir):
    """Render side-by-side annotated video (H.264)."""
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)
    query = get_baseline_query(obj_name)

    out_path = os.path.join(out_dir, f"{case['name']}_comparison.mp4")
    tmp_path = out_path.replace(".mp4", "_tmp.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (IMG_W, IMG_H))

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vis = frame_bgr.copy()

        # GT
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)
        if gt_in and gt_u is not None:
            cv2.circle(vis, (int(gt_u), int(gt_v)), 8, (0, 255, 0), 2)  # green = GT

        # YOLO-World
        ycx, ycy, yconf, yarea = detect_yoloworld(yolo_model, frame_rgb, obj_name)
        if yarea > 0:
            color = (255, 200, 0)  # cyan-ish = YOLO
            cv2.circle(vis, (int(ycx), int(ycy)), 6, color, -1)
            cv2.putText(vis, f"YOLO:{yconf:.2f}", (int(ycx)+10, int(ycy)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # OWL-ViT
        ocx, ocy, oconf, oarea = detect_owlvit(frame_rgb, query, owlvit_proc, owlvit_mdl, owlvit_dev)
        if oarea > 0:
            color = (0, 100, 255)  # orange = OWL-ViT
            cv2.circle(vis, (int(ocx), int(ocy)), 6, color, -1)
            cv2.putText(vis, f"OWL:{oconf:.2f}", (int(ocx)+10, int(ocy)+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Info
        cv2.putText(vis, f"Q: {query}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, f"f{frame_idx}", (5, IMG_H - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        legend_y = 30
        cv2.circle(vis, (5, legend_y), 4, (0, 255, 0), -1)
        cv2.putText(vis, "GT", (15, legend_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.circle(vis, (45, legend_y), 4, (255, 200, 0), -1)
        cv2.putText(vis, "YOLO", (55, legend_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
        cv2.circle(vis, (100, legend_y), 4, (0, 100, 255), -1)
        cv2.putText(vis, "OWL", (110, legend_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    # Convert to H.264 for VS Code playback
    os.system(f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p "{out_path}" 2>/dev/null')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, out_path)

    return out_path


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return
    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases from {eval_dir}")
    print(f"Baseline queries: {BASELINE_QUERIES}\n")

    # Load models
    print("Loading YOLO-World v2 (large)...")
    yolo_model = load_yoloworld("l")
    print("Loading OWL-ViT...")
    owlvit_proc, owlvit_mdl, owlvit_dev = load_owlvit()

    # Run detectors on all cases
    all_results = {}
    for case in cases:
        query = get_baseline_query(case["object"])
        print(f"\n{'='*80}")
        print(f"CASE: {case['name']}  (query: '{query}')")
        print(f"{'='*80}")

        # YOLO-World
        t1 = time.time()
        yolo_res = run_detector(
            case,
            lambda rgb, obj: detect_yoloworld(yolo_model, rgb, obj),
            "yolo_world_v2_l"
        )
        yt = time.time() - t1

        # OWL-ViT with baseline query
        t1 = time.time()
        owlvit_res = run_detector(
            case,
            lambda rgb, obj: detect_owlvit(rgb, get_baseline_query(obj),
                                           owlvit_proc, owlvit_mdl, owlvit_dev),
            "owlvit_baseline"
        )
        ot = time.time() - t1

        # Print comparison
        for det_name, res, dt in [("YOLO-World", yolo_res, yt), ("OWL-ViT", owlvit_res, ot)]:
            d_med = f"{res['det']['median']:.1f}" if res['det'].get('median') else "-"
            det_pct = f"{res['det_rate']*100:.1f}%"
            conf_med = f"{res['conf']['median']:.3f}" if res['conf'].get('median') else "-"
            lat = f"{res['latency_ms']['mean']:.0f}ms"
            print(f"  {det_name:<15}  det={det_pct:>6}  med={d_med:>7}px  "
                  f"FP={res['n_fp']:>3}  conf={conf_med:>6}  lat={lat:>6}  ({dt:.1f}s)")

        all_results[case["name"]] = {
            "query": query,
            "yolo_world": yolo_res,
            "owlvit_baseline": owlvit_res,
        }

    elapsed = time.time() - t0

    # Aggregate comparison
    print(f"\n\n{'#'*90}")
    print(f"AGGREGATE COMPARISON")
    print(f"{'#'*90}")
    print(f"{'Detector':<20}  {'det%':>6}  {'med_px':>8}  {'FP':>4}  {'conf_med':>9}  {'lat_ms':>7}")
    print(f"{'-'*70}")

    for det_key, det_label in [("yolo_world", "YOLO-World v2-L"),
                                ("owlvit_baseline", "OWL-ViT baseline")]:
        all_det = []; all_med = []; total_fp = 0; all_conf = []; all_lat = []
        for case_name, v in all_results.items():
            r = v[det_key]
            if r['n_gt_vis'] > 0:
                all_det.append(r['det_rate'])
            m = r.get('det', {}).get('median')
            if m is not None:
                all_med.append(m)
            total_fp += r.get('n_fp', 0)
            cm = r.get('conf', {}).get('median')
            if cm is not None:
                all_conf.append(cm)
            all_lat.append(r['latency_ms']['mean'])

        avg_det = sum(all_det)/len(all_det)*100 if all_det else 0
        avg_med = sum(all_med)/len(all_med) if all_med else float('inf')
        avg_conf = sum(all_conf)/len(all_conf) if all_conf else 0
        avg_lat = sum(all_lat)/len(all_lat) if all_lat else 0
        print(f"{det_label:<20}  {avg_det:5.1f}%  {avg_med:7.1f}px  {total_fp:4d}  "
              f"{avg_conf:8.3f}  {avg_lat:6.0f}ms")

    print(f"{'#'*90}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save results
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"yoloworld_vs_owlvit_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Render comparison videos for a few interesting cases
    print("\nRendering comparison videos...")
    vis_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", f"yolo_comparison_{ts}")
    os.makedirs(vis_dir, exist_ok=True)
    for case in cases:
        vpath = render_comparison_video(
            case, yolo_model, owlvit_proc, owlvit_mdl, owlvit_dev, vis_dir)
        print(f"  {case['name']} -> {vpath}")

    print(f"\nDone! Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
