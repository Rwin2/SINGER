#!/usr/bin/env python3
"""
Step-by-step analysis: per-frame detector error vs KF error.

Shows exactly how the KF helps (or hurts) compared to raw detection.
Outputs a CSV trace per case + aggregate statistics.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/step_analysis.py
"""
import os, sys, json, glob, time, csv
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

# ── Detectors ──

def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
    img = Image.fromarray(frame_rgb)
    queries = [[obj_name, f"a {obj_name}"]]
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


def analyze_case(case, proc, mdl, dev):
    """Run OWL-ViT + KF on one case, return per-frame trace."""
    kf = ConsensusInitEKF(
        n_init=3, init_radius_px=50.0, max_coast_frames=20,
        conf_gate=0.05, temporal_decay_gate=True,
    )
    kf.reset()

    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    Xro = case["Xro"]
    obj_name = case["object"]
    n_frames = len(centroids)

    trace = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area, box = detect_owlvit(
            frame_rgb, obj_name, proc, mdl, dev, threshold=0.02)
        detected = (area > 0 and conf >= 0.05)

        step_idx = centroids[frame_idx].get("step", frame_idx)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            kf_est = kf.step(xcr, b, e, conf, area)
        else:
            kf_est = kf.step(xcr, 0.0, 0.0, 0.0, 0)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        det_err = None
        kf_err = None
        if gt_in and gt_u is not None:
            if detected and cx is not None:
                det_err = float(np.hypot(cx - gt_u, cy - gt_v))
            if kf_est is not None:
                kf_err = float(np.hypot(kf_est[0] - gt_u, kf_est[1] - gt_v))

        row = {
            "frame": frame_idx,
            "gt_in": gt_in,
            "gt_u": gt_u,
            "gt_v": gt_v,
            "detected": detected,
            "det_cx": cx if detected else None,
            "det_cy": cy if detected else None,
            "det_conf": conf,
            "det_err_px": det_err,
            "kf_active": kf_est is not None,
            "kf_u": kf_est[0] if kf_est else None,
            "kf_v": kf_est[1] if kf_est else None,
            "kf_err_px": kf_err,
            "kf_coasting": kf.steps_coasting,
            "kf_mode": ("meas" if kf_est and kf.steps_coasting == 0
                        else f"coast({kf.steps_coasting})" if kf_est
                        else "off"),
        }
        trace.append(row)
        frame_idx += 1

    cap.release()
    return trace


def print_analysis(case_name, trace):
    """Print step-by-step analysis for one case."""
    n_frames = len(trace)
    gt_vis = sum(1 for r in trace if r["gt_in"])
    n_det = sum(1 for r in trace if r["detected"])
    n_det_gt = sum(1 for r in trace if r["detected"] and r["gt_in"])
    n_kf = sum(1 for r in trace if r["kf_active"])
    n_kf_gt = sum(1 for r in trace if r["kf_active"] and r["gt_in"])

    det_errs = [r["det_err_px"] for r in trace if r["det_err_px"] is not None]
    kf_errs = [r["kf_err_px"] for r in trace if r["kf_err_px"] is not None]

    # Frames where both exist — direct comparison
    both = [(r["det_err_px"], r["kf_err_px"])
            for r in trace if r["det_err_px"] is not None and r["kf_err_px"] is not None]
    kf_only = [r["kf_err_px"] for r in trace
               if r["kf_err_px"] is not None and r["det_err_px"] is None]

    print(f"\n{'='*80}")
    print(f"CASE: {case_name}")
    print(f"{'='*80}")
    print(f"Frames: {n_frames}  GT visible: {gt_vis}  "
          f"Detected: {n_det} ({n_det/max(gt_vis,1)*100:.1f}%)  "
          f"KF active: {n_kf} ({n_kf/max(gt_vis,1)*100:.1f}%)")

    if det_errs:
        d = np.array(det_errs)
        print(f"\nDetector (OWL-ViT) error when detected + GT visible ({len(d)} frames):")
        print(f"  mean={d.mean():.1f}px  median={np.median(d):.1f}px  "
              f"p10={np.percentile(d,10):.1f}  p90={np.percentile(d,90):.1f}  max={d.max():.1f}")

    if kf_errs:
        k = np.array(kf_errs)
        print(f"\nKF error when active + GT visible ({len(k)} frames):")
        print(f"  mean={k.mean():.1f}px  median={np.median(k):.1f}px  "
              f"p10={np.percentile(k,10):.1f}  p90={np.percentile(k,90):.1f}  max={k.max():.1f}")

    if both:
        det_b = np.array([b[0] for b in both])
        kf_b = np.array([b[1] for b in both])
        improvement = det_b - kf_b
        print(f"\nDirect comparison (both active, {len(both)} frames):")
        print(f"  Det mean={det_b.mean():.1f}px  KF mean={kf_b.mean():.1f}px  "
              f"KF improvement={improvement.mean():.1f}px ({improvement.mean()/max(det_b.mean(),0.01)*100:.0f}%)")
        n_kf_better = (improvement > 0).sum()
        n_det_better = (improvement < 0).sum()
        n_same = (improvement == 0).sum()
        print(f"  KF better: {n_kf_better}/{len(both)}  "
              f"Det better: {n_det_better}/{len(both)}  Same: {n_same}")

    if kf_only:
        ko = np.array(kf_only)
        print(f"\nKF-only frames (coasting, no detection, {len(ko)} frames):")
        print(f"  mean={ko.mean():.1f}px  median={np.median(ko):.1f}px  "
              f"max={ko.max():.1f}")

    # Step-by-step trace (compact, only frames with GT)
    print(f"\nFrame-by-frame trace (GT visible only):")
    print(f"{'frame':>6} {'det':>5} {'det_err':>8} {'kf':>5} {'kf_err':>8} {'mode':>12} {'conf':>6}")
    print(f"{'-'*60}")
    for r in trace:
        if not r["gt_in"]:
            continue
        det_s = "Y" if r["detected"] else "."
        det_e = f"{r['det_err_px']:.0f}" if r["det_err_px"] is not None else "-"
        kf_s = "Y" if r["kf_active"] else "."
        kf_e = f"{r['kf_err_px']:.0f}" if r["kf_err_px"] is not None else "-"
        mode = r["kf_mode"]
        conf = f"{r['det_conf']:.3f}"
        print(f"{r['frame']:>6} {det_s:>5} {det_e:>8} {kf_s:>5} {kf_e:>8} {mode:>12} {conf:>6}")


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return

    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases from {eval_dir}")

    print("Loading OWL-ViT...")
    proc, mdl, dev = load_owlvit()
    print(f"OWL-ViT on {dev}")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "analysis", ts)
    os.makedirs(out_dir, exist_ok=True)

    all_det_errs = []
    all_kf_errs = []
    all_kf_only_errs = []
    n_total_gt = 0
    n_total_det = 0
    n_total_kf = 0

    for case in cases:
        print(f"\nProcessing {case['name']}...")
        t1 = time.time()
        trace = analyze_case(case, proc, mdl, dev)
        dt = time.time() - t1
        print(f"  {len(trace)} frames in {dt:.1f}s")

        print_analysis(case["name"], trace)

        # Save CSV
        csv_path = os.path.join(out_dir, f"{case['name']}_trace.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trace[0].keys())
            writer.writeheader()
            writer.writerows(trace)

        # Aggregate
        for r in trace:
            if r["gt_in"]:
                n_total_gt += 1
                if r["detected"]:
                    n_total_det += 1
                if r["kf_active"]:
                    n_total_kf += 1
            if r["det_err_px"] is not None:
                all_det_errs.append(r["det_err_px"])
                if r["kf_err_px"] is not None:
                    all_kf_errs.append(r["kf_err_px"])
            elif r["kf_err_px"] is not None:
                all_kf_only_errs.append(r["kf_err_px"])

    elapsed = time.time() - t0

    # ── Grand Summary ──
    print(f"\n\n{'#'*80}")
    print(f"GRAND SUMMARY (all {len(cases)} cases)")
    print(f"{'#'*80}")
    print(f"Total GT-visible frames: {n_total_gt}")
    print(f"Total detected: {n_total_det} ({n_total_det/max(n_total_gt,1)*100:.1f}%)")
    print(f"Total KF active: {n_total_kf} ({n_total_kf/max(n_total_gt,1)*100:.1f}%)")

    if all_det_errs:
        d = np.array(all_det_errs)
        print(f"\nRaw detector error (all frames where detected + GT visible):")
        print(f"  N={len(d)}  mean={d.mean():.1f}px  median={np.median(d):.1f}px  "
              f"p90={np.percentile(d,90):.1f}  max={d.max():.1f}")

    if all_kf_errs:
        k = np.array(all_kf_errs)
        print(f"\nKF error (frames where det + KF + GT all active):")
        print(f"  N={len(k)}  mean={k.mean():.1f}px  median={np.median(k):.1f}px  "
              f"p90={np.percentile(k,90):.1f}  max={k.max():.1f}")

        # Direct comparison
        d_matched = np.array(all_det_errs[:len(all_kf_errs)])
        if len(d_matched) == len(k):
            improvement = d_matched - k
            print(f"\nDirect comparison (same frames):")
            print(f"  Det mean={d_matched.mean():.1f}px → KF mean={k.mean():.1f}px")
            print(f"  Improvement: {improvement.mean():.1f}px ({improvement.mean()/max(d_matched.mean(),0.01)*100:.0f}%)")

    if all_kf_only_errs:
        ko = np.array(all_kf_only_errs)
        print(f"\nKF coasting error (KF active but no detection):")
        print(f"  N={len(ko)}  mean={ko.mean():.1f}px  median={np.median(ko):.1f}px  max={ko.max():.1f}")

    # KF coverage boost
    if n_total_gt > 0:
        det_cov = n_total_det / n_total_gt * 100
        kf_cov = n_total_kf / n_total_gt * 100
        print(f"\nCoverage boost from KF:")
        print(f"  Detector only: {det_cov:.1f}% of GT-visible frames have an estimate")
        print(f"  With KF (coasting): {kf_cov:.1f}% of GT-visible frames have an estimate")
        print(f"  KF adds {kf_cov - det_cov:.1f}pp coverage through coasting")

    print(f"\nCSV traces saved to: {out_dir}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save summary JSON
    summary = {
        "n_cases": len(cases),
        "n_total_gt": n_total_gt,
        "n_total_det": n_total_det,
        "n_total_kf": n_total_kf,
        "det_mean": float(np.mean(all_det_errs)) if all_det_errs else None,
        "det_median": float(np.median(all_det_errs)) if all_det_errs else None,
        "kf_mean": float(np.mean(all_kf_errs)) if all_kf_errs else None,
        "kf_median": float(np.median(all_kf_errs)) if all_kf_errs else None,
        "kf_only_mean": float(np.mean(all_kf_only_errs)) if all_kf_only_errs else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
