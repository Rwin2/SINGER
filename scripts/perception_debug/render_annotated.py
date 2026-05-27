#!/usr/bin/env python3
"""
Render annotated videos with detector + KF markers overlaid on RGB.

Same visual style as eval_failed_val_videos.py:
  - Green crosshair = GT
  - Magenta crosshair = detector (OWL-ViT or CLIPSeg)
  - Cyan crosshair = KF (bright when measured, dim when coasting)
  - HUD: step, confidence, mode, errors

Outputs MP4 videos for visual inspection.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/render_annotated.py
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

FPS = 20


# ── Detectors ──

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


def detect_owlvit(frame_rgb, obj_name, proc, mdl, dev, threshold=0.02):
    """OWL-ViT detection. Returns (cx, cy, conf, area, box)."""
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


def detect_clipseg(frame_rgb, obj_name, proc, mdl, dev, sigmoid_thresh=0.5):
    """CLIPSeg detection. Returns (cx, cy, conf, npx, None)."""
    img = Image.fromarray(frame_rgb)
    inputs = proc(images=img, text=obj_name, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
    logits_np = logits.cpu().squeeze().numpy().astype(np.float32)
    if logits_np.shape != (IMG_H, IMG_W):
        logits_np = cv2.resize(logits_np, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    prob = 1.0 / (1.0 + np.exp(-logits_np))
    conf = float(prob.max())
    mask = prob > sigmoid_thresh
    n_above = int(mask.sum())
    if n_above < 30:
        return None, None, conf, 0, None
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return None, None, conf, 0, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label
    blob_area = int(areas[best_label - 1])
    if blob_area < 30:
        return None, None, conf, 0, None
    ys, xs = np.where(blob_mask)
    w = prob[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))
    return cx, cy, conf, blob_area, None


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
            "obj_target": npz["obj_target"],
        })
    return cases


# ── Main ──

def render_case(case, detector_name, detect_fn, det_proc, det_mdl, det_dev,
                kf, conf_gate, out_dir):
    """Render annotated video for one case."""
    kf.reset()
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    Xro = case["Xro"]
    obj_name = case["object"]
    n_frames = len(centroids)

    out_path = os.path.join(out_dir, f"{case['name']}_{detector_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (IMG_W, IMG_H))

    frame_idx = 0
    stats = {"n_det": 0, "n_tp": 0, "n_fp": 0, "kf_errs": [], "det_errs": []}

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect
        cx, cy, conf, area_or_npx, box = detect_fn(
            frame_rgb, obj_name, det_proc, det_mdl, det_dev)
        detected = (area_or_npx > 0 and conf >= conf_gate)

        # KF
        step_idx = centroids[frame_idx].get("step", frame_idx)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            kf_est = kf.step(xcr, b, e, conf, area_or_npx)
        else:
            kf_est = kf.step(xcr, 0.0, 0.0, 0.0, 0)

        # GT
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # ── Draw on frame_bgr ──
        # GT (green)
        if gt_in and gt_u is not None:
            draw_marker(frame_bgr, gt_u, gt_v, (50, 255, 50), "GT", sz=14, th=2)

        # Detector (magenta)
        det_err = None
        if detected:
            draw_marker(frame_bgr, cx, cy, (255, 0, 255),
                        f"{detector_name} {conf:.2f}", sz=14, th=2)
            if box is not None:
                draw_box(frame_bgr, box, (255, 0, 255), th=1)
            stats["n_det"] += 1
            if gt_in:
                stats["n_tp"] += 1
                det_err = float(np.hypot(cx - gt_u, cy - gt_v))
                stats["det_errs"].append(det_err)
            else:
                stats["n_fp"] += 1

        # KF (cyan/dim cyan)
        kf_err = None
        if kf_est is not None:
            kf_u, kf_v, kf_sigma = kf_est
            if kf.steps_coasting == 0:
                draw_marker(frame_bgr, kf_u, kf_v, (0, 255, 255), "KF", sz=16, th=2)
            else:
                draw_marker(frame_bgr, kf_u, kf_v, (0, 180, 180),
                            f"KF~{kf.steps_coasting}", sz=12, th=1)
            if gt_in and gt_u is not None:
                kf_err = float(np.hypot(kf_u - gt_u, kf_v - gt_v))
                stats["kf_errs"].append(kf_err)

        # HUD
        txt1 = f"step {frame_idx}/{n_frames}  conf={conf:.3f}  " \
               f"{'DETECTED' if detected else 'no detection'}"
        txt2 = ""
        if kf_est is not None:
            mode = "meas" if kf.steps_coasting == 0 else f"coast({kf.steps_coasting})"
            txt2 = f"KF:{mode}  sigma={kf_sigma:.4f}"
        parts = []
        if det_err is not None:
            parts.append(f"det={det_err:.0f}px")
        if kf_err is not None:
            parts.append(f"kf={kf_err:.0f}px")
        txt3 = "err: " + "  ".join(parts) if parts else ""

        draw_hud(frame_bgr, [(18, txt1), (34, txt2), (50, txt3)])

        writer.write(frame_bgr)
        frame_idx += 1

    cap.release()
    writer.release()

    # Summary
    kf_e = stats["kf_errs"]
    det_e = stats["det_errs"]
    n_gt = sum(1 for c in centroids if c.get("gt_in_frame", False))
    summary = {
        "n_frames": n_frames,
        "n_gt_visible": n_gt,
        "n_detected": stats["n_det"],
        "n_tp": stats["n_tp"],
        "n_fp": stats["n_fp"],
        "det_rate": stats["n_tp"] / max(n_gt, 1),
        "kf_mean": float(np.mean(kf_e)) if kf_e else None,
        "kf_median": float(np.median(kf_e)) if kf_e else None,
        "kf_p90": float(np.percentile(kf_e, 90)) if kf_e else None,
        "det_mean": float(np.mean(det_e)) if det_e else None,
        "det_median": float(np.median(det_e)) if det_e else None,
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

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load detectors
    print("Loading OWL-ViT...")
    owl_proc, owl_mdl, owl_dev = load_owlvit()
    print("Loading CLIPSeg...")
    cseg_proc, cseg_mdl, cseg_dev = load_clipseg()

    # Configs to render
    configs = [
        {
            "name": "owlvit",
            "detect_fn": detect_owlvit,
            "proc": owl_proc, "mdl": owl_mdl, "dev": owl_dev,
            "conf_gate": 0.05,
            "kf": ConsensusInitEKF(n_init=3, init_radius_px=50.0,
                                    max_coast_frames=20, conf_gate=0.05,
                                    temporal_decay_gate=True),
        },
        {
            "name": "clipseg",
            "detect_fn": detect_clipseg,
            "proc": cseg_proc, "mdl": cseg_mdl, "dev": cseg_dev,
            "conf_gate": 0.70,
            "kf": ConsensusInitEKF(n_init=3, init_radius_px=50.0,
                                    max_coast_frames=20, conf_gate=0.70,
                                    temporal_decay_gate=True),
        },
    ]

    all_summaries = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Rendering with {cfg['name']}")
        print(f"{'='*60}")
        summaries = {}
        for case in cases:
            t1 = time.time()
            path, summary = render_case(
                case, cfg["name"], cfg["detect_fn"],
                cfg["proc"], cfg["mdl"], cfg["dev"],
                cfg["kf"], cfg["conf_gate"], out_dir,
            )
            dt = time.time() - t1
            s = summary
            det_r = f"{s['det_rate']*100:.1f}%"
            kf_m = f"{s['kf_mean']:.1f}" if s['kf_mean'] else "-"
            kf_d = f"{s['kf_median']:.1f}" if s['kf_median'] else "-"
            det_d = f"{s['det_median']:.1f}" if s['det_median'] else "-"
            print(f"  {case['name']:<45}  "
                  f"KF={kf_m:>6}/{kf_d:>6}  det_med={det_d:>6}  "
                  f"det={det_r:>6}  FP={s['n_fp']}  ({dt:.1f}s)  → {os.path.basename(path)}")
            summaries[case["name"]] = summary
        all_summaries[cfg["name"]] = summaries

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Videos saved to: {out_dir}")

    # Aggregate
    print(f"\n{'='*80}")
    for det_name, summaries in all_summaries.items():
        kf_ms = [s["kf_mean"] for s in summaries.values() if s["kf_mean"]]
        kf_ds = [s["kf_median"] for s in summaries.values() if s["kf_median"]]
        det_rs = [s["det_rate"]*100 for s in summaries.values()]
        fps = sum(s["n_fp"] for s in summaries.values())
        print(f"{det_name}: KF mean={np.mean(kf_ms):.1f}  "
              f"KF med={np.mean(kf_ds):.1f}  det={np.mean(det_rs):.1f}%  FP={fps}")
    print(f"{'='*80}")

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)


if __name__ == "__main__":
    main()
