#!/usr/bin/env python3
"""
OWL-ViT + KF combined test.

OWL-ViT is an open-vocabulary object detector that outputs bounding boxes.
It was designed for detection (unlike CLIPSeg which does segmentation),
so it should be better at finding specific objects.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/owlvit_combined_test.py
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

# ── OWL-ViT Detection ──

def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device


def detect_owlvit(frame_rgb, obj_name, processor, model, device, threshold=0.05):
    """Detect object with OWL-ViT. Returns (cx, cy, conf, box_area)."""
    img = Image.fromarray(frame_rgb)
    queries = [[obj_name, f"a {obj_name}"]]
    inputs = processor(text=queries, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold)

    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()

    if len(boxes) == 0:
        return None, None, 0.0, 0

    # Pick highest-scoring detection
    best_idx = scores.argmax()
    box = boxes[best_idx]
    score = float(scores[best_idx])
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = int((box[2] - box[0]) * (box[3] - box[1]))
    return cx, cy, score, area


# ── Data Loading ──

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


# ── Configs ──

CONFIGS = {
    # OWL-ViT with different KF configs
    "owlvit_consensus3_td70": {
        "detect_threshold": 0.05,
        "conf_gate": 0.05,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.05, temporal_decay_gate=True),
    },
    "owlvit_consensus3_td10": {
        "detect_threshold": 0.10,
        "conf_gate": 0.10,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.10, temporal_decay_gate=True),
    },
    "owlvit_consensus2_td05": {
        "detect_threshold": 0.05,
        "conf_gate": 0.05,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=2, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.05, temporal_decay_gate=True),
    },
    "owlvit_baseline_05": {
        "detect_threshold": 0.05,
        "conf_gate": 0.05,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.05),
    },
    "owlvit_baseline_10": {
        "detect_threshold": 0.10,
        "conf_gate": 0.10,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.10),
    },
}


def run_test(case, config, processor, model, device):
    """Run OWL-ViT detection + KF on one case."""
    kf = config["kf_class"](**config["kf_kwargs"])
    kf.reset()
    det_thresh = config["detect_threshold"]
    conf_gate = config["conf_gate"]

    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    Xro = case["Xro"]
    obj_name = case["object"]
    n_frames = len(centroids)

    results = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area = detect_owlvit(frame_rgb, obj_name, processor, model,
                                             device, threshold=det_thresh)
        detected = (area > 0 and conf >= conf_gate)

        step_idx = centroids[frame_idx].get("step", frame_idx)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            est = kf.step(xcr, b, e, conf, area)
        else:
            est = kf.step(xcr, 0.0, 0.0, 0.0, 0)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        kf_err = None
        det_err = None
        if gt_in and gt_u is not None:
            if est is not None:
                kf_err = float(np.hypot(est[0] - gt_u, est[1] - gt_v))
            if detected and cx is not None:
                det_err = float(np.hypot(cx - gt_u, cy - gt_v))

        results.append({
            "frame": frame_idx,
            "kf_err": kf_err,
            "det_err": det_err,
            "conf": conf,
            "detected": detected,
            "gt_in": gt_in,
            "kf_active": est is not None,
            "coasting": kf.steps_coasting,
        })
        frame_idx += 1

    cap.release()

    kf_errs = [r["kf_err"] for r in results if r["kf_err"] is not None]
    det_errs = [r["det_err"] for r in results if r["det_err"] is not None]

    def _s(arr):
        if not arr: return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    gt_vis = sum(1 for r in results if r["gt_in"])
    n_det = sum(1 for r in results if r["detected"])
    n_tp = sum(1 for r in results if r["detected"] and r["gt_in"])
    n_fp = sum(1 for r in results if r["detected"] and not r["gt_in"])

    return {
        "kf_all": _s(kf_errs),
        "det": _s(det_errs),
        "n_frames": len(results),
        "n_gt_visible": gt_vis,
        "n_detected": n_det,
        "detection_rate": n_tp / max(gt_vis, 1),
        "n_false_pos": n_fp,
    }


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return

    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases from {eval_dir}")

    print("\nLoading OWL-ViT...")
    processor, model, device = load_owlvit()
    print(f"OWL-ViT on {device}")

    print(f"\nRunning {len(CONFIGS)} configs x {len(cases)} cases")

    all_results = {}
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")
        case_results = {}
        for case in cases:
            t1 = time.time()
            res = run_test(case, config, processor, model, device)
            dt = time.time() - t1

            kf = res["kf_all"]
            det = res["det"]
            kf_m = f"{kf['mean']:.1f}" if kf.get("mean") else "-"
            kf_d = f"{kf['median']:.1f}" if kf.get("median") else "-"
            det_d = f"{det['median']:.1f}" if det.get("median") else "-"
            dr = f"{res['detection_rate']*100:.1f}%"
            print(f"  {case['name']:<45}  KF={kf_m:>7}/{kf_d:>7}px  "
                  f"det_med={det_d:>7}px  det={dr:>6}  FP={res['n_false_pos']}  ({dt:.1f}s)")

            case_results[case["name"]] = res
        all_results[config_name] = case_results

    elapsed = time.time() - t0
    print(f"\n\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Aggregate
    print(f"\n{'='*100}")
    print(f"{'Config':<35} {'KF mean':>8} {'KF med':>8} {'KF p90':>8} "
          f"{'det_med':>8} {'det%':>6} {'FP':>4}")
    print(f"{'-'*100}")

    for config_name, case_results in all_results.items():
        means, meds, p90s, det_ms, det_rs, fps = [], [], [], [], [], []
        for cname, res in case_results.items():
            kf = res["kf_all"]
            if kf.get("mean"): means.append(kf["mean"])
            if kf.get("median"): meds.append(kf["median"])
            if kf.get("p90"): p90s.append(kf["p90"])
            det = res["det"]
            if det.get("median"): det_ms.append(det["median"])
            det_rs.append(res["detection_rate"] * 100)
            fps.append(res["n_false_pos"])

        m = f"{np.mean(means):.1f}" if means else "-"
        d = f"{np.mean(meds):.1f}" if meds else "-"
        p = f"{np.mean(p90s):.1f}" if p90s else "-"
        dm = f"{np.mean(det_ms):.1f}" if det_ms else "-"
        det = f"{np.mean(det_rs):.1f}"
        fp = f"{sum(fps)}"
        print(f"{config_name:<35} {m:>8} {d:>8} {p:>8} {dm:>8} {det:>6} {fp:>4}")

    print(f"{'='*100}")

    # Save
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"owlvit_combined_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
