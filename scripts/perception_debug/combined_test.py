#!/usr/bin/env python3
"""
Combined test: best CLIPSeg strategy + best KF config.

Runs multimax CLIPSeg detection (try multiple prompts, pick best) +
consensus-init temporal-decay inverse-depth EKF on saved RGB videos.

This is the end-to-end test that evaluates the full perception pipeline.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/combined_test.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
import cv2
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

# Import KF from our experiments
from perception_debug.kf_experiments import (
    InvDepthEKF, ConsensusInitEKF, IMG_W, IMG_H,
    predict_state, numerical_jacobian,
)
from perception_debug.clipseg_experiments import (
    load_clipseg, get_logits, get_multi_query_logits, extract_centroid,
)


# ── Detection Strategies ──

def detect_multimax(frame_rgb, obj_name, processor, model, device,
                    sigmoid_thresh=0.5):
    """Multi-prompt max: try several prompts, take highest confidence."""
    prompts = [
        obj_name,
        f"a photo of a {obj_name}",
        f"a {obj_name} in a room",
    ]
    logits_dict = get_multi_query_logits(frame_rgb, prompts, processor, model, device)

    best_cx, best_cy, best_conf, best_npx = None, None, 0.0, 0
    for q, logits in logits_dict.items():
        prob = 1.0 / (1.0 + np.exp(-logits))
        cx, cy, conf, npx = extract_centroid(prob, sigmoid_thresh)
        if conf > best_conf:
            best_cx, best_cy, best_conf, best_npx = cx, cy, conf, npx
    return best_cx, best_cy, best_conf, best_npx


def detect_multiavg(frame_rgb, obj_name, processor, model, device,
                    sigmoid_thresh=0.5):
    """Multi-prompt average: average logits from multiple prompts."""
    prompts = [
        obj_name,
        f"a photo of a {obj_name}",
        f"a {obj_name} in a room",
    ]
    logits_dict = get_multi_query_logits(frame_rgb, prompts, processor, model, device)
    avg_logits = np.mean(list(logits_dict.values()), axis=0)
    prob = 1.0 / (1.0 + np.exp(-avg_logits))
    return extract_centroid(prob, sigmoid_thresh)


def detect_single(frame_rgb, obj_name, processor, model, device,
                   sigmoid_thresh=0.5):
    """Baseline single prompt."""
    logits = get_logits(frame_rgb, obj_name, processor, model, device)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return extract_centroid(prob, sigmoid_thresh)


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


# ── Configs to Test ──

CONFIGS = {
    "single_consensus3_td70": {
        "detect_fn": "single",
        "conf_gate": 0.70,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.70, temporal_decay_gate=True),
    },
    "multimax_consensus3_td65": {
        "detect_fn": "multimax",
        "conf_gate": 0.65,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.65, temporal_decay_gate=True),
    },
    "multimax_consensus3_td70": {
        "detect_fn": "multimax",
        "conf_gate": 0.70,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.70, temporal_decay_gate=True),
    },
    "multiavg_consensus3_td70": {
        "detect_fn": "multiavg",
        "conf_gate": 0.70,
        "kf_class": ConsensusInitEKF,
        "kf_kwargs": dict(n_init=3, init_radius_px=50.0, max_coast_frames=20,
                          conf_gate=0.70, temporal_decay_gate=True),
    },
    "multimax_baseline75": {
        "detect_fn": "multimax",
        "conf_gate": 0.75,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.75),
    },
    "multiavg_baseline75": {
        "detect_fn": "multiavg",
        "conf_gate": 0.75,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.75),
    },
    # Original baseline for comparison
    "single_baseline60": {
        "detect_fn": "single",
        "conf_gate": 0.60,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.60),
    },
    "single_baseline75": {
        "detect_fn": "single",
        "conf_gate": 0.75,
        "kf_class": InvDepthEKF,
        "kf_kwargs": dict(conf_gate=0.75),
    },
}


def run_combined_test(case, config, processor, model, device):
    """Run detection + KF on one case."""
    detect_fn = {
        "single": detect_single,
        "multimax": detect_multimax,
        "multiavg": detect_multiavg,
    }[config["detect_fn"]]

    conf_gate = config["conf_gate"]
    kf = config["kf_class"](**config["kf_kwargs"])
    kf.reset()

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

        # Detection
        cx, cy, conf, npx = detect_fn(frame_rgb, obj_name, processor, model, device)
        detected = (npx > 0 and conf >= conf_gate)

        # KF step
        step_idx = centroids[frame_idx].get("step", frame_idx)
        xcr_idx = min(step_idx + 1, Xro.shape[1] - 1)
        xcr = Xro[:, xcr_idx]

        if detected:
            b = 2.0 * (cx / IMG_W) - 1.0
            e = 2.0 * (cy / IMG_H) - 1.0
            est = kf.step(xcr, b, e, conf, npx)
        else:
            est = kf.step(xcr, 0.0, 0.0, 0.0, 0)

        # GT from centroid log
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        kf_err = None
        cseg_err = None
        if gt_in and gt_u is not None:
            if est is not None:
                kf_err = float(np.hypot(est[0] - gt_u, est[1] - gt_v))
            if detected and cx is not None:
                cseg_err = float(np.hypot(cx - gt_u, cy - gt_v))

        results.append({
            "frame": frame_idx,
            "kf_err": kf_err,
            "cseg_err": cseg_err,
            "conf": conf,
            "detected": detected,
            "gt_in": gt_in,
            "kf_active": est is not None,
            "coasting": kf.steps_coasting,
        })
        frame_idx += 1

    cap.release()

    # Summary
    kf_errs = [r["kf_err"] for r in results if r["kf_err"] is not None]
    cseg_errs = [r["cseg_err"] for r in results if r["cseg_err"] is not None]
    kf_meas = [r["kf_err"] for r in results if r["kf_err"] is not None and r["coasting"] == 0]
    kf_coast = [r["kf_err"] for r in results if r["kf_err"] is not None and r["coasting"] > 0]

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
        "kf_measured": _s(kf_meas),
        "kf_coasting": _s(kf_coast),
        "cseg": _s(cseg_errs),
        "n_frames": len(results),
        "n_gt_visible": gt_vis,
        "n_detected": n_det,
        "n_true_pos": n_tp,
        "n_false_pos": n_fp,
        "detection_rate": n_tp / max(gt_vis, 1),
        "n_kf_active": sum(1 for r in results if r["kf_active"]),
    }


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found")
        return

    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases from {eval_dir}")
    for c in cases:
        print(f"  {c['name']}: {len(c['centroids'])} frames")

    print("\nLoading CLIPSeg...")
    processor, model, device = load_clipseg()
    print(f"CLIPSeg on {device}")

    print(f"\nRunning {len(CONFIGS)} configs x {len(cases)} cases")

    all_results = {}
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")
        case_results = {}
        for case in cases:
            t1 = time.time()
            res = run_combined_test(case, config, processor, model, device)
            dt = time.time() - t1
            case_results[case["name"]] = res

            kf = res["kf_all"]
            cseg = res["cseg"]
            kf_mean = f"{kf['mean']:.1f}" if kf.get("mean") else "-"
            kf_med = f"{kf['median']:.1f}" if kf.get("median") else "-"
            cseg_med = f"{cseg['median']:.1f}" if cseg.get("median") else "-"
            det = f"{res['detection_rate']*100:.1f}%"
            print(f"  {case['name']:<45}  KF={kf_mean:>6}/{kf_med:>6}px  "
                  f"CSEG={cseg_med:>6}px  det={det:>6}  "
                  f"FP={res['n_false_pos']}  ({dt:.1f}s)")

        all_results[config_name] = case_results

    elapsed = time.time() - t0
    print(f"\n\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Aggregate table
    print(f"\n{'='*100}")
    print(f"{'Config':<35} {'KF mean':>8} {'KF med':>8} {'KF p90':>8} "
          f"{'CSEG med':>8} {'det%':>6} {'FP':>4}")
    print(f"{'-'*100}")

    for config_name, case_results in all_results.items():
        means, meds, p90s, cseg_ms, det_rs, fps = [], [], [], [], [], []
        for cname, res in case_results.items():
            kf = res["kf_all"]
            if kf.get("mean"): means.append(kf["mean"])
            if kf.get("median"): meds.append(kf["median"])
            if kf.get("p90"): p90s.append(kf["p90"])
            cseg = res["cseg"]
            if cseg.get("median"): cseg_ms.append(cseg["median"])
            det_rs.append(res["detection_rate"] * 100)
            fps.append(res["n_false_pos"])

        m = f"{np.mean(means):.1f}" if means else "-"
        d = f"{np.mean(meds):.1f}" if meds else "-"
        p = f"{np.mean(p90s):.1f}" if p90s else "-"
        c = f"{np.mean(cseg_ms):.1f}" if cseg_ms else "-"
        det = f"{np.mean(det_rs):.1f}"
        fp = f"{sum(fps)}"
        print(f"{config_name:<35} {m:>8} {d:>8} {p:>8} {c:>8} {det:>6} {fp:>4}")

    print(f"{'='*100}")

    # Save
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"combined_test_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
