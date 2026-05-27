#!/usr/bin/env python3
"""
Prompt ablation experiment.

Tests different prompt strategies for OWL-ViT to find the optimal set:
1. Single basic prompt (baseline)
2. Rich 5-prompt set (current best)
3. Extended 10-prompt set (more variations)
4. Color-focused prompts (emphasize object color)
5. Context-aware prompts (include scene context)
6. Negative prompts (what the object is NOT)
7. Per-prompt analysis: which individual prompts work best?

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/prompt_ablation.py
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


# ── Prompt Sets ──

PROMPT_SETS = {
    "basic": {
        "green clock": ["green clock"],
        "leafblower": ["leafblower"],
        "drill": ["drill"],
    },
    "basic_2": {
        "green clock": ["green clock", "a green clock"],
        "leafblower": ["leafblower", "a leaf blower"],
        "drill": ["drill", "a power drill"],
    },
    "rich_5": {
        "green clock": ["green clock", "a green wall clock", "green analog clock",
                        "round green clock on table", "green colored clock"],
        "leafblower": ["leafblower", "a leaf blower", "leaf blower tool",
                       "garden leaf blower", "handheld blower"],
        "drill": ["drill", "a power drill", "electric drill",
                  "cordless drill", "hand drill tool"],
    },
    "extended_10": {
        "green clock": [
            "green clock", "a green wall clock", "green analog clock",
            "round green clock on table", "green colored clock",
            "small green clock", "clock with green frame", "green round clock",
            "analog clock green", "timepiece green",
        ],
        "leafblower": [
            "leafblower", "a leaf blower", "leaf blower tool",
            "garden leaf blower", "handheld blower",
            "electric leaf blower", "blower machine", "green leaf blower",
            "pink leaf blower", "garden blower tool",
        ],
        "drill": [
            "drill", "a power drill", "electric drill",
            "cordless drill", "hand drill tool",
            "yellow drill", "battery drill", "power tool drill",
            "handheld drill", "yellow cordless drill",
        ],
    },
    "color_focus": {
        "green clock": ["green clock", "green object", "bright green clock",
                        "green colored round object", "neon green clock"],
        "leafblower": ["green and pink leafblower", "colorful leaf blower",
                       "green pink blower", "multicolor leaf blower tool",
                       "bright colored leafblower"],
        "drill": ["yellow drill", "bright yellow power drill",
                  "yellow colored drill", "yellow tool", "yellow handheld drill"],
    },
    "scene_context": {
        "green clock": ["a green clock on a table", "green clock in a room",
                        "clock sitting on surface", "small clock indoors",
                        "green clock next to objects"],
        "leafblower": ["leaf blower on the ground", "leaf blower on a table",
                       "gardening tool on surface", "blower in indoor scene",
                       "leaf blower placed down"],
        "drill": ["drill on a table", "power drill in a room",
                  "cordless drill on surface", "tool on workbench",
                  "drill placed indoors"],
    },
    "descriptive": {
        "green clock": ["a small round green analog clock", "a bright green clock face",
                        "circular green clock with numbers", "green desktop clock",
                        "miniature green alarm clock"],
        "leafblower": ["a handheld garden leaf blower with a long nozzle",
                       "leaf blowing machine with handle",
                       "portable electric blower for leaves",
                       "garden cleanup blower tool",
                       "powered leaf debris blower"],
        "drill": ["a yellow battery-powered cordless drill",
                  "electric screwdriver drill tool",
                  "handheld power drill with trigger",
                  "yellow DeWalt style drill",
                  "cordless electric drill with battery"],
    },
}


def load_owlvit():
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    return proc, mdl, dev


def detect_owlvit(frame_rgb, prompts, proc, mdl, dev, threshold=0.02):
    img = Image.fromarray(frame_rgb)
    inputs = proc(text=[prompts], images=img, return_tensors="pt")
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
        npz = np.load(npz_files[0])
        cases.append({
            "name": case_dir,
            "video_path": rgb_vids[0],
            "centroids": log_data["centroids"],
            "object": log_data.get("object", case_dir),
        })
    return cases


def get_prompts(obj_name, prompt_set):
    for key, prompts in prompt_set.items():
        if key in obj_name.lower():
            return prompts
    return [obj_name]


def run_test(case, prompt_set, proc, mdl, dev):
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)
    prompts = get_prompts(obj_name, prompt_set)

    det_errs = []
    n_det = 0; n_tp = 0; n_fp = 0; n_gt_vis = 0
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area = detect_owlvit(
            frame_rgb, prompts, proc, mdl, dev)
        detected = (area > 0 and conf >= 0.02)

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

        frame_idx += 1

    cap.release()

    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    return {
        "det": _s(det_errs),
        "n_frames": frame_idx, "n_gt_vis": n_gt_vis,
        "n_det": n_det, "n_tp": n_tp, "n_fp": n_fp,
        "det_rate": n_tp / max(n_gt_vis, 1),
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

    print(f"Testing {len(PROMPT_SETS)} prompt sets x {len(cases)} cases = "
          f"{len(PROMPT_SETS)*len(cases)} runs\n")

    all_results = {}
    for set_name, prompt_set in PROMPT_SETS.items():
        print(f"\n{'='*80}")
        print(f"PROMPT SET: {set_name}")
        print(f"{'='*80}")
        case_results = {}
        for case in cases:
            t1 = time.time()
            res = run_test(case, prompt_set, proc, mdl, dev)
            dt = time.time() - t1
            d_med = f"{res['det']['median']:.1f}" if res['det'].get('median') else "-"
            det = f"{res['det_rate']*100:.1f}%"
            print(f"  {case['name']:<45}  det={det:>6}  med={d_med:>7}px  "
                  f"FP={res['n_fp']}  ({dt:.1f}s)")
            case_results[case["name"]] = res
        all_results[set_name] = case_results

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*90}")
    print(f"{'Prompt Set':<20s}  {'det%':>6}  {'det_med':>9}  {'FP':>4}")
    print(f"{'-'*90}")
    for set_name, case_results in all_results.items():
        all_det = []; all_med = []; total_fp = 0
        for case_name, v in case_results.items():
            if v['n_gt_vis'] > 0:
                all_det.append(v['det_rate'])
            m = v.get('det', {}).get('median')
            if m is not None:
                all_med.append(m)
            total_fp += v.get('n_fp', 0)
        avg_det = sum(all_det)/len(all_det)*100 if all_det else 0
        avg_med = sum(all_med)/len(all_med) if all_med else float('inf')
        print(f"{set_name:<20s}  {avg_det:5.1f}%  {avg_med:8.1f}px  {total_fp:4d}")
    print(f"{'#'*90}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"prompt_ablation_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
