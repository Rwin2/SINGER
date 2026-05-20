#!/usr/bin/env python3
"""
Combined best-of-everything experiment.

Tests the top prompt+filter combinations found overnight:
1. descriptive + CLIPSeg verify (ratio>=2) + EMA coast30
2. color_focus + CLIPSeg verify + EMA coast30
3. rich_5 + CLIPSeg verify + EMA coast30 (reference)
4. descriptive + EMA coast30 (no verify, for comparison)
5. per_object_best: clock=descriptive, leaf=color, drill=descriptive

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/combined_best.py
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
PROMPTS = {
    "rich_5": {
        "green clock": ["green clock", "a green wall clock", "green analog clock",
                        "round green clock on table", "green colored clock"],
        "leafblower": ["leafblower", "a leaf blower", "leaf blower tool",
                       "garden leaf blower", "handheld blower"],
        "drill": ["drill", "a power drill", "electric drill",
                  "cordless drill", "hand drill tool"],
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
    "per_object_best": {
        # Use descriptive for clock (best acc), color for leaf (best det), descriptive for drill
        "green clock": ["a small round green analog clock", "a bright green clock face",
                        "circular green clock with numbers", "green desktop clock",
                        "miniature green alarm clock"],
        "leafblower": ["green and pink leafblower", "colorful leaf blower",
                       "green pink blower", "multicolor leaf blower tool",
                       "bright colored leafblower"],
        "drill": ["a yellow battery-powered cordless drill",
                  "electric screwdriver drill tool",
                  "handheld power drill with trigger",
                  "yellow DeWalt style drill",
                  "cordless electric drill with battery"],
    },
}


def get_prompts(obj_name, prompt_set):
    for key, prompts in prompt_set.items():
        if key in obj_name.lower():
            return prompts
    return [obj_name]


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
        return None, None, 0.0, 0, None
    best = scores.argmax()
    box = boxes[best]
    score = float(scores[best])
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    area = int((box[2] - box[0]) * (box[3] - box[1]))
    return cx, cy, score, area, box


def verify_clipseg(frame_rgb, obj_name, box, cseg_proc, cseg_mdl, dev, ratio_thresh=2.0):
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


def run_test(case, prompt_set, use_verify, owl_proc, owl_mdl, cseg_proc, cseg_mdl, dev):
    filt = EMAFilter(alpha=0.5, max_coast=30)
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    prompts = get_prompts(obj_name, prompt_set)
    n_frames = len(centroids)

    filt_errs = []; det_errs = []
    n_det = 0; n_tp = 0; n_fp = 0; n_gt_vis = 0; n_est = 0
    verified_tp = 0; verified_fp = 0
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area, box = detect_owlvit(
            frame_rgb, prompts, owl_proc, owl_mdl, dev)
        detected = (area > 0 and conf >= 0.02)

        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        if gt_in:
            n_gt_vis += 1

        accepted = False
        if detected:
            n_det += 1
            is_tp = gt_in
            if is_tp:
                n_tp += 1
            else:
                n_fp += 1

            if use_verify:
                verified, ratio = verify_clipseg(
                    frame_rgb, obj_name, box, cseg_proc, cseg_mdl, dev)
                if verified:
                    accepted = True
                    if is_tp:
                        verified_tp += 1
                    else:
                        verified_fp += 1
            else:
                accepted = True
                verified_tp += (1 if is_tp else 0)
                verified_fp += (0 if is_tp else 1)

        est = filt.step(cx if accepted else 0, cy if accepted else 0, accepted)

        if gt_in and gt_u is not None:
            if est is not None:
                filt_errs.append(float(np.hypot(est[0] - gt_u, est[1] - gt_v)))
                n_est += 1
            if accepted and cx is not None:
                det_errs.append(float(np.hypot(cx - gt_u, cy - gt_v)))

        frame_idx += 1

    cap.release()

    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}

    return {
        "filt": _s(filt_errs), "det": _s(det_errs),
        "n_frames": frame_idx, "n_gt_vis": n_gt_vis,
        "n_det": n_det, "n_est": n_est, "n_tp": n_tp, "n_fp": n_fp,
        "verified_tp": verified_tp, "verified_fp": verified_fp,
        "det_rate": n_tp / max(n_gt_vis, 1),
        "verified_det_rate": verified_tp / max(n_gt_vis, 1),
        "coverage": n_est / max(n_gt_vis, 1),
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
    owl_proc, owl_mdl, dev = load_owlvit()
    print("Loading CLIPSeg...")
    cseg_proc, cseg_mdl, _ = load_clipseg()

    configs = [
        ("rich5_verify", "rich_5", True),
        ("rich5_noverify", "rich_5", False),
        ("descriptive_verify", "descriptive", True),
        ("descriptive_noverify", "descriptive", False),
        ("color_verify", "color_focus", True),
        ("color_noverify", "color_focus", False),
        ("per_obj_verify", "per_object_best", True),
        ("per_obj_noverify", "per_object_best", False),
    ]

    print(f"Testing {len(configs)} configs x {len(cases)} cases = {len(configs)*len(cases)} runs\n")

    all_results = {}
    for cfg_name, prompt_key, use_verify in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {cfg_name} (prompts={prompt_key}, verify={use_verify})")
        print(f"{'='*80}")
        prompt_set = PROMPTS[prompt_key]
        case_results = {}
        for case in cases:
            t1 = time.time()
            res = run_test(case, prompt_set, use_verify,
                          owl_proc, owl_mdl, cseg_proc, cseg_mdl, dev)
            dt = time.time() - t1
            f_med = f"{res['filt']['median']:.1f}" if res['filt'].get('median') else "-"
            vdet = f"{res['verified_det_rate']*100:.1f}%"
            cov = f"{res['coverage']*100:.1f}%"
            print(f"  {case['name']:<45}  vdet={vdet:>6}  cov={cov:>6}  "
                  f"med={f_med:>7}px  vFP={res['verified_fp']}  ({dt:.1f}s)")
            case_results[case["name"]] = res
        all_results[cfg_name] = case_results

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*100}")
    print(f"{'Config':<25s}  {'vdet%':>6}  {'cov%':>6}  {'filt_med':>9}  {'vFP':>4}  {'rawFP':>6}")
    print(f"{'-'*100}")
    for cfg_name, case_results in all_results.items():
        all_vdet = []; all_cov = []; all_med = []; total_vfp = 0; total_fp = 0
        for case_name, v in case_results.items():
            if v['n_gt_vis'] > 0:
                all_vdet.append(v['verified_det_rate'])
                all_cov.append(v['coverage'])
            m = v.get('filt', {}).get('median')
            if m is not None:
                all_med.append(m)
            total_vfp += v.get('verified_fp', 0)
            total_fp += v.get('n_fp', 0)
        avg_vdet = sum(all_vdet)/len(all_vdet)*100 if all_vdet else 0
        avg_cov = sum(all_cov)/len(all_cov)*100 if all_cov else 0
        avg_med = sum(all_med)/len(all_med) if all_med else float('inf')
        print(f"{cfg_name:<25s}  {avg_vdet:5.1f}%  {avg_cov:5.1f}%  {avg_med:8.1f}px  {total_vfp:4d}  {total_fp:6d}")
    print(f"{'#'*100}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"combined_best_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
