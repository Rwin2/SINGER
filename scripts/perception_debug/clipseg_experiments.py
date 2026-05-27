#!/usr/bin/env python3
"""
CLIPSeg detection experiments — improving recall + precision.

Loads RGB videos from eval runs, tests different CLIPSeg strategies:
1. Single-prompt baseline (current)
2. Dual-prompt: positive + negative prompts, differential score
3. Prompt engineering: different text queries
4. Multi-scale: run at multiple resolutions, combine

Compares detection rates and centroid accuracy against GT from centroid logs.

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/clipseg_experiments.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
import cv2
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

IMG_W, IMG_H = 640, 360


# ── CLIPSeg Strategies ──

def load_clipseg():
    """Load CLIPSeg model."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device).eval()
    return processor, model, device


def get_logits(frame_rgb, query, processor, model, device):
    """Run CLIPSeg once, return raw logits resized to IMG_H x IMG_W."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(frame_rgb)
    inputs = processor(images=img, text=query, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    logits_np = logits.cpu().squeeze().numpy().astype(np.float32)
    if logits_np.shape != (IMG_H, IMG_W):
        logits_np = cv2.resize(logits_np, (IMG_W, IMG_H),
                               interpolation=cv2.INTER_LINEAR)
    return logits_np


def get_multi_query_logits(frame_rgb, queries, processor, model, device):
    """Run CLIPSeg with multiple text queries simultaneously.
    Returns dict: query → logits (H×W)."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(frame_rgb)
    results = {}
    # CLIPSeg supports batch text queries with same image
    inputs = processor(images=[img] * len(queries), text=queries,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits  # (N, H_out, W_out)
    for i, q in enumerate(queries):
        l = logits[i].cpu().numpy().astype(np.float32)
        if l.shape != (IMG_H, IMG_W):
            l = cv2.resize(l, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        results[q] = l
    return results


def extract_centroid(prob, sigmoid_threshold=0.5, min_blob_pixels=30):
    """Extract centroid from probability map. Returns (cx, cy, confidence, n_pixels)."""
    confidence = float(prob.max())
    mask = prob > sigmoid_threshold
    n_above = int(mask.sum())
    if n_above < min_blob_pixels:
        return None, None, confidence, 0

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return None, None, confidence, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    blob_mask = labels == best_label
    blob_area = int(areas[best_label - 1])

    if blob_area < min_blob_pixels:
        return None, None, confidence, 0

    ys, xs = np.where(blob_mask)
    w = prob[blob_mask]
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))
    return cx, cy, confidence, blob_area


class SinglePrompt:
    """Baseline: single positive prompt."""
    def __init__(self, sigmoid_thresh=0.5):
        self.sigmoid_thresh = sigmoid_thresh
        self.name = f"single_t{sigmoid_thresh}"

    def detect(self, frame, obj_name, processor, model, device):
        logits = get_logits(frame, obj_name, processor, model, device)
        prob = 1.0 / (1.0 + np.exp(-logits))
        cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
        return cx, cy, conf, npx


class DualPrompt:
    """Dual-prompt: positive - max(negatives). PEACE-style.

    score = sigmoid(logit_pos - max(logit_neg1, logit_neg2, ...))
    """
    def __init__(self, neg_prompts=None, sigmoid_thresh=0.5, margin=0.0):
        self.neg_prompts = neg_prompts or ["wall", "floor", "ceiling", "background"]
        self.sigmoid_thresh = sigmoid_thresh
        self.margin = margin
        self.name = f"dual_t{sigmoid_thresh}_m{margin}"

    def detect(self, frame, obj_name, processor, model, device):
        all_queries = [obj_name] + self.neg_prompts
        logits_dict = get_multi_query_logits(frame, all_queries, processor, model, device)

        pos_logits = logits_dict[obj_name]
        neg_logits = np.stack([logits_dict[q] for q in self.neg_prompts], axis=0)
        max_neg = neg_logits.max(axis=0)

        # Differential score: positive - negative - margin
        diff = pos_logits - max_neg - self.margin
        prob = 1.0 / (1.0 + np.exp(-diff))

        cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
        return cx, cy, conf, npx


class EnhancedPrompt:
    """Try different text prompts for the same object."""
    def __init__(self, prompt_template=None, sigmoid_thresh=0.5):
        self.prompt_template = prompt_template
        self.sigmoid_thresh = sigmoid_thresh
        self.name = f"enhanced_t{sigmoid_thresh}"

    def detect(self, frame, obj_name, processor, model, device):
        # Generate enhanced prompt
        if self.prompt_template:
            query = self.prompt_template.format(obj=obj_name)
        else:
            query = f"a photo of a {obj_name}"
        logits = get_logits(frame, query, processor, model, device)
        prob = 1.0 / (1.0 + np.exp(-logits))
        cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
        return cx, cy, conf, npx


class MultiPromptMax:
    """Try multiple prompts, take the one with highest confidence."""
    def __init__(self, extra_prompts=None, sigmoid_thresh=0.5):
        self.extra_prompts = extra_prompts or []
        self.sigmoid_thresh = sigmoid_thresh
        self.name = f"multimax_t{sigmoid_thresh}"

    def detect(self, frame, obj_name, processor, model, device):
        prompts = [obj_name] + [p.format(obj=obj_name) if "{obj}" in p else p
                                 for p in self.extra_prompts]
        logits_dict = get_multi_query_logits(frame, prompts, processor, model, device)

        best_cx, best_cy, best_conf, best_npx = None, None, 0.0, 0
        for q, logits in logits_dict.items():
            prob = 1.0 / (1.0 + np.exp(-logits))
            cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
            if conf > best_conf:
                best_cx, best_cy, best_conf, best_npx = cx, cy, conf, npx
        return best_cx, best_cy, best_conf, best_npx


class MultiPromptAvg:
    """Average logits from multiple prompts before sigmoid."""
    def __init__(self, extra_prompts=None, sigmoid_thresh=0.5):
        self.extra_prompts = extra_prompts or []
        self.sigmoid_thresh = sigmoid_thresh
        self.name = f"multiavg_t{sigmoid_thresh}"

    def detect(self, frame, obj_name, processor, model, device):
        prompts = [obj_name] + [p.format(obj=obj_name) if "{obj}" in p else p
                                 for p in self.extra_prompts]
        logits_dict = get_multi_query_logits(frame, prompts, processor, model, device)

        avg_logits = np.mean(list(logits_dict.values()), axis=0)
        prob = 1.0 / (1.0 + np.exp(-avg_logits))
        cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
        return cx, cy, conf, npx


class DualPromptTuned:
    """Dual-prompt with object-specific negative prompts."""
    def __init__(self, sigmoid_thresh=0.5):
        self.sigmoid_thresh = sigmoid_thresh
        self.name = f"dual_tuned_t{sigmoid_thresh}"

    def _get_neg_prompts(self, obj_name):
        """Object-specific negatives."""
        base = ["wall", "floor", "ceiling"]
        if "clock" in obj_name:
            return base + ["lamp", "light fixture", "round object on wall"]
        elif "leafblower" in obj_name:
            return base + ["plant", "furniture", "vacuum cleaner"]
        elif "drill" in obj_name:
            return base + ["tool on table", "metal object", "small appliance"]
        return base + ["background", "other object"]

    def detect(self, frame, obj_name, processor, model, device):
        neg_prompts = self._get_neg_prompts(obj_name)
        all_queries = [obj_name] + neg_prompts
        logits_dict = get_multi_query_logits(frame, all_queries, processor, model, device)

        pos_logits = logits_dict[obj_name]
        neg_logits = np.stack([logits_dict[q] for q in neg_prompts], axis=0)
        max_neg = neg_logits.max(axis=0)

        diff = pos_logits - max_neg
        prob = 1.0 / (1.0 + np.exp(-diff))
        cx, cy, conf, npx = extract_centroid(prob, self.sigmoid_thresh)
        return cx, cy, conf, npx


# ── Data Loading ──

def find_eval_dir_with_npz():
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
    """Load all cases with RGB videos + centroid logs."""
    cases = []
    for case_dir in sorted(os.listdir(eval_dir)):
        case_path = os.path.join(eval_dir, case_dir)
        if not os.path.isdir(case_path):
            continue

        # Find RGB video
        rgb_vids = glob.glob(os.path.join(case_path, "*_rgb.mp4"))
        if not rgb_vids:
            continue

        # Find centroid log (for GT)
        log_files = glob.glob(os.path.join(case_path, "*_centroid_log.json"))
        if not log_files:
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


def run_detection_experiment(case, strategy, processor, model, device,
                             conf_gate=0.60, subsample=1):
    """Run a detection strategy on every frame of a video.

    Returns per-frame results with detection vs GT comparison.
    """
    cap = cv2.VideoCapture(case["video_path"])
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {case['video_path']}")
        return None

    centroids = case["centroids"]
    obj_name = case["object"]
    n_frames = len(centroids)

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        if frame_idx % subsample != 0:
            frame_idx += 1
            continue

        # Videos are at double resolution — resize to model input
        if frame.shape[:2] != (IMG_H, IMG_W):
            frame_small = cv2.resize(frame, (IMG_W, IMG_H))
        else:
            frame_small = frame

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        cx, cy, conf, npx = strategy.detect(frame_rgb, obj_name,
                                             processor, model, device)

        detected = (npx > 0 and conf >= conf_gate)

        # GT from centroid log
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        err = None
        if detected and gt_in and gt_u is not None:
            err = float(np.hypot(cx - gt_u, cy - gt_v))

        results.append({
            "frame": frame_idx,
            "conf": conf,
            "npx": npx,
            "detected": detected,
            "cx": cx,
            "cy": cy,
            "gt_in": gt_in,
            "err_px": err,
        })

        frame_idx += 1

    cap.release()

    # Summary
    gt_frames = [r for r in results if r["gt_in"]]
    det_frames = [r for r in results if r["detected"]]
    true_pos = [r for r in results if r["detected"] and r["gt_in"]]
    false_pos = [r for r in results if r["detected"] and not r["gt_in"]]
    errs = [r["err_px"] for r in results if r["err_px"] is not None]
    confs = [r["conf"] for r in results]

    summary = {
        "n_frames": len(results),
        "n_gt_visible": len(gt_frames),
        "n_detected": len(det_frames),
        "n_true_pos": len(true_pos),
        "n_false_pos": len(false_pos),
        "detection_rate": len(true_pos) / max(len(gt_frames), 1),
        "false_pos_rate": len(false_pos) / max(len(results) - len(gt_frames), 1)
                          if len(results) > len(gt_frames) else 0.0,
        "err_mean": float(np.mean(errs)) if errs else None,
        "err_median": float(np.median(errs)) if errs else None,
        "err_p90": float(np.percentile(errs, 90)) if errs else None,
        "conf_mean": float(np.mean(confs)),
        "conf_max": float(np.max(confs)),
        "conf_p75": float(np.percentile(confs, 75)),
    }

    return {"results": results, "summary": summary}


def build_strategies():
    """Build all detection strategies to test."""
    strategies = {}

    # 1. Baseline single-prompt at different thresholds
    for t in [0.3, 0.4, 0.5]:
        s = SinglePrompt(sigmoid_thresh=t)
        strategies[s.name] = s

    # 2. Dual-prompt with generic negatives
    for t in [0.3, 0.4, 0.5]:
        s = DualPrompt(sigmoid_thresh=t)
        strategies[s.name] = s

    # 3. Dual-prompt with margin
    for m in [1.0, 2.0]:
        s = DualPrompt(sigmoid_thresh=0.5, margin=m)
        strategies[s.name] = s

    # 4. Dual-prompt with object-specific negatives
    for t in [0.3, 0.5]:
        s = DualPromptTuned(sigmoid_thresh=t)
        strategies[s.name] = s

    # 5. Enhanced prompts
    templates = [
        "a photo of a {obj}",
        "a {obj} in an indoor room",
        "a close-up of a {obj}",
        "{obj} target object",
    ]
    for tmpl in templates:
        s = EnhancedPrompt(prompt_template=tmpl, sigmoid_thresh=0.5)
        s.name = f"prompt_{tmpl[:30].replace(' ','_')}_t0.5"
        strategies[s.name] = s

    # 6. Multi-prompt max
    s = MultiPromptMax(
        extra_prompts=["a photo of a {obj}", "a {obj} in a room"],
        sigmoid_thresh=0.5
    )
    strategies[s.name] = s

    # 7. Multi-prompt average
    s = MultiPromptAvg(
        extra_prompts=["a photo of a {obj}", "a {obj} in a room"],
        sigmoid_thresh=0.5
    )
    strategies[s.name] = s

    return strategies


def main():
    t0 = time.time()

    eval_dir = find_eval_dir_with_npz()
    if eval_dir is None:
        print("ERROR: No eval directory found")
        return

    print(f"Loading from: {eval_dir}")
    cases = load_cases(eval_dir)
    print(f"Found {len(cases)} cases:")
    for c in cases:
        print(f"  {c['name']}: {len(c['centroids'])} frames, obj={c['object']}")

    if not cases:
        print("ERROR: No cases found")
        return

    # Load CLIPSeg
    print("\nLoading CLIPSeg model...")
    processor, model, device = load_clipseg()
    print(f"CLIPSeg loaded on {device}")

    strategies = build_strategies()
    print(f"\nTesting {len(strategies)} detection strategies x {len(cases)} cases")

    conf_gates = [0.50, 0.60, 0.65, 0.70]

    all_results = {}
    for strat_name, strategy in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name}")
        print(f"{'='*60}")
        for case in cases:
            for cg in conf_gates:
                key = f"{strat_name}_cg{cg:.2f}"
                res = run_detection_experiment(
                    case, strategy, processor, model, device,
                    conf_gate=cg, subsample=1
                )
                if res is None:
                    continue
                s = res["summary"]
                if key not in all_results:
                    all_results[key] = {}
                all_results[key][case["name"]] = s

                det_rate = s["detection_rate"] * 100
                fp_rate = s["false_pos_rate"] * 100
                err_med = f"{s['err_median']:.1f}" if s['err_median'] else "-"
                print(f"  {case['name']:<40} cg={cg:.2f}  "
                      f"detect={det_rate:5.1f}%  FP={fp_rate:4.1f}%  "
                      f"err_med={err_med:>6}px  "
                      f"conf_max={s['conf_max']:.3f}  "
                      f"n_det={s['n_detected']}/{s['n_frames']}")

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Aggregate table
    print(f"\n{'='*110}")
    print(f"{'Strategy+gate':<55} {'detect%':>7} {'FP%':>5} {'err_med':>8} "
          f"{'err_mean':>8} {'err_p90':>8}")
    print(f"{'-'*110}")

    agg = {}
    for key in sorted(all_results.keys()):
        case_results = all_results[key]
        det_rates = []
        fp_rates = []
        err_meds = []
        err_means = []
        err_p90s = []

        for cname, s in case_results.items():
            det_rates.append(s["detection_rate"] * 100)
            fp_rates.append(s["false_pos_rate"] * 100)
            if s["err_median"] is not None:
                err_meds.append(s["err_median"])
            if s["err_mean"] is not None:
                err_means.append(s["err_mean"])
            if s["err_p90"] is not None:
                err_p90s.append(s["err_p90"])

        avg_det = np.mean(det_rates) if det_rates else 0
        avg_fp = np.mean(fp_rates) if fp_rates else 0
        avg_err_med = np.mean(err_meds) if err_meds else None
        avg_err_mean = np.mean(err_means) if err_means else None
        avg_err_p90 = np.mean(err_p90s) if err_p90s else None

        agg[key] = {
            "detect_pct": avg_det, "fp_pct": avg_fp,
            "err_median": avg_err_med, "err_mean": avg_err_mean,
            "err_p90": avg_err_p90,
        }

        m = f"{avg_err_med:.1f}" if avg_err_med else "-"
        n = f"{avg_err_mean:.1f}" if avg_err_mean else "-"
        p = f"{avg_err_p90:.1f}" if avg_err_p90 else "-"
        print(f"{key:<55} {avg_det:>7.1f} {avg_fp:>5.1f} {m:>8} {n:>8} {p:>8}")

    print(f"{'='*110}")

    # Find best by detection rate (among configs with FP < 5%)
    safe_configs = {k: v for k, v in agg.items() if v["fp_pct"] < 5.0}
    if safe_configs:
        best_det = max(safe_configs, key=lambda k: safe_configs[k]["detect_pct"])
        print(f"\n*** BEST (safe, FP<5%): {best_det} — "
              f"detect={safe_configs[best_det]['detect_pct']:.1f}% ***")

    # Save results
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"clipseg_grid_{ts}.json")
    save_data = {
        "all_results": all_results,
        "aggregate": agg,
        "meta": {
            "eval_dir": eval_dir,
            "n_cases": len(cases),
            "n_strategies": len(strategies),
            "elapsed_s": elapsed,
            "timestamp": ts,
        }
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
