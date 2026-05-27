#!/usr/bin/env python3
"""
Advanced detector experiments to push detection rate beyond 24%.

Tests:
1. OWL-ViT v2 (google/owlv2-base-patch16-ensemble) — better model
2. GroundingDINO (IDEA-Research/grounding-dino-tiny) — SOTA open-vocab detector
3. Hybrid OWL-ViT + CLIPSeg fusion (run both, pick best per frame)
4. Better OWL-ViT prompts (object-specific, richer descriptions)
5. Multi-threshold OWL-ViT (very low threshold + strong KF filtering)
6. ROI-guided re-detection (use KF prediction to crop, re-detect in crop)

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/advanced_detectors.py
"""
import os, sys, json, glob, time, traceback
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


# ── Detector Implementations ──

class OWLViTv1Detector:
    """Original OWL-ViT v1."""
    def __init__(self, threshold=0.02):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl = self.mdl.to(self.dev).eval()
        self.threshold = threshold

    def detect(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        queries = [[obj_name, f"a {obj_name}"]]
        inputs = self.proc(text=queries, images=img, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mdl(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(self.dev)
        results = self.proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)
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


class OWLViTv2Detector:
    """OWL-ViT v2 — better architecture, ensemble training."""
    def __init__(self, threshold=0.02):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        print("  Loading OWL-ViT v2 (owlv2-base-patch16-ensemble)...")
        self.proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.mdl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl = self.mdl.to(self.dev).eval()
        self.threshold = threshold

    def detect(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        queries = [[obj_name, f"a {obj_name}"]]
        inputs = self.proc(text=queries, images=img, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mdl(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(self.dev)
        results = self.proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)
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


class GroundingDINODetector:
    """GroundingDINO — state-of-art open-vocabulary detector."""
    def __init__(self, threshold=0.15, text_threshold=0.15):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("  Loading GroundingDINO (grounding-dino-tiny)...")
        self.proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.mdl = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl = self.mdl.to(self.dev).eval()
        self.threshold = threshold
        self.text_threshold = text_threshold

    def detect(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        text = f"{obj_name}."
        inputs = self.proc(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mdl(**inputs)
        results = self.proc.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]],
        )
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


class OWLViTRichPromptDetector:
    """OWL-ViT v1 with richer, object-specific prompts."""
    def __init__(self, threshold=0.02):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mdl = self.mdl.to(self.dev).eval()
        self.threshold = threshold
        # Richer prompts per object type
        self.prompt_map = {
            "green clock": [
                "green clock", "a green wall clock", "green analog clock",
                "round green clock on table", "green colored clock",
            ],
            "leafblower": [
                "leafblower", "a leaf blower", "leaf blower tool",
                "garden leaf blower", "handheld blower",
            ],
            "drill": [
                "drill", "a power drill", "electric drill",
                "cordless drill", "hand drill tool",
            ],
        }

    def _get_prompts(self, obj_name):
        for key, prompts in self.prompt_map.items():
            if key in obj_name.lower():
                return prompts
        return [obj_name, f"a {obj_name}", f"{obj_name} object"]

    def detect(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        prompts = self._get_prompts(obj_name)
        queries = [prompts]
        inputs = self.proc(text=queries, images=img, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mdl(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(self.dev)
        results = self.proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)
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


class HybridDetector:
    """Run OWL-ViT + CLIPSeg, pick best per frame based on confidence."""
    def __init__(self, owlvit_thresh=0.02, clipseg_thresh=0.5):
        from transformers import (OwlViTProcessor, OwlViTForObjectDetection,
                                  CLIPSegProcessor, CLIPSegForImageSegmentation)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # OWL-ViT
        self.owl_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_mdl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.owl_mdl = self.owl_mdl.to(self.dev).eval()
        self.owlvit_thresh = owlvit_thresh
        # CLIPSeg
        self.cseg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.cseg_mdl = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.cseg_mdl = self.cseg_mdl.to(self.dev).eval()
        self.clipseg_thresh = clipseg_thresh

    def _detect_owlvit(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        queries = [[obj_name, f"a {obj_name}"]]
        inputs = self.owl_proc(text=queries, images=img, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.owl_mdl(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(self.dev)
        results = self.owl_proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.owlvit_thresh)
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

    def _detect_clipseg(self, frame_rgb, obj_name):
        img = Image.fromarray(frame_rgb)
        inputs = self.cseg_proc(images=img, text=obj_name, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.cseg_mdl(**inputs).logits
        logits_np = logits.cpu().squeeze().numpy().astype(np.float32)
        if logits_np.shape != (IMG_H, IMG_W):
            logits_np = cv2.resize(logits_np, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        prob = 1.0 / (1.0 + np.exp(-logits_np))
        conf = float(prob.max())
        mask = prob > self.clipseg_thresh
        n_above = int(mask.sum())
        if n_above < 30:
            return None, None, conf, 0
        mask_u8 = mask.astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels <= 1:
            return None, None, conf, 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        blob_mask = labels == best_label
        blob_area = int(areas[best_label - 1])
        if blob_area < 30:
            return None, None, conf, 0
        ys, xs = np.where(blob_mask)
        w = prob[blob_mask]
        cx = float(np.average(xs, weights=w))
        cy = float(np.average(ys, weights=w))
        return cx, cy, conf, blob_area

    def detect(self, frame_rgb, obj_name):
        # Run both
        owl_cx, owl_cy, owl_conf, owl_area = self._detect_owlvit(frame_rgb, obj_name)
        cseg_cx, cseg_cy, cseg_conf, cseg_area = self._detect_clipseg(frame_rgb, obj_name)

        owl_valid = owl_area > 0 and owl_conf >= self.owlvit_thresh
        cseg_valid = cseg_area > 0 and cseg_conf >= 0.60

        if owl_valid and cseg_valid:
            # Both detected — pick OWL-ViT (more reliable based on experiments)
            # unless CLIPSeg confidence is much higher
            if cseg_conf > owl_conf + 0.3:
                return cseg_cx, cseg_cy, cseg_conf, cseg_area
            return owl_cx, owl_cy, owl_conf, owl_area
        elif owl_valid:
            return owl_cx, owl_cy, owl_conf, owl_area
        elif cseg_valid:
            return cseg_cx, cseg_cy, cseg_conf, cseg_area
        else:
            best_conf = max(owl_conf, cseg_conf)
            return None, None, best_conf, 0


# ── Test Runner ──

def run_test(case, detector, kf_kwargs):
    """Run detector + KF on one case."""
    kf = ConsensusInitEKF(**kf_kwargs)
    kf.reset()

    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    Xro = case["Xro"]
    obj_name = case["object"]
    n_frames = len(centroids)

    kf_errs = []
    det_errs = []
    n_det = 0
    n_tp = 0
    n_fp = 0
    n_gt_vis = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, conf, area = detector.detect(frame_rgb, obj_name)
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

        if gt_in:
            n_gt_vis += 1
            if detected:
                n_tp += 1
        if detected and not gt_in:
            n_fp += 1
        if detected:
            n_det += 1

        if gt_in and gt_u is not None:
            if kf_est is not None:
                kf_errs.append(float(np.hypot(kf_est[0] - gt_u, kf_est[1] - gt_v)))
            if detected and cx is not None:
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
        "kf": _s(kf_errs),
        "det": _s(det_errs),
        "n_frames": frame_idx,
        "n_gt_vis": n_gt_vis,
        "n_det": n_det,
        "n_tp": n_tp,
        "n_fp": n_fp,
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

    # KF config (best from previous experiments)
    kf_kwargs = dict(
        n_init=3, init_radius_px=50.0, max_coast_frames=20,
        conf_gate=0.05, temporal_decay_gate=True,
    )

    # Detectors to test
    detectors = {}

    print("\n--- Loading detectors ---")

    # 1. Baseline OWL-ViT v1
    print("Loading OWL-ViT v1 (baseline)...")
    detectors["owlvit_v1"] = OWLViTv1Detector(threshold=0.02)

    # 2. OWL-ViT v1 with rich prompts
    print("Loading OWL-ViT v1 + rich prompts...")
    detectors["owlvit_rich"] = OWLViTRichPromptDetector(threshold=0.02)

    # 3. OWL-ViT v2
    try:
        detectors["owlvit_v2"] = OWLViTv2Detector(threshold=0.02)
    except Exception as e:
        print(f"  OWL-ViT v2 failed to load: {e}")

    # 4. GroundingDINO
    try:
        detectors["grounding_dino"] = GroundingDINODetector(threshold=0.10, text_threshold=0.10)
    except Exception as e:
        print(f"  GroundingDINO failed to load: {e}")

    # 5. GroundingDINO with lower threshold
    try:
        detectors["grounding_dino_low"] = GroundingDINODetector(threshold=0.05, text_threshold=0.05)
    except Exception as e:
        print(f"  GroundingDINO low-thresh failed: {e}")

    # 6. Hybrid
    try:
        print("Loading Hybrid (OWL-ViT + CLIPSeg)...")
        detectors["hybrid_owl_cseg"] = HybridDetector(owlvit_thresh=0.02, clipseg_thresh=0.5)
    except Exception as e:
        print(f"  Hybrid failed: {e}")

    print(f"\nLoaded {len(detectors)} detectors: {list(detectors.keys())}")
    print(f"Testing {len(detectors)} detectors x {len(cases)} cases = {len(detectors)*len(cases)} runs")

    all_results = {}
    for det_name, detector in detectors.items():
        print(f"\n{'='*80}")
        print(f"DETECTOR: {det_name}")
        print(f"{'='*80}")
        case_results = {}
        for case in cases:
            t1 = time.time()
            try:
                res = run_test(case, detector, kf_kwargs)
            except Exception as e:
                print(f"  ERROR on {case['name']}: {e}")
                traceback.print_exc()
                res = {"kf": {"n": 0}, "det": {"n": 0}, "n_frames": 0,
                       "n_gt_vis": 0, "n_det": 0, "n_tp": 0, "n_fp": 0, "det_rate": 0}
            dt = time.time() - t1

            kf = res["kf"]
            det = res["det"]
            kf_m = f"{kf['mean']:.1f}" if kf.get("mean") else "-"
            kf_d = f"{kf['median']:.1f}" if kf.get("median") else "-"
            det_d = f"{det['median']:.1f}" if det.get("median") else "-"
            dr = f"{res['det_rate']*100:.1f}%"
            print(f"  {case['name']:<45}  KF={kf_m:>7}/{kf_d:>7}px  "
                  f"det_med={det_d:>7}px  det={dr:>6}  FP={res['n_fp']}  ({dt:.1f}s)")
            case_results[case["name"]] = res
        all_results[det_name] = case_results

    elapsed = time.time() - t0

    # ── Aggregate Table ──
    print(f"\n\n{'#'*100}")
    print(f"{'Detector':<25} {'KF mean':>8} {'KF med':>8} {'KF p90':>8} "
          f"{'det_med':>8} {'det%':>7} {'FP':>4} {'KF cov':>8}")
    print(f"{'-'*100}")

    for det_name, case_results in all_results.items():
        means, meds, p90s, det_ms, det_rs, fps = [], [], [], [], [], []
        kf_coverage = []
        for cname, res in case_results.items():
            kf = res["kf"]
            if kf.get("mean"):
                means.append(kf["mean"])
            if kf.get("median"):
                meds.append(kf["median"])
            if kf.get("p90"):
                p90s.append(kf["p90"])
            det = res["det"]
            if det.get("median"):
                det_ms.append(det["median"])
            det_rs.append(res["det_rate"] * 100)
            fps.append(res["n_fp"])
            if res["n_gt_vis"] > 0:
                kf_coverage.append(kf.get("n", 0) / res["n_gt_vis"] * 100)

        m = f"{np.mean(means):.1f}" if means else "-"
        d = f"{np.mean(meds):.1f}" if meds else "-"
        p = f"{np.mean(p90s):.1f}" if p90s else "-"
        dm = f"{np.mean(det_ms):.1f}" if det_ms else "-"
        det = f"{np.mean(det_rs):.1f}%"
        fp = f"{sum(fps)}"
        cov = f"{np.mean(kf_coverage):.1f}%" if kf_coverage else "-"
        print(f"{det_name:<25} {m:>8} {d:>8} {p:>8} {dm:>8} {det:>7} {fp:>4} {cov:>8}")

    print(f"{'#'*100}")
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Per-case comparison ──
    print(f"\n\n{'='*120}")
    print("PER-CASE COMPARISON (det rate %)")
    print(f"{'='*120}")
    det_names = list(all_results.keys())
    header = f"{'Case':<45}" + "".join(f"{n:>15}" for n in det_names)
    print(header)
    print("-" * 120)

    case_names = list(next(iter(all_results.values())).keys())
    for cname in case_names:
        parts = [f"{cname:<45}"]
        for det_name in det_names:
            res = all_results[det_name].get(cname, {})
            dr = res.get("det_rate", 0) * 100
            kf_m = res.get("kf", {}).get("mean")
            kf_s = f"{kf_m:.0f}px" if kf_m else "-"
            parts.append(f"{dr:>6.1f}%/{kf_s:>6}")

        print("".join(parts))

    # Save
    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"advanced_detectors_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
