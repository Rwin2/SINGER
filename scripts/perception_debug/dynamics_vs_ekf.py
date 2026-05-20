#!/usr/bin/env python3
"""
Dynamics-only vs Detection+EKF comparison.

After first detection:
  1. DYN_DET: Dynamics-only from first detection (no further measurement updates)
  2. DYN_GT:  Dynamics-only from GT (true depth) — the DGT baseline
  3. DET:     Raw online detection (no filtering)
  4. EKF:     Detection + InvDepthEKF (measurement updates)
  5. GT:      Ground truth

This shows whether the jumps come from EKF updates or from bad init.

Uses OWL-ViT with baseline queries (best det rate from 3-way test).

Usage:
    cd /data/erwinpi/SINGER
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:$PYTHONPATH \
    conda run --no-capture-output -n FiGS python -u scripts/perception_debug/dynamics_vs_ekf.py
"""
import os, sys, json, glob, time
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

from perception_debug.kf_experiments import (
    InvDepthEKF, predict_state, IMG_W, IMG_H, FX, FY, CX, CY, T_C2B,
    _xv_to_T,
)

BASELINE_QUERIES = [
    "green clock",
    "green and pink leafblower",
    "yellow handheld cordless drill on two boxes",
]


# ── Dynamics-only tracker ──

class DynamicsOnly:
    """After init, only run predict_state — no measurement updates."""
    def __init__(self, init_rho=0.3, name="DYN"):
        self.init_rho = init_rho
        self.state = None
        self.prev_xcr = None
        self.name = name

    def reset(self):
        self.state = None
        self.prev_xcr = None

    def init_from_pixel(self, cx, cy, xcr, rho=None):
        b = 2.0 * cx / IMG_W - 1.0
        e = 2.0 * cy / IMG_H - 1.0
        self.state = np.array([b, e, rho if rho else self.init_rho])
        self.prev_xcr = xcr.copy()

    def step(self, xcr):
        if self.state is None:
            return None
        s_new = predict_state(self.state, self.prev_xcr, xcr)
        if s_new is not None:
            self.state = s_new
        self.prev_xcr = xcr.copy()
        u = (self.state[0] + 1.0) * IMG_W / 2.0
        v = (self.state[1] + 1.0) * IMG_H / 2.0
        return int(round(u)), int(round(v))

    def get_pixel(self):
        if self.state is None:
            return None
        u = (self.state[0] + 1.0) * IMG_W / 2.0
        v = (self.state[1] + 1.0) * IMG_H / 2.0
        return int(round(u)), int(round(v))


def compute_true_depth(xcr, obj_target):
    """Compute true depth of object in camera frame."""
    T_c2w = _xv_to_T(xcr) @ T_C2B
    T_w2c = np.linalg.inv(T_c2w)
    pt = T_w2c @ np.array([*np.squeeze(obj_target), 1.0])
    return -pt[2]


# ── Detector ──

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


# ── Data ──

def get_query(obj_name):
    obj_lower = obj_name.lower()
    for q in BASELINE_QUERIES:
        if q in obj_lower or obj_lower in q:
            return q
        q_words = set(q.split())
        obj_words = set(obj_lower.replace("_", " ").split())
        if len(q_words & obj_words) >= 2:
            return q
    return obj_name


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


def px_to_be(cx, cy):
    return 2.0 * cx / IMG_W - 1.0, 2.0 * cy / IMG_H - 1.0


def process_case(case, proc, mdl, dev, out_dir):
    cap = cv2.VideoCapture(case["video_path"])
    centroids = case["centroids"]
    obj_name = case["object"]
    Xro = case["Xro"]
    obj_target = case["obj_target"]
    n_frames = min(len(centroids), Xro.shape[1])
    query = get_query(obj_name)

    # Trackers
    dyn_det = DynamicsOnly(init_rho=0.3, name="DYN_DET")   # init from 1st detection, guessed depth
    dyn_gt = DynamicsOnly(name="DYN_GT")                     # init from GT, true depth
    ekf = InvDepthEKF(
        conf_gate=0.02, process_noise_be=0.005,
        process_noise_rho=0.001, meas_noise=0.03,
        init_rho=0.3, init_sigma_rho=0.3,
        innovation_gate=None, adaptive_R=False,
        temporal_decay_gate=False,
    )

    # Video
    out_path = os.path.join(out_dir, f"{case['name']}_dyn_vs_ekf.mp4")
    tmp_path = out_path.replace(".mp4", "_tmp.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (IMG_W, IMG_H))

    # Error logs
    errs = {
        "det": [], "ekf": [], "dyn_det": [], "dyn_gt": [],
    }
    first_det_done = False
    first_gt_done = False

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        if frame_bgr.shape[:2] != (IMG_H, IMG_W):
            frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vis = frame_bgr.copy()

        xcr = Xro[:, frame_idx]
        c = centroids[frame_idx]
        gt_u = c.get("gt_u")
        gt_v = c.get("gt_v")
        gt_in = c.get("gt_in_frame", False)

        # -- Online detection --
        dcx, dcy, dconf, darea = detect_owlvit(frame_rgb, query, proc, mdl, dev)
        detected = (darea > 0 and dconf >= 0.02)

        # -- Init DYN_DET from first detection --
        if detected and not first_det_done:
            dyn_det.init_from_pixel(dcx, dcy, xcr, rho=0.3)
            first_det_done = True

        # -- Init DYN_GT from first GT --
        if gt_in and gt_u is not None and not first_gt_done:
            true_depth = compute_true_depth(xcr, obj_target)
            true_rho = 1.0 / max(true_depth, 0.1)
            dyn_gt.init_from_pixel(gt_u, gt_v, xcr, rho=true_rho)
            first_gt_done = True

        # -- Step dynamics-only trackers --
        dyn_det_px = dyn_det.step(xcr) if first_det_done else None
        dyn_gt_px = dyn_gt.step(xcr) if first_gt_done else None

        # -- Step EKF --
        if detected:
            b, e = px_to_be(dcx, dcy)
            ekf_out = ekf.step(xcr, b, e, dconf, darea)
        else:
            ekf_out = ekf.step(xcr, 0, 0, 0.0, 0)

        ekf_px = None
        if ekf_out is not None:
            ekf_px = (ekf_out[0], ekf_out[1])

        # -- Errors (only when GT visible) --
        if gt_in and gt_u is not None:
            if detected:
                errs["det"].append(float(np.hypot(dcx - gt_u, dcy - gt_v)))
            if ekf_px is not None:
                errs["ekf"].append(float(np.hypot(ekf_px[0] - gt_u, ekf_px[1] - gt_v)))
            if dyn_det_px is not None:
                errs["dyn_det"].append(float(np.hypot(dyn_det_px[0] - gt_u, dyn_det_px[1] - gt_v)))
            if dyn_gt_px is not None:
                errs["dyn_gt"].append(float(np.hypot(dyn_gt_px[0] - gt_u, dyn_gt_px[1] - gt_v)))

        # -- Render --
        # GT = green circle
        if gt_in and gt_u is not None:
            cv2.circle(vis, (int(gt_u), int(gt_v)), 10, (0, 255, 0), 2)

        # DYN_GT = orange (dynamics from GT with true depth — the DGT baseline)
        if dyn_gt_px is not None:
            cv2.drawMarker(vis, dyn_gt_px, (0, 165, 255),
                          cv2.MARKER_DIAMOND, 12, 2)

        # DYN_DET = cyan (dynamics from 1st detection, guessed depth)
        if dyn_det_px is not None:
            cv2.drawMarker(vis, dyn_det_px, (255, 255, 0),
                          cv2.MARKER_TILTED_CROSS, 12, 2)

        # DET = white dot (raw online detection)
        if detected:
            cv2.circle(vis, (int(dcx), int(dcy)), 5, (255, 255, 255), -1)

        # EKF = magenta cross (detection + EKF updates)
        if ekf_px is not None:
            cv2.drawMarker(vis, (int(ekf_px[0]), int(ekf_px[1])),
                          (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

        # Error text
        err_parts = []
        if gt_in and gt_u is not None:
            if detected:
                err_parts.append(f"DET:{np.hypot(dcx-gt_u, dcy-gt_v):.0f}")
            if ekf_px:
                err_parts.append(f"EKF:{np.hypot(ekf_px[0]-gt_u, ekf_px[1]-gt_v):.0f}")
            if dyn_det_px:
                err_parts.append(f"DYN_D:{np.hypot(dyn_det_px[0]-gt_u, dyn_det_px[1]-gt_v):.0f}")
            if dyn_gt_px:
                err_parts.append(f"DGT:{np.hypot(dyn_gt_px[0]-gt_u, dyn_gt_px[1]-gt_v):.0f}")

        # Info
        cv2.putText(vis, f'Q: "{query}"', (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        if err_parts:
            cv2.putText(vis, "  ".join(err_parts), (5, IMG_H - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(vis, f"f{frame_idx}/{n_frames}", (5, IMG_H - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Legend
        ly = 30
        items = [
            ((0, 255, 0), "GT"),
            ((255, 255, 255), "DET"),
            ((255, 0, 255), "EKF"),
            ((255, 255, 0), "DYN_DET"),
            ((0, 165, 255), "DGT"),
        ]
        for i, (col, name) in enumerate(items):
            x = 5 + i * 70
            cv2.circle(vis, (x, ly), 4, col, -1)
            cv2.putText(vis, name, (x+8, ly+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    # H.264
    os.system(f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p "{out_path}" 2>/dev/null')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, out_path)

    # Stats
    def _s(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90)),
                "min": float(np.min(a)), "max": float(np.max(a))}

    return {
        "query": query, "n_frames": frame_idx,
        "det": _s(errs["det"]),
        "ekf": _s(errs["ekf"]),
        "dyn_det": _s(errs["dyn_det"]),
        "dyn_gt": _s(errs["dyn_gt"]),
        "video": out_path,
    }


def main():
    t0 = time.time()
    eval_dir = find_eval_dir()
    if not eval_dir:
        print("ERROR: No eval dir found"); return
    cases = load_cases(eval_dir)
    print(f"Loaded {len(cases)} cases")

    print("Loading OWL-ViT...")
    proc, mdl, dev = load_owlvit()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(WORKSPACE, "scripts", "perception_debug",
                           "visualizations", f"dyn_vs_ekf_{ts}")
    os.makedirs(vis_dir, exist_ok=True)

    all_results = {}
    for case in cases:
        query = get_query(case["object"])
        print(f"\n{'='*80}")
        print(f"CASE: {case['name']}  (query: \"{query}\")")
        print(f"{'='*80}")

        res = process_case(case, proc, mdl, dev, vis_dir)

        for key, label in [("det", "DET(raw)"), ("ekf", "DET+EKF"),
                           ("dyn_det", "DYN_DET"), ("dyn_gt", "DGT")]:
            r = res[key]
            med = f"{r['median']:.1f}" if r.get('median') is not None else "-"
            mean = f"{r['mean']:.1f}" if r.get('mean') is not None else "-"
            mx = f"{r['max']:.0f}" if r.get('max') is not None else "-"
            n = r.get('n', 0)
            print(f"  {label:<12}  n={n:>4}  mean={mean:>7}px  "
                  f"med={med:>7}px  max={mx:>5}px")

        all_results[case["name"]] = res

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n\n{'#'*90}")
    print(f"AGGREGATE")
    print(f"{'#'*90}")
    print(f"{'Tracker':<12}  {'mean':>8}  {'median':>8}  {'p75':>8}  {'max':>8}")
    print(f"{'-'*55}")

    for key, label in [("det", "DET(raw)"), ("ekf", "DET+EKF"),
                       ("dyn_det", "DYN_DET"), ("dyn_gt", "DGT")]:
        means = []; meds = []; p75s = []; maxes = []
        for _, v in all_results.items():
            r = v[key]
            if r.get("mean") is not None:
                means.append(r["mean"])
                meds.append(r["median"])
                p75s.append(r["p75"])
                maxes.append(r["max"])
        if means:
            print(f"{label:<12}  {np.mean(means):7.1f}px  {np.mean(meds):7.1f}px  "
                  f"{np.mean(p75s):7.1f}px  {np.mean(maxes):7.0f}px")
        else:
            print(f"{label:<12}  no data")

    print(f"{'#'*90}")
    print(f"\nVideos: {vis_dir}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_dir = os.path.join(WORKSPACE, "scripts", "perception_debug", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dyn_vs_ekf_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
