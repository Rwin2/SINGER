"""
Analyze expert trajectory quality for occlusion-aware GT centroid BC dataset.

Produces 3 plots:
1. Non-occlusion rate per trajectory (histogram, per object)
2. Bearing distribution when visible
3. Bearing vs heading alignment when visible

Uses depth_at_gt from rollout data for true occlusion detection,
replicating the exact projection from pilot.py _compute_centroid.

GT target positions loaded from scene via _get_scene() — never hardcoded.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import _get_scene

# ── Constants (from pilot.py _compute_centroid) ──
FX, FY = 462.956, 463.002
CX, CY = 323.076, 181.184
IMG_W, IMG_H = 640, 360
OCCLUSION_THRESHOLD = 0.90
T_C2B = np.array([
    [ 0.0,  0.0, -1.0,  0.10],
    [ 1.0,  0.0,  0.0, -0.03],
    [ 0.0, -1.0,  0.0, -0.01],
    [ 0.0,  0.0,  0.0,  1.00],
])

OBJECT_COLORS = {
    "Clock": "#2196F3",
    "Leafblower": "#4CAF50",
    "Drill": "#FF9800",
}


def load_gt_targets(scene_name="flightroom_ssv_exp"):
    """Load GT 3D targets from the scene via _get_scene() — no hardcoding."""
    scenes_cfg = os.path.join(WORKSPACE, "configs", "scenes")
    scene_data = _get_scene(scene_name, scenes_cfg)
    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    gt_targets = {q: np.squeeze(t) for q, t in zip(queries, obj_targets)}
    # Build short names from queries
    short_names = {}
    for q in queries:
        if "clock" in q.lower():
            short_names[q] = "Clock"
        elif "leaf" in q.lower():
            short_names[q] = "Leafblower"
        elif "drill" in q.lower():
            short_names[q] = "Drill"
        else:
            short_names[q] = q[:15]
    print(f"Loaded GT targets from scene '{scene_name}':")
    for q, t in gt_targets.items():
        print(f"  {short_names[q]:15s} ({q}): {t.round(3)}")
    return gt_targets, short_names


def project_and_classify(Xro, depth_at_gt, obj_pos):
    """
    For each timestep, compute GT bearing/elevation and visibility.
    Replicates pilot.py _compute_centroid logic exactly.

    Returns:
        bearings: (N,) bearing in [-1, 1] (NaN if not visible)
        elevations: (N,) elevation in [-1, 1] (NaN if not visible)
        visible: (N,) bool — True if in FOV and not occluded
        in_fov: (N,) bool — True if in camera frame (regardless of occlusion)
        occluded: (N,) bool — True if in FOV but occluded
        heading_rates: (N-1,) yaw rate of the drone (rad/step)
    """
    n_steps = Xro.shape[1]
    n_data = min(n_steps, len(depth_at_gt))  # depth_at_gt has 120 entries, Xro has 121

    bearings = np.full(n_data, np.nan)
    elevations = np.full(n_data, np.nan)
    in_fov = np.zeros(n_data, dtype=bool)
    occluded = np.zeros(n_data, dtype=bool)
    visible = np.zeros(n_data, dtype=bool)
    geometric_depth = np.full(n_data, np.nan)

    for k in range(n_data):
        xcr = Xro[:, k]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(xcr[6:10]).as_matrix()
        T[:3, 3] = xcr[:3]
        T_c2w = T @ T_C2B
        T_w2c = np.linalg.inv(T_c2w)
        pt = T_w2c @ np.array([*obj_pos, 1.0])

        if pt[2] >= -0.01:  # behind camera
            continue

        depth = -pt[2]
        geometric_depth[k] = depth
        u = FX * pt[0] / depth + CX
        v = FY * (-pt[1]) / depth + CY

        if not (0 <= u < IMG_W and 0 <= v < IMG_H):
            continue

        in_fov[k] = True
        bearing = 2.0 * (u / IMG_W) - 1.0
        elevation = 2.0 * (v / IMG_H) - 1.0

        # Occlusion check
        d_rendered = depth_at_gt[k]
        if not np.isnan(d_rendered) and depth > 0.01:
            ratio = d_rendered / depth
            if ratio < OCCLUSION_THRESHOLD:
                occluded[k] = True
                continue

        visible[k] = True
        bearings[k] = bearing
        elevations[k] = elevation

    # Heading (yaw) from quaternion
    qx, qy, qz, qw = Xro[6:10, :n_data]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    yaw_rate = np.diff(yaw)  # (N-1,)

    # Required yaw to face object
    dx = obj_pos[0] - Xro[0, :n_data]
    dy = obj_pos[1] - Xro[1, :n_data]
    required_yaw = np.arctan2(dy, dx)

    # Yaw error (shortest angle)
    yaw_error = np.abs(yaw - required_yaw)
    yaw_error = np.where(yaw_error > np.pi, 2 * np.pi - yaw_error, yaw_error)

    # Bearing sign vs required heading direction
    # If bearing < 0 (object on left), drone should yaw left (positive yaw change in NED? depends on convention)
    # Simpler: just compare bearing sign with sign of (required_yaw - actual_yaw)
    heading_error_signed = required_yaw - yaw
    # Wrap to [-pi, pi]
    heading_error_signed = (heading_error_signed + np.pi) % (2 * np.pi) - np.pi

    return {
        "bearings": bearings,
        "elevations": elevations,
        "visible": visible,
        "in_fov": in_fov,
        "occluded": occluded,
        "geometric_depth": geometric_depth,
        "yaw": yaw,
        "yaw_rate": yaw_rate,
        "required_yaw": required_yaw,
        "yaw_error": yaw_error,
        "heading_error_signed": heading_error_signed,
    }


def load_all_trajectories(rollout_base, gt_targets, short_names):
    """Load all trajectory files and compute per-trajectory stats."""
    all_results = []

    for course_dir in sorted(os.listdir(rollout_base)):
        course_path = os.path.join(rollout_base, course_dir)
        if not os.path.isdir(course_path):
            continue

        traj_files = sorted([f for f in os.listdir(course_path)
                             if f.startswith("trajectories") and "val" not in f and f.endswith(".pt")])

        for tfile in traj_files:
            traj_data = torch.load(os.path.join(course_path, tfile),
                                   weights_only=False, map_location="cpu")
            for rep in traj_data["data"]:
                course = rep["course"]
                obj_pos = gt_targets.get(course)
                if obj_pos is None:
                    continue

                Xro = rep["Xro"]
                Uro = rep.get("Uro")
                depth_at_gt = rep.get("depth_at_gt")
                if depth_at_gt is None:
                    continue
                if isinstance(depth_at_gt, torch.Tensor):
                    depth_at_gt = depth_at_gt.numpy()

                info = project_and_classify(Xro, depth_at_gt, obj_pos)
                obj_label = short_names[course]

                n = len(info["visible"])

                # Extract yaw rate command wz = Uro[3,:] when visible
                # Uro has shape (4, N) = [uf, wx, wy, wz]
                wz_visible = np.array([])
                bearings_for_wz = np.array([])
                if Uro is not None and Uro.shape[1] >= n:
                    wz = Uro[3, :n]
                    wz_visible = wz[info["visible"]]
                    bearings_for_wz = info["bearings"][info["visible"]]

                all_results.append({
                    "object": obj_label,
                    "course": course,
                    "n_steps": n,
                    "visibility_rate": info["visible"].sum() / n,
                    "in_fov_rate": info["in_fov"].sum() / n,
                    "occlusion_rate": info["occluded"].sum() / max(1, info["in_fov"].sum()),
                    "bearings_visible": info["bearings"][info["visible"]],
                    "elevations_visible": info["elevations"][info["visible"]],
                    "heading_error_when_visible": info["heading_error_signed"][info["visible"]],
                    "yaw_error_when_visible": info["yaw_error"][info["visible"]],
                    "wz_visible": wz_visible,
                    "bearings_for_wz": bearings_for_wz,
                    "info": info,
                })

    return all_results


def make_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    objects = ["Clock", "Leafblower", "Drill"]

    # ═══════════════════════════════════════════════════════════════
    # PLOT 1: Non-occlusion (visibility) rate per trajectory
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Expert Trajectory Visibility Rate (in FOV & not occluded)", fontsize=14)

    for ax, obj in zip(axes, objects):
        rates = [r["visibility_rate"] for r in results if r["object"] == obj]
        fov_rates = [r["in_fov_rate"] for r in results if r["object"] == obj]
        ax.hist(rates, bins=20, range=(0, 1), alpha=0.8,
                color=OBJECT_COLORS[obj], label=f"Visible (not occluded)")
        ax.hist(fov_rates, bins=20, range=(0, 1), alpha=0.3,
                color="gray", label="In FOV (incl. occluded)")
        mean_vis = np.mean(rates)
        mean_fov = np.mean(fov_rates)
        ax.axvline(mean_vis, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean visible: {mean_vis:.1%}")
        ax.axvline(mean_fov, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Mean in FOV: {mean_fov:.1%}")
        ax.set_title(obj)
        ax.set_xlabel("Fraction of timesteps")
        ax.legend(fontsize=8)
        n = len(rates)
        occ_rates = [r["occlusion_rate"] for r in results if r["object"] == obj]
        mean_occ = np.mean(occ_rates) if occ_rates else 0
        ax.text(0.02, 0.95, f"n={n} trajs\nOccl|FOV: {mean_occ:.1%}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Number of trajectories")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_visibility_rate.png"), dpi=150)
    plt.close()
    print(f"Saved: 1_visibility_rate.png")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 2: Bearing distribution when visible
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("GT Bearing Distribution When Visible (not occluded)", fontsize=14)

    for ax, obj in zip(axes, objects):
        bearings = np.concatenate([r["bearings_visible"] for r in results if r["object"] == obj])
        ax.hist(bearings, bins=40, range=(-1, 1), alpha=0.8,
                color=OBJECT_COLORS[obj], density=True)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_title(f"{obj} (n={len(bearings)} samples)")
        ax.set_xlabel("Bearing [-1=left, +1=right]")
        mean_b = np.mean(bearings) if len(bearings) > 0 else 0
        std_b = np.std(bearings) if len(bearings) > 0 else 0
        ax.text(0.02, 0.95, f"mean={mean_b:.3f}\nstd={std_b:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_bearing_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved: 2_bearing_distribution.png")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 3: Bearing vs heading alignment when visible
    # ═══════════════════════════════════════════════════════════════
    # When object is visible: bearing tells "object is at angle B in image"
    # heading_error_signed tells "drone needs to yaw by H to face object"
    # If sign(bearing) == sign(heading_error), they agree (object on left, need to turn left)
    # If they disagree, expert is heading away from visible object
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Bearing vs Heading Error When Object is Visible", fontsize=14)

    for ax, obj in zip(axes, objects):
        bearings = np.concatenate([r["bearings_visible"] for r in results if r["object"] == obj])
        heading_err = np.concatenate([r["heading_error_when_visible"] for r in results if r["object"] == obj])

        if len(bearings) == 0:
            ax.set_title(f"{obj}: no visible samples")
            continue

        # Agreement: bearing and heading error have same sign
        agree = np.sign(bearings) == np.sign(heading_err)
        agree_pct = agree.mean()

        ax.scatter(bearings, np.degrees(heading_err), s=1, alpha=0.15,
                   color=OBJECT_COLORS[obj])
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)

        # Shade agreement quadrants
        ax.axhspan(0, 180, xmin=0.5, xmax=1.0, alpha=0.05, color="green")  # top-right: agree
        ax.axhspan(-180, 0, xmin=0, xmax=0.5, alpha=0.05, color="green")   # bottom-left: agree
        ax.axhspan(0, 180, xmin=0, xmax=0.5, alpha=0.05, color="red")      # top-left: disagree
        ax.axhspan(-180, 0, xmin=0.5, xmax=1.0, alpha=0.05, color="red")   # bottom-right: disagree

        ax.set_title(obj)
        ax.set_xlabel("Bearing [-1=left, +1=right]")
        ax.set_ylabel("Heading error to object (deg)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-180, 180)
        ax.text(0.02, 0.95, f"Agreement: {agree_pct:.1%}\nn={len(bearings)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_bearing_heading_alignment.png"), dpi=150)
    plt.close()
    print(f"Saved: 3_bearing_heading_alignment.png")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 4: GT Bearing vs Yaw Rate Command (wz = u[3])
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("GT Bearing vs Yaw Rate Command (wz) When Visible", fontsize=14)

    for ax, obj in zip(axes, objects):
        bearings = np.concatenate([r["bearings_for_wz"] for r in results if r["object"] == obj and len(r["bearings_for_wz"]) > 0])
        wz = np.concatenate([r["wz_visible"] for r in results if r["object"] == obj and len(r["wz_visible"]) > 0])

        if len(bearings) == 0:
            ax.set_title(f"{obj}: no data")
            continue

        agree = np.sign(bearings) == np.sign(wz)
        agree_pct = agree.mean()
        corr = np.corrcoef(bearings, wz)[0, 1] if len(bearings) > 1 else 0

        ax.scatter(bearings, wz, s=1, alpha=0.15, color=OBJECT_COLORS[obj])
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)

        # Shade agreement quadrants
        wz_lim = max(abs(wz.min()), abs(wz.max()), 0.5)
        ax.axhspan(0, wz_lim, xmin=0.5, xmax=1.0, alpha=0.05, color="green")
        ax.axhspan(-wz_lim, 0, xmin=0, xmax=0.5, alpha=0.05, color="green")
        ax.axhspan(0, wz_lim, xmin=0, xmax=0.5, alpha=0.05, color="red")
        ax.axhspan(-wz_lim, 0, xmin=0.5, xmax=1.0, alpha=0.05, color="red")

        ax.set_title(obj)
        ax.set_xlabel("GT Bearing [-1=left, +1=right]")
        ax.set_ylabel("Yaw rate cmd wz (rad/s)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-wz_lim, wz_lim)
        ax.text(0.02, 0.95, f"Sign agree: {agree_pct:.1%}\nCorr: {corr:.3f}\nn={len(bearings)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_bearing_vs_yaw_rate.png"), dpi=150)
    plt.close()
    print(f"Saved: 4_bearing_vs_yaw_rate.png")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 5: Heading Error vs Yaw Rate Command
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Heading Error vs Yaw Rate Command (wz) When Visible", fontsize=14)

    for ax, obj in zip(axes, objects):
        heading_err = np.concatenate([r["heading_error_when_visible"] for r in results if r["object"] == obj])
        wz = np.concatenate([r["wz_visible"] for r in results if r["object"] == obj and len(r["wz_visible"]) > 0])

        n_common = min(len(heading_err), len(wz))
        if n_common == 0:
            ax.set_title(f"{obj}: no data")
            continue
        heading_err = heading_err[:n_common]
        wz = wz[:n_common]

        agree = np.sign(heading_err) == np.sign(wz)
        agree_pct = agree.mean()
        corr = np.corrcoef(heading_err, wz)[0, 1] if n_common > 1 else 0

        ax.scatter(np.degrees(heading_err), wz, s=1, alpha=0.15, color=OBJECT_COLORS[obj])
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)

        wz_lim = max(abs(wz.min()), abs(wz.max()), 0.5)
        ax.set_title(obj)
        ax.set_xlabel("Heading error to object (deg)")
        ax.set_ylabel("Yaw rate cmd wz (rad/s)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-wz_lim, wz_lim)
        ax.text(0.02, 0.95, f"Sign agree: {agree_pct:.1%}\nCorr: {corr:.3f}\nn={n_common}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_heading_error_vs_yaw_rate.png"), dpi=150)
    plt.close()
    print(f"Saved: 5_heading_error_vs_yaw_rate.png")

    # ═══════════════════════════════════════════════════════════════
    # Summary stats
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERT TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 70)
    for obj in objects:
        obj_results = [r for r in results if r["object"] == obj]
        n_traj = len(obj_results)
        vis_rates = [r["visibility_rate"] for r in obj_results]
        fov_rates = [r["in_fov_rate"] for r in obj_results]
        occ_rates = [r["occlusion_rate"] for r in obj_results]
        bearings = np.concatenate([r["bearings_visible"] for r in obj_results])
        yaw_errs = np.concatenate([r["yaw_error_when_visible"] for r in obj_results])
        heading_errs = np.concatenate([r["heading_error_when_visible"] for r in obj_results])
        agree = (np.sign(bearings) == np.sign(heading_errs)).mean() if len(bearings) > 0 else 0

        print(f"\n{obj} ({n_traj} trajectories, {len(bearings)} visible samples)")
        print(f"  Visibility rate:   {np.mean(vis_rates):.1%} ± {np.std(vis_rates):.1%}")
        print(f"  In-FOV rate:       {np.mean(fov_rates):.1%} ± {np.std(fov_rates):.1%}")
        print(f"  Occlusion|FOV:     {np.mean(occ_rates):.1%}")
        print(f"  Bearing mean/std:  {np.mean(bearings):.3f} / {np.std(bearings):.3f}")
        print(f"  Yaw error (vis):   {np.degrees(np.mean(yaw_errs)):.1f}° ± {np.degrees(np.std(yaw_errs)):.1f}°")
        print(f"  Bearing-heading agreement: {agree:.1%}")

        wz_all = np.concatenate([r["wz_visible"] for r in obj_results if len(r["wz_visible"]) > 0])
        b_wz = np.concatenate([r["bearings_for_wz"] for r in obj_results if len(r["bearings_for_wz"]) > 0])
        if len(wz_all) > 1:
            wz_agree = (np.sign(b_wz) == np.sign(wz_all)).mean()
            wz_corr = np.corrcoef(b_wz, wz_all)[0, 1]
            print(f"  Bearing-wz agreement:  {wz_agree:.1%}")
            print(f"  Bearing-wz correlation: {wz_corr:.3f}")
            print(f"  wz mean/std:           {np.mean(wz_all):.4f} / {np.std(wz_all):.4f} rad/s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-base", default="/data/erwinpi/SINGER/cohorts/ssv_BC_GT_CENTROID_V3/rollout_data")
    parser.add_argument("--output-dir", default="/data/erwinpi/SINGER/cohorts/ssv_BC_GT_CENTROID_V3/analysis")
    parser.add_argument("--scene", default="flightroom_ssv_exp")
    args = parser.parse_args()

    gt_targets, short_names = load_gt_targets(args.scene)

    print("\nLoading trajectories...")
    results = load_all_trajectories(args.rollout_base, gt_targets, short_names)
    print(f"Loaded {len(results)} trajectories")

    make_plots(results, args.output_dir)
