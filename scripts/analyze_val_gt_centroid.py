"""
Analyze expert trajectory visibility on validation trajectories.

Same analysis as analyze_expert_gt_centroid.py but on val trajectories.
These trajectories don't have depth_at_gt, so occlusion detection is
skipped — only FOV visibility is checked.

GT target positions loaded from scene via _get_scene() — never hardcoded.
"""

import os
import sys
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import _get_scene

# ── Constants (from pilot.py _compute_centroid) ──
FX, FY = 462.956, 463.002
CX, CY = 323.076, 181.184
IMG_W, IMG_H = 640, 360
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


def project_and_classify(Xro, obj_pos):
    """
    For each timestep, compute GT bearing/elevation and FOV visibility.
    No occlusion check (no depth_at_gt available for val trajectories).
    """
    n_steps = Xro.shape[1]
    bearings = np.full(n_steps, np.nan)
    elevations = np.full(n_steps, np.nan)
    in_fov = np.zeros(n_steps, dtype=bool)
    visible = np.zeros(n_steps, dtype=bool)
    geometric_depth = np.full(n_steps, np.nan)
    u_pixels = np.full(n_steps, np.nan)
    v_pixels = np.full(n_steps, np.nan)

    for k in range(n_steps):
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
        u_pixels[k] = u
        v_pixels[k] = v

        if not (0 <= u < IMG_W and 0 <= v < IMG_H):
            continue

        in_fov[k] = True
        visible[k] = True  # no occlusion check possible
        bearings[k] = 2.0 * (u / IMG_W) - 1.0
        elevations[k] = 2.0 * (v / IMG_H) - 1.0

    # Heading
    qx, qy, qz, qw = Xro[6:10, :n_steps]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    dx = obj_pos[0] - Xro[0, :n_steps]
    dy = obj_pos[1] - Xro[1, :n_steps]
    required_yaw = np.arctan2(dy, dx)
    heading_error_signed = required_yaw - yaw
    heading_error_signed = (heading_error_signed + np.pi) % (2 * np.pi) - np.pi
    yaw_error = np.abs(heading_error_signed)

    return {
        "bearings": bearings,
        "elevations": elevations,
        "visible": visible,
        "in_fov": in_fov,
        "geometric_depth": geometric_depth,
        "u_pixels": u_pixels,
        "v_pixels": v_pixels,
        "yaw": yaw,
        "required_yaw": required_yaw,
        "yaw_error": yaw_error,
        "heading_error_signed": heading_error_signed,
    }


def load_val_trajectories(rollout_dir, gt_targets, short_names):
    """Load val trajectory files and group by object using course field."""
    val_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))

    all_results = []
    for tf in val_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None:
            continue
        if isinstance(tXUd, torch.Tensor):
            tXUd = tXUd.numpy()

        # Get course from rep data
        reps = data.get("data", [])
        if not reps or not isinstance(reps[0], dict):
            continue
        course = reps[0].get("course")
        if course not in gt_targets:
            continue
        obj_pos = gt_targets[course]
        # Expert trajectory state: tXUd rows 1:11 = [x,y,z,vx,vy,vz,qx,qy,qz,qw]
        Xro = tXUd[1:11, :]

        info = project_and_classify(Xro, obj_pos)
        obj_label = short_names[course]
        n = len(info["visible"])

        all_results.append({
            "object": obj_label,
            "course": course,
            "file": os.path.basename(tf),
            "n_steps": n,
            "visibility_rate": info["visible"].sum() / n,
            "in_fov_rate": info["in_fov"].sum() / n,
            "bearings_visible": info["bearings"][info["visible"]],
            "elevations_visible": info["elevations"][info["visible"]],
            "heading_error_when_visible": info["heading_error_signed"][info["visible"]],
            "yaw_error_when_visible": info["yaw_error"][info["visible"]],
            "info": info,
        })

    return all_results


def make_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    objects = ["Clock", "Leafblower", "Drill"]

    # ── PLOT 1: Visibility rate per trajectory ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Val Trajectory Visibility Rate (in FOV, no occlusion check)", fontsize=14)

    for ax, obj in zip(axes, objects):
        obj_results = [r for r in results if r["object"] == obj]
        rates = [r["visibility_rate"] for r in obj_results]
        if not rates:
            ax.set_title(f"{obj}: no data")
            continue
        ax.bar(range(len(rates)), rates, color=OBJECT_COLORS[obj], alpha=0.8)
        mean_vis = np.mean(rates)
        ax.axhline(mean_vis, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_vis:.1%}")
        ax.set_title(f"{obj} ({len(rates)} trajs)")
        ax.set_xlabel("Trajectory index")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        # Per-trajectory labels
        for i, r in enumerate(rates):
            ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=6, rotation=45)

    axes[0].set_ylabel("Fraction of timesteps visible")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_val_visibility_rate.png"), dpi=150)
    plt.close()
    print("Saved: 1_val_visibility_rate.png")

    # ── PLOT 2: Bearing distribution when visible ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Val GT Bearing Distribution When In FOV", fontsize=14)

    for ax, obj in zip(axes, objects):
        obj_results = [r for r in results if r["object"] == obj]
        if not obj_results:
            continue
        bearings = np.concatenate([r["bearings_visible"] for r in obj_results])
        if len(bearings) == 0:
            ax.set_title(f"{obj}: no visible samples")
            continue
        ax.hist(bearings, bins=40, range=(-1, 1), alpha=0.8,
                color=OBJECT_COLORS[obj], density=True)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_title(f"{obj} (n={len(bearings)} samples)")
        ax.set_xlabel("Bearing [-1=left, +1=right]")
        mean_b = np.mean(bearings)
        std_b = np.std(bearings)
        ax.text(0.02, 0.95, f"mean={mean_b:.3f}\nstd={std_b:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_val_bearing_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 2_val_bearing_distribution.png")

    # ── PLOT 3: Bearing vs heading alignment ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Val Bearing vs Heading Error When In FOV", fontsize=14)

    for ax, obj in zip(axes, objects):
        obj_results = [r for r in results if r["object"] == obj]
        if not obj_results:
            continue
        bearings = np.concatenate([r["bearings_visible"] for r in obj_results])
        heading_err = np.concatenate([r["heading_error_when_visible"] for r in obj_results])
        if len(bearings) == 0:
            continue

        agree = np.sign(bearings) == np.sign(heading_err)
        agree_pct = agree.mean()

        ax.scatter(bearings, np.degrees(heading_err), s=3, alpha=0.3,
                   color=OBJECT_COLORS[obj])
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.axhspan(0, 180, xmin=0.5, xmax=1.0, alpha=0.05, color="green")
        ax.axhspan(-180, 0, xmin=0, xmax=0.5, alpha=0.05, color="green")
        ax.axhspan(0, 180, xmin=0, xmax=0.5, alpha=0.05, color="red")
        ax.axhspan(-180, 0, xmin=0.5, xmax=1.0, alpha=0.05, color="red")
        ax.set_title(obj)
        ax.set_xlabel("Bearing [-1=left, +1=right]")
        ax.set_ylabel("Heading error to object (deg)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-180, 180)
        ax.text(0.02, 0.95, f"Agreement: {agree_pct:.1%}\nn={len(bearings)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_val_bearing_heading_alignment.png"), dpi=150)
    plt.close()
    print("Saved: 3_val_bearing_heading_alignment.png")

    # ── PLOT 4: Per-trajectory pixel position timeline ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("GT Pixel Position Over Time (u, v) — Val Trajectories", fontsize=14)

    for ax, obj in zip(axes, objects):
        obj_results = [r for r in results if r["object"] == obj]
        for i, r in enumerate(obj_results):
            info = r["info"]
            steps = np.arange(r["n_steps"])
            u = info["u_pixels"]
            v = info["v_pixels"]
            valid = ~np.isnan(u)
            in_frame = info["in_fov"]
            # Plot v pixel (vertical) which is the main issue for clock
            ax.plot(steps[valid], v[valid], '-', alpha=0.5, linewidth=0.8,
                    label=f"traj {i}" if i < 5 else None)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Top of image")
        ax.axhline(IMG_H, color="gray", linestyle="--", alpha=0.5, label="Bottom of image")
        if not obj_results:
            ax.set_title(f"{obj}: no data")
            continue
        ax.fill_between([0, max(r["n_steps"] for r in obj_results)],
                        0, IMG_H, alpha=0.05, color="green", label="FOV")
        ax.set_title(f"{obj} — v pixel (vertical)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("v pixel")
        ax.set_ylim(-200, IMG_H + 200)
        if obj == "Clock":
            ax.legend(fontsize=7, ncol=4, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_val_pixel_timeline.png"), dpi=150)
    plt.close()
    print("Saved: 4_val_pixel_timeline.png")

    # ── Summary stats ──
    print("\n" + "=" * 70)
    print("VAL TRAJECTORY ANALYSIS SUMMARY (no occlusion check)")
    print("=" * 70)
    for obj in objects:
        obj_results = [r for r in results if r["object"] == obj]
        n_traj = len(obj_results)
        if n_traj == 0:
            continue
        vis_rates = [r["visibility_rate"] for r in obj_results]
        bearings = np.concatenate([r["bearings_visible"] for r in obj_results])
        heading_errs = np.concatenate([r["heading_error_when_visible"] for r in obj_results])
        agree = (np.sign(bearings) == np.sign(heading_errs)).mean() if len(bearings) > 0 else 0

        print(f"\n{obj} ({n_traj} trajectories, {len(bearings)} visible samples)")
        print(f"  In-FOV rate:       {np.mean(vis_rates):.1%} +/- {np.std(vis_rates):.1%}")
        print(f"  Per-traj rates:    {' '.join(f'{r:.0%}' for r in vis_rates)}")
        if len(bearings) > 0:
            print(f"  Bearing mean/std:  {np.mean(bearings):.3f} / {np.std(bearings):.3f}")
            print(f"  Bearing-heading agreement: {agree:.1%}")
        # Altitude info
        for r in obj_results:
            z_mean = np.mean(r["info"]["geometric_depth"][~np.isnan(r["info"]["geometric_depth"])])
            print(f"    {r['file']}: mean depth={z_mean:.2f}m, vis={r['visibility_rate']:.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", default="/data/erwinpi/SINGER/cohorts/ssv_BC_CENTROID_V9/rollout_data/flightroom_ssv_exp")
    parser.add_argument("--output-dir", default="/data/erwinpi/SINGER/cohorts/ssv_BC_GT_CENTROID_V3/analysis/val_trajectories")
    parser.add_argument("--scene", default="flightroom_ssv_exp")
    args = parser.parse_args()

    gt_targets, short_names = load_gt_targets(args.scene)

    print("\nLoading val trajectories...")
    results = load_val_trajectories(args.rollout_dir, gt_targets, short_names)
    print(f"Loaded {len(results)} val trajectories")
    for obj in sorted(set(r["object"] for r in results)):
        n = sum(1 for r in results if r["object"] == obj)
        print(f"  {obj}: {n}")

    make_plots(results, args.output_dir)
