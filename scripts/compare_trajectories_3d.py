#!/usr/bin/env python3
"""
Compare Expert (MPC), BC Baseline, and DAgger Best trajectories in 3D Plotly.
Also generates per-timestep numerical analysis (matplotlib).

NEW: "Expert Takeover" paths — at waypoints along DAgger/BC trajectories,
shows what the expert MPC would do if it took control from the deviated state.

Usage:
  conda run -n FiGS python scripts/compare_trajectories_3d.py

Produces per-object HTML (3D) and PNG (2D metrics) in cohorts/<DAGGER_COHORT>/visualizations/
"""

import os, sys, yaml, json, glob
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ──
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(WORKSPACE)  # /data/erwinpi/SINGER
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

os.environ.setdefault("ACADOS_SOURCE_DIR", "/data/erwinpi/FiGS-Standalone/acados")
ld = os.environ.get("LD_LIBRARY_PATH", "")
if "/data/erwinpi/FiGS-Standalone/acados/lib" not in ld:
    os.environ["LD_LIBRARY_PATH"] = ld + ":/data/erwinpi/FiGS-Standalone/acados/lib"

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.tsampling.build_rrt_dataset as bd
from figs.utilities.trajectory_helper import process_branch
import figs.scene_editing.scene_editing_utils as scdt
from sousvide.control.pilot import Pilot

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Config ──
SCENE_NAME = "flightroom_ssv_exp"
FRAME_NAME = "carl"
ROLLOUT_NAME = "baseline"
POLICY_NAME = "vrmpc_rrt"

BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_CENTROID_V9"
PILOT_NAME = "InstinctJester"

N_BRANCHES = 5  # per object
SEED = 42

SUCCESS_RADIUS = 2.3
COLLISION_RADIUS = 0.15
HORIZONTAL_FOV = np.radians(85)

# Expert takeover waypoints (fractions of trajectory length)
TAKEOVER_FRACTIONS = [0.25, 0.50, 0.75]

SCENES_CFG_DIR = os.path.join(WORKSPACE, "configs", "scenes")
OUT_DIR = os.path.join(WORKSPACE, "cohorts", DAGGER_COHORT, "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)


def load_scene():
    """Load scene, simulator, and object targets."""
    print(f"[viz] Loading scene '{SCENE_NAME}'...")
    simulator = Simulator(SCENE_NAME, ROLLOUT_NAME)
    simulator.load_frame(FRAME_NAME)

    with open(os.path.join(SCENES_CFG_DIR, f"{SCENE_NAME}.yml")) as f:
        sc = yaml.safe_load(f)
    queries = sc.get("queries", [])
    similarities = sc.get("similarities", None)
    obj_targets, _, epcds_list, epcds_arr = bd.get_objectives(
        simulator.gsplat, queries, similarities, False
    )
    env_min = np.array(sc.get("minbound", [-1e6]*3), dtype=float)
    env_max = np.array(sc.get("maxbound", [1e6]*3), dtype=float)
    print(f"[viz] Scene loaded — {len(queries)} objects")
    return simulator, queries, obj_targets, epcds_arr, env_min, env_max, sc


def load_bc_branches(scene_name, queries, obj_targets):
    """Load BC trajectory branches for each object."""
    rollout_dir = os.path.join(WORKSPACE, "cohorts", BC_COHORT, "rollout_data", scene_name)
    traj_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories*.pt")))
    if not traj_files:
        print(f"[viz] WARNING: No trajectory files in {rollout_dir}")
        return {}

    obj_locs = np.array([np.squeeze(t) for t in obj_targets])
    per_obj = {i: [] for i in range(len(queries))}

    for tf in traj_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best_obj = int(np.argmin(dists))
        if float(dists[best_obj]) < 5.0:
            per_obj[best_obj].append(tXUd)

    result = {}
    for obj_idx, obj_name in enumerate(queries):
        branches = per_obj[obj_idx]
        if branches:
            result[obj_name] = branches
            print(f"[viz] '{obj_name}': {len(branches)} BC branches")
    return result


def load_pilot(cohort_name, model_filename="model.pth"):
    """Load a pilot with the specified model."""
    roster_dir = os.path.join(WORKSPACE, "cohorts", cohort_name, "roster", PILOT_NAME)
    model_path = os.path.join(roster_dir, model_filename)

    pilot = Pilot(cohort_name, PILOT_NAME)
    pilot.model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    pilot.model.to(DEVICE)
    pilot.model.eval()
    return pilot


def reset_pilot(pilot):
    """Reset pilot hidden state between runs."""
    pilot.hy_flag = False
    pilot.hy_idx = 0
    if hasattr(pilot, 'DxU') and pilot.DxU is not None:
        pilot.DxU.zero_()
    if hasattr(pilot, 'Znn') and pilot.Znn is not None:
        pilot.Znn.zero_()


def run_simulation(simulator, policy, tXUi, obj_name, t0=None, x0=None):
    """Run a simulation. Can override t0/x0 for takeover runs."""
    _t0 = float(tXUi[0, 0]) if t0 is None else t0
    tf = float(tXUi[0, -1])
    _x0 = tXUi[1:11, 0].copy() if x0 is None else x0.copy()
    if tf <= _t0:
        return None, None
    result = simulator.simulate(
        policy=policy, t0=_t0, tf=tf, x0=_x0,
        obj=np.zeros((18, 1)), query=obj_name,
        vision_processor=None, verbose=False,
    )
    return result[0], result[1]  # Tro, Xro


def run_expert_takeover(simulator, tXUi, obj_name, Xro_source, Tro_source):
    """Run expert MPC from waypoints along a source trajectory (DAgger/BC).

    Returns list of (waypoint_step, Xro_takeover) tuples.
    """
    N = Xro_source.shape[1]
    tf = float(tXUi[0, -1])
    takeovers = []

    for frac in TAKEOVER_FRACTIONS:
        step = int(frac * N)
        if step >= N - 10:  # need enough remaining time
            continue

        x0_takeover = Xro_source[:10, step].copy()
        t0_takeover = float(Tro_source[step]) if Tro_source is not None else float(tXUi[0, 0]) + frac * (tf - float(tXUi[0, 0]))

        if tf - t0_takeover < 0.5:  # skip if <0.5s remaining
            continue

        expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)
        Tro_tk, Xro_tk = run_simulation(
            simulator, expert, tXUi, obj_name,
            t0=t0_takeover, x0=x0_takeover,
        )
        if Xro_tk is not None:
            takeovers.append((step, frac, Xro_tk))

    return takeovers


def evaluate_run(Xro, obj_target, epcds_arr, env_min, env_max, tXUi):
    """Evaluation of a run — returns metrics dict."""
    positions = Xro[:3, :].T
    obj_flat = np.asarray(obj_target).flatten()[:3]
    goal_dists = np.linalg.norm(positions - obj_flat, axis=1)
    min_gd = float(goal_dists.min())
    final_gd = float(goal_dists[-1])

    in_zone = goal_dists <= SUCCESS_RADIUS
    reached = bool(in_zone.any())
    first_entry = int(np.argmax(in_zone)) if reached else -1

    # Collision detection
    collided = False
    coll_step = Xro.shape[1]
    if epcds_arr.shape[0] > 0:
        Xro_t = torch.from_numpy(Xro[:3].T).float().to(DEVICE)
        pc_t = torch.from_numpy(epcds_arr).float().to(DEVICE)
        dists = torch.cdist(Xro_t, pc_t)
        mask = (dists < COLLISION_RADIUS).any(dim=1)
        if mask.any():
            collided = True
            coll_step = int(mask.nonzero()[0][0])
        del Xro_t, pc_t, dists

    # OOB
    if env_min is not None and env_max is not None:
        below = (positions < env_min).any(axis=1)
        above = (positions > env_max).any(axis=1)
        oob_mask = below | above
        if oob_mask.any():
            oob_step = int(np.argmax(oob_mask))
            coll_step = min(coll_step, oob_step)
            collided = True

    coll_before = collided and (not reached or coll_step < first_entry)
    success = reached and not coll_before

    # Per-timestep metrics for numerical analysis
    # Position deviation from reference trajectory
    N_actual = Xro.shape[1]
    N_ref = tXUi.shape[1]
    ref_positions = tXUi[1:4, :].T  # [N_ref, 3]
    # Align by time: interpolate reference to match actual timestep count
    t_norm_actual = np.linspace(0, 1, N_actual)
    t_norm_ref = np.linspace(0, 1, N_ref)
    ref_interp = np.column_stack([
        np.interp(t_norm_actual, t_norm_ref, ref_positions[:, d])
        for d in range(3)
    ])
    pos_dev = np.linalg.norm(positions - ref_interp, axis=1)

    # Orientation: yaw error to goal
    qw, qx, qy, qz = Xro[6], Xro[7], Xro[8], Xro[9]
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    goal_vec = obj_flat[:2] - positions[:, :2]
    required_yaw = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])
    yaw_err = np.abs(yaw - required_yaw)
    yaw_err = np.minimum(yaw_err, 2*np.pi - yaw_err)

    # FOV: goal within half-FOV of heading
    fov_in = yaw_err <= (HORIZONTAL_FOV / 2)

    return {
        "success": success, "collision": coll_before,
        "goal_dist": final_gd, "min_goal_dist": min_gd,
        "first_entry": first_entry, "coll_step": coll_step,
        # Per-timestep
        "goal_dists": goal_dists,
        "pos_dev": pos_dev,
        "yaw_err_deg": np.degrees(yaw_err),
        "fov_in": fov_in,
    }


def get_point_cloud(simulator):
    """Get rescaled environment point cloud for visualization (world coordinates)."""
    epcds_pcd, _, _, _, _, _ = scdt.rescale_point_cloud(simulator.gsplat)
    pts = np.asarray(epcds_pcd.points)
    cols = np.clip(np.asarray(epcds_pcd.colors), 0, 1)
    return pts, cols


def create_comparison_figure(
    pts, cols, obj_target, obj_name,
    expert_runs, bc_runs, dagger_runs,
    reference_branches,
    dagger_takeovers=None,
    bc_takeovers=None,
    scene_bounds=None,
):
    """Create a 3D Plotly figure comparing all three policies + expert takeover paths."""
    rgb = (cols * 255).astype(int)
    rgb_strs = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    fig = go.Figure(layout=dict(width=3200, height=2400))

    # Point cloud (subsample for performance)
    n_pts = len(pts)
    step = max(1, n_pts // 100000)
    idx = np.arange(0, n_pts, step)
    fig.add_trace(go.Scatter3d(
        x=pts[idx, 0], y=pts[idx, 1], z=pts[idx, 2],
        mode="markers",
        marker=dict(size=1.5, color=[rgb_strs[i] for i in idx], opacity=0.6),
        name="Environment", showlegend=False,
        hoverinfo="skip",
    ))

    # Object marker
    obj_flat = np.asarray(obj_target).flatten()
    fig.add_trace(go.Scatter3d(
        x=[obj_flat[0]], y=[obj_flat[1]], z=[obj_flat[2]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="diamond"),
        name="Object Target",
    ))

    # Success zone circle
    theta = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=obj_flat[0] + SUCCESS_RADIUS * np.cos(theta),
        y=obj_flat[1] + SUCCESS_RADIUS * np.sin(theta),
        z=np.full(200, obj_flat[2]),
        mode="lines",
        line=dict(color="green", width=3, dash="dash"),
        name=f"Success Zone (r={SUCCESS_RADIUS}m)",
    ))

    # Reference trajectories (thin, gray)
    for i, tXUi in enumerate(reference_branches):
        ref_pos = tXUi[1:4, :].T
        fig.add_trace(go.Scatter3d(
            x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="Reference" if i == 0 else None,
            showlegend=(i == 0),
            legendgroup="reference",
            opacity=0.4,
        ))

    # Policy trajectories
    policies = [
        ("Expert (MPC)", expert_runs, "blue", 5),
        ("BC Baseline", bc_runs, "orange", 4),
        ("DAgger Best", dagger_runs, "limegreen", 4),
    ]

    for policy_name, runs, color, width in policies:
        for i, (Xro, Tro, ev, _takeovers) in enumerate(runs):
            pos = Xro[:3, :].T
            success = ev["success"]
            collision = ev["collision"]
            dash = None if success else "dot"
            opacity = 1.0 if success else 0.5

            fig.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=policy_name if i == 0 else None,
                showlegend=(i == 0),
                legendgroup=policy_name,
                opacity=opacity,
                hovertemplate=(
                    f"{policy_name} Run {i}<br>"
                    f"{'OK' if success else 'FAIL'}<br>"
                    f"goal_dist={ev['goal_dist']:.2f}m<br>"
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
                ),
            ))

            # Start marker (small diamond)
            fig.add_trace(go.Scatter3d(
                x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                mode="markers",
                marker=dict(size=4, color=color, symbol="diamond"),
                showlegend=False, legendgroup=policy_name,
            ))

            # Endpoint markers
            end_pos = pos[-1]
            if success:
                fig.add_trace(go.Scatter3d(
                    x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
                    mode="markers",
                    marker=dict(size=6, color=color, symbol="circle"),
                    showlegend=False, legendgroup=policy_name,
                ))
            elif collision:
                fig.add_trace(go.Scatter3d(
                    x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
                    mode="markers",
                    marker=dict(size=8, color="red", symbol="x"),
                    showlegend=False, legendgroup=policy_name,
                ))

    # Expert takeover paths from DAgger states
    takeover_legend_shown = {"dagger": False, "bc": False}
    for source_label, runs, tk_color in [
        ("dagger", dagger_runs, "cyan"),
        ("bc", bc_runs, "magenta"),
    ]:
        for i, (Xro_src, Tro_src, ev_src, takeovers) in enumerate(runs):
            if not takeovers:
                continue
            for step, frac, Xro_tk in takeovers:
                pos_tk = Xro_tk[:3, :].T
                # Waypoint marker (where takeover starts)
                wp = Xro_src[:3, step]
                show_legend = not takeover_legend_shown[source_label]
                legend_name = f"Expert Takeover (from {source_label.upper()})"

                # Waypoint diamond
                fig.add_trace(go.Scatter3d(
                    x=[wp[0]], y=[wp[1]], z=[wp[2]],
                    mode="markers",
                    marker=dict(size=6, color=tk_color, symbol="diamond",
                                line=dict(color="black", width=1)),
                    name=legend_name if show_legend else None,
                    showlegend=show_legend,
                    legendgroup=f"takeover_{source_label}",
                    hovertemplate=f"Takeover @ {frac:.0%} (run {i})<br>"
                                  f"x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>",
                ))

                # Takeover trajectory
                fig.add_trace(go.Scatter3d(
                    x=pos_tk[:, 0], y=pos_tk[:, 1], z=pos_tk[:, 2],
                    mode="lines",
                    line=dict(color=tk_color, width=3, dash="dashdot"),
                    showlegend=False,
                    legendgroup=f"takeover_{source_label}",
                    opacity=0.7,
                    hovertemplate=f"Expert takeover from {source_label} run {i} @ {frac:.0%}<br>"
                                  "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>",
                ))
                takeover_legend_shown[source_label] = True

    # Layout — match visualize_multiple_trajectories
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    if scene_bounds:
        mb = scene_bounds.get("minbound", [None]*3)
        xb = scene_bounds.get("maxbound", [None]*3)
        if mb[0] is not None: xmin = float(mb[0])
        if mb[1] is not None: ymin = float(mb[1])
        if mb[2] is not None: zmin = float(mb[2])
        if xb[0] is not None: xmax = float(xb[0])
        if xb[1] is not None: ymax = float(xb[1])
        if xb[2] is not None: zmax = float(xb[2])

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    R = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * 1.5
    az, el = np.deg2rad(45), np.deg2rad(10)
    eye = dict(
        x=R*np.cos(el)*np.cos(az),
        y=R*np.cos(el)*np.sin(az),
        z=R*np.sin(el),
    )

    fig.update_layout(
        scene_camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=dx, y=dy, z=dz),
            domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),
            xaxis=dict(
                title='', range=[xmin, xmax], autorange=False,
                showticklabels=False, showgrid=False, zeroline=False, showbackground=False,
            ),
            yaxis=dict(
                title='', range=[ymax, ymin], autorange=False,  # INVERTED
                showticklabels=False, showgrid=False, zeroline=False, showbackground=False,
            ),
            zaxis=dict(
                title='', range=[zmax, zmin], autorange=False,  # INVERTED
                showticklabels=False, showgrid=False, zeroline=False, showbackground=False,
            ),
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        width=3200,
        height=2400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=14)),
        title=f"<b>{obj_name}</b> — Expert (blue) vs BC (orange) vs DAgger (green) | Takeover: cyan/magenta",
    )

    return fig


def create_metrics_figure(obj_name, expert_runs, bc_runs, dagger_runs):
    """Create a matplotlib figure with per-timestep numerical analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{obj_name} — Per-Timestep Metrics", fontsize=16, fontweight="bold")

    policy_data = [
        ("Expert", expert_runs, "tab:blue", "-"),
        ("BC", bc_runs, "tab:orange", "--"),
        ("DAgger", dagger_runs, "tab:green", "-"),
    ]

    # --- Panel 1: Goal distance over normalized time ---
    ax = axes[0, 0]
    ax.set_title("Distance to Goal")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Distance (m)")
    ax.axhline(SUCCESS_RADIUS, color="green", ls="--", alpha=0.5, label=f"Success zone ({SUCCESS_RADIUS}m)")

    for label, runs, color, ls in policy_data:
        all_series = []
        for Xro, Tro, ev, _ in runs:
            t_norm = np.linspace(0, 1, len(ev["goal_dists"]))
            all_series.append(np.interp(np.linspace(0, 1, 101), t_norm, ev["goal_dists"]))
        if all_series:
            arr = np.array(all_series)
            t = np.linspace(0, 1, 101)
            ax.plot(t, arr.mean(axis=0), color=color, ls=ls, lw=2, label=label)
            ax.fill_between(t, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                            color=color, alpha=0.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Position deviation from reference ---
    ax = axes[0, 1]
    ax.set_title("Position Deviation from Reference")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Deviation (m)")

    for label, runs, color, ls in policy_data:
        all_series = []
        for Xro, Tro, ev, _ in runs:
            t_norm = np.linspace(0, 1, len(ev["pos_dev"]))
            all_series.append(np.interp(np.linspace(0, 1, 101), t_norm, ev["pos_dev"]))
        if all_series:
            arr = np.array(all_series)
            t = np.linspace(0, 1, 101)
            ax.plot(t, arr.mean(axis=0), color=color, ls=ls, lw=2, label=label)
            ax.fill_between(t, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                            color=color, alpha=0.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Yaw error to goal ---
    ax = axes[1, 0]
    ax.set_title("Yaw Error to Goal")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Yaw Error (deg)")
    ax.axhline(85/2, color="green", ls="--", alpha=0.5, label=f"FOV half-angle ({85/2:.0f} deg)")

    for label, runs, color, ls in policy_data:
        all_series = []
        for Xro, Tro, ev, _ in runs:
            t_norm = np.linspace(0, 1, len(ev["yaw_err_deg"]))
            all_series.append(np.interp(np.linspace(0, 1, 101), t_norm, ev["yaw_err_deg"]))
        if all_series:
            arr = np.array(all_series)
            t = np.linspace(0, 1, 101)
            ax.plot(t, arr.mean(axis=0), color=color, ls=ls, lw=2, label=label)
            ax.fill_between(t, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                            color=color, alpha=0.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Per-branch success/failure summary ---
    ax = axes[1, 1]
    ax.set_title("Per-Branch Outcome")
    labels_list = []
    expert_outcomes, bc_outcomes, dagger_outcomes = [], [], []
    for i in range(len(expert_runs)):
        labels_list.append(f"Br {i}")
        expert_outcomes.append(1 if expert_runs[i][2]["success"] else (-1 if expert_runs[i][2]["collision"] else 0))
        bc_outcomes.append(1 if bc_runs[i][2]["success"] else (-1 if bc_runs[i][2]["collision"] else 0))
        dagger_outcomes.append(1 if dagger_runs[i][2]["success"] else (-1 if dagger_runs[i][2]["collision"] else 0))

    x_pos = np.arange(len(labels_list))
    w = 0.25
    outcome_colors = {1: "green", 0: "gray", -1: "red"}

    for offset, outcomes, label in [(-w, expert_outcomes, "Expert"), (0, bc_outcomes, "BC"), (w, dagger_outcomes, "DAgger")]:
        for i, o in enumerate(outcomes):
            ax.bar(x_pos[i] + offset, 1, width=w, color=outcome_colors[o],
                   edgecolor="black", linewidth=0.5,
                   label=label if i == 0 else None)
            ax.text(x_pos[i] + offset, 0.5,
                    {1: "S", 0: "F", -1: "C"}[o],
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if o != 0 else "black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list)
    ax.set_yticks([])
    ax.set_ylabel("S=Success  F=Fail  C=Collision")
    # Deduplicate legend
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)
    ax.grid(False)

    plt.tight_layout()
    return fig


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load scene
    simulator, queries, obj_targets, epcds_arr, env_min, env_max, sc = load_scene()
    pts, cols = get_point_cloud(simulator)
    print(f"[viz] Point cloud: {len(pts)} points")

    # Load BC branches
    all_branches = load_bc_branches(SCENE_NAME, queries, obj_targets)

    # Load pilots
    print("[viz] Loading BC pilot...")
    bc_pilot = load_pilot(BC_COHORT)
    print("[viz] Loading DAgger best pilot...")
    dagger_pilot = load_pilot(DAGGER_COHORT, "model.pth")

    for obj_idx, obj_name in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"[viz] Processing '{obj_name}'")
        print(f"{'='*60}")

        obj_target = obj_targets[obj_idx] if obj_idx < len(obj_targets) else np.zeros(3)
        branches = all_branches.get(obj_name, [])
        if not branches:
            print(f"[viz] No branches for '{obj_name}', skipping")
            continue

        # Pick N_BRANCHES random branches
        n_avail = len(branches)
        chosen = np.random.choice(n_avail, size=min(N_BRANCHES, n_avail), replace=False)
        selected_branches = [branches[i] for i in chosen]

        # Each entry: (Xro, Tro, eval_dict, takeover_list)
        expert_runs, bc_runs, dagger_runs = [], [], []

        for br_idx, tXUi in enumerate(selected_branches):
            print(f"  Branch {br_idx+1}/{len(selected_branches)}...")

            # Expert (MPC)
            expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)
            Tro_exp, Xro_exp = run_simulation(simulator, expert, tXUi, obj_name)
            ev_exp = evaluate_run(Xro_exp, obj_target, epcds_arr, env_min, env_max, tXUi)
            expert_runs.append((Xro_exp, Tro_exp, ev_exp, []))  # no takeovers for expert
            print(f"    Expert: {'OK' if ev_exp['success'] else 'FAIL'}  gd={ev_exp['goal_dist']:.2f}m")

            # BC baseline
            reset_pilot(bc_pilot)
            Tro_bc, Xro_bc = run_simulation(simulator, bc_pilot, tXUi, obj_name)
            ev_bc = evaluate_run(Xro_bc, obj_target, epcds_arr, env_min, env_max, tXUi)
            # Expert takeover from BC states
            bc_takeovers = run_expert_takeover(simulator, tXUi, obj_name, Xro_bc, Tro_bc)
            bc_runs.append((Xro_bc, Tro_bc, ev_bc, bc_takeovers))
            print(f"    BC:     {'OK' if ev_bc['success'] else 'FAIL'}  gd={ev_bc['goal_dist']:.2f}m  ({len(bc_takeovers)} takeovers)")

            # DAgger best
            reset_pilot(dagger_pilot)
            Tro_dag, Xro_dag = run_simulation(simulator, dagger_pilot, tXUi, obj_name)
            ev_dag = evaluate_run(Xro_dag, obj_target, epcds_arr, env_min, env_max, tXUi)
            # Expert takeover from DAgger states
            dag_takeovers = run_expert_takeover(simulator, tXUi, obj_name, Xro_dag, Tro_dag)
            dagger_runs.append((Xro_dag, Tro_dag, ev_dag, dag_takeovers))
            print(f"    DAgger: {'OK' if ev_dag['success'] else 'FAIL'}  gd={ev_dag['goal_dist']:.2f}m  ({len(dag_takeovers)} takeovers)")

        # 3D figure
        fig_3d = create_comparison_figure(
            pts, cols, obj_target, obj_name,
            expert_runs, bc_runs, dagger_runs,
            selected_branches,
            scene_bounds=sc,
        )

        safe_name = obj_name.replace(" ", "_")[:40]
        html_path = os.path.join(OUT_DIR, f"compare_{safe_name}.html")
        fig_3d.write_html(html_path, auto_open=False)
        print(f"[viz] 3D saved  -> {html_path}")

        # 2D metrics figure
        fig_2d = create_metrics_figure(obj_name, expert_runs, bc_runs, dagger_runs)
        png_path = os.path.join(OUT_DIR, f"metrics_{safe_name}.png")
        fig_2d.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig_2d)
        print(f"[viz] Metrics   -> {png_path}")

        # Print summary
        exp_sr = sum(1 for _, _, ev, _ in expert_runs if ev["success"]) / len(expert_runs)
        bc_sr = sum(1 for _, _, ev, _ in bc_runs if ev["success"]) / len(bc_runs)
        dag_sr = sum(1 for _, _, ev, _ in dagger_runs if ev["success"]) / len(dagger_runs)
        print(f"  Summary: Expert {exp_sr:.0%}  BC {bc_sr:.0%}  DAgger {dag_sr:.0%}")

        # Print takeover insight: how often can expert recover from DAgger's deviated states?
        for source_label, runs in [("DAgger", dagger_runs), ("BC", bc_runs)]:
            n_tk = sum(len(tk) for _, _, _, tk in runs)
            if n_tk > 0:
                tk_successes = 0
                for _, _, _, takeovers in runs:
                    for step, frac, Xro_tk in takeovers:
                        ev_tk = evaluate_run(Xro_tk, obj_target, epcds_arr, env_min, env_max, tXUi)
                        if ev_tk["success"]:
                            tk_successes += 1
                print(f"  Expert-takeover from {source_label}: {tk_successes}/{n_tk} succeed ({100*tk_successes/n_tk:.0f}%)")

        torch.cuda.empty_cache()

    print(f"\n[viz] All done! Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
