#!/usr/bin/env python3
"""
3-way overlay plots only: Expert (green) + BC (red) + DAgger (orange)
on dramatic SEEN branches where BC fails but DAgger succeeds.
No full benchmark, just the overlay HTMLs.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=1 \
    conda run --no-capture-output -n FiGS python -u scripts/run_hw1_cross_cohort_overlay.py
"""
import os, sys, time, math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_all_branches, _evaluate_run, _swap_model, _get_pkl,
    _preload_bc_trajectories, _make_terminal_fn, _SCENE_CACHE,
    DEVICE, SUCCESS_RADIUS,
)
import figs.tsampling.build_rrt_dataset as bd
from sousvide.control.pilot import Pilot

SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_HW1_COMPREHENSIVE"
PILOT_NAME = "InstinctJester"
BC_MODEL = os.path.join(WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "model.pth")
DAGGER_MODEL = os.path.join(WORKSPACE, "cohorts", DAGGER_COHORT, "roster", PILOT_NAME, "model_round_8.pth")

# BC collision/miss -> DAgger success
OVERLAY_BRANCHES = {
    "green clock": [16, 86],
    "green and pink leafblower": [1, 65],
    "yellow handheld cordless drill on two boxes": [28],
}
DEV_LINE_EVERY = 8


def reset_pilot(pilot):
    pilot.hy_flag = False
    pilot.hy_idx = 0
    pilot.DxU.zero_()
    if hasattr(pilot, 'Znn') and isinstance(pilot.Znn, torch.Tensor):
        pilot.Znn.zero_()
    if hasattr(pilot, 'chunk_buf'):
        pilot.chunk_buf = None
        pilot.chunk_step = 0


def run_single(simulator, pilot, model_path, tXUi, obj_name, obj_target, scene_data):
    from scipy.spatial import cKDTree
    pilot = _swap_model(pilot, model_path)
    reset_pilot(pilot)
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None
    terminal_fn, _ = _make_terminal_fn(
        obj_target, pc_tree,
        env_min=scene_data.get("env_min"), env_max=scene_data.get("env_max"))
    result = simulator.simulate(
        policy=pilot, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
        x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=obj_name,
        vision_processor=None, verbose=False, early_stop_fn=terminal_fn)
    Xro = result[1]
    ev = _evaluate_run(Xro, obj_target, epcds_arr,
                       env_min=scene_data.get("env_min"),
                       env_max=scene_data.get("env_max"),
                       tXUi=tXUi, idx0=0)
    return Xro, ev


def build_overlay_html(expert_pos, bc_pos, dag_pos, bc_ev, dag_ev,
                       obj_target, simulator, obj_name, branch_id, save_path):
    import plotly.graph_objects as go
    goal_loc = np.asarray(obj_target).flatten()[:3]
    scene_cfg_file = os.path.join(SCENES_CFG, f"{SCENE}.yml")
    with open(scene_cfg_file) as f:
        sc = yaml.safe_load(f)
    _, _, epcds_list, _ = bd.get_objectives(
        simulator.gsplat, sc.get("queries", []), sc.get("similarities", None), False)
    radius_info = {"r1": sc.get("r1", 1.0), "r2": sc.get("r2", SUCCESS_RADIUS)}
    fig = bd.visualize_multiple_trajectories(
        [], epcds_list, goal_loc, radius_info, scene_cfg_file, simulator)

    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=goal_loc[0] + SUCCESS_RADIUS * np.cos(theta),
        y=goal_loc[1] + SUCCESS_RADIUS * np.sin(theta),
        z=np.full(200, goal_loc[2]),
        mode="lines", line=dict(color="limegreen", width=4, dash="dash"),
        name=f"Success zone (r={SUCCESS_RADIUS}m)"))

    # Expert (green)
    fig.add_trace(go.Scatter3d(
        x=expert_pos[:, 0], y=expert_pos[:, 1], z=expert_pos[:, 2],
        mode="lines", line=dict(color="green", width=7), name="Expert (RRT)"))
    fig.add_trace(go.Scatter3d(
        x=[expert_pos[-1, 0]], y=[expert_pos[-1, 1]], z=[expert_pos[-1, 2]],
        mode="markers", marker=dict(size=10, color="green", symbol="diamond-open"),
        showlegend=False))

    # BC (red)
    bc_s = "COLL" if bc_ev["collision"] else ("OK" if bc_ev["success"] else "MISS")
    fig.add_trace(go.Scatter3d(
        x=bc_pos[:, 0], y=bc_pos[:, 1], z=bc_pos[:, 2],
        mode="lines", line=dict(color="red", width=6),
        name=f"BC ({bc_s}, dev={bc_ev.get('mean_pos_dev', 0):.2f}m)"))
    fig.add_trace(go.Scatter3d(
        x=[bc_pos[-1, 0]], y=[bc_pos[-1, 1]], z=[bc_pos[-1, 2]],
        mode="markers", marker=dict(size=14, color="red", symbol="x"), showlegend=False))

    # DAgger (orange)
    dag_s = "COLL" if dag_ev["collision"] else ("OK" if dag_ev["success"] else "MISS")
    fig.add_trace(go.Scatter3d(
        x=dag_pos[:, 0], y=dag_pos[:, 1], z=dag_pos[:, 2],
        mode="lines", line=dict(color="orange", width=6),
        name=f"DAgger R8 ({dag_s}, dev={dag_ev.get('mean_pos_dev', 0):.2f}m)"))
    end_color = "orange" if dag_ev["success"] else "red"
    fig.add_trace(go.Scatter3d(
        x=[dag_pos[-1, 0]], y=[dag_pos[-1, 1]], z=[dag_pos[-1, 2]],
        mode="markers", marker=dict(size=14, color=end_color,
                                     symbol="circle-open" if dag_ev["success"] else "x"),
        showlegend=False))

    # Deviation lines
    n_exp, n_bc, n_dag = expert_pos.shape[0], bc_pos.shape[0], dag_pos.shape[0]
    first_bc = first_dag = True
    for step in range(0, n_exp, DEV_LINE_EVERY):
        if step < n_bc:
            fig.add_trace(go.Scatter3d(
                x=[expert_pos[step, 0], bc_pos[step, 0]],
                y=[expert_pos[step, 1], bc_pos[step, 1]],
                z=[expert_pos[step, 2], bc_pos[step, 2]],
                mode="lines+markers",
                line=dict(color="rgba(255,50,50,0.5)", width=4, dash="dot"),
                marker=dict(size=3, color="rgba(255,50,50,0.5)"),
                name="BC deviation" if first_bc else None,
                showlegend=first_bc, hoverinfo="skip"))
            first_bc = False
        if step < n_dag:
            fig.add_trace(go.Scatter3d(
                x=[expert_pos[step, 0], dag_pos[step, 0]],
                y=[expert_pos[step, 1], dag_pos[step, 1]],
                z=[expert_pos[step, 2], dag_pos[step, 2]],
                mode="lines+markers",
                line=dict(color="rgba(255,165,0,0.5)", width=4, dash="dot"),
                marker=dict(size=3, color="rgba(255,165,0,0.5)"),
                name="DAgger deviation" if first_dag else None,
                showlegend=first_dag, hoverinfo="skip"))
            first_dag = False

    fig.add_trace(go.Scatter3d(
        x=[expert_pos[0, 0]], y=[expert_pos[0, 1]], z=[expert_pos[0, 2]],
        mode="markers", marker=dict(size=12, color="white", symbol="circle",
                                     line=dict(color="black", width=2)),
        name="Start"))

    fig.update_layout(
        title=f"<b>{obj_name} — branch {branch_id}</b><br>"
              f"<span style='font-size:14px'>"
              f"Expert (green) | BC ({bc_s}, red) | DAgger ({dag_s}, orange)</span>",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12)))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"  [overlay] -> {save_path}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "cohorts", DAGGER_COHORT, "visualizations",
                           f"overlay_hw1_{ts}")

    print("=" * 70)
    print("[Overlay plots: Expert + BC + DAgger-HW1 R8]")
    print(f"  5 dramatic branches, output -> {out_dir}")
    print("=" * 70)

    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    _preload_bc_trajectories(BC_COHORT, [[SCENE, SCENE]], SCENES_CFG)

    pilot = Pilot(BC_COHORT, PILOT_NAME)
    pilot.set_mode('deploy')

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in OVERLAY_BRANCHES:
            continue
        obj_target = obj_targets[obj_idx]
        pkl_data = _get_pkl(SCENE, obj_name, SCENES_CFG)
        if pkl_data is None:
            continue
        all_branches = _load_all_branches(
            SCENE, obj_name, os.path.join(WORKSPACE, "cohorts", BC_COHORT),
            SCENES_CFG, pkl_data["tXUi"])

        for bid in OVERLAY_BRANCHES[obj_name]:
            if bid >= len(all_branches):
                continue
            tXUi = all_branches[bid]
            expert_pos = tXUi[1:4, :].T
            print(f"\n  {obj_name} br{bid}...")

            Xro_bc, ev_bc = run_single(simulator, pilot, BC_MODEL, tXUi, obj_name, obj_target, scene_data)
            Xro_dag, ev_dag = run_single(simulator, pilot, DAGGER_MODEL, tXUi, obj_name, obj_target, scene_data)

            bc_s = "COLL" if ev_bc["collision"] else ("OK" if ev_bc["success"] else "MISS")
            dag_s = "COLL" if ev_dag["collision"] else ("OK" if ev_dag["success"] else "MISS")
            print(f"    BC={bc_s} (goal={ev_bc['goal_dist']:.1f}m)  DAgger={dag_s} (goal={ev_dag['goal_dist']:.1f}m)")

            short_obj = obj_name.replace(" ", "_")[:30]
            build_overlay_html(
                expert_pos, Xro_bc[:3, :].T, Xro_dag[:3, :].T,
                ev_bc, ev_dag, obj_target, simulator, obj_name, bid,
                save_path=os.path.join(out_dir, f"{short_obj}_br{bid}_overlay.html"))

    print(f"\n{'='*70}")
    print(f"Done. Overlays in {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
