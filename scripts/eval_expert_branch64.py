#!/usr/bin/env python3
"""Expert (VehicleRateMPC) on clock branch 64 — with video + plotly."""
import os, sys, time
import numpy as np
import torch
import imageio
from torchvision.transforms import Resize

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_all_branches, _evaluate_run, _get_pkl,
    _preload_bc_trajectories, _make_terminal_fn, _save_benchmark_plotly,
    SUCCESS_RADIUS,
)
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from scipy.spatial import cKDTree

SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
PILOT_NAME = "InstinctJester"
POLICY_NAME = "vrmpc_rrt"
FRAME_NAME = "carl"
BRANCH_ID = 64
OBJ_NAME = "green clock"
FPS = 20
transform = Resize((720, 1280))

OUT_DIR = os.path.join(WORKSPACE, "cohorts", "SSV_DAGGER_HW1_15BR",
                       "visualizations", "expert_branch64")
os.makedirs(OUT_DIR, exist_ok=True)


def save_video(frames_np, path):
    n = frames_np.shape[0]
    out = torch.zeros((n, 720, 1280, 3))
    t = torch.from_numpy(frames_np)
    for i in range(n):
        img = t[i].permute(2, 0, 1)
        img = transform(img)
        out[i] = img.permute(1, 2, 0)
    imageio.mimwrite(path, out.numpy().astype("uint8"), fps=FPS)


def main():
    print("Loading scene...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    flights = [[SCENE, SCENE]]
    _preload_bc_trajectories(BC_COHORT, flights, SCENES_CFG)

    obj_idx = queries.index(OBJ_NAME)
    obj_target = obj_targets[obj_idx]

    pkl_data = _get_pkl(SCENE, OBJ_NAME, SCENES_CFG)
    all_branches = _load_all_branches(
        SCENE, OBJ_NAME, os.path.join(WORKSPACE, "cohorts", BC_COHORT),
        SCENES_CFG, pkl_data["tXUi"])

    tXUi = all_branches[BRANCH_ID]
    expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)

    terminal_fn, term_info = _make_terminal_fn(
        obj_target, pc_tree,
        env_min=scene_data.get("env_min"),
        env_max=scene_data.get("env_max"),
    )

    print(f"Running expert on '{OBJ_NAME}' branch={BRANCH_ID}...")
    _t = time.time()
    result = simulator.simulate(
        policy=expert, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
        x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=OBJ_NAME,
        vision_processor=None, verbose=False,
        early_stop_fn=terminal_fn,
    )
    Tro, Xro, Uro, Iro = result[0], result[1], result[2], result[3]
    sim_s = time.time() - _t

    ev = _evaluate_run(Xro, obj_target, epcds_arr,
                       env_min=scene_data.get("env_min"),
                       env_max=scene_data.get("env_max"),
                       tXUi=tXUi, idx0=0)

    status = "OK" if ev["success"] else ("COLL" if ev["collision"] else "MISS")
    reason = term_info.get("reason", "timeout")
    print(f"  [{status}]  goal={ev['goal_dist']:.2f}m  min={ev['min_goal_dist']:.2f}m"
          f"  fov={ev['fov_pct']:.0%}  stop={reason}"
          f"  clearance={ev['min_clearance']:.2f}m@step{ev['min_clearance_step']}"
          f"  ({sim_s:.0f}s)")

    # Save plotly
    plotly_path = os.path.join(OUT_DIR, "expert_clock_branch64.html")
    _save_benchmark_plotly(
        obj_runs=[(Xro.copy(), ev, tXUi.copy())],
        obj_target=obj_target,
        simulator=simulator,
        obj_name=f"Expert -- {OBJ_NAME} branch={BRANCH_ID} ({status})",
        save_path=plotly_path,
    )
    print(f"  Plotly -> {plotly_path}")

    # Save videos
    for ch_name, frames in Iro.items():
        if ch_name == "depth_raw":
            continue
        vid_path = os.path.join(OUT_DIR, f"expert_clock_b64_{ch_name}.mp4")
        save_video(frames, vid_path)
        print(f"  Video  -> {vid_path}")

    print("Done.")


if __name__ == "__main__":
    main()
