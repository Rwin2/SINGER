#!/usr/bin/env python3
"""Quick expert (VehicleRateMPC) evaluation on the 15 HW1 branches.
Checks if the expert achieves 100% success and has goal in FOV."""
import os, sys, time
import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_all_branches, _evaluate_run, _get_pkl,
    _preload_bc_trajectories, _make_terminal_fn,
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

SELECTED = {
    "green clock":        {"fail": [5, 115], "success": [47, 91, 64]},
    "green and pink leafblower": {"fail": [1, 111], "success": [37, 15, 81]},
    "yellow handheld cordless drill on two boxes": {"fail": [2, 27], "success": [10, 35, 17]},
}

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

    print("\n" + "=" * 70)
    print("Expert (VehicleRateMPC) evaluation on 15 HW1 branches")
    print("=" * 70)

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in SELECTED:
            continue
        obj_target = obj_targets[obj_idx]
        pkl_data = _get_pkl(SCENE, obj_name, SCENES_CFG)
        all_branches = _load_all_branches(
            SCENE, obj_name, os.path.join(WORKSPACE, "cohorts", BC_COHORT),
            SCENES_CFG, pkl_data["tXUi"])

        sel = SELECTED[obj_name]
        branch_ids = sel["fail"] + sel["success"]

        for br_id in branch_ids:
            tXUi = all_branches[br_id]
            expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)

            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )
            _t = time.time()
            result = simulator.simulate(
                policy=expert, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
                x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=obj_name,
                vision_processor=None, verbose=False,
                early_stop_fn=terminal_fn,
            )
            Xro = result[1]
            ev = _evaluate_run(Xro, obj_target, epcds_arr,
                               env_min=scene_data.get("env_min"),
                               env_max=scene_data.get("env_max"),
                               tXUi=tXUi, idx0=0)
            s = "OK" if ev["success"] else ("COLL" if ev["collision"] else "MISS")
            reason = term_info.get("reason", "timeout")
            fov = ev.get("fov_pct", float("nan"))
            gd = ev["goal_dist"]
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            sim_s = time.time() - _t
            print(f"  [EXPERT] {s:4s}  '{obj_name[:30]}'  branch={br_id}"
                  f"  d0={d0:.1f}m  goal={gd:.2f}m  min={ev['min_goal_dist']:.2f}m"
                  f"  fov={fov:.0%}  stop={reason}  ({sim_s:.0f}s)")

    print("\nDone.")

if __name__ == "__main__":
    main()
