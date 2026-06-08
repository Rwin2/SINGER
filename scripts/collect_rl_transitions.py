#!/usr/bin/env python3
"""
Collect IQL-ready transitions from the 17 failed R8 DAgger branches.

For each branch, runs BOTH expert (MPC) and learned pilot. At every step,
captures the 147-d Commander observation (xnn processed through VisionMLP +
HistoryEncoder) — the exact same observation the Commander sees during BC and
DAgger training.

Observations are captured by hooking into the Pilot's OODA loop:
  - Pilot runs:  pilot.OODA() already computes xnn; we just save it.
  - Expert runs:  same pattern as DAgger MixedPolicy — run pilot perception
                  alongside expert action.

Rewards are computed inline:
  -1 per step, +1000 if goal reached, -1000 if collision, +0.5 if GT visible.

Usage:
    cd /data/erwinpi/SINGER
    source activate FiGS
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=1 \
    python scripts/collect_rl_transitions.py
"""
import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import torch
from scipy.spatial import cKDTree

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))
sys.path.insert(0, os.path.join(WORKSPACE, "..", "FiGS-Standalone", "src"))

# ── Reuse existing code ───────────────────────────────────────────────
from sousvide.control.pilot import Pilot
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.simulator import Simulator
from sousvide.instruct.train_dagger import (
    _get_scene, _load_training_branches, _make_terminal_fn, _evaluate_run,
    COLLISION_RADIUS, SUCCESS_RADIUS,
)
from sousvide.visualize.analyze_simulated_experiments import (
    analyze_trajectory_performance,
)
from sousvide.flight.vision_processor_base import create_vision_processor

# GT visibility — reuse project_and_classify from existing script
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))
from analyze_expert_gt_centroid import project_and_classify

# ── Config ─────────────────────────────────────────────────────────────
SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_COMPREHENSIVE"
PILOT_NAME = "InstinctJester"

POLICY_NAME = "vrmpc_rrt"
FRAME_NAME = "carl"

OUTPUT_DIR = os.path.join(WORKSPACE, "cohorts", DAGGER_COHORT, "rl_transitions")

# The 17 failed branches from R8
FAILED_R8 = {
    "green clock": [5, 50, 71, 87, 88],
    "green and pink leafblower": [14, 29, 50, 52, 53, 68, 90, 91, 96, 101, 109],
    "yellow handheld cordless drill on two boxes": [7],
}

# Reward constants
REWARD_STEP = -1.0
REWARD_SUCCESS = 1000.0
REWARD_COLLISION = -1000.0
REWARD_VISIBLE = 0.5


# ── Pilot subclass that captures xnn ──────────────────────────────────

class XnnCapturePilot(Pilot):
    """Pilot that saves xnn at every step (instead of discarding it in control())."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xnn_log = []

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        unn, znn, adv, xnn, tsol = self.OODA(upr, tcr, xcr, obj, icr, zcr)
        self.xnn_log.append({k: v.detach().cpu() for k, v in xnn.items()})
        return unn, znn, adv, tsol

    def clear_log(self):
        self.xnn_log = []


class ExpertWithPerception:
    """Runs expert action but captures pilot's perception (xnn).

    Proxies all attributes (hz, etc.) from the expert so the simulator
    can access them transparently.
    """

    def __init__(self, expert, pilot):
        self.expert = expert
        self.pilot = pilot  # XnnCapturePilot

    def __getattr__(self, name):
        # Proxy any attribute the simulator needs (hz, etc.) from expert
        if name in ("expert", "pilot"):
            raise AttributeError(name)
        return getattr(self.expert, name)

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        # Run pilot perception to capture xnn
        _, _, _, xnn, _ = self.pilot.OODA(upr, tcr, xcr, obj, icr, zcr)
        self.pilot.xnn_log.append({k: v.detach().cpu() for k, v in xnn.items()})
        # Return expert's action
        return self.expert.control(tcr, xcr, upr, obj, icr, zcr)


# ── Extract 147-d Commander input from xnn ────────────────────────────

def extract_commander_input(pilot, xnn_log):
    """Run VisionMLP + HistoryEncoder on stored xnn to get 147-d Commander input.

    This is exactly what happens inside svnet.forward():
      z_par = HistoryEncoder(dxu_par)        # 70-d -> 8-d
      y_vis = VisionMLP(img_vis, tx_vis)     # (3,224,224) + 5-d -> 128-d
      commander_in = cat(tx_com, obj_com, z_par, y_vis)  # 8+3+8+128 = 147-d
    """
    device = next(pilot.model.parameters()).device
    commander_inputs = []

    with torch.no_grad():
        for xnn in xnn_log:
            tx_com = xnn["tx_com"].to(device).unsqueeze(0)
            obj_com = xnn["obj_com"].to(device).unsqueeze(0)
            dxu_par = xnn["dxu_par"].to(device).unsqueeze(0)
            img_vis = xnn["img_vis"].to(device).unsqueeze(0)
            tx_vis = xnn["tx_vis"].to(device).unsqueeze(0)

            _, z_par = pilot.model.network["HistoryEncoder"](dxu_par)
            y_vis, _ = pilot.model.network["VisionMLP"](img_vis, tx_vis)

            cin = torch.cat([tx_com, obj_com, z_par, y_vis], dim=-1)
            commander_inputs.append(cin.squeeze(0).cpu().numpy())

    return np.array(commander_inputs, dtype=np.float32)  # (T, 147)


# ── Build transitions with rewards ───────────────────────────────────

def build_transitions(Xro, Uro, Tro, eval_dict, gt_vis, commander_obs):
    """Build IQL-ready transitions with proper observations and rewards.

    Args:
        Xro: (10, T+1) state trajectory
        Uro: (4, T) actions
        Tro: (T+1,) timestamps
        eval_dict: from _evaluate_run (has collision_step, first_entry_step, success)
        gt_vis: from project_and_classify (has 'visible' array)
        commander_obs: (T, 147) Commander inputs from extract_commander_input

    Returns:
        dict with obs, action, next_obs, done, reward arrays
    """
    T = Uro.shape[1]

    # Observations: 147-d Commander input
    obs = commander_obs[:T]
    # next_obs: shift by 1. Last step uses the same obs (terminal).
    next_obs = np.zeros_like(obs)
    if T > 1:
        next_obs[:T - 1] = obs[1:]
    next_obs[T - 1] = obs[T - 1]  # terminal state

    # Actions: raw, no normalization (consistent with BC/DAgger Commander output)
    actions = Uro[:, :T].T.astype(np.float32)

    # Done flags
    done = np.zeros(T, dtype=bool)
    collision_step = eval_dict.get("collision_step", -1)
    first_entry = eval_dict.get("first_entry_step", -1)
    success = eval_dict.get("success", False)

    if collision_step >= 0 and collision_step < T:
        done[collision_step] = True
    elif success and 0 <= first_entry < T:
        done[first_entry] = True
    done[T - 1] = True  # episode boundary

    # Rewards: -1/step, +1000 success, -1000 collision, +0.5 if visible
    reward = np.full(T, REWARD_STEP, dtype=np.float32)

    # Visibility bonus
    if gt_vis is not None:
        visible = gt_vis["visible"]
        n_vis = min(T, len(visible))
        for t in range(n_vis):
            if visible[t]:
                reward[t] += REWARD_VISIBLE

    # Terminal rewards (at the step where the event occurs)
    if collision_step >= 0 and collision_step < T:
        reward[collision_step] += REWARD_COLLISION
    if success and 0 <= first_entry < T:
        reward[first_entry] += REWARD_SUCCESS

    # Raw state (for reference, not used as IQL observation)
    raw_state = Xro[:, :T].T.astype(np.float32)
    raw_next_state = Xro[:, 1:T + 1].T.astype(np.float32)

    return {
        "obs": obs,                           # (T, 147) Commander input
        "action": actions,                    # (T, 4) raw actions
        "next_obs": next_obs,                 # (T, 147) next Commander input
        "done": done,                         # (T,) episode boundaries
        "reward": reward,                     # (T,) step-wise reward
        "raw_state": raw_state,               # (T, 10) for diagnostics
        "raw_next_state": raw_next_state,     # (T, 10) for diagnostics
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  RL Transition Collection V2 — 147-d Commander Observations")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # ── 1. Load scene ──
    print("\n[1/5] Loading scene...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator = scene_data["simulator"]
    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    epcds_arr = scene_data["epcds_arr"]
    env_min = scene_data["env_min"]
    env_max = scene_data["env_max"]
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    # ── 2. Load training branches ──
    print("\n[2/5] Loading training branches...")
    all_branches = _load_training_branches(BC_COHORT, SCENE, SCENES_CFG)

    # ── 3. Load vision processor ──
    print("\n[3/5] Loading vision processor (CLIPSeg)...")
    vision_processor = create_vision_processor("clipseg")

    # ── 4. Load pilot with xnn capture ──
    print("\n[4/5] Loading pilot (DAgger best model)...")
    pilot = XnnCapturePilot(DAGGER_COHORT, PILOT_NAME)
    pilot.set_mode("deploy")
    print(f"  Model: cohorts/{DAGGER_COHORT}/roster/{PILOT_NAME}/model.pth")
    print(f"  Commander input dim: tx_com(8) + obj_com(3) + z_par(8) + y_vis(128) = 147")

    # ── 5. Build goal lookup ──
    query_to_idx = {q: i for i, q in enumerate(queries)}

    import yaml
    with open(os.path.join(SCENES_CFG, f"{SCENE}.yml")) as f:
        scene_cfg = yaml.safe_load(f)
    radii = scene_cfg["radii"]

    # ── 6. Simulate ──
    print(f"\n[5/5] Collecting transitions...")
    all_pilot_transitions = []
    all_expert_transitions = []
    summary = []

    for obj_name, branch_ids in FAILED_R8.items():
        matched_query = None
        for q in queries:
            if obj_name.lower() in q.lower() or q.lower() in obj_name.lower():
                matched_query = q
                break
        if matched_query is None:
            print(f"  WARNING: No match for '{obj_name}'")
            continue

        obj_idx = query_to_idx[matched_query]
        obj_target = np.squeeze(obj_targets[obj_idx])
        obj_branches = all_branches.get(matched_query, {})
        exclusion_radius = radii[obj_idx][0]

        print(f"\n  === {matched_query} ({len(branch_ids)} branches) ===")

        for br_id in branch_ids:
            if br_id not in obj_branches:
                print(f"  WARNING: br{br_id} not found")
                continue

            tXUi = obj_branches[br_id]
            d0 = np.linalg.norm(tXUi[1:4, 0] - obj_target[:3])
            print(f"\n  --- br{br_id} (d0={d0:.1f}m) ---")

            for policy_type in ["pilot", "expert"]:
                is_expert = (policy_type == "expert")

                # Build policy
                if is_expert:
                    expert_policy = VehicleRateMPC(
                        tXUi, POLICY_NAME, FRAME_NAME, "expert"
                    )
                    pilot.clear_log()
                    # DAgger pattern: pilot perception + expert action
                    policy = ExpertWithPerception(expert_policy, pilot)
                    print("Bypassing trajectory optimization. Using provided trajectory.")
                else:
                    pilot.clear_log()
                    policy = pilot

                # Early stopping
                early_stop_fn, term_info = _make_terminal_fn(
                    obj_target, pc_tree, env_min, env_max
                )
                simulator.load_frame(FRAME_NAME)
                term_info["reason"] = None
                term_info["step"] = -1

                # Simulate
                print(f"    [{policy_type}] Simulating...", end=" ", flush=True)
                t0_sim = time.time()
                Tro, Xro, Uro, Iro, Tsol, Adv = simulator.simulate(
                    policy,
                    tXUi[0, 0], tXUi[0, -1], tXUi[1:11, 0].copy(),
                    np.zeros((18, 1)),
                    query=matched_query,
                    vision_processor=vision_processor,
                    verbose=False,
                    early_stop_fn=early_stop_fn,
                    gt_target_3d=obj_target,
                )
                stop_reason = term_info.get("reason", "timeout")
                sim_s = time.time() - t0_sim

                # Evaluate
                eval_dict = _evaluate_run(
                    Xro, obj_target, epcds_arr,
                    env_min=env_min, env_max=env_max,
                    tXUi=tXUi, idx0=0,
                )

                # GT visibility
                depth_at_gt = Iro.get("depth_at_gt", None)
                gt_vis = None
                if depth_at_gt is not None:
                    gt_vis = project_and_classify(Xro, depth_at_gt, obj_target[:3])

                # Extract 147-d Commander input from captured xnn
                T = Uro.shape[1]
                xnn_log = pilot.xnn_log
                if len(xnn_log) < T:
                    print(f"WARNING: xnn_log ({len(xnn_log)}) < T ({T}), padding")
                    while len(xnn_log) < T:
                        xnn_log.append(xnn_log[-1])

                commander_obs = extract_commander_input(pilot, xnn_log[:T])
                print(f"obs={commander_obs.shape}", end=" ", flush=True)

                # Build transitions with rewards
                transitions = build_transitions(
                    Xro, Uro, Tro, eval_dict, gt_vis, commander_obs
                )

                # Add metadata
                transitions["object_idx"] = np.full(T, obj_idx, dtype=np.int32)
                transitions["branch_id"] = np.full(T, br_id, dtype=np.int32)

                # Status
                status = "OK" if eval_dict["success"] else (
                    "COLL" if eval_dict["collision"] else "MISS"
                )
                vis_pct = (gt_vis["visible"].mean()
                           if gt_vis is not None else float("nan"))
                ep_return = transitions["reward"].sum()

                print(f"{status}  T={T}  goal={eval_dict['goal_dist']:.2f}m  "
                      f"clr={eval_dict['min_clearance']:.2f}m  "
                      f"vis={vis_pct:.0%}  R={ep_return:.0f}  ({sim_s:.1f}s)")

                # Save per-branch .pt
                per_branch = {
                    "Tro": Tro, "Xro": Xro, "Uro": Uro,
                    "Iro": Iro,
                    "tXUd": tXUi, "obj": np.zeros((18, 1)),
                    "Ndata": T, "Tsol": Tsol, "Adv": Adv,
                    "course": SCENE, "objective": matched_query,
                    "eval": eval_dict,
                    "gt_visibility": gt_vis,
                    "transitions": transitions,
                    "xnn_log": xnn_log[:T],  # keep raw xnn for debugging
                }
                fname = f"br{br_id}_{matched_query.split()[0].lower()}_{policy_type}.pt"
                fpath = os.path.join(out_dir, fname)
                torch.save(per_branch, fpath)
                fsize = os.path.getsize(fpath) / 1024
                print(f"    -> {fname} ({fsize:.0f} KB)")

                # Accumulate
                if is_expert:
                    all_expert_transitions.append(transitions)
                else:
                    all_pilot_transitions.append(transitions)

                summary.append({
                    "object": matched_query,
                    "branch_id": br_id,
                    "policy": policy_type,
                    "status": status,
                    "T": T,
                    "goal_dist": eval_dict["goal_dist"],
                    "min_clearance": eval_dict["min_clearance"],
                    "gt_vis_pct": float(vis_pct) if not np.isnan(vis_pct) else None,
                    "episode_return": float(ep_return),
                    "stop_reason": stop_reason,
                    "file": fname,
                })

                # Free GPU memory
                del Iro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── Save combined .npz datasets ──
    def concat_and_save(all_tr, name):
        if not all_tr:
            return
        keys = all_tr[0].keys()
        combined = {}
        for k in keys:
            combined[k] = np.concatenate([tr[k] for tr in all_tr], axis=0)
        npz_path = os.path.join(out_dir, f"{name}_transitions.npz")
        np.savez_compressed(npz_path, **combined)
        n = combined["obs"].shape[0]
        n_eps = combined["done"].sum()
        ep_returns = []
        cur = 0.0
        for r, d in zip(combined["reward"], combined["done"]):
            cur += r
            if d:
                ep_returns.append(cur)
                cur = 0.0
        ep_returns = np.array(ep_returns)
        sz = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"  {name}: {n} transitions, {int(n_eps)} episodes, "
              f"mean_return={ep_returns.mean():.1f}, obs_dim={combined['obs'].shape[1]}, "
              f"({sz:.1f} MB)")

    print(f"\n{'=' * 70}")
    print("Saving combined datasets...")
    concat_and_save(all_pilot_transitions, "pilot")
    concat_and_save(all_expert_transitions, "expert")

    # Save summary
    summary_path = os.path.join(out_dir, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'=' * 70}")
    print(f"  {'Object':<15} {'Br':>4} {'Policy':>7} {'Status':>6} "
          f"{'T':>4} {'Goal':>6} {'Clr':>5} {'Vis%':>5} {'Return':>8}")
    print(f"  {'-' * 70}")
    for s in summary:
        short = s["object"].split()[-1][:12]
        vis_str = f"{s['gt_vis_pct']:.0%}" if s["gt_vis_pct"] is not None else "N/A"
        print(f"  {short:<15} {s['branch_id']:>4} {s['policy']:>7} {s['status']:>6} "
              f"{s['T']:>4} {s['goal_dist']:>6.2f} {s['min_clearance']:>5.2f} "
              f"{vis_str:>5} {s['episode_return']:>8.0f}")
    print(f"\n  DONE — {len(summary)} runs, "
          f"{sum(s['T'] for s in summary)} total transitions")


if __name__ == "__main__":
    main()
