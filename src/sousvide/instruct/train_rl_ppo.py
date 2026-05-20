"""
RL Fine-Tuning via PPO (Step 7b) — On-policy with KL constraint to DAgger reference.

Key difference from train_rl_finetune.py (off-policy actor-critic):
  - On-policy: collect rollouts, compute GAE advantages, PPO update, discard data
  - Value network V(s) only as baseline (not for actor gradient)
  - KL penalty to frozen DAgger policy prevents catastrophic forgetting
  - Clipped surrogate objective prevents large policy updates

Based on HW2 PPO implementation + RLHF literature (Ouyang et al. 2022).

Reuses from SINGER pipeline:
  - Pilot, Simulator, _evaluate_run, _get_scene from train_dagger.py
  - RLRolloutPolicy, extract_commander_features, compute_trajectory_rewards
    from train_rl_finetune.py
"""

import os
import gc
import copy
import json
import time
import random
import shutil
import pickle
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import cKDTree

# -- SINGER pipeline imports --------------------------------------------------
from sousvide.instruct.train_dagger import (
    _evaluate_run,
    _get_scene,
    _get_pkl,
    _preload_all_pkls,
    _preload_bc_trajectories,
    _load_all_branches,
    _save_model_checkpoint,
    _swap_model,
    _make_terminal_fn,
    _SCENE_CACHE,
    _BC_TRAJ_CACHE,
    _BRANCHES_CACHE,
    run_cross_cohort_benchmark,
    DEVICE,
    SUCCESS_RADIUS,
    COLLISION_RADIUS,
    HORIZONTAL_FOV,
)
from sousvide.instruct.train_rl_finetune import (
    extract_commander_features,
    commander_forward_from_features,
    compute_trajectory_rewards,
    RLRolloutPolicy,
)
from sousvide.control.pilot import Pilot

# Override DEVICE if CUDA_VISIBLE_DEVICES is set (allows GPU 1)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


# -------------------------------------------------------------------------------
# Unseen benchmark — uses validation trajectories never seen during training
# Reuses logic from scripts/eval_unseen_benchmark.py
# -------------------------------------------------------------------------------

def _load_val_trajectories_per_object(
    workspace_path: str,
    bc_cohort_name: str,
    scene_name: str,
    obj_targets,
    queries: list,
) -> dict:
    """Load trajectories_val*.pt and group by nearest object."""
    import glob as _glob
    rollout_dir = os.path.join(
        workspace_path, "cohorts", bc_cohort_name, "rollout_data", scene_name
    )
    val_files = sorted(_glob.glob(os.path.join(rollout_dir, "trajectories_val*.pt")))
    print(f"  [val] {len(val_files)} validation trajectory files in {rollout_dir}")

    obj_locs = np.array([np.squeeze(t) for t in obj_targets])
    per_obj = {q: [] for q in queries}

    for tf in val_files:
        try:
            data = torch.load(tf, map_location="cpu", weights_only=False)
        except Exception:
            continue
        tXUd = data.get("tXUd")
        if tXUd is None or tXUd.shape[0] != 18:
            continue
        goal = tXUd[1:4, -1]
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best = int(np.argmin(dists))
        if float(dists[best]) < 5.0:
            per_obj[queries[best]].append(tXUd)

    for q in queries:
        print(f"    '{q}': {len(per_obj[q])} val trajectories")
    return per_obj


def run_unseen_benchmark(
    pilot: Pilot,
    model,
    model_path: str,
    label: str,
    val_per_obj: dict,
    simulator,
    obj_targets,
    queries: list,
    epcds_arr,
    env_min=None,
    env_max=None,
) -> dict:
    """Evaluate model on UNSEEN validation trajectories (full trajectory from t=0).

    Returns dict with per-object and overall results, same format as
    run_cross_cohort_benchmark for easy comparison.
    """
    print(f"\n{'='*70}")
    print(f"[Unseen Benchmark] {label}  (val trajectories, full trajectory from t=0)")
    print(f"{'='*70}")

    # Load model weights and initialize for deployment
    _swap_model(pilot, model_path)
    pilot.set_mode("deploy")  # dummy forward pass to initialize model internals

    results = {}
    all_sr, all_cr = [], []
    all_run_diagnostics = []

    for obj_idx, obj_name in enumerate(queries):
        trajs = val_per_obj.get(obj_name, [])
        if not trajs:
            print(f"  [{label}] '{obj_name}' — no val trajectories, skip")
            continue
        obj_target = (
            obj_targets[obj_idx] if obj_idx < len(obj_targets)
            else trajs[0][1:4, -1]
        )

        successes, collisions, goal_dists = [], [], []
        # Build KD-tree once per object (same point cloud for all runs)
        _pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None
        for run_i, tXUi in enumerate(trajs):
            # Reset pilot state between runs
            pilot.hy_flag = False
            pilot.hy_idx = 0
            pilot.DxU.zero_()
            if hasattr(pilot, "Znn") and isinstance(pilot.Znn, torch.Tensor):
                pilot.Znn.zero_()
            pilot.set_mode("deploy")  # re-initialize after reset

            # Full trajectory from t=0
            x0 = tXUi[1:11, 0].copy()
            t0 = float(tXUi[0, 0])
            tf = float(tXUi[0, -1])

            # Terminal state: stop at success (goal zone) OR collision
            _terminal_fn, _term_info = _make_terminal_fn(
                obj_target, _pc_tree, env_min, env_max
            )

            try:
                result = simulator.simulate(
                    policy=pilot, t0=t0, tf=tf, x0=x0,
                    obj=np.zeros((18, 1)), query=obj_name,
                    vision_processor=None, verbose=False,
                    early_stop_fn=_terminal_fn,
                )
            except Exception as e:
                print(f"  [{label}] 💀 '{obj_name[:20]}' run {run_i+1}/{len(trajs)} crash: {e}")
                successes.append(0.0)
                collisions.append(1.0)
                goal_dists.append(float("inf"))
                continue

            Xro = result[1]
            ev = _evaluate_run(
                Xro, obj_target, epcds_arr,
                env_min=env_min, env_max=env_max,
                tXUi=tXUi, idx0=0,
            )
            success = float(ev["success"])
            collided = float(ev["collision"])
            gd = float(ev["goal_dist"])
            successes.append(success)
            collisions.append(collided)
            goal_dists.append(gd)

            d0 = float(np.linalg.norm(x0[:3] - np.squeeze(obj_target)))
            sim_time = result[0][-1] - result[0][0]
            status = "✓" if success else ("💥" if collided else "👁")

            # Comprehensive per-run log line
            detail_parts = [
                f"  [{label}] {status}  '{obj_name[:20]}'  val={run_i}  run {run_i+1}/{len(trajs)}",
                f"d0={d0:.1f}m  goal_dist={gd:.2f}m  min_goal={ev['min_goal_dist']:.2f}m@step{ev['min_goal_step']}",
                f"steps={ev['n_steps']}  ({sim_time:.0f}s)",
            ]
            if collided:
                cpos = ev.get("collision_position")
                cpos_str = f"({cpos[0]:.1f},{cpos[1]:.1f},{cpos[2]:.1f})" if cpos else "?"
                cs = ev["collision_step"]
                expert_pos_cs = tXUi[1:4, min(cs, tXUi.shape[1] - 1)]
                ep_str = f"({expert_pos_cs[0]:.1f},{expert_pos_cs[1]:.1f},{expert_pos_cs[2]:.1f})"
                model_pos_cs = result[1][:3, min(cs, result[1].shape[1] - 1)]
                dev = float(np.linalg.norm(model_pos_cs - expert_pos_cs))
                detail_parts.append(
                    f"COLL@step{cs}  type={ev['collision_type']}  "
                    f"pos={cpos_str}  expert={ep_str}  dev={dev:.2f}m  "
                    f"goal@coll={ev['collision_dist_to_goal']:.2f}m"
                )
            detail_parts.append(
                f"clearance={ev['min_clearance']:.2f}m@step{ev['min_clearance_step']}  "
                f"fov={ev['fov_pct']:.0%}  pos_dev={ev['mean_pos_dev']:.2f}m  "
                f"ori_dev={ev['mean_orient_dev_deg']:.1f}°"
            )
            print("  ".join(detail_parts))

            # ── Deep collision forensics ──
            forensics = {}
            if collided and ev["collision_step"] >= 0:
                cs = ev["collision_step"]
                Uro = result[2]  # model commands (4, N_ctl)
                goal_flat = np.squeeze(obj_target)

                # 1. Clearance trajectory: last 10 steps before + at collision
                clr_arr = ev.get("clearance_array")
                if clr_arr is not None:
                    window = 10
                    start = max(0, cs - window)
                    clr_window = clr_arr[start:cs + 1]
                    clr_str = " ".join([f"{c:.2f}" for c in clr_window])
                    print(f"    [forensic] clearance ramp (step {start}→{cs}): [{clr_str}]m")
                    forensics["clearance_window"] = clr_window.tolist()

                # 2. Model position trajectory around collision (last 5 steps + collision)
                pos_window = 5
                pos_start = max(0, cs - pos_window)
                print(f"    [forensic] model positions (step {pos_start}→{cs}):")
                model_positions = []
                for s in range(pos_start, min(cs + 1, Xro.shape[1])):
                    p = Xro[:3, s]
                    gd_s = np.linalg.norm(p - goal_flat)
                    model_positions.append({"step": s, "pos": p.tolist(), "goal_dist": float(gd_s)})
                    print(f"      step {s:3d}: ({p[0]:+7.2f}, {p[1]:+7.2f}, {p[2]:+7.2f})  goal_d={gd_s:.2f}m")
                forensics["model_positions"] = model_positions

                # 3. Model commands before collision
                cmd_window = 5
                cmd_start = max(0, cs - cmd_window)
                cmd_end = min(cs + 1, Uro.shape[1])
                print(f"    [forensic] model commands (step {cmd_start}→{cmd_end - 1}):")
                model_cmds = []
                for s in range(cmd_start, cmd_end):
                    u = Uro[:, s]
                    model_cmds.append({"step": s, "cmd": u.tolist()})
                    print(f"      step {s:3d}: thrust={u[0]:+.3f}  roll={u[1]:+.3f}  pitch={u[2]:+.3f}  yaw={u[3]:+.3f}")
                forensics["model_commands"] = model_cmds

                # 4. Goal bearing & elevation at collision
                drone_pos = Xro[:3, cs]
                delta = goal_flat - drone_pos
                bearing = np.degrees(np.arctan2(delta[1], delta[0]))
                horiz_dist = np.linalg.norm(delta[:2])
                elevation = np.degrees(np.arctan2(delta[2], horiz_dist))
                # Drone yaw from quaternion at collision
                if Xro.shape[0] >= 10:
                    qx, qy, qz, qw = Xro[6:10, cs]
                    drone_yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy),
                                                       1 - 2 * (qy**2 + qz**2)))
                    yaw_to_goal = bearing - drone_yaw
                    if yaw_to_goal > 180: yaw_to_goal -= 360
                    if yaw_to_goal < -180: yaw_to_goal += 360
                else:
                    drone_yaw = float('nan')
                    yaw_to_goal = float('nan')
                print(f"    [forensic] goal@collision: bearing={bearing:.1f}°  elevation={elevation:.1f}°  "
                      f"drone_yaw={drone_yaw:.1f}°  yaw_to_goal={yaw_to_goal:.1f}°  "
                      f"goal_dist={ev['collision_dist_to_goal']:.2f}m")
                forensics["goal_at_collision"] = {
                    "bearing": float(bearing), "elevation": float(elevation),
                    "drone_yaw": float(drone_yaw), "yaw_to_goal": float(yaw_to_goal),
                }

                # 5. Expert comparison: expert position + command at same steps
                if tXUi is not None and tXUi.shape[0] >= 15:
                    expert_start = max(0, cs - pos_window)
                    expert_end = min(cs + 1, tXUi.shape[1])
                    print(f"    [forensic] EXPERT positions (step {expert_start}→{expert_end - 1}):")
                    expert_positions = []
                    for s in range(expert_start, expert_end):
                        ep = tXUi[1:4, s]
                        mp = Xro[:3, min(s, Xro.shape[1] - 1)]
                        dev = np.linalg.norm(mp - ep)
                        egd = np.linalg.norm(ep - goal_flat)
                        expert_positions.append({"step": s, "pos": ep.tolist(),
                                                  "goal_dist": float(egd), "deviation": float(dev)})
                        print(f"      step {s:3d}: ({ep[0]:+7.2f}, {ep[1]:+7.2f}, {ep[2]:+7.2f})  "
                              f"goal_d={egd:.2f}m  model_dev={dev:.2f}m")

                    # Expert commands at same window
                    print(f"    [forensic] EXPERT commands (step {cmd_start}→{min(cmd_end, tXUi.shape[1]) - 1}):")
                    expert_cmds = []
                    for s in range(cmd_start, min(cmd_end, tXUi.shape[1])):
                        eu = tXUi[11:15, s]
                        expert_cmds.append({"step": s, "cmd": eu.tolist()})
                        print(f"      step {s:3d}: thrust={eu[0]:+.3f}  roll={eu[1]:+.3f}  pitch={eu[2]:+.3f}  yaw={eu[3]:+.3f}")
                    forensics["expert_positions"] = expert_positions
                    forensics["expert_commands"] = expert_cmds

                    # Expert clearance at collision step (if available)
                    if clr_arr is not None and cs < tXUi.shape[1]:
                        # Recompute clearance for expert position at collision step
                        ep_cs = tXUi[1:4, cs].reshape(1, 3)
                        ep_t = torch.from_numpy(ep_cs).float().to(ev.get("_device", "cpu"))
                        # We need pc_bench — it's epcds_arr passed to _evaluate_run
                        # Use a quick numpy computation instead
                        expert_clr = float(np.linalg.norm(
                            epcds_arr - ep_cs, axis=1
                        ).min()) if epcds_arr.shape[0] > 0 else float('inf')
                        print(f"    [forensic] expert clearance@step{cs}: {expert_clr:.2f}m  "
                              f"(model was {clr_arr[cs]:.2f}m)")
                        forensics["expert_clearance_at_collision"] = expert_clr

                print(f"    [forensic] ─── end collision forensics ───")

            # Collect per-run diagnostics for JSON export
            diag_entry = {
                "object": obj_name, "val_idx": run_i, "d0": d0,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "collision_type": ev["collision_type"],
                "collision_step": ev["collision_step"],
                "collision_position": ev.get("collision_position"),
                "collision_dist_to_goal": ev["collision_dist_to_goal"],
                "goal_dist": gd, "min_goal_dist": ev["min_goal_dist"],
                "min_goal_step": ev["min_goal_step"],
                "first_entry_step": ev["first_entry_step"],
                "n_steps": ev["n_steps"], "sim_time": sim_time,
                "min_clearance": ev["min_clearance"],
                "min_clearance_step": ev["min_clearance_step"],
                "fov_pct": ev["fov_pct"],
                "mean_pos_dev": ev["mean_pos_dev"],
                "mean_orient_dev_deg": ev["mean_orient_dev_deg"],
                "out_of_bounds": bool(ev["out_of_bounds"]),
            }
            if forensics:
                diag_entry["forensics"] = forensics
            all_run_diagnostics.append(diag_entry)

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_mean = float(np.mean(goal_dists)) if goal_dists else float("nan")
        gd_std = float(np.std(goal_dists)) if goal_dists else float("nan")
        print(
            f"  [{label}] ── '{obj_name[:25]}'  success={sr:.0%}  "
            f"collision={cr:.0%}  goal_dist={gd_mean:.2f}±{gd_std:.2f}m"
        )
        results[obj_name] = {
            "success_rate": sr,
            "collision_rate": cr,
            "goal_dist": gd_mean,
        }
        all_sr.append(sr)
        all_cr.append(cr)

    overall_sr = float(np.mean(all_sr)) if all_sr else 0.0
    overall_cr = float(np.mean(all_cr)) if all_cr else 0.0
    overall_gd = float(np.mean([r["goal_dist"] for r in results.values()])) if results else float("nan")

    print(f"\nOVERALL  {overall_gd:.2f}m  {overall_sr:.0%}  collision={overall_cr:.0%}")

    # Save per-run diagnostics JSON for offline analysis
    if all_run_diagnostics:
        diag_dir = Path(model_path).parent if model_path else Path(".")
        diag_path = diag_dir / f"unseen_diag_{label}.json"
        try:
            # Convert NaN to None for JSON serialization
            import math
            def _sanitize(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v
            sanitized = [
                {k: _sanitize(v) for k, v in d.items()}
                for d in all_run_diagnostics
            ]
            with open(diag_path, "w") as f:
                json.dump({"label": label, "runs": sanitized}, f, indent=2)
            print(f"  [diag] saved {len(all_run_diagnostics)} run diagnostics → {diag_path}")
        except Exception as e:
            print(f"  [diag] save failed: {e}")

    return {
        label: results,
        "__overall__": {
            "success_rate": overall_sr,
            "collision_rate": overall_cr,
            "goal_dist": overall_gd,
        },
        "__diagnostics__": all_run_diagnostics,
    }


# -------------------------------------------------------------------------------
# Value Network V(s) — estimates state value as baseline for GAE
# -------------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """V(s) network for PPO. Input = commander features (147D), output = scalar."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


# -------------------------------------------------------------------------------
# Stochastic policy wrapper for PPO
# -------------------------------------------------------------------------------

class StochasticCommander:
    """Wraps the deterministic CommanderSV with a learned log_std for PPO.

    PPO needs a stochastic policy to compute log-probs and entropy.
    We model: π(a|s) = N(μ(s), σ²) where μ(s) = CommanderSV(s) and σ is learned.
    """

    def __init__(self, model, log_std_init: float = -2.0, action_dim: int = 4):
        self.model = model
        self.commander = model.network["CommanderSV"]
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, device=DEVICE)
        )

    def forward(self, features: torch.Tensor):
        """Returns (mean, std) for the action distribution."""
        mu = self.commander.network(features)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor):
        """Compute log π(a|s) under the current Gaussian policy."""
        mu, std = self.forward(features)
        var = std ** 2
        log_p = -0.5 * (((actions - mu) ** 2) / var + torch.log(var) + np.log(2 * np.pi))
        return log_p.sum(dim=-1)  # sum over action dims

    def entropy(self, features: torch.Tensor):
        """Differential entropy of the Gaussian."""
        _, std = self.forward(features)
        return 0.5 * (1 + torch.log(2 * np.pi * std ** 2)).sum(dim=-1)

    def sample(self, features: torch.Tensor):
        """Sample action from the policy."""
        mu, std = self.forward(features)
        noise = torch.randn_like(mu)
        return torch.clamp(mu + std * noise, -1, 1)

    def mean_action(self, features: torch.Tensor):
        """Deterministic action (for evaluation)."""
        mu, _ = self.forward(features)
        return mu

    def parameters(self):
        """All trainable params: commander + log_std."""
        return list(self.commander.parameters()) + [self.log_std]


# -------------------------------------------------------------------------------
# GAE computation (from HW2, adapted)
# -------------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T,) per-step rewards
        values: (T,) V(s_t) estimates
        next_values: (T,) V(s_{t+1}) estimates
        dones: (T,) terminal flags (1.0 at terminal step)
        gamma: discount factor
        gae_lambda: GAE decay parameter

    Returns:
        advantages: (T,) GAE advantages
        returns: (T,) advantage + value (V targets)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(1, device=rewards.device)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * gae * (1 - dones[t])
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# -------------------------------------------------------------------------------
# PPO Rollout collector — collects on-policy data with log-probs
# -------------------------------------------------------------------------------

class PPORolloutPolicy:
    """Like RLRolloutPolicy but also records log_prob for PPO.

    Uses stochastic sampling during collection, deterministic during eval.
    """

    def __init__(self, pilot: Pilot, model, stochastic_commander: StochasticCommander,
                 eval_mode: bool = False):
        self.pilot = pilot
        self.model = model
        self.stochastic_commander = stochastic_commander
        self.eval_mode = eval_mode
        self.hz = pilot.hz
        self.nzcr = pilot.nzcr
        self.transitions: List[dict] = []

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        """Same interface as Pilot.control(), called by Simulator."""
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(upr, tcr, xcr, obj, icr, zcr)

        if xnn is not None:
            with torch.no_grad():
                tx_com = xnn["tx_com"].unsqueeze(0).to(DEVICE)
                obj_com = xnn["obj_com"].unsqueeze(0).to(DEVICE)
                dxu_par = xnn["dxu_par"].unsqueeze(0).to(DEVICE)
                img_vis = xnn["img_vis"].unsqueeze(0).to(DEVICE)
                tx_vis = xnn["tx_vis"].unsqueeze(0).to(DEVICE)
                features = extract_commander_features(
                    self.model, tx_com, obj_com, dxu_par, img_vis, tx_vis
                ).squeeze(0)  # stay on GPU for log_prob

            if self.eval_mode:
                with torch.no_grad():
                    action_t = self.stochastic_commander.mean_action(features.unsqueeze(0)).squeeze(0)
                    log_prob = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    action_t = self.stochastic_commander.sample(features.unsqueeze(0)).squeeze(0)
                    log_prob = self.stochastic_commander.log_prob(
                        features.unsqueeze(0), action_t.unsqueeze(0)
                    ).squeeze(0)

            action_np = action_t.cpu().numpy()

            self.transitions.append({
                "features": features.cpu(),
                "action": action_t.cpu(),
                "log_prob": log_prob.cpu().item(),
                "x": xcr.copy(),
                "t": tcr,
            })

            return action_np, znn, adv, tsol
        else:
            return u_pilot, znn, adv, tsol

    def reset(self):
        self.transitions = []
        self.pilot.hy_flag = False
        self.pilot.hy_idx = 0
        self.pilot.DxU.zero_()
        if hasattr(self.pilot, 'Znn'):
            self.pilot.Znn.zero_()


# -------------------------------------------------------------------------------
# Main PPO fine-tuning function
# -------------------------------------------------------------------------------

def train_rl_ppo(
    cohort_name: str,
    method_name: str,
    roster: List[str],
    flights: List[Tuple[str, str]],
    # Source model
    source_cohort: str = None,
    bc_cohort: str = None,
    # PPO hyperparameters
    n_iterations: int = 30,
    rollouts_per_iter: int = 8,
    ppo_epochs: int = 3,
    batch_size: int = 128,
    actor_lr: float = 1e-6,
    value_lr: float = 3e-4,
    clip_eps: float = 0.1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
    log_std_init: float = -2.0,
    value_hidden: int = 256,
    # Reward weights (same as off-policy version)
    goal_progress_weight: float = 1.0,
    collision_penalty_weight: float = 2.0,
    clearance_threshold: float = 0.5,
    fov_bonus: float = 0.05,
    success_bonus: float = 10.0,
    collision_penalty: float = 10.0,
    time_penalty: float = 0.01,
    # Stability
    max_grad_norm: float = 0.5,
    reset_to_best: bool = True,
    # Evaluation
    n_eval: int = 20,
    eval_seed: int = 42,
    patience: int = 8,
    use_unseen_eval: bool = False,
    # Logging
    log_dir: str = None,
) -> dict:
    """Fine-tune a pre-trained SINGER policy using PPO with KL constraint.

    On-policy: collect rollouts -> compute GAE -> PPO clipped update -> discard.
    KL penalty to frozen DAgger reference prevents catastrophic forgetting.
    """
    print("\n" + "=" * 70)
    print("[PPO-Finetune] On-policy PPO with KL constraint")
    print("=" * 70)

    workspace_path = str(Path(__file__).resolve().parents[3])
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)
    scenes_cfg_dir = os.path.join(workspace_path, "configs", "scenes")
    method_path = os.path.join(workspace_path, "configs", "method", method_name + ".json")

    os.makedirs(cohort_path, exist_ok=True)

    # -- Load method config -----------------------------------------------
    with open(method_path) as f:
        mcfg = json.load(f)
    frame_name = mcfg.get("sample_set", {}).get("frame", "carl")
    rollout_name = mcfg.get("sample_set", {}).get("rollout", "baseline")

    # -- Pre-load scenes --------------------------------------------------
    print("[PPO] Loading scenes...")
    for scene_name, _ in flights:
        _get_scene(scene_name, scenes_cfg_dir, frame_name, rollout_name)

    _preload_all_pkls(flights, scenes_cfg_dir)

    bc_cohort_name = bc_cohort or source_cohort
    if bc_cohort_name:
        n_bc_traj = _preload_bc_trajectories(bc_cohort_name, flights, scenes_cfg_dir)
        print(f"[PPO] Loaded {n_bc_traj} BC trajectories from {bc_cohort_name}")

    # -- Pre-load unseen validation trajectories if needed --------------------
    val_per_obj = None
    if use_unseen_eval:
        scene_name_0 = flights[0][0]
        scene_data_0 = _get_scene(scene_name_0, scenes_cfg_dir)
        val_per_obj = _load_val_trajectories_per_object(
            workspace_path, bc_cohort_name, scene_name_0,
            scene_data_0["obj_targets"], scene_data_0["queries"],
        )
        n_val = sum(len(v) for v in val_per_obj.values())
        print(f"[PPO] Loaded {n_val} UNSEEN validation trajectories for benchmark")

    # -- Load pilot + model (same as off-policy RL) -------------------------
    pilot_name = roster[0]
    source_path = os.path.join(workspace_path, "cohorts", source_cohort)
    roster_dir = os.path.join(cohort_path, "roster", pilot_name)
    os.makedirs(roster_dir, exist_ok=True)

    # Copy model from source if not exists
    model_path = os.path.join(roster_dir, "model.pth")
    if not os.path.exists(model_path):
        src_roster = os.path.join(source_path, "roster", pilot_name)
        dst_roster = roster_dir
        for fname in ("model.pth", "config.json", "losses_Commander.pt"):
            src = os.path.join(src_roster, fname)
            dst = os.path.join(dst_roster, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"[PPO] Copied {fname} from {source_cohort}")

    # Build pilot (same as train_rl_finetune.py)
    pilot = Pilot(cohort_name, pilot_name)
    pilot.set_mode("deploy")
    pilot.model.to(DEVICE)

    model = pilot.model
    commander = model.network["CommanderSV"]

    # Re-enable gradients on Commander
    for p in commander.parameters():
        p.requires_grad_(True)

    # Freeze VisionMLP + HistoryEncoder
    for p in model.network["HistoryEncoder"].parameters():
        p.requires_grad_(False)
    for p in model.network["VisionMLP"].parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in commander.parameters() if p.requires_grad)
    print(f"[PPO] Commander params: {n_params} (all requires_grad=True)")

    # -- Feature/action dimensions ----------------------------------------
    cfg_path = os.path.join(roster_dir, "config.json")
    with open(cfg_path) as f:
        pilot_cfg = json.load(f)
    nets = pilot_cfg["networks"]
    tx_com_dim = len(nets[2]["state"])
    obj_com_dim = len(nets[2]["objective"])
    z_par_dim = nets[0]["encoder_size"]
    y_vis_dim = nets[1]["output_size"]
    feature_dim = tx_com_dim + obj_com_dim + z_par_dim + y_vis_dim
    action_dim = nets[2]["output_size"]

    print(f"[PPO] Feature dim: {feature_dim} = {tx_com_dim}+{obj_com_dim}+{z_par_dim}+{y_vis_dim}")
    print(f"[PPO] Action dim: {action_dim}")

    # -- Create PPO components --------------------------------------------
    stochastic_cmd = StochasticCommander(model, log_std_init, action_dim)
    value_net = ValueNetwork(feature_dim, value_hidden).to(DEVICE)

    # Frozen reference policy (DAgger baseline) for KL penalty
    ref_commander_state = copy.deepcopy(commander.state_dict())
    ref_commander = copy.deepcopy(commander)
    for p in ref_commander.parameters():
        p.requires_grad_(False)
    ref_commander.eval()

    # -- Optimizers -------------------------------------------------------
    actor_opt = optim.Adam(stochastic_cmd.parameters(), lr=actor_lr)
    value_opt = optim.Adam(value_net.parameters(), lr=value_lr)

    # -- Logging ----------------------------------------------------------
    if log_dir is None:
        log_dir = os.path.join(cohort_path, "ppo_training",
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    metrics_path = os.path.join(log_dir, "metrics.json")
    best_success_rate = 0.0
    best_iter = -1
    patience_counter = 0
    all_metrics = []

    print(f"\n[PPO] Starting training: {n_iterations} iterations, {rollouts_per_iter} rollouts/iter")
    print(f"[PPO] ppo_epochs={ppo_epochs}, batch={batch_size}, actor_lr={actor_lr}, value_lr={value_lr}")
    print(f"[PPO] clip_eps={clip_eps}, gae_lambda={gae_lambda}, kl_coef={kl_coef}")
    print(f"[PPO] entropy_coef={entropy_coef}, max_grad_norm={max_grad_norm}")
    print(f"[PPO] reset_to_best={reset_to_best}")
    print()

    # =====================================================================
    # Main PPO Loop
    # =====================================================================
    for iteration in range(n_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"[PPO] Iteration {iteration}/{n_iterations-1}  |  best={best_success_rate:.0%} (iter {best_iter})")
        print(f"{'='*60}")

        # ==================================================================
        # Phase 1: Collect ON-POLICY rollouts
        # ==================================================================
        model.eval()
        all_features = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        n_transitions = 0
        iter_successes = []
        iter_collisions = []
        iter_rewards = []

        for scene_name, _ in flights:
            scene_data = _get_scene(scene_name, scenes_cfg_dir)
            simulator = scene_data["simulator"]
            obj_targets = scene_data["obj_targets"]
            queries = scene_data["queries"]
            pc_bench = scene_data.get("epcds_arr", np.zeros((0, 3)))
            # Build KD-tree once per scene for terminal state collision checks
            _pc_tree_scene = cKDTree(pc_bench) if pc_bench.shape[0] > 0 else None

            for obj_idx, obj_name in enumerate(queries):
                pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
                if pkl_data is None:
                    continue

                tXUi_default = pkl_data["tXUi"]
                obj_target = (
                    obj_targets[obj_idx] if obj_idx < len(obj_targets)
                    else pkl_data.get("obj_loc", np.zeros(3))
                )

                all_branches = _load_all_branches(
                    scene_name, obj_name, cohort_path,
                    scenes_cfg_dir, tXUi_default,
                )

                branch_idxs = np.random.choice(
                    len(all_branches), size=rollouts_per_iter,
                    replace=(rollouts_per_iter > len(all_branches)),
                )
                print(f"  [Branches] '{obj_name[:20]}': sampled {list(branch_idxs)} from {len(all_branches)}")

                for br_idx in branch_idxs:
                    tXUi = all_branches[br_idx]
                    x0 = tXUi[1:11, 0].copy()
                    t0 = float(tXUi[0, 0])
                    tf = float(tXUi[0, -1])

                    ppo_policy = PPORolloutPolicy(
                        pilot, model, stochastic_cmd, eval_mode=False
                    )
                    ppo_policy.reset()

                    # Terminal state: stop at success (goal zone) OR collision
                    _terminal_fn_r, _term_info_r = _make_terminal_fn(
                        obj_target, _pc_tree_scene,
                        scene_data.get("env_min"), scene_data.get("env_max"),
                    )

                    t_sim_start = time.time()
                    result = simulator.simulate(
                        policy=ppo_policy, t0=t0, tf=tf, x0=x0,
                        obj=np.zeros((18, 1)), query=obj_name,
                        vision_processor=None, verbose=False,
                        early_stop_fn=_terminal_fn_r,
                    )
                    sim_elapsed = time.time() - t_sim_start
                    Tro, Xro = result[0], result[1]

                    # Compute rewards
                    rewards, info = compute_trajectory_rewards(
                        Xro, obj_target, pc_bench,
                        env_min=scene_data.get("env_min"),
                        env_max=scene_data.get("env_max"),
                        goal_progress_weight=goal_progress_weight,
                        collision_penalty_weight=collision_penalty_weight,
                        clearance_threshold=clearance_threshold,
                        fov_bonus=fov_bonus,
                        success_bonus=success_bonus,
                        collision_penalty=collision_penalty,
                        time_penalty=time_penalty,
                    )

                    transitions = ppo_policy.transitions
                    terminal = info["n_steps"]
                    n_trans = min(len(transitions) - 1, terminal, len(rewards))

                    for t in range(n_trans):
                        all_features.append(transitions[t]["features"])
                        all_actions.append(transitions[t]["action"])
                        all_log_probs.append(transitions[t]["log_prob"])
                        all_rewards.append(rewards[t])
                        # done = 1 at terminal step
                        is_terminal = (t == n_trans - 1)
                        all_dones.append(1.0 if is_terminal else 0.0)

                    # Also store next_features for the last transition
                    if n_trans > 0:
                        all_features.append(transitions[n_trans]["features"])
                        # This extra feature is for V(s_{T}) — will be popped later

                    n_transitions += n_trans
                    iter_successes.append(float(info["success"]))
                    iter_collisions.append(float(info["collision"]))
                    iter_rewards.append(float(sum(rewards[:n_trans])))

                    short_obj = obj_name[:20].ljust(20)
                    status = "SUCCESS" if info["success"] else ("COLL" if info["collision"] else "MISS")
                    d0 = np.linalg.norm(x0[:3] - np.asarray(obj_target).flatten()[:3])
                    total_r = sum(rewards[:n_trans])
                    # Final drone position + goal centroid
                    final_pos = Xro[:3, info["terminal_step"]]
                    pos_str = f"({final_pos[0]:+.1f},{final_pos[1]:+.1f},{final_pos[2]:+.1f})"
                    goal_flat = np.asarray(obj_target).flatten()[:3]
                    goal_str = f"({goal_flat[0]:+.1f},{goal_flat[1]:+.1f},{goal_flat[2]:+.1f})"
                    rollout_parts = [
                        f"  [rollout] {short_obj} {status:7s}  br={br_idx:3d}  d0={d0:.1f}m",
                        f"goal={info['goal_dist']:.2f}m  min_goal={info['min_goal_dist']:.2f}m",
                        f"steps={info['n_steps']}  reward={total_r:.2f}  {info['terminal_reason']}",
                        f"clearance={info['mean_clearance']:.2f}m  fov={info['fov_pct']:.0%}",
                        f"drone={pos_str}  target={goal_str}  sim={sim_elapsed:.1f}s",
                    ]
                    if info["collision"]:
                        cs = info["terminal_step"]
                        expert_pos_at_coll = tXUi[1:4, min(cs, tXUi.shape[1] - 1)]
                        ep_str = f"({expert_pos_at_coll[0]:+.1f},{expert_pos_at_coll[1]:+.1f},{expert_pos_at_coll[2]:+.1f})"
                        dev = np.linalg.norm(final_pos - expert_pos_at_coll)
                        rollout_parts.append(
                            f"COLL@step{cs}  expert={ep_str}  dev={dev:.2f}m"
                        )
                    print("  ".join(rollout_parts))

        collect_time = time.time() - iter_start
        rollout_sr = np.mean(iter_successes) if iter_successes else 0.0
        rollout_cr = np.mean(iter_collisions) if iter_collisions else 0.0
        print(f"\n  [collect] {n_transitions} transitions  "
              f"success={rollout_sr:.0%}  collision={rollout_cr:.0%}  "
              f"mean_reward={np.mean(iter_rewards):.2f}  ({collect_time:.1f}s)")

        if n_transitions < 10:
            print(f"  [PPO] Too few transitions ({n_transitions}), skipping update")
            continue

        # ==================================================================
        # Phase 2: PPO Update
        # ==================================================================
        model.train()
        for p in model.network["HistoryEncoder"].parameters():
            p.requires_grad_(False)
        for p in model.network["VisionMLP"].parameters():
            p.requires_grad_(False)
        for p in commander.parameters():
            p.requires_grad_(True)

        # Build tensors — note: all_features has one extra per trajectory for V(s_T)
        # We need to separate features and next_features properly
        # Simpler approach: recompute V for each (s, s') pair
        feat_tensors = torch.stack(all_features[:n_transitions]).to(DEVICE)
        act_tensors = torch.stack(all_actions[:n_transitions]).to(DEVICE)
        rew_tensors = torch.tensor(all_rewards[:n_transitions], dtype=torch.float32, device=DEVICE)
        done_tensors = torch.tensor(all_dones[:n_transitions], dtype=torch.float32, device=DEVICE)
        old_lp_tensors = torch.tensor(all_log_probs[:n_transitions], dtype=torch.float32, device=DEVICE)

        # Build next_features: shift by 1 within each trajectory
        # For simplicity, use the stored features offset by 1
        # (transitions[t+1].features = s_{t+1})
        # We need to reconstruct this from the flat list accounting for trajectory boundaries
        # Actually the extra features we appended are at trajectory boundaries
        # Let's just use value bootstrapping: V(s') = 0 at done, V(s') = V(s_{t+1}) otherwise
        # We can compute V for all states and shift

        with torch.no_grad():
            values = value_net(feat_tensors)  # (T,)

            # For next_values, we need V(s_{t+1})
            # Since dones mark trajectory ends, we compute next_features inline
            # Shift features by 1, pad last with zeros (will be masked by done)
            next_feat = torch.zeros_like(feat_tensors)
            next_feat[:-1] = feat_tensors[1:]
            # At trajectory boundaries (done=1), next_value should be 0
            next_values = value_net(next_feat)
            next_values = next_values * (1 - done_tensors)  # zero at terminal

            advantages, returns = compute_gae(
                rew_tensors, values, next_values, done_tensors,
                gamma=gamma, gae_lambda=gae_lambda,
            )

            # Normalize advantages
            adv_std = advantages.std(unbiased=False)
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # PPO epochs
        T = n_transitions
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_val = 0.0
        total_kl = 0.0
        num_updates = 0

        update_start = time.time()
        all_params = stochastic_cmd.parameters() + list(value_net.parameters())

        for epoch in range(ppo_epochs):
            idx = torch.randperm(T, device=DEVICE)

            for i in range(0, T, batch_size):
                batch_idx = idx[i:i + batch_size]
                bs = batch_idx.shape[0]

                obs_b = feat_tensors[batch_idx]
                act_b = act_tensors[batch_idx]
                adv_b = advantages[batch_idx]
                ret_b = returns[batch_idx]
                olp_b = old_lp_tensors[batch_idx]

                # New log-probs under current policy
                new_log_prob = stochastic_cmd.log_prob(obs_b, act_b)
                ent = stochastic_cmd.entropy(obs_b).mean()

                # PPO clipped surrogate — clamp log-ratio to prevent exp overflow
                log_ratio = torch.clamp(new_log_prob - olp_b, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL to reference policy (use detached log_std for stable ref_log_prob)
                with torch.no_grad():
                    ref_mu = ref_commander.network(obs_b)
                    ref_std2 = stochastic_cmd.log_std.detach().exp() ** 2
                    ref_log_std2 = stochastic_cmd.log_std.detach() * 2
                ref_log_prob = -0.5 * (((act_b - ref_mu) ** 2) / ref_std2
                                       + ref_log_std2 + np.log(2 * np.pi)).sum(dim=-1)
                kl = (new_log_prob - ref_log_prob).mean()

                # Value loss
                v_pred = value_net(obs_b)
                value_loss = F.mse_loss(v_pred, ret_b)

                # Combined loss
                loss = (policy_loss
                        + value_coef * value_loss
                        - entropy_coef * ent
                        + kl_coef * kl)

                # NaN safety — skip this batch if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [train] WARNING: NaN/Inf loss at update {num_updates}, skipping batch")
                    actor_opt.zero_grad()
                    value_opt.zero_grad()
                    continue

                # Actor update
                actor_opt.zero_grad()
                value_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                actor_opt.step()
                value_opt.step()

                # Clamp log_std to prevent std collapse or explosion
                # std ∈ [exp(-4)=0.018, exp(0.5)=1.65]
                stochastic_cmd.log_std.data.clamp_(-4.0, 0.5)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_val += ent.item()
                total_kl += kl.item()
                num_updates += 1

        update_time = time.time() - update_start
        if num_updates > 0:
            print(f"  [train] {num_updates} updates  "
                  f"policy={total_policy_loss/num_updates:.4f}  "
                  f"value={total_value_loss/num_updates:.4f}  "
                  f"entropy={total_entropy_val/num_updates:.4f}  "
                  f"kl={total_kl/num_updates:.4f}  ({update_time:.1f}s)")
            print(f"  [train] log_std={stochastic_cmd.log_std.data.cpu().numpy()}")

        # Save updated model
        model_cpu = model.cpu()
        torch.save(model_cpu, model_path)
        model.to(DEVICE)

        # ==================================================================
        # Phase 3: Evaluation
        # ==================================================================
        model.eval()
        eval_start = time.time()
        label = f"ppo_iter{iteration}"

        if use_unseen_eval and val_per_obj is not None:
            # UNSEEN benchmark — validation trajectories never seen during training
            scene_name_eval = flights[0][0]
            sd = _get_scene(scene_name_eval, scenes_cfg_dir)
            bench_result = run_unseen_benchmark(
                pilot=pilot, model=model, model_path=model_path,
                label=label, val_per_obj=val_per_obj,
                simulator=sd["simulator"],
                obj_targets=sd["obj_targets"],
                queries=sd["queries"],
                epcds_arr=sd.get("epcds_arr", np.zeros((0, 3))),
                env_min=sd.get("env_min"),
                env_max=sd.get("env_max"),
            )
        else:
            # SEEN benchmark — training branches (same as DAgger)
            bench_result = run_cross_cohort_benchmark(
                models=[{
                    "label": label,
                    "cohort": cohort_name,
                    "pilot_name": pilot_name,
                    "model_path": model_path,
                }],
                flights=flights,
                scenes_cfg_dir=scenes_cfg_dir,
                benchmark_seed=eval_seed,
                max_trajectories=n_eval,
                bc_cohort=bc_cohort_name,
                cohort_path=cohort_path,
            )

        eval_time = time.time() - eval_start

        # Parse benchmark results
        if label in bench_result:
            obj_results = bench_result[label]
            obj_items = [(k, v) for k, v in obj_results.items() if k != "__overall__"]
            n_obj = len(obj_items)
            if n_obj > 0:
                avg_sr = np.mean([v.get("success_rate", 0) for _, v in obj_items])
                avg_cr = np.mean([v.get("collision_rate", 0) for _, v in obj_items])
                avg_gd = np.mean([v.get("goal_dist", float('inf')) for _, v in obj_items])
                print(f"\nOVERALL  {avg_gd:.2f}m  {avg_sr:.0%}  collision={avg_cr:.0%}  ({eval_time:.1f}s)")

                # Track best
                if avg_sr > best_success_rate:
                    best_success_rate = avg_sr
                    best_iter = iteration
                    patience_counter = 0
                    best_model_path = os.path.join(roster_dir, "model_best_ppo.pth")
                    model_cpu = model.cpu()
                    torch.save(model_cpu, best_model_path)
                    model.to(DEVICE)
                    print(f"  [BEST] New best: {avg_sr:.1%} -> {best_model_path}")
                else:
                    patience_counter += 1
                    if reset_to_best and best_iter >= 0:
                        best_model_path = os.path.join(roster_dir, "model_best_ppo.pth")
                        if os.path.isfile(best_model_path):
                            shutil.copy2(best_model_path, model_path)
                            model_cpu = torch.load(model_path, map_location=DEVICE)
                            model.load_state_dict(model_cpu.state_dict() if hasattr(model_cpu, 'state_dict') else model_cpu)
                            # Re-wrap commander reference
                            commander = model.network["CommanderSV"]
                            stochastic_cmd.commander = commander
                            stochastic_cmd.model = model
                            print(f"  [reset] Restored best model from iter {best_iter}")

                # Save metrics
                iter_metrics = {
                    "iteration": iteration,
                    "success_rate": float(avg_sr),
                    "collision_rate": float(avg_cr),
                    "goal_dist": float(avg_gd),
                    "rollout_sr": float(rollout_sr),
                    "rollout_cr": float(rollout_cr),
                    "n_transitions": n_transitions,
                    "policy_loss": total_policy_loss / max(num_updates, 1),
                    "value_loss": total_value_loss / max(num_updates, 1),
                    "entropy": total_entropy_val / max(num_updates, 1),
                    "kl": total_kl / max(num_updates, 1),
                }
                all_metrics.append(iter_metrics)

                with open(metrics_path, "w") as f:
                    json.dump(all_metrics, f, indent=2)

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[PPO] Early stopping: no improvement for {patience} iterations")
            break

        # Reset eval seed randomness so next iter samples different branches
        np.random.seed(None)
        torch.manual_seed(int(time.time()) % 2**32)

        # Cleanup per-iteration
        del all_features, all_actions, all_log_probs, all_rewards, all_dones
        del feat_tensors, act_tensors, rew_tensors, done_tensors, old_lp_tensors
        torch.cuda.empty_cache()
        gc.collect()

    # -- Restore best model -----------------------------------------------
    best_model_path = os.path.join(roster_dir, "model_best_ppo.pth")
    if os.path.isfile(best_model_path):
        shutil.copy2(best_model_path, model_path)
        print(f"\n[PPO] Restored best model (iter {best_iter}, {best_success_rate:.1%}) -> model.pth")

    # Save value network
    value_path = os.path.join(log_dir, "value_net.pth")
    torch.save(value_net.state_dict(), value_path)

    total_time = time.time() - iter_start
    print(f"\n[PPO] Training complete. Best: {best_success_rate:.1%} at iter {best_iter}")
    print(f"[PPO] Metrics saved to {metrics_path}")

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_success_rate": best_success_rate,
        "best_iter": best_iter,
        "metrics": all_metrics,
    }
