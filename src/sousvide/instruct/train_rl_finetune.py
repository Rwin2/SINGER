"""
RL Fine-Tuning (Step 7) -- Off-policy actor-critic with UTD=5.

Fine-tunes a pre-trained DAgger/BC CommanderSV using RL rewards
derived from existing SINGER metrics (goal distance, collision,
FOV, success). VisionMLP and HistoryEncoder stay frozen.

Reuses:
  - Pilot, Simulator, _evaluate_run, _get_scene from train_dagger.py
  - CollisionDetector from sousvide.rl.collision_detector
  - compute_mc_rewards from sousvide.rl.rl_helpers
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
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sousvide.control.pilot import Pilot
from sousvide.rl.collision_detector import CollisionDetector

# -- Reuse DAgger infrastructure -----------------------------------------------
from sousvide.instruct.train_dagger import (
    _evaluate_run,
    _get_scene,
    _get_pkl,
    _preload_all_pkls,
    _preload_bc_trajectories,
    _load_all_branches,
    _save_model_checkpoint,
    _swap_model,
    _SCENE_CACHE,
    _BC_TRAJ_CACHE,
    _BRANCHES_CACHE,
    run_cross_cohort_benchmark,
    DEVICE,
    SUCCESS_RADIUS,
    COLLISION_RADIUS,
    HORIZONTAL_FOV,
)


# -------------------------------------------------------------------------------
# Critic Network (Q-function ensemble)
# -------------------------------------------------------------------------------

class QCritic(nn.Module):
    """Ensemble of Q-networks for clipped double-Q learning.

    Input: commander feature vector (same dim as Commander input) + action (4D).
    Output: scalar Q-value per critic.
    """

    def __init__(self, feature_dim: int, action_dim: int = 4,
                 hidden_dim: int = 256, num_critics: int = 2):
        super().__init__()
        input_dim = feature_dim + action_dim
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_critics)
        ])
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([features, action], dim=-1).float()
        return [c(sa) for c in self.critics]


# -------------------------------------------------------------------------------
# Replay Buffer
# -------------------------------------------------------------------------------

class ReplayBuffer:
    """Simple replay buffer storing flat feature vectors (not images).

    Each transition: (features, action, reward, next_features, done)
    where features = cat(tx_com, obj_com, z_par, y_vis) -- the commander input.
    """

    def __init__(self, capacity: int = 500_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, features, action, reward, next_features, done):
        self.buffer.append((features, action, reward, next_features, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        feats, acts, rews, next_feats, dones = zip(*batch)
        return (
            torch.stack(feats).to(device),
            torch.stack(acts).to(device),
            torch.tensor(rews, dtype=torch.float32, device=device).unsqueeze(1),
            torch.stack(next_feats).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------------------------------------------------------------
# Feature extraction (runs SVNet up to Commander input, without Commander)
# -------------------------------------------------------------------------------

def extract_commander_features(model, tx_com, obj_com, dxu_par, img_vis, tx_vis):
    """Run HistoryEncoder + VisionMLP (frozen) and concatenate Commander input.

    Returns the feature vector that would be fed into CommanderSV.forward().
    This is: cat(tx_com, obj_com, z_par, y_vis) -- same as CommandSV.forward does internally.
    """
    with torch.no_grad():
        _, z_par = model.network["HistoryEncoder"](dxu_par)
        y_vis, _ = model.network["VisionMLP"](img_vis, tx_vis)
    # cat = what CommandSV sees
    return torch.cat((tx_com, obj_com, z_par, y_vis), dim=-1)


def commander_forward_from_features(model, features):
    """Run CommanderSV on pre-extracted features.

    features = cat(tx_com, obj_com, z_par, y_vis) -- must match dimensions.
    """
    commander = model.network["CommanderSV"]
    return commander.network(features)  # SimpleMLP forward


# -------------------------------------------------------------------------------
# Reward computation from trajectory data
# -------------------------------------------------------------------------------

def compute_trajectory_rewards(
    Xro: np.ndarray,
    obj_target: np.ndarray,
    pc_bench: np.ndarray,
    env_min: np.ndarray = None,
    env_max: np.ndarray = None,
    # Reward weights
    goal_progress_weight: float = 1.0,
    collision_penalty_weight: float = 2.0,
    clearance_threshold: float = 0.5,
    fov_bonus: float = 0.05,
    success_bonus: float = 10.0,
    collision_penalty: float = 10.0,
    time_penalty: float = 0.01,
) -> Tuple[np.ndarray, dict]:
    """Compute per-step rewards from a trajectory using existing SINGER metrics.

    Args:
        Xro: State trajectory (10xN)
        obj_target: 3D goal position
        pc_bench: Point cloud for collision detection (Mx3)
        env_min/max: Scene boundaries

    Returns:
        rewards: (N-1,) per-step rewards (one fewer than states since we need transitions)
        info: dict with success, collision, final_goal_dist, etc.
    """
    n_steps = Xro.shape[1]
    obj_target = np.asarray(obj_target).flatten()[:3]
    positions = Xro[:3, :].T  # (N, 3)

    # Goal distances
    goal_dists = np.linalg.norm(positions - obj_target, axis=1)

    # Goal entry check (determines analysis horizon)
    in_zone = goal_dists <= SUCCESS_RADIUS
    reached = bool(in_zone.any())
    first_entry = int(np.argmax(in_zone)) if reached else -1

    # Analysis horizon: only check collisions/OOB up to goal entry (or full trajectory)
    # Post-goal data is irrelevant — the episode is over once the drone reaches the goal.
    horizon = (first_entry + 1) if reached else n_steps

    # Collision detection via point-cloud proximity (only up to horizon)
    collision_step = n_steps
    collided = False
    if pc_bench.shape[0] > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(pc_bench)
        clearances_horizon = np.array([tree.query(p, k=1)[0] for p in positions[:horizon]])
        # Pad to full length for reward computation (post-horizon gets large clearance)
        clearances = np.full(n_steps, 10.0)
        clearances[:horizon] = clearances_horizon
    else:
        clearances = np.full(n_steps, 10.0)  # no obstacles

    for i in range(horizon):
        if clearances[i] < COLLISION_RADIUS:
            collided = True
            collision_step = i
            break

    # OOB check (only up to horizon)
    oob = False
    oob_step = n_steps
    if env_min is not None and env_max is not None:
        env_min_arr = np.asarray(env_min).flatten()[:3]
        env_max_arr = np.asarray(env_max).flatten()[:3]
        for i in range(horizon):
            if np.any(positions[i] < env_min_arr) or np.any(positions[i] > env_max_arr):
                oob = True
                oob_step = i
                break
    if oob:
        collided = True
        collision_step = min(collision_step, oob_step)

    # Success: reached goal AND no collision before goal entry
    success = reached and not collided

    # FOV at each step
    fov_mask = np.zeros(n_steps, dtype=bool)
    if Xro.shape[0] >= 10:
        for t in range(n_steps):
            qx, qy, qz, qw = Xro[6:10, t]
            cam_pos = Xro[:3, t]
            dx = obj_target[0] - cam_pos[0]
            dy = obj_target[1] - cam_pos[1]
            req_yaw = np.arctan2(dy, dx)
            act_yaw = np.arctan2(2 * (qw * qz + qx * qy),
                                 1 - 2 * (qy**2 + qz**2))
            yaw_err = abs(act_yaw - req_yaw)
            if yaw_err > np.pi:
                yaw_err = 2 * np.pi - yaw_err
            fov_mask[t] = yaw_err <= (HORIZONTAL_FOV / 2)

    # -- Determine terminal step --
    # Episode ends at the FIRST terminal event: goal entry, collision, or trajectory end.
    # This prevents post-goal drift from poisoning the critic.
    terminal_step = n_steps - 1  # default: end of trajectory
    terminal_reason = "timeout"

    if success:
        # Terminal at goal entry (the drone's job is done)
        terminal_step = first_entry
        terminal_reason = "success"
    elif collided:
        # Terminal at collision
        terminal_step = collision_step
        terminal_reason = "collision"

    # Effective transitions = steps up to AND including terminal step
    effective_steps = min(terminal_step + 1, n_steps - 1)

    # -- Per-step rewards (for transitions t -> t+1) --
    rewards = np.zeros(n_steps - 1)

    for t in range(effective_steps):
        r = 0.0

        # 1. Goal progress: positive if getting closer
        r += goal_progress_weight * (goal_dists[t] - goal_dists[t + 1])

        # 2. Collision avoidance: penalty when close to obstacles
        proximity = max(0.0, clearance_threshold - clearances[t + 1])
        r -= collision_penalty_weight * proximity

        # 3. FOV bonus
        if fov_mask[t + 1]:
            r += fov_bonus

        # 4. Time penalty
        r -= time_penalty

        rewards[t] = r

    # Terminal rewards (at the terminal step)
    if effective_steps > 0:
        term_idx = effective_steps - 1
        if success:
            rewards[term_idx] += success_bonus
        elif collided:
            rewards[term_idx] -= collision_penalty

    info = {
        "success": success,
        "collision": collided,
        "terminal_step": terminal_step,
        "terminal_reason": terminal_reason,
        "goal_dist": float(goal_dists[terminal_step]),
        "min_goal_dist": float(goal_dists[:effective_steps + 1].min()) if effective_steps > 0 else float(goal_dists[0]),
        "fov_pct": float(fov_mask[:effective_steps].mean()) if effective_steps > 0 else 0.0,
        "mean_clearance": float(clearances[:effective_steps].mean()) if effective_steps > 0 else 0.0,
        "n_steps": effective_steps,
    }
    return rewards, info


# -------------------------------------------------------------------------------
# RL Rollout collector (uses Pilot + Simulator, captures features)
# -------------------------------------------------------------------------------

class RLRolloutPolicy:
    """Wrapper around Pilot that collects (features, action) at each step
    for RL training. Compatible with Simulator.simulate() interface."""

    def __init__(self, pilot: Pilot, model, noise_std: float = 0.02):
        self.pilot = pilot
        self.model = model
        self.hz = pilot.hz
        self.nzcr = pilot.nzcr
        self.noise_std = noise_std  # exploration noise
        self.transitions: List[dict] = []
        self._prev_features = None

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        """Same interface as Pilot.control(), called by Simulator."""
        # Run full OODA to get action + xnn features
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(upr, tcr, xcr, obj, icr, zcr)

        # Extract commander features from xnn (frozen VisionMLP + HistoryEncoder)
        if xnn is not None:
            with torch.no_grad():
                tx_com = xnn["tx_com"].unsqueeze(0).to(DEVICE)
                obj_com = xnn["obj_com"].unsqueeze(0).to(DEVICE)
                dxu_par = xnn["dxu_par"].unsqueeze(0).to(DEVICE)
                img_vis = xnn["img_vis"].unsqueeze(0).to(DEVICE)
                tx_vis = xnn["tx_vis"].unsqueeze(0).to(DEVICE)
                features = extract_commander_features(
                    self.model, tx_com, obj_com, dxu_par, img_vis, tx_vis
                ).squeeze(0).cpu()

            # Add exploration noise (small for near-optimal policy)
            action = u_pilot.copy()
            if self.noise_std > 0:
                noise = np.random.randn(*action.shape) * self.noise_std
                action = np.clip(action + noise, -1, 1)

            # Store transition (features saved, reward computed post-hoc)
            self.transitions.append({
                "features": features,
                "action": torch.tensor(action, dtype=torch.float32),
                "x": xcr.copy(),
                "t": tcr,
            })
            self._prev_features = features

            return action, znn, adv, tsol
        else:
            return u_pilot, znn, adv, tsol

    def reset(self):
        self.transitions = []
        self._prev_features = None
        self.pilot.hy_flag = False
        self.pilot.hy_idx = 0
        self.pilot.DxU.zero_()
        if hasattr(self.pilot, 'Znn'):
            self.pilot.Znn.zero_()


# -------------------------------------------------------------------------------
# Soft target update
# -------------------------------------------------------------------------------

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# -------------------------------------------------------------------------------
# Main RL fine-tuning function
# -------------------------------------------------------------------------------

def train_rl_finetune(
    cohort_name: str,
    method_name: str,
    roster: List[str],
    flights: List[Tuple[str, str]],
    # Source model
    source_cohort: str = None,
    bc_cohort: str = None,
    # RL hyperparameters
    n_iterations: int = 20,
    rollouts_per_iter: int = 6,
    utd: int = 5,
    batch_size: int = 128,
    critic_lr: float = 3e-4,
    actor_lr: float = 1e-6,
    critic_target_tau: float = 0.005,
    discount: float = 0.99,
    noise_std: float = 0.02,
    stddev_clip: float = 0.3,
    num_critics: int = 2,
    critic_hidden: int = 256,
    replay_capacity: int = 200_000,
    warmup_transitions: int = 500,
    # Reward weights
    goal_progress_weight: float = 1.0,
    collision_penalty_weight: float = 2.0,
    clearance_threshold: float = 0.5,
    fov_bonus: float = 0.05,
    success_bonus: float = 10.0,
    collision_penalty: float = 10.0,
    time_penalty: float = 0.01,
    # BC regularization
    bc_reg_freq: int = 1,
    bc_reg_weight: float = 1.0,
    # Stability
    actor_grad_clip: float = 0.5,
    critic_warmup_iters: int = 2,
    reset_to_best: bool = True,
    # Evaluation
    n_eval: int = 20,
    eval_seed: int = 42,
    patience: int = 5,
    # Logging
    log_dir: str = None,
    # Resume
    resume_warmup: bool = True,
) -> dict:
    """Fine-tune a pre-trained SINGER policy using off-policy actor-critic (UTD=5).

    The actor IS the existing CommanderSV (we optimize its parameters).
    A new critic ensemble learns Q-values from rollout rewards.

    Key stability features vs v1:
      - Critic-only warmup: no actor updates for first `critic_warmup_iters` iters
      - Action-level BC reg: MSE between current and frozen BC actions (not param L2)
      - Gradient clipping on actor updates
      - Reset-to-best: reload best checkpoint when performance drops
    """
    print("\n" + "=" * 70)
    print("[RL-Finetune] Off-policy Actor-Critic with UTD=%d" % utd)
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
    print("[RL] Loading scenes...")
    for scene_name, _ in flights:
        _get_scene(scene_name, scenes_cfg_dir, frame_name, rollout_name)

    # Pre-load pkl trajectories (fallback for _load_all_branches)
    _preload_all_pkls(flights, scenes_cfg_dir)

    # Pre-load BC rollout trajectories for multi-branch sampling
    # (same mechanism as DAgger — loads branches from BC cohort's rollout_data)
    bc_cohort_name = bc_cohort or source_cohort
    if bc_cohort_name:
        n_bc_traj = _preload_bc_trajectories(bc_cohort_name, flights, scenes_cfg_dir)
        print(f"[RL] Loaded {n_bc_traj} BC trajectories from {bc_cohort_name}")

    # -- Initialize pilot from source cohort ------------------------------
    pilot_name = roster[0]
    print(f"[RL] Pilot: {pilot_name}")

    # Copy model from source cohort if needed
    if source_cohort:
        src_roster = os.path.join(workspace_path, "cohorts", source_cohort, "roster", pilot_name)
        dst_roster = os.path.join(cohort_path, "roster", pilot_name)
        dst_model = os.path.join(dst_roster, "model.pth")
        if not os.path.isfile(dst_model):
            os.makedirs(dst_roster, exist_ok=True)
            for fname in ("model.pth", "config.json", "losses_Commander.pt"):
                src = os.path.join(src_roster, fname)
                dst = os.path.join(dst_roster, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"[RL] Copied {fname} from {source_cohort}")

    pilot = Pilot(cohort_name, pilot_name)
    pilot.set_mode("deploy")
    pilot.model.to(DEVICE)

    model = pilot.model
    commander = model.network["CommanderSV"]

    # Re-enable gradients on Commander (may be saved with requires_grad=False from DAgger)
    for p in commander.parameters():
        p.requires_grad_(True)
    print(f"[RL] Commander params: {sum(p.numel() for p in commander.parameters())} "
          f"(all requires_grad=True)")

    # Save pre-RL model
    roster_dir = os.path.join(cohort_path, "roster", pilot_name)
    model_path = os.path.join(roster_dir, "model.pth")
    pre_rl_path = os.path.join(roster_dir, "model_before_rl.pth")
    if os.path.isfile(model_path) and not os.path.isfile(pre_rl_path):
        shutil.copy2(model_path, pre_rl_path)

    # -- Determine feature dimension --------------------------------------
    cfg_path = os.path.join(roster_dir, "config.json")
    with open(cfg_path) as f:
        pilot_cfg = json.load(f)
    nets = pilot_cfg["networks"]
    tx_com_dim = len(nets[2]["state"])       # Commander state indices
    obj_com_dim = len(nets[2]["objective"])   # Commander objective indices
    z_par_dim = nets[0]["encoder_size"]       # HistoryEncoder output
    y_vis_dim = nets[1]["output_size"]        # VisionMLP output
    feature_dim = tx_com_dim + obj_com_dim + z_par_dim + y_vis_dim
    action_dim = nets[2]["output_size"]       # 4: thrust, wx, wy, wz

    print(f"[RL] Feature dim: {feature_dim} = {tx_com_dim}+{obj_com_dim}+{z_par_dim}+{y_vis_dim}")
    print(f"[RL] Action dim: {action_dim}")

    # -- Create critic ----------------------------------------------------
    critic = QCritic(feature_dim, action_dim, critic_hidden, num_critics).to(DEVICE)
    critic_target = QCritic(feature_dim, action_dim, critic_hidden, num_critics).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    # -- Optimizers -------------------------------------------------------
    actor_opt = optim.Adam(commander.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    # -- Replay buffer ----------------------------------------------------
    replay = ReplayBuffer(capacity=replay_capacity)

    # -- Setup logging ----------------------------------------------------
    if log_dir is None:
        log_dir = os.path.join(cohort_path, "rl_training", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # -- Resume from warmup checkpoint if available -----------------------
    warmup_ckpt_dir = os.path.join(cohort_path, "rl_warmup_checkpoint")
    warmup_loaded = False
    if resume_warmup and os.path.isdir(warmup_ckpt_dir):
        ckpt_critic = os.path.join(warmup_ckpt_dir, "critic.pth")
        ckpt_target = os.path.join(warmup_ckpt_dir, "critic_target.pth")
        ckpt_replay = os.path.join(warmup_ckpt_dir, "replay_buffer.pkl")
        if all(os.path.isfile(p) for p in [ckpt_critic, ckpt_target, ckpt_replay]):
            critic.load_state_dict(torch.load(ckpt_critic, map_location=DEVICE))
            critic_target.load_state_dict(torch.load(ckpt_target, map_location=DEVICE))
            with open(ckpt_replay, "rb") as f:
                replay.buffer = pickle.load(f)
            warmup_loaded = True
            print(f"[RL] Resumed warmup checkpoint: critic + {len(replay)} replay transitions")

    # Save a snapshot of the BC Commander weights for BC regularization
    bc_commander_state = copy.deepcopy(commander.state_dict())

    # Frozen BC commander for action-level regularization
    bc_commander = copy.deepcopy(commander)
    for p in bc_commander.parameters():
        p.requires_grad_(False)
    bc_commander.eval()

    # -- Training metrics -------------------------------------------------
    best_success_rate = 0.0
    best_iter = -1
    patience_counter = 0
    all_metrics = []

    print(f"\n[RL] Starting training: {n_iterations} iterations, {rollouts_per_iter} rollouts/iter")
    print(f"[RL] UTD={utd}, batch={batch_size}, critic_lr={critic_lr}, actor_lr={actor_lr}")
    print(f"[RL] actor_grad_clip={actor_grad_clip}, critic_warmup_iters={critic_warmup_iters}")
    print(f"[RL] bc_reg_weight={bc_reg_weight}, reset_to_best={reset_to_best}")
    print()

    start_iter = critic_warmup_iters if warmup_loaded else 0
    if warmup_loaded:
        print(f"[RL] Skipping warmup iters 0..{critic_warmup_iters-1} (loaded from checkpoint)")

    for iteration in range(start_iter, n_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"[RL] Iteration {iteration}/{n_iterations-1}  |  replay={len(replay)}  |  best={best_success_rate:.0%} (iter {best_iter})")
        print(f"{'='*60}")

        # ==================================================================
        # Phase 1: Collect rollouts
        # ==================================================================
        model.eval()
        iter_rewards = []
        iter_successes = []
        iter_collisions = []
        n_transitions = 0

        for scene_name, _ in flights:
            scene_data = _get_scene(scene_name, scenes_cfg_dir)
            simulator = scene_data["simulator"]
            obj_targets = scene_data["obj_targets"]
            queries = scene_data["queries"]
            pc_bench = scene_data.get("epcds_arr", np.zeros((0, 3)))

            for obj_idx, obj_name in enumerate(queries):
                pkl_data = _get_pkl(scene_name, obj_name, scenes_cfg_dir)
                if pkl_data is None:
                    continue

                tXUi_default = pkl_data["tXUi"]
                obj_target = (
                    obj_targets[obj_idx] if obj_idx < len(obj_targets)
                    else pkl_data.get("obj_loc", np.zeros(3))
                )

                # Load multi-branch trajectories (same as DAgger pipeline)
                all_branches = _load_all_branches(
                    scene_name, obj_name, cohort_path,
                    scenes_cfg_dir, tXUi_default,
                )
                branch_idxs = np.random.choice(
                    len(all_branches), size=rollouts_per_iter,
                    replace=(rollouts_per_iter > len(all_branches)),
                )

                for br_idx in branch_idxs:
                    tXUi = all_branches[br_idx]
                    # Full trajectory: start from beginning, run to end
                    x0 = tXUi[1:11, 0].copy()
                    t0 = float(tXUi[0, 0])
                    tf = float(tXUi[0, -1])
                    init_goal_dist = np.linalg.norm(x0[:3] - np.asarray(obj_target).flatten()[:3])

                    # Create RL rollout policy wrapper
                    rl_policy = RLRolloutPolicy(pilot, model, noise_std=noise_std)
                    rl_policy.reset()

                    from scipy.spatial import cKDTree as _cKDTree
                    from sousvide.instruct.train_dagger import _make_terminal_fn
                    _pc_tree = _cKDTree(pc_bench) if pc_bench.shape[0] > 0 else None
                    _terminal_fn, _term_info = _make_terminal_fn(
                        obj_target, _pc_tree,
                        env_min=scene_data.get("env_min"),
                        env_max=scene_data.get("env_max"),
                    )
                    result = simulator.simulate(
                        policy=rl_policy, t0=t0, tf=tf, x0=x0,
                        obj=np.zeros((18, 1)), query=obj_name,
                        vision_processor=None, verbose=False,
                        early_stop_fn=_terminal_fn,
                    )
                    Tro, Xro = result[0], result[1]

                    # Compute rewards from trajectory
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

                    # Store transitions in replay buffer (only up to terminal event)
                    transitions = rl_policy.transitions
                    terminal = info["n_steps"]  # effective steps (up to terminal)
                    n_trans = min(len(transitions) - 1, terminal, len(rewards))
                    for t in range(n_trans):
                        feat_t = transitions[t]["features"]
                        act_t = transitions[t]["action"]
                        r_t = rewards[t]
                        feat_next = transitions[t + 1]["features"]
                        # done=True at terminal event (goal entry, collision, or end)
                        done = 1.0 if (t == n_trans - 1) else 0.0
                        replay.push(feat_t, act_t, r_t, feat_next, done)

                    n_transitions += n_trans
                    iter_rewards.append(sum(rewards[:n_trans]))
                    iter_successes.append(info["success"])
                    iter_collisions.append(info["collision"])

                    status = "SUCCESS" if info["success"] else ("COLL" if info["collision"] else "MISS")
                    print(f"  [rollout] {obj_name[:15]:15s}  {status:7s}  "
                          f"d0={init_goal_dist:.1f}m  goal={info['goal_dist']:.2f}m  "
                          f"min={info['min_goal_dist']:.2f}m  steps={info['n_steps']}  "
                          f"reward={sum(rewards[:n_trans]):.2f}  {info['terminal_reason']}")

        collect_time = time.time() - iter_start
        rollout_sr = np.mean(iter_successes) if iter_successes else 0.0
        rollout_cr = np.mean(iter_collisions) if iter_collisions else 0.0
        print(f"\n  [collect] {n_transitions} transitions  "
              f"success={rollout_sr:.0%}  collision={rollout_cr:.0%}  "
              f"mean_reward={np.mean(iter_rewards):.2f}  ({collect_time:.1f}s)")

        # ==================================================================
        # Phase 2: RL updates (UTD=5)
        # ==================================================================
        if len(replay) < warmup_transitions:
            print(f"  [train] Warming up replay ({len(replay)}/{warmup_transitions})")
            # Still do evaluation to get baseline
        else:
            model.train()
            # Freeze VisionMLP + HistoryEncoder, ensure Commander is unfrozen
            for p in model.network["HistoryEncoder"].parameters():
                p.requires_grad_(False)
            for p in model.network["VisionMLP"].parameters():
                p.requires_grad_(False)
            for p in commander.parameters():
                p.requires_grad_(True)

            # Critic-only warmup: no actor updates during early iterations
            actor_enabled = iteration >= critic_warmup_iters
            if not actor_enabled:
                print(f"  [train] Critic-only warmup (iter {iteration} < {critic_warmup_iters})")

            n_updates = n_transitions * utd
            critic_losses = []
            actor_losses = []
            bc_losses = []
            q_values = []

            update_start = time.time()
            for update_i in range(n_updates):
                # -- Critic update ----------------------------------------
                obs, action, reward, next_obs, done = replay.sample(batch_size, DEVICE)

                with torch.no_grad():
                    next_action = commander_forward_from_features(model, next_obs)
                    noise_t = (torch.randn_like(next_action) * noise_std).clamp(-stddev_clip, stddev_clip)
                    next_action = (next_action + noise_t).clamp(-1, 1)

                    target_qs = critic_target(next_obs, next_action)
                    idx = random.sample(range(len(target_qs)), min(2, len(target_qs)))
                    target_q = torch.min(target_qs[idx[0]], target_qs[idx[1]])
                    target = reward + (1 - done) * discount * target_q

                current_qs = critic(obs, action)
                critic_loss = sum(F.mse_loss(q, target) for q in current_qs)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                soft_update_params(critic, critic_target, critic_target_tau)
                critic_losses.append(critic_loss.item())
                q_values.append(float(current_qs[0].mean().detach()))

                # -- Actor update (every UTD steps, after critic warmup) --
                if actor_enabled and (update_i + 1) % utd == 0:
                    for p in critic.parameters():
                        p.requires_grad_(False)

                    obs_actor, _, _, _, _ = replay.sample(batch_size, DEVICE)
                    obs_actor = obs_actor.detach()
                    action_pred = commander_forward_from_features(model, obs_actor)
                    qs = critic(obs_actor, action_pred)
                    actor_loss = -sum(q for q in qs).mean() / len(qs)

                    # Action-level BC regularization
                    bc_loss = torch.tensor(0.0, device=DEVICE)
                    if bc_reg_weight > 0 and bc_reg_freq > 0 and (update_i // utd) % bc_reg_freq == 0:
                        with torch.no_grad():
                            bc_action = bc_commander.network(obs_actor)
                        bc_loss = F.mse_loss(action_pred, bc_action)
                        bc_losses.append(bc_loss.item())

                    total_loss = actor_loss + bc_reg_weight * bc_loss

                    actor_opt.zero_grad()
                    total_loss.backward()
                    if actor_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(commander.parameters(), actor_grad_clip)
                    actor_opt.step()

                    for p in critic.parameters():
                        p.requires_grad_(True)

                    actor_losses.append(actor_loss.item())

            update_time = time.time() - update_start
            mean_q = np.mean(q_values) if q_values else 0.0
            mean_critic = np.mean(critic_losses) if critic_losses else 0.0
            mean_actor = np.mean(actor_losses) if actor_losses else 0.0
            print(f"  [train] {n_updates} updates  critic={mean_critic:.4f}  "
                  f"actor={mean_actor:.4f}  Q={mean_q:.3f}  ({update_time:.1f}s)")
            if bc_losses:
                print(f"  [train] bc_action_loss={np.mean(bc_losses):.6f}")

        # Save warmup checkpoint after last warmup iter (critic + replay buffer)
        if iteration == critic_warmup_iters - 1 and not warmup_loaded:
            os.makedirs(warmup_ckpt_dir, exist_ok=True)
            torch.save(critic.state_dict(), os.path.join(warmup_ckpt_dir, "critic.pth"))
            torch.save(critic_target.state_dict(), os.path.join(warmup_ckpt_dir, "critic_target.pth"))
            with open(os.path.join(warmup_ckpt_dir, "replay_buffer.pkl"), "wb") as f:
                pickle.dump(replay.buffer, f)
            print(f"  [warmup-ckpt] Saved critic + {len(replay)} replay transitions to {warmup_ckpt_dir}")

        # Save updated model
        model_cpu = model.cpu()
        torch.save(model_cpu, model_path)
        model.to(DEVICE)

        # ==================================================================
        # Phase 3: Evaluation — uses run_cross_cohort_benchmark (identical
        # to the DAgger benchmark that produced the 88% V9 number)
        # ==================================================================
        model.eval()
        eval_start = time.time()

        bench_result = run_cross_cohort_benchmark(
            models=[{
                "label": f"rl_iter{iteration}",
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

        # Extract metrics from benchmark result
        label_key = f"rl_iter{iteration}"
        bench_data = bench_result.get(label_key, {})
        overall = bench_data.get("__overall__", {})
        eval_sr = overall.get("success_rate", 0.0)
        eval_cr = overall.get("collision_rate", 0.0)
        eval_gd = overall.get("goal_dist", float('inf'))
        eval_time = time.time() - eval_start

        # Print per-object results
        for obj_name, obj_data in bench_data.items():
            if obj_name == "__overall__":
                continue
            print(f"  [eval] {obj_name[:20]:20s}  success={obj_data['success_rate']:.0%}  collision={obj_data['collision_rate']:.0%}")

        print(f"\n  [EVAL] iter {iteration}  SUCCESS={eval_sr:.1%}  COLLISION={eval_cr:.1%}  "
              f"goal_dist={eval_gd:.2f}m  ({eval_time:.1f}s)")

        # -- Best model checkpoint ----------------------------------------
        if eval_sr > best_success_rate:
            best_success_rate = eval_sr
            best_iter = iteration
            patience_counter = 0
            best_model_path = os.path.join(roster_dir, "model_best_rl.pth")
            model_cpu = model.cpu()
            torch.save(model_cpu, best_model_path)
            model.to(DEVICE)
            print(f"  [BEST] New best: {eval_sr:.1%} -> {best_model_path}")
        else:
            patience_counter += 1
            print(f"  [patience] {patience_counter}/{patience}")

            # Reset-to-best: reload best model if performance drops
            if reset_to_best and best_iter >= 0:
                best_model_path = os.path.join(roster_dir, "model_best_rl.pth")
                if os.path.isfile(best_model_path):
                    model_loaded = torch.load(best_model_path, map_location=DEVICE)
                    model.load_state_dict(model_loaded.state_dict())
                    model.to(DEVICE)
                    commander = model.network["CommanderSV"]
                    for p in commander.parameters():
                        p.requires_grad_(True)
                    actor_opt = optim.Adam(commander.parameters(), lr=actor_lr)
                    print(f"  [reset] Restored best model from iter {best_iter}")

        # -- Save iteration metrics ---------------------------------------
        # Safe access: variables only exist inside the training block
        _cl = locals()
        _critic = _cl.get('critic_losses', [])
        _actor = _cl.get('actor_losses', [])
        _q = _cl.get('q_values', [])
        _bc = _cl.get('bc_losses', [])
        iter_metrics = {
            "iteration": iteration,
            "rollout_success": float(rollout_sr),
            "rollout_collision": float(rollout_cr),
            "rollout_reward": float(np.mean(iter_rewards)) if iter_rewards else 0.0,
            "eval_success": float(eval_sr),
            "eval_collision": float(eval_cr),
            "eval_goal_dist": float(eval_gd),
            "critic_loss": float(np.mean(_critic)) if _critic else 0.0,
            "actor_loss": float(np.mean(_actor)) if _actor else 0.0,
            "mean_q": float(np.mean(_q)) if _q else 0.0,
            "bc_action_loss": float(np.mean(_bc)) if _bc else 0.0,
            "replay_size": len(replay),
            "n_transitions": n_transitions,
            "time_collect": collect_time,
            "time_eval": eval_time,
        }
        all_metrics.append(iter_metrics)

        # Save metrics to disk
        metrics_path = os.path.join(log_dir, "rl_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        # -- Early stopping -----------------------------------------------
        if patience_counter >= patience:
            print(f"\n[RL] Early stopping at iter {iteration} (patience={patience})")
            break

        # Reset eval seed randomness for next iter rollouts
        np.random.seed(None)
        torch.manual_seed(int(time.time()) % 2**32)

    # -- Restore best model -----------------------------------------------
    best_model_path = os.path.join(roster_dir, "model_best_rl.pth")
    if os.path.isfile(best_model_path):
        shutil.copy2(best_model_path, model_path)
        print(f"\n[RL] Restored best model (iter {best_iter}, {best_success_rate:.1%}) -> model.pth")

    # Save critic for potential future use
    critic_path = os.path.join(log_dir, "critic.pth")
    torch.save(critic.state_dict(), critic_path)

    total_time = time.time() - iter_start
    print(f"\n[RL] Training complete. Best: {best_success_rate:.1%} at iter {best_iter}")
    print(f"[RL] Metrics saved to {metrics_path}")

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_success_rate": best_success_rate,
        "best_iter": best_iter,
        "metrics": all_metrics,
    }
