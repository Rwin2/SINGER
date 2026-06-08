#!/usr/bin/env python3
"""
IQL training for SINGER — wires up hw3 IQL with SINGER transitions.

Key differences from hw3:
  - Observations: 147-d Commander input (VisionMLP + HistoryEncoder features)
  - Actions: raw 4-d (no tanh, no normalization — same as BC/DAgger Commander)
  - Actor: linear output MLP matching Commander architecture (no tanh)
  - Actor initialized from trained Commander weights

Bypasses RL_Trainer (which requires gym.make) and directly:
  1. Creates IQLAgent with SINGER dims (ob=147, ac=4)
  2. Loads .npz transitions into the replay buffer
  3. Runs the training loop
  4. Saves trained actor (as a Commander replacement)

Usage:
    cd /data/erwinpi/SINGER
    source activate FiGS
    CUDA_VISIBLE_DEVICES=1 \
    python scripts/train_iql.py \
        --config-file configs/experiment/ssv_iql_v2.yml \
        --use-wandb
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Add hw3 to sys.path for IQL components
HW3_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hw3", "hw3")
sys.path.insert(0, os.path.abspath(HW3_ROOT))

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.utils import (
    OptimizerSpec,
    MemoryOptimizedReplayBuffer,
    make_continuous_q_network,
    make_value_network,
)
from cs224r.critics.iql_critic import IQLCritic

SINGER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Linear Actor (no tanh, matches Commander) ─────────────────────────

class LinearGaussianActor(nn.Module):
    """Actor with linear output (no tanh). Matches SINGER Commander architecture.

    Unlike hw3's MLPPolicyAWAC which uses tanh(mean_net), this outputs raw
    unbounded actions — consistent with BC/DAgger Commander training.
    """

    def __init__(self, ob_dim, ac_dim, hidden_sizes, lr, lambda_awac=0.1,
                 max_grad_norm=1.0, dropout=0.0, freeze_hidden=False):
        super().__init__()
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.lambda_awac = lambda_awac
        self.max_grad_norm = max_grad_norm

        # Build MLP (same structure as Commander: SimpleMLP with ReLU + Dropout)
        layers = []
        prev_size = ob_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, ac_dim))
        # NO activation on output — linear, matching Commander
        self.mean_net = nn.Sequential(*layers)

        # Freeze hidden layers if requested (only train output layer + logstd)
        if freeze_hidden:
            output_idx = len(layers) - 1  # last layer is the output Linear
            for i, layer in enumerate(self.mean_net):
                if i < output_idx:
                    for p in layer.parameters():
                        p.requires_grad = False
            n_frozen = sum(1 for p in self.mean_net.parameters() if not p.requires_grad)
            n_total = sum(1 for p in self.mean_net.parameters())
            print(f"  [freeze] Frozen {n_frozen}/{n_total} parameter tensors (hidden layers)")

        # Learned log-std for Gaussian (needed for AWAC log_prob)
        self.logstd = nn.Parameter(
            torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
        )

        self.mean_net.to(ptu.device)
        self.logstd.to(ptu.device)

        # Only optimize trainable params
        trainable = [p for p in self.mean_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable + [self.logstd], lr=lr)

    def forward(self, obs):
        mean = self.mean_net(obs)  # linear output, no tanh
        std = torch.exp(torch.clamp(self.logstd, min=-5.0, max=2.0))
        return torch.distributions.MultivariateNormal(
            mean, scale_tril=torch.diag(std).unsqueeze(0).expand(mean.shape[0], -1, -1)
        )

    def get_action(self, obs, deterministic=False):
        if obs.ndim == 1:
            obs = obs[None]
        obs = ptu.from_numpy(obs)
        dist = self(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        return ptu.to_numpy(action[0])

    def update(self, observations, actions, adv_n):
        """AWAC policy update — same loss as hw3, no tanh."""
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)

        dist = self(observations)
        log_prob = dist.log_prob(actions)
        weights = torch.exp(adv_n / self.lambda_awac).clamp(max=50.0)
        loss = -(log_prob * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


# ── IQL Agent (adapted from hw3, uses LinearGaussianActor) ────────────

class SingerIQLAgent:
    """IQL agent for SINGER. Same algorithm as hw3, linear actor."""

    def __init__(self, params):
        self.batch_size = params["batch_size"]
        self.learning_freq = params.get("learning_freq", 1)
        self.optimizer_spec = params["optimizer_spec"]
        self.rew_shift = params.get("rew_shift", 0.0)
        self.rew_scale = params.get("rew_scale", 1.0)
        self.t = 0
        self.num_param_updates = 0

        ac_dim = params["ac_dim"]

        # Replay buffer
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            params.get("replay_buffer_size", int(2e6)),
            1, float_obs=True, continuous_action=True, ac_dim=ac_dim,
        )

        # Critic (Q + V networks) — same as hw3
        self.critic = IQLCritic(params, self.optimizer_spec)

        # Actor — linear output, no tanh
        self.actor = LinearGaussianActor(
            ob_dim=params["ob_dim"],
            ac_dim=ac_dim,
            hidden_sizes=params["actor_hidden_sizes"],
            lr=params["actor_lr"],
            lambda_awac=params["awac_lambda"],
            max_grad_norm=params.get("grad_norm_clipping", 10),
            dropout=params.get("actor_dropout", 0.0),
            freeze_hidden=params.get("freeze_hidden", False),
        )

        self.eval_policy = self.actor

    def estimate_advantage(self, ob_no, ac_na):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        with torch.no_grad():
            v = self.critic.v_net(ob_no).squeeze(-1)
            q = self.critic.get_q(ob_no, ac_na)
            adv = q - v
        return adv

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample(batch_size)
        return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)):
            env_reward = (re_n + self.rew_shift) * self.rew_scale

            # 1. V-update
            critic_loss = self.critic.update_v(ob_no, ac_na)

            # 2. Actor-update (AWAC)
            adv = self.estimate_advantage(ob_no, ac_na)
            actor_loss = self.actor.update(ob_no, ac_na, adv_n=adv)

            # 3. Q-update
            critic_loss.update(
                self.critic.update_q(ob_no, ac_na, next_ob_no, env_reward, terminal_n)
            )

            # 4. Soft target update
            self.critic.update_target_network()

            log["Critic V Loss"] = critic_loss["Training V Loss"]
            log["Critic Q Loss"] = critic_loss["Training Q Loss"]
            log["Actor Loss"] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log


# ── Helpers ───────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_commander_weights(agent, cohort, pilot_name, model_file="model_best.pth"):
    """Initialize actor from trained Commander weights.

    The Commander is inside the Pilot's SVNet. We extract its weights and
    map them onto the actor's mean_net.
    """
    model_path = os.path.join(
        SINGER_ROOT, "cohorts", cohort, "roster", pilot_name, model_file
    )
    if not os.path.exists(model_path):
        print(f"  WARNING: Commander weights not found at {model_path}")
        return False

    # Load full pilot model
    sys.path.insert(0, os.path.join(SINGER_ROOT, "src"))
    from sousvide.control.pilot import Pilot
    pilot = Pilot(cohort, pilot_name)

    # Extract Commander weights
    commander = pilot.model.network["CommanderSV"]
    commander_sd = commander.state_dict()

    # Map Commander weights onto actor.mean_net
    # Commander is a SimpleMLP with .networks = nn.Sequential(...)
    # Actor's mean_net is also nn.Sequential with same structure
    actor_sd = agent.actor.mean_net.state_dict()

    # Check compatibility
    cmd_keys = [k for k in commander_sd.keys() if "networks" in k]
    act_keys = list(actor_sd.keys())

    print(f"  Commander keys: {len(cmd_keys)}, Actor keys: {len(act_keys)}")

    if len(cmd_keys) != len(act_keys):
        print(f"  WARNING: key count mismatch, skipping weight init")
        return False

    # Map: commander.networks.X -> mean_net.X
    new_sd = {}
    for ck, ak in zip(sorted(cmd_keys), sorted(act_keys)):
        if commander_sd[ck].shape != actor_sd[ak].shape:
            print(f"  WARNING: shape mismatch {ck}{commander_sd[ck].shape} vs {ak}{actor_sd[ak].shape}")
            return False
        new_sd[ak] = commander_sd[ck]

    agent.actor.mean_net.load_state_dict(new_sd)
    print(f"  Loaded Commander weights from {model_path}")
    del pilot
    return True


def load_transitions(cfg):
    """Load .npz transitions into arrays."""
    data_dir = os.path.join(SINGER_ROOT, cfg["data_dir"])
    datasets = []

    if cfg.get("use_expert_data", True):
        path = os.path.join(data_dir, "expert_transitions.npz")
        d = np.load(path)
        datasets.append(d)
        print(f"  Expert: {d['obs'].shape[0]} transitions, obs_dim={d['obs'].shape[1]}")

    if cfg.get("use_pilot_data", True):
        path = os.path.join(data_dir, "pilot_transitions.npz")
        d = np.load(path)
        datasets.append(d)
        print(f"  Pilot:  {d['obs'].shape[0]} transitions, obs_dim={d['obs'].shape[1]}")

    obs = np.concatenate([d["obs"] for d in datasets]).astype(np.float32)
    actions = np.concatenate([d["action"] for d in datasets]).astype(np.float32)
    next_obs = np.concatenate([d["next_obs"] for d in datasets]).astype(np.float32)
    rewards = np.concatenate([d["reward"] for d in datasets]).astype(np.float32)
    dones = np.concatenate([d["done"] for d in datasets])

    print(f"  Total: {obs.shape[0]} transitions, obs_dim={obs.shape[1]}, "
          f"ac_dim={actions.shape[1]}, {int(dones.sum())} episodes")

    return obs, actions, next_obs, rewards, dones


def load_into_buffer(agent, obs, actions, next_obs, rewards, dones):
    buf = agent.replay_buffer
    n = len(obs)

    if buf.obs is None:
        buf.obs = np.empty([buf.size] + list(obs.shape[1:]), dtype=np.float32)
        buf.action = np.empty([buf.size, actions.shape[1]], dtype=np.float32)
        buf.reward = np.empty([buf.size], dtype=np.float32)
        buf.done = np.empty([buf.size], dtype=bool)

    buf.obs[:n] = obs
    buf.action[:n] = actions
    buf.reward[:n] = rewards
    buf.done[:n] = dones
    buf.num_in_buffer = n
    buf.next_idx = n % buf.size

    if buf.next_obs_explicit is None:
        buf.next_obs_explicit = np.empty(
            [buf.size] + list(next_obs.shape[1:]), dtype=np.float32
        )
    buf.next_obs_explicit[:n] = next_obs

    print(f"  Loaded {n} transitions into buffer (capacity {buf.size})")


def export_actor_to_model(agent, cfg, roster_dir):
    """Write IQL actor weights back into full Pilot model.pth for benchmarking.

    The benchmark code does `torch.load(model.pth)` which expects a full SVNet.
    We load the source model, overwrite the Commander weights with IQL actor
    weights, and save as model.pth in the IQL cohort's roster.
    """
    sys.path.insert(0, os.path.join(SINGER_ROOT, "src"))
    from sousvide.control.pilot import Pilot
    pilot = Pilot(cfg["source_cohort"], cfg["source_pilot"])

    commander = pilot.model.network["CommanderSV"]
    commander_sd = commander.state_dict()
    actor_sd = agent.actor.mean_net.state_dict()

    # Reverse map: actor mean_net keys -> Commander keys
    cmd_keys = sorted([k for k in commander_sd if "networks" in k])
    act_keys = sorted(actor_sd.keys())

    for ck, ak in zip(cmd_keys, act_keys):
        commander_sd[ck] = actor_sd[ak]

    commander.load_state_dict(commander_sd)
    model_path = os.path.join(roster_dir, "model.pth")
    torch.save(pilot.model.cpu(), model_path)
    print(f"  Exported IQL actor → {model_path}")
    del pilot


# ── Training loop ─────────────────────────────────────────────────────

def train(cfg, use_wandb=False):
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(use_gpu=True, gpu_id=cfg.get("gpu_id", 0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cohort = cfg["cohort"]
    logdir = os.path.join(SINGER_ROOT, "cohorts", cohort, f"iql_{timestamp}")
    os.makedirs(logdir, exist_ok=True)
    print(f"\nLogdir: {logdir}")

    # Copy source model (never overwrite original)
    roster_dir = os.path.join(SINGER_ROOT, "cohorts", cohort, "roster",
                              cfg.get("roster", ["InstinctJester"])[0])
    os.makedirs(roster_dir, exist_ok=True)
    src_path = os.path.join(
        SINGER_ROOT, "cohorts", cfg["source_cohort"], "roster",
        cfg["source_pilot"], cfg.get("source_model", "model_best.pth")
    )
    dst_path = os.path.join(roster_dir, "model_pre_iql.pth")
    if not os.path.exists(dst_path) and os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"  Copied source model to {dst_path}")

    # Save config
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        yaml.dump(cfg, f)

    # Load data
    print("\nLoading transitions...")
    obs, actions, next_obs, rewards, dones = load_transitions(cfg)
    ob_dim = obs.shape[1]
    ac_dim = actions.shape[1]
    print(f"  ob_dim={ob_dim}, ac_dim={ac_dim}")

    # Build agent params
    critic_lr = cfg.get("learning_rate", 3e-4)
    agent_params = {
        "ob_dim": ob_dim,
        "ac_dim": ac_dim,
        "discrete": False,
        "gamma": cfg.get("gamma", 0.99),
        "q_func": make_continuous_q_network(
            hidden_size=cfg.get("q_hidden_size", 256),
            n_layers=cfg.get("q_n_layers", 2),
        ),
        "v_func": make_value_network(
            hidden_size=cfg.get("q_hidden_size", 256),
            n_layers=cfg.get("q_n_layers", 2),
        ),
        "optimizer_spec": OptimizerSpec(
            constructor=optim.Adam,
            optim_kwargs=dict(lr=critic_lr),
            learning_rate_schedule=lambda epoch: 1.0,
        ),
        "replay_buffer_size": int(2e6),
        "learning_freq": 1,
        "batch_size": cfg.get("batch_size", 256),
        "grad_norm_clipping": cfg.get("grad_norm_clipping", 10),
        "iql_expectile": cfg.get("iql_expectile", 0.9),
        "iql_tau": cfg.get("iql_tau", 0.005),
        "awac_lambda": cfg.get("awac_lambda", 0.1),
        "rew_shift": cfg.get("rew_shift", 0.0),
        "rew_scale": cfg.get("rew_scale", 1.0),
        # Actor config
        "actor_lr": cfg.get("actor_learning_rate", 1e-4),
        "actor_hidden_sizes": cfg.get("actor_hidden_sizes", [100, 100]),
        "actor_dropout": cfg.get("actor_dropout", 0.2),
        "freeze_hidden": cfg.get("freeze_hidden", False),
    }

    # Build agent
    print("\nBuilding IQL agent...")
    print(f"  Actor: {ob_dim} -> {agent_params['actor_hidden_sizes']} -> {ac_dim} (linear output)")
    print(f"  Q/V nets: {cfg.get('q_hidden_size', 256)} x {cfg.get('q_n_layers', 2)}")
    agent = SingerIQLAgent(agent_params)

    # Initialize actor from Commander weights
    print("\nInitializing actor from Commander weights...")
    loaded = load_commander_weights(
        agent, cfg["source_cohort"], cfg["source_pilot"],
        cfg.get("source_model", "model_best.pth"),
    )
    if not loaded:
        print("  Starting from random initialization")

    # Load data into buffer
    load_into_buffer(agent, obs, actions, next_obs, rewards, dones)

    # Dataset stats
    ep_returns = []
    cur = 0.0
    for r, d in zip(rewards, dones):
        cur += r
        if d:
            ep_returns.append(cur)
            cur = 0.0
    ep_returns = np.array(ep_returns)
    print(f"\n  Episode returns: mean={ep_returns.mean():.1f} "
          f"min={ep_returns.min():.1f} max={ep_returns.max():.1f} "
          f"n={len(ep_returns)}")

    # Wandb
    if use_wandb:
        import wandb
        wandb.init(project="singer", name=f"IQL_{cohort}_{timestamp}", config=cfg)

    # Training
    num_steps = cfg.get("num_iql_steps", 100000)
    log_freq = cfg.get("log_freq", 1000)
    ckpt_freq = cfg.get("checkpoint_freq", 10000)
    batch_size = cfg.get("batch_size", 256)

    print(f"\nTraining: {num_steps} steps, batch={batch_size}")
    print(f"  gamma={agent_params['gamma']}, expectile={agent_params['iql_expectile']}, "
          f"awac_lambda={agent_params['awac_lambda']}, tau={agent_params['iql_tau']}")

    start = time.time()
    logs = {"v": [], "q": [], "actor": []}

    for step in range(1, num_steps + 1):
        batch = agent.sample(batch_size)
        log = agent.train(*batch)

        if log:
            logs["v"].append(log.get("Critic V Loss", 0))
            logs["q"].append(log.get("Critic Q Loss", 0))
            logs["actor"].append(log.get("Actor Loss", 0))

        if step % log_freq == 0:
            elapsed = time.time() - start
            avg_v = np.mean(logs["v"][-log_freq:]) if logs["v"] else 0
            avg_q = np.mean(logs["q"][-log_freq:]) if logs["q"] else 0
            avg_a = np.mean(logs["actor"][-log_freq:]) if logs["actor"] else 0
            print(f"[{step:>6d}/{num_steps}] V={avg_v:.4f} Q={avg_q:.4f} "
                  f"Actor={avg_a:.4f} ({step/elapsed:.0f} steps/s, {elapsed:.0f}s)")
            if use_wandb:
                import wandb
                wandb.log({"v_loss": avg_v, "q_loss": avg_q,
                           "actor_loss": avg_a, "step": step}, step=step)

        if step % ckpt_freq == 0:
            ckpt = os.path.join(logdir, f"checkpoint_{step}")
            os.makedirs(ckpt, exist_ok=True)
            agent.actor.save(os.path.join(ckpt, "actor.pt"))
            torch.save({
                "q_net": agent.critic.q_net.state_dict(),
                "q_net_target": agent.critic.q_net_target.state_dict(),
                "q_net2": agent.critic.q_net2.state_dict(),
                "q_net2_target": agent.critic.q_net2_target.state_dict(),
                "v_net": agent.critic.v_net.state_dict(),
            }, os.path.join(ckpt, "critic.pt"))
            print(f"  -> Checkpoint: {ckpt}")

    # Final save
    final = os.path.join(logdir, "final")
    os.makedirs(final, exist_ok=True)
    agent.actor.save(os.path.join(final, "actor.pt"))
    torch.save({
        "q_net": agent.critic.q_net.state_dict(),
        "q_net_target": agent.critic.q_net_target.state_dict(),
        "q_net2": agent.critic.q_net2.state_dict(),
        "q_net2_target": agent.critic.q_net2_target.state_dict(),
        "v_net": agent.critic.v_net.state_dict(),
    }, os.path.join(final, "critic.pt"))

    # Save loss curves
    np.savez(os.path.join(logdir, "losses.npz"),
             v=np.array(logs["v"]), q=np.array(logs["q"]),
             actor=np.array(logs["actor"]))

    # Export IQL actor back into full Pilot model.pth for benchmarking
    export_actor_to_model(agent, cfg, roster_dir)

    total = time.time() - start
    final_v = np.mean(logs["v"][-log_freq:])
    final_q = np.mean(logs["q"][-log_freq:])
    final_a = np.mean(logs["actor"][-log_freq:])

    summary = {
        "cohort": cohort, "steps": num_steps, "time_s": total,
        "n_transitions": len(obs), "n_episodes": len(ep_returns),
        "final_v": float(final_v), "final_q": float(final_q),
        "final_actor": float(final_a),
    }
    with open(os.path.join(logdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {num_steps} steps in {total:.0f}s")
    print(f"  V={final_v:.4f}  Q={final_q:.4f}  Actor={final_a:.4f}")
    print(f"  Output: {logdir}")

    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--freeze-hidden", action="store_true",
                        help="Freeze hidden layers, only train output + logstd")
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    if args.num_steps is not None:
        cfg["num_iql_steps"] = args.num_steps
    if args.freeze_hidden:
        cfg["freeze_hidden"] = True
    train(cfg, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
