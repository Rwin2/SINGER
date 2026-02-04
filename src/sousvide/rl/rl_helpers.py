"""
Helper functions for RL training pipeline.

Includes:
- Trajectory loading and parsing
- Advantage computation (GAE)
- KL divergence computation
- Expert policy querying
- Batch data preparation
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
import shutil
from typing import Tuple, Dict, List, Optional
from pathlib import Path


def move_simulation_results(cohort_path: str, validation_dir: str, episode: int = None, pilot_name: str = None):
    # Move entire simulation_data directory to validation_dir
    sim_data_dir = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(sim_data_dir):
        raise FileNotFoundError(f"No simulation_data directory found at {sim_data_dir}")

    dest_sim_dir = os.path.join(validation_dir, "simulation_data")
    shutil.move(sim_data_dir, dest_sim_dir)
    print(f"  ✓ Moved simulation_data directory to {dest_sim_dir}")


def filter_validation_trajectories(validation_dir: str, pilot_name: str, selected_indices: List[int]):
    """
    Filter trajectory files in validation directory to only keep selected trajectories.

    Handles structure: validation_dir/simulation_data/{timestamp}/trajectories/

    Each .pt file contains exactly one trajectory (wrapped in a list), so this function
    simply deletes files corresponding to unselected trajectory indices. It also deletes
    matching expert trajectory files to keep them in sync.

    Args:
        validation_dir: Path to validation_data directory (contains simulation_data subdirectory)
        pilot_name: Name of the pilot (to filter files)
        selected_indices: List of trajectory indices to keep (from stratified sampling)
    """
    # Navigate to simulation_data directory
    sim_data_dir = os.path.join(validation_dir, "simulation_data")
    if not os.path.exists(sim_data_dir):
        print(f"[WARNING] simulation_data directory not found at {sim_data_dir}")
        return

    # Find timestamp directory (should be only one after move)
    timestamp_dirs = sorted(glob.glob(os.path.join(sim_data_dir, "*")))
    if not timestamp_dirs:
        print(f"[WARNING] No timestamp directories found in {sim_data_dir}")
        return

    timestamp_dir = timestamp_dirs[-1]  # Use most recent (should only be one)
    trajectories_dir = os.path.join(timestamp_dir, "trajectories")

    if not os.path.exists(trajectories_dir):
        print(f"[WARNING] Trajectories directory not found at {trajectories_dir}")
        return

    # Get all trajectory files in sorted order (same as load_simulation_results)
    traj_files = sorted(glob.glob(os.path.join(trajectories_dir, "*.pt")))
    if not traj_files:
        print(f"[WARNING] No trajectory files found in {trajectories_dir}")
        return

    # Build mapping from trajectory index to file path
    # Each file contains exactly 1 trajectory, so index = file number
    traj_idx_to_file = {}  # traj_idx -> (file_path, traj_key)
    current_traj_idx = 0

    for traj_file in traj_files:
        filename = os.path.basename(traj_file)

        # Skip tester trajectories (files without _traj{idx}_ pattern)
        if "_traj" not in filename:
            continue

        # Only process pilot files (not expert files)
        if f"_{pilot_name}.pt" not in filename:
            continue

        # Extract trajectory key for matching with expert files
        # e.g., "sim_data_scene_obj_traj0_InstinctJester.pt" -> "sim_data_scene_obj_traj0"
        traj_key = filename.replace(f"_{pilot_name}.pt", "")

        # Map this trajectory index to its file
        traj_idx_to_file[current_traj_idx] = (traj_file, traj_key)
        current_traj_idx += 1

    print(f"[VALIDATION] Found {current_traj_idx} total policy trajectories in {len(traj_idx_to_file)} files")

    # Convert selected indices to set for fast lookup
    selected_set = set(selected_indices)

    # Track which expert trajectory keys to keep
    expert_keys_to_keep = set()

    # Delete unselected policy files
    policy_deleted = 0
    policy_kept = 0

    for traj_idx, (traj_file, traj_key) in traj_idx_to_file.items():
        filename = os.path.basename(traj_file)

        if traj_idx in selected_set:
            # Keep this trajectory
            policy_kept += 1
            expert_keys_to_keep.add(traj_key)
        else:
            # Delete this trajectory file
            try:
                os.remove(traj_file)
                policy_deleted += 1
                print(f"  [DELETED] {filename}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {filename}: {e}")

    # Delete expert files that don't have matching policy trajectories
    expert_deleted = 0
    expert_kept = 0

    for traj_file in traj_files:
        filename = os.path.basename(traj_file)

        # Only process expert files
        if "_expert.pt" not in filename:
            continue

        # Extract trajectory key
        # e.g., "sim_data_scene_obj_traj0_expert.pt" -> "sim_data_scene_obj_traj0"
        traj_key = filename.replace("_expert.pt", "")

        if traj_key in expert_keys_to_keep:
            # Keep this expert trajectory (matches a selected policy trajectory)
            expert_kept += 1
        else:
            # Delete this expert trajectory (no matching policy trajectory)
            try:
                os.remove(traj_file)
                expert_deleted += 1
                print(f"  [DELETED] {filename}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {filename}: {e}")

    print(f"[VALIDATION] Policy: {policy_kept} kept, {policy_deleted} deleted")
    print(f"[VALIDATION] Expert: {expert_kept} kept, {expert_deleted} deleted")


def load_simulation_results(cohort_path: str, episode: int = None, pilot_name: str = None) -> Tuple[list, Dict, Dict]:
    """
    Load trajectory simulation results from disk.

    Searches for most recent simulation data directory and loads all trajectory files.

    Args:
        cohort_path: Path to cohort directory
        episode: Episode number (for naming, optional)
        pilot_name: Optional pilot name to filter trajectories (e.g., "InstinctJester")
                   If provided, only loads trajectories from this pilot (skips expert)

    Returns:
        trajectories: List of Xro state arrays (one per simulated trajectory)
        metadata: Dict with goal_locations, collision_radii, scene names
        raw_data: Dict with full trajectory data for later use
    """
    # Find latest simulation_data directory
    sim_data_dir = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(sim_data_dir):
        raise FileNotFoundError(f"No simulation_data directory found at {sim_data_dir}")

    # Get most recent timestamp directory
    timestamp_dirs = sorted(glob.glob(os.path.join(sim_data_dir, "*")))
    if not timestamp_dirs:
        raise FileNotFoundError(f"No simulation timestamp directories found in {sim_data_dir}")

    latest_sim_dir = timestamp_dirs[-1]
    trajectories_dir = os.path.join(latest_sim_dir, "trajectories")

    if not os.path.exists(trajectories_dir):
        raise FileNotFoundError(f"No trajectories directory found at {trajectories_dir}")

    # Load all trajectory files
    traj_files = sorted(glob.glob(os.path.join(trajectories_dir, "*.pt")))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {trajectories_dir}")

    trajectories = []
    metadata = {
        "goal_locations": [],
        "collision_radii": [],
        "exclusion_radii": [],
        "scene_names": [],
        "course_names": [],
    }
    raw_data = []

    for traj_file in traj_files:
        # Skip tester trajectories (files without _traj{idx}_ pattern)
        filename = os.path.basename(traj_file)
        if "_traj" not in filename:
            continue

        # Skip expert trajectories if we only want a specific pilot
        if pilot_name is not None and f"_{pilot_name}.pt" not in filename:
            continue

        # Load trajectory data
        data_list = torch.load(traj_file)  # List of trajectory dicts

        for data in data_list:
            # Extract state trajectory
            Xro = data.get("Xro", None)
            if Xro is not None:
                trajectories.append(Xro)

            # Extract metadata
            if "goal_location" in data:
                metadata["goal_locations"].append(np.array(data["goal_location"]))
            if "collision_radius" in data:
                metadata["collision_radii"].append(data["collision_radius"])
            if "exclusion_radius" in data:
                metadata["exclusion_radii"].append(data["exclusion_radius"])
            if "course" in data:
                metadata["scene_names"].append(data.get("course", "unknown"))

            raw_data.append(data)

    # Convert lists to numpy arrays where appropriate
    if metadata["goal_locations"]:
        metadata["goal_locations"] = np.array(metadata["goal_locations"])
    if metadata["collision_radii"]:
        metadata["collision_radii"] = np.array(metadata["collision_radii"])

    return trajectories, metadata, raw_data


def compute_advantages_gae(rewards: np.ndarray,
                          values: np.ndarray,
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages using Generalized Advantage Estimation (GAE).

    Args:
        rewards: Per-timestep rewards (T,)
        values: Per-timestep value estimates (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (0 = TD residual, 1 = MC return)

    Returns:
        advantages: GAE advantages (T,)
        returns: Discounted cumulative returns (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0

    # Compute GAE backwards from end of trajectory
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]

        # TD residual
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE accumulation
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    # Compute returns
    returns = advantages + values

    return advantages, returns


def compute_advantages_mc(rewards: np.ndarray,
                         gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages using Monte Carlo returns (no value network needed).

    This is the simplest advantage estimation - just use actual discounted returns.
    Advantages are computed by subtracting the mean return (simple baseline).

    Args:
        rewards: Per-timestep rewards (T,)
        gamma: Discount factor

    Returns:
        advantages: MC advantages (T,) - normalized returns
        returns: Discounted cumulative returns (T,)
    """
    T = len(rewards)
    returns = np.zeros(T)

    # Compute returns backward (sum of discounted future rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Advantage = return - mean(returns) as simple baseline
    # This centers the advantages around 0
    mean_return = np.mean(returns)
    advantages = returns - mean_return

    return advantages, returns


def compute_mc_rewards(clearances: np.ndarray,
                       collision_detected: bool,
                       collision_index: int,
                       query_in_view: np.ndarray = None,
                       # Reward component toggles
                       use_collision: bool = True,
                       use_clearance: bool = True,
                       use_query_in_view: bool = False,
                       # Reward weights
                       collision_penalty: float = 5.0,
                       clearance_weight: float = 1.0,
                       query_reward: float = 0.5,
                       clearance_threshold: float = 0.3,
                       success_reward: float = 10.0) -> np.ndarray:
    """
    Compute per-timestep rewards for Monte Carlo returns.
    
    Each reward component can be toggled on/off for ablation studies.
    
    Args:
        clearances: Per-timestep clearance distances (T,)
        collision_detected: Whether collision occurred
        collision_index: Index of collision (or len(clearances) if none)
        query_in_view: Per-timestep query visibility scores (T,), optional
        
        use_collision: Include terminal collision/success penalty
        use_clearance: Include per-step clearance-based penalties
        use_query_in_view: Include per-step query visibility reward
        
        collision_penalty: Weight for collision terminal penalty
        clearance_weight: Weight for per-step clearance reward
        query_reward: Weight for query-in-view reward
        clearance_threshold: Distance threshold for clearance penalty (meters)
        success_reward: Terminal reward for collision-free trajectory
        
    Returns:
        rewards: (T,) array of per-timestep rewards
    """
    T = len(clearances)
    rewards = np.zeros(T)
    
    # Truncate to collision point if collision occurred
    effective_T = min(T, collision_index + 1) if collision_detected else T
    
    for t in range(effective_T):
        reward_t = 0.0
        
        # Component 1: Clearance-based penalty (proximity to obstacles)
        if use_clearance:
            proximity = max(0.0, clearance_threshold - clearances[t])
            reward_t -= clearance_weight * proximity
        
        # Component 2: Query-in-view reward (encourage keeping target visible)
        if use_query_in_view and query_in_view is not None and t < len(query_in_view):
            # query_in_view should be in [0, 1] where 1 = fully visible
            reward_t += query_reward * query_in_view[t]
        
        rewards[t] = reward_t
    
    # Component 3: Terminal collision/success reward
    if use_collision:
        if collision_detected and collision_index < T:
            rewards[min(collision_index, T-1)] -= collision_penalty
        elif not collision_detected:
            # Small success bonus at end
            rewards[-1] += success_reward * 0.1  # Smaller than collision penalty
    
    return rewards


def compute_simple_advantage(rewards: np.ndarray,
                            values: np.ndarray,
                            gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages using simple TD residuals.

    Simpler alternative to GAE for faster computation.

    Args:
        rewards: Per-timestep rewards (T,)
        values: Per-timestep value estimates (T,)
        gamma: Discount factor

    Returns:
        advantages: TD advantages (T,)
        returns: Discounted returns (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T)

    for t in range(T):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        advantages[t] = rewards[t] + gamma * next_value - values[t]

    returns = advantages + values

    return advantages, returns


def compute_kl_divergence_continuous(mu1: torch.Tensor,
                                     sigma1: torch.Tensor,
                                     mu2: torch.Tensor,
                                     sigma2: torch.Tensor,
                                     eps: float = 1e-8) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussian distributions (for continuous actions).

    KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))

    Args:
        mu1: Mean of first distribution (batch_size, action_dim)
        sigma1: Std dev of first distribution (batch_size, action_dim)
        mu2: Mean of second distribution (batch_size, action_dim)
        sigma2: Std dev of second distribution (batch_size, action_dim)
        eps: Small epsilon for numerical stability

    Returns:
        kl_div: KL divergence (scalar, mean over batch)
    """
    # Avoid log(0)
    sigma1 = torch.clamp(sigma1, min=eps)
    sigma2 = torch.clamp(sigma2, min=eps)

    # KL(N1 || N2) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 1/2
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2

    kl = (torch.log(sigma2 / sigma1) +
          (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5)

    return kl.mean()


def compute_bc_loss_from_trajectories(policy_dir, expert_dir, pilot_name, collision_detector):
    """Compute BC loss between policy and expert trajectories.
    
    Only compares actions up to the first collision in the policy trajectory.
    This ensures fair comparison - we don't penalize divergence after failure.
    """
    bc_losses = []
    
    # Load expert trajectories
    expert_trajs = {}
    for expert_file in glob.glob(os.path.join(expert_dir, "*_expert.pt")):
        filename = os.path.basename(expert_file)
        # Extract scene_objective key
        key = filename.replace("sim_data_", "").replace("_expert.pt", "")
        data_list = torch.load(expert_file)
        expert_trajs[key] = data_list
    
    # Load policy trajectories and compute BC loss
    for policy_file in glob.glob(os.path.join(policy_dir, f"*_{pilot_name}.pt")):
        filename = os.path.basename(policy_file)
        key = filename.replace("sim_data_", "").replace(f"_{pilot_name}.pt", "")
        
        if key not in expert_trajs:
            continue
        
        policy_data_list = torch.load(policy_file)
        expert_data_list = expert_trajs[key]
        
        # Match trajectories and compute BC loss
        for p_data, e_data in zip(policy_data_list, expert_data_list):
            p_actions = p_data.get("Uro", None)  # Policy actions
            e_actions = e_data.get("Uro", None)  # Expert actions
            p_states = p_data.get("Xro", None)   # Policy states for collision detection
            
            if p_actions is not None and e_actions is not None:
                # Find first collision in policy trajectory
                collision_idx = p_actions.shape[1]  # Default: full trajectory
                if p_states is not None and collision_detector is not None:
                    positions = p_states[:3, :].T  # (T, 3)
                    clearances = collision_detector.compute_clearances(positions)
                    collision_mask = clearances < collision_detector.collision_radius
                    if np.any(collision_mask):
                        collision_idx = np.argmax(collision_mask)
                
                # Only compare up to collision (or min length)
                compare_len = min(p_actions.shape[1], e_actions.shape[1], collision_idx)
                if compare_len > 0:
                    p_act = torch.tensor(p_actions[:, :compare_len], dtype=torch.float32)
                    e_act = torch.tensor(e_actions[:, :compare_len], dtype=torch.float32)
                    
                    # Compute MSE loss
                    bc_loss = F.mse_loss(p_act, e_act).item()
                    bc_losses.append(bc_loss)
    
    return np.mean(bc_losses) if bc_losses else float('nan')


def compute_bc_loss_from_observations(obs_dir, policy_network, device, batch_size=32):
    """Compute BC loss by running policy on expert states (batched).
    
    This is the proper BC validation metric:
    - Load pre-generated observation files (expert states with embeddings)
    - Batch samples together for efficient forward pass
    - Compare policy output vs expert action
    
    This avoids trajectory divergence issues since we're always evaluating
    on the expert's distribution of states.
    
    Args:
        obs_dir: Directory containing observation files (observations_val_*.pt)
        policy_network: The policy network to evaluate
        device: torch device
        batch_size: Number of samples to process at once (must be > 1 for proper batching)
        
    Returns:
        Mean BC loss across all timesteps, total timesteps
    """
    # First, collect all observation-action pairs
    all_observations = []  # List of (xnn_dict, expert_action)
    
    for obs_file in glob.glob(os.path.join(obs_dir, "observations_val_*.pt")):
        try:
            obs_data = torch.load(obs_file, weights_only=False)
            
            for traj_data in obs_data["data"]:
                Xnn_list = traj_data["Xnn"]
                Ynn_list = traj_data["Ynn"]
                
                for t in range(len(Xnn_list)):
                    xnn = Xnn_list[t]
                    ynn = Ynn_list[t]
                    
                    # Get expert action
                    if isinstance(ynn, dict):
                        expert_action = ynn.get("unn", None)
                    else:
                        expert_action = ynn
                    
                    if expert_action is None or not isinstance(xnn, dict):
                        continue
                    
                    all_observations.append((xnn, expert_action))
                    
        except Exception as e:
            print(f"  ⚠ Error loading {os.path.basename(obs_file)}: {e}")
            continue
    
    if not all_observations:
        return float('nan'), 0
    
    # Process in batches
    bc_losses = []
    total_timesteps = 0
    
    policy_network.eval()
    with torch.no_grad():
        for batch_start in range(0, len(all_observations), batch_size):
            batch_end = min(batch_start + batch_size, len(all_observations))
            batch = all_observations[batch_start:batch_end]
            
            # Skip batches of size 1 (causes dimension issues with squeeze)
            if len(batch) < 2:
                continue
            
            try:
                # Helper to convert to tensor (handles both numpy and existing tensors)
                def to_tensor(x):
                    if isinstance(x, torch.Tensor):
                        return x.clone().detach().float()
                    return torch.tensor(x, dtype=torch.float32)
                
                # Collect batch tensors
                tx_com_list, obj_com_list, dxu_par_list = [], [], []
                img_vis_list, tx_vis_list, expert_actions_list = [], [], []
                
                for xnn, expert_action in batch:
                    tx_com_list.append(to_tensor(xnn["tx_com"]))
                    obj_com_list.append(to_tensor(xnn["obj_com"]))
                    dxu_par_list.append(to_tensor(xnn["dxu_par"]))
                    img_vis_list.append(to_tensor(xnn["img_vis"]))
                    tx_vis_list.append(to_tensor(xnn["tx_vis"]))
                    expert_actions_list.append(to_tensor(expert_action))
                
                # Stack into batched tensors
                tx_com_batch = torch.stack(tx_com_list).to(device)
                obj_com_batch = torch.stack(obj_com_list).to(device)
                dxu_par_batch = torch.stack(dxu_par_list).to(device)
                img_vis_batch = torch.stack(img_vis_list).to(device)
                tx_vis_batch = torch.stack(tx_vis_list).to(device)
                expert_actions_batch = torch.stack(expert_actions_list).to(device)
                
                # Forward pass through policy
                policy_output = policy_network(
                    tx_com_batch, obj_com_batch, dxu_par_batch,
                    img_vis_batch, tx_vis_batch
                )
                policy_actions = policy_output[0][:, :4]  # Get actions (first 4 dims)
                
                # Compute MSE loss for batch
                bc_loss = F.mse_loss(policy_actions, expert_actions_batch).item()
                bc_losses.append(bc_loss * len(batch))  # Weight by batch size
                total_timesteps += len(batch)
                
            except Exception as e:
                print(f"  ⚠ Batch forward pass error: {e}")
                continue
    
    policy_network.train()
    
    if bc_losses and total_timesteps > 0:
        return sum(bc_losses) / total_timesteps, total_timesteps
    else:
        return float('nan'), 0


def select_high_risk_states(clearances: np.ndarray,
                           collision_index: int,
                           clearance_threshold: float = 0.3,
                           top_k_pct: float = 0.1) -> np.ndarray:
    """
    Select indices of high-risk states for expert querying.

    Prioritizes states close to obstacles or just before collision.

    Args:
        clearances: Per-timestep clearance distances (T,)
        collision_index: Index of first collision (-1 if none)
        clearance_threshold: Threshold distance for "risky" states
        top_k_pct: Fraction of trajectory to select as expert queries

    Returns:
        risk_indices: Indices of selected high-risk states
    """
    T = len(clearances)
    risk_scores = np.zeros(T)

    # Score 1: Proximity to obstacles
    for t in range(T):
        proximity = max(0.0, clearance_threshold - clearances[t])
        risk_scores[t] += proximity

    # Score 2: Proximity to collision point
    if collision_index >= 0:
        for t in range(max(0, collision_index - 5), collision_index + 1):
            risk_scores[t] += 10.0  # High weight for near-collision states

    # Select top-k by risk score
    k = max(1, int(T * top_k_pct))
    risk_indices = np.argsort(-risk_scores)[:k]

    return np.sort(risk_indices)


def prepare_batch_data(trajectories: list,
                      metadata: Dict,
                      collision_detector,
                      device: torch.device) -> Dict:
    """
    Prepare trajectory data for batch training.

    Computes collision metrics, rewards, and advantages for all trajectories.

    Args:
        trajectories: List of Xro state arrays
        metadata: Dict with goal_locations, collision_radii
        collision_detector: CollisionDetector instance
        device: Torch device (cuda or cpu)

    Returns:
        batch_data: Dict with processed trajectories, rewards, advantages, etc.
    """
    from sousvide.rl.collision_detector import compute_collision_rewards

    batch_data = {
        "states": [],
        "clearances": [],
        "rewards": [],
        "returns": [],
        "advantages": [],
        "collision_masks": [],
        "trajectory_lengths": [],
        "analyses": []
    }

    for i, Xro in enumerate(trajectories):
        # Analyze trajectory for collisions
        analysis = collision_detector.analyze_trajectory(Xro)
        batch_data["analyses"].append(analysis)

        # Extract clearances and collision info
        clearances = analysis["clearances"]
        collision_detected = analysis["collision"]
        collision_index = analysis["collision_index"]

        batch_data["clearances"].append(clearances)

        # Compute rewards
        per_step_rewards, terminal_reward = compute_collision_rewards(
            clearances,
            collision_detected,
            collision_index,
            collision_penalty=5.0,
            clearance_threshold=0.3
        )

        # Add terminal reward to last timestep
        full_rewards = per_step_rewards.copy()
        full_rewards[-1] += terminal_reward

        batch_data["rewards"].append(full_rewards)
        batch_data["trajectory_lengths"].append(len(clearances))

        # Placeholder for advantages (computed later with value estimates)
        batch_data["collision_masks"].append(np.zeros_like(clearances))

    # Compute aggregate statistics
    all_clearances = np.concatenate(batch_data["clearances"])
    collision_count = sum(a["collision"] for a in batch_data["analyses"])
    collision_rate = collision_count / len(trajectories) if trajectories else 0.0

    batch_data["collision_rate"] = collision_rate
    batch_data["mean_clearance"] = float(np.mean(all_clearances))
    batch_data["min_clearance"] = float(np.min(all_clearances))
    batch_data["std_clearance"] = float(np.std(all_clearances))

    return batch_data


def compute_policy_gradient_loss(log_probs: torch.Tensor,
                                advantages: torch.Tensor,
                                entropy_coeff: float = 0.01) -> torch.Tensor:
    """
    Compute policy gradient loss (simplified PPO-style).

    Args:
        log_probs: Log probabilities of taken actions (batch_size,)
        advantages: Estimated advantages (batch_size,)
        entropy_coeff: Entropy regularization coefficient

    Returns:
        loss: Scalar policy gradient loss
    """
    # Policy gradient: -mean(log_prob * advantage)
    # Negative because we want to maximize expected return
    policy_loss = -(log_probs * advantages).mean()

    return policy_loss


def compute_bc_loss(policy_outputs: torch.Tensor,
                   expert_actions: torch.Tensor) -> torch.Tensor:
    """
    Compute behavior cloning loss (MSE).

    Args:
        policy_outputs: Policy action predictions (batch_size, action_dim)
        expert_actions: Expert action labels (batch_size, action_dim)

    Returns:
        loss: Scalar BC loss
    """
    return F.mse_loss(policy_outputs, expert_actions)


def compute_kl_loss(policy_outputs_mean: torch.Tensor,
                   policy_outputs_std: torch.Tensor,
                   expert_outputs_mean: torch.Tensor,
                   expert_outputs_std: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence loss between current and initial policy.

    Args:
        policy_outputs_mean: Current policy action means
        policy_outputs_std: Current policy action std devs
        expert_outputs_mean: Initial BC policy action means
        expert_outputs_std: Initial BC policy action std devs

    Returns:
        loss: Scalar KL loss
    """
    return compute_kl_divergence_continuous(
        policy_outputs_mean, policy_outputs_std,
        expert_outputs_mean, expert_outputs_std
    )


def normalize_advantages(advantages: np.ndarray,
                        eps: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages to zero mean and unit variance.

    Improves training stability.

    Args:
        advantages: Raw advantages (batch_size,)
        eps: Small epsilon for numerical stability

    Returns:
        normalized_advantages: Normalized advantages
    """
    # CRITICAL FIX: Clip extreme advantages before normalization
    advantages_clipped = np.clip(advantages, -100.0, 100.0)
    
    mean = np.mean(advantages_clipped)
    std = np.std(advantages_clipped)
    
    # Prevent division by very small std (would amplify noise)
    if std < 0.01:
        std = 1.0
    
    normalized = (advantages_clipped - mean) / (std + eps)
    
    # Final safety clip after normalization
    normalized = np.clip(normalized, -10.0, 10.0)
    
    return normalized


def prepare_state_batch(trajectory_data: Dict, pilot, device: torch.device) -> List[Tuple]:
    """
    Extract state tuples from trajectory for value estimation.

    Converts trajectory data (states, images, objectives, history) into the format
    expected by SVNet.forward() for batch value prediction by the critic.

    Args:
        trajectory_data: Dict with trajectory states, images, objectives, history
                        Expected keys: "states", "objectives", "images", "history"
        pilot:          Pilot object (used for feature extraction if needed)
        device:         Torch device (cpu or cuda)

    Returns:
        List of (tx_com, obj_com, dxu_par, img_vis, tx_vis) tuples for each timestep
    """
    states_batch = []

    # Get trajectory length
    num_timesteps = len(trajectory_data["states"])

    for t in range(num_timesteps):
        # Extract raw state at timestep t
        state_t = trajectory_data["states"][t]
        obj_t = trajectory_data.get("objectives", [np.zeros(0)])[t] if t < len(trajectory_data.get("objectives", [])) else np.zeros(0)
        img_t = trajectory_data["images"][t]
        history_t = trajectory_data["history"][t]

        # Convert to model inputs (same format as pilot.decide())
        # Note: During training, network expects batched inputs, but during inference it handles unbatched
        # We'll add batch dimension later when calling the network
        tx_com = torch.tensor(state_t, dtype=torch.float32, device=device)
        obj_com = torch.tensor(obj_t, dtype=torch.float32, device=device)

        # Select last 5 frames from history (HistoryEncoder expects 15 features × 5 frames = 75 elements)
        history_subset = history_t[:, -5:]  # (15, 5)
        dxu_par = torch.tensor(history_subset, dtype=torch.float32, device=device).flatten()  # (75,)

        # Preprocess image using pilot's preprocessing pipeline (resize, crop, normalize)
        # This ensures images match what the network was trained on (224x224)
        try:
            # Debug: Check image format
            if t == 0:  # Only print for first timestep
                print(f"[DEBUG] Raw image shape: {img_t.shape}, dtype: {img_t.dtype}, range: [{img_t.min()}, {img_t.max()}]")

            img_preprocessed = pilot.process_image(img_t)  # Returns tensor (C, H, W)
            img_vis = img_preprocessed.to(device)

            if t == 0:
                print(f"[DEBUG] Preprocessed image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
                print(f"[DEBUG] tx_com shape: {tx_com.shape}")
                print(f"[DEBUG] dxu_par shape: {dxu_par.shape}")
        except Exception as e:
            print(f"[DEBUG] Image preprocessing failed at timestep {t}: {e}")
            import traceback
            traceback.print_exc()
            raise

        tx_vis = tx_com  # Use same state for vision input

        states_batch.append((tx_com, obj_com, dxu_par, img_vis, tx_vis))

    return states_batch


def extract_trajectory_data_for_critic(raw_trajectory: Dict,
                                       pilot,
                                       use_rgb: bool = True) -> Dict:
    """
    Extract and format trajectory data for critic value estimation.

    Converts raw simulation data (Xro, Iro, Uro) into the format expected by
    prepare_state_batch() for critic forward passes.

    Args:
        raw_trajectory: Dict from load_simulation_results() with keys:
                       - "Xro": State trajectory (state_dim × N)
                       - "Iro": Images dict {"rgb": (N, H, W, 3), "semantic": ..., "depth": ...}
                       - "Uro": Control inputs (4 × N)
                       - "obj": Objectives (18 × 1)
                       - "Tro": Time array (N+1,)
        pilot:         Pilot instance (used to get history window size)
        use_rgb:       If True, use RGB images; else use semantic images

    Returns:
        Dict with keys for prepare_state_batch():
        - "states": List of state vectors (11,) per timestep [time; state]
        - "images": List of image arrays (H, W, 3) per timestep
        - "objectives": List of objective vectors per timestep
        - "history": List of delta history (15, nhy) per timestep
    """
    # Extract raw data
    Xro = raw_trajectory["Xro"]  # (state_dim, N)
    Uro = raw_trajectory["Uro"]  # (4, N)
    obj = raw_trajectory.get("obj", np.zeros((18, 1)))  # (18, 1) or (18,)
    Tro = raw_trajectory["Tro"]  # (N+1,)

    # Check if Iro exists (required for critic)
    if "Iro" not in raw_trajectory:
        raise KeyError("Iro")  # Will be caught and handled by caller

    Iro = raw_trajectory["Iro"]  # Dict with image arrays

    # Select image type
    if use_rgb and "rgb" in Iro:
        images_raw = Iro["rgb"]  # (N, H, W, 3)
    else:
        images_raw = Iro.get("semantic", Iro.get("rgb"))  # Fallback to semantic or rgb

    # Get trajectory length from IMAGES (not states, since Xro includes initial state)
    # Xro has shape (state_dim, N+1) where N+1 includes initial state
    # Iro has N images (one per timestep, excluding initial state)
    N = len(images_raw)

    # Verify Xro has N+1 states (initial state + N timesteps)
    if Xro.shape[1] != N + 1:
        raise ValueError(f"Expected Xro to have {N+1} states (initial + {N} timesteps), but got {Xro.shape[1]}")

    # Prepare output lists
    states_list = []
    images_list = []
    objectives_list = []
    history_list = []

    # Get history window size from pilot
    nhy = pilot.DxU.shape[1]  # History window size (typically 20)

    # Process each timestep (image index t corresponds to state index t+1)
    for t in range(N):
        # Image index t corresponds to timestep t+1 (since index 0 is initial state)
        state_idx = t + 1

        # State: [time; x; y; z; vx; vy; vz; qx; qy; qz; qw]
        # Xro is (state_dim, N+1), we use state_idx to skip initial state
        state_t = np.concatenate(([Tro[state_idx]], Xro[:10, state_idx]))  # (11,) = [time; state[10]]

        # Image: (H, W, 3)
        image_t = images_raw[t]  # Image at timestep t

        # Objective: (18,) or (18, 1) → flatten to (18,)
        if obj.ndim == 2:
            objective_t = obj[:, 0] if obj.shape[1] > 0 else obj[:, -1]
        else:
            objective_t = obj

        # History deltas: (15, nhy)
        # For now, create a simplified history using velocity and control data
        # Full version would track actual delta states over time window
        # Format: [dt, dx, dy, dz, dvx, dvy, dvz, dqx, dqy, dqz, dqw, u1, u2, u3, u4]

        # Create history window - fill with recent data
        history_t = np.zeros((15, nhy))

        # Fill the most recent entry (last column) with current data
        if state_idx > 0:
            history_t[0, -1] = Tro[state_idx] - Tro[state_idx-1]  # Time delta
            history_t[1:4, -1] = Xro[0:3, state_idx] - Xro[0:3, state_idx-1]  # Position delta
            history_t[4:7, -1] = Xro[3:6, state_idx] - Xro[3:6, state_idx-1]  # Velocity delta
            history_t[7:11, -1] = Xro[6:10, state_idx]  # Quaternion (simplified - should be delta)
            history_t[11:15, -1] = Uro[:, state_idx-1]  # Previous control (Uro is 0-indexed)

        # Fill earlier history entries if available
        for i in range(min(state_idx, nhy-1)):
            lookback = state_idx - nhy + 1 + i
            if lookback >= 0 and lookback < state_idx:
                history_t[0, i] = Tro[lookback+1] - Tro[lookback]
                history_t[1:4, i] = Xro[0:3, lookback+1] - Xro[0:3, lookback]
                history_t[4:7, i] = Xro[3:6, lookback+1] - Xro[3:6, lookback]
                history_t[7:11, i] = Xro[6:10, lookback+1]
                history_t[11:15, i] = Uro[:, lookback]

        states_list.append(state_t)
        images_list.append(image_t)
        objectives_list.append(objective_t)
        history_list.append(history_t)

    return {
        "states": states_list,
        "images": images_list,
        "objectives": objectives_list,
        "history": history_list
    }


def generate_observations_from_trajectory(raw_trajectory: Dict, pilot, device: torch.device) -> List[Dict]:
    """
    Generate observation data from raw trajectory using OODA pipeline (matching BC training).

    This function replicates the observation_generator.py pipeline to ensure critic
    receives data in the same format it was trained on.

    Args:
        raw_trajectory: Dict with keys Xro, Iro, Uro, obj, Tro
        pilot: Pilot instance
        device: torch device

    Returns:
        List of observation dicts suitable for network.get_commander_data extractor
    """
    # Extract raw data
    Xro = raw_trajectory["Xro"]  # (state_dim, N+1)
    Uro = raw_trajectory["Uro"]  # (4, N)
    obj = raw_trajectory.get("obj", np.zeros(18))  # (18,)
    Tro = raw_trajectory["Tro"]  # (N+1,)

    if "Iro" not in raw_trajectory:
        raise KeyError("Iro not in trajectory")

    Iro = raw_trajectory["Iro"]  # Dict with image arrays

    # Get images
    if isinstance(Iro, dict):
        if "semantic" in Iro:
            images_raw = Iro["semantic"]  # (N, H, W, 3) - vision-processor-processed semantic images
        elif "rgb" in Iro:
            images_raw = Iro["rgb"]  # Fallback to raw RGB if semantic not available
        else:
            images_raw = list(Iro.values())[0] if Iro else None
    else:
        images_raw = Iro

    if images_raw is None:
        raise ValueError("No valid images in Iro")

    # Get trajectory length from images
    N = len(images_raw)

    # Verify shapes
    if Xro.shape[1] != N + 1:
        raise ValueError(f"Expected Xro to have {N+1} states, got {Xro.shape[1]}")

    # Flatten obj if needed
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        obj = obj.flatten()
    if len(obj) < 18:
        obj = np.concatenate([obj, np.zeros(18 - len(obj))])

    # Generate observations using OODA (matching observation_generator.py)
    observations = []
    upr = np.zeros(4)  # Previous control
    znn_cr = torch.zeros(pilot.model.Nz).to(device)  # Previous latent state

    for k in range(N):
        # Current state and control
        tcr = Tro[k + 1]  # Time at this step (skip initial state)
        xcr = Xro[:, k + 1]  # State at this step
        ucr = Uro[:, k]  # Control at this step
        img_cr = images_raw[k]  # Image at this step

        # Run OODA cycle (this is what observation_generator.py does)
        _, znn_cr, _, xnn, _ = pilot.OODA(upr, tcr, xcr, obj, img_cr, znn_cr)

        # Create observation dict (format expected by get_commander_data)
        observation = {"tx_com": None, "obj_com": None, "dxu_par": None,
                      "img_vis": None, "tx_vis": None}
        observation["xnn"] = xnn  # This will be processed by get_commander_data
        observation["ynn"] = {"unn": ucr}  # Control for labeling

        observations.append(observation)

        # Update for next iteration
        upr = ucr

    return observations


def extract_commander_inputs_from_observations(observations: List[Dict], pilot, device: torch.device):
    """
    Extract commander inputs from observations using network's built-in extractor.

    This uses the same pipeline as BC training to ensure compatibility.

    Args:
        observations: List of observation dicts from generate_observations_from_trajectory
        pilot: Pilot instance
        device: torch device

    Returns:
        Tuple of (input_tuples, labels) suitable for batching
    """
    extractor = pilot.model.get_commander_data

    inputs_list = []

    for obs in observations:
        # Use the network's built-in extractor (same as BC training)
        inputs, _ = extractor(obs["xnn"], obs["ynn"])
        inputs_list.append(inputs)

    # inputs is a tuple of (tx_com, obj_com, dxu_par, img_vis, tx_vis)
    # We need to batch these same way DataLoader does
    if not inputs_list:
        raise ValueError("No observations to extract")

    # Check structure of first element
    first_inputs = inputs_list[0]
    if not isinstance(first_inputs, tuple):
        raise ValueError(f"Expected tuple from extractor, got {type(first_inputs)}")

    # Stack each component
    tx_com_batch = torch.stack([inp[0] for inp in inputs_list])
    obj_com_batch = torch.stack([inp[1] for inp in inputs_list])
    dxu_par_batch = torch.stack([inp[2] for inp in inputs_list])
    img_vis_batch = torch.stack([inp[3] for inp in inputs_list])
    tx_vis_batch = torch.stack([inp[4] for inp in inputs_list])

    return (tx_com_batch, obj_com_batch, dxu_par_batch, img_vis_batch, tx_vis_batch)


def get_policy_expert_file_paths(cohort_path: str, pilot_name: str, policy_indices: list = None):
    """
    Get file paths for policy and expert trajectories without loading data.

    Useful for on-the-fly loading to minimize memory usage.

    Args:
        cohort_path: Path to cohort directory
        pilot_name: Name of the policy pilot
        policy_indices: Optional list of trajectory indices. If None, returns all pairs.

    Returns:
        tuple of (policy_file_paths, expert_file_paths): Dicts mapping traj_idx to file paths
    """
    # Find latest simulation_data directory
    sim_data_dir = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(sim_data_dir):
        raise FileNotFoundError(f"No simulation_data directory found at {sim_data_dir}")

    # Get most recent timestamp directory
    timestamp_dirs = sorted(glob.glob(os.path.join(sim_data_dir, "*")))
    if not timestamp_dirs:
        raise FileNotFoundError(f"No simulation timestamp directories found in {sim_data_dir}")

    latest_sim_dir = timestamp_dirs[-1]
    trajectories_dir = os.path.join(latest_sim_dir, "trajectories")

    if not os.path.exists(trajectories_dir):
        raise FileNotFoundError(f"No trajectories directory found at {trajectories_dir}")

    # Load all trajectory files
    traj_files = sorted(glob.glob(os.path.join(trajectories_dir, "*.pt")))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {trajectories_dir}")

    # Separate policy and expert trajectories by filename
    policy_trajs = {}
    expert_trajs = {}

    for traj_file in traj_files:
        filename = os.path.basename(traj_file)

        # Skip tester trajectories (files without _traj{idx}_ pattern)
        if "_traj" not in filename:
            continue

        # Parse filename: sim_data_{scene}_{obj}_traj{idx}_{pilot}.pt
        if f"_{pilot_name}.pt" in filename:
            # Extract trajectory key (scene_obj_trajidx)
            traj_key = filename.replace(f"_{pilot_name}.pt", "")
            policy_trajs[traj_key] = traj_file
        elif "_expert.pt" in filename:
            traj_key = filename.replace("_expert.pt", "")
            expert_trajs[traj_key] = traj_file

    # Match policy and expert trajectories
    policy_file_paths = {}
    expert_file_paths = {}

    # If policy_indices provided, convert to a set for O(1) lookup
    target_indices = set(policy_indices) if policy_indices is not None else None

    for traj_idx, traj_key in enumerate(sorted(policy_trajs.keys())):
        # If filtering by indices, skip if not in the target set
        if target_indices is not None and traj_idx not in target_indices:
            continue

        if traj_key in expert_trajs:
            policy_file_paths[traj_idx] = policy_trajs[traj_key]
            expert_file_paths[traj_idx] = expert_trajs[traj_key]

    return policy_file_paths, expert_file_paths


def load_policy_expert_pairs(cohort_path: str, pilot_name: str, policy_indices: list = None):
    """
    Load and match policy-expert trajectory pairs from simulation results.

    Args:
        cohort_path: Path to cohort directory
        pilot_name: Name of the policy pilot
        policy_indices: Optional list of trajectory indices to load. If None, loads all pairs.
                       This is useful for loading only expert data for failed trajectories.

    Returns:
        matched_pairs: List of dicts with 'policy' and 'expert' trajectory data
    """
    # Find latest simulation_data directory
    sim_data_dir = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(sim_data_dir):
        raise FileNotFoundError(f"No simulation_data directory found at {sim_data_dir}")

    # Get most recent timestamp directory
    timestamp_dirs = sorted(glob.glob(os.path.join(sim_data_dir, "*")))
    if not timestamp_dirs:
        raise FileNotFoundError(f"No simulation timestamp directories found in {sim_data_dir}")

    latest_sim_dir = timestamp_dirs[-1]
    trajectories_dir = os.path.join(latest_sim_dir, "trajectories")

    if not os.path.exists(trajectories_dir):
        raise FileNotFoundError(f"No trajectories directory found at {trajectories_dir}")

    # Load all trajectory files
    traj_files = sorted(glob.glob(os.path.join(trajectories_dir, "*.pt")))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {trajectories_dir}")

    # Separate policy and expert trajectories by filename
    policy_trajs = {}
    expert_trajs = {}

    for traj_file in traj_files:
        filename = os.path.basename(traj_file)

        # Skip tester trajectories (files without _traj{idx}_ pattern)
        if "_traj" not in filename:
            continue

        # Parse filename: sim_data_{scene}_{obj}_traj{idx}_{pilot}.pt
        if f"_{pilot_name}.pt" in filename:
            # Extract trajectory key (scene_obj_trajidx)
            traj_key = filename.replace(f"_{pilot_name}.pt", "")
            policy_trajs[traj_key] = traj_file
        elif "_expert.pt" in filename:
            traj_key = filename.replace("_expert.pt", "")
            expert_trajs[traj_key] = traj_file

    # Match policy and expert trajectories
    matched_pairs = []

    # If policy_indices provided, convert to a set for O(1) lookup
    target_indices = set(policy_indices) if policy_indices is not None else None

    for traj_idx, traj_key in enumerate(sorted(policy_trajs.keys())):
        # If filtering by indices, skip if not in the target set
        if target_indices is not None and traj_idx not in target_indices:
            continue

        if traj_key in expert_trajs:
            # Load both
            policy_data_list = torch.load(policy_trajs[traj_key])
            expert_data_list = torch.load(expert_trajs[traj_key])

            # They should have the same structure (list of trajectory dicts)
            for policy_data, expert_data in zip(policy_data_list, expert_data_list):
                matched_pairs.append({
                    'policy': policy_data,
                    'expert': expert_data,
                    'trajectory_key': traj_key,
                    'trajectory_index': traj_idx
                })

    print(f"[DEBUG] Matched {len(matched_pairs)} policy-expert trajectory pairs")
    return matched_pairs


def compute_state_divergence(policy_states, expert_states):
    """
    Compute state divergence between policy and expert trajectories.

    Args:
        policy_states: Policy state trajectory (T, state_dim) or (n_agents, T, state_dim)
        expert_states: Expert state trajectory (same shape)

    Returns:
        divergence: L2 distance between states at each timestep (T,)
    """
    # Handle multi-agent case - take first agent
    if policy_states.ndim == 3:
        policy_states = policy_states[0]  # (T, state_dim)
        expert_states = expert_states[0]

    # Truncate to minimum length (policy and expert may have different trajectory lengths)
    min_len = min(len(policy_states), len(expert_states))
    policy_states = policy_states[:min_len]
    expert_states = expert_states[:min_len]

    # Compute L2 distance for position (first 3 dims typically x, y, z)
    position_policy = policy_states[:, :3]
    position_expert = expert_states[:, :3]

    divergence = np.linalg.norm(position_policy - position_expert, axis=1)
    return divergence


def compute_onset_signals(divergence, values, advantages):
    """
    Compute onset detection signals from divergence and values.

    Finds "early warning signs" where trajectories start to diverge,
    not just where they end up (collision points).

    Args:
        divergence: State divergence at each timestep (T,)
        values: Value estimates at each timestep (T,)
        advantages: GAE advantages at each timestep (T,)

    Returns:
        onset_weights: Continuous weights for BC loss focusing on divergence onset (T,)
    """
    # Ensure all arrays have the same length (handle off-by-one errors)
    min_len = min(len(divergence), len(values), len(advantages))
    if min_len == 0:
        return np.array([])

    divergence = divergence[:min_len]
    values = values[:min_len]
    advantages = advantages[:min_len]

    T = min_len

    # Compute divergence rate (how fast trajectories are separating)
    divergence_rate = np.gradient(divergence)
    divergence_rate = np.maximum(divergence_rate, 0)  # Only positive (separating)

    # Normalize to [0, 1]
    div_rate_max = divergence_rate.max()
    if div_rate_max > 1e-8:
        divergence_rate_norm = divergence_rate / div_rate_max
    else:
        divergence_rate_norm = np.zeros(T)

    # Compute value drops (where value function starts decreasing)
    value_drops = np.diff(values, prepend=values[0])
    value_drops = -value_drops  # Positive = value dropping
    value_drops = np.maximum(value_drops, 0)  # Only drops, not increases

    # Normalize to [0, 1]
    value_drop_max = value_drops.max()
    if value_drop_max > 1e-8:
        value_drops_norm = value_drops / value_drop_max
    else:
        value_drops_norm = np.zeros(T)

    # Onset signal: Combine divergence rate and value drops
    # High where both signals are strong (multiplicative)
    onset_signal = divergence_rate_norm * value_drops_norm

    # Combine with advantage magnitude for final weighting
    # This focuses on: early divergence + dropping value + high-consequence states
    adv_abs = np.abs(advantages)
    adv_abs_max = adv_abs.max()
    if adv_abs_max > 1e-8:
        adv_norm = adv_abs / adv_abs_max
    else:
        adv_norm = np.zeros(T)

    # Final weight: onset signal modulated by advantage
    onset_weights = onset_signal * adv_norm

    return onset_weights


def backward_value_propagation(
    policy_traj: Dict,
    expert_traj: Dict,
    gamma: float = 0.95,
    divergence_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Rigorous backward value propagation to identify critical divergence states.
    
    Unlike heuristic onset signals, this computes exact backward values:
    V(t) = r(t) + γ V(t+1)
    
    This is a rigorous dynamic programming approach that identifies the exact
    timestep where the policy trajectory begins diverging from expert behavior
    in terms of cumulative future returns.
    
    Args:
        policy_traj: Policy trajectory dict with 'Xro' (states) and optionally 'rewards'
        expert_traj: Expert trajectory dict with 'Xro' (states) and optionally 'rewards'
        gamma: Discount factor for value computation
        divergence_threshold: Threshold for "significant" value gap (default 0.5)
        
    Returns:
        V_policy: Backward values for policy trajectory (N,)
        V_expert: Backward values for expert trajectory (M,)
        critical_idx: Index of critical state (first significant divergence), or None
    
    Example:
        >>> V_policy, V_expert, critical_idx = backward_value_propagation(
        ...     policy_traj={'Xro': states_policy, 'rewards': rewards_policy},
        ...     expert_traj={'Xro': states_expert, 'rewards': rewards_expert},
        ...     gamma=0.95
        ... )
        >>> if critical_idx is not None:
        ...     print(f"Critical divergence at timestep {critical_idx}")
        ...     print(f"Value gap: {V_expert[critical_idx] - V_policy[critical_idx]:.3f}")
    """
    # Extract states
    policy_states = policy_traj.get('Xro', None)
    expert_states = expert_traj.get('Xro', None)
    
    if policy_states is None or expert_states is None:
        # Cannot compute values without states
        return np.array([]), np.array([]), None
    
    # Get rewards (use zeros if not available - caller should provide these)
    policy_rewards = policy_traj.get('rewards', np.zeros(len(policy_states)))
    expert_rewards = expert_traj.get('rewards', np.zeros(len(expert_states)))
    
    # Ensure rewards are numpy arrays
    if isinstance(policy_rewards, list):
        policy_rewards = np.array(policy_rewards)
    if isinstance(expert_rewards, list):
        expert_rewards = np.array(expert_rewards)
    
    # Backward value accumulation: V(t) = r(t) + γ V(t+1)
    N = len(policy_rewards)
    M = len(expert_rewards)
    
    V_policy = np.zeros(N)
    V_expert = np.zeros(M)
    
    # Initialize terminal values (last timestep has no future return)
    if N > 0:
        V_policy[-1] = policy_rewards[-1]
    if M > 0:
        V_expert[-1] = expert_rewards[-1]
    
    # Backward pass for policy trajectory
    for t in range(N - 2, -1, -1):
        V_policy[t] = policy_rewards[t] + gamma * V_policy[t + 1]
    
    # Backward pass for expert trajectory
    for t in range(M - 2, -1, -1):
        V_expert[t] = expert_rewards[t] + gamma * V_expert[t + 1]
    
    # Identify critical state: first timestep where value gap becomes significant
    # Align trajectories by taking minimum length
    min_len = min(N, M)
    
    if min_len == 0:
        return V_policy, V_expert, None
    
    value_gap = V_expert[:min_len] - V_policy[:min_len]
    
    # Critical state: where value gap exceeds threshold AND is increasing
    # This indicates the onset of divergence (not just noise)
    critical_idx = None
    
    for t in range(min_len - 1):
        if value_gap[t] > divergence_threshold:
            # Check if gap is growing (divergence onset, not just temporary)
            if t == 0 or value_gap[t] > value_gap[t - 1]:
                critical_idx = t
                break
    
    return V_policy, V_expert, critical_idx
