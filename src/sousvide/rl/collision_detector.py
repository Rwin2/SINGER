"""
RL-focused collision detection and clearance computation for policy fine-tuning.

This module extends the existing trajectory analysis (visualize/analyze_simulated_experiments.py)
with RL-specific functionality:
- Per-timestep clearance distances (continuous reward signal)
- Time-to-collision (TTC) metric
- Integration with RL reward computation
- Efficient KDTree-based spatial queries

Uses analyze_trajectory_performance as the underlying analysis engine.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Optional


class CollisionDetector:
    """
    Detects collisions and computes clearance metrics for drone trajectories.

    Uses efficient KDTree spatial indexing for per-timestep collision checking.
    """

    def __init__(self, point_cloud: np.ndarray, collision_radius: float = 0.3):
        """
        Initialize collision detector with environment point cloud.

        Args:
            point_cloud: Environment geometry (3×N or N×3 format)
            collision_radius: Radius for collision detection (meters)
        """
        # Normalize point cloud to N×3 format
        if point_cloud.shape[0] == 3:
            self.point_cloud = point_cloud.T
        elif point_cloud.shape[1] == 3:
            self.point_cloud = point_cloud
        else:
            raise ValueError(f"Point cloud shape {point_cloud.shape} not recognized. Expected 3×N or N×3")

        self.collision_radius = collision_radius
        self.kdtree = cKDTree(self.point_cloud)

    def compute_clearances(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute minimum clearance (distance to nearest obstacle) at each timestep.

        Args:
            positions: Trajectory positions (N×3 array, one position per row)

        Returns:
            clearances: Minimum distance to obstacles at each timestep (N,)
        """
        clearances = np.zeros(len(positions))

        for i, pos in enumerate(positions):
            # Query single nearest neighbor distance
            distance, _ = self.kdtree.query(pos, k=1)
            clearances[i] = distance

        return clearances

    def detect_collision(self, positions: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        """
        Detect if trajectory collides with obstacles.

        Args:
            positions: Trajectory positions (N×3 array)

        Returns:
            collision_detected: Boolean indicating if collision occurred
            collision_index: Index of first collision timestep (or len(positions) if none)
            clearances: Per-timestep clearance distances
        """
        clearances = self.compute_clearances(positions)

        # Find first collision
        collision_detected = False
        collision_index = len(positions)

        for i, clearance in enumerate(clearances):
            if clearance < self.collision_radius:
                collision_detected = True
                collision_index = i
                break

        return collision_detected, collision_index, clearances

    def analyze_trajectory(self, Xro: np.ndarray) -> Dict:
        """
        Comprehensive RL-focused trajectory analysis.

        Args:
            Xro: State trajectory (10×N or higher, where first 3 rows are x,y,z)

        Returns:
            dict with keys:
                'collision': bool, whether collision occurred
                'collision_index': int, index of first collision (-1 if none)
                'clearances': (N,) array of per-timestep distances to obstacles
                'min_clearance': float, minimum clearance along trajectory
                'time_to_collision': int, steps until collision (-1 if none)
                'clearance_percentile_10': float, 10th percentile clearance
                'clearance_percentile_25': float, 25th percentile clearance
                'mean_clearance': float, average clearance along trajectory
        """
        # Extract positions
        positions = Xro[0:3, :].T  # (N, 3)

        # Detect collisions and compute clearances
        collision_detected, collision_index, clearances = self.detect_collision(positions)

        # Compute per-trajectory metrics
        result = {
            'collision': collision_detected,
            'collision_index': collision_index if collision_detected else -1,
            'clearances': clearances,
            'min_clearance': float(np.min(clearances)),
            'time_to_collision': collision_index if collision_detected else -1,
            'clearance_percentile_10': float(np.percentile(clearances, 10)),
            'clearance_percentile_25': float(np.percentile(clearances, 25)),
            'mean_clearance': float(np.mean(clearances)),
            'trajectory_length': len(positions),
            'collision_radius': self.collision_radius,
        }

        return result


def compute_collision_rewards(clearances: np.ndarray,
                             collision_detected: bool,
                             collision_index: int,
                             collision_penalty: float = 5.0,
                             clearance_weight: float = 1.0,
                             success_reward: float = 10.0,
                             clearance_threshold: float = 0.3) -> Tuple[np.ndarray, float]:
    """
    Compute RL rewards from collision detection results.

    Reward structure:
    - Per-step: Continuous penalty based on proximity to obstacles
    - Terminal: Success/failure bonus

    Args:
        clearances: Per-timestep clearance distances
        collision_detected: Whether collision occurred
        collision_index: Index of collision (or len(clearances) if none)
        collision_penalty: Weight for collision penalty (scales continuous reward)
        clearance_weight: Weight for per-step clearance reward
        success_reward: Terminal reward for collision-free trajectory
        clearance_threshold: Threshold distance for collision (meters)

    Returns:
        per_step_rewards: (N,) array of per-timestep rewards
        terminal_reward: scalar terminal reward
    """
    N = len(clearances)

    # Per-step rewards: negative when close to obstacles
    # r_t = -α * max(0, threshold - clearance_t)
    # This gives 0 reward when clearance >= threshold, increasingly negative as clearance decreases
    per_step_rewards = np.zeros(N)

    for i in range(N):
        if i < collision_index:  # Only before collision
            proximity = max(0.0, clearance_threshold - clearances[i])
            per_step_rewards[i] = -collision_penalty * proximity * clearance_weight

    # Terminal reward
    if collision_detected:
        terminal_reward = -success_reward
    else:
        terminal_reward = success_reward

    return per_step_rewards, terminal_reward


def batch_analyze_trajectories(trajectories: list,
                              collision_detector: CollisionDetector) -> list:
    """
    Analyze multiple trajectories in batch.

    Args:
        trajectories: List of state arrays (Xro format: 10×N or higher)
        collision_detector: Initialized CollisionDetector instance

    Returns:
        List of analysis dictionaries (one per trajectory)
    """
    results = []
    for Xro in trajectories:
        result = collision_detector.analyze_trajectory(Xro)
        results.append(result)

    return results


def compute_aggregate_collision_stats(analysis_results: list) -> Dict:
    """
    Compute summary statistics from batch trajectory analyses.

    Args:
        analysis_results: List of dictionaries from analyze_trajectory()

    Returns:
        dict with aggregate metrics
    """
    if not analysis_results:
        return {}

    collision_rates = [r['collision'] for r in analysis_results]
    min_clearances = [r['min_clearance'] for r in analysis_results]
    mean_clearances = [r['mean_clearance'] for r in analysis_results]
    collision_indices = [r['collision_index'] for r in analysis_results if r['collision']]

    stats = {
        'num_trajectories': len(analysis_results),
        'collision_rate': float(np.mean(collision_rates)),
        'collision_count': int(np.sum(collision_rates)),
        'mean_min_clearance': float(np.mean(min_clearances)),
        'std_min_clearance': float(np.std(min_clearances)),
        'mean_clearance_overall': float(np.mean(mean_clearances)),
        'std_clearance_overall': float(np.std(mean_clearances)),
    }

    if collision_indices:
        stats['mean_collision_index'] = float(np.mean(collision_indices))
        stats['std_collision_index'] = float(np.std(collision_indices))

    return stats
