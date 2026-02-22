"""
Reinforcement Learning module for policy fine-tuning.

This module provides tools for improving behavior-cloned policies through RL,
with a focus on collision avoidance.
"""

from .collision_detector import (
    CollisionDetector,
    compute_collision_rewards,
    batch_analyze_trajectories,
    compute_aggregate_collision_stats
)

from .rl_helpers import (
    move_simulation_results,
    filter_validation_trajectories,
    load_simulation_results,
    compute_advantages_gae,
    compute_advantages_mc,
    compute_mc_rewards,
    compute_simple_advantage,
    compute_kl_divergence_continuous,
    select_high_risk_states,
    prepare_batch_data,
    compute_policy_gradient_loss,
    compute_bc_loss,
    compute_kl_loss,
    normalize_advantages,
    prepare_state_batch,
    extract_trajectory_data_for_critic,
    generate_observations_from_trajectory,
    extract_commander_inputs_from_observations,
    load_policy_expert_pairs,
    get_policy_expert_file_paths,
    compute_state_divergence,
    compute_onset_signals,
    backward_value_propagation
)

__all__ = [
    # Collision detection
    'CollisionDetector',
    'compute_collision_rewards',
    'batch_analyze_trajectories',
    'compute_aggregate_collision_stats',
    # Simulation data management
    'move_simulation_results',
    'filter_validation_trajectories',
    'load_simulation_results',
    # RL helpers
    'compute_advantages_gae',
    'compute_advantages_mc',
    'compute_mc_rewards',
    'compute_simple_advantage',
    'compute_kl_divergence_continuous',
    'select_high_risk_states',
    'prepare_batch_data',
    'compute_policy_gradient_loss',
    'compute_bc_loss',
    'compute_kl_loss',
    'normalize_advantages',
    'prepare_state_batch',
    'extract_trajectory_data_for_critic',
    'generate_observations_from_trajectory',
    'extract_commander_inputs_from_observations',
    # Divergence-based BC
    'load_policy_expert_pairs',
    'get_policy_expert_file_paths',
    'compute_state_divergence',
    'compute_onset_signals',
    'backward_value_propagation'
]
