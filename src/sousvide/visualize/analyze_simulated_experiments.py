import json

import numpy as np
from scipy.spatial import cKDTree

def analyze_trajectory_performance(Xro, goal_location, point_cloud, exclusion_radius, 
                                 collision_radius, trajectory_name="trajectory"):
    """
    Analyze trajectory performance including goal reaching, collisions, and summary statistics.
    
    Parameters:
        Tro: Time array of actual trajectory
        Xro: State array of actual trajectory (11 x N)
             States: [x, y, z, vx, vy, vz, qx, qy, qz, qw, ...]
             where [qx, qy, qz, qw] is the quaternion orientation
        goal_location: 3D coordinates of the goal [x, y, z]
        point_cloud: Point cloud array for collision detection (3 x N or N x 3)
        exclusion_radius: Radius around goal for success detection (r1) - success occurs when within r1 + r2
        collision_radius: Radius for collision detection with point cloud (r2) - also used for soft success zone
        trajectory_name: Name for logging purposes
    
    Returns:
        dict: Analysis results containing terminal point, statistics, etc.
    """
    if Xro.shape[1] == 0:
        return {
            "success": False,
            "collision": True,
            "terminal_reason": "empty_trajectory",
            "terminal_index": 0,
            "terminal_position": np.array([0, 0, 0]),
            "distance_to_goal": float('inf'),
            "normalized_distance": 1.0,
            "yaw_error": float('inf'),
            "trajectory_length": 0
        }
    
    # Extract position data (x, y, z are typically the first 3 states)
    positions = Xro[0:3, :].T  # N x 3 array
    
    # Extract quaternion components (Xro[6:10] = [qx, qy, qz, qw])
    qx, qy, qz, qw = Xro[6:10, :]
    
    # Convert quaternion to yaw using standard formula
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    yaw_angles = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    # --- per-timestep required yaw and yaw-error series (vectorized) ---
    # Vector from each position to goal
    goal_vectors = (goal_location.reshape(1, 3) - positions)  # (N x 3)
    required_world_yaws = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])  # (N,)
    # Shortest-angle yaw error in [0, pi]
    yaw_error_series = np.abs(yaw_angles - required_world_yaws)
    yaw_error_series = np.where(yaw_error_series > np.pi, 2*np.pi - yaw_error_series, yaw_error_series)

    
    # Prepare point cloud for KDTree (ensure it's N x 3)
    if point_cloud.shape[0] == 3:
        # Point cloud is 3 x N, transpose to N x 3
        point_cloud_kdtree_format = point_cloud.T
    elif point_cloud.shape[1] == 3:
        # Point cloud is already N x 3
        point_cloud_kdtree_format = point_cloud
    else:
        raise ValueError(f"Point cloud shape {point_cloud.shape} not recognized. Expected 3xN or Nx3")
    
    # Build KDTree for efficient collision detection
    obstacle_kdtree = cKDTree(point_cloud_kdtree_format)
    
    # Calculate distances to goal at each point
    goal_distances = np.linalg.norm(positions - goal_location, axis=1)
    
    # Check for goal reaching (within exclusion radius + collision radius for soft success)
    # This allows success if drone gets within its own radius of the exclusion zone
    soft_success_radius = exclusion_radius + 2*collision_radius
    goal_reached_mask = goal_distances <= soft_success_radius
    goal_reached_indices = np.where(goal_reached_mask)[0]
    
    # Find first goal reaching event (if any)
    first_goal_index = goal_reached_indices[0] if len(goal_reached_indices) > 0 else len(positions)
    
    # Check for collisions with point cloud using KDTree
    collision_detected = False
    collision_index = len(positions)
    
    # Use the collision radius from configuration (r2)
    collision_threshold = collision_radius
    
    # Check collision at each timestep
    for i in range(len(positions)):
        pos = positions[i]
        # Use KDTree to find points within collision radius
        indices = obstacle_kdtree.query_ball_point(pos, r=collision_threshold)
        if len(indices) > 0:
            collision_detected = True
            collision_index = i
            break
    
    # Determine terminal point and reason based on what happened first
    # If both happen at the same timestep, collision takes priority for safety
    terminal_index = len(positions) - 1  # Default to end of trajectory
    terminal_reason = "trajectory_end"
    success = False
    
    if collision_detected and first_goal_index < len(positions):
        # Both collision and goal detected - check which happens first
        if collision_index < first_goal_index:
            # Collision happened first
            terminal_index = collision_index
            terminal_reason = "collision"
            success = False
        elif collision_index == first_goal_index:
            # Both happen at same time - collision takes priority for safety
            terminal_index = collision_index
            terminal_reason = "collision"
            success = False
        else:
            # Goal reached first
            terminal_index = first_goal_index
            terminal_reason = "goal_reached"
            success = True
    elif collision_detected:
        # Only collision detected
        terminal_index = collision_index
        terminal_reason = "collision"
        success = False
    elif first_goal_index < len(positions):
        # Only goal reached
        terminal_index = first_goal_index
        terminal_reason = "goal_reached"
        success = True
    
    # Get terminal state
    terminal_position = positions[terminal_index]
    terminal_yaw = yaw_angles[terminal_index]
    
    # Calculate final distance to goal
    final_distance = np.linalg.norm(terminal_position - goal_location)
    
    # Calculate normalized distance: (final_distance - success_radius) / (start_distance - success_radius)
    # This shows what fraction of the "remaining distance to goal" was NOT covered
    # Use the same radius threshold as success determination for consistency
    start_position = positions[0]
    start_distance = np.linalg.norm(start_position - goal_location)
    
    # Account for soft success radius in both distances (same as success criteria)
    effective_final_distance = max(0.0, final_distance - soft_success_radius)
    effective_start_distance = max(0.0, start_distance - soft_success_radius)
    
    # Normalized distance: 0.0 = reached goal, 1.0 = no progress made
    if effective_start_distance > 1e-6:  # Avoid division by zero
        normalized_distance = effective_final_distance / effective_start_distance
    else:
        # If we start within exclusion radius, consider it perfect (0.0)
        normalized_distance = 0.0 if effective_final_distance < 1e-6 else 1.0

    # Debug check for abnormally high normalized distances
    if normalized_distance > 15:
        print(f"WARNING: Abnormally high normalized distance detected for {trajectory_name}")
        print(f"  Normalized distance: {normalized_distance:.3f}")
        print(f"  Start position: {start_position}")
        print(f"  Start distance to goal: {start_distance:.3f}m")
        print(f"  Terminal position: {terminal_position}")
        print(f"  Final distance to goal: {final_distance:.3f}m")
        print(f"  Goal location: {goal_location}")
        print(f"  Exclusion radius (r1): {exclusion_radius:.3f}m")
        print(f"  Collision radius (r2): {collision_radius:.3f}m")
        print(f"  Soft success radius (r1+2*r2): {soft_success_radius:.3f}m")
        print(f"  Effective start distance: {effective_start_distance:.3f}m")
        print(f"  Effective final distance: {effective_final_distance:.3f}m")
        print(f"  Terminal reason: {terminal_reason}")
        print(f"  Success: {success}")
        print(f"  Trajectory length: {len(positions)} points")
        print(f"  Calculation: {effective_final_distance:.3f} / {effective_start_distance:.3f} = {normalized_distance:.3f}")
        if effective_start_distance < 1e-6:
            print(f"  NOTE: Very small start distance may cause numerical issues")
    
    # Calculate camera field-of-view based yaw/heading analysis
    horizontal_fov = np.radians(85)  # 85° horizontal FOV

    # Per-timestep FOV membership from the series computed earlier
    goal_in_camera_fov_series = yaw_error_series <= (horizontal_fov / 2)

    # (Optional but nice): if we're effectively at the goal, force zero error + in-FOV.
    # close_mask = goal_distances <= 1e-6
    # yaw_error_series[close_mask] = 0.0
    # goal_in_camera_fov_series[close_mask] = True

    # Terminal yaw/error & FOV derived consistently from series
    terminal_yaw = yaw_angles[terminal_index]
    yaw_error = float(yaw_error_series[terminal_index])
    goal_in_camera_fov = bool(goal_in_camera_fov_series[terminal_index])

    # # Calculate camera field-of-view based yaw/heading analysis
    # # Camera specifications
    # horizontal_fov = np.radians(85)  # 85° horizontal FOV
    
    # goal_direction = goal_location - terminal_position
    # goal_direction_magnitude = np.linalg.norm(goal_direction)
    
    # # Initialize FOV-based metrics
    # goal_in_camera_fov = False
    # yaw_error = 0.0
    
    # # Only calculate FOV analysis if we're not already at the goal
    # if goal_direction_magnitude > 1e-6:  # Small threshold to avoid division by zero
    #     # Ensure goal_direction is a 1D array (flatten if needed)
    #     goal_direction_flat = goal_direction.flatten()
        
    #     # Ensure we have at least 2D coordinates for horizontal plane calculation
    #     if len(goal_direction_flat) < 2:
    #         # Edge case: goal direction has insufficient dimensions
    #         print(f"Warning: Goal direction has insufficient dimensions: {goal_direction.shape}")
    #         print(f"Goal location: {goal_location}")
    #         print(f"Terminal position: {terminal_position}")
    #         goal_in_camera_fov = False
    #         yaw_error = np.pi  # Maximum error
    #     else:
    #         # SIMPLIFIED APPROACH: Assume quaternions are already in world frame
    #         # Calculate the required world yaw to point toward goal from terminal position
    #         required_world_yaw = np.arctan2(goal_direction_flat[1], goal_direction_flat[0])
            
    #         # If quaternions are in world frame, terminal_yaw should be the actual world yaw
    #         actual_world_yaw = terminal_yaw
            
    #         # Calculate yaw error as the shortest angular distance between actual and required yaw
    #         yaw_error = abs(actual_world_yaw - required_world_yaw)
            
    #         # Normalize yaw error to [0, π] (shortest angular distance)
    #         if yaw_error > np.pi:
    #             yaw_error = 2 * np.pi - yaw_error
            
    #         # Check if goal is within horizontal FOV (half-angle on each side)
    #         goal_in_camera_fov = yaw_error <= (horizontal_fov / 2)
    # else:
    #     # If we're at the goal, consider it in FOV
    #     goal_in_camera_fov = True
    #     yaw_error = 0.0
    
    # Calculate trajectory length
    trajectory_length = 0
    for i in range(1, len(positions)):
        trajectory_length += np.linalg.norm(positions[i] - positions[i-1])
    
    results = {
        "success": success,
        "collision": (terminal_reason == "collision"),
        "terminal_reason": terminal_reason,
        "terminal_index": terminal_index,
        "terminal_position": terminal_position,
        "terminal_yaw": terminal_yaw,
        "distance_to_goal": final_distance,
        "normalized_distance": normalized_distance,
        "yaw_error": yaw_error,
        "yaw_error_degrees": np.degrees(yaw_error),
        "goal_in_camera_fov": goal_in_camera_fov,
        "yaw_error_series": yaw_error_series,
        "yaw_error_degrees_series": np.degrees(yaw_error_series),
        "goal_in_camera_fov_series": goal_in_camera_fov_series,
        "yaw_error_mean_upto_terminal": float(np.mean(yaw_error_series[:terminal_index+1])),
        "yaw_error_max_upto_terminal": float(np.max(yaw_error_series[:terminal_index+1])),
        "trajectory_length": trajectory_length,
        "goal_location": goal_location,
        "exclusion_radius": exclusion_radius,
        "collision_radius": collision_radius,
        "soft_success_radius": soft_success_radius,  # Effective success zone used for both success and normalized distance
        "trajectory_name": trajectory_name
    }
    
    return results


def compute_aggregate_statistics(individual_results):
    """
    Compute aggregate statistics from individual trajectory analyses.
    
    Parameters:
        individual_results: List of dictionaries from analyze_trajectory_performance
    
    Returns:
        dict: Aggregate statistics
    """
    if not individual_results:
        return {}
    
    # Extract relevant metrics
    successes            = [r["success"] for r in individual_results]
    collisions           = [r["collision"] for r in individual_results]
    distances            = [r["distance_to_goal"] for r in individual_results]
    normalized_distances = [r["normalized_distance"] for r in individual_results]
    yaw_errors           = [r["yaw_error"] for r in individual_results]
    yaw_errors_deg       = [r["yaw_error_degrees"] for r in individual_results]
    trajectory_lengths   = [r["trajectory_length"] for r in individual_results]
    
    # Extract FOV-based metrics
    goal_in_camera_fov = [r["goal_in_camera_fov"] for r in individual_results]
    
    # Compute aggregate statistics
    aggregate_stats = {
        "total_trajectories": len(individual_results),
        "success_rate": np.mean(successes),
        "collision_rate": np.mean(collisions),
        "completion_rate": 1.0 - np.mean(collisions),  # Trajectories that didn't collide
        
        # Distance statistics
        "mean_distance_to_goal": np.mean(distances),
        "std_distance_to_goal": np.std(distances),
        "median_distance_to_goal": np.median(distances),
        "min_distance_to_goal": np.min(distances),
        "max_distance_to_goal": np.max(distances),
        
        # Normalized distance statistics
        "mean_normalized_distance": np.mean(normalized_distances),
        "std_normalized_distance": np.std(normalized_distances),
        "median_normalized_distance": np.median(normalized_distances),
        
        # Yaw error statistics
        "mean_yaw_error": np.mean(yaw_errors),
        "std_yaw_error": np.std(yaw_errors),
        "median_yaw_error": np.median(yaw_errors),
        "mean_yaw_error_degrees": np.mean(yaw_errors_deg),
        "std_yaw_error_degrees": np.std(yaw_errors_deg),
        "median_yaw_error_degrees": np.median(yaw_errors_deg),
        
        # Camera field-of-view statistics
        "camera_fov_success_rate": np.mean(goal_in_camera_fov),
        
        # Trajectory length statistics
        "mean_trajectory_length": np.mean(trajectory_lengths),
        "std_trajectory_length": np.std(trajectory_lengths),
        "median_trajectory_length": np.median(trajectory_lengths),
        
        # Terminal reasons breakdown
        "terminal_reasons": {}
    }
    
    # Count terminal reasons
    terminal_reasons = [r["terminal_reason"] for r in individual_results]
    for reason in set(terminal_reasons):
        aggregate_stats["terminal_reasons"][reason] = terminal_reasons.count(reason)
    
    # Success-specific statistics (only for successful trajectories)
    successful_results = [r for r in individual_results if r["success"]]
    if successful_results:
        successful_distances = [r["distance_to_goal"] for r in successful_results]
        successful_yaw_errors = [r["yaw_error_degrees"] for r in successful_results]
        successful_camera_fov = [r["goal_in_camera_fov"] for r in successful_results]
        
        aggregate_stats["successful_trajectories"] = {
            "count": len(successful_results),
            "mean_distance_to_goal": np.mean(successful_distances),
            "std_distance_to_goal": np.std(successful_distances),
            "mean_yaw_error_degrees": np.mean(successful_yaw_errors),
            "std_yaw_error_degrees": np.std(successful_yaw_errors),
            "camera_fov_success_rate": np.mean(successful_camera_fov)
        }
    
    return aggregate_stats


def load_trajectory_analysis(analysis_file_path):
    """
    Load and display trajectory analysis results from a saved JSON file.
    
    Parameters:
        analysis_file_path: Path to the trajectory analysis JSON file
    
    Returns:
        dict: Loaded analysis data
    """
    with open(analysis_file_path, 'r') as f:
        analysis_data = json.load(f)
    
    print("TRAJECTORY ANALYSIS RESULTS")
    print("="*50)
    print(f"Scene: {analysis_data['scene_name']}")
    print(f"Objective: {analysis_data['objective']}")
    print(f"Trajectories tested: {analysis_data['num_trajectories']}")
    print(f"Pilots: {', '.join(analysis_data['pilots_tested'])}")
    print(f"Goal location: {analysis_data['goal_location']}")
    print(f"Exclusion radius (r1): {analysis_data['exclusion_radius']:.2f}m")
    if 'collision_radius' in analysis_data:
        print(f"Collision radius (r2): {analysis_data['collision_radius']:.2f}m")
        # Calculate and display soft success radius if both radii are available
        soft_radius = analysis_data['exclusion_radius'] + analysis_data['collision_radius']
        print(f"Soft success radius (r1+r2): {soft_radius:.2f}m")
    
    print(f"\nPILOT PERFORMANCE COMPARISON:")
    for pilot_name, pilot_stats in analysis_data['pilot_specific_statistics'].items():
        print(f"  {pilot_name}:")
        print(f"    Total simulations: {pilot_stats['total_trajectories']}")
        print(f"    Success rate: {pilot_stats['success_rate']:.1%}")
        print(f"    Collision rate: {pilot_stats['collision_rate']:.1%}")
        print(f"    Mean distance to goal: {pilot_stats['mean_distance_to_goal']:.2f} ± {pilot_stats['std_distance_to_goal']:.2f}m")
        print(f"    Mean normalized distance: {pilot_stats['mean_normalized_distance']:.3f} ± {pilot_stats['std_normalized_distance']:.3f}")
        print(f"    Mean yaw error: {pilot_stats['mean_yaw_error_degrees']:.1f} ± {pilot_stats['std_yaw_error_degrees']:.1f}°")
        if 'camera_fov_success_rate' in pilot_stats:
            print(f"    Camera FOV success rate: {pilot_stats['camera_fov_success_rate']:.1%}")
    
    return analysis_data