import os
import json
import yaml
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torchvision.io import write_video
from torchvision.transforms import Resize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from figs.simulator import Simulator
import figs.utilities.trajectory_helper as th
import figs.tsampling.build_rrt_dataset as bd
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.dynamics.model_specifications import generate_specifications
import sousvide.synthesize.rollout_generator as gd
import sousvide.visualize.record_flight as rf
import sousvide.flight.vision_preprocess_alternate as vp
from sousvide.control.pilot import Pilot

from sousvide.visualize.plot_flight import quad_frame
from sousvide.visualize.analyze_simulated_experiments import analyze_trajectory_performance, compute_aggregate_statistics, load_trajectory_analysis

def visualize_filtered_trajectories(cohort_name: str, method_name: str, 
                                   flights: List[Tuple[str, str]], 
                                   obj_name: str,
                                   max_trajectories: int = None,
                                   simulate: bool = False,
                                   roster: List[str] = None,
                                   review: bool = False,
                                   force_reanalysis: bool = False):
    """
    Load filtered trajectories and visualize multiple trajectories in a single figure.
    Optionally simulate each trajectory or review previously saved simulation results.
    
    Parameters:
        cohort_name: Name of the cohort
        method_name: Name of the method configuration
        flights: List of (scene_name, course_name) tuples
        obj_name: Name of the objective to visualize
        max_trajectories: Maximum number of trajectories to visualize (None for all)
        simulate: Whether to simulate the trajectories
        roster: List of pilot names to use for simulation (if simulate=True)
        review: Whether to analyze previously saved simulation data (simulate=False)
        force_reanalysis: Whether to force regeneration of analysis even if existing file found
    """
    
    # Get workspace path
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Set default roster if not provided (needed for both simulate and review)
    if roster is None:
        roster = []
    
    # Extract method configs
    method_path = os.path.join(workspace_path, "configs", "method", method_name + ".json")
    scenes_cfg_dir = os.path.join(workspace_path, "configs", "scenes")
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)
    
    with open(method_path) as json_file:
        method_config = json.load(json_file)
    
    sample_set_config = method_config["sample_set"]
    rollout_name = sample_set_config["rollout"]
    
    # If simulating, load additional configurations
    if simulate:
        test_set_config = method_config["test_set"]
        trajectory_set_config = method_config["trajectory_set"]
        frame_set_config = method_config["frame_set"]
        
        base_policy_name = sample_set_config["policy"]
        base_frame_name = sample_set_config["frame"]
        use_clip = sample_set_config["clipseg"]
        Nrep = test_set_config["reps"]
        
        # Load perception config for vision processor
        perception_cfg_dir = os.path.join(workspace_path, "configs", "perception")
        with open(os.path.join(perception_cfg_dir, "onnx_benchmark_config.json")) as json_file:
            perception_config = json.load(json_file)
        onnx_model_path = perception_config.get("onnx_model_path", None)
        
        # Load policy and frame configs
        policy_path = os.path.join(workspace_path, "configs", "policy", base_policy_name + ".json")
        frame_path = os.path.join(workspace_path, "configs", "frame", base_frame_name + ".json")
        
        with open(policy_path) as json_file:
            policy_config = json.load(json_file)
        with open(frame_path) as json_file:
            base_frame_config = json.load(json_file)
        
        # Set up vision processor
        if use_clip:
            print("Image Processing Model set to CLIPSeg.")
            if onnx_model_path is not None:
                print("Using ONNX model for CLIPSeg.")
                vision_processor = vp.CLIPSegHFModel(
                    hf_model="CIDAS/clipseg-rd64-refined",
                    onnx_model_path=onnx_model_path
                )
            else:
                print("Using HuggingFace model for CLIPSeg.")
                vision_processor = vp.CLIPSegHFModel(
                    hf_model="CIDAS/clipseg-rd64-refined"
                )
        else:
            vision_processor = None
        
        # Initialize video transform
        transform = Resize((720, 1280), antialias=True)
        
        # Generate base drone specifications
        base_frame_specs = generate_specifications(base_frame_config)
        
        # Compute simulation variables
        Trep = np.zeros(Nrep)
    
    # Find the most recent simulation data directory
    simulation_data_base = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(simulation_data_base):
        raise ValueError(f"No simulation data found in {simulation_data_base}")
    
    # Get the most recent timestamped directory
    timestamped_dirs = [d for d in os.listdir(simulation_data_base) 
                       if os.path.isdir(os.path.join(simulation_data_base, d))]
    if not timestamped_dirs:
        raise ValueError(f"No timestamped directories found in {simulation_data_base}")
    
    latest_dir = max(timestamped_dirs)
    simulation_data_dir = os.path.join(simulation_data_base, latest_dir)
    print(f"Loading simulation data from: {simulation_data_dir}")
    
    # Create organized subdirectories for data products if simulating
    if simulate:
        trajectories_dir = os.path.join(simulation_data_dir, "trajectories")
        videos_dir = os.path.join(simulation_data_dir, "videos")
        rrt_data_dir = os.path.join(simulation_data_dir, "rrt_planning")
        visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
        
        for directory in [trajectories_dir, videos_dir, rrt_data_dir, visualizations_dir]:
            os.makedirs(directory, exist_ok=True)
    
    for scene_name, course_name in flights:
        # Load scene configuration
        scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)
        
        objectives = scene_cfg["queries"]
        radii = scene_cfg["radii"]
        altitudes = scene_cfg["altitudes"]
        similarities = scene_cfg.get("similarities", None)
        
        if obj_name not in objectives:
            print(f"Objective '{obj_name}' not found in scene '{scene_name}'. Available: {objectives}")
            continue
        
        obj_index = objectives.index(obj_name)
        
        # Generate simulator to get environment data
        simulator = Simulator(scene_name, rollout_name)
        
        # Get objectives and environment data (same as in deploy_ssv.py)
        obj_targets, _, epcds_list, epcds_arr = bd.get_objectives(
            simulator.gsplat, objectives, similarities, False
        )
        
        # Load filtered trajectories - check both new and old structure
        rrt_data_dir = os.path.join(simulation_data_dir, "rrt_planning")
        if os.path.exists(rrt_data_dir):
            filtered_file = os.path.join(rrt_data_dir, f"{scene_name}_filtered_{obj_name}.pkl")
        else:
            raise ValueError(f"RRT planning directory not found: {rrt_data_dir}")
            
        if not os.path.exists(filtered_file):
            print(f"Filtered trajectories file not found: {filtered_file}")
            return
        
        with open(filtered_file, "rb") as f:
            filtered_trajectories = pickle.load(f)
        
        print(f"Loaded {len(filtered_trajectories)} filtered trajectories for {obj_name}")
        
        # Limit trajectories if specified
        if max_trajectories and len(filtered_trajectories) > max_trajectories:
            filtered_trajectories = filtered_trajectories[:max_trajectories]
            print(f"Limiting to {max_trajectories} trajectories")
        
        # Process RRT objectives to get centroids
        env_bounds = {}
        if "minbound" in scene_cfg and "maxbound" in scene_cfg:
            env_bounds["minbound"] = np.array(scene_cfg["minbound"])
            env_bounds["maxbound"] = np.array(scene_cfg["maxbound"])
        
        goal_poses, obj_centroids = th.process_RRT_objectives(
            obj_targets, epcds_arr, env_bounds, radii, altitudes
        )
        
        # Parameterize each trajectory and collect debug info
        all_debug_info = []
        all_trajectories = []  # Store the actual tXUi trajectories for simulation
        for idx, trajectory in enumerate(filtered_trajectories):
            print(f"Parameterizing trajectory {idx + 1}/{len(filtered_trajectories)}")
            
            # Parameterize this specific trajectory using its index
            traj_list, node_list, debug_info = th.parameterize_RRT_trajectories(
                [trajectory], obj_centroids[obj_index], 1.0, 20, randint=0
            )
            
            if debug_info is not None and len(traj_list) > 0:
                # Add trajectory index for identification
                debug_info['trajectory_index'] = idx
                all_debug_info.append(debug_info)
                
                # Store the actual parameterized trajectory for simulation
                all_trajectories.append(traj_list[0])  # traj_list[0] is the parameterized trajectory
        
        print(f"Successfully parameterized {len(all_debug_info)} trajectories")
        
        # Simulate trajectories if requested
        if simulate:
            print(f"Simulating {len(all_debug_info)} trajectories...")
            
            # Generate drone instances
            Frames = gd.generate_frames(Trep, base_frame_config, frame_set_config)
            
            # Get goal location and point cloud for post-processing
            goal_location = obj_targets[obj_index]    # 3D coordinates of the goal
            point_cloud = epcds_arr                   # Point cloud for collision detection
            exclusion_radius = radii[obj_index][0]    # r1: Inner radius for success detection
            # collision_radius = radii[obj_index][1]  # r2: Collision check radius
            collision_radius = 0.15
            
            # Store all trajectory analysis results
            all_trajectory_analyses = []
            
            # Simulate each parameterized trajectory
            for debug_idx, (debug_info, tXUi) in enumerate(zip(all_debug_info, all_trajectories)):
                print(f"Simulating trajectory {debug_idx + 1}/{len(all_debug_info)}")
                
                # Get trajectory data - tXUi is the parameterized trajectory 
                
                # Create trajectory dataset entry similar to deploy_ssv.py
                trajectory_data = {
                    "tXUi": tXUi,
                    **debug_info
                }
                
                # Simulation parameters
                t0, tf = tXUi[0, 0], tXUi[0, -1]
                x0 = tXUi[1:11, 0]
                
                # Pilot list (expert + roster)
                pilot_list = ["expert"] + roster
                
                # Apply perturbations
                Perturbations = gd.generate_perturbations(
                    Tsps=Trep,
                    tXUi=tXUi,
                    trajectory_set_config=test_set_config
                )
                
                # Store analysis results for this trajectory across all pilots
                trajectory_pilot_analyses = {}
                
                # Simulate with each pilot
                for pilot_name in pilot_list:
                    print(f"  Simulating with pilot '{pilot_name}'")
                    
                    # File paths for saving simulation results - use organized subdirectories
                    traj_file = os.path.join(
                        trajectories_dir, 
                        f"sim_data_{scene_name}_{obj_name}_traj{debug_idx}_{pilot_name}.pt"
                    )
                    vid_file = os.path.join(
                        videos_dir, 
                        f"sim_video_{scene_name}_{obj_name}_traj{debug_idx}_{pilot_name}.mp4"
                    )
                    
                    # Set up policy
                    if pilot_name == "expert":
                        policy = VehicleRateMPC(tXUi, base_policy_name, base_frame_name, pilot_name)
                    else:
                        policy = Pilot(cohort_name, pilot_name)
                        policy.set_mode('deploy')
                    
                    results = []
                    pilot_analyses = []  # Store analysis for each perturbation
                    
                    for idx, (frame, perturbation) in enumerate(zip(Frames, Perturbations)):
                        simulator.load_frame(frame)
                        
                        # Simulate trajectory
                        Tro, Xro, Uro, Iro, Tsol, Adv = simulator.simulate(
                            policy, perturbation["t0"], tXUi[0, -1], perturbation["x0"], 
                            np.zeros((18, 1)), query=obj_name, clipseg=vision_processor, verbose=False
                        )
                        
                        # Analyze trajectory performance
                        trajectory_analysis = analyze_trajectory_performance(
                            Xro, goal_location, point_cloud, exclusion_radius, 
                            collision_radius, trajectory_name=f"traj{debug_idx}_{pilot_name}_rep{idx}"
                        )
                        pilot_analyses.append(trajectory_analysis)
                        
                        # Save trajectory result
                        trajectory = {
                            "Tro": Tro, "Xro": Xro, "Uro": Uro,
                            "tXUd": tXUi, "obj": np.zeros((18, 1)), "Ndata": Uro.shape[1], 
                            "Tsol": Tsol, "Adv": Adv,
                            "rollout_id": method_name + "_" + str(idx).zfill(5),
                            "course": course_name,
                            "frame": frame,
                            "goal_location": goal_location.tolist(),  # Add goal location at top level for easy access
                            "exclusion_radius": exclusion_radius,    # Add radii for reference
                            "collision_radius": collision_radius,
                            "analysis": trajectory_analysis  # Add analysis to trajectory data
                        }
                        results.append(trajectory)
                    
                    # Store pilot-specific analyses
                    trajectory_pilot_analyses[pilot_name] = pilot_analyses
                    
                    # Save simulation results
                    torch.save(results, traj_file)
                    
                    # Process and save videos
                    if vision_processor is not None:
                        imgs = {
                            "semantic": Iro["semantic"],
                            "rgb": Iro["rgb"]
                        }
                    else:
                        imgs = {
                            "semantic": Iro["semantic"]
                        }
                    
                    for key in imgs:
                        # Prepare and write video
                        frames = torch.zeros((imgs[key].shape[0], 720, 1280, 3))
                        imgs_t = torch.from_numpy(imgs[key])
                        for i in range(imgs_t.shape[0] - 1):
                            img = imgs_t[i].permute(2, 0, 1)
                            img = transform(img)
                            frames[i] = img.permute(1, 2, 0)
                        
                        # Save video with key in filename
                        key_vid_file = vid_file.replace('.mp4', f'_{key}.mp4')
                        write_video(key_vid_file, frames, fps=20)
                    
                    print(f"    Saved simulation results for pilot '{pilot_name}'")
                
                # Store this trajectory's analysis across all pilots
                all_trajectory_analyses.append({
                    "trajectory_index": debug_idx,
                    "pilot_analyses": trajectory_pilot_analyses
                })
            
            # Compute aggregate statistics across all trajectories and pilots
            print("Computing aggregate trajectory statistics...")
            
            # Group analyses by pilot for individual pilot statistics
            pilot_specific_analyses = {}
            
            for traj_analysis in all_trajectory_analyses:
                for pilot_name, pilot_analyses in traj_analysis["pilot_analyses"].items():
                    if pilot_name not in pilot_specific_analyses:
                        pilot_specific_analyses[pilot_name] = []
                    pilot_specific_analyses[pilot_name].extend(pilot_analyses)
            
            # Compute pilot-specific aggregate statistics
            pilot_aggregates = {}
            for pilot_name, pilot_analyses in pilot_specific_analyses.items():
                pilot_aggregates[pilot_name] = compute_aggregate_statistics(pilot_analyses)
            
            # Create comprehensive analysis summary
            analysis_summary = {
                "scene_name": scene_name,
                "objective": obj_name,
                "goal_location": goal_location.tolist(),
                "exclusion_radius": exclusion_radius,
                "collision_radius": collision_radius,
                "point_cloud_size": len(point_cloud),
                "num_trajectories": len(all_debug_info),
                "pilots_tested": pilot_list,
                "timestamp": simulation_data_dir.split('/')[-1],
                "pilot_specific_statistics": pilot_aggregates,
                "all_trajectory_analyses": all_trajectory_analyses
            }
            
            # Save analysis summary to visualizations directory
            analysis_file = os.path.join(
                visualizations_dir, 
                f"trajectory_analysis_{scene_name}_{obj_name}.json"
            )
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_for_json(item) for item in obj]
                else:
                    return obj
            
            analysis_summary_json = convert_numpy_for_json(analysis_summary)
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_summary_json, f, indent=2)
            
            print(f"Saved trajectory analysis to: {analysis_file}")
            
            # Print summary statistics
            print("\n" + "="*70)
            print("TRAJECTORY ANALYSIS SUMMARY")
            print("="*70)
            print(f"Scene: {scene_name}, Objective: {obj_name}")
            print(f"Goal exclusion radius (r1): {exclusion_radius:.2f}m")
            print(f"Collision check radius (r2): {collision_radius:.2f}m")
            print(f"Total trajectories tested: {len(all_debug_info)}")
            print(f"Pilots tested: {', '.join(pilot_list)}")
            
            print("\nPilot performance comparison:")
            for pilot_name, pilot_stats in pilot_aggregates.items():
                print(f"  {pilot_name}:")
                print(f"    Total simulations: {pilot_stats['total_trajectories']}")
                print(f"    Success rate: {pilot_stats['success_rate']:.1%}")
                print(f"    Collision rate: {pilot_stats['collision_rate']:.1%}")
                print(f"    Mean distance to goal: {pilot_stats['mean_distance_to_goal']:.2f}m")
                print(f"    Mean normalized distance: {pilot_stats['mean_normalized_distance']:.3f}")
                print(f"    Mean yaw error: {pilot_stats['mean_yaw_error_degrees']:.1f}°")
                print(f"    Camera FOV success rate: {pilot_stats['camera_fov_success_rate']:.1%}")
            
            print("="*70)
            
            print("Simulation completed!")
        
        # Review previously saved simulation data if requested
        elif review:
            print(f"Reviewing saved simulation data for {len(all_debug_info)} trajectories...")
            
            # Look for existing analysis file (check visualizations folder first, then main directory for backward compatibility)
            visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
            analysis_file = os.path.join(
                visualizations_dir, 
                f"trajectory_analysis_{scene_name}_{obj_name}.json"
            )
            
            # Fallback to old location if not found in visualizations directory
            if not os.path.exists(analysis_file):
                analysis_file = os.path.join(
                    visualizations_dir,
                    f"trajectory_analysis_{scene_name}_{obj_name}.json"
                )
            
            if os.path.exists(analysis_file) and not force_reanalysis:
                print(f"Found existing analysis file: {analysis_file}")
                analysis_data = load_trajectory_analysis(analysis_file)
                
            else:
                if force_reanalysis and os.path.exists(analysis_file):
                    print(f"Forcing reanalysis despite existing file: {analysis_file}")
                else:
                    print(f"No existing analysis file found at: {analysis_file}")
                print("Looking for individual simulation files to reconstruct analysis...")
                
                # Get goal location and point cloud for analysis
                goal_location = obj_targets[obj_index]  # 3D coordinates of the goal
                point_cloud = epcds_arr  # Point cloud for collision detection
                exclusion_radius = radii[obj_index][0]  # r1: Inner radius for success detection
                collision_radius = 0.15
                
                # Look for simulation files in the trajectories subdirectory
                trajectories_dir = os.path.join(simulation_data_dir, "trajectories")
                
                # Check if organized structure exists, fallback to old structure
                if os.path.exists(trajectories_dir):
                    search_dir = trajectories_dir
                    print(f"Using new organized structure: {search_dir}")
                else:
                    raise ValueError(f"Trajectories directory not found: {trajectories_dir}")
                
                sim_file_pattern = f"sim_data_{scene_name}_{obj_name}_traj*_*.pt"
                sim_files = []
                for file_name in os.listdir(search_dir):
                    if file_name.startswith(f"sim_data_{scene_name}_{obj_name}_traj") and file_name.endswith(".pt"):
                        sim_files.append(file_name)
                
                if not sim_files:
                    print(f"No simulation files found matching pattern: {sim_file_pattern}")
                    print("Available files in directory:")
                    for file_name in os.listdir(search_dir):
                        if file_name.endswith(".pt"):
                            print(f"  {file_name}")
                    print("Cannot perform review without simulation data.")
                    continue
                
                print(f"Found {len(sim_files)} simulation files to analyze")
                
                # Parse simulation files to extract trajectory and pilot information
                trajectory_pilot_data = {}  # {traj_idx: {pilot_name: [rep_files]}}
                
                for sim_file in sim_files:
                    # Parse filename: sim_data_{scene}_{obj}_traj{idx}_{pilot}_{rep}.pt
                    # or sim_data_{scene}_{obj}_traj{idx}_{pilot}.pt (for single rep)
                    base_name = sim_file.replace(f"sim_data_{scene_name}_{obj_name}_", "").replace(".pt", "")
                    
                    if base_name.startswith("traj"):
                        # Extract trajectory index and pilot name
                        parts = base_name.split("_")
                        traj_part = parts[0]  # "trajX"
                        traj_idx = int(traj_part.replace("traj", ""))
                        
                        # Respect max_trajectories limit - skip trajectories beyond the limit
                        if max_trajectories is not None and traj_idx >= max_trajectories:
                            continue
                        
                        # Rest is pilot name (might include underscores)
                        pilot_name = "_".join(parts[1:])
                        
                        # Handle repetition suffix if present
                        if pilot_name.endswith("_0") or pilot_name.endswith("_1") or pilot_name.endswith("_2"):
                            # Remove repetition suffix
                            pilot_name = "_".join(pilot_name.split("_")[:-1])
                        
                        if traj_idx not in trajectory_pilot_data:
                            trajectory_pilot_data[traj_idx] = {}
                        
                        if pilot_name not in trajectory_pilot_data[traj_idx]:
                            trajectory_pilot_data[traj_idx][pilot_name] = []
                        
                        trajectory_pilot_data[traj_idx][pilot_name].append(sim_file)
                
                # Analyze each trajectory and pilot combination
                all_trajectory_analyses = []
                
                for traj_idx in sorted(trajectory_pilot_data.keys()):
                    print(f"Analyzing trajectory {traj_idx}")

                    # trajectory_pilot_analyses = {}

                    # for pilot_name, sim_file_list in trajectory_pilot_data[traj_idx].items():
                    #     sim_data = torch.load(os.path.join(search_dir, sim_file_list[0]))
                    #     num_reps = len(sim_data)

                    #     for rep_idx in range(num_reps):

                    
                    trajectory_pilot_analyses = {}
                    
                    for pilot_name, sim_file_list in trajectory_pilot_data[traj_idx].items():
                        print(f"  Analyzing {len(sim_file_list)} repetitions for pilot: {pilot_name}")
                        
                        pilot_analyses = []
                        
                        for sim_file in sim_file_list:
                            sim_file_path = os.path.join(search_dir, sim_file)
                            
                            try:
                                # Load simulation data
                                sim_data = torch.load(sim_file_path)
                                
                                # Process ALL trajectory repetitions in the file, not just the first one
                                for rep_idx, trajectory_data in enumerate(sim_data):
                                    Tro = trajectory_data.get('Tro', None)  # Time array might not exist
                                    Xro = trajectory_data['Xro']  # State trajectory
                                    
                                    # Use stored analysis if available, otherwise recompute
                                    if 'analysis' in trajectory_data and not force_reanalysis:
                                        # Use pre-computed analysis (single source of truth)
                                        analysis_result = trajectory_data['analysis']
                                        print(f"    Using stored analysis for {pilot_name} rep {rep_idx}")
                                    else:
                                        # Recompute analysis (only if forced or no stored analysis)
                                        print(f"    Recomputing analysis for {pilot_name} rep {rep_idx} (force_reanalysis={force_reanalysis})")
                                        analysis_result = analyze_trajectory_performance(
                                            Xro, goal_location, point_cloud, exclusion_radius, 
                                            collision_radius, trajectory_name=f"{scene_name}_{obj_name}_traj{traj_idx}_{pilot_name}_rep{rep_idx}"
                                        )
                                    
                                    pilot_analyses.append(analysis_result)
                                
                            except Exception as e:
                                print(f"    Error analyzing {sim_file}: {e}")
                                continue
                        
                        trajectory_pilot_analyses[pilot_name] = pilot_analyses
                    
                    # Store this trajectory's analysis across all pilots
                    all_trajectory_analyses.append({
                        "trajectory_index": traj_idx,
                        "pilot_analyses": trajectory_pilot_analyses
                    })
                
                # Compute aggregate statistics (same as simulation path)
                print("Computing aggregate trajectory statistics...")
                
                # Group analyses by pilot for individual pilot statistics
                pilot_specific_analyses = {}
                
                for traj_analysis in all_trajectory_analyses:
                    for pilot_name, pilot_analyses in traj_analysis["pilot_analyses"].items():
                        if pilot_name not in pilot_specific_analyses:
                            pilot_specific_analyses[pilot_name] = []
                        pilot_specific_analyses[pilot_name].extend(pilot_analyses)
                
                # Compute pilot-specific aggregate statistics
                pilot_aggregates = {}
                for pilot_name, pilot_analyses in pilot_specific_analyses.items():
                    pilot_aggregates[pilot_name] = compute_aggregate_statistics(pilot_analyses)
                
                # Create comprehensive analysis summary
                pilot_list = list(pilot_specific_analyses.keys())
                analysis_summary = {
                    "scene_name": scene_name,
                    "objective": obj_name,
                    "goal_location": goal_location.tolist(),
                    "exclusion_radius": exclusion_radius,
                    "collision_radius": collision_radius,
                    "point_cloud_size": len(point_cloud),
                    "num_trajectories": len(trajectory_pilot_data),
                    "pilots_tested": pilot_list,
                    "timestamp": simulation_data_dir.split('/')[-1],
                    "pilot_specific_statistics": pilot_aggregates,
                    "all_trajectory_analyses": all_trajectory_analyses
                }
                
                # Save reconstructed analysis
                def convert_numpy_for_json(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_for_json(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_for_json(item) for item in obj]
                    else:
                        return obj
                
                analysis_summary_json = convert_numpy_for_json(analysis_summary)
                
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_summary_json, f, indent=2)
                
                print(f"Saved reconstructed trajectory analysis to: {analysis_file}")
                
                # Print summary statistics (same as simulation path)
                print("\n" + "="*70)
                print("TRAJECTORY ANALYSIS SUMMARY (REVIEW)")
                print("="*70)
                print(f"Scene: {scene_name}, Objective: {obj_name}")
                print(f"Goal exclusion radius (r1): {exclusion_radius:.2f}m")
                print(f"Collision check radius (r2): {collision_radius:.2f}m")
                print(f"Total trajectories analyzed: {len(trajectory_pilot_data)}")
                print(f"Pilots found: {', '.join(pilot_list)}")
                
                print("\nPilot performance comparison:")
                for pilot_name, pilot_stats in pilot_aggregates.items():
                    print(f"  {pilot_name}:")
                    print(f"    Total simulations: {pilot_stats['total_trajectories']}")
                    print(f"    Success rate: {pilot_stats['success_rate']:.1%}")
                    print(f"    Collision rate: {pilot_stats['collision_rate']:.1%}")
                    print(f"    Mean distance to goal: {pilot_stats['mean_distance_to_goal']:.2f}m")
                    print(f"    Mean normalized distance: {pilot_stats['mean_normalized_distance']:.3f}")
                    print(f"    Mean yaw error: {pilot_stats['mean_yaw_error_degrees']:.1f}°")
                    print(f"    Camera FOV success rate: {pilot_stats['camera_fov_success_rate']:.1%}")
                
                print("="*70)
                print("Review analysis completed!")
        
        
        # Visualize all trajectories in a single figure
        if all_debug_info:
            radius_info = {'r1': radii[obj_index][0], 'r2': radii[obj_index][1]}
            
            # Create the original RRT figure with debug info trajectories
            fig_rrt = bd.visualize_multiple_trajectories(
                all_debug_info, epcds_list, obj_targets[obj_index], 
                radius_info, scene_cfg_file, simulator
            )
            
            # Create visualizations subdirectory to keep HTML files organized
            visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
            os.makedirs(visualizations_dir, exist_ok=True)
            
            # Save original RRT figure
            rrt_plot_path = os.path.join(
                visualizations_dir,
                f"rrt_plot_{scene_name}_{obj_name}.html"
            )
            fig_rrt.write_html(rrt_plot_path)
            print(f"Original RRT plot saved to: {rrt_plot_path}")
            
            # If simulation was run or review was performed, create separate figures
            if (simulate or review) and 'analysis_file' in locals():
                
                # 1. Enhanced figure with both RRT and simulation trajectories
                action_type = "simulation" if simulate else "review"
                print(f"Creating enhanced RRT plot with {action_type} trajectories...")
                fig_enhanced = add_simulation_trajectories_to_figure(
                    fig_rrt, analysis_file, scene_name, obj_name
                )
                
                enhanced_plot_path = os.path.join(
                    visualizations_dir,
                    f"enhanced_rrt_plot_{scene_name}_{obj_name}.html"
                )
                fig_enhanced.write_html(enhanced_plot_path)
                print(f"Enhanced RRT plot saved to: {enhanced_plot_path}")
                
                # 2. Simulation-only figure without RRT trajectories
                print(f"Creating {action_type}-only plot...")
                fig_sim_only = create_simulation_only_figure(
                    analysis_file, scene_name, obj_name, epcds_list,
                    obj_targets[obj_index], radius_info, scene_cfg_file, simulator
                )
                
                sim_only_plot_path = os.path.join(
                    visualizations_dir,
                    f"{action_type}_only_plot_{scene_name}_{obj_name}.html"
                )
                fig_sim_only.write_html(sim_only_plot_path)
                print(f"{action_type.capitalize()}-only plot saved to: {sim_only_plot_path}")
                
                # Show both figures
                fig_enhanced.show()
                fig_sim_only.show()
            else:
                # Just show the original RRT figure if no simulation or review
                fig_rrt.show()
        else:
            print("No valid debug info generated for visualization")


def create_simulation_only_figure(analysis_file_path, scene_name, obj_name, epcds_list, 
                                 goal_location, radius_info, scene_cfg_file, simulator):
    """
    Create a figure with only simulation trajectories (no RRT debug_info trajectories).
    Uses the same base environment setup as RRT figures for consistency.
    
    Parameters:
        analysis_file_path: Path to the trajectory analysis JSON file
        scene_name: Scene name for finding simulation files
        obj_name: Objective name for finding simulation files
        epcds_list: Point cloud list for environment
        goal_location: Goal location coordinates
        radius_info: Dictionary with r1 and r2 radius values
        scene_cfg_file: Scene configuration file path
        simulator: Simulator instance
    
    Returns:
        plotly.graph_objects.Figure: Figure with simulation trajectories only
    """
    try:
        # Create a base figure using the same function that creates RRT figures,
        # but with empty debug_info so we get the environment without RRT trajectories
        empty_debug_info = []  # No RRT trajectories
        
        # Use the same visualization function that creates the RRT plots
        # This ensures we get the exact same environment setup (point cloud, goal, etc.)
        fig = bd.visualize_multiple_trajectories(
            empty_debug_info, epcds_list, goal_location, 
            radius_info, scene_cfg_file, simulator
        )
        
        # Now add simulation trajectories using the same function that works for enhanced plots
        fig = add_simulation_trajectories_to_figure(fig, analysis_file_path, scene_name, obj_name)
        
        # Update the title to indicate this is simulation-only
        current_title = fig.layout.title.text if fig.layout.title else ""
        fig.update_layout(
            title=f"Simulation Results: {scene_name} - {obj_name}<br><sub>Individual Pilot Performance (Simulation Only)</sub>"
        )
        
        print(f"Created simulation-only plot using same environment setup as RRT figures")
        
    except Exception as e:
        print(f"Warning: Could not create simulation-only figure: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to empty figure
        import plotly.graph_objects as go
        fig = go.Figure()
    
    return fig


def add_simulation_trajectories_to_figure(fig, analysis_file_path, scene_name, obj_name):
    """
    Add actual simulation trajectories to an existing plotly figure.
    
    Parameters:
        fig: Existing plotly figure (from bd.visualize_multiple_trajectories)
        analysis_file_path: Path to the trajectory analysis JSON file
        scene_name: Scene name for finding simulation files
        obj_name: Objective name for finding simulation files
    
    Returns:
        plotly.graph_objects.Figure: Enhanced figure with simulation trajectories
    """
    try:
        # Import required modules
        import plotly.graph_objects as go
        import torch
        
        # Load analysis data
        with open(analysis_file_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Storage for trajectory data and markers
        pilot_trajectories = {}  # Group by pilot name: {pilot_name: [traj_data, ...]}
        collision_points = []
        goal_points = []
        
        # Extract trajectory data from individual analyses
        # Handle both old and new organized directory structures
        analysis_dir = os.path.dirname(analysis_file_path)
        # New organized structure: analysis file is in visualizations/ subdirectory
        main_simulation_dir = os.path.dirname(analysis_dir)  # Go up one level to main simulation directory
        # Check if trajectories are in the organized trajectories/ subdirectory
        trajectories_subdir = os.path.join(main_simulation_dir, "trajectories")
        if os.path.exists(trajectories_subdir):
            simulation_dir = trajectories_subdir  # Use trajectories/ subdirectory
        else:
            raise ValueError(f"Trajectories directory not found: {trajectories_subdir}")

        for traj_analysis in analysis_data['all_trajectory_analyses']:
            traj_idx = traj_analysis['trajectory_index']
            
            for pilot_name, pilot_analyses in traj_analysis['pilot_analyses'].items():
                for rep_idx, analysis in enumerate(pilot_analyses):
                    # Get the simulation data file path
                    sim_file = os.path.join(
                        simulation_dir, 
                        f"sim_data_{scene_name}_{obj_name}_traj{traj_idx}_{pilot_name}.pt"
                    )
                    
                    # Load simulation data if file exists
                    if os.path.exists(sim_file):
                        try:
                            sim_data = torch.load(sim_file, map_location='cpu')
                            if rep_idx < len(sim_data):
                                trajectory_data = sim_data[rep_idx]
                                Xro = trajectory_data['Xro']  # State trajectory
                                
                                # Extract positions (x, y, z are typically first 3 states)
                                if Xro.shape[1] > 0:  # Check if trajectory has points
                                    positions = Xro[0:3, :].T  # N x 3 array
                                    
                                    # Store trajectory grouped by pilot
                                    traj_data = {
                                        'x': positions[:, 0],
                                        'y': positions[:, 1], 
                                        'z': positions[:, 2],
                                        'pilot': pilot_name,
                                        'traj_idx': traj_idx,
                                        'rep_idx': rep_idx
                                    }
                                    
                                    # Group trajectories by pilot name
                                    if pilot_name not in pilot_trajectories:
                                        pilot_trajectories[pilot_name] = []
                                    pilot_trajectories[pilot_name].append(traj_data)
                                    
                                    # Mark collision and goal points based on analysis
                                    terminal_idx = analysis['terminal_index']
                                    if terminal_idx < len(positions):
                                        terminal_pos = positions[terminal_idx]
                                        
                                        if analysis['terminal_reason'] == 'collision':
                                            collision_points.append({
                                                'x': terminal_pos[0],
                                                'y': terminal_pos[1],
                                                'z': terminal_pos[2],
                                                'pilot': pilot_name
                                            })
                                        elif analysis['terminal_reason'] == 'goal_reached':
                                            goal_points.append({
                                                'x': terminal_pos[0],
                                                'y': terminal_pos[1],
                                                'z': terminal_pos[2],
                                                'pilot': pilot_name
                                            })
                        except Exception as e:
                            print(f"Warning: Could not load simulation data from {sim_file}: {e}")
        
        # Import plotly for adding traces
        import plotly.graph_objects as go
        
        # Define colors for different pilots
        pilot_colors = {
            'expert': 'blue',
            # Add more colors for other pilots - cycle through distinct colors
        }
        default_colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        color_idx = 0
        
        # Add trajectories for each pilot individually
        total_trajectories = 0
        for pilot_name, trajectories in pilot_trajectories.items():
            # Get color for this pilot
            if pilot_name in pilot_colors:
                color = pilot_colors[pilot_name]
            else:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1
            
            # Add all trajectories for this pilot
            for i, traj in enumerate(trajectories):
                fig.add_trace(go.Scatter3d(
                    x=traj['x'], y=traj['y'], z=traj['z'],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=pilot_name if i == 0 else None,  # Only show legend for first trajectory of each pilot
                    showlegend=(i == 0),
                    hovertemplate=f"{pilot_name} Traj {traj['traj_idx']}<br>" +
                                 "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
                ))
            
            total_trajectories += len(trajectories)
        
        # Add collision markers as red X
        if collision_points:
            collision_x = [cp['x'] for cp in collision_points]
            collision_y = [cp['y'] for cp in collision_points] 
            collision_z = [cp['z'] for cp in collision_points]
            fig.add_trace(go.Scatter3d(
                x=collision_x, y=collision_y, z=collision_z,
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='red',
                    line=dict(width=3)
                ),
                name=f'Collisions ({len(collision_points)})',
                hovertemplate="Collision<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            ))
        
        # Add goal achievement markers as green O
        if goal_points:
            goal_x = [gp['x'] for gp in goal_points]
            goal_y = [gp['y'] for gp in goal_points]
            goal_z = [gp['z'] for gp in goal_points]
            fig.add_trace(go.Scatter3d(
                x=goal_x, y=goal_y, z=goal_z,
                mode='markers',
                marker=dict(
                    symbol='circle-open',
                    size=10,
                    color='green',
                    line=dict(width=3)
                ),
                name=f'Goal Reached ({len(goal_points)})',
                hovertemplate="Goal Reached<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            ))

        # Update the title to indicate enhanced view
        current_title = fig.layout.title.text if fig.layout.title else ""
        fig.update_layout(
            title=f"{current_title}<br><sub>Enhanced with Simulation Results</sub>"
        )
        
        # Print summary of what was added
        pilot_counts = {pilot: len(trajs) for pilot, trajs in pilot_trajectories.items()}
        pilot_summary = ", ".join([f"{pilot}: {count}" for pilot, count in pilot_counts.items()])
        print(f"Enhanced RRT plot with {total_trajectories} total trajectories ({pilot_summary})")
        print(f"Added {len(collision_points)} collision markers and {len(goal_points)} goal achievement markers")
        
    except Exception as e:
        print(f"Warning: Could not enhance figure with simulation trajectories: {e}")
        import traceback
        traceback.print_exc()
    
    return fig


def create_simulation_violin_plots(cohort_name: str, scene_name: str = None, 
                                   objectives: List[str] = None, pilots: List[str] = None,
                                   figsize: tuple = (16, 12),
                                   include_failures: bool = True, separate_failure_analysis: bool = True,
                                   include_column_charts: bool = True):
    """
    Create violin plots for simulated experiments using pre-computed analysis results.
    
    This function loads analysis data saved by analyze_simulated_experiments.py and creates
    violin plots comparing expert vs pilot performance using normalized distance to goal.
    Optionally includes column charts for success rates and query-in-view rates.
    
    Parameters:
        cohort_name: Name of the cohort containing simulation data
        scene_name: Specific scene to analyze (if None, analyzes all scenes found)
        objectives: List of objectives to analyze (if None, analyzes all found)
        pilots: List of pilot names to include (if None, includes all found)
        figsize: Figure size for the plots
        include_failures: Whether to include failed trajectories (collisions/timeouts) in violin plots
        separate_failure_analysis: Whether to show separate analysis of failure rates
        include_column_charts: Whether to create additional column charts with confidence intervals
    
    Returns:
        pandas.DataFrame: DataFrame with all simulation results for further analysis
    """
    try:
        import pandas as pd
        import seaborn as sns
        sns.set_theme(style="whitegrid",palette="deep")
        import matplotlib.pyplot as plt
        from scipy import stats
        import json
        pandas_available = True
    except ImportError:
        print("Error: pandas, seaborn, matplotlib, and scipy are required for violin plots")
        return None
    
    # Get workspace path
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Extract configs
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)
    
    # Find the most recent simulation data directory
    simulation_data_base = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(simulation_data_base):
        raise ValueError(f"No simulation data found in {simulation_data_base}")
    
    # Get the most recent timestamped directory
    timestamped_dirs = [d for d in os.listdir(simulation_data_base) 
                       if os.path.isdir(os.path.join(simulation_data_base, d))]
    if not timestamped_dirs:
        raise ValueError(f"No timestamped directories found in {simulation_data_base}")
    
    latest_dir = max(timestamped_dirs)
    simulation_data_dir = os.path.join(simulation_data_base, latest_dir)
    print(f"Loading simulation data from: {simulation_data_dir}")

    # Visualizations directory
    visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
    
    # Find all analysis JSON files
    analysis_files = []
    if os.path.exists(visualizations_dir):
        for file_name in os.listdir(visualizations_dir):
            if file_name.startswith("trajectory_analysis_") and file_name.endswith(".json"):
                analysis_files.append(os.path.join(visualizations_dir, file_name))
    
    if not analysis_files:
        raise ValueError(f"No analysis JSON files found in {visualizations_dir}. "
                        f"Please run analyze_simulated_experiments.py first.")
    
    print(f"Found {len(analysis_files)} analysis files")
    
    # Load all analysis data and extract trajectory results
    all_results = []
    experiment_info = {}  # Track scene/objective combinations
    
    for analysis_file in analysis_files:
        print(f"Loading analysis: {os.path.basename(analysis_file)}")
        
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            scene = analysis_data['scene_name']
            objective = analysis_data['objective']
            experiment_id = f"{scene}_{objective}"
            
            # Apply filters if specified
            if scene_name and scene != scene_name:
                continue
            if objectives and objective not in objectives:
                continue
            
            # Store experiment info
            experiment_info[experiment_id] = {
                'scene': scene,
                'objective': objective,
                'goal_location': analysis_data['goal_location'],
                'exclusion_radius': analysis_data.get('exclusion_radius', 0),
                'collision_radius': analysis_data.get('collision_radius', 0)
            }
            
            # Extract individual trajectory results
            for traj_analysis in analysis_data['all_trajectory_analyses']:
                traj_idx = traj_analysis['trajectory_index']
                
                for pilot_name, pilot_analyses in traj_analysis['pilot_analyses'].items():
                    # Apply pilot filter
                    if pilots and pilot_name not in pilots:
                        continue
                    
                    for rep_idx, analysis in enumerate(pilot_analyses):
                        all_results.append({
                            'scene': scene,
                            'objective': objective,
                            'pilot': pilot_name,
                            'trajectory_index': traj_idx,
                            'repetition': rep_idx,
                            'normalized_distance': analysis['normalized_distance'],
                            'distance_to_goal': analysis['distance_to_goal'],
                            'success': analysis['success'],
                            'collision': analysis['collision'],
                            'goal_in_camera_fov': analysis['goal_in_camera_fov'],
                            'yaw_error_degrees': analysis['yaw_error_degrees'],
                            'terminal_reason': analysis['terminal_reason'],
                            'trajectory_length': analysis['trajectory_length'],
                            'experiment_id': experiment_id
                        })
            
        except Exception as e:
            print(f"  Error loading {analysis_file}: {e}")
            continue
    
    if not all_results:
        print("No simulation results match the specified filters")
        return None
    
    # Create DataFrame
    df_sim = pd.DataFrame(all_results)
    
    print(f"\nLoaded {len(df_sim)} simulation results")
    print(f"Experiments: {sorted(df_sim['experiment_id'].unique())}")
    print(f"Pilots: {sorted(df_sim['pilot'].unique())}")
    
    # Analyze failure patterns before filtering
    if separate_failure_analysis and len(df_sim) > 0:
        print(f"\n" + "="*60)
        print("FAILURE ANALYSIS (ALL TRAJECTORIES)")
        print("="*60)
        
        # Overall failure rates by pilot
        failure_summary = df_sim.groupby('pilot').agg({
            'success': 'mean',
            'collision': 'mean'
        }).round(3)
        
        print("Success and Collision Rates by Pilot:")
        for pilot in failure_summary.index:
            success_rate = failure_summary.loc[pilot, 'success']
            collision_rate = failure_summary.loc[pilot, 'collision']
            timeout_rate = 1.0 - success_rate - collision_rate
            print(f"  {pilot}:")
            print(f"    Success: {success_rate:.1%}")
            print(f"    Collision: {collision_rate:.1%}")
            print(f"    Timeout: {timeout_rate:.1%}")
        
        from statsmodels.stats.proportion import proportion_confint

        def proportion_ci(series, method='wilson', alpha=0.05):
            successes = series.sum()
            n = len(series)
            mean = successes / n if n > 0 else np.nan
            lower, upper = proportion_confint(successes, n, alpha=alpha, method=method) if n > 0 else (np.nan, np.nan)
            return pd.Series({'mean': mean, 'ci_lower': lower, 'ci_upper': upper})

        # Failure rates by objective
        print(f"\nFailure Rates by Objective:")
        obj_failure_summary = df_sim.groupby(['pilot', 'objective']).agg({
            'success': 'mean',
            'collision': 'mean',
            'goal_in_camera_fov': 'mean'
        }).round(3)

        # Count collisions where goal_in_camera_fov is False
        collision_mask = (df_sim['collision'] == True)
        collision_and_not_in_view = df_sim[collision_mask & (df_sim['goal_in_camera_fov'] == False)]
        num_collisions = collision_mask.sum()
        num_collisions_not_in_view = len(collision_and_not_in_view)
        if num_collisions > 0:
            percent_not_in_view = 100.0 * num_collisions_not_in_view / num_collisions
            # Compute Clopper-Pearson confidence interval for proportion
            alpha = 0.05
            lower, upper = clopper_pearson_ci(num_collisions_not_in_view, num_collisions, alpha)
            print(f"\nOf {num_collisions} collisions, {num_collisions_not_in_view} ({percent_not_in_view:.1f}%) occurred when goal_in_camera_fov=False.")
            print(f"95% CI for this proportion: [{100.0*lower:.1f}%, {100.0*upper:.1f}%]")
        else:
            print("\nNo collisions found in the data.")

        # Compute by objective as well
        print("\nCollision analysis by objective:")
        for objective in df_sim['objective'].unique():
            obj_mask = (df_sim['objective'] == objective)
            obj_collision_mask = obj_mask & (df_sim['collision'] == True)
            obj_collision_and_not_in_view = df_sim[obj_collision_mask & (df_sim['goal_in_camera_fov'] == False)]
            obj_num_collisions = obj_collision_mask.sum()
            obj_num_collisions_not_in_view = len(obj_collision_and_not_in_view)
            if obj_num_collisions > 0:
                obj_percent_not_in_view = 100.0 * obj_num_collisions_not_in_view / obj_num_collisions
                lower, upper = clopper_pearson_ci(obj_num_collisions_not_in_view, obj_num_collisions, alpha)
                print(f"  Objective '{objective}': {obj_num_collisions_not_in_view}/{obj_num_collisions} ({obj_percent_not_in_view:.1f}%) collisions occurred when goal_in_camera_fov=False.")
                print(f"    95% CI: [{100.0*lower:.1f}%, {100.0*upper:.1f}%]")
            else:
                print(f"  Objective '{objective}': No collisions found.")

        for metric in ['success', 'collision', 'goal_in_camera_fov']:
            print(f"\n{metric.capitalize()} rates and CIs by pilot and objective:")
            summary = df_sim.groupby(['pilot', 'objective'])[metric].apply(proportion_ci)
            print(summary)
        
        for (pilot, objective), rates in obj_failure_summary.iterrows():
            success_rate = rates['success']
            collision_rate = rates['collision']
            print(f"  {pilot} - {objective}: Success {success_rate:.1%}, Collision {collision_rate:.1%}")
    
    # Filter data for violin plots based on include_failures parameter
    if include_failures:
        df_plot = df_sim.copy()
        plot_subtitle = "All Trajectories (Including Failures)"
        print(f"\n📊 Violin plots will include ALL {len(df_plot)} trajectories (successes + failures)")
    else:
        # Only include successful trajectories for violin plots
        df_plot = df_sim[df_sim['success'] == True].copy()
        plot_subtitle = "Successful Trajectories Only"
        print(f"\n📊 Violin plots will include only SUCCESSFUL trajectories: {len(df_plot)}/{len(df_sim)} ({len(df_plot)/len(df_sim)*100:.1f}%)")
        
        if len(df_plot) == 0:
            print("❌ No successful trajectories found! Cannot create violin plots.")
            print("💡 Try setting include_failures=True to see all trajectory performance.")
            return df_sim
    
    # Set random seed for consistent stripplot jitter positions
    np.random.seed(42)
    
    # Create violin plots (similar to review_experiments.ipynb style)
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()
    
    # Get unique experiments for subplot layout
    experiments = df_plot['experiment_id'].unique()
    n_experiments = len(experiments)
    
    # Calculate subplot layout
    if n_experiments <= 3:
        nrows, ncols = 1, n_experiments
    elif n_experiments <= 6:
        nrows, ncols = 2, 3
    elif n_experiments <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 4  # Adjust as needed
    
    # Get consistent pilot ordering (same as column charts)
    pilot_order = sorted(df_plot['pilot'].unique())
    
    # Individual experiment plots
    for i, exp_id in enumerate(experiments):
        plt.subplot(nrows, ncols, i+1)
        df_exp = df_plot[df_plot['experiment_id'] == exp_id]
        
        # Create violin plot with consistent pilot order
        sns.violinplot(data=df_exp, x="pilot", y="normalized_distance",
                      order=pilot_order, inner=None, cut=0, density_norm="width",
                      color="#ffffff", edgecolor="#505050", linewidth=1.0)
        # sns.violinplot(data=df_exp, x="pilot", y="normalized_distance",
        #               order=pilot_order, inner=None, cut=0, density_norm="width")
        
        # # Add strip plot for individual points
        sns.stripplot(data=df_exp, x="pilot", y="normalized_distance",
                     order=pilot_order, jitter=0.2,
                     color="#2690db", alpha=0.7) 
                    #  color="#ffc400", alpha=0.7)
        # sns.swarmplot(data=df_exp, x="pilot", y="normalized_distance",
        #               order=pilot_order, alpha=0.7, linewidth=0.3, size=3)
        # sns.swarmplot(data=df_exp, x="pilot", y="normalized_distance",
        #               order=pilot_order, alpha=0.7, edgecolor="k", linewidth=0.3, size=12)
        # color=["#e9ae0d" if pilot == 'expert' else "#2690db"
        # Add medians with consistent pilot ordering
        medians = df_exp.groupby("pilot")["normalized_distance"].median()
        for j, pilot in enumerate(pilot_order):
            if pilot in medians.index:
                median_val = medians[pilot]
                plt.scatter(j, median_val, zorder=3, s=40, color="#FF6600", marker="o")
        
        plt.title(f"{exp_id.replace('_', ' ')}", fontsize=10)
        plt.ylabel("Normalized Distance to Goal" if i % ncols == 0 else "")
        plt.xlabel("")
        if i < n_experiments - ncols:  # Not bottom row
            plt.xticks([])
        else:
            plt.xticks(rotation=45, fontsize=8)
        plt.grid(axis="y", linestyle=":", alpha=0.3)
    
    # Aggregate plot if we have remaining space
    if n_experiments < nrows * ncols:
        plt.subplot(nrows, ncols, n_experiments + 1)
        # Create violin plot with consistent pilot order using ALL filtered data
        sns.violinplot(data=df_plot, x="pilot", y="normalized_distance",
                      order=pilot_order, inner=None, cut=0, density_norm="width",
                      color="#ffffff", edgecolor="#505050", linewidth=1.0)
        
        # Add strip plot for individual points
        sns.stripplot(data=df_plot, x="pilot", y="normalized_distance",
                     order=pilot_order, jitter=0.2,
                     color="#2690db", alpha=0.7)
        # sns.swarmplot(data=df_exp, x="pilot", y="normalized_distance",
        #               order=pilot_order, alpha=0.7, edgecolor="k", linewidth=0.3, size=12)
        
        # Add aggregate medians with consistent pilot ordering  
        medians = df_plot.groupby("pilot")["normalized_distance"].median()
        for j, pilot in enumerate(pilot_order):
            if pilot in medians.index:
                median_val = medians[pilot]
                plt.scatter(j, median_val, zorder=3, s=50, color="red", marker="o")
        
        plt.title("AGGREGATE", fontweight="bold")
        plt.ylabel("Normalized Distance")
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(axis="y", linestyle=":", alpha=0.3)
    
    plt.tight_layout()
    
    # Add main title with subtitle indicating what data is shown
    fig.suptitle(f"Simulation Performance Analysis\n{plot_subtitle}", 
                fontsize=10, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9)  # Make room for title
    
    # Save plot if requested
    os.makedirs(visualizations_dir, exist_ok=True)
    plot_path = os.path.join(visualizations_dir, "violin_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {plot_path}")
    
    plt.show()
    
    # Create column charts with confidence intervals if requested
    if include_column_charts and len(df_sim) > 0:
        create_performance_column_charts(df_sim, cohort_name, os.path.join(visualizations_dir, "column_charts"))
    
    # Print summary statistics
    print("\n" + "="*70)
    print(f"SIMULATION VIOLIN PLOT ANALYSIS - {plot_subtitle.upper()}")
    print("="*70)
    
    print(f"\n1. OVERALL MEDIANS BY PILOT ({len(df_plot)} trajectories):")
    overall_medians = df_plot.groupby("pilot")["normalized_distance"].median().sort_values()
    for pilot, median_val in overall_medians.items():
        print(f"   {pilot}: {median_val:.3f}")
    
    print("\n2. SUCCESS RATES BY PILOT (from all data):")
    if 'success' in df_sim.columns and df_sim['success'].notna().any():
        success_rates = df_sim.groupby("pilot")["success"].mean()
        for pilot, rate in success_rates.items():
            print(f"   {pilot}: {rate:.1%}")
    else:
        print("   (Success rate data not available)")
    
    print(f"\n3. PERFORMANCE BY EXPERIMENT ({plot_subtitle.lower()}):")
    exp_summary = df_plot.groupby(["experiment_id", "pilot"])["normalized_distance"].median().unstack()
    print(exp_summary)
    
    print(f"\n4. VARIANCE ANALYSIS ({plot_subtitle.lower()}):")
    variance_summary = df_plot.groupby("pilot")["normalized_distance"].std()
    for pilot, std_val in variance_summary.items():
        print(f"   {pilot}: σ = {std_val:.3f}")
    
    # Add interpretation guidance
    print(f"\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    if include_failures:
        print("✅ Violin plots include ALL trajectories (successes + failures)")
        print("  • Lower normalized distance = better performance")
        print("  • 0.0 = perfect (reached goal)")  
        print("  • 1.0 = no progress made")
        print("  • >1.0 = moved away from goal")
        print("  • Very high values (>2.0) typically indicate collisions or major failures")
    else:
        print("✅ Violin plots show SUCCESSFUL trajectories only")
        print("  • This isolates navigation quality from basic safety")
        print("  • Lower normalized distance = better precision when successful")
        print("  • 0.0 = perfect goal reaching")
        print("  • Values should mostly be < 1.0 since these are successful attempts")
        print("  • Refer to 'Success Rates' above for overall safety performance")
    
    print("\n💡 RECOMMENDATIONS:")
    if include_failures:
        print("  • Large tails in violin plots indicate frequent failures")
        print("  • Compare both central tendency (median) and spread (failures)")
        print("  • Consider running with include_failures=False to isolate navigation skill")
    else:
        print("  • This shows navigation precision for successful flights only")
        print("  • Combine with overall success rates for complete picture")
        print("  • Run with include_failures=True to see complete performance distribution")
    
    print(f"\n💡 DATA SOURCE:")
    print(f"  • Analysis data loaded from: {len(analysis_files)} JSON files")
    print(f"  • Data pre-computed by analyze_simulated_experiments.py")
    print(f"  • No filename parsing or trajectory reloading required")
    
    return df_sim


def create_simulation_time_history_plots(cohort_name: str, scene_name: str = None, 
                                   objectives: List[str] = None, pilots: List[str] = None,
                                   figsize: tuple = (16, 12),
                                   include_failures: bool = True, separate_failure_analysis: bool = True,
                                   include_column_charts: bool = True):
    """
    Create time history plots for simulated experiments using pre-computed analysis results.
    
    This function loads analysis data saved by analyze_simulated_experiments.py and creates
    time history plots comparing expert vs pilot performance using position and orientation.
    Optionally includes column charts for success rates and query-in-view rates.
    
    Parameters:
        cohort_name: Name of the cohort containing simulation data
        scene_name: Specific scene to analyze (if None, analyzes all scenes found)
        objectives: List of objectives to analyze (if None, analyzes all found)
        pilots: List of pilot names to include (if None, includes all found)
        figsize: Figure size for the plots
        include_failures: Whether to include failed trajectories (collisions/timeouts) in violin plots
        separate_failure_analysis: Whether to show separate analysis of failure rates
        include_column_charts: Whether to create additional column charts with confidence intervals
    
    Returns:
    """
    try:
        import pandas as pd
        import seaborn as sns
        sns.set_theme(style="whitegrid",palette="deep")
        import matplotlib.pyplot as plt
        from scipy import stats
        import json
        pandas_available = True
    except ImportError:
        print("Error: pandas, seaborn, matplotlib, and scipy are required for plotting!")
        return None
    
    # Get workspace path
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Extract configs
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)
    
    # Find the most recent simulation data directory
    simulation_data_base = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(simulation_data_base):
        raise ValueError(f"No simulation data found in {simulation_data_base}")
    
    # Get the most recent timestamped directory
    timestamped_dirs = [d for d in os.listdir(simulation_data_base) 
                       if os.path.isdir(os.path.join(simulation_data_base, d))]
    if not timestamped_dirs:
        raise ValueError(f"No timestamped directories found in {simulation_data_base}")
    
    latest_dir = max(timestamped_dirs)
    simulation_data_dir = os.path.join(simulation_data_base, latest_dir)
    print(f"Loading simulation data from: {simulation_data_dir}")

    # Visualizations directory
    visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
    
    # Find all analysis JSON files
    analysis_files = []
    if os.path.exists(visualizations_dir):
        for file_name in os.listdir(visualizations_dir):
            if file_name.startswith("trajectory_analysis_") and file_name.endswith(".json"):
                analysis_files.append(os.path.join(visualizations_dir, file_name))
    
    if not analysis_files:
        raise ValueError(f"No analysis JSON files found in {visualizations_dir}. "
                        f"Please run analyze_simulated_experiments.py first.")
    
    print(f"Found {len(analysis_files)} analysis files")
    
    # Load all analysis data and extract trajectory results
    all_results = []
    experiment_info = {}  # Track scene/objective combinations
    
    for analysis_file in analysis_files:
        print(f"Loading analysis: {os.path.basename(analysis_file)}")
        
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            scene = analysis_data['scene_name']
            objective = analysis_data['objective']
            experiment_id = f"{scene}_{objective}"
            
            # Apply filters if specified
            if scene_name and scene != scene_name:
                continue
            if objectives and objective not in objectives:
                continue
            
            # Store experiment info
            experiment_info[experiment_id] = {
                'scene': scene,
                'objective': objective,
                'goal_location': analysis_data['goal_location'],
                'exclusion_radius': analysis_data.get('exclusion_radius', 0),
                'collision_radius': analysis_data.get('collision_radius', 0)
            }
            
            # Extract individual trajectory results
            for traj_analysis in analysis_data['all_trajectory_analyses']:
                traj_idx = traj_analysis['trajectory_index']
                
                for pilot_name, pilot_analyses in traj_analysis['pilot_analyses'].items():
                    # Apply pilot filter
                    if pilots and pilot_name not in pilots:
                        continue
                    
                    for rep_idx, analysis in enumerate(pilot_analyses):
                        all_results.append({
                            'scene': scene,
                            'objective': objective,
                            'pilot': pilot_name,
                            'trajectory_index': traj_idx,
                            'repetition': rep_idx,
                            'normalized_distance': analysis['normalized_distance'],
                            'distance_to_goal': analysis['distance_to_goal'],
                            'success': analysis['success'],
                            'collision': analysis['collision'],
                            'goal_in_camera_fov': analysis['goal_in_camera_fov'],
                            "yaw_error_series": analysis['yaw_error_series'],
                            "yaw_error_degrees_series": analysis['yaw_error_degrees_series'],
                            "goal_in_camera_fov_series": analysis['goal_in_camera_fov_series'],
                            "yaw_error_mean_upto_terminal": analysis['yaw_error_mean_upto_terminal'],
                            "yaw_error_max_upto_terminal": analysis['yaw_error_max_upto_terminal'],
                            'yaw_error_degrees': analysis['yaw_error_degrees'],
                            'terminal_reason': analysis['terminal_reason'],
                            'trajectory_length': analysis['trajectory_length'],
                            'experiment_id': experiment_id
                        })
            
        except Exception as e:
            print(f"  Error loading {analysis_file}: {e}")
            continue
    
    if not all_results:
        print("No simulation results match the specified filters")
        return None
    
    # Create DataFrame
    df_sim = pd.DataFrame(all_results)
    
    print(f"\nLoaded {len(df_sim)} simulation results")
    print(f"Experiments: {sorted(df_sim['experiment_id'].unique())}")
    print(f"Pilots: {sorted(df_sim['pilot'].unique())}")

    # =============================
    # Build long-form yaw-error data
    # =============================

    # Helper to explode each row's series into (idx, t_norm, value_deg)
    def _row_series_to_long(row, n_interp=101):
        y = np.asarray(row["yaw_error_degrees_series"], dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            return None

        # raw index axis
        t_idx = np.arange(y.size)

        if n_interp is None or y.size == 1:
            # No interpolation; normalized time just from indices
            t_norm = (t_idx / max(1, (y.size - 1))).astype(float)
            df = pd.DataFrame({
                "experiment_id": row["experiment_id"],
                "scene": row["scene"],
                "objective": row["objective"],
                "pilot": row["pilot"],
                "trajectory_index": row["trajectory_index"],
                "repetition": row["repetition"],
                "success": row["success"],
                "collision": row["collision"],
                "terminal_reason": row["terminal_reason"],
                "yaw_error_deg": y,
                "t_norm": t_norm
            })
            return df

        # Interpolate onto a common normalized grid for aggregation
        t_norm_grid = np.linspace(0.0, 1.0, n_interp)
        # Map normalized grid to original indices
        x_src = (t_idx / max(1, (y.size - 1))).astype(float)

        # Clean y for interpolation (replace non-finite with nearest finite)
        if not np.isfinite(y).all():
            # Simple forward/back fill on the 1D array
            y_clean = y.copy()
            # forward fill
            for i in range(1, y_clean.size):
                if not np.isfinite(y_clean[i]):
                    y_clean[i] = y_clean[i-1]
            # backward fill
            for i in range(y_clean.size - 2, -1, -1):
                if not np.isfinite(y_clean[i]):
                    y_clean[i] = y_clean[i+1]
            y = y_clean

        y_interp = np.interp(t_norm_grid, x_src, y)

        df = pd.DataFrame({
            "experiment_id": row["experiment_id"],
            "scene": row["scene"],
            "objective": row["objective"],
            "pilot": row["pilot"],
            "trajectory_index": row["trajectory_index"],
            "repetition": row["repetition"],
            "success": row["success"],
            "collision": row["collision"],
            "terminal_reason": row["terminal_reason"],
            "yaw_error_deg": y_interp,
            "t_norm": t_norm_grid
        })
        return df

    # Apply filters for failures if requested
    df_filtered = df_sim.copy()
    if not include_failures:
        df_filtered = df_filtered[(df_filtered["success"] == True) & (df_filtered["collision"] == False)]

    # Build long-form (interpolate to common grid for nice aggregation)
    long_parts = []
    for _, row in df_filtered.iterrows():
        df_part = _row_series_to_long(row, n_interp=101)  # set to None to disable interpolation
        if df_part is not None:
            long_parts.append(df_part)

    if not long_parts:
        print("No yaw-error series available after filtering.")
        return None

    df_long = pd.concat(long_parts, ignore_index=True)

    # =============================
    # Plot helpers
    # =============================
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="deep")

    def _plot_group_time_series(df_group, title, save_name, show_pilot_means=True):
        """
        df_group: long-form rows for a *single* group (experiment_id or objective),
                  with columns [t_norm, yaw_error_deg, pilot, trajectory_index, repetition]
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 1) Light per-trial lines (all pilots)
        #    Make semi-transparent to show density
        #    Group by unique trajectory instance to avoid mixing repetitions
        for (pilot, traj_idx, rep), sub in df_group.groupby(["pilot", "trajectory_index", "repetition"], sort=False):
            ax.plot(sub["t_norm"], sub["yaw_error_deg"], alpha=0.15, linewidth=1)

        # 2) Pilot-wise means (optional)
        if show_pilot_means:
            for pilot, sub_p in df_group.groupby("pilot", sort=False):
                # Mean across all trials for this pilot at each t_norm
                pilot_mean = sub_p.groupby("t_norm", sort=True)["yaw_error_deg"].mean()
                ax.plot(pilot_mean.index.values, pilot_mean.values, linewidth=2, label=f"{pilot} mean")

        # 3) Overall mean ± 95% CI
        grouped = df_group.groupby("t_norm", sort=True)["yaw_error_deg"]
        mean = grouped.mean()
        sem = grouped.sem()  # standard error of mean
        # Use normal approx for 95% CI
        ci = 1.96 * sem.fillna(0.0)
        ax.plot(mean.index.values, mean.values, linewidth=3, label="All pilots mean", zorder=5)
        ax.fill_between(mean.index.values, mean.values - ci, mean.values + ci, alpha=0.25, label="95% CI", zorder=4)

        ax.set_title(title)
        ax.set_xlabel("Normalized time in trajectory (0 → 1)")
        ax.set_ylabel("Yaw error (degrees)")
        ax.set_xlim(0, 1)
        ax.grid(True, which="both", axis="both", alpha=0.3)

        # Legend (only if pilot means are shown and not too crowded)
        if show_pilot_means and df_group["pilot"].nunique() <= 8:
            ax.legend(loc="upper right", frameon=True)
        else:
            # Always keep the All pilots mean + CI visible
            handles, labels = ax.get_legend_handles_labels()
            keep = [i for i, lab in enumerate(labels) if lab in ("All pilots mean", "95% CI")]
            if keep:
                ax.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="upper right", frameon=True)

        # Save
        os.makedirs(visualizations_dir, exist_ok=True)
        out_path = os.path.join(visualizations_dir, save_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
        plt.close(fig)

    # =============================
    # PLOT: By Experiment (scene + objective)
    # =============================
    for exp_id, sub in df_long.groupby("experiment_id", sort=False):
        scene_lbl = sub["scene"].iloc[0]
        obj_lbl = sub["objective"].iloc[0]
        title = f"Yaw Error Time Series — Experiment: {exp_id} (scene={scene_lbl}, query={obj_lbl})"
        fname = f"yaw_error_by_experiment_{exp_id}.png"
        _plot_group_time_series(sub, title, fname, show_pilot_means=True)

    # =============================
    # PLOT: By Query (objective across scenes)
    # =============================
    for obj, sub in df_long.groupby("objective", sort=False):
        # If you want to compare pilots on a single plot per objective, leave as-is.
        title = f"Yaw Error Time Series — By Query: {obj}"
        fname = f"yaw_error_by_query_{obj}.png"
        _plot_group_time_series(sub, title, fname, show_pilot_means=True)

    # (Optional) If you want separate failure analysis overlays, you can slice df_long
    # and re-run the same plotting, e.g. only-collisions vs non-collisions:
    if separate_failure_analysis:
        if df_long["collision"].any():
            sub_fail = df_long[df_long["collision"] == True]
            for exp_id, sub in sub_fail.groupby("experiment_id", sort=False):
                scene_lbl = sub["scene"].iloc[0]
                obj_lbl = sub["objective"].iloc[0]
                title = f"[FAILURES ONLY] Yaw Error — Experiment: {exp_id} (scene={scene_lbl}, query={obj_lbl})"
                fname = f"yaw_error_by_experiment_{exp_id}_FAILURES.png"
                _plot_group_time_series(sub, title, fname, show_pilot_means=False)

        sub_nonfail = df_long[(df_long["collision"] == False)]
        if not sub_nonfail.empty:
            for exp_id, sub in sub_nonfail.groupby("experiment_id", sort=False):
                scene_lbl = sub["scene"].iloc[0]
                obj_lbl = sub["objective"].iloc[0]
                title = f"[NON-FAILURES] Yaw Error — Experiment: {exp_id} (scene={scene_lbl}, query={obj_lbl})"
                fname = f"yaw_error_by_experiment_{exp_id}_NONFAIL.png"
                _plot_group_time_series(sub, title, fname, show_pilot_means=True)

    print("Finished generating yaw-error time-history plots.")
    return {
        "df_long": df_long,            # long-form data for further analyses
        "experiments": sorted(df_long["experiment_id"].unique()),
        "objectives": sorted(df_long["objective"].unique()),
        "pilots": sorted(df_long["pilot"].unique())
    }


def clopper_pearson_ci(x, n, alpha=0.05):
    """
    Calculate Clopper-Pearson exact binomial confidence interval.
    
    Parameters:
        x: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval
    """
    from scipy import stats
    
    if n == 0:
        return (0.0, 1.0)
    
    if x == 0:
        lower = 0.0
        upper = 1 - (alpha/2)**(1/n)
    elif x == n:
        lower = (alpha/2)**(1/n)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha/2, x, n-x+1)
        upper = stats.beta.ppf(1-alpha/2, x+1, n-x)
    
    return (lower, upper)


def create_performance_column_charts(df_sim, cohort_name, base_save_path=None):
    """
    Create column charts for success rates and query-in-view rates with 95% confidence intervals.
    
    Parameters:
        df_sim: DataFrame with simulation results
        cohort_name: Name of the cohort for titles
        base_save_path: Optional base path for saving plots (will append suffix)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import beta
    sns.set_theme(style="whitegrid",palette="deep")
    
    # Calculate success rates and confidence intervals by pilot
    pilot_stats = []
    
    # Get pilots in consistent order (same as violin plots)
    unique_pilots = sorted(df_sim['pilot'].unique())  # Sort for consistency
    
    for pilot in unique_pilots:
        pilot_data = df_sim[df_sim['pilot'] == pilot]
        n_total = len(pilot_data)
        
        # Success rate
        n_success = pilot_data['success'].sum()
        success_rate = n_success / n_total if n_total > 0 else 0
        success_ci = clopper_pearson_ci(n_success, n_total)
        
        # Query-in-view rate (goal_in_camera_fov)
        if 'goal_in_camera_fov' in pilot_data.columns:
            n_in_view = pilot_data['goal_in_camera_fov'].sum()
            in_view_rate = n_in_view / n_total if n_total > 0 else 0
            in_view_ci = clopper_pearson_ci(n_in_view, n_total)
        else:
            n_in_view = 0
            in_view_rate = 0
            in_view_ci = (0, 0)
        
        pilot_stats.append({
            'pilot': pilot,
            'n_total': n_total,
            'n_success': n_success,
            'success_rate': success_rate,
            'success_ci_lower': success_ci[0],
            'success_ci_upper': success_ci[1],
            'n_in_view': n_in_view,
            'in_view_rate': in_view_rate,
            'in_view_ci_lower': in_view_ci[0],
            'in_view_ci_upper': in_view_ci[1]
        })
    
    df_stats = pd.DataFrame(pilot_stats)
    
    # Create figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Success Rate Chart
    x_pos = np.arange(len(df_stats))
    
    # Calculate error bars (asymmetric)
    success_yerr_lower = df_stats['success_rate'] - df_stats['success_ci_lower']
    success_yerr_upper = df_stats['success_ci_upper'] - df_stats['success_rate']
    success_yerr = np.array([success_yerr_lower, success_yerr_upper])
    
    
    # Using seaborn barplot instead of matplotlib bar
    bars1 = sns.barplot(data=df_stats, x='pilot', y='success_rate', 
                    #    palette=["#e9ae0d" if pilot == 'expert' else '#2690db' 
                       palette=["#2690db" if pilot == 'expert' else "#FF6600"#"#35db26" 
                               for pilot in df_stats['pilot']], 
                               alpha=0.8, ax=ax1)
    # Add error bars manually since seaborn barplot doesn't support asymmetric error bars
    for i, (pilot, rate, lower, upper) in enumerate(zip(df_stats['pilot'], df_stats['success_rate'], 
                                                       df_stats['success_ci_lower'], df_stats['success_ci_upper'])):
        ax1.errorbar(i, rate, yerr=[[rate - lower], [upper - rate]], 
                    fmt='none', capsize=5, color="#505050", capthick=1)
        
    # # Using seaborn barplot instead of matplotlib bar
    # bars1 = sns.barplot(data=df_stats, x='pilot', y='success_rate', 
    #                    palette=["#e9ae0d" if pilot == 'expert' else '#2690db' 
    #                            for pilot in df_stats['pilot']], ax=ax1)
    # # Add error bars manually since seaborn barplot doesn't support asymmetric error bars
    # for i, (pilot, rate, lower, upper) in enumerate(zip(df_stats['pilot'], df_stats['success_rate'], 
    #                                                    df_stats['success_ci_lower'], df_stats['success_ci_upper'])):
    #     ax1.errorbar(i, rate, yerr=[[rate - lower], [upper - rate]], 
    #                 fmt='none', capsize=5, color='#414141', capthick=1)
    
    ax1.set_xlabel('Pilot', fontweight='bold')
    ax1.set_ylabel('Success Rate', fontweight='bold')
    ax1.set_title('Success Rate by Pilot\n(95% CI)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_stats['pilot'], rotation=45)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add sample size annotations
    for i, (n_total, n_success, rate) in enumerate(zip(df_stats['n_total'], df_stats['n_success'], df_stats['success_rate'])):
        ax1.text(i, rate / 2,
                f'n={n_total}\n({n_success} success)',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Query-in-View Rate Chart
    if df_stats['in_view_rate'].sum() > 0:  # Only create if we have data
        in_view_yerr_lower = df_stats['in_view_rate'] - df_stats['in_view_ci_lower']
        in_view_yerr_upper = df_stats['in_view_ci_upper'] - df_stats['in_view_rate']
        in_view_yerr = np.array([in_view_yerr_lower, in_view_yerr_upper])
        
        # Using seaborn barplot instead of matplotlib bar
        bars2 = sns.barplot(data=df_stats, x='pilot', y='in_view_rate', 
                        #    palette=["#e9ae0d" if pilot == 'expert' else '#2690db' 
                        palette=["#2690db" if pilot == 'expert' else "#FF6600" 
                                for pilot in df_stats['pilot']], 
                                alpha=0.8, ax=ax2)
        # Add error bars manually since seaborn barplot doesn't support asymmetric error bars
        for i, (pilot, rate, lower, upper) in enumerate(zip(df_stats['pilot'], df_stats['in_view_rate'], 
                                                        df_stats['in_view_ci_lower'], df_stats['in_view_ci_upper'])):
            ax2.errorbar(i, rate, yerr=[[rate - lower], [upper - rate]], 
                        fmt='none', capsize=5, color="#505050", capthick=1)
            
        # bars2 = ax2.bar(x_pos, df_stats['in_view_rate'], 
        #                 yerr=in_view_yerr, capsize=5,
        #                 color=["#e9ae0d" if pilot == 'expert' else "#2690db" 
        #                        for pilot in df_stats['pilot']],
        #                 alpha=0.8, edgecolor='gray', linewidth=1)
        
        ax2.set_xlabel('Pilot', fontweight='bold')
        ax2.set_ylabel('Query-in-View Rate', fontweight='bold')
        ax2.set_title('Query-in-View Rate by Pilot\n(95% CI)', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df_stats['pilot'], rotation=45)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add sample size annotations
        # for i, (n_total, n_success, rate) in enumerate(zip(df_stats[''], df_stats['n_in_view'], df_stats['n_in_view'])):
        #     ax2.text(i, rate / 2,
        #             f'n={n_total}\n({n_success} in view)',
        #             ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        for i, (n_total, n_in_view, rate) in enumerate(zip(df_stats['n_total'], df_stats['n_in_view'], df_stats['in_view_rate'])):
            ax2.text(i, rate / 2,
                f'n={n_total}\n({n_in_view} in view)',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    else:
        ax2.text(0.5, 0.5, 'No Query-in-View Data Available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, style='italic')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Query-in-View Rate by Pilot\n(No Data Available)', fontweight='bold')
    
    plt.suptitle(f'Performance Comparison - {cohort_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if base_save_path:
        # Remove extension if present and add our suffix
        if base_save_path.endswith('.png'):
            base_save_path = base_save_path[:-4]
        
        column_chart_path = f"{base_save_path}_column_charts.png"
        plt.savefig(column_chart_path, dpi=300, bbox_inches='tight')
        print(f"Column charts saved to: {column_chart_path}")
    
    plt.show()
    
    # Print statistical summary
    print(f"\n" + "="*70)
    print("COLUMN CHART STATISTICS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    
    print("\n1. SUCCESS RATES:")
    for _, row in df_stats.iterrows():
        ci_width = row['success_ci_upper'] - row['success_ci_lower']
        print(f"   {row['pilot']:15s}: {row['success_rate']:.1%} "
              f"[{row['success_ci_lower']:.1%}, {row['success_ci_upper']:.1%}] "
              f"(n={row['n_total']}, CI width: {ci_width:.1%})")
    
    if df_stats['in_view_rate'].sum() > 0:
        print("\n2. QUERY-IN-VIEW RATES:")
        for _, row in df_stats.iterrows():
            ci_width = row['in_view_ci_upper'] - row['in_view_ci_lower']
            print(f"   {row['pilot']:15s}: {row['in_view_rate']:.1%} "
                  f"[{row['in_view_ci_lower']:.1%}, {row['in_view_ci_upper']:.1%}] "
                  f"(n={row['n_total']}, CI width: {ci_width:.1%})")
    else:
        print("\n2. QUERY-IN-VIEW RATES: No data available")
    
    print("\n💡 INTERPRETATION:")
    print("  • Error bars show 95% Clopper-Pearson exact binomial confidence intervals")
    print("  • Wider intervals indicate smaller sample sizes or rates near 0.5")
    print("  • Non-overlapping intervals suggest statistically significant differences")
    print("  • Sample sizes (n) are shown above each bar")