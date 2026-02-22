import os
import json
import yaml
import pickle
import gc
from datetime import datetime
from typing import List, Union, Literal, Tuple
from re import T, X

# CRITICAL: Set matplotlib to non-interactive backend BEFORE any other imports
# This prevents figures from being created in browser (memory leak)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

import numpy as np
import torch
from torchvision.io import write_video
from torchvision.transforms import Resize
from acados_template import AcadosSimSolver

# from synthesize.solvers import min_snap as ms
# import synthesize.nerf_utils as nf
# import synthesize.trajectory_helper as th
# import synthesize.generate_data as gd
# from synthesize.build_rrt_dataset import get_objectives, generate_rrt_paths

from sousvide.control.pilot import Pilot
from figs.control.vehicle_rate_mpc import VehicleRateMPC
# from figs.tsplines import min_snap as ms
# import sousvide.synthesize.synthesize_helper as sh
import sousvide.synthesize.rollout_generator as gd
import sousvide.visualize.record_flight as rf
from torchvision.io import write_video
from torchvision.transforms import Resize
from figs.simulator import Simulator
import figs.utilities.trajectory_helper as th
from figs.utilities.display_config import get_figure_display
from figs.dynamics.model_specifications import generate_specifications
# import figs.visualize.generate_videos as gv

import figs.tsampling.build_rrt_dataset as bd
import figs.scene_editing.scene_editing_utils as scdt
import figs.visualize.rich_visuals as rv

# import visualize.plot_synthesize as ps
# import visualize.record_flight as rf

# import sousvide.flight.vision_preprocess as vp
# import sousvide.flight.vision_preprocess_alternate as vp
from sousvide.flight.vision_processor_base import create_vision_processor
from sousvide.visualize.analyze_simulated_experiments import analyze_trajectory_performance, compute_aggregate_statistics, load_trajectory_analysis


def simulate_trajectory():
    # Load trajectory dataset from files
    print("Review mode enabled. Loading trajectory dataset from files.")
    trajectory_dataset = {}
    for scene_name, course_name in flights:
        # Generate simulator
        simulator = Simulator(scene_name,rollout_name)

        scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
        combined_prefix = os.path.join(scenes_cfg_dir, scene_name)
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)
        objectives      = scene_cfg["queries"]
        print(f"Objectives for {scene_name}: {objectives}")

        for objective in objectives:
            # Use pkl_dir if provided, otherwise fall back to scenes_cfg_dir
            pkl_base_dir = pkl_dir if pkl_dir is not None else scenes_cfg_dir
            combined_prefix = os.path.join(pkl_base_dir, scene_name)
            if filename is not None:
                combined_file_path = filename
            else:
                combined_file_path = f"{combined_prefix}_{objective}.pkl"
            with open(combined_file_path, "rb") as f:
                data = pickle.load(f)
                trajectory_dataset[objective] = data
        
        # Print trajectory dataset contents after loading all objectives
        print("Trajectory dataset contents:")
        for key, value in trajectory_dataset.items():
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: ndarray shape {v.shape}")
                elif isinstance(v, (list, dict)):
                    print(f"    {k}: {type(v).__name__} (length {len(v)})")
                else:
                    print(f"    {k}: {type(v).__name__} ({v})")


    # print(asdghasdgh)
    # === 8) Initialize Drone Config & Transform ===
    # base_cfg   = generate_specifications(base_frame_config)
    transform  = Resize((720, 1280), antialias=True)

    # === 9) Generate Drone Instances ===
    Frames = gd.generate_frames(
    Trep, base_frame_config, frame_set_config
    )

    # print(f"trajectory dataset: {list(trajectory_dataset.keys())}")
    # === 10) Simulation Loop: for each objective, for each pilot ===
    for obj_name, data in trajectory_dataset.items():
        tXUi   = data["tXUi"]
        print(f"txui shape: {tXUi.shape}")
        t0, tf = tXUi[0, 0], tXUi[0, -1]
        x0     = tXUi[1:11, 0]
        
        # prepend expert to pilots
        pilot_list = ["expert"] + roster

        # apply any perturbations
        Perturbations  = gd.generate_perturbations(
            Tsps=Trep,
            tXUi=tXUi,
            trajectory_set_config=test_set_config
        )

        for pilot_name in pilot_list:
            print("-" * 70)
            print(f"Simulating pilot '{pilot_name}' on objective '{obj_name}'")

            traj_file = os.path.join(
                trajectories_dir, f"sim_data_{scene_name}_{obj_name}_{pilot_name}.pt"
            )
            vid_file = os.path.join(
                videos_dir, f"sim_video_{scene_name}_{obj_name}_{pilot_name}.mp4"
            )

            if pilot_name == "expert":
                policy = VehicleRateMPC(tXUi,base_policy_name,base_frame_name,pilot_name)
            else:
                policy = Pilot(cohort_name,pilot_name)
                policy.set_mode('deploy')

            results = []
            for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):

                simulator.load_frame(frame)

                # Simulate Trajectory
            #FIXME
                # if obj_name.startswith("loiter_"):
                #     # For loiter trajectories, simulate with special loiter parameters
                #     print(f"simulating loiter trajectory with query: null")
                #     Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                #         policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),
                #         query="null",clipseg=vision_processor,verbose=verbose)
            #
                # else:
                # Normal simulation for non-loiter trajectories
                Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                    policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),
                    query=obj_name,vision_processor=vision_processor,verbose=verbose)
                # Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                #     policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),query=obj_name,clipseg=vision_processor)

                # Save Trajectory
                trajectory = {
                    "Tro":Tro,"Xro":Xro,"Uro":Uro,
                    "Iro":Iro,  # Rendered images (rgb, depth, semantic) for critic training
                    "tXUd":tXUi,"obj":np.zeros((18,1)),"Ndata":Uro.shape[1],"Tsol":Tsol,"Adv":Adv,
                    "rollout_id":method_name+"_"+str(idx).zfill(5),
                    "course":course_name,
                    "frame":frame}
                results.append(trajectory)

                # rollout = {
                #     "Tro": Tro, "Xro": Xro, "Uro": Uro,
                #     "Xid": tXUi[1:11, :],
                #     "obj": np.zeros((18, 1)),
                #     "Ndata": Uro.shape[1],
                #     "Tsol": Tsol, "Adv": Adv,
                #     "rollout_id": "acados_rollout",
                #     "course": scene_name,
                #     "drone": drone_name,
                #     "method": method_name,
                #     "objective": obj_name
                # }
                # results.append(rollout)
                # mpc_expert.clear_generated_code()

            # semantic_imgs = Iro["semantic"]

            # if visualize_rrt:
            #     combined_file_path = os.path.join(f"{combined_prefix}_{obj_name}.pkl")#f"{combined_prefix}_{obj_name}.pkl"
            #     with open(combined_file_path, 'rb') as f:
            #         trajectory_data = pickle.load(f)
            #         th.debug_figures_RRT(trajectory_data["obj_loc"],trajectory_data["positions"],trajectory_data["trajectory"],
            #                             trajectory_data["smooth_trajectory"],trajectory_data["times"])

            if vision_processor is not None:
                imgs = {
                    "semantic": Iro["semantic"],
                    "rgb": Iro["rgb"],
                    "depth": Iro["depth"]
                }
            else:
                imgs = {
                    "semantic": Iro["semantic"]
                }

            # run for each key in imgs
            for key in imgs:
                if use_flight_recorder:
                    fr = rf.FlightRecorder(
                        Xro.shape[0], Uro.shape[0],
                        20, tXUi[0, -1],
                        [224, 398, 3],
                        np.zeros((18, 1)),
                        cohort_name, scene_name, pilot_name
                    )
                    fr.simulation_import(
                        imgs[key], Tro, Xro, Uro, tXUi, Tsol, Adv
                    )
                    fr.save()
                else:
                    torch.save(results, traj_file)

                    # prepare and write video
                    frames = torch.zeros(
                        (imgs[key].shape[0], 720, 1280, 3)
                    )
                    imgs_t = torch.from_numpy(imgs[key])
                    for i in range(imgs_t.shape[0] - 1):
                        img = imgs_t[i].permute(2, 0, 1)
                        img = transform(img)
                        frames[i] = img.permute(1, 2, 0)

                    # save video with key in filename
                    key_vid_file = vid_file.replace('.mp4', f'_{key}.mp4')
                    write_video(key_vid_file, frames, fps=20)






def simulate_rollouts(
        workspace_path:str,
        cohort_name:str, cohort_path:str,
        method_name:str,
        pilot:Pilot,
        flights:List[Tuple[str,str]], scenes_cfg_dir:str, objectives_all:dict,
        max_trajectories:Union[int,None]=None,
        review:bool=False,
        disable_visualization:bool=False,
        show_progress:bool=True,
        progress_bar:tuple=None,  # Optional (progress, task) tuple from parent
):
    # Find the most recent simulation data directory
    simulation_data_base = os.path.join(cohort_path, "simulation_data")
    if not os.path.exists(simulation_data_base):
        print(f"Creating simulation_data directory: {simulation_data_base}")
        os.makedirs(simulation_data_base, exist_ok=True)

    # Get the most recent timestamped directory
    timestamped_dirs = [d for d in os.listdir(simulation_data_base)
                    if os.path.isdir(os.path.join(simulation_data_base, d))]

    if not timestamped_dirs or review:  # Fixed: review=True forces regeneration
        print("Running new simulations...")
        simulate_roster(
            cohort_name=cohort_name,
            method_name=method_name,
            flights=flights,
            roster=[pilot.name],
        )

        # Re-scan for timestamped directories after simulate_roster creates new data
        timestamped_dirs = [d for d in os.listdir(simulation_data_base)
                        if os.path.isdir(os.path.join(simulation_data_base, d))]

    latest_dir = max(timestamped_dirs)
    simulation_data_dir = os.path.join(simulation_data_base, latest_dir)
    print(f"Loading simulation data from: {simulation_data_dir}")
    # if simulate:
    trajectories_dir = os.path.join(simulation_data_dir, "trajectories")
    videos_dir = os.path.join(simulation_data_dir, "videos")
    rrt_data_dir = os.path.join(simulation_data_dir, "rrt_planning")
    visualizations_dir = os.path.join(simulation_data_dir, "visualizations")
    
    for directory in [trajectories_dir, videos_dir, rrt_data_dir, visualizations_dir]:
        os.makedirs(directory, exist_ok=True)

    # If simulating, load additional configurations
    method_path = os.path.join(workspace_path, "configs", "method", method_name + ".json")
    with open(method_path) as json_file:
        method_config = json.load(json_file)
    # if simulate:
    test_set_config = method_config["test_set"]
    trajectory_set_config = method_config["trajectory_set"]
    frame_set_config = method_config["frame_set"]

    sample_set_config = method_config["sample_set"]
    rollout_name = sample_set_config["rollout"]
    
    base_policy_name = sample_set_config["policy"]
    base_frame_name = sample_set_config["frame"]
    vision_processor_type = sample_set_config.get("vision_processor", "none")
    Nrep = test_set_config["reps"]

    # Load perception config for vision processor (for ONNX paths if needed)
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

    # Create vision processor using factory function
    if vision_processor_type.lower() != 'none':
        print(f"Image Processing Model set to {vision_processor_type.upper()}")
        processor_kwargs = {}

        # Add ONNX path for CLIPSeg if available
        if vision_processor_type.lower() == 'clipseg' and onnx_model_path is not None:
            processor_kwargs['onnx_model_path'] = onnx_model_path
            print(f"Using ONNX model for CLIPSeg at {onnx_model_path}")

        vision_processor = create_vision_processor(vision_processor_type, **processor_kwargs)
    else:
        print("Vision processor disabled (set to 'none')")
        vision_processor = None
    
    # Initialize video transform
    transform = Resize((720, 1280), antialias=True)
    
    # Generate base drone specifications
    base_frame_specs = generate_specifications(base_frame_config)
    
    # Compute simulation variables
    Trep = np.zeros(Nrep)
    
    for scene_name, course_name in flights:
        # Load scene configuration
        scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)

        # Override visualization flag if requested (for RL training)
        if disable_visualization:
            scene_cfg["visualize"] = False
        
        objectives = objectives_all[scene_name]
        radii = scene_cfg["radii"]
        altitudes = scene_cfg["altitudes"]
        similarities = scene_cfg.get("similarities", None)
        
        # if obj_name not in objectives:
        #     print(f"Objective '{obj_name}' not found in scene '{scene_name}'. Available: {objectives}")
        #     continue
        
        # obj_index = objectives.index(obj_name)
        
        # Generate simulator to get environment data
        rollout_name = sample_set_config["rollout"]
        simulator = Simulator(scene_name, rollout_name)
        
        # Get objectives and environment data (same as in deploy_ssv.py)
        obj_targets, _, epcds_list, epcds_arr = bd.get_objectives(
            simulator.gsplat, objectives, similarities, False
        )
        
        for obj_index, obj_name in enumerate(objectives):
            print(f"Simulating '{scene_name}', objective '{obj_name}'")
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
            # if simulate:
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
            
            # Use passed-in progress bar or create new one if enabled
            from contextlib import nullcontext
            
            if progress_bar is not None:
                # Use existing progress bar from parent (RL training)
                progress_ctx, parent_task = progress_bar
                num_trajectories = len(all_debug_info)
                use_existing_progress = True
                
                # Add a subtask to parent's progress for trajectory simulation
                # CRITICAL: Must include ALL fields that parent's Progress expects!
                # Copy all fields from parent task to ensure compatibility
                task_fields = {}
                parent_task_obj = progress_ctx._tasks[parent_task]
                for field_name, field_value in parent_task_obj.fields.items():
                    # Copy field with appropriate default value based on type
                    if isinstance(field_value, (int, float)):
                        task_fields[field_name] = 0.0
                    else:
                        task_fields[field_name] = field_value
                
                # Override units field for trajectories
                task_fields['units'] = 'steps'
                
                traj_task = progress_ctx.add_task(
                    "[cyan]Simulating trajectories[/]",
                    total=num_trajectories,
                    visible=True,
                    **task_fields
                )
                # Don't use context manager - parent's Progress is already active
            elif show_progress:
                # Create new progress bar (standalone mode)
                # Only create if no parent progress exists
                progress_ctx = rv.get_generation_progress()
                num_trajectories = len(all_debug_info)
                use_existing_progress = False
                traj_task = None
            else:
                # No progress bar at all
                progress_ctx = None
                traj_task = None
                num_trajectories = len(all_debug_info)
                use_existing_progress = False
            
            # Only enter context if we created a new progress bar
            if progress_bar is None and show_progress:
                progress_context = progress_ctx
            else:
                progress_context = nullcontext()
            
            with progress_context:
                # Simulate each parameterized trajectory
                for debug_idx, (debug_info, tXUi) in enumerate(zip(all_debug_info, all_trajectories)):
                    if not show_progress:
                        print(f"Simulating trajectory {debug_idx + 1}/{len(all_debug_info)}")
                    
                    # Get trajectory data - tXUi is the parameterized trajectory 
                    
                    # Create trajectory dataset entry similar to simulate_roster
                    trajectory_data = {
                        "tXUi": tXUi,
                        **debug_info
                    }
                    
                    # Simulation parameters
                    t0, tf = tXUi[0, 0], tXUi[0, -1]
                    x0 = tXUi[1:11, 0]
                    
                    # Pilot list (expert + roster)
                    pilot_list = [pilot.name] + ["expert"]
                    
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
                        if not show_progress:
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
                            
                            # Create progress callback if progress is enabled
                            sim_progress_callback = None
                            if show_progress and progress_ctx is not None:
                                # Calculate expected control steps for this trajectory
                                t0_sim, tf_sim = perturbation["t0"], tXUi[0, -1]
                                hz_ctl = 20  # Default control frequency
                                expected_steps = int((tf_sim - t0_sim) * hz_ctl)
                                
                                # Create/update simulation task
                                # Check if this is a training progress bar (has 'loss' field) or generation progress bar
                                # Training progress bars need loss field, generation progress bars don't
                                task_fields = {"units": "steps"}
                                # Check if parent task has 'loss' field (indicates training progress bar)
                                if use_existing_progress and parent_task is not None:
                                    parent_task_obj = progress_ctx._tasks[parent_task]
                                    if 'loss' in parent_task_obj.fields:
                                        task_fields["loss"] = 0.0  # Add loss field for training progress bars
                                
                                if traj_task is None:
                                    traj_task = progress_ctx.add_task(
                                        f"[cyan]Traj {debug_idx+1}/{num_trajectories} | {pilot_name}",
                                        total=expected_steps, **task_fields
                                    )
                                else:
                                    progress_ctx.update(traj_task, 
                                        description=f"[cyan]Traj {debug_idx+1}/{num_trajectories} | {pilot_name}",
                                        completed=0, total=expected_steps)
                                
                                last_step = [-1]
                                def sim_progress_callback(current_step, total_steps, sim_time):
                                    if current_step > last_step[0]:
                                        progress_ctx.update(traj_task, completed=current_step)
                                        progress_ctx.refresh()
                                        last_step[0] = current_step
                            
                            # Simulate trajectory
                            Tro, Xro, Uro, Iro, Tsol, Adv = simulator.simulate(
                                policy, perturbation["t0"], tXUi[0, -1], perturbation["x0"],
                                np.zeros((18, 1)), query=obj_name, vision_processor=vision_processor, verbose=False,
                                progress_callback=sim_progress_callback
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
                                "Iro": Iro,  # Rendered images (rgb, depth, semantic) for critic training
                                "tXUd": tXUi, "obj": np.zeros((18, 1)), "Ndata": Uro.shape[1],
                                "Tsol": Tsol, "Adv": Adv,
                                "rollout_id": method_name + "_" + str(idx).zfill(5),
                                "course": course_name,
                                "objective": obj_name,  # Add objective name for validation grouping
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
                        
                        if not show_progress:
                            print(f"    Saved simulation results for pilot '{pilot_name}'")
                    
                    # Store this trajectory's analysis across all pilots
                    all_trajectory_analyses.append({
                        "trajectory_index": debug_idx,
                        "pilot_analyses": trajectory_pilot_analyses
                    })
                    
                    # Update trajectory progress if using progress bar
                    if traj_task is not None and progress_ctx is not None:
                        progress_ctx.update(traj_task, advance=1)
                    
                    # CRITICAL: Free GPU memory after each trajectory to prevent OOM
                    # Delete large tensors and clear CUDA cache
                    if 'Iro' in locals():
                        del Iro
                    if 'Xro' in locals():
                        del Xro
                    if 'Uro' in locals():
                        del Uro
                    if 'imgs_t' in locals():
                        del imgs_t
                    if 'frames' in locals():
                        del frames
                    
                    # CRITICAL: Close all matplotlib figures to prevent browser memory leak
                    plt.close('all')
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
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
    
    # ====================================================================
    # MEMORY CLEANUP: Free vision processor and other heavy objects
    # ====================================================================
    if vision_processor is not None:
        # Delete the vision processor to free model memory (especially CLIPSeg)
        del vision_processor
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()

def simulate_roster(cohort_name:str,method_name:str,
                    flights:List[Tuple[str,str]],
                    roster:List[str],
                    use_flight_recorder:bool=False,
                    review:bool=False,
                    filename:str=None,
                    pkl_dir:str=None,
                    verbose:bool=False):
                    # visualize_rrt:bool=False):
    
    # Some useful path(s)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Extract method configs
    method_path = os.path.join(workspace_path,"configs","method",method_name+".json")

    # Extract scene configs
    scenes_cfg_dir  = os.path.join(workspace_path, "configs", "scenes")

    # Set course config path
    course_cfg_path = os.path.join(workspace_path, "configs", "course")

    # Extract Perception configs
    perception_cfg_dir = os.path.join(workspace_path, "configs", "perception")
    with open(os.path.join(perception_cfg_dir, "onnx_benchmark_config.json")) as json_file:
        perception_config = json.load(json_file)
    onnx_model_path = perception_config.get("onnx_model_path", None)
        
    with open(method_path) as json_file:
        method_config = json.load(json_file)

    test_set_config = method_config["test_set"]
    sample_set_config = method_config["sample_set"]
    trajectory_set_config = method_config["trajectory_set"]
    frame_set_config = method_config["frame_set"]

    base_policy_name = sample_set_config["policy"]
    base_frame_name = sample_set_config["frame"]

    # rollout_name = test_set_config["rollout"]
    Nrep = test_set_config["reps"]
    # trajectory_set_config = test_set_config["trajectory_set"]
    # frame_set_config = test_set_config["frame_set"]

    # Compute simulation variables
    Trep = np.zeros(Nrep)
    
    # sample_set_config = method_config["sample_set"]

    rrt_mode = sample_set_config["rrt_mode"]
    loitering = sample_set_config["loitering"]
    Tdt_ro = sample_set_config["duration"]
    Nro_tp = sample_set_config["reps"]
    Ntp_sc = sample_set_config["rate"]
    err_tol = sample_set_config["tolerance"]
    rollout_name = sample_set_config["rollout"]
    policy_name = sample_set_config["policy"]
    frame_name = sample_set_config["frame"]

    # Get vision processor type from config (default to 'none' if not specified)
    vision_processor_type = sample_set_config.get("vision_processor", "none")

    # Extract policy and frame
    policy_path = os.path.join(workspace_path,"configs","policy",policy_name+".json")
    frame_path  = os.path.join(workspace_path,"configs","frame",frame_name+".json")

    with open(policy_path) as json_file:
        policy_config = json.load(json_file)

    with open(frame_path) as json_file:
        base_frame_config = json.load(json_file)

    hz_ctl = policy_config["hz"]

    # Create cohort folder
    cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)

    if not os.path.exists(cohort_path):
        os.makedirs(cohort_path)

    # Generate base drone specifications
    base_frame_specs = generate_specifications(base_frame_config)

    # Create vision processor using factory function
    if vision_processor_type.lower() != 'none':
        print(f"Initializing vision processor: {vision_processor_type.upper()}")
        processor_kwargs = {}

        # Add ONNX path for CLIPSeg if available
        if vision_processor_type.lower() == 'clipseg' and onnx_model_path is not None:
            processor_kwargs['onnx_model_path'] = onnx_model_path
            print(f"Using ONNX model for CLIPSeg at {onnx_model_path}")

        vision_processor = create_vision_processor(vision_processor_type, **processor_kwargs)
    else:
        print("Vision processor disabled (set to 'none')")
        vision_processor = None

    # Print some useful information
    print("==========================================================================")
    print("Cohort         :",cohort_name)
    print("Method         :",method_name)
    print("Policy         :",policy_name)
    print("Frame          :",frame_name)
    print("Flights        :",flights)

    # Create timestamped simulation data directory with organized subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_data_dir = os.path.join(cohort_path, "simulation_data", timestamp)
    
    # Create subdirectories for different file types
    rrt_data_dir = os.path.join(simulation_data_dir, "rrt_planning")      # .pkl files (RRT trees, filtered paths)
    trajectories_dir = os.path.join(simulation_data_dir, "trajectories")  # .pt files (simulation data)
    videos_dir = os.path.join(simulation_data_dir, "videos")              # .mp4 files (simulation videos)
    
    for directory in [simulation_data_dir, rrt_data_dir, trajectories_dir, videos_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print(f"Saving simulation data to: {simulation_data_dir}")
    print(f"  RRT planning data: {rrt_data_dir}")
    print(f"  Trajectory data: {trajectories_dir}")
    print(f"  Videos: {videos_dir}")

    if not review:
        if rrt_mode:
            trajectory_dataset = {}
            for scene_name,course_name in flights:
                scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
                combined_prefix = os.path.join(scenes_cfg_dir, scene_name)
                with open(scene_cfg_file) as f:
                    scene_cfg = yaml.safe_load(f)

                objectives      = scene_cfg["queries"]
                radii           = scene_cfg["radii"]
                n_branches      = scene_cfg["nbranches"]
                hover_mode      = scene_cfg["hoverMode"]
                visualize_flag = scene_cfg["visualize"]
                altitudes       = scene_cfg["altitudes"]
                similarities    = scene_cfg.get("similarities", None)
                num_trajectories = scene_cfg.get("numTraj", "all")
                n_iter_rrt = scene_cfg["N"]
                env_bounds      = {}
                if "minbound" in scene_cfg and "maxbound" in scene_cfg:
                    env_bounds["minbound"] = np.array(scene_cfg["minbound"])
                    env_bounds["maxbound"] = np.array(scene_cfg["maxbound"])

                # Generate simulator
                simulator = Simulator(scene_name,rollout_name)

                # RRT-based trajectories
                obj_targets, _, epcds_list, epcds_arr = bd.get_objectives(
                    simulator.gsplat, objectives, similarities, visualize_flag
                )

                # Goal poses and centroids
                goal_poses, obj_centroids = th.process_RRT_objectives(
                    obj_targets, epcds_arr, env_bounds, radii, altitudes
                )

                # Obstacle centroids and rings
#FIXME
                # if loitering:
                rings, obstacles = th.process_obstacle_clusters_and_sample(
                    epcds_arr, env_bounds)
                print(f"obstacles poses : {obstacles}")
                print(f"rings poses shape: {len(rings)}")
                # Generate RRT paths
                raw_rrt_paths = bd.generate_rrt_paths(
                    scene_cfg_file, simulator, epcds_list, epcds_arr, objectives,
                    goal_poses, obj_centroids, env_bounds, rings, obstacles, n_iter_rrt,
                    viz=visualize_flag
                )
                total_paths = sum(len(paths) for paths in raw_rrt_paths.values())
                print(f"Total number of raw_rrt_paths: {total_paths}")
            #FIXME
                # Save the complete RRT tree for each objective
                for obj_name in objectives:
                    rrt_tree_file = os.path.join(rrt_data_dir, f"{scene_name}_rrt_tree_{obj_name}.pkl")
                    with open(rrt_tree_file, "wb") as f:
                        pickle.dump(raw_rrt_paths[obj_name], f)
                    print(f"Saved complete RRT tree for {obj_name}: {len(raw_rrt_paths[obj_name])} paths")
            #
                # else:
                #     # Generate RRT paths
                #     raw_rrt_paths = bd.generate_rrt_paths(
                #         scene_cfg_file, simulator, epcds_list, epcds_arr, objectives,
                #         goal_poses, obj_centroids, env_bounds, Niter_RRT=n_iter_rrt
                #     )
#
                # Filter and parameterize trajectories
                all_trajectories = {}
                raw_filtered = {}
                for i, obj_name in enumerate(objectives):
                    print(f"Processing objective: {obj_name}")
                    branches = raw_rrt_paths[obj_name]
                    alt_set  = th.set_RRT_altitude(branches, altitudes[i])
                    filtered = th.filter_branches(alt_set, n_branches[i], hover_mode)
                    raw_filtered[obj_name] = filtered
                    print(f"{obj_name}: {len(filtered)} branches")
                #FIXME
                    # Save all filtered trajectories
                    filtered_file = os.path.join(rrt_data_dir, f"{scene_name}_filtered_{obj_name}.pkl")
                    with open(filtered_file, "wb") as f:
                        pickle.dump(filtered, f)
                    print(f"Saved {len(filtered)} filtered trajectories for {obj_name}")
                #
                    idx = np.random.randint(len(filtered))
                    print(f"Selected branch index for {obj_name}: {idx}")
                #NOTE this function behaves differently with randint=idx
                    traj_list, node_list, debug_info = th.parameterize_RRT_trajectories(
                        filtered, obj_centroids[i], 1.0, 20, randint=idx
                    )
                    print(f"Parameterized: {len(traj_list)} trajectories")
                    print(f"chosen_traj.shape: {traj_list[idx].shape}")

                    chosen_traj  = traj_list[idx]
                    chosen_nodes = node_list[idx]

                    if loitering:
                        Tsps = gd.compute_intervals(chosen_traj, Tdt_ro)
                        loiter_tXUd_list = gd.generate_loiter_trajectories(
                            tXUd_rrt         = chosen_traj,
                            Tpd              = chosen_traj[0],
                            Tsps             = Tsps,
                            base_frame_specs = base_frame_specs,
                            course_cfg_path  = course_cfg_path,
                            simulate         = True
                        )
                        # print(f"what on earth is loiter_tXUD? is it a list or an array? {type(loiter_tXUd)}")
                        # print(f"okay its a list, how big is it and what does it contain? {len(loiter_tXUd)} {loiter_tXUd[0].shape}")
                        loiter_tXUd = loiter_tXUd_list[0]
                        combined_data = {
                        "tXUi": loiter_tXUd,
                        "nodes": chosen_nodes,
                        **debug_info
                        }
                        combined_file = f"{combined_prefix}_loiter_{obj_name}.pkl"
                        with open(combined_file, "wb") as f:
                            pickle.dump(combined_data, f)
                        trajectory_dataset[obj_name] = combined_data
                        print(f"Saved trajectory dataset for {obj_name} and loiter_{obj_name}")
                    else:
                        combined_data = {
                            "tXUi": chosen_traj,
                            "nodes": chosen_nodes,
                            **debug_info
                        }

                        combined_file = f"{combined_prefix}_{obj_name}.pkl"
                        with open(combined_file, "wb") as f:
                            pickle.dump(combined_data, f)
                        
                        trajectory_dataset[obj_name] = combined_data
                        print(f"Saved trajectory dataset for {obj_name}")

                        if debug_info is not None and visualize_flag and get_figure_display():
                            # env_pcd_dict, env_pcd_array, env_bounds, env_pcd_o3d, env_pcd_mask, env_pcd_attr = scdt.rescale_point_cloud(simulator.gsplat, viz=False)
                            radius_info = {'r1': radii[i][0], 'r2': radii[i][1]}
                            fig = bd.visualize_single_trajectory(debug_info, epcds_list, obj_targets[i], radius_info, scene_cfg_file, simulator)
                            fig.show()
#FIXME                   
                    # for idx in range(len(obstacles)):
                    #     print(f"Rings for loiter: {[rings[idx]]}")
                    #     all_trajectories[f"loiter_{idx}"], _ = th.parameterize_RRT_trajectories(
                    #         [rings[idx]], obstacles[idx], constant_velocity=1.0, sampling_frequency=20, loiter=True)
                    # objectives.extend([f"null" for _ in range(len(obstacles))])
                    # idx = np.random.randint(len(obstacles))
                    # print(f"Selected loiter index for obstacles: {idx}")
                    # Take the first element of rings and create a new list with just that element
                    # rings_idx = [rings[idx]]
                    # print(f"Rings for loiter: {rings_idx}")
                    # traj_list, node_list, debug_info = th.parameterize_RRT_trajectories(
                    #     rings_idx, obstacles[idx], 1.0, 20, randint=idx, loiter=True)
                    # print(f"Parameterized: {len(traj_list)} loiter trajectories for obstacle {idx}")
                    # print(f"chosen_traj.shape: {traj_list[0].shape}")
                    # combined_data = {
                    #     "tXUi": traj_list[0],
                    #     "nodes": node_list[0],
                    #     **debug_info
                    # }
                    # combined_file = f"{combined_prefix}_loiter_{idx}.pkl"
                    # with open(combined_file, "wb") as f:
                    #     pickle.dump(combined_data, f)
                    # trajectory_dataset[f"loiter_{idx}"] = combined_data
#
    else:
        # Load trajectory dataset from files
        print("Review mode enabled. Loading trajectory dataset from files.")
        trajectory_dataset = {}
        for scene_name, course_name in flights:
            # Generate simulator
            simulator = Simulator(scene_name,rollout_name)

            scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
            combined_prefix = os.path.join(scenes_cfg_dir, scene_name)
            with open(scene_cfg_file) as f:
                scene_cfg = yaml.safe_load(f)
            objectives      = scene_cfg["queries"]
            print(f"Objectives for {scene_name}: {objectives}")

            for objective in objectives:
                # Use pkl_dir if provided, otherwise fall back to scenes_cfg_dir
                pkl_base_dir = pkl_dir if pkl_dir is not None else scenes_cfg_dir
                combined_prefix = os.path.join(pkl_base_dir, scene_name)
                if filename is not None:
                    combined_file_path = filename
                else:
                    combined_file_path = f"{combined_prefix}_{objective}.pkl"
                with open(combined_file_path, "rb") as f:
                    data = pickle.load(f)
                    trajectory_dataset[objective] = data
            
            # Print trajectory dataset contents after loading all objectives
            print("Trajectory dataset contents:")
            for key, value in trajectory_dataset.items():
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        print(f"    {k}: ndarray shape {v.shape}")
                    elif isinstance(v, (list, dict)):
                        print(f"    {k}: {type(v).__name__} (length {len(v)})")
                    else:
                        print(f"    {k}: {type(v).__name__} ({v})")
    # print(asdghasdgh)
    # === 8) Initialize Drone Config & Transform ===
    # base_cfg   = generate_specifications(base_frame_config)
    transform  = Resize((720, 1280), antialias=True)

    # === 9) Generate Drone Instances ===
    Frames = gd.generate_frames(
    Trep, base_frame_config, frame_set_config
    )

    # print(f"trajectory dataset: {list(trajectory_dataset.keys())}")
    # === 10) Simulation Loop: for each objective, for each pilot ===
    for obj_name, data in trajectory_dataset.items():
        tXUi   = data["tXUi"]
        print(f"txui shape: {tXUi.shape}")
        t0, tf = tXUi[0, 0], tXUi[0, -1]
        x0     = tXUi[1:11, 0]
        
        # prepend expert to pilots
        pilot_list = ["expert"] + roster

        # apply any perturbations
        Perturbations  = gd.generate_perturbations(
            Tsps=Trep,
            tXUi=tXUi,
            trajectory_set_config=test_set_config
        )

        for pilot_name in pilot_list:
            print("-" * 70)
            print(f"Simulating pilot '{pilot_name}' on objective '{obj_name}'")

            traj_file = os.path.join(
                trajectories_dir, f"sim_data_{scene_name}_{obj_name}_{pilot_name}.pt"
            )
            vid_file = os.path.join(
                videos_dir, f"sim_video_{scene_name}_{obj_name}_{pilot_name}.mp4"
            )

            if pilot_name == "expert":
                policy = VehicleRateMPC(tXUi,base_policy_name,base_frame_name,pilot_name)
            else:
                policy = Pilot(cohort_name,pilot_name)
                policy.set_mode('deploy')

            results = []
            for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):

                simulator.load_frame(frame)

                # Simulate Trajectory
            #FIXME
                # if obj_name.startswith("loiter_"):
                #     # For loiter trajectories, simulate with special loiter parameters
                #     print(f"simulating loiter trajectory with query: null")
                #     Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                #         policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),
                #         query="null",clipseg=vision_processor,verbose=verbose)
            #
                # else:
                # Normal simulation for non-loiter trajectories
                Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                    policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),
                    query=obj_name,vision_processor=vision_processor,verbose=verbose)
                # Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                #     policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),query=obj_name,clipseg=vision_processor)

                # Save Trajectory
                trajectory = {
                    "Tro":Tro,"Xro":Xro,"Uro":Uro,
                    "Iro":Iro,  # Rendered images (rgb, depth, semantic) for critic training
                    "tXUd":tXUi,"obj":np.zeros((18,1)),"Ndata":Uro.shape[1],"Tsol":Tsol,"Adv":Adv,
                    "rollout_id":method_name+"_"+str(idx).zfill(5),
                    "course":course_name,
                    "frame":frame}
                results.append(trajectory)

                # rollout = {
                #     "Tro": Tro, "Xro": Xro, "Uro": Uro,
                #     "Xid": tXUi[1:11, :],
                #     "obj": np.zeros((18, 1)),
                #     "Ndata": Uro.shape[1],
                #     "Tsol": Tsol, "Adv": Adv,
                #     "rollout_id": "acados_rollout",
                #     "course": scene_name,
                #     "drone": drone_name,
                #     "method": method_name,
                #     "objective": obj_name
                # }
                # results.append(rollout)
                # mpc_expert.clear_generated_code()

            # semantic_imgs = Iro["semantic"]

            # if visualize_rrt:
            #     combined_file_path = os.path.join(f"{combined_prefix}_{obj_name}.pkl")#f"{combined_prefix}_{obj_name}.pkl"
            #     with open(combined_file_path, 'rb') as f:
            #         trajectory_data = pickle.load(f)
            #         th.debug_figures_RRT(trajectory_data["obj_loc"],trajectory_data["positions"],trajectory_data["trajectory"],
            #                             trajectory_data["smooth_trajectory"],trajectory_data["times"])

            if vision_processor is not None:
                imgs = {
                    "semantic": Iro["semantic"],
                    "rgb": Iro["rgb"],
                    "depth": Iro["depth"]
                }
            else:
                imgs = {
                    "semantic": Iro["semantic"]
                }

            # run for each key in imgs
            for key in imgs:
                if use_flight_recorder:
                    fr = rf.FlightRecorder(
                        Xro.shape[0], Uro.shape[0],
                        20, tXUi[0, -1],
                        [224, 398, 3],
                        np.zeros((18, 1)),
                        cohort_name, scene_name, pilot_name
                    )
                    fr.simulation_import(
                        imgs[key], Tro, Xro, Uro, tXUi, Tsol, Adv
                    )
                    fr.save()
                else:
                    torch.save(results, traj_file)

                    # prepare and write video
                    frames = torch.zeros(
                        (imgs[key].shape[0], 720, 1280, 3)
                    )
                    imgs_t = torch.from_numpy(imgs[key])
                    for i in range(imgs_t.shape[0] - 1):
                        img = imgs_t[i].permute(2, 0, 1)
                        img = transform(img)
                        frames[i] = img.permute(1, 2, 0)

                    # save video with key in filename
                    key_vid_file = vid_file.replace('.mp4', f'_{key}.mp4')
                    write_video(key_vid_file, frames, fps=20)

    # === 11) Final Summary Report ===
    print("\n" + "="*80)
    print("🎯 SIMULATION COMPLETE - DATA GENERATION SUMMARY")
    print("="*80)
    
    # Count files in each subdirectory
    def count_files_and_size(directory):
        """Count files and calculate total size in a directory"""
        if not os.path.exists(directory):
            return 0, 0
        
        file_count = 0
        total_size = 0
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                file_count += 1
                total_size += os.path.getsize(file_path)
        return file_count, total_size
    
    def format_size(bytes_size):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    # Count and measure each file type
    rrt_count, rrt_size = count_files_and_size(rrt_data_dir)
    traj_count, traj_size = count_files_and_size(trajectories_dir)
    video_count, video_size = count_files_and_size(videos_dir)
    
    # Check if rrt_plots directory exists (from notebook auto-save)
    rrt_plots_dir = os.path.join(simulation_data_dir, "rrt_plots")
    plots_count, plots_size = count_files_and_size(rrt_plots_dir)
    
    total_files = rrt_count + traj_count + video_count + plots_count
    total_size = rrt_size + traj_size + video_size + plots_size
    
    print(f"📁 Data stored in: {simulation_data_dir}")
    print(f"   └── Timestamp: {timestamp}")
    print()
    
    if rrt_count > 0:
        print(f"🗂️  RRT Planning Data:")
        print(f"   └── {rrt_count:2d} files | {format_size(rrt_size):>8s} | {rrt_data_dir}")
    
    if traj_count > 0:
        print(f"🚁 Trajectory Data:")
        print(f"   └── {traj_count:2d} files | {format_size(traj_size):>8s} | {trajectories_dir}")
    
    if video_count > 0:
        print(f"🎬 Simulation Videos:")
        print(f"   └── {video_count:2d} files | {format_size(video_size):>8s} | {videos_dir}")
    
    if plots_count > 0:
        print(f"📊 RRT Visualization Plots:")
        print(f"   └── {plots_count:2d} files | {format_size(plots_size):>8s} | {rrt_plots_dir}")
    
    print()
    print(f"📈 TOTAL: {total_files} files | {format_size(total_size)} storage consumed")
    print()
    
    # Breakdown by pilot and objective if we have trajectory data
    if traj_count > 0:
        print(f"🎮 Simulation Coverage:")
        pilots_tested = set()
        objectives_tested = set()
        
        for filename in os.listdir(trajectories_dir):
            if filename.startswith("sim_data_") and filename.endswith(".pt"):
                # Parse filename: sim_data_{scene}_{objective}_{pilot}.pt
                parts = filename[9:-3].split("_")  # Remove "sim_data_" and ".pt"
                if len(parts) >= 2:
                    # Find pilot (last part) and objective (everything before pilot)
                    pilot = parts[-1]
                    objective = "_".join(parts[:-1])
                    
                    # Clean up objective (remove scene prefix)
                    for scene_name, _ in flights:
                        if objective.startswith(scene_name + "_"):
                            objective = objective[len(scene_name) + 1:]
                            break
                    
                    pilots_tested.add(pilot)
                    objectives_tested.add(objective)
        
        print(f"   └── {len(pilots_tested)} pilots: {', '.join(sorted(pilots_tested))}")
        print(f"   └── {len(objectives_tested)} objectives: {', '.join(sorted(objectives_tested))}")
    
    print("="*80)

            # if use_flight_recorder:
            #     fr = rf.FlightRecorder(
            #         Xro.shape[0], Uro.shape[0],
            #         20, tXUi[0, -1],
            #         [224, 398, 3],
            #         np.zeros((18, 1)),
            #         cohort_name, scene_name, pilot_name
            #     )
            #     fr.simulation_import(
            #         Iro, Tro, Xro, Uro, tXUi, Tsol, Adv
            #     )
            #     fr.save()
            # else:
            #     # print(f"Simulated {len(drone_instances)} rollouts.")
            #     torch.save(results, traj_file)

            #     # prepare and write video
            #     frames = torch.zeros(
            #         (Iro.shape[0], 720, 1280, 3)
            #     )
            #     imgs_t = torch.from_numpy(Iro)
            #     for i in range(imgs_t.shape[0] - 1):
            #         img = imgs_t[i].permute(2, 0, 1)
            #         img = transform(img)
            #         frames[i] = img.permute(1, 2, 0)

            #     write_video(vid_file, frames, fps=20)
