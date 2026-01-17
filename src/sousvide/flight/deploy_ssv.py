import os
import json
import yaml
import pickle
from datetime import datetime
from typing import List, Union, Literal, Tuple
from re import T, X

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
from figs.dynamics.model_specifications import generate_specifications
# import figs.visualize.generate_videos as gv

import figs.tsampling.build_rrt_dataset as bd
import figs.scene_editing.scene_editing_utils as scdt

# import visualize.plot_synthesize as ps
# import visualize.record_flight as rf

# import sousvide.flight.vision_preprocess as vp
import sousvide.flight.vision_preprocess_alternate as vp


def simulate_roster(cohort_name:str,method_name:str,
                    flights:List[Tuple[str,str]],
                    roster:List[str],
                    use_flight_recorder:bool=False,
                    review:bool=False,
                    filename:str=None,
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
    use_clip   = sample_set_config["clipseg"]


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

    if use_clip:
        print("SIL Model set to CLIPSeg.")
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

                        if debug_info is not None:
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

            for objective in objectives:
                combined_prefix = os.path.join(scenes_cfg_dir, scene_name)
                if filename is not None:
                    combined_file_path = filename
                else:
                    combined_file_path = f"{combined_prefix}_{objective}.pkl"
                with open(combined_file_path, "rb") as f:
                    data = pickle.load(f)
                    trajectory_dataset[objective] = data
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
                    query=obj_name,clipseg=vision_processor,verbose=verbose)
                # Tro,Xro,Uro,Iro,Tsol,Adv = simulator.simulate(
                #     policy,perturbation["t0"],tXUi[0,-1],perturbation["x0"],np.zeros((18,1)),query=obj_name,clipseg=vision_processor)

                # Save Trajectory
                trajectory = {
                    "Tro":Tro,"Xro":Xro,"Uro":Uro,
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
                    "rgb": Iro["rgb"]
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
