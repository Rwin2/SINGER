import numpy as np
import torch
import os
import json
from typing import List, Tuple

import sousvide.visualize.plot_synthesize as ps

from sousvide.control.pilot import Pilot
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.tsplines import min_snap as ms
import sousvide.synthesize.synthesize_helper as sh
import sousvide.synthesize.rollout_generator as rg
import sousvide.visualize.record_flight as rf
from torchvision.io import write_video
from torchvision.transforms import Resize
from figs.simulator import Simulator
import figs.utilities.trajectory_helper as th
from figs.dynamics.model_specifications import generate_specifications
import figs.visualize.generate_videos as gv

def simulate_roster(cohort_name:str,method_name:str,
                    scene_name:str,course_name:str,
                    roster:List[str],
                    use_flight_recorder:bool=False):

    # Some useful path(s)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_path = os.path.join(workspace_path,"cohorts",cohort_name,"output")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Extract configs
    method_path = os.path.join(workspace_path,"configs","method",method_name+".json")
    with open(method_path) as json_file:
        method_config = json.load(json_file)

    test_set_config = method_config["test_set"]
    sample_set_config = method_config["sample_set"]

    base_policy_name = sample_set_config["policy"]
    base_frame_name = sample_set_config["frame"]

    rollout_name = test_set_config["rollout"]
    Nrep = test_set_config["reps"]
    trajectory_set_config = test_set_config["trajectory_set"]
    frame_set_config = test_set_config["frame_set"]

    # Compute simulation variables
    Trep = np.zeros(Nrep)

    frame_path  = os.path.join(workspace_path,"configs","frame",base_frame_name+".json")
    with open(frame_path) as json_file:
        base_frame_config = json.load(json_file)  
    
    base_frame_specs = generate_specifications(base_frame_config)

    # Add Expert to Roster
    roster = ["expert"]+roster

    # Initialize simulator for scene
    sim = Simulator(scene_name,rollout_name)

    # Load course_config
    course_path = os.path.join(workspace_path,"configs","course",course_name+".json")
    with open(course_path) as json_file:
        course_config = json.load(json_file)

    # Compute ideal trajectory spline
    output = ms.solve(course_config)
    if output is not False:
        Tpi,CPi = output
    else:
        raise ValueError("Desired trajectory not feasible. Aborting.")

    # Generate sample variables
    Frames = rg.generate_frames(Trep,base_frame_config,frame_set_config)        
    Perturbations  = rg.generate_perturbations(Trep,Tpi,CPi,trajectory_set_config)
    
    # Simulate samples across roster
    for pilot_name in roster:
        # Load Pilot
        if pilot_name == "expert":
            policy = VehicleRateMPC(course_name,base_policy_name,base_frame_name,pilot_name)
        else:
            policy = Pilot(cohort_name,pilot_name)
            policy.set_mode('deploy')

        # Save paths
        trajectories_path = os.path.join(output_path,"sim_"+course_name+"_"+policy.name+".pt")
        video_path = os.path.join(output_path, "sim_"+course_name+"_"+policy.name+".mp4")

        # Compute ideal trajectory variables
        tXUd = th.TS_to_tXU(Tpi,CPi,base_frame_specs,policy.hz)
        obj = sh.ts_to_obj(Tpi,CPi)

        # Simulate trajectory across samples
        trajectories = []
        for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):
            # Load Frame
            sim.load_frame(frame)

            # Simulate Trajectory
            Tro,Xro,Uro,Iro,Tsol,Adv = sim.simulate(
                policy,perturbation["t0"],tXUd[0,-1],perturbation["x0"],obj)

            # Save Trajectory
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,
                "tXUd":tXUd,"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,"Adv":Adv,
                "rollout_id":method_name+"_"+str(idx).zfill(5),
                "course":course_name,
                "frame":frame}
            trajectories.append(trajectory)

        # Print some diagnostics
        print("Simulated",len(Frames),"rollouts for",policy.name)

        # Save simulations
        print("Saving simulation data...")

        # Save all trajectory data
        torch.save(trajectories,trajectories_path)
        
        # Save last trajectory as video/flight recorder
        if use_flight_recorder:
            # Save Flight Recorder class to mirror real-world deployment
            flight_record = rf.FlightRecorder(
                Xro.shape[0],Uro.shape[0],
                20,tXUd[0,-1],[360,640,3],obj,cohort_name,course_name,policy.name)
            flight_record.simulation_import(Iro,Tro,Xro,Uro,tXUd,Tsol,Adv)
            flight_record.save()
        else:
            # Save video on last trajectory
            gv.images_to_mp4(Iro,video_path, policy.hz)       # Save the video
