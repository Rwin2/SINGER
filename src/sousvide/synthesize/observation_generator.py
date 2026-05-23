import numpy as np
import os
import torch

from typing import Dict,Union,List

import sousvide.synthesize.data_utils as du

from sousvide.control.pilot import Pilot
from tqdm import tqdm

def generate_observation_data(
    cohort: str,
    roster: List[str],
    subsample: float = 1.0,
    validation_mode: bool = False
) -> None:
    """
    Takes rollout data and generates observations for each pilot in the cohort.
    In validation_mode, produces two sets of observations from the "val" and "rollout" videos.
    """
    # Base paths
    workspace_path    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path       = os.path.join(workspace_path, "cohorts", cohort)
    rollout_folder    = os.path.join(cohort_path, "rollout_data")

    Pilots = [Pilot(cohort, name) for name in roster]

    print("=" * 90)

    for pilot in Pilots:
        print(f"Pilot: {pilot.name}, Augment μ={pilot.da_cfg['mean']} ±{pilot.da_cfg['std']}, subsample=1/{int(1/subsample)}")
        print("-" * 90)

        courses = sorted(os.listdir(rollout_folder))
        Nobs = 0

        for course in courses:
            folder = os.path.join(rollout_folder, course)

            if validation_mode:
                # pick up val vs rollout files
                traj_files = sorted([f for f in os.listdir(folder) if f.startswith("trajectories_val") and f.endswith(".pt")])
                img_files  = sorted([f for f in os.listdir(folder) if f.startswith("imgdata_val") and f.endswith(".pt")])
                vid_val    = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("video_val") and not f.startswith("video_val_rollout") and f.endswith(".mp4")])
                vid_roll   = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("video_val_rollout") and f.endswith(".mp4")])

                print(f"Course {course}: Found {len(traj_files)} trajectory files, {len(img_files)} image files, {len(vid_val)} validation videos, and {len(vid_roll)} rollout videos.")

                has_roll = len(vid_roll) > 0
                if has_roll:
                    for tfile, ifile, vval, vrol in tqdm(zip(traj_files, img_files, vid_val, vid_roll), total=len(traj_files), desc=f"Processing Rollout & Validation Files for Course: {course}"):
                        traj_ds = torch.load(os.path.join(folder, tfile))
                        img_ds  = torch.load(os.path.join(folder, ifile))

                        # Observations on semantic-validation video
                        obs_val = generate_observations(pilot, traj_ds, img_ds, vval, subsample)
                        Nobs += obs_val["Nobs"]
                        save_observations(
                            cohort_path, course, pilot.name,
                            obs_val,
                            validation_mode=True,
                            suffix="val"
                        )
                        # Observations on rollout-validation video
                        obs_rol = generate_observations(pilot, traj_ds, img_ds, vrol, subsample)
                        Nobs += obs_rol["Nobs"]
                        save_observations(
                            cohort_path, course, pilot.name,
                            obs_rol,
                            validation_mode=True,
                            suffix="val_rollout"
                        )
                else:
                    for tfile, ifile, vval in tqdm(zip(traj_files, img_files, vid_val), total=len(traj_files), desc=f"Processing Validation Files for Course: {course}"):
                        traj_ds = torch.load(os.path.join(folder, tfile))
                        img_ds  = torch.load(os.path.join(folder, ifile))

                        # Observations on semantic-validation video
                        obs_val = generate_observations(pilot, traj_ds, img_ds, vval, subsample)
                        Nobs += obs_val["Nobs"]
                        save_observations(
                            cohort_path, course, pilot.name,
                            obs_val,
                            validation_mode=True,
                            suffix="val"
                        )
            else:
                traj_files = sorted([f for f in os.listdir(folder)
                                     if f.startswith("trajectories") and not f.startswith("trajectories_val") and f.endswith(".pt")])
                img_files  = sorted([f for f in os.listdir(folder)
                                     if f.startswith("imgdata") and not f.startswith("imgdata_val") and f.endswith(".pt")])
                vid_files  = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                     if f.startswith("video") and not f.startswith("video_val") and f.endswith(".mp4")])

                print(f"Course {course}: Found {len(traj_files)} trajectory files, {len(img_files)} image files, {len(vid_files)} training videos.")

                for tfile, ifile, vid in tqdm(zip(traj_files, img_files, vid_files), total=len(traj_files), desc=f"Processing Training Files for Course: {course}"):
                    traj_ds = torch.load(os.path.join(folder, tfile))
                    img_ds  = torch.load(os.path.join(folder, ifile))

                    obs = generate_observations(pilot, traj_ds, img_ds, vid, subsample)
                    Nobs += obs["Nobs"]
                    save_observations(cohort_path, course, pilot.name, obs)

        print(f"Extracted {Nobs} observations from {len(courses)} course(s).")
    print("=" * 90)


def generate_observations(pilot:Pilot,
                            trajectory_data_set:Dict[str,Union[str,int,Dict[str,Union[np.ndarray,float,str]]]],
                            image_data_set:Dict[str,Union[str,int,Dict[str,Union[np.ndarray,float,str]]]],
                            video,
                            subsample:float=1) -> Dict[str,Union[str,int,Dict[str,Union[np.ndarray,float,str]]]]:
    
    # Initialize the observation data dictionary
    Observations, Img_Obsv = [], []

    # Unpack augmenteation variables
    aug_type = np.array(pilot.da_cfg["type"])
    aug_mean = np.array(pilot.da_cfg["mean"])
    aug_std = np.array(pilot.da_cfg["std"])

    # Set subsample step
    nss = int(1/subsample)

    # Generate observations
    Nobs = 0
    for trajectory_data,image_data in zip(trajectory_data_set["data"],image_data_set["data"]):
        # Unpack data
        Tro,Xro,Uro = trajectory_data["Tro"],trajectory_data["Xro"],trajectory_data["Uro"]
        obj,Ndata = trajectory_data["obj"],trajectory_data["Ndata"]
        rollout_id,course = trajectory_data["rollout_id"],trajectory_data["course"]
        frame = trajectory_data["frame"]

        # Decompress and extract the image data
        # Imgs = du.decompress_data(image_data)["images"]
        Imgs = du.load_video_frames(video,image_data)

        # Check if images are raw or processed. Raw images are in (N,H,W,C) format while
        # processed images are in (N,C,H,W) format.
        height,width = Imgs.shape[1],Imgs.shape[2]

        if height == 224 and width == 224:
            Imgs = np.transpose(Imgs, (0, 3, 1, 2))

        # Create Rollout Data
        Xnn,Ynn,Obsv = [],[],[]
        upr = np.zeros(4)
        znn_cr = torch.zeros(pilot.model.Nz).to(pilot.device)
        for k in range(Ndata):
            # Generate current state (with/without noise augmentation)
            if aug_type == "additive":
                xcr = Xro[:,k] + np.random.normal(aug_mean,aug_std)
            elif aug_type == "multiplicative":
                xcr = Xro[:,k] * (1 + np.random.normal(aug_mean,aug_std))
            else:
                xcr = Xro[:,k]

            # Extract other data
            tcr = Tro[k]
            ucr = Uro[:,k]
            img_cr = Imgs[k,:,:,:]

            # Generate the sfti data
            _,znn_cr,_,xnn,_ = pilot.OODA(upr,tcr,xcr,obj,img_cr,znn_cr)
            ynn = {"unn":ucr,"mfn":np.array([frame["mass"],frame["force_normalized"]]),"onn":xcr}

            # Store the data
            if k % nss == 0:
                Xnn.append(xnn)
                Ynn.append(ynn)

            # Loop updates
            upr = ucr

        # Store the observation data
        observations = {
            "Xnn":Xnn,
            "Ynn":Ynn,
            "Ndata":len(Xnn),
            "rollout_id":rollout_id,
            "course":course,"frame":frame
        }
        Observations.append(observations)
        Nobs += len(Xnn)

    observations_data_set = {"data":Observations,
                    "set":trajectory_data_set["set"],
                    "Nobs":Nobs,
                    "course":trajectory_data_set["course"]}

    return observations_data_set


def save_observations(
    cohort_path: str,
    course_name: str,
    pilot_name: str,
    observations: Dict[str, Union[np.ndarray, int, str]],
    validation_mode: bool = False,
    suffix: str = ""
) -> None:
    """
    Saves the observation data to a .pt file in folders corresponding to pilot name
    within the course directory within the cohort directory.

    In validation_mode with a provided suffix (e.g. "val" or "rollout"),
    it appends the suffix to the filename: observations_<suffix><set>.pt.
    Otherwise it uses observations<set>.pt.

    Args:
        cohort_path:    Cohort path.
        course_name:    Name of the course.
        pilot_name:     Name of the pilot.
        observations:   Observation data dict, must include key "set".
        validation_mode: Whether this is validation data (defaults to False).
        suffix:         Optional suffix for validation streams (e.g. "val" or "rollout").

    Returns:
        None: Writes a .pt file to disk.
    """
    # Create directory for this pilot and course if it doesn't exist
    obs_pilot_dir = os.path.join(cohort_path, "observation_data", pilot_name)
    os.makedirs(obs_pilot_dir, exist_ok=True)
    obs_course_dir = os.path.join(obs_pilot_dir, course_name)
    os.makedirs(obs_course_dir, exist_ok=True)

    # Determine filename suffix
    if suffix:
        suffix_str = f"_{suffix}"
    else:
        suffix_str = ""

    # Build filename and path
    set_id = str(observations.get("set", ""))
    filename = f"observations{suffix_str}{set_id}.pt"
    observations_data_path = os.path.join(obs_course_dir, filename)

    # Save to disk
    torch.save(observations, observations_data_path)