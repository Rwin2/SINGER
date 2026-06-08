import numpy as np
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from tqdm.auto import trange
from sousvide.control.pilot import Pilot
from typing import List,Tuple,Literal,Union,Dict,Callable
from enum import Enum

class ObservationData(Dataset):
    def __init__(self, Xnn:List[Dict[str,torch.Tensor]], Ynn:List[torch.Tensor],extractor:Callable,
                 augment_fn=None):
        self.Xnn = Xnn
        self.Ynn = Ynn
        self.extractor = extractor
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.Xnn)

    def __getitem__(self,idx):
        xnn = self.Xnn[idx]
        ynn = self.Ynn[idx]
        if self.augment_fn is not None:
            xnn = self.augment_fn(xnn)
        return self.extractor(xnn,ynn)
    
def ensure_torch_tensor(variable):
    if isinstance(variable, np.ndarray):
        return torch.from_numpy(variable).float()
    elif isinstance(variable, torch.Tensor):
        return variable.float()
    else:
        raise ValueError("The variable is neither a NumPy array nor a PyTorch tensor.")

def generate_dataset(observation_data_path:str,student:Pilot,
                     mode:Literal["Parameter","Odometry","Commander"],device:torch.device,
                     augment_fn=None) -> Dataset:
    """
    Generate a Pytorch Dataset from the given list of observation data path.

    Args:
        observation_data_path:  Observation data path.
        augment_fn:  Optional function applied to each xnn dict at __getitem__ time.

    Returns:
        dset:  The Pytorch Dataset object.
    """
    Xnn_ds,Ynn_ds = extract_data(observation_data_path)
    extractor = student.model.get_data[mode]

    return ObservationData(Xnn_ds,Ynn_ds,extractor,augment_fn=augment_fn)

def get_true_states(observation_data_path:str):
    """
    Get the true states from the given list of observation data paths.

    Args:
        observation_data_path:  Observation data path.

    Returns:
        TrueStates:  The list of true states.
    """
    # Load Data Files
    dir_path  =  os.path.dirname(observation_data_path)
    base_name = os.path.basename(observation_data_path)
    traj_name = "trajectories_"+"_".join(base_name.split("_")[1:])
    traj_path = os.path.join(dir_path,traj_name)

    # Load Trajectory Data
    traj_data = torch.load(traj_path)

    assert len(traj_data['data'])==1, "Only one trajectory is supported."

    Tact,Xact = traj_data['data'][0]['Tro'],traj_data['data'][0]['Xro']

    return Tact,Xact

def get_input_images(observation_data_path:str,student:Pilot):
    """
    Get the input images from the given list of observation data paths.

    Args:
        observation_data_path:  Observation data path.

    Returns:
        Images:  The list of input images.
    """
    Xnn_ds,_ = extract_data(observation_data_path)
    extractor = student.model.get_data["Images"]

    Images = []
    for xnn in Xnn_ds:
        Images.append(extractor(xnn))

    Images = torch.stack(Images)

    return Images

def get_input_data(observation_data_path:str,student:Pilot):
    """
    Get the input from the given list of observation data paths.

    Args:
        observation_data_path:  Observation data path.

    Returns:
        Inputs:  The list of inputs.
    """
    Xnn_ds,Ynn_ds = extract_data(observation_data_path)
    extractor = student.model.get_data["Commander"]

    Inputs = []
    for xnn,ynn in zip(Xnn_ds,Ynn_ds):
        Inputs.append(extractor(xnn,ynn)[0])

    return Inputs

def extract_data(observation_data_path:str):
    """
    Extract the observation data from the given list of observation data paths.

    Args:
        observation_data_path:  Observation data path.

    Returns:
        Xnn_ds:  The list of input data.
        Ynn_ds:  The list of output data.
    """
    # Load Data Files
    observation_data = torch.load(observation_data_path)

    # Extract the observation data
    Xnn_ds,Ynn_ds = [],[]
    for observations in observation_data["data"]:
        # Extract the inputs to GPU
        Xnn = []
        for xnn_raw in observations["Xnn"]:
            for key,value in xnn_raw.items():
                xnn_raw[key] = ensure_torch_tensor(value)
            Xnn.append(xnn_raw)

        # Extract the labels to GPU
        Ynn = []
        for ynn_raw in observations["Ynn"]:
            for key, value in ynn_raw.items():
                ynn_raw[key] = ensure_torch_tensor(value)
            Ynn.append(ynn_raw)

        # Append to the list
        Xnn_ds.extend(Xnn)
        Ynn_ds.extend(Ynn)
    
    return Xnn_ds,Ynn_ds

def extract_data_chunked(observation_data_path: str, chunk_horizon: int = 5, action_dim: int = 4):
    """
    Extract observation data with action chunking: window sequential actions
    within each rollout into chunks of size (chunk_horizon * action_dim).

    Reuses extract_data's tensor conversion but preserves rollout boundaries
    for correct windowing.

    Args:
        observation_data_path:  Path to .pt observation file.
        chunk_horizon:          Number of future actions per chunk (H).
        action_dim:             Action dimensionality (default 4 for SINGER).

    Returns:
        Xnn_ds:  List of input dicts (one per valid timestep).
        Ynn_ds:  List of output dicts with chunked unn of shape (H * action_dim,).
    """
    observation_data = torch.load(observation_data_path)

    Xnn_ds, Ynn_ds = [], []
    for observations in observation_data["data"]:
        # Convert to tensors (same as extract_data)
        Xnn = []
        for xnn_raw in observations["Xnn"]:
            for key, value in xnn_raw.items():
                xnn_raw[key] = ensure_torch_tensor(value)
            Xnn.append(xnn_raw)

        Ynn = []
        for ynn_raw in observations["Ynn"]:
            for key, value in ynn_raw.items():
                ynn_raw[key] = ensure_torch_tensor(value)
            Ynn.append(ynn_raw)

        N = len(Xnn)
        # Window within this rollout: for timestep i, chunk = [unn_i, ..., unn_{i+H-1}]
        for i in range(N - chunk_horizon + 1):
            # Build chunked action label
            unn_chunk = torch.cat([Ynn[i + j]["unn"] for j in range(chunk_horizon)])
            # Copy ynn dict with chunked unn
            ynn_chunked = {
                "unn": unn_chunk,   # shape: (H * action_dim,)
                "mfn": Ynn[i]["mfn"],
                "onn": Ynn[i]["onn"],
            }
            Xnn_ds.append(Xnn[i])
            Ynn_ds.append(ynn_chunked)

    return Xnn_ds, Ynn_ds


def extract_data_dynamics(observation_data_path: str, chunk_horizon: int = 5,
                           action_dim: int = 4, state_keys=("obj_com",)):
    """
    Extract chunked actions + future state targets for dynamics loss.

    Returns same (Xnn_ds, Ynn_ds) as extract_data_chunked but also adds
    future state values to each ynn dict under 'future_{key}' keys.

    For DreamZero-style dynamics loss: predict future bearing/elevation
    alongside future actions.

    Args:
        observation_data_path:  Path to .pt observation file.
        chunk_horizon:          Number of future steps (H).
        action_dim:             Action dimensionality.
        state_keys:             Which xnn keys to extract as future states.

    Returns:
        Xnn_ds, Ynn_ds: same as extract_data_chunked, with added future state tensors.
    """
    observation_data = torch.load(observation_data_path)

    Xnn_ds, Ynn_ds = [], []
    for observations in observation_data["data"]:
        Xnn = []
        for xnn_raw in observations["Xnn"]:
            for key, value in xnn_raw.items():
                xnn_raw[key] = ensure_torch_tensor(value)
            Xnn.append(xnn_raw)

        Ynn = []
        for ynn_raw in observations["Ynn"]:
            for key, value in ynn_raw.items():
                ynn_raw[key] = ensure_torch_tensor(value)
            Ynn.append(ynn_raw)

        N = len(Xnn)
        for i in range(N - chunk_horizon + 1):
            unn_chunk = torch.cat([Ynn[i + j]["unn"] for j in range(chunk_horizon)])
            ynn_chunked = {
                "unn": unn_chunk,
                "mfn": Ynn[i]["mfn"],
                "onn": Ynn[i]["onn"],
            }
            # Add future state values for dynamics loss
            for key in state_keys:
                future_vals = torch.stack([Xnn[i + j][key] for j in range(1, chunk_horizon + 1)
                                           if (i + j) < N])
                if future_vals.shape[0] < chunk_horizon:
                    # Pad with last available value
                    pad = future_vals[-1:].expand(chunk_horizon - future_vals.shape[0], -1)
                    future_vals = torch.cat([future_vals, pad])
                ynn_chunked[f"future_{key}"] = future_vals.reshape(-1)  # flatten

            Xnn_ds.append(Xnn[i])
            Ynn_ds.append(ynn_chunked)

    return Xnn_ds, Ynn_ds


def generate_dataset_chunked(observation_data_path: str, student, mode: str,
                              device, chunk_horizon: int = 5, action_dim: int = 4):
    """Generate a chunked Dataset from observation data. Wrapper around extract_data_chunked."""
    Xnn_ds, Ynn_ds = extract_data_chunked(observation_data_path, chunk_horizon, action_dim)
    extractor = student.model.get_data[mode]
    return ObservationData(Xnn_ds, Ynn_ds, extractor)


def get_data_paths(cohort_name: str,
                   student_name: str,
                   course_name: Union[str, None] = None
                   ) -> Tuple[List[str], List[str], List[str]]:
    """
    Walk each course directory and gather:
      • normal files    -> split into train/test just like before
      • rollout files   -> all filenames matching observations_val*.pt

    Returns train_paths, test_paths, rollout_paths.
    """
    # base folder: .../cohorts/<cohort>/observation_data/<student>/
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    base = os.path.join(workspace_path, "cohorts", cohort_name,
                        "observation_data", student_name)

    # choose which course folders to scan
    if course_name is None:
        course_paths = [d.path for d in os.scandir(base) if d.is_dir()]
    else:
        course_paths = [os.path.join(base, course_name)]

    train_paths, test_paths, validation_paths, rollout_paths = [], [], [], []
    for course_path in course_paths:
        data_files = []
        val_files = []
        rollout_files = []
        for entry in os.scandir(course_path):
            fn = entry.name
            if not fn.endswith(".pt"):
                continue
            if fn.startswith("observations_val_rollout"):
                rollout_files.append(entry.path)
            elif fn.startswith("observations_val"):
                val_files.append(entry.path)
            elif fn.startswith("observations"):
                data_files.append(entry.path)
        data_files.sort()
        rollout_files.sort()
        val_files.sort()

        # split the normal data_files into train/test
        if len(data_files) == 0 and len(val_files) == 0 and len(rollout_files) == 0:
            # print(f"Warning: No observation files found in {course_path}")
            raise ValueError(f"No observation files in {course_path}")
        elif len(data_files) == 0 and (len(val_files) > 0 or len(rollout_files) > 0):
            print(f"Warning: No training observation files found in {course_path}")
        elif len(data_files) == 1:
            train_paths.append(data_files[0])
            test_paths.append(data_files[0])
        else:
            train_paths.extend(data_files[:-1])
            test_paths.append(data_files[-1])

        # collect all rollout (val) files
        validation_paths.extend(val_files)
        rollout_paths.extend(rollout_files)

    return train_paths, test_paths, validation_paths, rollout_paths


