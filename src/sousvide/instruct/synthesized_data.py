import numpy as np
import os
import time
import torch
from torch import NoneType, nn
from torch.utils.data import DataLoader,Dataset
from tqdm.notebook import trange
from sousvide.control.pilot import Pilot
from typing import List,Tuple,Literal,Union,Dict,Callable
from enum import Enum

class ObservationData(Dataset):
    def __init__(self, Xnn:List[Dict[str,torch.Tensor]], Ynn:List[torch.Tensor],extractor:Callable):      
        self.Xnn = Xnn
        self.Ynn = Ynn
        self.extractor = extractor

    def __len__(self):
        return len(self.Xnn)

    def __getitem__(self,idx):
        xnn = self.Xnn[idx]
        ynn = self.Ynn[idx]

        return self.extractor(xnn,ynn)
    
def ensure_torch_tensor(variable):
    if isinstance(variable, np.ndarray):
        return torch.from_numpy(variable).float()
    elif isinstance(variable, torch.Tensor):
        return variable.float()
    else:
        raise ValueError("The variable is neither a NumPy array nor a PyTorch tensor.")

def generate_dataset(observation_data_path:str,student:Pilot,
                     mode:Literal["Parameter","Odometry","Commander"],device:torch.device) -> Dataset:
    """
    Generate a Pytorch Dataset from the given list of observation data path.

    Args:
        observation_data_path:  Observation data path.

    Returns:
        dset:  The Pytorch Dataset object.
    """
    Xnn_ds,Ynn_ds = extract_data(observation_data_path)
    extractor = student.model.get_data[mode]

    return ObservationData(Xnn_ds,Ynn_ds,extractor)

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

def get_data_paths_old(cohort_name:str,
                   student_name:str,
                   course_name:Union[str,None]=None
                   ) -> Tuple[List[str],str]:
    """
    Get the paths to the observation data files for training or testing. If mode is 'train',
    the paths are shuffled. This way, we can mix the course data a little better. However, we
    need to keep the order constant across epochs and so we use a rng_seed to lock the randomness.
    If mode is 'test', the first file is selected.

    Args:
        cohort_name:  The name of the cohort.
        student_name: The name of the student.
        course_name:  The name of the course.

    Returns:
        train_data:  The list of training data paths.
        test_data:   The list of testing data paths.
    """

    # Some useful path(s)
    workspace_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    observation_data_path = os.path.join(
        workspace_path,"cohorts",cohort_name,"observation_data",student_name)

    # Get course paths
    if course_name is None:
        course_paths = [course.path for course in os.scandir(observation_data_path) if course.is_dir()]
    else:
        course_paths = [os.path.join(observation_data_path,course_name)]

    # Split into training and testing data
    train_data,test_data = [],[]
    for course_path in course_paths:
        # Get data files for the course
        data_paths = []
        for file in os.scandir(course_path):
            data_paths.append(file.path)

        data_paths.sort()

        if len(data_paths) == 1:
            train_data.append(data_paths[0])
            test_data.append(data_paths[0])
        elif len(data_paths) > 1:
            train_data.extend(data_paths[:-1])
            test_data.append(data_paths[-1])
        else:
            raise ValueError("No data found.")

    return train_data,test_data

