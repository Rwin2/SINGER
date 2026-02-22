import numpy as np
import os
import time
import json
import glob
import gc
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from rich.progress import Progress
import wandb
import figs.visualize.rich_visuals as rv
from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import *
from sousvide.rl import rl_helpers as rrl
from typing import List,Tuple,Literal,Dict
from enum import Enum

# Set matplotlib to non-interactive backend to avoid memory issues with many figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure matplotlib to suppress figure warnings during training
plt.rcParams['figure.max_open_warning'] = 0  # Disable the warning entirely


def train_roster(cohort_name:str,roster:List[str],
                 mode:Literal["Parameter","Odometry","Commander"],
                 Neps:int,lim_sv:int,
                 lr:float=1e-4,batch_size:int=64):

    # Initialize Rich progress bar
    progress = rv.get_training_progress()
    train_desc = "[bold dark_green]Training Progress[/]"

    with progress:
        # Create outer task for students
        outer_task = progress.add_task(train_desc, total=len(roster), loss=0.0, units="students")

        for student_name in roster:
            # Load Student
            student = Pilot(cohort_name,student_name)
            student.set_mode('train')

            # Create student progress bar
            student_desc = f"[bold green3]{student_name:>8}[/]"
            student_task = progress.add_task(student_desc, total=Neps, loss=0.0, units='epochs')
            student_bar = (progress, student_task)

            # Train the student
            train_student(cohort_name,student,mode,Neps,lim_sv,lr,batch_size,student_bar)

            # Update outer task
            progress.update(outer_task, advance=1)
            progress.refresh()

def train_student(cohort_name:str,student:Pilot,
                  mode:Literal["Parameter","Odometry","Commander"],
                  Neps:int,lim_sv:int,lr:float,batch_size:int,
                  progress_bar:tuple[Progress,int]|None=None):

    # Pytorch Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Loss function
    criterion = nn.MSELoss(reduction='mean')

    # Send to GPU
    student.model.to(device)

    # Select model components
    if mode in student.model.get_network:
        model = student.model.get_network[mode]["Train"]
    else:
        # Update progress bar if provided
        if progress_bar is not None:
            progress, student_task = progress_bar
            progress.update(student_task, description=f"{student.name} has no model for {mode}.")
        return

    # Set parameters to optimize over
    opt = optim.Adam(model.parameters(),lr=lr)

    # Some Useful Paths
    student_path = student.path
    model_path   = os.path.join(student_path,"model.pth")
    losses_path  = os.path.join(student_path,"losses_"+mode+".pt")
    best_model_path   = os.path.join(student_path, "best_model.pth")
    last_model_path   = os.path.join(student_path, "last_model.pth")

    # Say who we are training
    print("Training Student: ",student.name)
    print(f"Loss Mode: {loss_mode}")

    # Setup Loss Variables (load if exists)
    losses: Dict[str, List] = {}
    if os.path.exists(losses_path):
        losses = torch.load(losses_path)

        print("Re-training existing network.")
        print("Previous Epochs: ", sum(losses["Neps"]), losses["Neps"])

        losses["train"].append([]),losses["test"].append([]),losses["validation"].append([]),losses["rollout"].append([])
        losses["Neps"].append(0),losses["Nspl"].append(0),losses["t_train"].append(0)
    else:
        losses = {
            "train": [None], "test": [None], "validation":[None], "rollout":[None],
            "Neps": [None], "Nspl": [None], "t_train": [None]
        }
        print("Training new network.")

    Loss_train,Loss_tests,Loss_validation,Loss_rollouts = [],[],[],[]
    best_test_loss = float('inf')
    best_val_loss = float('inf')
    last_val_loss = float('inf')
    best_rol_loss = float('inf')
    last_rol_loss = float('inf')
    last_cem_loss = float('nan')
    # Record the start time
    start_time = time.time()

    # Training + Testing Loop
    for ep in range(Neps):
        # Lock/Unlock Networks
        unlock_networks(student,mode)

        # Get Observation Data Files (Paths)
        od_train_files,od_test_files,od_val_files,od_rol_files = get_data_paths(cohort_name,student.name)

        # Training
        loss_log_tn =[]
        for od_train_file in od_train_files:
            # Load Datasets
            dataset = generate_dataset(od_train_file,student,mode,device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

            # Training
            for input,label in dataloader:
                # Move to GPU
                input,label = tuple(tensor.to(device) for tensor in input),label.to(device)

                # Forward Pass
                if use_energy_loss:
                    energy_scores, _ = model(*input, oracle_command=label)
                    loss = criterion(energy_scores, oracle_position=0)
                else:
                    prediction,_ = model(*input)
                    loss = criterion(prediction,label)

                # Backward Pass
                loss.backward()
                opt.step()
                opt.zero_grad()

                # Save loss logs
                loss_log_tn.append((label.shape[0],loss.item()))

        # Testing
        log_log_tt = []
        for od_test_file in od_test_files:
            # Load Datasets
            dataset = generate_dataset(od_test_file,student,mode,device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

            # Testing
            for input,label in dataloader:
                # Move to GPU
                input,label = tuple(tensor.to(device) for tensor in input),label.to(device)

                # Forward Pass
                if use_energy_loss:
                    energy_scores, _ = model(*input, oracle_command=label)
                    loss = criterion(energy_scores, oracle_position=0)
                else:
                    prediction,_ = model(*input)
                    loss = criterion(prediction,label)

                # Save loss logs
                log_log_tt.append((label.shape[0],loss.item()))

        # # Validation (commented out: redundant with rollouts, uses same data)
        # if bool(od_val_files) and ((ep + 1) % 5 == 0) or (ep + 1 == Neps):
        #     val_log = []
        #     for val_file in od_val_files:
        #         ds = generate_dataset(val_file, student, mode, device)
        #         loader = DataLoader(ds, batch_size=batch_size,
        #                             shuffle=False, drop_last=False)
        #         for inp, lbl in loader:
        #             inp, lbl = (t.to(device) for t in inp), lbl.to(device)
        #             if use_energy_loss:
        #                 energy_scores, _ = model(*inp, oracle_command=lbl)
        #                 loss_val = criterion(energy_scores, oracle_position=0)
        #             else:
        #                 pred, _ = model(*inp)
        #                 loss_val = criterion(pred, lbl)
        #             val_log.append((lbl.shape[0], loss_val.item()))
        #     if val_log:
        #         Ntv = sum(n for n, _ in val_log)
        #         loss_validation = sum(n * l for n, l in val_log) / Ntv
        #     else:
        #         loss_validation = float('nan')
        #     last_val_loss = loss_validation
        # else:
        #     loss_validation = last_val_loss

        # Rollouts
        if bool(od_rol_files) and ((ep + 1) % 5 == 0) or (ep + 1 == Neps):
            rol_log = []
            cem_log = [] if use_energy_loss else None
            for rol_file in od_rol_files:
                ds = generate_dataset(rol_file, student, mode, device)
                loader = DataLoader(ds, batch_size=batch_size,
                                    shuffle=False, drop_last=False)
                for inp, lbl in loader:
                    inp, lbl = tuple(t.to(device) for t in inp), lbl.to(device)
                    if use_energy_loss:
                        # CEM CE loss (training mode - oracle at position 0)
                        energy_scores, _ = model(*inp, oracle_command=lbl)
                        cem_loss = criterion(energy_scores, oracle_position=0)

                        # MSE loss (inference mode - compare inferred action to expert)
                        with torch.no_grad():
                            _, inferred_action = model(*inp)
                            mse_loss = F.mse_loss(inferred_action, lbl)

                        rol_log.append((lbl.shape[0], mse_loss.item()))  # rollout loss = MSE
                        cem_log.append((lbl.shape[0], cem_loss.item()))  # cem loss = CE
                    else:
                        pred, _ = model(*inp)
                        loss_rol = criterion(pred, lbl)
                        rol_log.append((lbl.shape[0], loss_rol.item()))
            if rol_log:
                Ntv = sum(n for n, _ in rol_log)
                loss_rollouts = sum(n * l for n, l in rol_log) / Ntv
                if use_energy_loss and cem_log:
                    loss_cem = sum(n * l for n, l in cem_log) / Ntv
                else:
                    loss_cem = None
            else:
                loss_rollouts = float('nan')
                loss_cem = None
            last_rol_loss = loss_rollouts
            if use_energy_loss and loss_cem is not None:
                last_cem_loss = loss_cem
        else:
            loss_rollouts = last_rol_loss
            loss_cem = last_cem_loss if use_energy_loss else None

        # Loss Diagnostics
        Ntn = np.sum([n for n,_ in loss_log_tn])
        Ntt = np.sum([n for n,_ in log_log_tt])
        loss_train = np.sum([n*loss for n,loss in loss_log_tn])/Ntn
        loss_tests = np.sum([n*loss for n,loss in log_log_tt])/Ntt

        Loss_train.append(loss_train)
        Loss_tests.append(loss_tests)
        # if bool(od_val_files):  # Commented out: redundant with rollouts
        #     Loss_validation.append(loss_validation)
        if bool(od_rol_files):
            Loss_rollouts.append(loss_rollouts)

        # Update progress bar if provided
        if progress_bar is not None:
            progress, student_task = progress_bar
            progress.update(student_task, loss=loss_train, advance=1)
            progress.refresh()

        # Log losses to wandb
        # # Commented out: validation is redundant with rollouts
        # if bool(od_val_files) and bool(od_rol_files):
        #     wandb.log({
        #         "train/epoch loss": loss_train,
        #         "test/epoch loss": loss_tests,
        #         "test/epoch validation loss": loss_validation,
        #         "test/epoch rollout loss:": loss_rollouts,
        #         "epoch": ep,
        #     })
        # elif bool(od_val_files):
        #     wandb.log({
        #         "train/epoch loss": loss_train,
        #         "test/epoch loss": loss_tests,
        #         "test/epoch validation_loss": loss_validation,
        #         "epoch": ep,
        #     })
        if bool(od_rol_files):
            log_dict = {
                "train/epoch loss": loss_train,
                "test/epoch loss": loss_tests,
                "test/epoch rollout loss": loss_rollouts,
                "epoch": ep,
            }
            if use_energy_loss and loss_cem is not None and not np.isnan(loss_cem):
                log_dict["test/epoch cem loss"] = loss_cem
            wandb.log(log_dict)
        else:
            wandb.log({
                "train/epoch loss": loss_train,
                "test/epoch loss": loss_tests,
                "epoch": ep,
            })

        # if loss_validation < best_val_loss and loss_tests < best_test_loss:  # Commented out: use rollout loss instead
        #     best_val_loss = loss_validation
        #     best_test_loss = loss_tests
        #     torch.save(student.model, best_model_path)
        if loss_rollouts < best_rol_loss and loss_tests < best_test_loss:
            best_rol_loss = loss_rollouts
            best_test_loss = loss_tests
            torch.save(student.model, best_model_path)
        elif ep+1== Neps:
            # If we reach the end of training, save the last model
            torch.save(student.model, last_model_path)

        # Save at intermediate steps and at the end
        if ((ep+1) % lim_sv == 0) or (ep+1==Neps):
            # Lock the networks
            unlock_networks(student,"None")

            # Record the end time
            end_time = time.time()
            t_train = end_time - start_time

            torch.save(student.model,model_path)

            losses["train"][-1] = Loss_train
            losses["test"][-1] = Loss_tests
            # if bool(od_val_files):  # Commented out: redundant with rollouts
            #     losses["validation"][-1] = Loss_validation
            if bool(od_rol_files):
                losses["rollout"][-1] = Loss_rollouts
            losses["Neps"][-1] = ep+1
            losses["Nspl"][-1] = Ntn
            losses["t_train"][-1] = t_train

            # Save Loss
            torch.save(losses,losses_path)

def unlock_networks(student:Pilot,
                  target:Literal["All","None","Parameter","Odometry","Commander"]):
    """
    Locks/Unlocks the networks based on the training mode.

    """

    if target == "All":
        for param in student.model.parameters():
            param.requires_grad = True
    else:
        for param in student.model.parameters():
            param.requires_grad = False

        if target == "None":
            return
        elif target == "Parameter":
            for param in student.model.get_network["Parameter"]["Unlock"].parameters():
                param.requires_grad = True
        elif target == "Commander":
            for param in student.model.get_network["Commander"]["Unlock"].parameters():
                param.requires_grad = True
        elif target == "Odometry":
            for param in student.model.get_network["Odometry"]["Unlock"].parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid Training Mode")


def train_rl_policy(cohort_name: str, roster: List[str],
                   method_name: str, flights: List[Tuple[str, str]],
                   Neps: int = 50, lim_sv: int = 5,
                   lr_rl: float = 1e-5, lr_value: float = 1e-4,
                   batch_size: int = 64,
                   collision_penalty: float = 5.0,
                   clearance_threshold: float = 0.3,
                   lambda_bc: float = 2.0,
                   lambda_ref: float = 1.0,
                   lambda_value: float = 0.5,
                   gamma: float = 0.95,
                   gae_lambda: float = 0.95,
                   lambda_contrastive: float = 0.5,
                   contrastive_margin: float = 1.0,
                   lambda_dpg: float = 1.0,
                   train_on_failures_only: bool = False,
                   bc_weighting_mode: str = "onset",
                   advantage_method: str = "gae",
                   mc_reward_config: dict = None,
                   validation_config: dict = None,
                   max_trajectories: int = 20,
                   validation_percentage: float = 0.5):
    """
    Train pilots using reinforcement learning with behavior cloning regularization.

    This function fine-tunes behavior-cloned policies using RL objectives focused on
    collision avoidance. The training loop follows a hybrid approach:
    - Collect rollouts with the learned policy
    - Detect collisions and compute per-timestep clearances
    - Compute RL rewards based on collision avoidance
    - Update policy with combined RL + BC + reference policy regularization losses

    Args:
        cohort_name: Name of the cohort
        roster: List of pilot names to train
        method_name: Name of the method configuration
        flights: List of (scene_name, course_name) tuples for rollout generation
        Neps: Number of RL episodes (default: 50)
        lim_sv: Save checkpoint every N episodes (default: 5)
        lr_rl: Learning rate for policy gradients (default: 1e-5, 10x lower for fine-tuning stability)
        lr_value: Learning rate for value function (default: 1e-4, 10x faster than actor)
        batch_size: Batch size for gradient updates (default: 64)
        collision_penalty: Weight for collision penalty in reward (default: 5.0)
        clearance_threshold: Distance threshold for collision detection (default: 0.3m)
        lambda_bc: Weight for behavior cloning regularization loss (default: 2.0, high to prevent forgetting)
        lambda_ref: Weight for reference policy regularization (MSE to frozen BC policy) (default: 1.0, high)
        lambda_value: Weight for critic loss (value function learning) (default: 0.5)
        gamma: Discount factor for returns (default: 0.95, shorter horizon for collision avoidance)
        gae_lambda: GAE lambda for advantage estimation (default: 0.95)

    Returns:
        None (saves checkpoints to cohort roster directories)
    """
    # Import RL-specific modules
    from sousvide.rl.collision_detector import CollisionDetector, compute_collision_rewards
    from sousvide.flight.deploy_ssv import simulate_roster
    from sousvide.visualize.analyze_simulated_experiments import analyze_trajectory_performance
    import figs.utilities.trajectory_helper as th
    from figs.utilities.display_config import set_figure_display

    # CRITICAL: Disable figure display during RL training to prevent memory leaks
    set_figure_display(False)

    progress = rv.get_training_progress()
    train_desc = "[bold dark_green]RL Training Progress[/]"

    with progress:
        # Create outer task for pilots
        outer_task = progress.add_task(train_desc, total=len(roster), loss=0.0, units="pilots")

        for pilot_name in roster:
            # Load pilot (pre-trained BC policy)
            pilot = Pilot(cohort_name, pilot_name)
            pilot.set_mode('deploy')

            # Create pilot progress bar
            pilot_desc = f"[bold green3]{pilot_name:>8}[/]"
            pilot_task = progress.add_task(pilot_desc, total=Neps, loss=0.0, units="episodes")
            pilot_bar = (progress, pilot_task)

            # Train the pilot with RL
            train_rl_pilot(cohort_name, [pilot_name], method_name, flights, Neps, lim_sv,
                          lr_rl, lr_value, batch_size,
                          collision_penalty, clearance_threshold,
                          lambda_bc, lambda_ref,
                          gamma,
                          train_on_failures_only,
                          bc_weighting_mode,
                          advantage_method, 
                          mc_reward_config,
                          validation_percentage,
                          max_trajectories,
                          pilot_bar)

            # Update outer task
            progress.update(outer_task, advance=1)
            progress.refresh()


def train_rl_pilot(cohort_name: str, roster: List[str], 
                   method_name: str, flights: List[Tuple[str, str]], Neps: int, lim_sv: int,
                  lr_rl: float, lr_value: float, batch_size: int,
                  collision_penalty: float, clearance_threshold: float,
                  lambda_bc: float, lambda_ref: float,
                  gamma: float,
                  train_on_failures_only: bool = False,
                  bc_weighting_mode: str = "onset",
                  advantage_method: str = "gae",
                  mc_reward_config: dict = None,
                  validation_percentage: float = 0.1,
                  max_trajectories: int = 20,
                  progress_bar: tuple[Progress, int] | None = None):
    """
    Train a single pilot using RL with BC regularization.
    """
    import json
    import yaml
    import os
    from sousvide.rl.collision_detector import CollisionDetector, compute_collision_rewards
    from sousvide.flight.deploy_ssv import simulate_rollouts
    from sousvide.visualize.analyze_simulated_experiments import analyze_trajectory_performance
    from figs.simulator import Simulator
    import figs.scene_editing.scene_editing_utils as scdt


    # Load pilot from roster (expect single pilot name)
    if len(roster) != 1:
        raise ValueError(f"train_rl_pilot expects a single pilot, got {len(roster)}")
    pilot_name = roster[0]
    pilot = Pilot(cohort_name, pilot_name)
    pilot.set_mode('deploy')

    # Pytorch config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Send to GPU
    pilot.model.to(device)

    # Get workspace path and cohort path
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(pilot.path))))
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)

    # Load scene configs for environment information
    scenes_cfg_dir = os.path.join(workspace_path, "configs", "scenes")

    # ====================================================================
    # Load GSplat and point clouds
    # ====================================================================
    print("=" * 70)
    print("ONE-TIME SETUP: Loading scenes and point clouds")
    print("=" * 70)

    # Load method config to get rollout name
    method_path = os.path.join(workspace_path, "configs", "method", method_name + ".json")
    with open(method_path) as json_file:
        method_config = json.load(json_file)

    sample_set_config = method_config["sample_set"]
    rollout_name = sample_set_config["rollout"]
    print(f"Using rollout config: {rollout_name}")

    collision_detectors = {}
    scene_configs = {}
    objective_configs = {}

    for scene_name, course_name in flights:
        scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)

        objectives = scene_cfg["queries"]

        # Store scene config for later use
        scene_configs[scene_name] = scene_cfg
        objective_configs[scene_name] = objectives

        # Load GSplat and extract point cloud ONCE (before training loop)
        try:
            print(f"\n[{scene_name}] Loading 3D Gaussian Splatting scene...")
            simulator = Simulator(scene_name, rollout_name)

            print(f"[{scene_name}] Extracting environment point cloud...")
            _, env_pcd_array, env_bounds, _, _, _ = scdt.rescale_point_cloud(
                simulator.gsplat,
                viz=False,
                cull=False,
                verbose=False
            )

            # Initialize collision detector with real point cloud
            collision_detectors[scene_name] = CollisionDetector(
                point_cloud=env_pcd_array,
                collision_radius=clearance_threshold
            )

            print(f"[{scene_name}] ✓ Point cloud loaded: {env_pcd_array.shape[0]:,} points")

            # GSplat and simulator can now be garbage collected
            # We only need the point cloud in the collision detector

        except Exception as e:
            print(f"[{scene_name}] ✗ Error loading scene: {e}")
            print(f"[{scene_name}]   Falling back to placeholder point cloud")
            import traceback
            traceback.print_exc()
            collision_detectors[scene_name] = CollisionDetector(
                point_cloud=np.random.randn(1000, 3),
                collision_radius=clearance_threshold
            )

    print("\n" + "=" * 70)
    print("Setup complete! Point clouds cached in memory for all episodes.")
    print("=" * 70 + "\n")


    # Setup loss tracking
    pilot_path = pilot.path
    rl_model_path = os.path.join(pilot_path, "rl_model.pth")
    rl_value_path = os.path.join(pilot_path, "rl_value.pth")
    rl_losses_path = os.path.join(pilot_path, "rl_losses.pt")
    rl_best_model_path = os.path.join(pilot_path, "rl_best_model.pth")

    print(f"RL Training Pilot: {pilot.name}")

    # Initialize loss tracking
    losses = {
        "collision_rate": [],
        "mean_clearance": [],
        "val_collision_rate": [],      # Validation metrics
        "val_mean_clearance": [],      # Validation metrics
        "loss_rl": [],
        "loss_bc": [],
        "loss_ref": [],                # Reference policy regularization (MSE to frozen BC policy)
        "loss_value": [],              # Value loss (critic)
        "loss_contrastive": [],        # Contrastive Q-learning loss
        "loss_total": [],
        "episodes": []
    }

    if os.path.exists(rl_losses_path):
        losses = torch.load(rl_losses_path)
        start_episode = len(losses["episodes"])
        print(f"✓ Continuing RL training from episode {start_episode}/{Neps}")
        print(f"  Previous best collision rate: {min(losses['collision_rate']):.2%}")
    else:
        start_episode = 0
        print("Starting new RL training.")
    
    # If we've already completed all episodes, exit
    if start_episode >= Neps:
        print(f"✓ Training already complete ({start_episode}/{Neps} episodes)")
        return

    # Save initial BC policy as reference for regularization
    pilot_bc = Pilot(cohort_name, pilot.name)
    pilot_bc.set_mode('deploy')

    # HistoryEncoder left unaltered
    for param in pilot.model.network["HistoryEncoder"].parameters():
        param.requires_grad = False
    print("\n[FROZEN] HistoryEncoder frozen - temporal encoding fixed from BC training")
    
    actor_params = list(pilot.model.network["CommanderSV"].parameters())
    vision_params = list(pilot.model.network["VisionMLP"].parameters())
    
    # Create separate optimizers
    # Actor optimizer: updates CommanderSV + VisionMLP
    # Vision features are shaped by policy objectives only
    opt_actor = optim.Adam(actor_params + vision_params, lr=lr_rl)
    
    # Enable automatic mixed precision to reduce GPU memory usage
    from torch.cuda.amp import autocast, GradScaler
    use_amp = torch.cuda.is_available()
    scaler_actor = GradScaler(enabled=use_amp)
    if use_amp:
        print("\n[AMP] Automatic Mixed Precision enabled - reducing GPU memory usage")
    
    print(f"\n[DEBUG] Optimizer setup:")
    print(f"  Actor LR: {lr_rl:.2e}")
    print(f"    - CommanderSV: {len(actor_params)} param groups")
    print(f"    - VisionMLP: {len(vision_params)} param groups")

    print("=" * 70)

    try:
        from sousvide.rl import (
            move_simulation_results,
            load_simulation_results,
            prepare_batch_data,
            CollisionDetector,
            compute_simple_advantage,
            compute_advantages_gae,
            compute_advantages_mc,
            compute_mc_rewards,
            normalize_advantages,
            select_high_risk_states,
            prepare_state_batch
        )
    except ImportError as e:
        print(f"Error importing RL modules: {e}")
        raise e

    # ====================================================================
    # GENERATE FIXED VALIDATION DATA & INITIAL EVALUATION
    # ====================================================================
    validation_data_dir = os.path.join(cohort_path, "validation_data")
    simulate_rollouts(
        workspace_path=workspace_path,
        cohort_name=cohort_name,
        cohort_path=cohort_path,
        method_name=method_name,
        pilot=pilot,
        flights=flights,
        scenes_cfg_dir=scenes_cfg_dir,
        objectives_all=objective_configs,
        max_trajectories=max_trajectories,
        review=True,  # Use the RRT paths we just generated
        disable_visualization=True,
        show_progress=True,
        progress_bar=progress_bar,
    )
    # Close any matplotlib figures that may have been created during simulation
    plt.close('all')

    move_simulation_results(cohort_path, validation_data_dir)
    # print(f"\n[VALIDATION] Fixed validation data generated with {validation_trajectories} trajectories per flight.")
    trajectories, metadata, raw_data = load_simulation_results(validation_data_dir, pilot_name=pilot.name)
    if not trajectories:
        raise RuntimeError(f"No trajectories found for pilot {pilot.name} in episode {episode + 1}")

    total_trajectories = len(trajectories)
    validation_trajectories = int(total_trajectories * validation_percentage)
    print(f"\nTrajectory Configuration:")
    print(f"  Training: {total_trajectories} trajectories per episode")
    print(f"  Validation: {validation_trajectories} trajectories ({validation_percentage:.0%} of training)")

    # ====================================================================
    # BATCHED PROCESSING: Load and analyze trajectories in batches to avoid OOM
    # ====================================================================
    batch_size = 32  # Process up to 32 trajectories at a time
    print(f"[MEMORY] Processing {total_trajectories} trajectories in batches of {batch_size}")

    collision_rate_ep = 0.0
    mean_clearance_ep = 0.0
    success_indices = []
    failure_indices = []
    all_analyses = []
    all_clearances = []

    for batch_start in range(0, total_trajectories, batch_size):
        batch_end = min(batch_start + batch_size, total_trajectories)
        batch_trajectories = trajectories[batch_start:batch_end]
        batch_metadata = {key: metadata[key][batch_start:batch_end] if isinstance(metadata[key], list) else metadata[key]
                         for key in metadata}

        print(f"  [Batch {batch_start//batch_size + 1}] Processing trajectories {batch_start+1}-{batch_end}")

        # Prepare batch data (compute rewards, clearances, etc.)
        batch_data = prepare_batch_data(
            batch_trajectories,
            batch_metadata,
            collision_detectors[list(scene_configs.keys())[0]],
            device
        )

        # Accumulate collision metrics
        collision_rate_ep += batch_data["collision_rate"] * len(batch_trajectories)
        mean_clearance_ep += batch_data["mean_clearance"] * len(batch_trajectories)

        # Accumulate all analyses and clearances
        all_analyses.extend(batch_data["analyses"])
        all_clearances.extend(batch_data["clearances"])

        # Track success/failure indices (adjust for batch offset)
        for local_idx, analysis in enumerate(batch_data["analyses"]):
            global_idx = batch_start + local_idx
            if analysis["collision"]:
                failure_indices.append(global_idx)
            else:
                success_indices.append(global_idx)

        # Free memory after each batch
        del batch_trajectories, batch_metadata, batch_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average the metrics
    collision_rate_ep /= max(1, total_trajectories)
    mean_clearance_ep /= max(1, total_trajectories)

    # Reconstruct batch_data from accumulated results (needed for subsequent code)
    batch_data = {
        "analyses": all_analyses,
        "clearances": all_clearances,
        "collision_rate": collision_rate_ep,
        "mean_clearance": mean_clearance_ep
    }

    # Calculate proportional counts
    total_traj = total_trajectories
    success_ratio = len(success_indices) / max(1, total_traj)
    failure_ratio = len(failure_indices) / max(1, total_traj)
    
    num_success = max(1, int(validation_trajectories * success_ratio))
    num_failure = max(1, int(validation_trajectories * failure_ratio))
    
    # Adjust if rounding causes mismatch
    if num_success + num_failure > validation_trajectories:
        num_failure = validation_trajectories - num_success
    elif num_success + num_failure < validation_trajectories:
        num_success = validation_trajectories - num_failure
    
    # Pick in order (first N from each list)
    validation_success = success_indices[:min(num_success, len(success_indices))]
    validation_failure = failure_indices[:min(num_failure, len(failure_indices))]
    
    trajectory_indices = sorted(validation_success + validation_failure)
    
    print(f"[VALIDATION] Selected {len(trajectory_indices)} trajectories: "
          f"{len(validation_success)} success ({success_ratio:.1%}), "
          f"{len(validation_failure)} failure ({failure_ratio:.1%})")

    # Clean up validation directory to only keep selected trajectories
    rrl.filter_validation_trajectories(validation_data_dir, pilot.name, trajectory_indices)

    # ====================================================================
    # OPTIMIZATION: Filter RRT .pkl files to only contain selected trajectories
    # This avoids re-simulating unnecessary trajectories during validation
    # ====================================================================
    print(f"\n[OPTIMIZATION] Filtering RRT .pkl files to selected {len(trajectory_indices)} trajectories...")

    # Build mapping from global trajectory indices to (scene, objective, local_idx)
    import re
    # Navigate through validation_data_dir/simulation_data/{timestamp}/trajectories/
    validation_sim_data_dir = os.path.join(validation_data_dir, "simulation_data")
    validation_ts_dirs = sorted(glob.glob(os.path.join(validation_sim_data_dir, "*")))
    if not validation_ts_dirs:
        raise FileNotFoundError(f"No timestamp directories found in {validation_sim_data_dir}")
    validation_ts_dir = validation_ts_dirs[-1]
    validation_traj_dir = os.path.join(validation_ts_dir, "trajectories")
    traj_files = sorted(glob.glob(os.path.join(validation_traj_dir, f"*_traj*_{pilot.name}.pt")))

    trajectory_map = []  # List of (global_idx, scene, obj, local_idx)
    for global_idx, traj_file in enumerate(traj_files):
        filename = os.path.basename(traj_file)
        # Extract scene+objective and local trajectory index
        # Pattern: sim_data_{scene}_{obj}_traj{idx}_{pilot}.pt
        parts = filename.replace(f"sim_data_", "").replace(f"_traj", "_TRAJ_MARKER_").replace(f"_{pilot.name}.pt", "").split("_TRAJ_MARKER_")
        if len(parts) == 2:
            scene_obj = parts[0]  # e.g., "sv_917_3_left_gemsplat_microwave"
            local_idx = int(parts[1])

            # Extract scene and objective by splitting at last underscore
            scene_obj_parts = scene_obj.rsplit("_", 1)
            if len(scene_obj_parts) == 2:
                scene = scene_obj_parts[0]
                obj = scene_obj_parts[1]
                trajectory_map.append((global_idx, scene, obj, local_idx))

    # Group selected trajectories by (scene, objective)
    from collections import defaultdict
    pkl_indices = defaultdict(list)  # {(scene, obj): [local_idx1, local_idx2, ...]}

    for global_idx in trajectory_indices:
        if global_idx < len(trajectory_map):
            _, scene, obj, local_idx = trajectory_map[global_idx]
            pkl_indices[(scene, obj)].append(local_idx)

    # Sort indices for each pkl file
    for key in pkl_indices:
        pkl_indices[key] = sorted(pkl_indices[key])

    print(f"  Mapped {len(trajectory_indices)} global indices to {len(pkl_indices)} .pkl files")

    # Filter each .pkl file to keep only selected trajectories
    validation_rrt_dir = os.path.join(validation_ts_dir, "rrt_planning")
    for (scene, obj), local_indices in pkl_indices.items():
        pkl_file = os.path.join(validation_rrt_dir, f"{scene}_filtered_{obj}.pkl")

        if not os.path.exists(pkl_file):
            print(f"  [WARNING] .pkl file not found: {pkl_file}")
            continue

        # Load full trajectory list
        with open(pkl_file, "rb") as f:
            all_trajectories = pickle.load(f)

        # Filter to selected indices
        filtered_trajectories = [all_trajectories[i] for i in local_indices if i < len(all_trajectories)]

        # Save filtered list back
        with open(pkl_file, "wb") as f:
            pickle.dump(filtered_trajectories, f)

        print(f"  {scene}/{obj}: Filtered {len(all_trajectories)} → {len(filtered_trajectories)} trajectories")

    print(f"[OPTIMIZATION] RRT .pkl files filtered successfully!")

    # for traj_idx in trajectory_indices:
    #     rewards = batch_data["rewards"][traj_idx]
    #     clearances = batch_data["clearances"][traj_idx]

    #     if traj_idx < len(raw_data):
    #         try:
    #             # Generate observations using OODA pipeline
    #             observations = rrl.generate_observations_from_trajectory(
    #                 raw_data[traj_idx], pilot, device
    #             )

    #             # Extract commander inputs using network's built-in extractor
    #             batched_inputs = rrl.extract_commander_inputs_from_observations(
    #                 observations, pilot, device
    #             )
    #             tx_com_batch, obj_com_batch, dxu_par_batch, img_vis_batch, tx_vis_batch = batched_inputs

    #             # Align rewards with observations
    #             # Observations are generated for states 1..N (skipping initial state 0)
    #             # But rewards are computed for states 0..N (including initial state)
    #             # Skip the first reward to match observations
    #             if len(rewards) == len(tx_com_batch) + 1:
    #                 rewards = rewards[1:]  # Skip reward for initial state
    #                 clearances = clearances[1:]  # Also skip initial clearance

    #             # Accumulate inputs for mini-batch training
    #             all_tx_com.append(tx_com_batch)
    #             all_obj_com.append(obj_com_batch)
    #             all_dxu_par.append(dxu_par_batch)
    #             all_img_vis.append(img_vis_batch)
    #             all_tx_vis.append(tx_vis_batch)

    #             # Store trajectory states and rewards for on-the-fly advantage computation
    #             # Advantages will be computed during mini-batch training with values from forward pass
    #             all_trajectory_states.append(batched_inputs)
    #             all_trajectory_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
    #             # Use actual tensor length, not reward length
    #             trajectory_lengths.append(len(tx_com_batch))

    #             print(f"  Trajectory {traj_idx}: Stored {len(tx_com_batch)} states and {len(rewards)} rewards")

    #         except Exception as e:
    #             import traceback
    #             print(f"  Warning: Failed to process trajectory {traj_idx}: {e}")
    #             if "[DEBUG]" in str(e):  # Our debug errors
    #                 traceback.print_exc()
    #             print("  Skipping this trajectory")
    #     else:
    #         # Missing raw data - skip this trajectory
    #         print(f"  Trajectory {traj_idx}: Missing raw data, skipping")

    # # Verify we have trajectory data
    # if len(all_trajectory_states) == 0:
    #     raise RuntimeError(f"No valid trajectories found for validation dataset of pilot {pilot.name}")

    # print(f"[DEBUG] Collected {len(all_trajectory_states)} validation trajectories with total {sum(trajectory_lengths)} timesteps")

    # print("[DEBUG] Loading expert trajectories for BC loss...")

    # # Load policy-expert trajectory pairs and prepare per-trajectory expert data
    # all_expert_data = []  # List of (expert_actions, bc_weights) tuples, one per trajectory
    # try:
    #     from sousvide.rl import (
    #         load_policy_expert_pairs,
    #         compute_state_divergence,
    #         compute_onset_signals
    #     )

    #     matched_pairs = load_policy_expert_pairs(cohort_path, pilot.name)
    #     print(f"[DEBUG] Found {len(matched_pairs)} policy-expert trajectory pairs")

    #     # Process each matched pair to extract expert actions and BC weights
    #     for pair_idx, pair in enumerate(matched_pairs):
    #         if pair_idx >= len(all_trajectory_states):
    #             # More expert pairs than policy trajectories we collected
    #             break

    #         policy_traj = pair['policy']
    #         expert_traj = pair['expert']

    #         # Extract states for divergence computation
    #         policy_states = policy_traj.get('Xro', None)
    #         expert_states = expert_traj.get('Xro', None)

    #         if policy_states is None or expert_states is None:
    #             print(f"  [WARNING] Missing states for pair {pair_idx}, skipping BC for this trajectory")
    #             all_expert_data.append(None)
    #             continue

    #         try:
    #             # Expert Uro contains MPC actions
    #             expert_uro = expert_traj.get('Uro', None)
                
    #             if expert_uro is None:
    #                 raise ValueError("Missing expert Uro actions")
                
    #             # Expert actions are Uro transposed to (N, 4) for timestep-action matching
    #             # Uro shape: (4, N) where N is number of timesteps
    #             # Convert to (N, 4) tensor for comparison with policy outputs
    #             expert_actions = torch.tensor(
    #                 expert_uro.T,  # Transpose (4, N) -> (N, 4)
    #                 dtype=torch.float32,
    #                 device=device
    #             )

    #             # Compute state divergence for onset signal computation
    #             divergence = compute_state_divergence(policy_states, expert_states)

    #             # Store expert data for this trajectory (weights will be computed on-the-fly)
    #             # We'll compute bc_weights after we have values from the forward pass
    #             all_expert_data.append({
    #                 'expert_actions': expert_actions,  # Now actual MPC actions from Uro
    #                 'divergence': divergence,
    #                 'policy_states': policy_states,
    #                 'expert_states': expert_states
    #             })

    #             if pair_idx < 3:
    #                 print(f"  [DEBUG] Pair {pair_idx}: Loaded {len(expert_actions)} expert MPC actions from Uro")

    #         except Exception as e:
    #             print(f"  [WARNING] Failed to process expert trajectory {pair_idx}: {e}")
    #             all_expert_data.append(None)

    #     print(f"[DEBUG] Successfully loaded expert data for {sum(1 for x in all_expert_data if x is not None)} trajectories")

    # except Exception as e:
    #     print(f"[WARNING] Failed to load expert trajectories: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     all_expert_data = []

    # ====================================================================
    # SAVE EPISODE 0 VALIDATION METRICS (Pre-finetune baseline)
    # ====================================================================
    # Compute collision rate from selected validation trajectories
    selected_analyses = [batch_data["analyses"][i] for i in trajectory_indices]
    val_collision_count = sum(1 for analysis in selected_analyses if analysis["collision"])
    val_collision_rate_ep0 = val_collision_count / len(selected_analyses) if selected_analyses else 0.0

    # Compute mean clearance from selected validation trajectories
    selected_clearances = [batch_data["clearances"][i] for i in trajectory_indices]
    all_clearances = np.concatenate(selected_clearances) if selected_clearances else np.array([])
    val_mean_clearance_ep0 = float(np.mean(all_clearances)) if len(all_clearances) > 0 else 0.0

    print(f"\n[EPISODE 0 - Pre-finetune] Validation metrics:")
    print(f"  Collision rate: {val_collision_rate_ep0:.2%} ({val_collision_count}/{len(selected_analyses)} trajectories)")
    print(f"  Mean clearance: {val_mean_clearance_ep0:.3f}m")

    # Save to losses dict
    losses["val_collision_rate"].append(val_collision_rate_ep0)
    losses["val_mean_clearance"].append(val_mean_clearance_ep0)
    losses["episodes"].append(0)
    # Episode 0 training metrics (baseline = validation metrics)
    losses["collision_rate"].append(val_collision_rate_ep0)
    losses["mean_clearance"].append(val_mean_clearance_ep0)
    losses["loss_rl"].append(0.0)
    losses["loss_bc"].append(0.0)
    losses["loss_ref"].append(0.0)
    losses["loss_total"].append(0.0)

    print(f"[EPISODE 0] Saved baseline validation metrics to losses dict")

    # ====================================================================
    # MAIN RL TRAINING LOOP
    # ====================================================================
    for episode in range(start_episode, Neps):
        print(f"\n--- Episode {episode + 1}/{Neps} ---")
        # ====================================================================
        # STEP 1: Collect rollouts with current policy
        # ====================================================================
        print("Collecting rollouts...")
        
        from sousvide.flight.deploy_ssv import simulate_rollouts

        simulate_rollouts(
            workspace_path=workspace_path,
            cohort_name=cohort_name, cohort_path=cohort_path,
            method_name=method_name,
            pilot=pilot,
            flights=flights, scenes_cfg_dir=scenes_cfg_dir, objectives_all=objective_configs,
            max_trajectories=max_trajectories,  
            review=True,  
            disable_visualization=True,
            show_progress=True,  
            progress_bar=progress_bar
        )

        # ====================================================================
        # STEP 2: Load trajectories and analyze for collision metrics
        # ====================================================================
        print("Loading and analyzing trajectories...")
        # Load simulation results (includes raw_data with Iro for critic)
        # Only load POLICY trajectories (not expert) to compute advantages
        trajectories, metadata, raw_data = load_simulation_results(cohort_path, pilot_name=pilot.name)

        if not trajectories:
            raise RuntimeError(f"No trajectories found for pilot {pilot.name} in episode {episode + 1}")

        # Close any matplotlib figures that may have been created during simulation
        plt.close('all')

        # Collision detectors already loaded during setup phase (no reload needed!)
        print(f"Training on {len(trajectories)} trajectories")

        # ====================================================================
        # BATCHED BATCH DATA PREPARATION: Avoid loading all trajectories at once
        # ====================================================================
        batch_size = 32  # Process 32 trajectories at a time
        print(f"[MEMORY] Preparing batch data in batches of {batch_size}")

        all_rewards = []
        all_clearances = []
        all_analyses = []
        collision_rate_ep = 0.0
        mean_clearance_ep = 0.0

        for batch_start in range(0, len(trajectories), batch_size):
            batch_end = min(batch_start + batch_size, len(trajectories))
            batch_trajectories = trajectories[batch_start:batch_end]
            batch_metadata = {key: metadata[key][batch_start:batch_end] if isinstance(metadata[key], list) else metadata[key]
                             for key in metadata}

            print(f"  [Batch {batch_start//batch_size + 1}] Processing trajectories {batch_start+1}-{batch_end}")

            # Prepare batch data (compute rewards, clearances, etc.)
            batch_data = prepare_batch_data(
                batch_trajectories,
                batch_metadata,
                collision_detectors[list(scene_configs.keys())[0]],  # Use first detector
                device
            )

            # Accumulate data
            all_rewards.extend(batch_data["rewards"])
            all_clearances.extend(batch_data["clearances"])
            all_analyses.extend(batch_data["analyses"])
            collision_rate_ep += batch_data["collision_rate"] * len(batch_trajectories)
            mean_clearance_ep += batch_data["mean_clearance"] * len(batch_trajectories)

            # Free memory after each batch
            del batch_trajectories, batch_metadata, batch_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Reconstruct batch_data from accumulated results
        collision_rate_ep /= max(1, len(trajectories))
        mean_clearance_ep /= max(1, len(trajectories))
        batch_data = {
            "rewards": all_rewards,
            "clearances": all_clearances,
            "analyses": all_analyses,
            "collision_rate": collision_rate_ep,
            "mean_clearance": mean_clearance_ep
        }

        print(f"Episode {episode + 1}: Collision rate = {collision_rate_ep:.2%}, "
                f"Mean clearance = {mean_clearance_ep:.3f}m")
        
        # ====================================================================
        # STEP 3: Compute advantages and prepare training data
        # ====================================================================
        print("Computing advantages...")

        # Store trajectory data for on-the-fly advantage computation during training
        all_trajectory_states = []  # Store state inputs for each trajectory
        all_trajectory_rewards = []  # Store rewards for each trajectory
        trajectory_lengths = []  # Track actual length of each trajectory

        # Accumulate batched inputs across all trajectories for mini-batch training
        all_tx_com = []
        all_obj_com = []
        all_dxu_par = []
        all_img_vis = []
        all_tx_vis = []

        # Filter trajectories if training only on failures
        trajectory_indices = list(range(len(batch_data["rewards"])))
        if train_on_failures_only:
            # Only include trajectories where policy encountered collision
            trajectory_indices = [i for i, analysis in enumerate(batch_data["analyses"])
                                 if analysis["collision"]]
            print(f"[DEBUG] Training on failures only: {len(trajectory_indices)}/{len(batch_data['analyses'])} trajectories had collisions")
            if not trajectory_indices:
                print(f"[WARNING] No failed trajectories in this episode. Skipping training.")
                continue

        for traj_idx in trajectory_indices:
            rewards = batch_data["rewards"][traj_idx]
            clearances = batch_data["clearances"][traj_idx]

            if traj_idx < len(raw_data):
                try:
                    # Generate observations using OODA pipeline (matching BC training)
                    observations = rrl.generate_observations_from_trajectory(
                        raw_data[traj_idx], pilot, device
                    )

                    # Extract commander inputs using network's built-in extractor
                    batched_inputs = rrl.extract_commander_inputs_from_observations(
                        observations, pilot, device
                    )
                    tx_com_batch, obj_com_batch, dxu_par_batch, img_vis_batch, tx_vis_batch = batched_inputs

                    # Align rewards with observations
                    # Observations are generated for states 1..N (skipping initial state 0)
                    # But rewards are computed for states 0..N (including initial state)
                    # Skip the first reward to match observations
                    if len(rewards) == len(tx_com_batch) + 1:
                        rewards = rewards[1:]  # Skip reward for initial state
                        clearances = clearances[1:]  # Also skip initial clearance

                    # Accumulate inputs for mini-batch training
                    all_tx_com.append(tx_com_batch)
                    all_obj_com.append(obj_com_batch)
                    all_dxu_par.append(dxu_par_batch)
                    all_img_vis.append(img_vis_batch)
                    all_tx_vis.append(tx_vis_batch)

                    # Store trajectory states and rewards for on-the-fly advantage computation
                    # Advantages will be computed during mini-batch training with values from forward pass
                    all_trajectory_states.append(batched_inputs)
                    all_trajectory_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
                    # Use actual tensor length, not reward length
                    trajectory_lengths.append(len(tx_com_batch))

                    print(f"  Trajectory {traj_idx}: Stored {len(tx_com_batch)} states and {len(rewards)} rewards")

                except Exception as e:
                    import traceback
                    print(f"  Warning: Failed to process trajectory {traj_idx}: {e}")
                    if "[DEBUG]" in str(e):  # Our debug errors
                        traceback.print_exc()
                    print("  Skipping this trajectory")
            else:
                # Missing raw data - skip this trajectory
                print(f"  Trajectory {traj_idx}: Missing raw data, skipping")

        # Verify we have trajectory data
        if len(all_trajectory_states) == 0:
            print("[ERROR] No valid trajectories collected! Skipping episode.")
            continue

        print(f"[DEBUG] Collected {len(all_trajectory_states)} trajectories with total {sum(trajectory_lengths)} timesteps")

        # Delete policy trajectories and metadata now that we've extracted all needed data
        # This frees significant memory before loading expert trajectories
        del trajectories, metadata, raw_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ====================================================================
        # STEP 3.5: Load expert trajectories for BC loss (per-trajectory matching)
        # ====================================================================
        print("[DEBUG] Loading expert trajectories for BC loss...")

        # Load policy-expert trajectory pairs and prepare per-trajectory expert data
        all_expert_data = []  # List of (expert_actions, bc_weights) tuples, one per trajectory
        try:
            from sousvide.rl import (
                get_policy_expert_file_paths,
                compute_state_divergence,
                compute_onset_signals
            )

            # Only load expert trajectories for failed trajectories to save memory
            # trajectory_indices contains the indices of failed trajectories (from line 1105-1106)
            print(f"[MEMORY] Loading expert trajectories on-the-fly for {len(trajectory_indices)} failed trajectories")

            # Get file paths without loading data
            policy_file_paths, expert_file_paths = get_policy_expert_file_paths(
                cohort_path, pilot.name, policy_indices=trajectory_indices
            )
            print(f"[DEBUG] Found {len(policy_file_paths)} policy-expert trajectory pairs")

            # Process each trajectory by loading policy-expert pair on-the-fly
            for pair_idx, traj_idx in enumerate(trajectory_indices):
                if pair_idx >= len(all_trajectory_states):
                    # More expert pairs than policy trajectories we collected
                    break

                # Skip if we don't have file paths for this trajectory
                if traj_idx not in policy_file_paths or traj_idx not in expert_file_paths:
                    print(f"  [WARNING] Missing file paths for trajectory {traj_idx}, skipping BC for this trajectory")
                    all_expert_data.append(None)
                    continue

                try:
                    # Load policy and expert trajectories on-the-fly to minimize memory usage
                    policy_data_list = torch.load(policy_file_paths[traj_idx])
                    expert_data_list = torch.load(expert_file_paths[traj_idx])

                    # Extract first perturbation from each
                    policy_traj = policy_data_list[0] if policy_data_list else None
                    expert_traj = expert_data_list[0] if expert_data_list else None

                    if policy_traj is None or expert_traj is None:
                        print(f"  [WARNING] Empty trajectory data for pair {pair_idx}, skipping BC for this trajectory")
                        all_expert_data.append(None)
                        del policy_data_list, expert_data_list
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    # Extract states for divergence computation
                    policy_states = policy_traj.get('Xro', None)
                    expert_states = expert_traj.get('Xro', None)

                    if policy_states is None or expert_states is None:
                        print(f"  [WARNING] Missing states for pair {pair_idx}, skipping BC for this trajectory")
                        all_expert_data.append(None)
                        del policy_data_list, expert_data_list, policy_traj, expert_traj
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    # Expert Uro contains MPC actions
                    expert_uro = expert_traj.get('Uro', None)

                    if expert_uro is None:
                        raise ValueError("Missing expert Uro actions")

                    # Expert actions are Uro transposed to (N, 4) for timestep-action matching
                    # Uro shape: (4, N) where N is number of timesteps
                    # Convert to (N, 4) tensor for comparison with policy outputs
                    expert_actions = torch.tensor(
                        expert_uro.T,  # Transpose (4, N) -> (N, 4)
                        dtype=torch.float32,
                        device=device
                    )

                    # Compute state divergence for onset signal computation
                    divergence = compute_state_divergence(policy_states, expert_states)

                    # Store expert data for this trajectory (weights will be computed on-the-fly)
                    # We'll compute bc_weights after we have values from the forward pass
                    all_expert_data.append({
                        'expert_actions': expert_actions,  # Now actual MPC actions from Uro
                        'divergence': divergence,
                        'policy_states': policy_states,
                        'expert_states': expert_states
                    })

                    if pair_idx < 3:
                        print(f"  [DEBUG] Pair {pair_idx}: Loaded {len(expert_actions)} expert MPC actions from Uro")

                except Exception as e:
                    print(f"  [WARNING] Failed to process expert trajectory {pair_idx}: {e}")
                    all_expert_data.append(None)

                finally:
                    # Always delete loaded data to free memory before next iteration
                    try:
                        del policy_data_list, expert_data_list, policy_traj, expert_traj
                    except:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print(f"[DEBUG] Successfully loaded expert data for {sum(1 for x in all_expert_data if x is not None)} trajectories")

        except Exception as e:
            print(f"[WARNING] Failed to load expert trajectories: {e}")
            import traceback
            traceback.print_exc()
            all_expert_data = []

        # ====================================================================
        # STEP 4: Pre-compute advantages, BC weights, and critical states per trajectory
        # ====================================================================
        # Clear CUDA cache to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_timesteps = sum(trajectory_lengths)
        num_trajectories = len(all_trajectory_states)
        print(f"[DEBUG] Pre-computing advantages for {num_trajectories} trajectories with {total_timesteps} total timesteps")
        
        # Episode-wide lists (flattened across all trajectories)
        episode_advantages = []
        episode_returns = []
        episode_bc_weights = []
        critical_timestep_indices = []  # Global indices of critical timesteps for contrastive loss
        
        # Pre-compute per trajectory, then flatten (like old code)
        for traj_idx, (traj_states, traj_rewards, traj_len) in enumerate(
            zip(all_trajectory_states, all_trajectory_rewards, trajectory_lengths)):
            
            # Unpack trajectory state inputs
            tx_com_traj, obj_com_traj, dxu_par_traj, img_vis_traj, tx_vis_traj = traj_states
            
            # Verify consistency between trajectory_lengths and actual tensor sizes
            actual_traj_len = len(tx_com_traj)
            if actual_traj_len != traj_len:
                print(f"[ERROR] Trajectory {traj_idx}: trajectory_lengths says {traj_len}, but tensor has {actual_traj_len} timesteps!")
                print(f"  Rewards length: {len(traj_rewards)}")
                # Use actual tensor length
                traj_len = actual_traj_len
            
            # Compute Q-values for advantage estimation (no gradients needed here)
            with torch.no_grad():
                # Forward pass to get policy outputs
                policy_out, _ = pilot.model(
                    tx_com_traj, obj_com_traj, dxu_par_traj,
                    img_vis_traj, tx_vis_traj, return_value=False
                )
                
            # Compute advantages using selected method
            rewards_np = traj_rewards.cpu().numpy()
            
            if advantage_method == "monte_carlo":
                # Monte Carlo: use actual returns, no value network
                # Get MC reward config (with defaults)
                mc_cfg = mc_reward_config or {}
                
                # Get clearances for this trajectory from batch_data if available
                traj_clearances = batch_data["clearances"][traj_idx] if traj_idx < len(batch_data["clearances"]) else None
                
                # Check if trajectory had collision (stored in analyses)
                traj_analysis = batch_data["analyses"][traj_idx] if traj_idx < len(batch_data["analyses"]) else None
                traj_collision = traj_analysis["collision"] if traj_analysis else False
                collision_idx = traj_analysis["collision_index"] if traj_analysis and traj_collision else traj_len
                
                # Compute MC rewards with configurable components
                if traj_clearances is not None:
                    # Ensure clearances match observation length (traj_len)
                    clearances_arr = np.array(traj_clearances)
                    if len(clearances_arr) > traj_len:
                        clearances_arr = clearances_arr[:traj_len]
                    elif len(clearances_arr) < traj_len:
                        # Pad with last clearance value
                        pad_val = clearances_arr[-1] if len(clearances_arr) > 0 else 1.0
                        clearances_arr = np.concatenate([clearances_arr, np.full(traj_len - len(clearances_arr), pad_val)])
                    
                    # Adjust collision index to be within trajectory bounds
                    collision_idx = min(collision_idx, traj_len - 1) if traj_collision else traj_len
                    
                    mc_rewards = compute_mc_rewards(
                        clearances=clearances_arr,
                        collision_detected=traj_collision,
                        collision_index=collision_idx,
                        query_in_view=None,  # TODO: Add query visibility signal when available
                        use_collision=mc_cfg.get("use_collision", True),
                        use_clearance=mc_cfg.get("use_clearance", True),
                        use_query_in_view=mc_cfg.get("use_query_in_view", False),
                        collision_penalty=mc_cfg.get("collision_penalty", collision_penalty),
                        clearance_weight=mc_cfg.get("clearance_weight", 1.0),
                        query_reward=mc_cfg.get("query_reward", 0.5),
                        clearance_threshold=mc_cfg.get("clearance_threshold", clearance_threshold),
                        success_reward=mc_cfg.get("success_reward", 10.0)
                    )
                else:
                    raise ValueError(f"Trajectory {traj_idx}: Missing clearances for MC reward computation")
                
                # Compute MC advantages
                advantages, returns = compute_advantages_mc(mc_rewards, gamma=gamma)
                
                # Ensure advantages match traj_len
                if len(advantages) != traj_len:
                    if len(advantages) > traj_len:
                        advantages = advantages[:traj_len]
                        returns = returns[:traj_len]
                    else:
                        # Pad with zeros
                        advantages = np.concatenate([advantages, np.zeros(traj_len - len(advantages))])
                        returns = np.concatenate([returns, np.zeros(traj_len - len(returns))])
            else:
                raise ValueError(f"Unknown advantage_method: {advantage_method}. Use 'monte_carlo' or 'gae'.")
            
            # Normalize advantages for this trajectory
            advantages = normalize_advantages(advantages)
            
            # Compute BC weights for this trajectory
            bc_weights = np.zeros_like(advantages)
            if traj_idx < len(all_expert_data) and all_expert_data[traj_idx] is not None:
                expert_data = all_expert_data[traj_idx]
                divergence = expert_data['divergence']
                
                # Pad divergence to match policy trajectory length (expert may be shorter)
                # Divergence was computed only for min(len(policy), len(expert))
                # Pad with zeros (no divergence signal) for timesteps after expert finished
                if len(divergence) < len(advantages):
                    padding = np.zeros(len(advantages) - len(divergence))
                    divergence = np.concatenate([divergence, padding])
                elif len(divergence) > len(advantages):
                    # Truncate if somehow longer (shouldn't happen, but be safe)
                    divergence = divergence[:len(advantages)]
                
                # Compute BC weights based on selected mode
                if bc_weighting_mode == "onset":
                    # Onset weights: divergence rate × value drops × |advantages|
                    # More sophisticated - early warning signal
                    from sousvide.rl import compute_onset_signals
                    bc_weights = compute_onset_signals(
                        divergence, 
                        returns,  # Use returns from GAE
                        advantages  # Use advantages from GAE
                    )
                elif bc_weighting_mode == "advantage_u_shaped":
                    # Pure advantage weighting: |A(s,a)| normalized
                    # Simpler - AWAC-style
                    adv_abs = np.abs(advantages)
                    adv_max = adv_abs.max()
                    if adv_max > 1e-8:
                        bc_weights = adv_abs / adv_max
                    else:
                        bc_weights = np.zeros_like(advantages)
                elif bc_weighting_mode == "advantage_monotonic":
                    # Use max(0, mean(G) - G_t) to only weight collision-proximity steps
                    mean_return = np.mean(returns)
                    collision_proximity = np.maximum(0, mean_return - returns)
                    proximity_max = collision_proximity.max()
                    if proximity_max > 1e-8:
                        bc_weights = collision_proximity / proximity_max
                    else:
                        bc_weights = np.zeros_like(advantages)
                elif bc_weighting_mode == "uniform":
                    # Uniform weights: all timesteps weighted equally
                    bc_weights = np.ones_like(advantages)
                else:
                    raise ValueError(f"Unknown bc_weighting_mode: {bc_weighting_mode}. "
                                   f"Choose from: 'onset', 'advantage', 'uniform'")
            
            # Identify critical timestep for contrastive loss (if expert data exists)
            critical_idx = None
            if traj_idx < len(all_expert_data) and all_expert_data[traj_idx] is not None:
                expert_data = all_expert_data[traj_idx]
                
                # Compute backward values to identify critical state
                from sousvide.rl import backward_value_propagation
                
                policy_traj_dict = {
                    'Xro': expert_data['policy_states'],
                    'rewards': rewards_np
                }
                expert_rewards_len = len(expert_data['expert_states'])
                expert_rewards = rewards_np[:expert_rewards_len] if len(rewards_np) >= expert_rewards_len else np.zeros(expert_rewards_len)
                expert_traj_dict = {
                    'Xro': expert_data['expert_states'],
                    'rewards': expert_rewards
                }
                
                V_policy, V_expert, critical_idx = backward_value_propagation(
                    policy_traj_dict, expert_traj_dict, gamma=gamma
                )
            
            # Track which global timestep index gets contrastive loss
            global_start_idx = len(episode_advantages)
            if critical_idx is not None and critical_idx < len(advantages):
                # Store global index and which expert data to use
                critical_timestep_indices.append({
                    'global_idx': global_start_idx + critical_idx,
                    'expert_data_idx': traj_idx
                })
            
            # Flatten advantages, returns, and BC weights into episode-wide lists
            episode_advantages.extend(advantages)
            episode_returns.extend(returns)
            episode_bc_weights.extend(bc_weights)
            
            if traj_idx < 3:
                print(f"  Trajectory {traj_idx}: {len(advantages)} timesteps, critical_idx={critical_idx}")
        
        # Convert to tensors
        episode_advantages = torch.tensor(np.array(episode_advantages), dtype=torch.float32, device=device)
        episode_returns = torch.tensor(np.array(episode_returns), dtype=torch.float32, device=device)
        episode_bc_weights = torch.tensor(np.array(episode_bc_weights), dtype=torch.float32, device=device)
        
        print(f"[DEBUG] Total timesteps: {len(episode_advantages)}, Critical timesteps: {len(critical_timestep_indices)}")
        
        # ====================================================================
        # STEP 5: Concatenate all trajectory tensors for mini-batch processing
        # ====================================================================
        print(f"[DEBUG] Concatenating trajectory tensors...")
        
        # Concatenate all state inputs from all trajectories
        all_tx_com = []
        all_obj_com = []
        all_dxu_par = []
        all_img_vis = []
        all_tx_vis = []
        all_expert_actions = []  # Expert actions (for BC loss)
        
        for traj_idx, (traj_states, _, _) in enumerate(
            zip(all_trajectory_states, all_trajectory_rewards, trajectory_lengths)):
            tx_com_traj, obj_com_traj, dxu_par_traj, img_vis_traj, tx_vis_traj = traj_states
            
            # Get actual trajectory length from tensors
            traj_len = len(tx_com_traj)
            
            all_tx_com.append(tx_com_traj)
            all_obj_com.append(obj_com_traj)
            all_dxu_par.append(dxu_par_traj)
            all_img_vis.append(img_vis_traj)
            all_tx_vis.append(tx_vis_traj)
            
            # Add expert actions (padded with zeros if no expert data)
            if traj_idx < len(all_expert_data) and all_expert_data[traj_idx] is not None:
                expert_actions = all_expert_data[traj_idx]['expert_actions']
                
                # Ensure expert actions match trajectory length (truncate or pad)
                if len(expert_actions) != traj_len:
                    if len(expert_actions) < traj_len:
                        # Pad with zeros if expert trajectory is shorter
                        padding = torch.zeros((traj_len - len(expert_actions), 4), dtype=torch.float32, device=device)
                        expert_actions = torch.cat([expert_actions, padding], dim=0)
                    else:
                        # Truncate if expert trajectory is longer
                        expert_actions = expert_actions[:traj_len]
                
                all_expert_actions.append(expert_actions)
            else:
                # No expert data - pad with zeros to match trajectory length
                all_expert_actions.append(torch.zeros((traj_len, 4), dtype=torch.float32, device=device))
        
        # Concatenate into single tensors (all timesteps together)
        tx_com_batch = torch.cat(all_tx_com, dim=0)
        obj_com_batch = torch.cat(all_obj_com, dim=0)
        dxu_par_batch = torch.cat(all_dxu_par, dim=0)
        img_vis_batch = torch.cat(all_img_vis, dim=0)
        tx_vis_batch = torch.cat(all_tx_vis, dim=0)
        expert_actions_batch = torch.cat(all_expert_actions, dim=0)
        
        # CRITICAL: Use actual concatenated tensor length, not sum of trajectory_lengths
        actual_total_timesteps = len(tx_com_batch)
        print(f"[DEBUG] Concatenated {actual_total_timesteps} timesteps from {num_trajectories} trajectories")
        print(f"[DEBUG] Batch shapes - tx_com: {tx_com_batch.shape}, img_vis: {img_vis_batch.shape}, expert_actions: {expert_actions_batch.shape}")
        
        # Verify all batches have same length
        assert len(tx_com_batch) == len(expert_actions_batch) == len(episode_advantages), \
            f"Batch size mismatch: tx_com={len(tx_com_batch)}, expert_actions={len(expert_actions_batch)}, advantages={len(episode_advantages)}"
        
        # Update total_timesteps to actual value
        total_timesteps = actual_total_timesteps


        total_timesteps = sum(trajectory_lengths)
        num_trajectories = len(all_trajectory_states)
        print(f"[DEBUG] Pre-computing advantages for {num_trajectories} trajectories with {total_timesteps} total timesteps")

        actual_total_timesteps = len(tx_com_batch)
        print(f"[DEBUG] Concatenated {actual_total_timesteps} timesteps from {num_trajectories} trajectories")
        print(f"[DEBUG] Batch shapes - tx_com: {tx_com_batch.shape}, img_vis: {img_vis_batch.shape}, expert_actions: {expert_actions_batch.shape}")
        
        # Verify all batches have same length
        assert len(tx_com_batch) == len(expert_actions_batch) == len(episode_advantages), \
            f"Batch size mismatch: tx_com={len(tx_com_batch)}, expert_actions={len(expert_actions_batch)}, advantages={len(episode_advantages)}"
        
        # Update total_timesteps to actual value
        total_timesteps = actual_total_timesteps
        # ====================================================================
        # Shuffle timesteps to break temporal correlation (matches BC training)
        # ====================================================================
        # Generate random permutation indices
        shuffle_indices = torch.randperm(actual_total_timesteps, device=device)
        
        # Apply same permutation to all tensors to preserve correspondence
        tx_com_batch = tx_com_batch[shuffle_indices]
        obj_com_batch = obj_com_batch[shuffle_indices]
        dxu_par_batch = dxu_par_batch[shuffle_indices]
        img_vis_batch = img_vis_batch[shuffle_indices]
        tx_vis_batch = tx_vis_batch[shuffle_indices]
        expert_actions_batch = expert_actions_batch[shuffle_indices]
        episode_advantages = episode_advantages[shuffle_indices]
        episode_returns = episode_returns[shuffle_indices]
        episode_bc_weights = episode_bc_weights[shuffle_indices]


        # ====================================================================
        # STEP 6: Mini-batch SGD training
        # ====================================================================
        loss_rl_ep = 0.0
        loss_bc_ep = 0.0
        loss_ref_ep = 0.0

        # Mini-batch SGD across all timesteps
        batch_size = 64  # Process 64 timesteps at a time
        num_batches = max(1, (total_timesteps + batch_size - 1) // batch_size)
        print(f"[DEBUG] Training with {num_batches} mini-batches of size {batch_size}")
        
        # Progress bar for mini-batches
        if progress_bar is not None:
            progress, pilot_task = progress_bar
            print(f"[DEBUG] progress_bar exists, about to add batch_task...")
            try:
                # Add loss=0.0 and units to match the schema expected by the progress object
                batch_task = progress.add_task(
                    f"[bold cyan]Mini-batches[/]",
                    total=num_batches,
                    loss=0.0,
                    units="batches"
                )
                print(f"[DEBUG] trajectory_task created: {batch_task}")
            except Exception as e:
                print(f"[DEBUG] ERROR creating batch_task: {e}")
                batch_task = None
        else:
            batch_task = None
            print(f"[DEBUG] progress_bar is None, batch_task = None")

        # Mini-batch training loop (process fixed-size batches, not full trajectories)
        print(f"[DEBUG] Processing {num_batches} mini-batches of {batch_size} timesteps")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_timesteps)
            
            if batch_idx == 0 or batch_idx % 10 == 0:
                print(f"[DEBUG] Processing mini-batch {batch_idx + 1}/{num_batches} (timesteps {start_idx}:{end_idx})")
            
            # Slice mini-batch from concatenated tensors
            tx_com_batch_slice = tx_com_batch[start_idx:end_idx]
            obj_com_batch_slice = obj_com_batch[start_idx:end_idx]
            dxu_par_batch_slice = dxu_par_batch[start_idx:end_idx]
            img_vis_batch_slice = img_vis_batch[start_idx:end_idx]
            tx_vis_batch_slice = tx_vis_batch[start_idx:end_idx]
            expert_actions_slice = expert_actions_batch[start_idx:end_idx]
            
            # Slice advantages, returns, BC weights
            batch_advantages = episode_advantages[start_idx:end_idx]
            batch_returns = episode_returns[start_idx:end_idx]
            batch_bc_weights = episode_bc_weights[start_idx:end_idx]
            
            if batch_idx == 0:
                print(f"[DEBUG] Mini-batch shapes: policy_batch={end_idx - start_idx}, bc_weights={len(batch_bc_weights)}, expert_actions={len(expert_actions_slice)}")
            
            # Forward pass through policy to get actions (with AMP for memory efficiency)
            with autocast(enabled=use_amp):
                policy_out, _ = pilot.model(
                    tx_com_batch_slice, obj_com_batch_slice, dxu_par_batch_slice,
                    img_vis_batch_slice, tx_vis_batch_slice, return_value=False
                )
                
                # ---- Behavior cloning loss (L_BC) ----
                # Use pre-computed BC weights
                if batch_idx == 0:
                    print(f"[DEBUG] BC shapes - policy_out: {policy_out.shape}, expert_actions: {expert_actions_slice.shape}, bc_weights: {batch_bc_weights.shape}")
                
                actual_batch_size = len(policy_out)
                assert len(batch_bc_weights) == actual_batch_size, \
                    f"BC weight size mismatch: {len(batch_bc_weights)} vs {actual_batch_size}"
                assert len(expert_actions_slice) == actual_batch_size, \
                    f"Expert action size mismatch: {len(expert_actions_slice)} vs {actual_batch_size}"
                
                action_diff = torch.clamp(
                    (policy_out - expert_actions_slice) ** 2,
                    max=100.0
                )
                weighted_diff = batch_bc_weights.unsqueeze(1) * action_diff
                loss_bc = weighted_diff.mean()
                
                if batch_idx == 0:
                    print(f"[DEBUG] BC loss (weighted): {loss_bc.item():.6f}, mean weight: {batch_bc_weights.mean().item():.4f}")

                # ---- Reference policy regularization (L_ref) ----
                # Keep policy close to initial BC policy to prevent catastrophic forgetting
                if batch_idx == 0:
                    print(f"[DEBUG] Computing reference policy regularization...")
                
                # Get reference policy (initial BC policy) predictions
                with torch.no_grad():
                    policy_out_ref, _ = pilot_bc.model(
                        tx_com_batch_slice, obj_com_batch_slice, dxu_par_batch_slice,
                        img_vis_batch_slice, tx_vis_batch_slice, return_value=True
                    )
                
                # MSE between current policy and frozen reference policy
                loss_ref = F.mse_loss(policy_out, policy_out_ref)
                
                if batch_idx == 0:
                    print(f"[DEBUG] Reference policy loss: {loss_ref.item():.6f}")


                # ---- Combined loss ----
                loss_bc_clipped = torch.clamp(loss_bc, max=1000.0)
                loss_ref_clipped = torch.clamp(loss_ref, max=1000.0)
                
                # lambda_dpg controls DPG weight (set to 0 to disable DPG entirely)
                loss_actor = lambda_bc * loss_bc_clipped + lambda_ref * loss_ref_clipped
                loss_total = loss_actor
            loss_total = torch.clamp(loss_total, max=5000.0)
            
            if batch_idx == 0:
                print(f"[DEBUG] Loss computed: {loss_total.item():.6f}")
                print(f"[DEBUG] Individual - BC: {loss_bc.item():.6f}, "f"Ref: {loss_ref.item():.6f}")

            # Backward pass with separate optimizers and mixed precision
            if batch_idx == 0:
                print(f"[DEBUG] Starting backward pass...")
            try:
                # Update actor (policy gradient + BC + reference regularization)
                if batch_idx == 0:
                    print(f"[DEBUG] Updating actor...")
                opt_actor.zero_grad()
                scaler_actor.scale(loss_actor).backward()
                scaler_actor.unscale_(opt_actor)
                torch.nn.utils.clip_grad_norm_(actor_params + vision_params, max_norm=1.0)
                scaler_actor.step(opt_actor)
                scaler_actor.update()
                
                # Check for NaN or Inf in parameters
                has_nan = False
                for name, param in pilot.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"[WARNING] NaN/Inf detected in {name}! Stopping training.")
                        has_nan = True
                        break
                
                if has_nan:
                    print("[ERROR] NaN/Inf detected, stopping training")
                    break
            except Exception as e:
                print(f"[DEBUG] ERROR in backward pass: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Accumulate episode losses
            loss_bc_ep += loss_bc.item()
            loss_ref_ep += loss_ref.item()
            
            # Update progress bar
            if progress_bar is not None and batch_task is not None:
                progress.update(batch_task, advance=1, loss=loss_total.item())

        print(f"[DEBUG] Mini-batch loop completed. Processed {num_batches} batches.")
        
        # Average losses over batches
        loss_bc_ep /= max(1, num_batches)
        loss_ref_ep /= max(1, num_batches)
        loss_total_ep = lambda_bc * loss_bc_ep + lambda_ref * loss_ref_ep

        # Free GPU memory after episode
        del tx_com_batch, obj_com_batch, dxu_par_batch, img_vis_batch, tx_vis_batch
        del episode_advantages, episode_returns, episode_bc_weights, expert_actions_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ====================================================================
        # STEP 7: Logging and checkpointing
        # ====================================================================
        print(f"[DEBUG] Episode {episode + 1} complete!")
        print(f"Losses - RL: {loss_rl_ep:.6f}, BC: {loss_bc_ep:.6f}, Ref: {loss_ref_ep:.6f}, "f"Total: {loss_total_ep:.6f}")

        # Store loss history
        losses["collision_rate"].append(collision_rate_ep)
        losses["mean_clearance"].append(mean_clearance_ep)
        losses["loss_rl"].append(loss_rl_ep)
        losses["loss_bc"].append(loss_bc_ep)
        losses["loss_ref"].append(loss_ref_ep)
        losses["loss_total"].append(loss_total_ep)
        losses["episodes"].append(episode + 1)

        # ====================================================================
        # VALIDATION: Evaluate updated pilot on fixed validation set
        # ====================================================================
        print(f"\n[VALIDATION] Evaluating episode {episode+1} on validation set...")

        # Create temporary validation directory with fixed RRT paths
        import shutil

        # Generate timestamp for this validation run (plain format for correct sorting)
        val_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        val_temp_dir = os.path.join(cohort_path, "simulation_data", val_timestamp)
        val_rrt_dir = os.path.join(val_temp_dir, "rrt_planning")

        # Copy validation RRT paths
        os.makedirs(val_rrt_dir, exist_ok=True)
        # Navigate through validation_data_dir/simulation_data/{timestamp}/rrt_planning/
        validation_sim_data_dir = os.path.join(validation_data_dir, "simulation_data")
        validation_timestamp_dirs = sorted(glob.glob(os.path.join(validation_sim_data_dir, "*")))
        if not validation_timestamp_dirs:
            raise FileNotFoundError(f"No timestamp directories found in {validation_sim_data_dir}")
        validation_timestamp_dir = validation_timestamp_dirs[-1]
        validation_rrt_source = os.path.join(validation_timestamp_dir, "rrt_planning")

        if os.path.exists(validation_rrt_source):
            for rrt_file in glob.glob(os.path.join(validation_rrt_source, "*.pkl")):
                shutil.copy2(rrt_file, val_rrt_dir)
            print(f"  Copied validation RRT paths to {val_temp_dir}")
        else:
            raise FileNotFoundError(f"Validation RRT directory not found: {validation_rrt_source}")

        # Re-simulate validation trajectories with updated pilot
        # review=False tells it to USE EXISTING RRT paths (from val_temp_dir)
        from sousvide.flight.deploy_ssv import simulate_rollouts

        simulate_rollouts(
            workspace_path=workspace_path,
            cohort_name=cohort_name,
            cohort_path=cohort_path,  # Will find val_temp_dir (most recent timestamp)
            method_name=method_name,
            pilot=pilot,  # Updated pilot (just trained in this episode)
            flights=flights,
            scenes_cfg_dir=scenes_cfg_dir,
            objectives_all=objective_configs,
            max_trajectories=None,  # Simulate ALL validation trajectories
            review=False,  # ← CRITICAL: Use existing RRT paths, don't regenerate!
            disable_visualization=True,
            show_progress=True,  # Suppress progress for validation
            progress_bar=progress_bar,
        )

        # Load validation trajectories (from val_temp_dir)
        # These are ONLY the selected trajectories (pkl files were pre-filtered)
        from sousvide.rl import load_simulation_results, prepare_batch_data

        val_trajectories, val_metadata, val_raw_data = load_simulation_results(
            cohort_path,  # Loads from simulation_data/{latest} which is val_temp_dir
            pilot_name=pilot.name
        )

        if not val_trajectories:
            print(f"[WARNING] No validation trajectories found for episode {episode+1}")
            val_collision_rate = float('nan')
            val_mean_clearance = float('nan')
        else:
            print(f"  Using {len(val_trajectories)} validation trajectories")
            # Batch processing of validation trajectories to avoid OOM
            batch_size = 32
            all_analyses = []
            all_clearances = []

            print(f"[MEMORY] Processing validation trajectories in batches of {batch_size}")
            for batch_start in range(0, len(val_trajectories), batch_size):
                batch_end = min(batch_start + batch_size, len(val_trajectories))
                val_batch_trajectories = val_trajectories[batch_start:batch_end]
                val_batch_metadata = {key: val_metadata[key][batch_start:batch_end] if isinstance(val_metadata[key], list) else val_metadata[key]
                                     for key in val_metadata}

                print(f"  [Val Batch {batch_start//batch_size + 1}] Processing trajectories {batch_start+1}-{batch_end}")

                # Process this batch
                val_batch_data = prepare_batch_data(
                    val_batch_trajectories,
                    val_batch_metadata,
                    collision_detectors[list(scene_configs.keys())[0]],
                    device
                )

                # Accumulate results
                all_analyses.extend(val_batch_data["analyses"])
                all_clearances.extend(val_batch_data["clearances"])

                # Free memory after batch
                del val_batch_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Recompute aggregate metrics from accumulated results
            all_clearances_flat = np.concatenate(all_clearances)
            collision_count = sum(a["collision"] for a in all_analyses)
            val_collision_rate = collision_count / len(val_trajectories) if val_trajectories else 0.0
            val_mean_clearance = float(np.mean(all_clearances_flat))

            print(f"[VALIDATION] Episode {episode+1} metrics:")
            print(f"  Collision rate: {val_collision_rate:.2%}")
            print(f"  Mean clearance: {val_mean_clearance:.3f}m")
            print(f"  (Simulated {len(val_trajectories)} selected trajectories)")

        # Save validation metrics to losses dict
        losses["val_collision_rate"].append(val_collision_rate)
        losses["val_mean_clearance"].append(val_mean_clearance)

        # Close matplotlib figures
        plt.close('all')

        # Clean up temporary validation directory (saves disk space)
        # Note: Using plain timestamp format ensures simulate_rollouts() finds it via max() sorting
        if os.path.exists(val_temp_dir):
            shutil.rmtree(val_temp_dir)
            print(f"  Cleaned up validation directory: {val_timestamp}")

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Free GPU memory after episode (only if variables exist)
        try:
            del tx_com_batch, obj_com_batch, dxu_par_batch, img_vis_batch, tx_vis_batch
        except (NameError, UnboundLocalError):
            pass

        try:
            del episode_advantages, episode_returns, episode_bc_weights, expert_actions_batch
        except (NameError, UnboundLocalError):
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # DELETE ACCUMULATED DATA FROM ENTIRE EPISODE
        try:
            del all_tx_com, all_obj_com, all_dxu_par, all_img_vis, all_tx_vis
        except (NameError, UnboundLocalError):
            pass

        try:
            del trajectories, metadata, raw_data
        except (NameError, UnboundLocalError):
            pass

        # Force Python garbage collection
        import gc
        gc.collect()

        # Save checkpoint periodically
        if ((episode + 1) % lim_sv == 0) or (episode + 1 == Neps):
            torch.save(pilot.model, rl_model_path)
            torch.save(losses, rl_losses_path)
            print(f"[CHECKPOINT] Saved at episode {episode + 1}")
        # Update episode progress bar
        if progress_bar is not None:
            progress, pilot_task = progress_bar
            progress.update(pilot_task, advance=1, loss=loss_total_ep)