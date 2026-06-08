#!/usr/bin/env python3
"""
Gaussian BC training for SINGER.

Same as standard BC pipeline but:
  - Adds a learned logstd parameter to the Commander
  - Uses NLL loss instead of MSE
  - At deployment, uses deterministic mean (logstd discarded)

Reuses the exact same observation data, datasets, and model architecture.
Only the loss function and one extra parameter change.

Usage:
    cd /data/erwinpi/SINGER
    source activate FiGS
    CUDA_VISIBLE_DEVICES=0 python scripts/train_gaussian_bc.py \
        --config-file configs/experiment/ssv_gaussian_bc_v1.yml \
        --use-wandb --wandb-project singer
"""
import argparse
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

import yaml
import wandb as wandb_module

from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import (
    generate_dataset, get_data_paths,
)
from sousvide.instruct.train_policy import unlock_networks


class GaussianNLLLoss(nn.Module):
    """NLL loss for a diagonal Gaussian: -log N(action | mean, diag(exp(logstd)²)).

    When logstd is fixed, this reduces to MSE (up to a constant).
    When logstd is learned, the model learns per-dimension uncertainty.
    """

    def __init__(self, logstd: nn.Parameter):
        super().__init__()
        self.logstd = logstd

    def forward(self, prediction, target):
        # Clamp logstd to prevent collapse/explosion (same as hw3)
        logstd = torch.clamp(self.logstd, min=-5.0, max=2.0)
        var = torch.exp(2 * logstd)  # σ²

        # NLL = 0.5 * Σ_d [ (y - μ)² / σ² + log(σ²) ]
        mse_per_dim = (target - prediction) ** 2 / var
        log_var = torch.log(var)

        # Mean over batch and action dims
        nll = 0.5 * (mse_per_dim + log_var).mean()
        return nll


def train_gaussian_bc(cfg, use_wandb=False, wandb_project="singer"):
    cohort = cfg["cohort"]
    source_cohort = cfg.get("source_cohort", "ssv_BC_CENTROID_V9")
    roster = cfg.get("roster", ["InstinctJester"])
    n_epochs = cfg.get("Nep_com", 150)
    lr = cfg.get("lr", 1e-4)
    batch_size = cfg.get("batch_size", 64)
    lim_sv = cfg.get("lim_sv", 10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    for student_name in roster:
        print(f"\n{'='*70}")
        print(f"[Gaussian BC] Training {student_name} in cohort '{cohort}'")
        print(f"  Source observations: {source_cohort}")
        print(f"  Epochs: {n_epochs}, LR: {lr}, Batch: {batch_size}")
        print(f"{'='*70}\n")

        # --- Step 1: Copy source model into new cohort ---
        src_roster = os.path.join(WORKSPACE, "cohorts", source_cohort, "roster", student_name)
        dst_roster = os.path.join(WORKSPACE, "cohorts", cohort, "roster", student_name)
        os.makedirs(dst_roster, exist_ok=True)

        # Copy config.json
        src_config = os.path.join(src_roster, "config.json")
        dst_config = os.path.join(dst_roster, "config.json")
        if os.path.exists(src_config) and not os.path.exists(dst_config):
            import shutil
            shutil.copy2(src_config, dst_config)
            print(f"  Copied config.json from {source_cohort}")

        # Copy initial model (we'll train from scratch with same architecture)
        src_model = os.path.join(src_roster, "model.pth")
        dst_model = os.path.join(dst_roster, "model.pth")
        if os.path.exists(src_model) and not os.path.exists(dst_model):
            import shutil
            shutil.copy2(src_model, dst_model)
            print(f"  Copied initial model from {source_cohort}")

        # --- Step 2: Symlink observation data ---
        src_obs = os.path.join(WORKSPACE, "cohorts", source_cohort, "observation_data", student_name)
        dst_obs = os.path.join(WORKSPACE, "cohorts", cohort, "observation_data", student_name)
        if os.path.exists(src_obs) and not os.path.exists(dst_obs):
            os.makedirs(os.path.dirname(dst_obs), exist_ok=True)
            os.symlink(src_obs, dst_obs)
            print(f"  Symlinked observation data from {source_cohort}")

        # --- Step 3: Train History Encoder first (same as BC, MSE) ---
        n_hist_epochs = cfg.get("Nep_his", 100)
        if cfg.get("train_history", True):
            print(f"\n[Phase 1] Training HistoryEncoder ({n_hist_epochs} epochs, MSE)...")
            student = Pilot(cohort, student_name)
            student.set_mode('train')
            student.model.to(device)

            from sousvide.instruct.train_policy import train_student
            if use_wandb:
                wandb_module.init(
                    project=wandb_project,
                    name=f"GaussBC_{cohort}_History",
                    config=cfg, reinit=True,
                )
            train_student(cohort, student, "Parameter", n_hist_epochs, lim_sv, lr, batch_size)
            if use_wandb and wandb_module.run is not None:
                wandb_module.finish()
            del student
            torch.cuda.empty_cache()
            print("  HistoryEncoder training done.\n")

        # --- Step 4: Train Commander with Gaussian NLL ---
        print(f"\n[Phase 2] Training Commander with Gaussian NLL ({n_epochs} epochs)...")
        student = Pilot(cohort, student_name)
        student.set_mode('train')
        student.model.to(device)

        # Get Commander network
        model = student.model.get_network["Commander"]["Train"]  # full SVNet

        # Create learned logstd parameter (4-d, one per action dim)
        ac_dim = 4
        logstd = nn.Parameter(torch.zeros(ac_dim, device=device))

        # NLL criterion
        criterion = GaussianNLLLoss(logstd)
        mse_criterion = nn.MSELoss(reduction='mean')  # for logging comparison

        # Optimizer: Commander + VisionMLP + logstd
        unlock_networks(student, "Commander")
        commander_params = list(student.model.get_network["Commander"]["Unlock"].parameters())
        opt = optim.Adam(list(commander_params) + [logstd], lr=lr)

        # Paths
        student_path = student.path
        model_path = os.path.join(student_path, "model.pth")
        best_model_path = os.path.join(student_path, "best_model.pth")
        logstd_path = os.path.join(student_path, "logstd.pt")
        losses_path = os.path.join(student_path, "losses_Commander.pt")

        if use_wandb:
            wandb_module.init(
                project=wandb_project,
                name=f"GaussBC_{cohort}_Commander",
                config={**cfg, "loss": "gaussian_nll", "logstd_init": 0.0},
                reinit=True,
            )

        best_test_loss = float('inf')
        all_train_losses = []
        all_test_losses = []
        all_mse_losses = []

        start_time = time.time()

        for ep in range(n_epochs):
            # Unlock Commander + VisionMLP
            unlock_networks(student, "Commander")

            # Get data paths (from symlinked observation data)
            od_train_files, od_test_files, _, _ = get_data_paths(cohort, student_name)

            # Training
            train_log = []
            for od_file in od_train_files:
                dataset = generate_dataset(od_file, student, "Commander", device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

                for inp, lbl in dataloader:
                    inp = tuple(t.to(device) for t in inp)
                    lbl = lbl.to(device)

                    prediction, _ = model(*inp)
                    loss = criterion(prediction, lbl)

                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    train_log.append((lbl.shape[0], loss.item()))

            # Testing
            test_log = []
            mse_log = []
            with torch.no_grad():
                for od_file in od_test_files:
                    dataset = generate_dataset(od_file, student, "Commander", device)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                    for inp, lbl in dataloader:
                        inp = tuple(t.to(device) for t in inp)
                        lbl = lbl.to(device)

                        prediction, _ = model(*inp)
                        test_log.append((lbl.shape[0], criterion(prediction, lbl).item()))
                        mse_log.append((lbl.shape[0], mse_criterion(prediction, lbl).item()))

            # Compute epoch losses
            n_train = sum(n for n, _ in train_log)
            n_test = sum(n for n, _ in test_log)
            loss_train = sum(n * l for n, l in train_log) / n_train
            loss_test = sum(n * l for n, l in test_log) / n_test
            loss_mse = sum(n * l for n, l in mse_log) / n_test

            all_train_losses.append(loss_train)
            all_test_losses.append(loss_test)
            all_mse_losses.append(loss_mse)

            # Current std values
            with torch.no_grad():
                std_vals = torch.exp(torch.clamp(logstd, -5, 2))

            if (ep + 1) % 10 == 0 or ep == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {ep+1:>3d}/{n_epochs}  "
                      f"NLL_train={loss_train:.6f}  NLL_test={loss_test:.6f}  "
                      f"MSE_test={loss_mse:.6f}  "
                      f"std=[{', '.join(f'{s:.4f}' for s in std_vals.cpu())}]  "
                      f"({elapsed:.0f}s)")

            if use_wandb and wandb_module.run is not None:
                log_dict = {
                    "train/nll_loss": loss_train,
                    "test/nll_loss": loss_test,
                    "test/mse_loss": loss_mse,
                    "epoch": ep,
                }
                for i, s in enumerate(std_vals.cpu()):
                    log_dict[f"std/dim_{i}"] = s.item()
                log_dict["std/mean"] = std_vals.mean().item()
                wandb_module.log(log_dict)

            # Save best model
            if loss_test < best_test_loss:
                best_test_loss = loss_test
                unlock_networks(student, "None")
                torch.save(student.model, best_model_path)
                torch.save({"logstd": logstd.detach().cpu()}, logstd_path)
                if (ep + 1) % 10 == 0:
                    print(f"    -> New best test NLL: {best_test_loss:.6f}")

            # Periodic save
            if ((ep + 1) % lim_sv == 0) or (ep + 1 == n_epochs):
                unlock_networks(student, "None")
                torch.save(student.model, model_path)

        # Final summary
        total_time = time.time() - start_time
        final_std = torch.exp(torch.clamp(logstd, -5, 2)).detach().cpu()
        print(f"\n  Done! {n_epochs} epochs in {total_time:.0f}s")
        print(f"  Final NLL: train={all_train_losses[-1]:.6f} test={all_test_losses[-1]:.6f}")
        print(f"  Final MSE: {all_mse_losses[-1]:.6f}")
        print(f"  Learned std: [{', '.join(f'{s:.4f}' for s in final_std)}]")
        print(f"  Best test NLL: {best_test_loss:.6f}")
        print(f"  Models: {model_path}, {best_model_path}")

        # Save loss curves
        losses = {
            "train": [all_train_losses],
            "test": [all_test_losses],
            "mse": [all_mse_losses],
            "Neps": [n_epochs],
            "Nspl": [n_train],
            "t_train": [total_time],
        }
        torch.save(losses, losses_path)

        if use_wandb and wandb_module.run is not None:
            wandb_module.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="singer")
    parser.add_argument("--skip-history", action="store_true",
                        help="Skip HistoryEncoder training (reuse from source)")
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    if args.skip_history:
        cfg["train_history"] = False

    train_gaussian_bc(cfg, use_wandb=args.use_wandb, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
