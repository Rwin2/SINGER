#!/usr/bin/env python3
"""
Test script for Flow Matching BC training on SINGER.

Three modes (set via MODE variable):
  1. "fm_only"      — Flow matching on single-step 4D actions
  2. "fm_chunked"   — Flow matching on action chunks (H=5, 20D)
  3. "fm_dynamics"  — Flow matching + dynamics loss (predict future bearing/elev)

Reuses V9 observation data. Transfers HistoryEncoder + VisionMLP from V9.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/test_flow_matching.py [--mode fm_only|fm_chunked|fm_dynamics]
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import (
    extract_data, extract_data_chunked, ObservationData, get_data_paths
)
from sousvide.instruct.train_policy import unlock_networks
from sousvide.control.policies.FlowMatchingCommander import (
    FlowMatchingSchedule, SinusoidalTimeEmbedding, FlowMatchingCommanderWrapper
)
import sousvide.control.policies.BaseNetworks as bn

# ============================================================
# Configuration
# ============================================================
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4

CHUNK_HORIZON = 5
CHUNK_EXECUTE = 2
ACTION_DIM = 4
NUM_FLOW_STEPS = 10

SOURCE_COHORT = "ssv_BC_CENTROID_V9"
SOURCE_PILOT = "InstinctJester"


def transfer_frozen_weights(source_cohort, source_pilot, target_model):
    """Transfer HistoryEncoder + VisionMLP weights from source."""
    source = Pilot(source_cohort, source_pilot)
    source_model_obj = torch.load(os.path.join(source.path, "model.pth"),
                                   map_location="cpu", weights_only=False)
    source_state = source_model_obj.state_dict()
    target_state = target_model.svnet.state_dict()

    transferred = 0
    for key in source_state:
        if "CommanderSV" in key:
            continue
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
            transferred += 1

    target_model.svnet.load_state_dict(target_state)
    print(f"Transferred {transferred} weight tensors (HistoryEncoder + VisionMLP)")


def setup_cohort(mode):
    """Setup cohort directory and symlink observations."""
    cohort_name = f"ssv_BC_FM_{mode.upper()}"
    # fm_chunked uses chunked pilot config (with action_chunk settings)
    if mode == "fm_chunked":
        pilot_name = "InstinctJester_chunked"
    else:
        pilot_name = SOURCE_PILOT

    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create cohort dirs
    roster_path = os.path.join(workspace, "cohorts", cohort_name,
                                "roster", pilot_name)
    os.makedirs(roster_path, exist_ok=True)

    # Copy pilot config (use chunked config for fm_chunked)
    config_name = pilot_name if mode == "fm_chunked" else SOURCE_PILOT
    src_config = os.path.join(workspace, "configs", "pilots", f"{config_name}.json")
    dst_config = os.path.join(roster_path, "config.json")
    if not os.path.exists(dst_config):
        import shutil
        shutil.copy2(src_config, dst_config)

    # Symlink observations
    src_obs = os.path.join(workspace, "cohorts", SOURCE_COHORT,
                            "observation_data", SOURCE_PILOT)
    dst_obs_base = os.path.join(workspace, "cohorts", cohort_name,
                                 "observation_data")
    dst_obs = os.path.join(dst_obs_base, pilot_name)
    if not os.path.exists(dst_obs):
        os.makedirs(dst_obs_base, exist_ok=True)
        os.symlink(src_obs, dst_obs)
        print(f"Linked observations → {dst_obs}")

    return cohort_name, pilot_name, roster_path


def train_flow_matching(mode="fm_only"):
    """Train Commander with flow matching loss."""
    output_size = ACTION_DIM if mode == "fm_only" else CHUNK_HORIZON * ACTION_DIM

    print("=" * 60)
    print(f"FLOW MATCHING BC TEST — mode: {mode}")
    print(f"  Output dim: {output_size}")
    print(f"  Flow steps: {NUM_FLOW_STEPS}, Epochs: {EPOCHS}")
    print("=" * 60)

    cohort_name, pilot_name, roster_path = setup_cohort(mode)

    # Initialize base pilot (for HistoryEncoder + VisionMLP)
    student = Pilot(cohort_name, pilot_name)
    student.set_mode('train')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create flow matching wrapper
    fm_model = FlowMatchingCommanderWrapper(
        student.model, output_size=output_size,
        num_flow_steps=NUM_FLOW_STEPS,
        time_embed_dim=32, hidden_sizes=[256, 128]
    ).to(device)

    # Transfer frozen weights
    transfer_frozen_weights(SOURCE_COHORT, SOURCE_PILOT, fm_model)

    # Freeze HistoryEncoder + VisionMLP
    for param in fm_model.svnet.parameters():
        param.requires_grad = False
    # Only train fm_commander + time_embed
    trainable = list(fm_model.fm_commander.parameters()) + list(fm_model.time_embed.parameters())
    opt = optim.Adam(trainable, lr=LR)

    schedule = FlowMatchingSchedule(
        action_dim=output_size, device=str(device),
        num_steps=NUM_FLOW_STEPS, gaussian_weighting=True
    )

    # Data paths
    train_paths, test_paths, _, _ = get_data_paths(cohort_name, pilot_name)
    print(f"Data: {len(train_paths)} train, {len(test_paths)} test files")

    # Extractor to get inputs + labels
    extractor = student.model.get_data["Commander"]

    # Training loop
    losses_train, losses_test = [], []
    best_test_loss = float('inf')
    start_time = time.time()

    with trange(EPOCHS, desc=f"FM Training ({mode})") as pbar:
        for ep in pbar:
            # --- Train ---
            fm_model.train()
            epoch_loss = []

            for train_file in train_paths:
                if mode == "fm_only":
                    Xnn_ds, Ynn_ds = extract_data(train_file)
                else:
                    Xnn_ds, Ynn_ds = extract_data_chunked(
                        train_file, CHUNK_HORIZON, ACTION_DIM)

                dataset = ObservationData(Xnn_ds, Ynn_ds, extractor)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=False)

                for inp, label in loader:
                    inp = tuple(t.to(device) for t in inp)
                    label = label.to(device)
                    B = label.shape[0]

                    # Flow matching: sample t, interpolate, predict velocity
                    t = schedule.sample_timesteps(B, device)
                    x_t, target_v = schedule.interpolate(label, t)

                    # Forward: predict velocity
                    pred_v = fm_model.forward_flow(
                        *inp, noisy_action=x_t, flow_time=t)

                    # Weighted MSE loss
                    weights = schedule.compute_loss_weights(t)
                    loss = ((pred_v - target_v) ** 2).mean(dim=-1)
                    loss = (loss * weights).mean()

                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    epoch_loss.append((B, loss.item()))

            N_train = sum(n for n, _ in epoch_loss)
            avg_train = sum(n * l for n, l in epoch_loss) / N_train
            losses_train.append(avg_train)

            # --- Test ---
            fm_model.eval()
            test_loss = []
            with torch.no_grad():
                for test_file in test_paths:
                    if mode == "fm_only":
                        Xnn_ds, Ynn_ds = extract_data(test_file)
                    else:
                        Xnn_ds, Ynn_ds = extract_data_chunked(
                            test_file, CHUNK_HORIZON, ACTION_DIM)

                    dataset = ObservationData(Xnn_ds, Ynn_ds, extractor)
                    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, drop_last=False)

                    for inp, label in loader:
                        inp = tuple(t.to(device) for t in inp)
                        label = label.to(device)
                        B = label.shape[0]

                        t = schedule.sample_timesteps(B, device)
                        x_t, target_v = schedule.interpolate(label, t)
                        pred_v = fm_model.forward_flow(*inp, x_t, t)

                        loss = ((pred_v - target_v) ** 2).mean()
                        test_loss.append((B, loss.item()))

            N_test = sum(n for n, _ in test_loss)
            avg_test = sum(n * l for n, l in test_loss) / N_test
            losses_test.append(avg_test)

            if avg_test < best_test_loss:
                best_test_loss = avg_test
                torch.save(fm_model,
                           os.path.join(roster_path, "model.pth"))
                torch.save(fm_model,
                           os.path.join(roster_path, "best_model.pth"))

            torch.save(fm_model,
                       os.path.join(roster_path, "last_model.pth"))

            pbar.set_postfix(train=f"{avg_train:.5f}", test=f"{avg_test:.5f}",
                           best=f"{best_test_loss:.5f}")

    elapsed = time.time() - start_time

    torch.save({"train": losses_train, "test": losses_test,
                "epochs": EPOCHS, "mode": mode,
                "flow_steps": NUM_FLOW_STEPS},
               os.path.join(roster_path, "losses_FlowMatching.pt"))

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"  Final train loss: {losses_train[-1]:.5f}")
    print(f"  Best test loss:   {best_test_loss:.5f}")
    print(f"  Model saved to:   {roster_path}/fm_model_best.pth")

    return fm_model, losses_train, losses_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fm_only",
                        choices=["fm_only", "fm_chunked"])
    args = parser.parse_args()
    train_flow_matching(args.mode)
