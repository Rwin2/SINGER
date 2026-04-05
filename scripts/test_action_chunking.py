#!/usr/bin/env python3
"""
Test script for action chunking BC training.

Reuses V9 observation data with windowed action chunks.
Transfers HistoryEncoder + VisionMLP weights from V9 BC model.
Only trains CommanderSV with 20D output (5 steps × 4 actions).

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    conda run -n FiGS python scripts/test_action_chunking.py
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import (
    extract_data_chunked, ObservationData, get_data_paths
)
from sousvide.instruct.train_policy import unlock_networks


# ============================================================
# Configuration
# ============================================================
CHUNK_HORIZON = 5       # Predict 5 future actions
CHUNK_EXECUTE = 2       # Execute 2 before re-querying
ACTION_DIM = 4          # 4D actions
EPOCHS = 30             # Small test
BATCH_SIZE = 64
LR = 1e-4

SOURCE_COHORT = "ssv_BC_CENTROID_V9"
SOURCE_PILOT = "InstinctJester"
TARGET_COHORT = "ssv_BC_CHUNKED_TEST"
TARGET_PILOT = "InstinctJester_chunked"


def transfer_weights(source_cohort, source_pilot, target_pilot_obj):
    """Transfer HistoryEncoder + VisionMLP weights from source to target.
    Commander weights are NOT transferred (different output dim)."""
    source = Pilot(source_cohort, source_pilot)
    source_model = torch.load(os.path.join(source.path, "model.pth"),
                              map_location="cpu", weights_only=False)
    source_state = source_model.state_dict()

    target_state = target_pilot_obj.model.state_dict()

    transferred = 0
    skipped = 0
    for key in source_state:
        if "CommanderSV" in key:
            skipped += 1
            continue  # Skip Commander (different output_size)
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
            transferred += 1
        else:
            skipped += 1

    target_pilot_obj.model.load_state_dict(target_state)
    print(f"Weight transfer: {transferred} transferred, {skipped} skipped (Commander)")
    return target_pilot_obj


def create_symlink_observations(source_cohort, source_pilot, target_cohort, target_pilot):
    """Symlink V9 observation data to chunked cohort (avoid duplication)."""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    source_obs = os.path.join(workspace, "cohorts", source_cohort,
                               "observation_data", source_pilot)
    target_obs_base = os.path.join(workspace, "cohorts", target_cohort,
                                    "observation_data")
    target_obs = os.path.join(target_obs_base, target_pilot)

    if os.path.exists(target_obs):
        print(f"Observation link already exists: {target_obs}")
        return

    # Symlink source pilot's observation data under target pilot name
    os.makedirs(target_obs_base, exist_ok=True)
    os.symlink(source_obs, target_obs)
    print(f"Linked observations: {source_obs} → {target_obs}")


def train_chunked_bc():
    """Train Commander with action chunks using V9 observation data."""
    print("=" * 60)
    print("ACTION CHUNKING BC TEST")
    print(f"  Chunk horizon: {CHUNK_HORIZON}, Execute: {CHUNK_EXECUTE}")
    print(f"  Output dim: {CHUNK_HORIZON * ACTION_DIM} = {CHUNK_HORIZON}×{ACTION_DIM}")
    print(f"  Source: {SOURCE_COHORT}/{SOURCE_PILOT}")
    print(f"  Target: {TARGET_COHORT}/{TARGET_PILOT}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    # Step 1: Create observation symlink
    create_symlink_observations(SOURCE_COHORT, SOURCE_PILOT,
                                 TARGET_COHORT, TARGET_PILOT)

    # Step 2: Initialize target pilot (creates new Commander with 20D output)
    student = Pilot(TARGET_COHORT, TARGET_PILOT)
    student.set_mode('train')

    # Step 3: Transfer HistoryEncoder + VisionMLP weights from V9
    student = transfer_weights(SOURCE_COHORT, SOURCE_PILOT, student)

    # Step 4: Setup training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student.model.to(device)
    criterion = nn.MSELoss(reduction='mean')

    # Get Commander model for training
    model = student.model.get_network["Commander"]["Train"]
    opt = optim.Adam(model.parameters(), lr=LR)

    # Get data paths (uses symlinked V9 data)
    train_paths, test_paths, val_paths, rol_paths = get_data_paths(
        TARGET_COHORT, TARGET_PILOT)

    print(f"\nData files: {len(train_paths)} train, {len(test_paths)} test")

    # Step 5: Training loop
    losses_train = []
    losses_test = []
    best_test_loss = float('inf')
    start_time = time.time()

    with trange(EPOCHS, desc="Training Chunked BC") as pbar:
        for ep in pbar:
            # Unlock Commander for training (freeze HistoryEncoder + VisionMLP)
            unlock_networks(student, "Commander")

            # --- Train ---
            epoch_loss = []
            for train_file in train_paths:
                # Use chunked dataset
                Xnn_ds, Ynn_ds = extract_data_chunked(
                    train_file, CHUNK_HORIZON, ACTION_DIM)
                extractor = student.model.get_data["Commander"]
                dataset = ObservationData(Xnn_ds, Ynn_ds, extractor)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                     shuffle=True, drop_last=False)

                for inp, label in loader:
                    inp = tuple(t.to(device) for t in inp)
                    label = label.to(device)

                    pred, _ = model(*inp)
                    loss = criterion(pred, label)

                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    epoch_loss.append((label.shape[0], loss.item()))

            # Weighted average train loss
            N_train = sum(n for n, _ in epoch_loss)
            avg_train = sum(n * l for n, l in epoch_loss) / N_train
            losses_train.append(avg_train)

            # --- Test ---
            test_loss = []
            with torch.no_grad():
                for test_file in test_paths:
                    Xnn_ds, Ynn_ds = extract_data_chunked(
                        test_file, CHUNK_HORIZON, ACTION_DIM)
                    extractor = student.model.get_data["Commander"]
                    dataset = ObservationData(Xnn_ds, Ynn_ds, extractor)
                    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, drop_last=False)

                    for inp, label in loader:
                        inp = tuple(t.to(device) for t in inp)
                        label = label.to(device)
                        pred, _ = model(*inp)
                        loss = criterion(pred, label)
                        test_loss.append((label.shape[0], loss.item()))

            N_test = sum(n for n, _ in test_loss)
            avg_test = sum(n * l for n, l in test_loss) / N_test
            losses_test.append(avg_test)

            # Save best model
            if avg_test < best_test_loss:
                best_test_loss = avg_test
                torch.save(student.model.state_dict(),
                           os.path.join(student.path, "model.pth"))
                torch.save(student.model.state_dict(),
                           os.path.join(student.path, "best_model.pth"))

            # Save last model always
            torch.save(student.model.state_dict(),
                       os.path.join(student.path, "last_model.pth"))

            pbar.set_postfix(train=f"{avg_train:.5f}", test=f"{avg_test:.5f}",
                           best=f"{best_test_loss:.5f}")

    elapsed = time.time() - start_time

    # Save losses
    torch.save({"train": losses_train, "test": losses_test,
                "epochs": EPOCHS, "chunk_horizon": CHUNK_HORIZON,
                "chunk_execute": CHUNK_EXECUTE},
               os.path.join(student.path, "losses_Commander.pt"))

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"  Final train loss: {losses_train[-1]:.5f}")
    print(f"  Final test loss:  {losses_test[-1]:.5f}")
    print(f"  Best test loss:   {best_test_loss:.5f}")
    print(f"  Model saved to:   {student.path}/model.pth")

    return student, losses_train, losses_test


if __name__ == "__main__":
    student, losses_train, losses_test = train_chunked_bc()
