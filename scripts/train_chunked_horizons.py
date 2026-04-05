#!/usr/bin/env python3
"""
Train action chunking BC at different horizons to find optimal H and K.

Tests: H=5/K=2, H=10/K=3, H=10/K=5
All reuse V9 observation data. Transfers HistoryEncoder + VisionMLP from V9 BC.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/train_chunked_horizons.py [--horizons 10]
"""

import os
import sys
import time
import json
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import (
    extract_data_chunked, ObservationData, get_data_paths
)
from sousvide.instruct.train_policy import unlock_networks

WORKSPACE = os.path.dirname(os.path.abspath(__file__)).replace('/scripts', '')
SOURCE_COHORT = "ssv_BC_CENTROID_V9"
SOURCE_PILOT = "InstinctJester"

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4


def transfer_frozen_weights(source_cohort, source_pilot, target_pilot):
    """Transfer HistoryEncoder + VisionMLP weights from source model."""
    source = Pilot(source_cohort, source_pilot)
    source_model = torch.load(os.path.join(source.path, "model.pth"),
                               map_location="cpu", weights_only=False)
    source_state = source_model.state_dict()
    target_state = target_pilot.model.state_dict()

    transferred = 0
    for key in source_state:
        if "CommanderSV" in key:
            continue
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
            transferred += 1

    target_pilot.model.load_state_dict(target_state)
    print(f"  Transferred {transferred} weight tensors (HistoryEncoder + VisionMLP)")


def setup_cohort(cohort_name, pilot_name):
    """Create cohort dir and symlink V9 observations."""
    roster_path = os.path.join(WORKSPACE, "cohorts", cohort_name, "roster", pilot_name)
    os.makedirs(roster_path, exist_ok=True)

    # Copy pilot config
    src_config = os.path.join(WORKSPACE, "configs", "pilots", f"{pilot_name}.json")
    dst_config = os.path.join(roster_path, "config.json")
    if not os.path.exists(dst_config):
        shutil.copy2(src_config, dst_config)

    # Symlink observations
    src_obs = os.path.join(WORKSPACE, "cohorts", SOURCE_COHORT,
                            "observation_data", SOURCE_PILOT)
    dst_obs_base = os.path.join(WORKSPACE, "cohorts", cohort_name, "observation_data")
    dst_obs = os.path.join(dst_obs_base, pilot_name)
    if not os.path.exists(dst_obs):
        os.makedirs(dst_obs_base, exist_ok=True)
        os.symlink(src_obs, dst_obs)

    return roster_path


def train_chunked_bc(horizon, execute_steps, epochs=EPOCHS):
    """Train chunked BC with given horizon and execute_steps."""
    action_dim = 4
    output_size = horizon * action_dim
    pilot_name = f"InstinctJester_chunked_h{horizon}"

    # Use existing config if it exists, otherwise use h10 config as template
    config_path = os.path.join(WORKSPACE, "configs", "pilots", f"{pilot_name}.json")
    if not os.path.exists(config_path):
        # Create config from template
        base_config = os.path.join(WORKSPACE, "configs", "pilots", "InstinctJester_chunked.json")
        with open(base_config) as f:
            cfg = json.load(f)
        cfg["networks"][2]["output_size"] = output_size
        # Scale hidden sizes for larger outputs
        if horizon >= 10:
            cfg["networks"][2]["hidden_sizes"] = [256, 256]
        cfg["action_chunk"]["horizon"] = horizon
        cfg["action_chunk"]["execute_steps"] = execute_steps
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"  Created config: {config_path}")

    cohort_name = f"ssv_BC_CHUNKED_H{horizon}K{execute_steps}"
    roster_path = setup_cohort(cohort_name, pilot_name)

    print(f"\n{'='*60}")
    print(f"CHUNKED BC: H={horizon}, K={execute_steps}, output={output_size}D")
    print(f"  Cohort: {cohort_name}")
    print(f"  Pilot: {pilot_name}")
    print(f"{'='*60}")

    # Initialize pilot
    student = Pilot(cohort_name, pilot_name)
    student.set_mode('train')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student.model.to(device)

    # Transfer frozen weights
    transfer_frozen_weights(SOURCE_COHORT, SOURCE_PILOT, student)

    # Freeze everything, then unlock Commander
    for param in student.model.parameters():
        param.requires_grad = False
    for param in student.model.get_network["Commander"]["Unlock"].parameters():
        param.requires_grad = True

    model = student.model.get_network["Commander"]["Train"]
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean')

    # Data paths
    train_paths, test_paths, _, _ = get_data_paths(cohort_name, pilot_name)
    extractor = student.model.get_data["Commander"]
    print(f"  Data: {len(train_paths)} train, {len(test_paths)} test files")

    # Training
    losses_train, losses_test = [], []
    best_test_loss = float('inf')
    model_path = os.path.join(roster_path, "model.pth")
    best_path = os.path.join(roster_path, "best_model.pth")
    last_path = os.path.join(roster_path, "last_model.pth")
    start_time = time.time()

    with trange(epochs, desc=f"Chunked H={horizon}") as pbar:
        for ep in pbar:
            # Train
            student.model.train()
            epoch_loss = []
            for f in train_paths:
                Xnn, Ynn = extract_data_chunked(f, horizon, action_dim)
                ds = ObservationData(Xnn, Ynn, extractor)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                for inp, label in loader:
                    inp = tuple(t.to(device) for t in inp)
                    label = label.to(device)
                    pred, _ = model(*inp)
                    loss = criterion(pred, label)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    epoch_loss.append((label.shape[0], loss.item()))

            N = sum(n for n, _ in epoch_loss)
            avg_train = sum(n * l for n, l in epoch_loss) / N

            # Test
            student.model.eval()
            test_loss = []
            with torch.no_grad():
                for f in test_paths:
                    Xnn, Ynn = extract_data_chunked(f, horizon, action_dim)
                    ds = ObservationData(Xnn, Ynn, extractor)
                    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
                    for inp, label in loader:
                        inp = tuple(t.to(device) for t in inp)
                        label = label.to(device)
                        pred, _ = model(*inp)
                        loss = criterion(pred, label)
                        test_loss.append((label.shape[0], loss.item()))

            N_test = sum(n for n, _ in test_loss)
            avg_test = sum(n * l for n, l in test_loss) / N_test
            losses_train.append(avg_train)
            losses_test.append(avg_test)

            if avg_test < best_test_loss:
                best_test_loss = avg_test
                torch.save(student.model, model_path)
                torch.save(student.model, best_path)

            torch.save(student.model, last_path)
            pbar.set_postfix(train=f"{avg_train:.5f}", test=f"{avg_test:.5f}",
                           best=f"{best_test_loss:.5f}")

    elapsed = time.time() - start_time
    torch.save({"train": losses_train, "test": losses_test,
                "epochs": epochs, "horizon": horizon, "execute_steps": execute_steps},
               os.path.join(roster_path, "losses_Commander.pt"))

    print(f"\n  Training complete in {elapsed:.0f}s")
    print(f"  Best test loss: {best_test_loss:.5f}")
    print(f"  Model: {model_path}")

    return cohort_name, pilot_name, best_test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="10",
                        help="Comma-separated horizons to test (e.g., '10,20')")
    parser.add_argument("--execute-steps", default=None,
                        help="Comma-separated K values (default: H//3 for each)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    if args.execute_steps:
        exec_steps = [int(k) for k in args.execute_steps.split(",")]
    else:
        exec_steps = [max(2, h // 3) for h in horizons]

    results = []
    for h, k in zip(horizons, exec_steps):
        cohort, pilot, loss = train_chunked_bc(h, k, args.epochs)
        results.append({"horizon": h, "execute_steps": k,
                        "cohort": cohort, "pilot": pilot, "best_test_loss": loss})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  H={r['horizon']}, K={r['execute_steps']}: "
              f"best_test={r['best_test_loss']:.5f}  cohort={r['cohort']}")
