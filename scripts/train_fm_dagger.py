#!/usr/bin/env python3
"""
Flow Matching + DAgger for SINGER.

This script runs DAgger using a flow matching loss instead of MSE for Commander retraining.
Uses the standard SINGER DAgger pipeline but swaps the loss function.

Key idea: DAgger collects (state, expert_action) corrections. Instead of training
Commander with MSE(pred_action, expert_action), we train with flow matching:
  - Sample t ~ U[0,1]
  - Interpolate: x_t = t*expert_action + (1-t)*noise
  - Target velocity: v = expert_action - noise
  - Loss: MSE(predicted_velocity, target_velocity)

At inference: Euler-integrate velocity field from noise -> action (10 steps).

This handles multimodal expert corrections better than MSE.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n FiGS python scripts/train_fm_dagger.py \
        --base-cohort ssv_BC_FM_FM_ONLY \
        --base-pilot InstinctJester \
        --n-iterations 8 \
        [--smoke-test]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up ACADOS
os.environ.setdefault("ACADOS_SOURCE_DIR", "/data/erwinpi/FiGS-Standalone/acados")

from sousvide.control.pilot import Pilot
from sousvide.instruct.train_dagger import (
    train_dagger_policy, _get_scene, _get_pkl, _preload_all_pkls,
    _evaluate_run, _swap_model, _save_model_checkpoint, DEVICE,
    MixedPolicy,
)
from sousvide.instruct.synthesized_data import (
    generate_dataset, get_data_paths, extract_data, ObservationData
)
from sousvide.control.policies.FlowMatchingCommander import (
    FlowMatchingSchedule, FlowMatchingCommanderWrapper
)


def retrain_commander_fm(
    cohort_name: str,
    pilot_name: str,
    annotations: list,
    n_epochs: int = 15,
    lr: float = 2e-5,
    num_flow_steps: int = 10,
    gaussian_weighting: bool = False,
    bc_data_paths: list = None,
    oversample: int = 1,
):
    """
    Retrain Commander using Flow Matching loss on DAgger annotations + BC data.

    Replaces _retrain_commander_ewc's MSE loss with FM velocity matching.

    Args:
        cohort_name: Current DAgger cohort
        pilot_name: Pilot name
        annotations: List of DAgger annotation dicts with 'xnn' and 'u' keys
        n_epochs: Training epochs per DAgger iteration
        lr: Learning rate
        num_flow_steps: ODE integration steps for FM
        gaussian_weighting: Use Gaussian t-sampling (DreamZero-style)
        bc_data_paths: Optional list of BC observation file paths to mix in
        oversample: Repeat DAgger data this many times
    """
    if not annotations:
        print("  [retrain_fm] No annotations — skipping.")
        return

    default_mfn = np.array([0.3, 0.3], dtype=np.float32)

    # Convert annotations to Xnn, Ynn format
    Xnn_dag, Ynn_dag = [], []
    for ann in annotations:
        xnn = ann.get("xnn")
        if not xnn:
            continue
        ynn = {
            "unn": torch.tensor(ann["u"], dtype=torch.float32),
            "mfn": torch.tensor(default_mfn),
            "onn": torch.tensor(ann["x"], dtype=torch.float32),
        }
        Xnn_dag.append(xnn)
        Ynn_dag.append(ynn)

    if not Xnn_dag:
        print("  [retrain_fm] No valid xnn entries — skipping.")
        return

    if oversample > 1:
        Xnn_dag = Xnn_dag * oversample
        Ynn_dag = Ynn_dag * oversample

    print(f"  [retrain_fm] {len(Xnn_dag)} DAgger samples")

    # Load pilot
    pilot = Pilot(cohort_name, pilot_name)
    pilot.set_mode('train')
    model = pilot.model

    # Check if model is FM wrapper or standard SVNet
    is_fm_wrapper = isinstance(model, FlowMatchingCommanderWrapper)

    if is_fm_wrapper:
        model.to(DEVICE)
        # Freeze SVNet (HistoryEncoder + VisionMLP)
        for param in model.svnet.parameters():
            param.requires_grad = False
        # Train FM commander + time embedding
        trainable = list(model.fm_commander.parameters()) + list(model.time_embed.parameters())
        output_size = model.output_size
    else:
        # Standard SVNet — wrap it with FM
        print("  [retrain_fm] Wrapping SVNet with FlowMatchingCommander")
        output_size = 4  # standard 4D action
        fm_model = FlowMatchingCommanderWrapper(
            model, output_size=output_size,
            num_flow_steps=num_flow_steps,
            time_embed_dim=32, hidden_sizes=[256, 128]
        ).to(DEVICE)
        # Transfer Commander weights as initialization
        # (The FM commander is new, so this starts from scratch)
        for param in fm_model.svnet.parameters():
            param.requires_grad = False
        trainable = list(fm_model.fm_commander.parameters()) + list(fm_model.time_embed.parameters())
        model = fm_model
        # Update pilot's model
        pilot.model = model

    optimizer = optim.Adam(trainable, lr=lr)
    schedule = FlowMatchingSchedule(
        action_dim=output_size, device=str(DEVICE),
        num_steps=num_flow_steps, gaussian_weighting=gaussian_weighting
    )

    extractor = pilot.model.get_data["Commander"] if hasattr(pilot.model, 'get_data') else model.get_data["Commander"]

    # Combine BC + DAgger data
    all_Xnn, all_Ynn = list(Xnn_dag), list(Ynn_dag)

    if bc_data_paths:
        for bc_path in bc_data_paths:
            try:
                bc_Xnn, bc_Ynn = extract_data(bc_path)
                all_Xnn.extend(bc_Xnn)
                all_Ynn.extend(bc_Ynn)
            except Exception as e:
                print(f"  [retrain_fm] Warning: could not load BC data {bc_path}: {e}")

    print(f"  [retrain_fm] Total training samples: {len(all_Xnn)} "
          f"({len(Xnn_dag)} DAgger + {len(all_Xnn) - len(Xnn_dag)} BC)")

    dataset = ObservationData(all_Xnn, all_Ynn, extractor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

    # Training loop
    best_loss = float('inf')
    model_path = os.path.join(pilot.path, "model.pth")
    best_path = os.path.join(pilot.path, "best_model.pth")

    for ep in range(n_epochs):
        model.train()
        epoch_loss = []

        for inp, label in loader:
            inp = tuple(t.to(DEVICE) for t in inp)
            label = label.to(DEVICE)
            B = label.shape[0]

            # Flow matching training
            t = schedule.sample_timesteps(B, DEVICE)
            x_t, target_v = schedule.interpolate(label, t)

            if is_fm_wrapper or isinstance(model, FlowMatchingCommanderWrapper):
                pred_v = model.forward_flow(*inp, noisy_action=x_t, flow_time=t)
            else:
                raise RuntimeError("Model is not FM-compatible")

            weights = schedule.compute_loss_weights(t)
            loss = ((pred_v - target_v) ** 2).mean(dim=-1)
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss.append((B, loss.item()))

        N = sum(n for n, _ in epoch_loss)
        avg_loss = sum(n * l for n, l in epoch_loss) / N

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, model_path)
            torch.save(model, best_path)

        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            print(f"    Epoch {ep+1}/{n_epochs}: loss={avg_loss:.5f} (best={best_loss:.5f})")

    torch.save(model, os.path.join(pilot.path, "last_model.pth"))
    print(f"  [retrain_fm] Done. Best loss: {best_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cohort", default="ssv_BC_FM_FM_ONLY",
                        help="BC cohort to start from")
    parser.add_argument("--base-pilot", default="InstinctJester")
    parser.add_argument("--n-iterations", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test with 2 iterations, 5 rollouts")
    args = parser.parse_args()

    print("=" * 60)
    print("FM + DAgger is not yet integrated into train_dagger_policy().")
    print("This is a placeholder showing the approach.")
    print("To fully integrate, _retrain_commander_ewc in train_dagger.py")
    print("needs to support a 'loss_type' parameter (mse vs flow_matching).")
    print("=" * 60)
    print("\nFor now, use the standard DAgger pipeline with the FM-trained")
    print("BC model as the starting point:")
    print(f"\n  python ssv_muilti3dgs_campaign.py dagger \\")
    print(f"    --config-file configs/experiment/ssv_dagger_fm.yml")
