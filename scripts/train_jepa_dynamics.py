#!/usr/bin/env python3
"""
JEPA dynamics auxiliary loss for SINGER chunked BC.

Learns to predict the *target encoder*'s y_vis at multiple future horizons
(t+1, t+5, t+10, t+20 steps -> 50ms / 250ms / 500ms / 1s ahead) on top of the
existing chunked H=10 K=3 BC training. Inspired by JEPA / BYOL: target is an
EMA copy of the student's own VisionMLP, gradients flow only through the
student via a small predictor MLP, anti-collapse comes from the EMA + stop-grad.

This is the "DreamZero done small" — instead of fine-tuning a 14B video
diffusion model, we predict the 128-dim learned visual feature, which is
SINGER's full learned visual perception (the only path image data takes into
the policy).

Warm-starts from the existing chunked_h10 best checkpoint. New cohort:
ssv_BC_JEPA_H10K3. Does NOT touch the BC pipeline files.

Usage:
    cd /data/erwinpi/SINGER
    bash scripts/run_jepa_nohup.sh
"""
import argparse
import copy
import gc
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import (
    ensure_torch_tensor,
    get_data_paths,
)

# ---- config -----------------------------------------------------------------
SOURCE_COHORT = "ssv_BC_CHUNKED_H10K3"          # warm-start
SOURCE_PILOT  = "InstinctJester_chunked_h10"
TARGET_COHORT = "ssv_BC_JEPA_H10K3"
TARGET_PILOT  = "InstinctJester_chunked_h10"    # same profile (H=10 chunked)

CHUNK_HORIZON   = 10
ACTION_DIM      = 4
FUTURE_HORIZONS = (1, 5, 10, 20)                # in 50ms steps -> 50ms..1s
JEPA_DIM        = 128                           # VisionMLP output_size
TX_COM_DIM      = 8                             # CommandSV state slice (z + quat + ?), see config

EPOCHS          = 25
BATCH_SIZE      = 64
LR              = 5e-5                          # warm-start: smaller than fresh
JEPA_WEIGHT     = 0.5                           # L_total = L_action + w * L_jepa
EMA_TAU         = 0.99


# ---- data -------------------------------------------------------------------
def extract_jepa_samples(path: str,
                         chunk_horizon: int = CHUNK_HORIZON,
                         action_dim: int = ACTION_DIM,
                         future_horizons=FUTURE_HORIZONS):
    """Build (xnn_now, future_xnn_list, ynn_chunked) tuples per timestep.

    Trims the rollout tail so both the H-step action chunk AND the deepest
    future-horizon (max(future_horizons)) are inside the rollout.
    """
    # Force-load to CPU: SINGER's saved obs files contain CUDA tensors from
    # prior sessions. Without map_location='cpu' they pile up in VRAM.
    obs = torch.load(path, map_location="cpu")
    max_h = max(future_horizons)
    tail = max(chunk_horizon, max_h + 1)

    samples = []
    for observations in obs["data"]:
        Xnn = []
        for xnn_raw in observations["Xnn"]:
            for k, v in xnn_raw.items():
                t = ensure_torch_tensor(v)
                xnn_raw[k] = t.detach().cpu() if t.is_cuda else t
            Xnn.append(xnn_raw)
        Ynn = []
        for ynn_raw in observations["Ynn"]:
            for k, v in ynn_raw.items():
                t = ensure_torch_tensor(v)
                ynn_raw[k] = t.detach().cpu() if t.is_cuda else t
            Ynn.append(ynn_raw)

        N = len(Xnn)
        if N < tail + 1:
            continue
        for i in range(N - tail):
            unn_chunk = torch.cat([Ynn[i + j]["unn"] for j in range(chunk_horizon)])
            ynn_chunked = {
                "unn": unn_chunk,
                "mfn": Ynn[i]["mfn"],
                "onn": Ynn[i]["onn"],
            }
            futures = [Xnn[i + h] for h in future_horizons]
            samples.append((Xnn[i], futures, ynn_chunked))
    return samples


class JEPADataset(Dataset):
    def __init__(self, samples, extractor):
        self.samples = samples
        self.extractor = extractor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xnn_now, futures, ynn = self.samples[idx]
        inp, label = self.extractor(xnn_now, ynn)        # 5-tuple, (H*Adim,)
        future_imgs   = torch.stack([f["img_vis"] for f in futures])  # (Hf, C, H, W)
        future_tx_vis = torch.stack([f["tx_vis"] for f in futures])   # (Hf, S)
        return inp, label, future_imgs, future_tx_vis


# ---- model glue -------------------------------------------------------------
def build_predictor(y_vis_dim: int, tx_com_dim: int, action_chunk_dim: int,
                    n_horizons: int):
    """Tiny MLP: [y_vis_now ; tx_com ; action_chunk_pred] -> n_horizons * y_vis_dim."""
    in_dim = y_vis_dim + tx_com_dim + action_chunk_dim
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_horizons * y_vis_dim),
    )


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, tau: float):
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)
    for bt, bs in zip(target.buffers(), source.buffers()):
        bt.data.copy_(bs.data)


def warm_start_from(source_cohort: str, source_pilot: str, target: Pilot):
    src = Pilot(source_cohort, source_pilot)
    src_state = torch.load(
        os.path.join(src.path, "model.pth"),
        map_location="cpu",
        weights_only=False,
    ).state_dict()
    tgt_state = target.model.state_dict()
    n = 0
    for k in src_state:
        if k in tgt_state and src_state[k].shape == tgt_state[k].shape:
            tgt_state[k] = src_state[k]
            n += 1
    target.model.load_state_dict(tgt_state)
    print(f"  Warm-started {n} tensors from {source_cohort}/{source_pilot}")


def setup_target_cohort():
    import shutil, json
    roster = os.path.join(WORKSPACE, "cohorts", TARGET_COHORT, "roster", TARGET_PILOT)
    os.makedirs(roster, exist_ok=True)

    # Pilot config: copy from source pilot config
    src_cfg = os.path.join(WORKSPACE, "configs", "pilots", f"{TARGET_PILOT}.json")
    dst_cfg = os.path.join(roster, "config.json")
    if not os.path.exists(dst_cfg):
        shutil.copy2(src_cfg, dst_cfg)

    # Symlink the H10 cohort's observation_data (same data, no need to regen)
    src_obs = os.path.join(WORKSPACE, "cohorts", SOURCE_COHORT,
                           "observation_data", SOURCE_PILOT)
    dst_obs_base = os.path.join(WORKSPACE, "cohorts", TARGET_COHORT, "observation_data")
    dst_obs = os.path.join(dst_obs_base, TARGET_PILOT)
    if not os.path.exists(dst_obs):
        os.makedirs(dst_obs_base, exist_ok=True)
        os.symlink(src_obs, dst_obs)
    return roster


# ---- training loop ----------------------------------------------------------
def run_epoch(loader, student, target_vmlp, predictor, opt, criterion,
              device, jepa_w, train: bool, action_chunk_dim: int):
    student.model.train(train)
    predictor.train(train)
    target_vmlp.eval()

    sums = {"action": 0.0, "jepa": 0.0, "total": 0.0, "n": 0}
    n_horizons = len(FUTURE_HORIZONS)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inp, label, future_imgs, future_tx_vis in loader:
            inp = tuple(t.to(device, non_blocking=True) for t in inp)
            label = label.to(device, non_blocking=True)
            future_imgs   = future_imgs.to(device, non_blocking=True)
            future_tx_vis = future_tx_vis.to(device, non_blocking=True)

            # ---- action prediction (and y_vis_now via the same forward) ----
            tx_com, obj_com, dxu_par, img_vis, tx_vis = inp
            y_vis_now, _ = student.model.network["VisionMLP"](img_vis, tx_vis)
            _, z_par = student.model.network["HistoryEncoder"](dxu_par)
            action_pred, _ = student.model.network["CommanderSV"](
                tx_com, obj_com, z_par, y_vis_now)

            L_action = criterion(action_pred, label)

            # ---- JEPA: predict target_vmlp's y_vis at the 4 future horizons ----
            pred_in = torch.cat([y_vis_now, tx_com, action_pred.detach()], dim=-1)
            future_pred = predictor(pred_in).view(-1, n_horizons, JEPA_DIM)

            B, Hf, C, H, W = future_imgs.shape
            with torch.no_grad():
                target_y = target_vmlp(
                    future_imgs.view(B * Hf, C, H, W),
                    future_tx_vis.view(B * Hf, -1),
                )[0].view(B, Hf, JEPA_DIM)
            L_jepa = criterion(future_pred, target_y.detach())

            L_total = L_action + jepa_w * L_jepa

            if train:
                opt.zero_grad()
                L_total.backward()
                opt.step()
                ema_update(target_vmlp, student.model.network["VisionMLP"], EMA_TAU)

            B0 = label.shape[0]
            sums["action"] += L_action.item() * B0
            sums["jepa"]   += L_jepa.item() * B0
            sums["total"]  += L_total.item() * B0
            sums["n"]      += B0

    return {k: (v / sums["n"] if k != "n" else v) for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--jepa-weight", type=float, default=JEPA_WEIGHT)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 1 epoch, first train file only")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    roster = setup_target_cohort()
    student = Pilot(TARGET_COHORT, TARGET_PILOT)
    student.set_mode("train")
    student.model.to(device)

    warm_start_from(SOURCE_COHORT, SOURCE_PILOT, student)

    # Predictor & target encoder
    action_chunk_dim = CHUNK_HORIZON * ACTION_DIM
    predictor = build_predictor(JEPA_DIM, TX_COM_DIM, action_chunk_dim,
                                len(FUTURE_HORIZONS)).to(device)
    target_vmlp = copy.deepcopy(student.model.network["VisionMLP"]).to(device)
    for prm in target_vmlp.parameters():
        prm.requires_grad = False

    # Unlock everything that should train: all of student.model + predictor
    for prm in student.model.parameters():
        prm.requires_grad = True

    params = list(student.model.parameters()) + list(predictor.parameters())
    opt = optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss(reduction="mean")

    train_paths, test_paths, _, _ = get_data_paths(TARGET_COHORT, TARGET_PILOT)
    if args.smoke:
        train_paths = train_paths[:1]
        test_paths  = test_paths[:1]
    print(f"Files: {len(train_paths)} train, {len(test_paths)} test")

    extractor = student.model.get_data["Commander"]

    losses = []
    best_test = float("inf")
    best_path = os.path.join(roster, "best_model.pth")
    last_path = os.path.join(roster, "last_model.pth")
    model_path = os.path.join(roster, "model.pth")

    epochs = 1 if args.smoke else args.epochs
    t0 = time.time()
    with trange(epochs, desc="JEPA H10") as pbar:
        for ep in pbar:
            # Build datasets fresh per epoch to keep memory bounded
            for split, paths, train_flag in (
                ("train", train_paths, True),
                ("test",  test_paths,  False),
            ):
                ep_sum = {"action": 0.0, "jepa": 0.0, "total": 0.0, "n": 0}
                for f in paths:
                    samples = extract_jepa_samples(f)
                    if not samples:
                        continue
                    ds = JEPADataset(samples, extractor)
                    loader = DataLoader(
                        ds, batch_size=args.batch_size,
                        shuffle=train_flag, num_workers=0, pin_memory=False,
                    )
                    s = run_epoch(loader, student, target_vmlp, predictor, opt,
                                  criterion, device, args.jepa_weight,
                                  train=train_flag,
                                  action_chunk_dim=action_chunk_dim)
                    for k in ("action", "jepa", "total"):
                        ep_sum[k] += s[k] * s["n"]
                    ep_sum["n"] += s["n"]
                    del samples, ds, loader
                    gc.collect()
                avg = {k: ep_sum[k] / max(ep_sum["n"], 1) for k in ("action","jepa","total")}
                if split == "train":
                    tr = avg
                else:
                    te = avg

            losses.append({"epoch": ep, "train": tr, "test": te})
            if te["action"] < best_test:
                best_test = te["action"]
                torch.save(student.model, best_path)
                torch.save(student.model, model_path)
            torch.save(student.model, last_path)

            pbar.set_postfix(
                tr_a=f"{tr['action']:.5f}", tr_j=f"{tr['jepa']:.5f}",
                te_a=f"{te['action']:.5f}", te_j=f"{te['jepa']:.5f}",
                best=f"{best_test:.5f}",
            )

    torch.save({"losses": losses, "horizons": FUTURE_HORIZONS,
                "jepa_weight": args.jepa_weight, "lr": args.lr,
                "warm_from": f"{SOURCE_COHORT}/{SOURCE_PILOT}"},
               os.path.join(roster, "losses_JEPA.pt"))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min  best test action loss = {best_test:.5f}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
