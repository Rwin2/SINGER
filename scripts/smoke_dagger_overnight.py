#!/usr/bin/env python3
"""
Overnight DAgger smoke test on 15 selected branches (9 fail + 6 ok from BC).
Runs: expert eval → BC eval → full DAgger (10 rounds, 30 epochs).

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=1 \
    nohup /home/erwinpi/miniconda3/envs/FiGS/bin/python -u scripts/smoke_dagger_overnight.py \
        > logs/smoke_dagger_overnight.log 2>&1 &
"""
import os, sys, time, json, shutil, random
from datetime import datetime

import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_training_branches, _select_branches_for_round,
    _collect_and_evaluate, _retrain_commander, _swap_model, DAggerPolicy,
    _reset_pilot, _wandb_log_round, DEVICE,
)
from sousvide.instruct.benchmark import evaluate_branches
from sousvide.control.pilot import Pilot

# ── Config ──
SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
COHORT = "SSV_DAGGER_SMOKE_15BR"
PILOT_NAME = "InstinctJester"
BC_MODEL = os.path.join(WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "model.pth")

N_ROUNDS = 10
N_EPOCHS = 30
LR = 2e-5
SEED = 42
MAX_SUCCESS_PER_OBJ = 2

# 15 branches: 9 failures + 6 successes from BC round 1 eval
BRANCH_IDS = {
    "green clock": [0, 4, 8, 20, 5],
    "green and pink leafblower": [1, 3, 8, 48, 43],
    "yellow handheld cordless drill on two boxes": [0, 7, 8, 10, 5],
}

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "visualizations", f"smoke_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE, "logs"), exist_ok=True)

    print("=" * 70)
    print(f"[Smoke DAgger] 15 branches, {N_ROUNDS} rounds, {N_EPOCHS} epochs/round")
    print(f"  Output: {out_dir}")
    print(f"  Started: {ts}")
    print("=" * 70, flush=True)

    # ── Load scene + branches ──
    print("\n[Phase 0] Loading scene + branches...", flush=True)
    scene_data = _get_scene(SCENE, SCENES_CFG)
    all_branches_full = _load_training_branches(BC_COHORT, SCENE, SCENES_CFG)

    # Subset to our 15 branches
    subset = {}
    for obj, ids in BRANCH_IDS.items():
        if obj in all_branches_full:
            subset[obj] = {bid: all_branches_full[obj][bid] for bid in ids
                           if bid in all_branches_full[obj]}
            print(f"  {obj[:35]}: {len(subset[obj])} branches (ids={ids})", flush=True)

    n_total = sum(len(v) for v in subset.values())
    print(f"  Total: {n_total} branches\n", flush=True)

    # ── Phase 1: Expert evaluation ──
    print("=" * 70)
    print("[Phase 1] Expert (MPC) evaluation on 15 branches")
    print("=" * 70, flush=True)

    expert_branches = {obj: [(bid, br) for bid, br in brs.items()] for obj, brs in subset.items()}
    expert_results, expert_diag, expert_sr, expert_cr = evaluate_branches(
        None, None, expert_branches, scene_data,
        label="Expert", output_dir=out_dir, save_plots=True, is_expert=True)
    print(f"\n[Expert] Overall: {expert_sr:.0%} SR, {expert_cr:.0%} CR", flush=True)

    # ── Phase 2: BC baseline evaluation ──
    print("\n" + "=" * 70)
    print("[Phase 2] BC baseline evaluation on 15 branches")
    print("=" * 70, flush=True)

    pilot = Pilot(BC_COHORT, PILOT_NAME)
    pilot.set_mode("deploy")
    bc_results, bc_diag, bc_sr, bc_cr = evaluate_branches(
        pilot, BC_MODEL, expert_branches, scene_data,
        label="BC", output_dir=out_dir, save_plots=True)
    print(f"\n[BC] Overall: {bc_sr:.0%} SR, {bc_cr:.0%} CR", flush=True)

    # ── Phase 3: DAgger training (10 rounds) ──
    print("\n" + "=" * 70)
    print(f"[Phase 3] DAgger training — {N_ROUNDS} rounds, {N_EPOCHS} epochs/round")
    print("=" * 70, flush=True)

    rng = random.Random(SEED)
    cohort_path = os.path.join(WORKSPACE, "cohorts", COHORT)
    model_dir = os.path.join(cohort_path, "roster", PILOT_NAME)
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy2(BC_MODEL, os.path.join(model_dir, "model.pth"))
    shutil.copy2(os.path.join(WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "config.json"),
                 os.path.join(model_dir, "config.json"))

    model_path = BC_MODEL
    all_annotations = []
    round_results = []

    for r in range(1, N_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"[DAgger Round {r}/{N_ROUNDS}]  ({len(all_annotations)} annotations)")
        print(f"{'='*70}", flush=True)

        # Select branches
        if r == 1:
            prior_failures = {}
        else:
            prior_failures = {}
            for d in round_results[-1].get("eval_diag", []):
                if not d["success"]:
                    prior_failures.setdefault(d["object"], []).append(d["branch_id"])

        selected = _select_branches_for_round(subset, prior_failures, rng, r, MAX_SUCCESS_PER_OBJ)

        # Collect
        new_anns, coll_results, coll_diag, coll_sr, coll_cr = \
            _collect_and_evaluate(pilot, model_path, selected, scene_data,
                                   round_i=r, output_dir=out_dir, is_dagger=True)
        all_annotations.extend(new_anns)
        print(f"\n  Collection: {len(new_anns)} annotations, {coll_sr:.0%} SR", flush=True)

        # Retrain
        print(f"\n  Retraining ({N_EPOCHS} epochs, lr={LR})...", flush=True)
        agg_dir = os.path.join(cohort_path, "dagger_data", PILOT_NAME)
        os.makedirs(agg_dir, exist_ok=True)
        agg_file = os.path.join(agg_dir, "aggregated_annotations.pt")
        torch.save(all_annotations, agg_file)
        dst_model = os.path.join(model_dir, "model.pth")
        shutil.copy2(BC_MODEL, dst_model)
        _retrain_commander(COHORT, PILOT_NAME, agg_file, Nep=N_EPOCHS, lim_sv=10,
                           lr=LR, bc_cohort_name=BC_COHORT, dagger_only=False,
                           freeze_vision=True)
        model_path = dst_model
        round_model = os.path.join(model_dir, f"model_round_{r}.pth")
        shutil.copy2(dst_model, round_model)

        # Evaluate on ALL 15 branches
        print(f"\n  Evaluating on all {n_total} branches...", flush=True)
        eval_branches_all = {obj: [(bid, br) for bid, br in brs.items()]
                              for obj, brs in subset.items()}
        eval_results, eval_diag, eval_sr, eval_cr = evaluate_branches(
            pilot, model_path, eval_branches_all, scene_data,
            label=f"Eval_R{r}", output_dir=out_dir, save_plots=True)

        round_results.append({
            "round": r, "n_annotations": len(new_anns),
            "n_total_annotations": len(all_annotations),
            "collection": {"sr": coll_sr, "cr": coll_cr},
            "eval": {"sr": eval_sr, "cr": eval_cr},
            "eval_per_object": eval_results,
            "eval_diag": eval_diag,
        })

        # Progress table
        print(f"\n{'='*70}")
        print(f"[Progress] After Round {r}/{N_ROUNDS}")
        print(f"  {'Round':8} {'Coll SR':>8} {'Eval SR':>8} {'Eval CR':>8} {'Ann':>8}")
        print(f"  {'-'*40}")
        for rr in round_results:
            print(f"  R{rr['round']:<7} {rr['collection']['sr']:>7.0%}"
                  f" {rr['eval']['sr']:>7.0%} {rr['eval']['cr']:>7.0%}"
                  f" {rr['n_total_annotations']:>8}")
        print(flush=True)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"[FINAL] Expert: {expert_sr:.0%} SR, {expert_cr:.0%} CR")
    print(f"[FINAL] BC baseline: {bc_sr:.0%} SR, {bc_cr:.0%} CR")
    if round_results:
        final = round_results[-1]
        print(f"[FINAL] DAgger R{final['round']}: {final['eval']['sr']:.0%} SR, {final['eval']['cr']:.0%} CR")
    print(f"{'='*70}", flush=True)

    # Save summary
    summary = {
        "timestamp": ts, "branches": BRANCH_IDS, "n_rounds": N_ROUNDS,
        "expert_sr": expert_sr, "expert_cr": expert_cr,
        "expert_per_object": expert_results,
        "bc_sr": bc_sr, "bc_cr": bc_cr,
        "bc_per_object": bc_results,
        "rounds": [{k: v for k, v in rr.items() if k != "eval_diag"} for rr in round_results],
    }
    with open(os.path.join(out_dir, "smoke_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary -> {out_dir}/smoke_summary.json", flush=True)


if __name__ == "__main__":
    main()
