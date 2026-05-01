#!/usr/bin/env python3
"""
HW1-style DAgger on 15 TRAINING branches (5/object: 3 success + 2 worst fail).

Pure HW1 formalism:
  - Learner rollout (beta=0), expert relabels at learner-visited states
  - Aggregate all annotations raw (no filtering)
  - Retrain from BC weights each round on BC + DAgger data

Crash-resilient: saves annotations + results after each step, resumes on restart.

Branch selection from canonical bc_vs_dagger results.json (seed=42 mapping):
  Clock:       5, 115 (COLL)  +  47, 91, 64 (OK)
  Leafblower:  1, 111 (COLL)  +  37, 15, 81 (OK)
  Boxes:       2, 27  (COLL)  +  10, 35, 17 (OK)

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=1 \
    conda run --no-capture-output -n FiGS python -u scripts/smoke_test_dagger_hw1.py
"""
import os, sys, time, json, math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _load_all_branches, _evaluate_run, _swap_model, _get_pkl,
    _preload_bc_trajectories, _save_benchmark_plotly, _make_terminal_fn,
    _retrain_commander,
    DEVICE, SUCCESS_RADIUS,
)
from sousvide.control.pilot import Pilot
from figs.control.vehicle_rate_mpc import VehicleRateMPC

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
PILOT_NAME = "InstinctJester"
BC_MODEL_PATH = os.path.join(
    WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "model.pth")

POLICY_NAME = "vrmpc_rrt"
FRAME_NAME = "carl"

N_ROUNDS = 5
N_EPOCHS_PER_ROUND = 30
LR = 2e-5
BATCH_SIZE = 64
COHORT = "SSV_DAGGER_HW1_15BR"

# Checkpoint directory
CKPT_DIR = os.path.join(WORKSPACE, "cohorts", COHORT, "checkpoints")

# Selected branches: actual_branch_id from all_branches[]
# 3 success + 2 worst collision per object
SELECTED = {
    "green clock":        {"fail": [5, 115], "success": [47, 91, 64]},
    "green and pink leafblower": {"fail": [1, 111], "success": [37, 15, 81]},
    "yellow handheld cordless drill on two boxes": {"fail": [2, 27], "success": [10, 35, 17]},
}


# ──────────────────────────────────────────────────────────────────────────────
# HW1-style DAgger Policy
# ──────────────────────────────────────────────────────────────────────────────
class DAggerPolicy:
    """Pure learner rollout + expert relabeling. Pilot uses its OWN prev action."""
    def __init__(self, expert: VehicleRateMPC, pilot: Pilot):
        self.expert = expert
        self.pilot = pilot
        self.hz = pilot.hz
        self.nzcr = pilot.nzcr
        self.annotations = []
        self._u_pilot_prev = np.zeros(4)

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        u_expert, _, _, _ = self.expert.control(tcr, xcr, upr, obj, icr, zcr)
        u_pilot, znn, adv, xnn, tsol = self.pilot.OODA(
            self._u_pilot_prev, tcr, xcr, obj, icr, zcr)
        self._u_pilot_prev = u_pilot.copy()
        xnn_cpu = {k: v.detach().cpu() for k, v in xnn.items()} if xnn else {}
        self.annotations.append({
            "xnn": xnn_cpu, "x": xcr.copy(), "u": u_expert.copy(),
            "t": tcr, "query": obj,
        })
        return u_pilot, znn, adv, tsol

    def reset(self):
        self.annotations = []
        self._u_pilot_prev = np.zeros(4)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _short(name):
    return name.replace(" ", "_")[:30]


def reset_pilot(pilot):
    pilot.hy_flag = False
    pilot.hy_idx = 0
    pilot.DxU.zero_()
    if hasattr(pilot, 'Znn') and isinstance(pilot.Znn, torch.Tensor):
        pilot.Znn.zero_()
    if hasattr(pilot, 'chunk_buf'):
        pilot.chunk_buf = None
        pilot.chunk_step = 0


def run_sim(simulator, policy, tXUi, obj_name, early_stop_fn=None):
    return simulator.simulate(
        policy=policy, t0=float(tXUi[0, 0]), tf=float(tXUi[0, -1]),
        x0=tXUi[1:11, 0].copy(), obj=np.zeros((18, 1)), query=obj_name,
        vision_processor=None, verbose=False,
        early_stop_fn=early_stop_fn,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ckpt_path(round_i, step):
    """Return checkpoint file path for a given round and step."""
    return os.path.join(CKPT_DIR, f"round_{round_i}_{step}.pt")


def _save_checkpoint(round_i, step, data):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = _ckpt_path(round_i, step)
    torch.save(data, path)
    print(f"  [ckpt] Saved {step} -> {path}")


def _load_checkpoint(round_i, step):
    path = _ckpt_path(round_i, step)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def find_resume_point():
    """Find the last completed round and step.
    Returns (last_completed_round, all_prior_annotations).
    Round 0 = no DAgger done yet (start from scratch).
    """
    if not os.path.isdir(CKPT_DIR):
        return 0, []

    all_annotations = []
    last_completed = 0

    for r in range(1, N_ROUNDS + 1):
        # Check if collection checkpoint exists
        coll_data = _load_checkpoint(r, "collected")
        if coll_data is None:
            break
        all_annotations.extend(coll_data["annotations"])

        # Check if retrain + eval completed
        eval_data = _load_checkpoint(r, "evaluated")
        if eval_data is None:
            # Collection done but retrain/eval not — resume from retrain
            print(f"  [resume] Round {r}: collection done, retrain/eval pending")
            return r - 1, all_annotations  # we'll redo retrain+eval for round r
        last_completed = r

    return last_completed, all_annotations


# ──────────────────────────────────────────────────────────────────────────────
# Collection + evaluation in one pass (no separate BC baseline needed)
# ──────────────────────────────────────────────────────────────────────────────
def collect_and_evaluate(pilot, model_path, branches_dict, scene_data, round_i,
                         output_dir=None, is_dagger=True):
    """
    Run learner on all branches. If is_dagger=True, also collect expert annotations.
    Returns (annotations, per_object_results, diagnostics, overall_sr, overall_cr).
    The collection IS the evaluation — no separate eval pass needed.
    """
    from scipy.spatial import cKDTree
    pilot = _swap_model(pilot, model_path)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    all_annotations = []
    per_object = {}
    all_diag = []
    label = f"R{round_i}" if is_dagger else "BC"

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        branches = branches_dict[obj_name]
        successes, collisions, goal_dists = [], [], []
        obj_runs_for_plot = []

        for br_id, tXUi in branches:
            reset_pilot(pilot)
            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )

            if is_dagger:
                expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)
                dagger_pol = DAggerPolicy(expert, pilot)
                policy = dagger_pol
            else:
                policy = pilot

            _t = time.time()
            result = run_sim(simulator, policy, tXUi, obj_name,
                             early_stop_fn=terminal_fn)
            Xro = result[1]
            sim_s = time.time() - _t

            ev = _evaluate_run(Xro, obj_target, epcds_arr,
                               env_min=scene_data.get("env_min"),
                               env_max=scene_data.get("env_max"),
                               tXUi=tXUi, idx0=0)

            s = float(ev["success"])
            c = float(ev["collision"])
            gd = float(ev["goal_dist"])
            successes.append(s); collisions.append(c); goal_dists.append(gd)
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            status = "OK" if s else ("COLL" if c else "MISS")
            reason = term_info.get("reason", "timeout")
            dev = ev.get("mean_pos_dev", float("nan"))
            fov = ev.get("fov_pct", float("nan"))

            if is_dagger:
                n_ann = len(dagger_pol.annotations)
                all_annotations.extend(dagger_pol.annotations)
                print(f"  [{label}] {status:4s}  '{_short(obj_name)}'  branch={br_id}"
                      f"  d0={d0:.1f}m  goal={gd:.2f}m  {n_ann} ann"
                      f"  fov={fov:.0%}  stop={reason}  ({sim_s:.0f}s)")
            else:
                print(f"  [{label}] {status:4s}  '{_short(obj_name)}'  branch={br_id}"
                      f"  d0={d0:.1f}m  goal={gd:.2f}m"
                      f"  fov={fov:.0%}  stop={reason}  ({sim_s:.0f}s)")

            all_diag.append({
                "object": obj_name, "branch_id": br_id, "d0": d0,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "goal_dist": gd, "min_goal_dist": float(ev["min_goal_dist"]),
                "fov_pct": float(fov) if not math.isnan(fov) else None,
                "stop_reason": reason,
                "sim_time_s": round(sim_s, 1),
            })
            if output_dir:
                obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy()))

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_m = float(np.mean(goal_dists)) if goal_dists else float("nan")
        print(f"  [{label}] -- '{obj_name[:25]}'  success={sr:.0%}  collision={cr:.0%}"
              f"  goal={gd_m:.2f}m  n={len(branches)}")
        per_object[obj_name] = {
            "success_rate": sr, "collision_rate": cr, "goal_dist": gd_m,
            "n_eval": len(branches),
        }

        if output_dir and obj_runs_for_plot:
            html_path = os.path.join(output_dir, "plots", f"{label}_{_short(obj_name)}.html")
            _save_benchmark_plotly(
                obj_runs=obj_runs_for_plot, obj_target=obj_target,
                simulator=simulator,
                obj_name=f"{label} -- {obj_name} ({sr:.0%} success)",
                save_path=html_path,
            )

    overall_sr = float(np.mean([v["success_rate"] for v in per_object.values()]))
    overall_cr = float(np.mean([v["collision_rate"] for v in per_object.values()]))
    return all_annotations, per_object, all_diag, overall_sr, overall_cr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "visualizations", f"hw1_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("[HW1 DAgger — 15 Training Branches (3 OK + 2 COLL per object)]")
    print(f"  BC model    : {BC_MODEL_PATH}")
    print(f"  Rounds      : {N_ROUNDS}")
    print(f"  Epochs/round: {N_EPOCHS_PER_ROUND}")
    print(f"  LR          : {LR}")
    print(f"  Cohort      : {COHORT}")
    print(f"  Checkpoints : {CKPT_DIR}")
    print(f"  Output      : {out_dir}")
    print("=" * 70)

    # ── Phase 0: Load scene + branches ──
    print("\n[Phase 0] Loading scene + BC trajectories...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    queries = scene_data["queries"]
    flights = [[SCENE, SCENE]]
    _preload_bc_trajectories(BC_COHORT, flights, SCENES_CFG)

    all_branch_data = {}
    selected_branches = {}
    for obj_name in queries:
        if obj_name not in SELECTED:
            continue
        pkl_data = _get_pkl(SCENE, obj_name, SCENES_CFG)
        if pkl_data is None:
            continue
        all_branches = _load_all_branches(
            SCENE, obj_name, os.path.join(WORKSPACE, "cohorts", BC_COHORT),
            SCENES_CFG, pkl_data["tXUi"])
        all_branch_data[obj_name] = all_branches

        sel = SELECTED[obj_name]
        branch_ids = sel["fail"] + sel["success"]
        selected_branches[obj_name] = [(bid, all_branches[bid]) for bid in branch_ids]
        print(f"  '{obj_name}': {len(branch_ids)} branches  ids={branch_ids}"
              f"  (from {len(all_branches)} total)")

    # ── Check for resume ──
    last_completed, prior_annotations = find_resume_point()
    if last_completed > 0:
        print(f"\n  [resume] Resuming from round {last_completed + 1}"
              f" ({len(prior_annotations)} prior annotations)")

    pilot = Pilot(BC_COHORT, PILOT_NAME)
    pilot.set_mode('deploy')

    # ── DAgger rounds ──
    model_path = BC_MODEL_PATH
    all_annotations = list(prior_annotations)
    round_results = []

    # If resuming, reload the model from last completed round
    if last_completed > 0:
        model_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "roster", PILOT_NAME)
        round_model = os.path.join(model_dir, f"model_round_{last_completed}.pth")
        fallback_model = os.path.join(model_dir, "model.pth")
        if os.path.exists(round_model):
            model_path = round_model
            print(f"  [resume] Using model from round {last_completed}: {model_path}")
        elif os.path.exists(fallback_model):
            model_path = fallback_model
            print(f"  [resume] Using model.pth (round {last_completed}): {model_path}")

    start_round = last_completed + 1

    # Round 0 = BC baseline (collection without DAgger = just eval)
    if start_round <= 1:
        # First DAgger round also serves as BC baseline — the learner IS the BC model
        # on the first pass, so its performance = BC performance
        pass

    for r in range(start_round, N_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"[DAgger Round {r}/{N_ROUNDS}]"
              f"  (total annotations so far: {len(all_annotations)})")
        print(f"{'='*70}")

        # ── Collect (= run learner + expert relabel) ──
        # Check if collection already done for this round
        coll_ckpt = _load_checkpoint(r, "collected")
        if coll_ckpt is not None:
            new_anns = coll_ckpt["annotations"]
            print(f"  [resume] Round {r} collection loaded from checkpoint"
                  f" ({len(new_anns)} annotations)")
        else:
            print(f"\n  Collecting DAgger data (learner rollout + expert relabel)...")
            new_anns, coll_results, coll_diag, coll_sr, coll_cr = \
                collect_and_evaluate(pilot, model_path, selected_branches,
                                     scene_data, round_i=r, output_dir=out_dir,
                                     is_dagger=True)
            print(f"  Round {r}: {len(new_anns)} new annotations"
                  f"  |  learner: {coll_sr:.0%} success, {coll_cr:.0%} collision")

            # Save collection checkpoint
            _save_checkpoint(r, "collected", {
                "annotations": new_anns,
                "results": coll_results,
                "diagnostics": coll_diag,
                "overall_sr": coll_sr,
                "overall_cr": coll_cr,
            })

        all_annotations.extend(new_anns)
        print(f"  Total annotations: {len(all_annotations)}")

        # ── Retrain ──
        print(f"\n  Retraining Commander (BC + DAgger, {N_EPOCHS_PER_ROUND} epochs, lr={LR})...")

        # Save aggregated annotations for _retrain_commander
        agg_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "dagger_data", PILOT_NAME)
        os.makedirs(agg_dir, exist_ok=True)
        agg_file = os.path.join(agg_dir, "aggregated_annotations.pt")
        torch.save(all_annotations, agg_file)

        # Reload BC weights and retrain using pipeline function
        import shutil
        model_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "roster", PILOT_NAME)
        os.makedirs(model_dir, exist_ok=True)
        dst_model = os.path.join(model_dir, "model.pth")
        shutil.copy2(BC_MODEL_PATH, dst_model)

        _retrain_commander(
            cohort_name=COHORT,
            pilot_name=PILOT_NAME,
            aggregated_file=agg_file,
            Nep=N_EPOCHS_PER_ROUND,
            lim_sv=10,
            lr=LR,
            bc_cohort_name=BC_COHORT,
            dagger_only=False,
            freeze_vision=True,
        )
        model_path = dst_model

        # Save per-round model weights
        round_model = os.path.join(model_dir, f"model_round_{r}.pth")
        shutil.copy2(dst_model, round_model)
        print(f"  Model saved -> {model_path}  (copy: {round_model})")

        # ── Evaluate retrained model ──
        print(f"\n  Evaluating retrained model...")
        _, eval_results, eval_diag, eval_sr, eval_cr = \
            collect_and_evaluate(pilot, model_path, selected_branches,
                                 scene_data, round_i=r, output_dir=out_dir,
                                 is_dagger=False)
        print(f"\n  Round {r} eval: {eval_sr:.0%} success, {eval_cr:.0%} collision")

        # Save eval checkpoint
        _save_checkpoint(r, "evaluated", {
            "results": eval_results,
            "diagnostics": eval_diag,
            "overall_sr": eval_sr,
            "overall_cr": eval_cr,
        })

        round_results.append({
            "round": r,
            "n_new_annotations": len(new_anns),
            "n_total_annotations": len(all_annotations),
            "collection": {"sr": coll_ckpt["overall_sr"] if coll_ckpt else coll_sr,
                           "cr": coll_ckpt["overall_cr"] if coll_ckpt else coll_cr},
            "eval": {"sr": eval_sr, "cr": eval_cr},
            "eval_per_object": eval_results,
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"[SUMMARY] HW1 DAgger -- {N_ROUNDS} rounds on 15 branches")
    print(f"{'='*70}")
    header = f"  {'Round':8} {'Coll SR':>8} {'Coll CR':>8} {'Eval SR':>8} {'Eval CR':>8} {'Annotations':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rr in round_results:
        print(f"  R{rr['round']:<7} {rr['collection']['sr']:>7.0%}"
              f" {rr['collection']['cr']:>7.0%}"
              f" {rr['eval']['sr']:>7.0%}"
              f" {rr['eval']['cr']:>7.0%}"
              f" {rr['n_total_annotations']:>12}")

    print(f"\n  Per-object eval success rates:")
    obj_names = [q for q in queries if q in SELECTED]
    print(f"  {'Object':30} " + " ".join(f"{'R'+str(rr['round']):>6}" for rr in round_results))
    for obj_name in obj_names:
        cells = []
        for rr in round_results:
            sr = rr["eval_per_object"].get(obj_name, {}).get("success_rate", float("nan"))
            cells.append(f"{sr:>5.0%}")
        print(f"  {obj_name[:30]:30} " + " ".join(cells))
    print(f"{'='*70}")

    # Save results
    summary = {
        "timestamp": ts,
        "config": {
            "n_rounds": N_ROUNDS, "n_epochs_per_round": N_EPOCHS_PER_ROUND,
            "lr": LR, "batch_size": BATCH_SIZE, "cohort": COHORT,
        },
        "selected_branches": {k: {"fail": v["fail"], "success": v["success"]}
                              for k, v in SELECTED.items()},
        "rounds": round_results,
    }
    out_path = os.path.join(out_dir, "hw1_dagger_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    print(f"Checkpoints in {CKPT_DIR}")


if __name__ == "__main__":
    main()
