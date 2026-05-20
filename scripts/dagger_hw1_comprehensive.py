#!/usr/bin/env python3
"""
Comprehensive HW1-style DAgger using ALL TRAINING branches per object.

IMPORTANT: Only uses training branches (trajectories[0-9]*.pt), NOT validation
branches (trajectories_val*.pt). Per scene config: 110 branches/object for
clock and leafblower, 30 for boxes = 250 training branches total.

Per round:
  Round 1: ALL training branches (to establish initial failure set).
  Round 2+: ALL prior failure branches + MAX_SUCCESS_PER_OBJ random successes.
  Expert relabels at learner-visited states (beta=0 = pure learner rollout).
  Retrain from BC weights each round on BC + aggregated DAgger data.

Evaluation: each round is evaluated on ALL training branches (full coverage).

Crash-resilient: per-round checkpoints, resumes on restart.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=1 \
    conda run --no-capture-output -n FiGS python -u scripts/dagger_hw1_comprehensive.py
"""
import os, sys, time, json, math, random, glob
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
    _retrain_commander, _SCENE_CACHE, _BC_TRAJ_CACHE, _PKL_CACHE,
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

N_ROUNDS = 10
N_EPOCHS_PER_ROUND = 30
LR = 2e-5
BATCH_SIZE = 64
COHORT = "SSV_DAGGER_HW1_COMPREHENSIVE"

# Per round after round 1: use ALL failure branches + up to this many
# randomly sampled success branches per object.
MAX_SUCCESS_PER_OBJ = 5

# Seed for reproducibility
SEED = 42

# Checkpoint directory
CKPT_DIR = os.path.join(WORKSPACE, "cohorts", COHORT, "checkpoints")


# ──────────────────────────────────────────────────────────────────────────────
# Training-only branch loader (excludes validation trajectories)
# ──────────────────────────────────────────────────────────────────────────────
def _load_training_branches_only(bc_cohort_name, scene_name, scenes_cfg_dir):
    """
    Load ONLY training BC trajectories (trajectories[0-9]*.pt, NOT _val*).
    Returns {obj_name: {branch_id: tXUi_array}}.
    """
    workspace = str(Path(__file__).resolve().parents[1])
    scene_data = _SCENE_CACHE.get(scene_name)
    if scene_data is None:
        raise RuntimeError(f"Scene '{scene_name}' not loaded")

    queries = scene_data["queries"]
    obj_targets = scene_data["obj_targets"]
    obj_locs = np.array([np.squeeze(t) for t in obj_targets])

    rollout_dir = os.path.join(
        workspace, "cohorts", bc_cohort_name, "rollout_data", scene_name)

    # ONLY training files: trajectories00000.pt, trajectories00001.pt, ...
    # EXCLUDE: trajectories_val00000.pt, etc.
    train_files = sorted(glob.glob(os.path.join(rollout_dir, "trajectories[0-9]*.pt")))

    per_obj = {i: [] for i in range(len(queries))}
    per_obj_files = {i: [] for i in range(len(queries))}

    for tf in train_files:
        data = torch.load(tf, map_location="cpu", weights_only=False)
        tXUd = data.get("tXUd")
        if tXUd is None or not hasattr(tXUd, "shape") or tXUd.shape[0] != 18:
            continue
        goal = np.array(tXUd[1:4, -1])
        dists = np.linalg.norm(obj_locs - goal, axis=1)
        best = int(np.argmin(dists))
        if float(dists[best]) < 5.0:
            per_obj[best].append(tXUd)
            per_obj_files[best].append(os.path.basename(tf))

    result = {}
    for obj_idx, obj_name in enumerate(queries):
        trajs = per_obj[obj_idx]
        if not trajs:
            continue
        result[obj_name] = {i: tr for i, tr in enumerate(trajs)}
        files_str = per_obj_files[obj_idx]
        print(f"  [Train-Only] '{obj_name}': {len(trajs)} training branches"
              f"  (files: {files_str[0]}..{files_str[-1]})")

        # Also populate caches so downstream helpers work
        key = f"{scene_name}_{obj_name}"
        _BC_TRAJ_CACHE[key] = trajs
        if key not in _PKL_CACHE:
            _PKL_CACHE[key] = {"tXUi": trajs[0], "obj_loc": trajs[0][1:4, -1]}

    return result


# ──────────────────────────────────────────────────────────────────────────────
# HW1-style DAgger Policy
# ──────────────────────────────────────────────────────────────────────────────
class DAggerPolicy:
    """Pure learner rollout + expert relabeling."""
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


def select_branches_for_round(all_branches_dict, prior_failures, rng, round_i):
    """
    Round 1: ALL training branches.
    Round 2+: ALL prior failure branches + MAX_SUCCESS_PER_OBJ random successes.
    """
    selected = {}
    for obj_name, branches in all_branches_dict.items():
        all_ids = list(branches.keys())
        if round_i == 1:
            selected[obj_name] = [(bid, branches[bid]) for bid in all_ids]
        else:
            fail_ids = prior_failures.get(obj_name, [])
            success_ids = [bid for bid in all_ids if bid not in fail_ids]
            n_success = min(MAX_SUCCESS_PER_OBJ, len(success_ids))
            sampled_success = rng.sample(success_ids, n_success) if success_ids else []
            round_ids = sorted(set(fail_ids + sampled_success))
            selected[obj_name] = [(bid, branches[bid]) for bid in round_ids]

        n_fail = len(prior_failures.get(obj_name, []))
        n_succ = len(selected[obj_name]) - n_fail if round_i > 1 else 0
        print(f"  '{obj_name[:30]}': {len(selected[obj_name])} branches for R{round_i}"
              f"  (from {len(all_ids)} total"
              + (f", {n_fail} fail + {n_succ} success" if round_i > 1 else "")
              + ")")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ckpt_path(round_i, step):
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
    """Find the last completed round. Returns (last_completed_round, all_prior_annotations)."""
    if not os.path.isdir(CKPT_DIR):
        return 0, []

    all_annotations = []
    last_completed = 0

    for r in range(1, N_ROUNDS + 1):
        coll_data = _load_checkpoint(r, "collected")
        if coll_data is None:
            break
        all_annotations.extend(coll_data["annotations"])

        eval_data = _load_checkpoint(r, "evaluated")
        if eval_data is None:
            print(f"  [resume] Round {r}: collection done, retrain/eval pending")
            return r - 1, all_annotations
        last_completed = r

    return last_completed, all_annotations


def get_failures_from_checkpoint(round_i):
    """Extract failure branch IDs per object from a round's eval diagnostics."""
    eval_data = _load_checkpoint(round_i, "evaluated")
    if eval_data is None:
        coll_data = _load_checkpoint(round_i, "collected")
        if coll_data is None:
            return {}
        diag = coll_data.get("diagnostics", [])
    else:
        diag = eval_data.get("diagnostics", [])

    failures = {}
    for d in diag:
        if not d["success"]:
            obj = d["object"]
            if obj not in failures:
                failures[obj] = []
            failures[obj].append(d["branch_id"])
    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Collection + evaluation (with detailed per-trajectory logging)
# ──────────────────────────────────────────────────────────────────────────────
def collect_and_evaluate(pilot, model_path, branches_dict, scene_data, round_i,
                         output_dir=None, is_dagger=True):
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
    label = f"R{round_i}" if is_dagger else f"Eval_R{round_i}"
    mode = "DAgger" if is_dagger else "Eval"

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        branches = branches_dict[obj_name]
        successes, collisions, goal_dists = [], [], []
        obj_runs_for_plot = []

        obj_short = _short(obj_name)
        n_branches = len(branches)
        print(f"\n  [{mode}] === {obj_name} ({n_branches} branches) ===")

        for bi, (br_id, tXUi) in enumerate(branches):
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
            Tro, Xro, Uro, Iro = result[0], result[1], result[2], result[3]
            sim_s = time.time() - _t

            ev = _evaluate_run(Xro, obj_target, epcds_arr,
                               env_min=scene_data.get("env_min"),
                               env_max=scene_data.get("env_max"),
                               tXUi=tXUi, idx0=0)

            s = float(ev["success"])
            c = float(ev["collision"])
            gd = float(ev["goal_dist"])
            min_gd = float(ev["min_goal_dist"])
            successes.append(s); collisions.append(c); goal_dists.append(gd)
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            status = "OK" if s else ("COLL" if c else "MISS")
            reason = term_info.get("reason", "timeout")
            fov = ev.get("fov_pct", float("nan"))
            n_steps = Xro.shape[1] - 1

            # Detailed per-trajectory log line
            traj_id = f"{obj_short}/br{br_id:03d}"
            ann_str = ""
            if is_dagger:
                n_ann = len(dagger_pol.annotations)
                all_annotations.extend(dagger_pol.annotations)
                ann_str = f"  ann={n_ann:4d}"

            fov_str = f"{fov:.0%}" if not math.isnan(fov) else "N/A"
            reason_str = reason if reason is not None else "None"
            print(f"  [{label}] {bi+1:3d}/{n_branches}  {status:4s}  {traj_id}"
                  f"  d0={d0:5.1f}m  goal={gd:5.2f}m  min={min_gd:5.2f}m"
                  f"  fov={fov_str:>4s}  steps={n_steps:3d}"
                  f"  stop={reason_str:<12s}{ann_str}  ({sim_s:.0f}s)")

            all_diag.append({
                "object": obj_name, "branch_id": br_id, "traj_id": traj_id,
                "d0": d0, "success": bool(ev["success"]),
                "collision": bool(ev["collision"]),
                "goal_dist": gd, "min_goal_dist": min_gd,
                "fov_pct": float(fov) if not math.isnan(fov) else None,
                "stop_reason": reason, "n_steps": n_steps,
                "sim_time_s": round(sim_s, 1),
            })
            if output_dir:
                obj_runs_for_plot.append((Xro.copy(), ev, tXUi.copy()))

        sr = float(np.mean(successes)) if successes else 0.0
        cr = float(np.mean(collisions)) if collisions else 0.0
        gd_m = float(np.mean(goal_dists)) if goal_dists else float("nan")
        n_ok = int(sum(successes))
        n_fail = len(successes) - n_ok
        n_coll = int(sum(collisions))
        print(f"  [{mode}] --- '{obj_name[:30]}'  success={sr:.0%} ({n_ok}/{len(successes)})"
              f"  collision={cr:.0%} ({n_coll})  miss={n_fail - n_coll}"
              f"  avg_goal={gd_m:.2f}m")
        per_object[obj_name] = {
            "success_rate": sr, "collision_rate": cr, "goal_dist": gd_m,
            "n_eval": len(branches), "n_success": n_ok, "n_collision": n_coll,
        }

        # Plotly 3D trajectory plot per object per round
        if output_dir and obj_runs_for_plot:
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            html_path = os.path.join(plots_dir, f"{label}_{obj_short}.html")
            try:
                _save_benchmark_plotly(
                    obj_runs=obj_runs_for_plot, obj_target=obj_target,
                    simulator=simulator,
                    obj_name=f"{label} -- {obj_name} ({sr:.0%} success, {n_ok}/{len(successes)})",
                    save_path=html_path,
                )
                print(f"  [plotly] -> {html_path}")
            except Exception as e:
                print(f"  [plotly] WARNING: {e}")

    overall_sr = float(np.mean([v["success_rate"] for v in per_object.values()]))
    overall_cr = float(np.mean([v["collision_rate"] for v in per_object.values()]))

    # Summary table
    print(f"\n  [{mode}] ========== Round {round_i} Summary ==========")
    print(f"  {'Object':35} {'SR':>6} {'CR':>6} {'Goal':>7} {'N':>4}")
    print(f"  {'-'*60}")
    for obj_name, stats in per_object.items():
        print(f"  {obj_name[:35]:35} {stats['success_rate']:>5.0%}"
              f" {stats['collision_rate']:>5.0%}"
              f" {stats['goal_dist']:>6.2f}m {stats['n_eval']:>4}")
    print(f"  {'-'*60}")
    print(f"  {'OVERALL':35} {overall_sr:>5.0%} {overall_cr:>5.0%}")
    print(f"  {'='*60}")

    return all_annotations, per_object, all_diag, overall_sr, overall_cr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "visualizations", f"hw1_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("[HW1 DAgger COMPREHENSIVE — training branches only]")
    print(f"  BC model         : {BC_MODEL_PATH}")
    print(f"  Rounds           : {N_ROUNDS}")
    print(f"  Epochs/round     : {N_EPOCHS_PER_ROUND}")
    print(f"  LR               : {LR}")
    print(f"  Max success/obj  : {MAX_SUCCESS_PER_OBJ}")
    print(f"  Cohort           : {COHORT}")
    print(f"  Checkpoints      : {CKPT_DIR}")
    print(f"  Output           : {out_dir}")
    print("=" * 70)

    # ── Phase 0: Load scene + TRAINING-ONLY branches ──
    print("\n[Phase 0] Loading scene + TRAINING-ONLY BC trajectories...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    all_branch_data = _load_training_branches_only(BC_COHORT, SCENE, SCENES_CFG)

    n_total = sum(len(v) for v in all_branch_data.values())
    print(f"\n  Total training branches: {n_total}")
    for obj, brs in all_branch_data.items():
        print(f"    {obj[:40]}: {len(brs)}")

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

    for r in range(start_round, N_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"[DAgger Round {r}/{N_ROUNDS}]"
              f"  (total annotations so far: {len(all_annotations)})")
        print(f"{'='*70}")

        # ── Select branches for this round ──
        if r == 1:
            prior_failures = {}
        else:
            prior_failures = get_failures_from_checkpoint(r - 1)
            n_fail = sum(len(v) for v in prior_failures.values())
            print(f"  Prior round had {n_fail} failure branches across objects:")
            for obj, fids in prior_failures.items():
                print(f"    {obj[:30]}: {len(fids)} failures  ids={fids[:10]}{'...' if len(fids)>10 else ''}")

        selected_branches = select_branches_for_round(
            all_branch_data, prior_failures, rng, r)

        n_round_branches = sum(len(v) for v in selected_branches.values())
        print(f"  Total branches this round: {n_round_branches}")

        # ── Collect (= run learner + expert relabel) ──
        coll_ckpt = _load_checkpoint(r, "collected")
        if coll_ckpt is not None:
            new_anns = coll_ckpt["annotations"]
            coll_sr = coll_ckpt["overall_sr"]
            coll_cr = coll_ckpt["overall_cr"]
            coll_results = coll_ckpt.get("results", {})
            print(f"  [resume] Round {r} collection loaded from checkpoint"
                  f" ({len(new_anns)} annotations, {coll_sr:.0%} SR)")
        else:
            print(f"\n  Collecting DAgger data ({n_round_branches} branches)...")
            new_anns, coll_results, coll_diag, coll_sr, coll_cr = \
                collect_and_evaluate(pilot, model_path, selected_branches,
                                     scene_data, round_i=r, output_dir=out_dir,
                                     is_dagger=True)
            print(f"\n  Round {r} collection: {len(new_anns)} annotations"
                  f"  |  {coll_sr:.0%} success, {coll_cr:.0%} collision")

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
        print(f"  [retrain] {len(all_annotations)} annotation samples")

        agg_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "dagger_data", PILOT_NAME)
        os.makedirs(agg_dir, exist_ok=True)
        agg_file = os.path.join(agg_dir, "aggregated_annotations.pt")
        torch.save(all_annotations, agg_file)

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

        round_model = os.path.join(model_dir, f"model_round_{r}.pth")
        shutil.copy2(dst_model, round_model)
        print(f"  Model saved -> {model_path}  (copy: {round_model})")

        # ── Evaluate retrained model on ALL training branches ──
        print(f"\n  Evaluating retrained model on ALL {n_total} training branches...")
        eval_branches = {obj: [(bid, br) for bid, br in brs.items()]
                         for obj, brs in all_branch_data.items()}
        _, eval_results, eval_diag, eval_sr, eval_cr = \
            collect_and_evaluate(pilot, model_path, eval_branches,
                                 scene_data, round_i=r, output_dir=out_dir,
                                 is_dagger=False)

        _save_checkpoint(r, "evaluated", {
            "results": eval_results,
            "diagnostics": eval_diag,
            "overall_sr": eval_sr,
            "overall_cr": eval_cr,
        })

        round_results.append({
            "round": r,
            "n_collect_branches": n_round_branches,
            "n_new_annotations": len(new_anns),
            "n_total_annotations": len(all_annotations),
            "collection": {"sr": coll_sr, "cr": coll_cr},
            "eval": {"sr": eval_sr, "cr": eval_cr},
            "eval_per_object": eval_results,
        })

        # ── Per-round summary table ──
        print(f"\n{'='*70}")
        print(f"[Progress] After Round {r}/{N_ROUNDS}")
        print(f"{'='*70}")
        print(f"  {'Round':8} {'Branches':>9} {'Coll SR':>8} {'Coll CR':>8}"
              f" {'Eval SR':>8} {'Eval CR':>8} {'Annotations':>12}")
        print(f"  {'-'*67}")
        for rr in round_results:
            print(f"  R{rr['round']:<7} {rr['n_collect_branches']:>9}"
                  f" {rr['collection']['sr']:>7.0%}"
                  f" {rr['collection']['cr']:>7.0%}"
                  f" {rr['eval']['sr']:>7.0%}"
                  f" {rr['eval']['cr']:>7.0%}"
                  f" {rr['n_total_annotations']:>12}")

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"[FINAL SUMMARY] HW1 DAgger COMPREHENSIVE — {N_ROUNDS} rounds")
    print(f"{'='*70}")
    print(f"  {'Round':8} {'Branches':>9} {'Coll SR':>8} {'Coll CR':>8}"
          f" {'Eval SR':>8} {'Eval CR':>8} {'Annotations':>12}")
    print(f"  {'-'*67}")
    for rr in round_results:
        print(f"  R{rr['round']:<7} {rr['n_collect_branches']:>9}"
              f" {rr['collection']['sr']:>7.0%}"
              f" {rr['collection']['cr']:>7.0%}"
              f" {rr['eval']['sr']:>7.0%}"
              f" {rr['eval']['cr']:>7.0%}"
              f" {rr['n_total_annotations']:>12}")

    print(f"\n  Per-object eval success rates (all {n_total} training branches):")
    obj_names = list(all_branch_data.keys())
    header_rounds = " ".join(f"{'R'+str(rr['round']):>6}" for rr in round_results)
    print(f"  {'Object':35} {header_rounds}")
    for obj_name in obj_names:
        cells = []
        for rr in round_results:
            sr = rr["eval_per_object"].get(obj_name, {}).get("success_rate", float("nan"))
            cells.append(f"{sr:>5.0%}")
        print(f"  {obj_name[:35]:35} " + " ".join(cells))

    # Per-object collision rates
    print(f"\n  Per-object collision rates:")
    print(f"  {'Object':35} {header_rounds}")
    for obj_name in obj_names:
        cells = []
        for rr in round_results:
            cr = rr["eval_per_object"].get(obj_name, {}).get("collision_rate", float("nan"))
            cells.append(f"{cr:>5.0%}")
        print(f"  {obj_name[:35]:35} " + " ".join(cells))
    print(f"{'='*70}")

    # Save results
    summary = {
        "timestamp": ts,
        "config": {
            "n_rounds": N_ROUNDS, "n_epochs_per_round": N_EPOCHS_PER_ROUND,
            "lr": LR, "batch_size": BATCH_SIZE, "cohort": COHORT,
            "max_success_per_obj": MAX_SUCCESS_PER_OBJ, "seed": SEED,
            "n_total_training_branches": n_total,
        },
        "per_object_branch_counts": {obj: len(brs) for obj, brs in all_branch_data.items()},
        "rounds": round_results,
    }
    out_path = os.path.join(out_dir, "hw1_comprehensive_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    print(f"Checkpoints in {CKPT_DIR}")
    print(f"Plotly plots in {os.path.join(out_dir, 'plots')}")


if __name__ == "__main__":
    main()
