#!/usr/bin/env python3
"""
HW1-style DAgger on 15 TRAINING branches (5/object: 3 success + 2 worst fail).

Phase 1: Verify seed=42 branch mapping (6 sims: 1 fail + 1 success per object)
Phase 2: DAgger rounds on the 15 selected branches

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

# Selected branches: actual_branch_id from all_branches[]
# 3 success + 2 worst collision per object
SELECTED = {
    "green clock":        {"fail": [5, 115], "success": [47, 91, 64]},
    "green and pink leafblower": {"fail": [1, 111], "success": [37, 15, 81]},
    "yellow handheld cordless drill on two boxes": {"fail": [2, 27], "success": [10, 35, 17]},
}

# Expected BC outcomes from results.json (for verification)
VERIFY = {
    "green clock":        {"fail_branch": 5,  "fail_goal": 6.17, "success_branch": 47, "success_goal": 2.89},
    "green and pink leafblower": {"fail_branch": 1, "fail_goal": 2.92, "success_branch": 37, "success_goal": 2.15},
    "yellow handheld cordless drill on two boxes": {"fail_branch": 2, "fail_goal": 1.45, "success_branch": 10, "success_goal": 1.69},
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


def evaluate_branches(pilot, model_path, branches_dict, scene_data, label,
                      output_dir=None):
    """Evaluate pilot on selected branches. Returns per-object results + diagnostics."""
    from scipy.spatial import cKDTree
    pilot = _swap_model(pilot, model_path)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None

    per_object = {}
    all_diag = []

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        branches = branches_dict[obj_name]  # list of (branch_id, tXUi)
        successes, collisions, goal_dists = [], [], []
        obj_runs_for_plot = []

        for br_id, tXUi in branches:
            reset_pilot(pilot)
            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )
            _t = time.time()
            result = run_sim(simulator, pilot, tXUi, obj_name,
                             early_stop_fn=terminal_fn)
            Xro = result[1]
            ev = _evaluate_run(Xro, obj_target, epcds_arr,
                               env_min=scene_data.get("env_min"),
                               env_max=scene_data.get("env_max"),
                               tXUi=tXUi, idx0=0)
            s, c, gd = float(ev["success"]), float(ev["collision"]), float(ev["goal_dist"])
            successes.append(s); collisions.append(c); goal_dists.append(gd)
            d0 = float(np.linalg.norm(tXUi[1:4, 0] - np.squeeze(obj_target)))
            sim_s = time.time() - _t
            status = "OK" if s else ("COLL" if c else "MISS")
            dev = ev.get("mean_pos_dev", float("nan"))
            fov = ev.get("fov_pct", float("nan"))

            print(f"  [{label}] {status:4s}  '{_short(obj_name)}'  branch={br_id}"
                  f"  d0={d0:.1f}m  goal={gd:.2f}m  min={ev['min_goal_dist']:.2f}m"
                  f"  dev={dev:.2f}m  fov={fov:.0%}  ({sim_s:.0f}s)")

            all_diag.append({
                "object": obj_name, "branch_id": br_id, "d0": d0,
                "success": bool(ev["success"]), "collision": bool(ev["collision"]),
                "goal_dist": gd, "min_goal_dist": float(ev["min_goal_dist"]),
                "mean_pos_dev": float(dev) if not math.isnan(dev) else None,
                "fov_pct": float(fov) if not math.isnan(fov) else None,
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
    return per_object, all_diag, overall_sr, overall_cr


def collect_dagger_data(pilot, model_path, branches_dict, scene_data, round_i):
    """Deploy learner, expert relabels at visited states."""
    from scipy.spatial import cKDTree
    pilot = _swap_model(pilot, model_path)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    epcds_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))
    pc_tree = cKDTree(epcds_arr) if epcds_arr.shape[0] > 0 else None
    all_annotations = []

    for obj_idx, obj_name in enumerate(queries):
        if obj_name not in branches_dict:
            continue
        obj_target = obj_targets[obj_idx]
        for br_id, tXUi in branches_dict[obj_name]:
            reset_pilot(pilot)
            expert = VehicleRateMPC(tXUi, POLICY_NAME, FRAME_NAME, PILOT_NAME)
            dagger_pol = DAggerPolicy(expert, pilot)
            terminal_fn, term_info = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )
            _t = time.time()
            run_sim(simulator, dagger_pol, tXUi, obj_name,
                    early_stop_fn=terminal_fn)
            n_ann = len(dagger_pol.annotations)
            reason = term_info.get("reason", "timeout")
            print(f"  [R{round_i}] '{_short(obj_name)}'  branch={br_id}"
                  f"  {n_ann} annotations  {reason}  ({time.time()-_t:.0f}s)")
            all_annotations.extend(dagger_pol.annotations)

    return all_annotations


def retrain_commander_from_bc(dagger_annotations, round_i):
    """Reload BC weights, train on BC + DAgger. Only CommanderSV."""
    import sousvide.instruct.train_policy_unified as tp

    student = Pilot(COHORT, PILOT_NAME)
    bc_weights = torch.load(BC_MODEL_PATH, map_location="cpu", weights_only=False)
    student.model = bc_weights
    student.set_mode('train')

    Xnn_dag, Ynn_dag = [], []
    default_mfn = np.array([0.3, 0.3], dtype=np.float32)
    for ann in dagger_annotations:
        xnn = ann.get("xnn")
        if not xnn:
            continue
        ynn = {
            "unn": np.array(ann["u"], dtype=np.float32),
            "mfn": default_mfn.copy(),
            "onn": np.array(ann["x"], dtype=np.float32),
        }
        Xnn_dag.append(xnn)
        Ynn_dag.append(ynn)

    print(f"  [retrain R{round_i}] {len(Xnn_dag)} DAgger samples")

    obs_data = {
        "data": [{"Xnn": Xnn_dag, "Ynn": Ynn_dag, "Ndata": len(Xnn_dag),
                  "rollout_id": 0, "course": "dagger",
                  "frame": {"mass": 0.3, "force_normalized": 0.3}}],
        "set": "", "Nobs": len(Xnn_dag), "course": "dagger",
    }
    course_dir = os.path.join(
        WORKSPACE, "cohorts", COHORT, "observation_data", PILOT_NAME, "dagger")
    os.makedirs(course_dir, exist_ok=True)
    torch.save(obs_data, os.path.join(course_dir, "observations_dagger.pt"))

    bc_obs = os.path.join(WORKSPACE, "cohorts", BC_COHORT, "observation_data", PILOT_NAME)
    dag_obs = os.path.join(WORKSPACE, "cohorts", COHORT, "observation_data", PILOT_NAME)
    if os.path.isdir(bc_obs):
        for entry in os.scandir(bc_obs):
            if not entry.is_dir() or entry.name == "dagger":
                continue
            link = os.path.join(dag_obs, entry.name)
            if not os.path.exists(link):
                os.symlink(entry.path, link)

    if hasattr(student.model, 'get_network') and "Commander" in student.model.get_network:
        student.model.get_network["Commander"]["Unlock"] = nn.ModuleList(
            [student.model.network["CommanderSV"]])
        n_vis = sum(p.numel() for p in student.model.network["VisionMLP"].parameters())
        n_cmd = sum(p.numel() for p in student.model.network["CommanderSV"].parameters())
        print(f"  [retrain R{round_i}] VisionMLP frozen ({n_vis:,}) -- CommanderSV ({n_cmd:,}) trained")

    tp.train_student(COHORT, student, "Commander", N_EPOCHS_PER_ROUND,
                     lim_sv=10, lr=LR, batch_size=BATCH_SIZE)

    model_dir = os.path.join(WORKSPACE, "cohorts", COHORT, "roster", PILOT_NAME)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(student.model, model_path)
    print(f"  [retrain R{round_i}] Model saved -> {model_path}")
    return model_path


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
    print(f"  Output      : {out_dir}")
    print("=" * 70)

    # ── Phase 0: Load scene + branches ──
    print("\n[Phase 0] Loading scene + BC trajectories...")
    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    flights = [[SCENE, SCENE]]
    _preload_bc_trajectories(BC_COHORT, flights, SCENES_CFG)

    # Build branch lookup: obj_name -> list of (branch_id, tXUi)
    all_branch_data = {}
    selected_branches = {}
    for obj_idx, obj_name in enumerate(queries):
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

    # ── Phase 1: Verify mapping (1 fail + 1 success per object = 6 sims) ──
    print(f"\n{'='*70}")
    print("[Phase 1] Verifying seed=42 branch mapping (6 sims)...")
    print(f"{'='*70}")

    pilot = Pilot(BC_COHORT, PILOT_NAME)
    pilot.set_mode('deploy')

    verify_branches = {}
    for obj_name in queries:
        if obj_name not in VERIFY:
            continue
        v = VERIFY[obj_name]
        ab = all_branch_data[obj_name]
        verify_branches[obj_name] = [
            (v["fail_branch"], ab[v["fail_branch"]]),
            (v["success_branch"], ab[v["success_branch"]]),
        ]

    v_results, v_diag, _, _ = evaluate_branches(
        pilot, BC_MODEL_PATH, verify_branches, scene_data, label="VERIFY")

    # Check verification
    all_ok = True
    print(f"\n  Verification check:")
    for obj_name, diags in zip(queries, []):
        pass  # handled below
    for d in v_diag:
        obj = d["object"]
        v = VERIFY[obj]
        bid = d["branch_id"]
        if bid == v["fail_branch"]:
            expected_coll = True
            expected_goal = v["fail_goal"]
        else:
            expected_coll = False
            expected_goal = v["success_goal"]

        goal_diff = abs(d["goal_dist"] - expected_goal)
        status_match = d["collision"] == expected_coll if expected_coll else d["success"]
        ok = status_match and goal_diff < 2.0  # allow some tolerance due to stochastic sim

        sym = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"    {sym}  '{_short(obj)}'  branch={bid}"
              f"  expected={'COLL' if expected_coll else 'OK'}(goal~{expected_goal:.1f}m)"
              f"  got={'COLL' if d['collision'] else 'OK'}(goal={d['goal_dist']:.2f}m)"
              f"  diff={goal_diff:.2f}m")

    if not all_ok:
        print("\n  WARNING: Some verifications failed. Proceeding anyway (sim is stochastic).")
    else:
        print("\n  All verifications PASSED. Branch mapping confirmed.")

    # ── Phase 2: BC baseline on all 15 branches ──
    print(f"\n{'='*70}")
    print("[Phase 2] BC baseline on 15 selected branches...")
    print(f"{'='*70}")
    bc_results, bc_diag, bc_sr, bc_cr = evaluate_branches(
        pilot, BC_MODEL_PATH, selected_branches, scene_data, label="BC",
        output_dir=out_dir)
    print(f"\n  BC baseline: {bc_sr:.0%} success, {bc_cr:.0%} collision")

    # ── Phase 3: DAgger rounds ──
    model_path = BC_MODEL_PATH
    all_annotations = []
    round_results = [{"round": 0, "label": "BC", "results": bc_results,
                      "overall_sr": bc_sr, "overall_cr": bc_cr}]

    for r in range(1, N_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"[Phase 3] DAgger Round {r}/{N_ROUNDS}")
        print(f"{'='*70}")

        # Collect
        print(f"\n  Collecting DAgger data...")
        anns = collect_dagger_data(pilot, model_path, selected_branches,
                                   scene_data, r)
        all_annotations.extend(anns)
        print(f"  Round {r}: {len(anns)} new, {len(all_annotations)} total")

        # Retrain
        print(f"\n  Retraining Commander...")
        model_path = retrain_commander_from_bc(all_annotations, r)

        # Evaluate
        print(f"\n  Evaluating after Round {r}...")
        r_res, r_diag, r_sr, r_cr = evaluate_branches(
            pilot, model_path, selected_branches, scene_data, label=f"R{r}",
            output_dir=out_dir)
        print(f"\n  Round {r}: {r_sr:.0%} success, {r_cr:.0%} collision")

        round_results.append({"round": r, "label": f"R{r}", "results": r_res,
                              "overall_sr": r_sr, "overall_cr": r_cr,
                              "n_annotations": len(all_annotations),
                              "diag": r_diag})

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"[SUMMARY] HW1 DAgger -- {N_ROUNDS} rounds on 15 branches")
    print(f"{'='*70}")
    header = f"  {'Round':8} {'Success':>8} {'Collision':>10} {'Annotations':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rr in round_results:
        n_ann = rr.get("n_annotations", 0)
        print(f"  {rr['label']:8} {rr['overall_sr']:>7.0%} {rr['overall_cr']:>9.0%} {n_ann:>12}")

    print(f"\n  Per-object success rates:")
    print(f"  {'Object':30} " + " ".join(f"{'R'+str(rr['round']):>6}" for rr in round_results))
    for obj_name in queries:
        if obj_name not in SELECTED:
            continue
        cells = []
        for rr in round_results:
            sr = rr["results"].get(obj_name, {}).get("success_rate", float("nan"))
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
        "verification": v_diag,
        "rounds": [{
            "round": rr["round"], "label": rr["label"],
            "overall_sr": rr["overall_sr"], "overall_cr": rr["overall_cr"],
            "n_annotations": rr.get("n_annotations", 0),
            "per_object": rr["results"],
        } for rr in round_results],
    }
    out_path = os.path.join(out_dir, "hw1_dagger_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
    print(f"Plotly visualizations in {out_dir}/plots/")


if __name__ == "__main__":
    main()
