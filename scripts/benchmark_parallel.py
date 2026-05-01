#!/usr/bin/env python3
"""
Parallel canonical benchmark: run 50 sims at a time using simulate_parallel().
Compares wall-clock time and results with sequential benchmark.

Usage:
    cd /data/erwinpi/SINGER
    ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n FiGS python -u scripts/benchmark_parallel.py
"""
import os, sys, time, json, copy, threading, subprocess
import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

from sousvide.instruct.train_dagger import (
    _get_scene, _preload_all_pkls, _preload_bc_trajectories,
    _load_all_branches, _make_terminal_fn, _evaluate_run,
    _swap_model, _BC_TRAJ_CACHE, _get_pkl, Pilot, DEVICE,
)

SCENE = "flightroom_ssv_exp"
SCENES_CFG = os.path.join(WORKSPACE, "configs", "scenes")
BC_COHORT = "ssv_BC_CENTROID_V9"
DAGGER_COHORT = "SSV_DAGGER_CENTROID_V9"
PILOT_NAME = "InstinctJester"
SEED = 42
MAX_BRANCHES = 50

# ── GPU Monitor ──────────────────────────────────────────────────────────
gpu_samples = []
_stop_monitor = threading.Event()

def _gpu_monitor_thread(interval=1.0):
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    while not _stop_monitor.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits", f"--id={gpu_id}"],
                text=True
            ).strip()
            parts = [x.strip() for x in out.split(",")]
            gpu_samples.append({
                "t": time.time(),
                "mem_mb": float(parts[0]),
                "mem_total_mb": float(parts[1]),
                "power_w": float(parts[2]),
                "util_pct": float(parts[3]),
            })
        except Exception:
            pass
        _stop_monitor.wait(interval)


def main():
    print("=" * 70)
    print("PARALLEL BENCHMARK — 50 sims at a time")
    print("=" * 70)

    # Start GPU monitor
    monitor = threading.Thread(target=_gpu_monitor_thread, daemon=True)
    monitor.start()

    # ── Load scene ────────────────────────────────────────────────────
    print("\nLoading scene...")
    flights = [[SCENE, None]]
    _get_scene(SCENE, SCENES_CFG)
    _preload_all_pkls(flights, SCENES_CFG)
    if not _BC_TRAJ_CACHE:
        _preload_bc_trajectories(BC_COHORT, flights, SCENES_CFG)

    scene_data = _get_scene(SCENE, SCENES_CFG)
    simulator = scene_data["simulator"]
    obj_targets = scene_data["obj_targets"]
    queries = scene_data["queries"]
    pc_arr = scene_data.get("epcds_arr", np.zeros((0, 3)))

    from scipy.spatial import cKDTree
    pc_tree = cKDTree(pc_arr) if pc_arr.shape[0] > 0 else None

    # ── Sample branches (same seed as sequential benchmark) ──────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cohort_path = os.path.join(WORKSPACE, "cohorts", BC_COHORT)
    branches_per_obj = {}
    branch_idxs_per_obj = {}

    for obj_name in queries:
        pkl_data = _get_pkl(SCENE, obj_name, SCENES_CFG)
        if pkl_data is None:
            continue
        tXUi_default = pkl_data["tXUi"]
        all_branches = _load_all_branches(SCENE, obj_name, cohort_path, SCENES_CFG, tXUi_default)
        n_sample = min(MAX_BRANCHES, len(all_branches))
        idxs = np.random.choice(len(all_branches), size=n_sample,
                                replace=(n_sample < MAX_BRANCHES and len(all_branches) > 0))
        branches_per_obj[obj_name] = [all_branches[i] for i in idxs]
        branch_idxs_per_obj[obj_name] = idxs
        print(f"  '{obj_name}': {n_sample} branches from {len(all_branches)}")

    # ── Models to benchmark ──────────────────────────────────────────
    models = [
        {"label": "BC_Centroid_V9", "cohort": BC_COHORT, "pilot_name": PILOT_NAME,
         "model_path": os.path.join(WORKSPACE, "cohorts", BC_COHORT, "roster", PILOT_NAME, "model.pth")},
        {"label": "DAgger_Centroid_V9", "cohort": DAGGER_COHORT, "pilot_name": PILOT_NAME,
         "model_path": os.path.join(WORKSPACE, "cohorts", DAGGER_COHORT, "roster", PILOT_NAME, "model.pth")},
    ]

    all_results = {}
    total_t0 = time.time()

    for model_cfg in models:
        label = model_cfg["label"]
        print(f"\n{'='*70}")
        print(f"[Parallel] {label}")
        print(f"{'='*70}")

        # Create base pilot
        pilot_base = Pilot(model_cfg["cohort"], model_cfg["pilot_name"])
        pilot_base.set_mode("deploy")
        pilot_base.model.to(DEVICE)
        pilot_base = _swap_model(pilot_base, model_cfg["model_path"])

        label_results = {}

        for obj_idx, obj_name in enumerate(queries):
            if obj_name not in branches_per_obj:
                continue

            obj_target = obj_targets[obj_idx] if obj_idx < len(obj_targets) else np.zeros(3)
            branches = branches_per_obj[obj_name]
            br_idxs = branch_idxs_per_obj[obj_name]
            n_runs = len(branches)

            # Build terminal function (shared, stateless)
            terminal_fn, _ = _make_terminal_fn(
                obj_target, pc_tree,
                env_min=scene_data.get("env_min"),
                env_max=scene_data.get("env_max"),
            )

            # Create N pilot copies with reset state
            pilots = []
            for _ in range(n_runs):
                p = copy.deepcopy(pilot_base)
                p.hy_flag = False
                p.hy_idx = 0
                p.DxU.zero_()
                if hasattr(p, 'Znn'):
                    p.Znn.zero_()
                if hasattr(p, 'chunk_buf'):
                    p.chunk_buf = None
                    p.chunk_step = 0
                pilots.append(p)

            # Build sim configs
            configs = []
            for run_i, tXUi in enumerate(branches):
                x0 = tXUi[1:11, 0].copy()
                t_start = float(tXUi[0, 0])
                t_end = float(tXUi[0, -1])
                configs.append({
                    "policy": pilots[run_i],
                    "t0": t_start,
                    "tf": t_end,
                    "x0": x0,
                    "obj": np.zeros((18, 1)),
                    "query": obj_name,
                    "early_stop_fn": terminal_fn,
                })

            # Run parallel batch
            print(f"\n  Running {n_runs} parallel sims for '{obj_name}'...")
            batch_t0 = time.time()
            results = simulator.simulate_parallel(configs, verbose=True)
            batch_elapsed = time.time() - batch_t0
            print(f"  Batch completed in {batch_elapsed:.1f}s  ({batch_elapsed/n_runs:.1f}s/sim effective)")

            # Evaluate each result
            goal_dists, successes, collisions = [], [], []
            for run_i, (Tro, Xro, Uro, Iro, Tsol, Adv) in enumerate(results):
                tXUi = branches[run_i]
                br_idx = int(br_idxs[run_i])
                goal_dist = float(np.linalg.norm(Xro[:3, -1] - obj_target))

                ev = _evaluate_run(Xro, obj_target, pc_arr,
                                   env_min=scene_data.get("env_min"),
                                   env_max=scene_data.get("env_max"),
                                   tXUi=tXUi, idx0=0)
                success = ev["success"]
                collided = ev["collision"]
                init_d = float(np.linalg.norm(Xro[:3, 0] - np.asarray(obj_target).flatten()[:3]))

                goal_dists.append(goal_dist)
                successes.append(success)
                collisions.append(collided)

                status = "ok" if success else ("COLL" if collided else "miss")
                print(f"    run {run_i+1:2d}/{n_runs}  br={br_idx:3d}  d0={init_d:.1f}m  "
                      f"goal={goal_dist:.2f}m  steps={ev['n_steps']}  {status}")

            sr = float(np.mean(successes))
            cr = float(np.mean(collisions))
            gd = float(np.mean(goal_dists))
            print(f"  [{label}] '{obj_name}':  success={sr:.0%}  collision={cr:.0%}  goal={gd:.2f}m")

            label_results[obj_name] = {
                "goal_dist": gd, "success_rate": sr, "collision_rate": cr, "n_eval": n_runs,
            }

            # Cleanup pilot copies
            del pilots
            torch.cuda.empty_cache()

        # Overall
        all_sr = [v["success_rate"] for v in label_results.values()]
        all_cr = [v["collision_rate"] for v in label_results.values()]
        all_gd = [v["goal_dist"] for v in label_results.values()]
        label_results["__overall__"] = {
            "goal_dist": float(np.mean(all_gd)),
            "success_rate": float(np.mean(all_sr)),
            "collision_rate": float(np.mean(all_cr)),
        }
        all_results[label] = label_results

        del pilot_base
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0
    _stop_monitor.set()

    # ── Load sequential results for comparison ───────────────────────
    seq_path = os.path.join(
        WORKSPACE, "cohorts", DAGGER_COHORT, "visualizations",
        "canonical_benchmark_results.json")
    seq_results = {}
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            seq_results = json.load(f)

    # ── Print comparison ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: Sequential vs Parallel")
    print("=" * 70)
    print(f"\n  Parallel total wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  GPU samples collected: {len(gpu_samples)}")

    if gpu_samples:
        mem_vals = [s["mem_mb"] for s in gpu_samples]
        pwr_vals = [s["power_w"] for s in gpu_samples]
        util_vals = [s["util_pct"] for s in gpu_samples]
        print(f"  GPU memory:  avg={np.mean(mem_vals):.0f}MB  max={max(mem_vals):.0f}MB")
        print(f"  GPU power:   avg={np.mean(pwr_vals):.0f}W  max={max(pwr_vals):.0f}W")
        print(f"  GPU util:    avg={np.mean(util_vals):.0f}%  max={max(util_vals):.0f}%")

    for label in all_results:
        print(f"\n  --- {label} ---")
        print(f"  {'Object':<35}  {'Parallel SR':>12}  {'Sequential SR':>14}  {'Par CR':>8}  {'Seq CR':>8}")
        for obj_name in all_results[label]:
            if obj_name == "__overall__":
                continue
            par = all_results[label][obj_name]
            seq = seq_results.get(label, {}).get(obj_name, {})
            seq_sr = seq.get("success_rate", float('nan'))
            seq_cr = seq.get("collision_rate", float('nan'))
            print(f"  {obj_name[:35]:<35}  {par['success_rate']:>11.0%}  {seq_sr:>13.0%}"
                  f"  {par['collision_rate']:>7.0%}  {seq_cr:>7.0%}")

        par_o = all_results[label].get("__overall__", {})
        seq_o = seq_results.get(label, {}).get("__overall__", {})
        print(f"  {'OVERALL':<35}  {par_o.get('success_rate',0):>11.0%}  "
              f"{seq_o.get('success_rate', float('nan')):>13.0%}"
              f"  {par_o.get('collision_rate',0):>7.0%}  "
              f"{seq_o.get('collision_rate', float('nan')):>7.0%}")

    # ── Save results ─────────────────────────────────────────────────
    out_path = os.path.join(WORKSPACE, "parallel_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": all_results,
            "wall_clock_s": total_elapsed,
            "gpu_stats": {
                "n_samples": len(gpu_samples),
                "mem_avg_mb": float(np.mean([s["mem_mb"] for s in gpu_samples])) if gpu_samples else 0,
                "mem_max_mb": float(max(s["mem_mb"] for s in gpu_samples)) if gpu_samples else 0,
                "power_avg_w": float(np.mean([s["power_w"] for s in gpu_samples])) if gpu_samples else 0,
                "power_max_w": float(max(s["power_w"] for s in gpu_samples)) if gpu_samples else 0,
                "util_avg_pct": float(np.mean([s["util_pct"] for s in gpu_samples])) if gpu_samples else 0,
                "util_max_pct": float(max(s["util_pct"] for s in gpu_samples)) if gpu_samples else 0,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
