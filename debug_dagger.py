"""
debug_dagger.py — Unit tests for DAgger fixes.
Run: conda run -n FiGS python debug_dagger.py
"""
import os, sys, shutil, tempfile, traceback
import numpy as np
import torch

WORKSPACE = "/data/erwinpi/SINGER"
COHORT    = "ssv_dagger_rrt_smoke"
PILOT_NAME = "InstinctJester"

sys.path.insert(0, os.path.join(WORKSPACE, "src"))

PASS_FAIL = []

def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    PASS_FAIL.append((name, ok))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sousvide.instruct.train_dagger import (
        _retrain_commander,
        MixedPolicy,
        _filter_deviation_annotations,
        _swap_model,
        _save_model_checkpoint,
        DEVICE,
    )
    import sousvide.instruct.train_policy as tp
    from sousvide.control.pilot import Pilot
    print(f"[info] Imports OK  device={DEVICE}")
except Exception as e:
    print(f"[FATAL] Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — _retrain_commander lr=1e-5
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Test 1: retrain lr ---")
try:
    # Create a tiny synthetic aggregated file
    tmp_dir = tempfile.mkdtemp()
    agg_file = os.path.join(tmp_dir, "agg.pt")

    # We need a valid xnn: load pilot to get an actual xnn shape
    pilot_ref = Pilot(COHORT, PILOT_NAME)
    pilot_ref.set_mode("deploy")
    pilot_ref.model.to(DEVICE)

    # Build fake xnn matching the model's expected input keys
    # Use a real forward pass with zeros to get the right shape
    dummy_xnn = {}
    # Try to get xnn keys from pilot observe method
    # We'll create plausible dummy tensors based on typical SINGER architecture
    # The safe approach: build fake observations with zeros of shape (1,) per key
    # and let the training loop handle it or skip gracefully.
    # Instead, let's just create minimal BC-format data directly.

    # Minimal annotation: no xnn (will be skipped by retrain)
    # Actually we need valid xnn. Load a real one from existing obs data if available.
    obs_path = os.path.join(WORKSPACE, "cohorts", COHORT,
                            "observation_data", PILOT_NAME)
    sample_xnn = None
    for root, dirs, files in os.walk(obs_path):
        for fn in files:
            if fn.endswith(".pt"):
                try:
                    d = torch.load(os.path.join(root, fn), weights_only=False)
                    if isinstance(d, dict) and "data" in d:
                        for entry in d["data"]:
                            Xnn_list = entry.get("Xnn", [])
                            if Xnn_list:
                                sample_xnn = Xnn_list[0]
                                break
                    if sample_xnn is not None:
                        break
                except Exception:
                    pass
        if sample_xnn is not None:
            break

    if sample_xnn is None:
        report("Test1_retrain_lr", False, "no sample xnn found in obs data — skipping")
    else:
        # Build a small aggregated annotation list (5 samples)
        annotations = []
        for _ in range(5):
            annotations.append({
                "xnn": {k: v.clone() for k, v in sample_xnn.items()},
                "u":   np.random.randn(4).astype(np.float32),
                "x":   np.random.randn(10).astype(np.float32),
                "t":   0.0,
                "query": np.zeros(3),
            })
        torch.save(annotations, agg_file)

        # Patch tp.train_roster to capture lr
        captured_lr = []
        orig_train_roster = tp.train_roster
        def mock_train_roster(cohort_name, roster, mode, Neps, lim_sv, lr=1e-4, batch_size=64):
            captured_lr.append(lr)
            print(f"  [mock] train_roster called: lr={lr}  Neps={Neps}")
            # Call real function with Neps=1 (fast)
            return orig_train_roster(cohort_name, roster, mode, min(Neps, 1), lim_sv=lim_sv, lr=lr, batch_size=64)
        tp.train_roster = mock_train_roster

        _retrain_commander(COHORT, PILOT_NAME, agg_file, Nep=1, lim_sv=1, lr=1e-5)

        tp.train_roster = orig_train_roster

        if captured_lr and abs(captured_lr[0] - 1e-5) < 1e-12:
            report("Test1_retrain_lr", True, f"lr={captured_lr[0]} correctly passed as 1e-5")
        else:
            report("Test1_retrain_lr", False, f"lr was {captured_lr} — expected 1e-5")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    del pilot_ref
    torch.cuda.empty_cache()

except Exception as e:
    report("Test1_retrain_lr", False, str(e))
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — MixedPolicy annotation collection
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Test 2: MixedPolicy annotation collection ---")
try:
    # We test MixedPolicy.control() and reset_annotations()
    # without running a full simulator — just call control() directly with
    # fake state to confirm annotations accumulate with the right keys.
    from sousvide.instruct.expert_controllers import OnlineRRTExpert, PotentialFieldExpert
    import yaml

    scenes_cfg_dir = os.path.join(WORKSPACE, "configs", "scenes")
    scene_name = "flightroom_ssv_exp"
    with open(os.path.join(scenes_cfg_dir, f"{scene_name}.yml")) as f:
        sc = yaml.safe_load(f)
    goal = np.array([1.0, 2.0, 1.5])

    # Minimal expert: PotentialField (no pkl needed)
    expert = PotentialFieldExpert(goal=goal, point_cloud=np.zeros((0, 3)))

    pilot_mp = Pilot(COHORT, PILOT_NAME)
    pilot_mp.set_mode("deploy")
    pilot_mp.model.to(DEVICE)

    mixed = MixedPolicy(expert=expert, pilot=pilot_mp, beta=1.0)

    # Synthetic state: 10-dim drone state [x, y, z, vx, vy, vz, q0, q1, q2, q3]
    # Quaternion (indices 6:10) must be a valid unit quaternion — use identity [0,0,0,1]
    xcr = np.zeros(10, dtype=np.float64)
    xcr[:3] = [5.0, 0.0, 1.0]   # position
    xcr[6:10] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (w=1 in scipy convention x,y,z,w)
    upr = np.zeros(4, dtype=np.float64)
    tcr = 0.0

    # For control() we need icr to be a valid image since pilot.OODA → observe → process_image.
    # Provide a dummy uint8 RGB image of the correct size (640×480 typical).
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    obj_18 = np.zeros((18, 1))  # correct shape for pilot.observe

    # Call control() 3 times
    for step in range(3):
        xcr[0] += 0.1 * step
        tcr = float(step) * 0.1
        u_out, znn, adv, tsol = mixed.control(tcr, xcr, upr, obj_18, dummy_image, None)
        upr = u_out.copy()

    # Check annotations
    ann = mixed.annotations
    keys_ok  = all("xnn" in a and "u" in a and "x" in a and "t" in a for a in ann)
    u_nonzero = not np.allclose(mixed._u_exp_prev, 0.0)

    if len(ann) == 3 and keys_ok:
        report("Test2_annotation_keys",    True,  f"{len(ann)} annotations with correct keys")
    else:
        report("Test2_annotation_keys",    False, f"len={len(ann)}  keys_ok={keys_ok}")

    if u_nonzero:
        report("Test2_u_exp_prev_nonzero", True,  f"_u_exp_prev={mixed._u_exp_prev}")
    else:
        report("Test2_u_exp_prev_nonzero", False, "_u_exp_prev still zero after steps")

    # Test reset
    mixed.reset_annotations()
    if len(mixed.annotations) == 0 and np.allclose(mixed._u_exp_prev, 0.0):
        report("Test2_reset_annotations",  True)
    else:
        report("Test2_reset_annotations",  False, f"after reset: len={len(mixed.annotations)} u={mixed._u_exp_prev}")

    del pilot_mp, mixed, expert
    torch.cuda.empty_cache()

except Exception as e:
    report("Test2_MixedPolicy",  False, str(e))
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — _filter_deviation_annotations
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Test 3: _filter_deviation_annotations ---")
try:
    N = 20
    # Goal placed at the END of the reference trajectory but close enough for max_goal_dist
    # All drone positions in deviation zone are within max_goal_dist=50m
    goal = np.array([10.0, 0.0, 1.0])  # reachable from trajectory

    # Synthetic tXUi: rows [t, x, y, z, ...]  shape (11, N)
    tXUi = np.zeros((11, N))
    tXUi[0, :] = np.linspace(0, 2, N)
    tXUi[1, :] = np.linspace(0, 5, N)  # x: 0→5
    tXUi[2, :] = 0.0                    # y ref = 0
    tXUi[3, :] = 1.0                    # z ref = 1

    # Xro: drone positions (shape 10×N, rows 0,1,2 = x,y,z)
    Xro = np.zeros((10, N))
    Xro[0, :]  = np.linspace(0, 5, N)   # x matches reference
    Xro[1, :]  = 0.0                     # y matches reference
    Xro[2, :]  = 1.0                     # z matches reference
    # Steps 5-10: large deviation in y (>0.3m) — goal is at x=10, so goal_dist≈8m < 50
    Xro[1, 5:11] = 1.0   # y drifts 1m from reference (0)
    # Steps 15-20: within 5m of goal (10,0,1)
    Xro[0, 15:] = 10.0   # x=10 = goal x
    Xro[1, 15:] = 0.0
    Xro[2, 15:] = 0.5    # dist = 0.5 < 5m ✓
    # Steps 0-4: x in [0..1], y=0, z=1 → no deviation, goal_dist > 5m (x=0..1, goal at x=10 → dist ~9m)
    # but goal_dist < 50 so they pass max_goal_dist, but no deviation → should be discarded

    # Fake annotations (one per timestep)
    annotations = [{"idx": i, "xnn": {}, "u": np.zeros(4), "x": np.zeros(10), "t": float(i)} for i in range(N)]

    kept = _filter_deviation_annotations(
        annotations=annotations,
        Xro=Xro,
        tXUi=tXUi,
        obj_target=goal,
        idx0=0,
        deviation_threshold=0.3,
        close_approach_dist=5.0,
        collision_steps=None,
        max_goal_dist=50.0,
    )

    kept_idxs = [a["idx"] for a in kept]
    # Steps 5-10 should be kept (deviation > 0.3m)
    dev_kept = all(i in kept_idxs for i in range(5, 11))
    # Steps 15-19 should be kept (near goal < 5m)
    close_kept = all(i in kept_idxs for i in range(15, N))
    # Steps 0-4 should NOT be kept (no deviation, far from goal)
    far_discarded = not any(i in kept_idxs for i in range(5))

    # Steps 0-4: x in [0..1.3], y=0, z=1 → far from goal(50,50,0), no y-deviation → should be discarded
    # Steps 5-10: y=1.0m drift → should be kept
    # Steps 15-19: right at goal → should be kept
    report("Test3_deviation_kept",  dev_kept,   f"idxs 5-10 in kept={kept_idxs}")
    report("Test3_close_kept",      close_kept,  f"idxs 15-19 in kept={kept_idxs}")
    report("Test3_far_discarded",   far_discarded, f"idxs 0-4 not in kept={kept_idxs}")

    # Test collision cutoff: collision at step 7 → steps ≥7 with deviation should be excluded
    kept_coll = _filter_deviation_annotations(
        annotations=annotations,
        Xro=Xro,
        tXUi=tXUi,
        obj_target=goal,
        idx0=0,
        deviation_threshold=0.3,
        close_approach_dist=5.0,
        collision_steps=[7],
        max_goal_dist=50.0,
    )
    kept_coll_idxs = [a["idx"] for a in kept_coll]
    # Step 7 onward should be excluded (collision cutoff)
    coll_cutoff_ok = not any(i in kept_coll_idxs for i in range(7, N))
    report("Test3_collision_cutoff", coll_cutoff_ok, f"cutoff@7 → kept={kept_coll_idxs}")

except Exception as e:
    report("Test3_filter", False, str(e))
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Best model restore
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Test 4: Best model restore ---")
try:
    import copy

    tmp_dir = tempfile.mkdtemp()
    pilot_t4 = Pilot(COHORT, PILOT_NAME)
    pilot_t4.set_mode("deploy")
    pilot_t4.model.to(DEVICE)

    # Save "before" state: record one param value
    param_before = next(pilot_t4.model.parameters()).data.clone()

    # Save current model as "best" checkpoint (simulate a better iter's weights)
    best_model_path = os.path.join(tmp_dir, "model_best.pth")
    _save_model_checkpoint(pilot_t4, best_model_path)

    # Corrupt the pilot's current model weights (simulate training degradation)
    with torch.no_grad():
        for p in pilot_t4.model.parameters():
            p.fill_(0.0)

    param_corrupted = next(pilot_t4.model.parameters()).data.clone()

    # Apply restore logic (from Fix 1)
    best_score = 0.7  # > 0.0
    if os.path.isfile(best_model_path) and best_score > 0.0:
        pilot_t4 = _swap_model(pilot_t4, best_model_path)
        pilot_t4.set_mode("deploy")
        roster_model_path = os.path.join(tmp_dir, "roster_model.pth")
        torch.save(pilot_t4.model.cpu(), roster_model_path)
        pilot_t4.model.to(DEVICE)

    param_after = next(pilot_t4.model.parameters()).data.clone()

    # Check: restored model matches original, NOT corrupted
    restored_matches_before = torch.allclose(param_after, param_before.to(param_after.device))
    restored_differs_from_corrupted = not torch.allclose(param_after, param_corrupted.to(param_after.device))
    roster_saved = os.path.exists(os.path.join(tmp_dir, "roster_model.pth"))

    report("Test4_model_restored_matches_best",
           restored_matches_before,
           f"param match: {restored_matches_before}  corrupt≠restored: {restored_differs_from_corrupted}")
    report("Test4_roster_pth_saved",
           roster_saved,
           f"roster file exists: {roster_saved}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    del pilot_t4
    torch.cuda.empty_cache()

except Exception as e:
    report("Test4_best_model_restore", False, str(e))
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DEBUG SUMMARY")
print("="*60)
n_pass = sum(1 for _, ok in PASS_FAIL if ok)
n_fail = sum(1 for _, ok in PASS_FAIL if not ok)
for name, ok in PASS_FAIL:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\nTotal: {n_pass}/{len(PASS_FAIL)} passed  ({n_fail} failed)")
if n_fail == 0:
    print("ALL TESTS PASSED")
