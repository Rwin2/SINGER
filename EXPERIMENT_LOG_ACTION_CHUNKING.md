# Experiment Log: Action Chunking & Flow Matching for SINGER

**Branch**: `feature/action-chunking-flow-matching`
**Date started**: 2026-04-04
**Base model**: V9 DAgger (88.0% success, 8.0% collision)
**Base BC**: ssv_BC_CENTROID_V9 (80.7% success)

## Branch Map
| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Stable release | V9 DAgger (88%) |
| `feature/centroid-v9` | Centroid feature dev | Merged into main |
| `feature/action-chunking-flow-matching` | **This branch**: FM + chunking experiments | Active |
| `feature/position-prediction` | (planned) Waypoint prediction + MPC tracking | Not started |
| `feature/dynamics-regularization` | (planned) Physics-informed action prediction | Not started |

## Position vs Action Prediction Analysis (2026-04-05)

### Current: Action prediction (4D thrust+rates)
- Directly outputs flight controller commands
- No physics reasoning — network must implicitly learn dynamics
- Works with DAgger (expert corrects actions directly)

### Option A: Waypoint prediction + MPC tracking
- Network predicts `(x,y,z)` waypoints, existing MPC tracks them
- Physics-consistent, bounded, but slower inference (MPC latency)
- Branch: `feature/position-prediction` (to be created)

### Option B: Physics-informed regularization (RECOMMENDED FIRST)
- Keep action prediction, add dynamics loss during training
- Forward-simulate predicted actions, compare with actual next-state
- `extract_data_dynamics()` already implemented in synthesized_data.py!
- Branch: `feature/dynamics-regularization` (to be created)

### Why not H=50 like Pi-0?
- SINGER Commander MLP has ~25K params vs Pi-0's 3B params
- Trajectories are 120 steps (6s) — H=50 = 2.5s = almost half the trajectory
- 4D action space has lower multimodality than 7-DoF robot arms
- H=10-20 likely optimal for SINGER's scale

---

## Plan

### Phase 1: Action Chunking BC (Small Test)
- H=5 (predict 5 future 4D actions = 20D output), K=2 (execute 2 before re-query)
- Reuse existing V9 observation data (no regeneration)
- Create chunked dataset wrapper (window sequential Ynn["unn"] into chunks)
- New pilot config: InstinctJester_chunked.json (output_size=20)
- Modify pilot.act() with chunk buffer
- Train BC (small: 30 epochs) → evaluate vs baseline

### Phase 2: Flow Matching BC (if Phase 1 works)
- Replace MSE with flow matching loss in Commander training
- Add time embedding (32D) to Commander input
- Euler integration at inference (10 steps)
- Compare vs MSE baseline

### Phase 3: Combined + DAgger
- Best of Phase 1+2 + DAgger
- Compare vs V9 DAgger baseline

---

## Changes Tracking

### Files Modified
- `src/sousvide/instruct/synthesized_data.py` — added `extract_data_chunked()` and `generate_dataset_chunked()` for windowed action chunks
- `src/sousvide/control/pilot.py` — added chunk buffer support in `__init__` + `act()` (backward compatible, enabled via config)

### Files Created
- `configs/pilots/InstinctJester_chunked.json` — pilot config with output_size=20 (5×4), hidden=[256,128], action_chunk settings
- `configs/experiment/ssv_bc_chunked_test.yml` — experiment config for chunked BC test
- `scripts/test_action_chunking.py` — standalone training script with weight transfer from V9
- `cohorts/ssv_BC_CHUNKED_TEST/` — cohort dir with symlinked V9 observation data

### Design Decisions
1. **Reuse V9 observations**: Symlinked existing data, windowing done at dataset loading time (no regeneration needed)
2. **Transfer learning**: HistoryEncoder + VisionMLP weights copied from V9 BC model; only Commander retrained
3. **Larger Commander**: [256,128] hidden (was [100,100]) to handle 20D output vs 4D
4. **Conservative chunking**: H=5, K=2 (predict 0.25s ahead at 20Hz, execute 0.1s)
5. **Backward compatible**: chunk buffering in pilot.act() only activated when action_chunk config present

---

## Training Status

### Phase 1: Action Chunking BC (30 epochs)
- **Started**: 2026-04-04 22:46 PST
- **Completed**: 2026-04-05 03:44 PST (~5 hours)
- **Epoch timing**: ~10 min/epoch
- **Bug found**: Training script saved `model.state_dict()` instead of full model object. Fixed post-training with `scripts/fix_model_format.py`.
- **Best test loss**: 0.02406 at epoch 23
- **Final train loss**: 0.00258, Final test loss: 0.03000

### Phase 2a: Flow Matching BC (fm_only, 30 epochs, 4D actions)
- **Started**: 2026-04-05 04:50 PST
- **Completed**: 2026-04-05 09:19 PST (4h 30m)
- **Best test loss**: 0.20731 at epoch 13
- **Final train loss**: 0.09335
- Note: FM loss not directly comparable to MSE loss

### Phase 2b: Flow Matching + Chunking (fm_chunked, 30 epochs, 20D)
- **Started**: 2026-04-05 09:22 PST
- **Status**: RUNNING on GPU 0

### Optimization note for future
Loading 330 files per epoch is inefficient. Consider:
- Pre-compute chunked dataset once, save as single .pt file
- Use in-memory caching after first load
- Or reduce to subset for quick tests

## Results

### Phase 1: Action Chunking BC (benchmark: 20 traj/object, seed=42)
| Metric | V9 BC Baseline (4D) | Chunked BC (20D, H=5 K=2) |
|--------|--------------------|-----------------------------|
| Train loss | 0.012 | 0.00258 |
| Best test loss | — | 0.02406 (epoch 23) |
| Success rate | 100.0% | 100.0% |
| Collision rate | 0.0% | 0.0% |
| Goal distance | 1.85m | 2.00m |

Per-object breakdown:
| Object | V9 Baseline GoalDist | Chunked BC GoalDist |
|--------|---------------------|---------------------|
| Clock | 1.91±0.14m | 1.84±0.19m |
| Leafblower | 1.97±0.48m | 2.50±0.35m |
| Boxes (cordless) | 1.67±0.39m | 1.67±0.38m |

**Conclusion**: Action chunking matches baseline success rate. Slightly higher goal distance for leafblower (2.50 vs 1.97m) but comparable overall. The 20D output prediction doesn't degrade performance.

### Phase 2a: Flow Matching BC (fm_only, 4D)
- Best test loss: 0.20731 (epoch 13), final train loss: 0.09335
- Note: FM loss = velocity prediction MSE, not comparable to BC action MSE
- Benchmark pending (need full-range benchmark with 50 traj/object)

### Phase 2b: Flow Matching + Chunking (fm_chunked, 20D)
- **Started**: 2026-04-05 09:22 PST
- **Status**: RUNNING (last checkpoint: epoch ~27-28 of 30)
- ETA: ~2026-04-05 14:30

### Phase 2c: Longer Horizon — Chunked MSE H=10 K=3 (40D output)
- **Started**: 2026-04-05 13:00 PST
- **Status**: RUNNING (epoch 0/30)
- Pilot config: `InstinctJester_chunked_h10.json` (hidden=[256,256], output=40D)
- Cohort: `ssv_BC_CHUNKED_H10K3`
- Reasoning: H=5 may be too short for temporal coherence. Pi-0 uses H=50.

### Phase 3: Full Trajectory Benchmark (50 traj, full range)
- **Script**: `scripts/eval_fair_benchmark.py` (uses `full_range=True`)
- **Started**: 2026-04-05 15:58 PST
- **Status**: RUNNING (V9_BC_baseline, clock run 7/50 at 16:09)
- **ETA**: ~15 hours (750 sims × 75s each) — overnight run
- **Models**: baseline, v9_dagger, chunked_H5, fm_only, fm_chunked
- H=10 will be added after its training completes
- Log: `full_benchmark.log`

### Phase 4: DAgger on best variant
(pending Phase 3 results)

### Phase 5 (planned): FM loss during DAgger retraining
- Placeholder script: `scripts/train_fm_dagger.py`
- Most pragmatic approach: start DAgger from FM BC model, use standard MSE retraining
- Advanced: modify `_retrain_commander_ewc()` to support FM loss

---

## Code Review (2026-04-05 session)

### Issues Found
1. **`eval_all_experiments.py --full-range`**: Inflates success rates by including easy starts
2. **`FlowMatchingCommanderWrapper.forward()`**: Runs HistoryEncoder+VisionMLP twice (perf, not correctness)
3. **`FlowMatchingCommander.py:223`**: Hardcoded `input_size=147` — breaks if objective features change
4. **Gaussian timestep weighting**: May not be optimal for 4D action space (DreamZero designed for images)

### Files Created This Session
- `scripts/eval_fair_benchmark.py` — Fair benchmark with second-half starts
- `scripts/train_chunked_horizons.py` — Train multiple chunk horizons
- `scripts/train_fm_dagger.py` — FM+DAgger placeholder
- `configs/pilots/InstinctJester_chunked_h10.json` — H=10 K=3 pilot config
- `FLOW_MATCHING_CHUNKING_ANALYSIS.md` — Comprehensive analysis document
