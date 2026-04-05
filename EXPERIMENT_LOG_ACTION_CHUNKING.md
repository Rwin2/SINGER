# Experiment Log: Action Chunking & Flow Matching for SINGER

**Branch**: `feature/action-chunking-flow-matching`
**Date started**: 2026-04-04
**Base model**: V9 DAgger (88.0% success, 8.0% collision)
**Base BC**: ssv_BC_CENTROID_V9 (80.7% success)

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
- **Status**: RUNNING on GPU 0 (8GB, 100% util)
- **Estimated completion**: ~5 hours (each epoch processes 330 files × ~480 chunked samples)
- **Model updates observed**: best_model.pth updated at 22:59, 23:09, 23:19
- **Bottleneck**: Data loading (330 .pt files loaded + chunked per epoch)
- **Epoch timing**: ~10 min/epoch → 30 epochs ≈ 5 hours
- **Expected completion**: ~2026-04-05 03:46 AM PST

### Optimization note for future
Loading 330 files per epoch is inefficient. Consider:
- Pre-compute chunked dataset once, save as single .pt file
- Use in-memory caching after first load
- Or reduce to subset for quick tests

## Results

### Phase 1: Action Chunking BC
| Metric | V9 BC Baseline (4D) | Chunked BC (20D, H=5 K=2) |
|--------|--------------------|-----------------------------|
| Train loss | 0.012 | TBD |
| Test loss | TBD | TBD |
| Success rate | 80.7% | TBD |
| Collision rate | 13.3% | TBD |
| Goal distance | 2.07m | TBD |

### Phase 2: Flow Matching BC
(pending Phase 1 results)

### Phase 3: Combined + DAgger
(pending Phase 1+2 results)
