# SINGER BC + DAgger Architecture Analysis
## Generated 2026-03-13

---

## 1. Pipeline Overview

```
STEP 1: generate-rollouts     → Expert MPC flies RRT trajectories, records (T,X,U) + video
STEP 2: generate-rollouts --val → Same but for validation split
STEP 3: generate-observations  → Pilot.OODA() on rollout data → (Xnn, Ynn) observation files
STEP 4: train-history          → Train HistoryEncoder on observation data
STEP 5: train-command          → Train Commander on observation data (BC training)
STEP 6: dagger                 → Iterative DAgger refinement on the BC-trained model
```

---

## 2. Key Data Structures

### 2.1 RRT Trajectory (tXUi) — `(18, N)` array
```
Row 0:      time
Rows 1-10:  state [x, y, z, vx, vy, vz, qx, qy, qz, qw]
Rows 11-13: angular velocity [wx, wy, wz]
Row 14:     unused
Rows 15-17: motor commands [m1, m2, m3, m4]
```
Stored in `.pkl` files under `configs/scenes/`.

### 2.2 Rollout Data — `rollout_data/{course}/trajectories_*.pt`
```python
{
    "data": [{
        "Tro": ndarray(Nctl+1,),     # Time vector
        "Xro": ndarray(10, Nctl+1),  # State: [x,y,z,vx,vy,vz,qx,qy,qz,qw]
        "Uro": ndarray(4, Nctl),     # Control: [thrust, wx, wy, wz]
        "obj": ndarray(18, 1),       # Goal
        "Ndata": int,
        "rollout_id": int,
        "course": str,
        "frame": {"mass": float, "force_normalized": float}
    }],
    "set": "", "course": str
}
```

### 2.3 Observation Data — `observation_data/{pilot}/{course}/observations*.pt`
```python
{
    "data": [{
        "Xnn": [xnn_dict_0, xnn_dict_1, ...],   # NN input features (from pilot.OODA)
        "Ynn": [{"unn": (4,), "mfn": (2,), "onn": (10,)}, ...],  # Expert labels
        "Ndata": int,
        "rollout_id": int,
        "course": str,
        "frame": {"mass": float, "force_normalized": float}
    }],
    "set": "", "Nobs": int, "course": str
}
```

### 2.4 xnn Dictionary (NN inputs from Pilot.OODA → decide → extract_inputs)
```python
{
    "tx_com":  tensor,   # Current time + state
    "obj_com": tensor,   # Goal/objective
    "dxu_par": tensor,   # Delta history (position, velocity, quaternion, actions)
    "img_vis": tensor,   # Image/perception features
    "tx_vis":  tensor,   # Visual state
}
```

### 2.5 Ynn Dictionary (Expert labels)
```python
{
    "unn": ndarray(4,),   # Expert action [thrust, wx, wy, wz]
    "mfn": ndarray(2,),   # [mass, force_normalized]
    "onn": ndarray(10,),  # Observed state [x,y,z,vx,vy,vz,qx,qy,qz,qw]
}
```

---

## 3. Pilot Architecture

### 3.1 OODA Loop (`control/pilot.py`)
```
Pilot.OODA(upr, tcr, xcr, obj, icr, zcr):
  ├─ observe()  → Store tcr, xcr, obj, img. Convert to tensors. Build tx_cr.
  ├─ orient()   → Compute DxU (delta history, 11×20) and Znn (feature history, Nz×20)
  ├─ decide()   → xnn = model.extract_inputs(tx_cr, obj, img, DxU, Znn, hy_idx)
  └─ act()      → unn, znn = model(*model.get_commander_inputs(xnn))
  Return: (unn, znn, adv, xnn, tsol)
```

### 3.2 Key Pilot Properties
- `Pilot.control(tcr, xcr, upr, obj, icr, zcr)` wraps `OODA(upr, tcr, xcr, ...)` — arg order swap!
- `Pilot.nzcr = None` → zcr always becomes `torch.zeros(0)` in observe()
- `DxU`: (11, 20) rolling history buffer. Updated each timestep.
- `Znn`: (Nz, 20) feature history. Updated each timestep.
- `hy_idx`: Rolling buffer index (wraps 0-19).
- Hz: 20 Hz control rate.

---

## 4. Training Pipeline (`train_policy.py`)

### 4.1 train_roster / train_student
```python
train_roster(cohort_name, roster, mode, Neps, lim_sv, lr=1e-4, batch_size=64, course_name=None)
  └─ For each pilot in roster:
       student = Pilot(cohort_name, student_name)
       student.set_mode('train')
       train_student(cohort_name, student, mode, Neps, ...)
```

### 4.2 Training Loop (per epoch)
```
1. unlock_networks(student, mode)  — freeze all params, unlock mode-specific params
2. get_data_paths(cohort, student, course_name) → (train_files, test_files, val_files, rollout_files)
3. For each train_file:
     dataset = generate_dataset(file, student, mode, device)
     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
     For each (input, label) in dataloader:
       prediction, _ = model(*input)
       loss = MSELoss(prediction, label)
       loss.backward() → opt.step() → opt.zero_grad()
4. Same loop for test_files (no backward)
5. Periodically: validation + rollout evaluation
6. Save model.pth at lim_sv intervals and at end
```

### 4.3 Data Path Discovery (`synthesized_data.py::get_data_paths()`)
```python
# Base: cohorts/{cohort}/observation_data/{student}/
# If course_name=None: scan ALL course subdirectories
# If course_name="dagger": scan only dagger/ subdirectory

For each course directory:
  observations_val_rollout*.pt → rollout_paths
  observations_val*.pt         → validation_paths
  observations*.pt (not val)   → data_files

  data_files split: [:-1] → train, [-1] → test
  Single file: used for both train and test
```

### 4.4 Data Volumes (Full Run)
- BC observation files: 330 files × ~160 samples = ~52,800 total
- DAgger: 1 file with ~300 samples per iteration
- DAgger is ~0.6% of total training data (but high-loss samples contribute more gradient)

---

## 5. DAgger Implementation (`train_dagger.py`)

### 5.1 Per-Iteration Flow
```
For each iteration:
  1. Reset to best model (if reset_to_best=True and iter > 0)
  2. Collect DAgger rollouts:
     For each (scene, object):
       a. Load scene (from _SCENE_CACHE) + pkl (from _PKL_CACHE)
       b. Perturb x0 with ±start_pos_noise
       c. Build expert (MPC tracking BC reference tXUi)
       d. MixedPolicy = β * expert + (1-β) * pilot
       e. simulator.simulate(mixed_policy, t0, tf, x0, query)
       f. Annotate: at each step, record {xnn, x, u_expert, t, query}
       g. Filter annotations (deviation + near-goal + collision cutoff + max filters)
  3. Aggregate (online mode: current iter only, no accumulation)
  4. Save DAgger annotations to dagger/observations_dagger.pt (BC obs format)
  5. Retrain Commander: train_roster(course_name=None) → BC + DAgger mixed
  6. Evaluate: n_eval runs per object from 2nd half of tXUi
  7. Track best model (by mean success_rate, goal_dist tiebreaker)
  8. Early stopping: if 2 consecutive iters without improvement → break
  9. Beta decay: β *= β_decay
```

### 5.2 MixedPolicy
```python
class MixedPolicy:
  control(tcr, xcr, upr, obj, icr, zcr):
    u_expert = expert.control(tcr, xcr, upr, obj, icr, zcr)
    u_pilot, znn, _, xnn, _ = pilot.OODA(self._u_exp_prev, tcr, xcr, obj, icr, zcr)
    # CRITICAL: pass expert action history to pilot (not mixed action)
    self._u_exp_prev = u_expert.copy()
    annotations.append({"xnn": xnn_cpu, "x": xcr, "u": u_expert, ...})
    u_out = u_expert if rand() < β else u_pilot
    return u_out, znn, adv, tsol
```

### 5.3 Annotation Filtering (`_filter_deviation_annotations`)
```
1. Hard cutoff at first collision step (post-crash physics are garbage)
2. For each timestep before cutoff:
   a. Skip if deviation > max_deviation_dist (extreme excursion)
   b. Skip if goal_dist > max_annotation_goal_dist (runaway)
   c. Always keep if goal_dist < close_approach_dist (final approach)
   d. For runaway trajectories: skip deviation-based annotations
   e. Keep if deviation > deviation_threshold
```

### 5.4 DAgger Data Conversion (`_retrain_commander`)
```
DAgger annotations → BC observation format:
  For each annotation {xnn, x, u}:
    Xnn.append(xnn)           # NN input dict (same format as BC)
    Ynn.append({
      "unn": u_expert,        # Expert action label
      "mfn": [0.3, 0.3],      # Default mass/force
      "onn": x_state           # State observation
    })

  Save as: observation_data/{pilot}/dagger/observations_dagger.pt

  Also: symlink BC obs course dirs into DAgger cohort
        (skips if dirs already exist — full run has its own)

  Then: train_roster(course_name=None) → trains on BC + DAgger mixed
```

### 5.5 V3 Validated Settings
```yaml
aggregate_dagger: false   # Online mode (no accumulation)
reset_to_best: true       # Reset to best before each iter
eval_seed: 42             # Fixed seed for reproducible eval
# Per-iter eval samples from 2nd half of tXUi (matches benchmark)
# Early stopping after 2 consecutive drops
```

---

## 6. Consistency Between BC and DAgger

| Aspect | BC Pipeline | DAgger Retraining | Match? |
|--------|-------------|-------------------|--------|
| xnn source | `pilot.OODA()` | `pilot.OODA()` | ✅ |
| xnn keys | tx_com, obj_com, dxu_par, img_vis, tx_vis | same | ✅ |
| Ynn format | {unn, mfn, onn} | {unn, mfn, onn} | ✅ |
| Obs file format | {data: [{Xnn, Ynn, ...}]} | identical | ✅ |
| Data extraction | extract_data() | extract_data() | ✅ |
| Training function | train_roster("Commander") | train_roster("Commander") | ✅ |
| Optimizer | Adam(lr=1e-4) | Adam(lr=dagger_lr) | ✅ (different lr OK) |
| Action history (upr) | Previous expert action | Previous expert action | ✅ |
| Data augmentation | Noise on state (in obs gen) | None (expert state as-is) | ⚠️ Expected |

### Critical Design Decisions
1. **Expert action history**: MixedPolicy passes `u_exp_prev` (pure expert) to pilot, NOT mixed action. This matches BC training where upr = ucr (expert action always).
2. **Online mode**: No DAgger data accumulation. Each iter trains on current annotations only + BC. Prevents redundant data dilution.
3. **Reset to best**: Restore best model before each iteration. Prevents catastrophic cascade.
4. **Eval from 2nd half**: Per-iter eval samples from 2nd half of tXUi, matching benchmark distribution.

---

## 7. File System Layout
```
SINGER/
├── configs/
│   ├── experiment/   *.yml configs
│   ├── method/       rrt.json
│   ├── frame/        carl.json
│   ├── policy/       vrmpc_rrt.json
│   └── scenes/       flightroom_ssv_exp.yml + *.pkl (RRT trajectories)
│
├── cohorts/{cohort}/
│   ├── rollout_data/{course}/
│   │   ├── trajectories_*.pt   (Step 1)
│   │   ├── imgdata_*.pt
│   │   └── video_*.mp4
│   ├── observation_data/{pilot}/
│   │   ├── {course}/observations*.pt   (Step 3)
│   │   └── dagger/observations_dagger.pt   (Step 6)
│   ├── roster/{pilot}/
│   │   ├── model.pth           (best/latest)
│   │   ├── best_model.pth
│   │   ├── last_model.pth
│   │   └── losses_*.pt
│   └── dagger_data/{pilot}/
│       ├── dagger_iter_*.pt    (per-iteration raw annotations)
│       ├── dagger_aggregated.pt
│       └── benchmark_{timestamp}/
│           ├── model_before_dagger.pth
│           ├── model_best_dagger.pth
│           └── benchmark_results.json
│
├── src/sousvide/
│   ├── control/pilot.py            (Pilot, OODA loop)
│   ├── instruct/train_policy.py    (train_roster, train_student)
│   ├── instruct/train_dagger.py    (DAgger: train_dagger_policy)
│   ├── instruct/synthesized_data.py (get_data_paths, extract_data)
│   ├── synthesize/rollout_generator.py (generate_rollout_data)
│   └── synthesize/observation_generator.py (generate_observation_data)
│
└── notebooks/ssv_muilti3dgs_campaign_coruscant.py  (CLI entry point)
```

---

## 8. Previous DAgger Results

### V3 Smoke Test (3 objects, 4 iterations, seed=42)
| Metric | Before (BC) | After (DAgger V3) | Delta |
|--------|-------------|-------------------|-------|
| Success | 38.9% | 61.1% | +22.2pp |
| Collision | 34.4% | 13.3% | -21.1pp |
| Goal dist | 2.65m | 2.22m | -0.43m |

Config: `configs/experiment/ssv_dagger_v3_smoke.yml`
Best model: iter 1 (sr=71%, gd=1.97m)

### Per-object breakdown
| Object | BC | DAgger V3 |
|--------|-------|-----------|
| Clock | 100%, 1.45m | 100%, 0.71m |
| Leafblower | 0%, 100% coll | 60%, 20% coll |
| Drill | 17%, 3.60m | 23%, 4.49m |
