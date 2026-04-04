# SINGER — Detailed Technical Reference

This document contains in-depth implementation details for future reference. For a concise overview, see [README.md](../README.md).

---

## Observations Format

An **observation** is a single training sample:

- **Xnn** (neural network input):
  - `tx_com`: Time + drone state (position, velocity, orientation)
  - `obj_com`: Centroid features [bearing, elevation, apparent_size] from CLIPSeg
  - `dxu_par`: Rolling 20-step delta history
  - `img_vis`: CLIPSeg image embeddings
  - `tx_vis`: Visual state information

- **Ynn** (expert label):
  - `unn`: Expert action `[thrust, wx, wy, wz]` (4D)
  - `mfn`: Vehicle parameters `[mass, force_normalized]` (2D)
  - `onn`: Observed state `[x,y,z, vx,vy,vz, qx,qy,qz,qw]` (10D)

Observation file format (`.pt`):
```python
{
    "data": [{
        "Xnn": [xnn_dict_0, xnn_dict_1, ...],
        "Ynn": [ynn_dict_0, ynn_dict_1, ...],
        "Ndata": int,
        "rollout_id": int,
        "course": str,
    }],
    "set": "", "Nobs": int, "course": str
}
```

---

## RRT Branch Generation (Steps 1-2)

For each target object, RRT* builds a random exploration tree:
1. **RRT* sampling**: 2500 random nodes in 2D bounds `[-3.7,3.7] x [-8,8]`, rewired for shortest paths
2. **Branch extraction**: `nbranches=110` distinct collision-free routes per object
3. **Parameterization** (`process_branch()` in `trajectory_helper.py`):
   - Altitude set to -1.0m
   - Waypoints interpolated at 1.5 m/s with cubic spline smoothing
   - Velocities via `np.gradient`, normalized to constant speed
   - Quaternions from velocity heading, slerped toward target-facing orientation
   - 2s hover padding at end
   - Result: 18xN `tXUi` array

**tXUi format** (18xN):
```
Row 0:      time
Rows 1-10:  state [x, y, z, vx, vy, vz, qx, qy, qz, qw]
Rows 11-13: angular velocity [wx, wy, wz]
Row 14:     unused
Rows 15-17: motor commands [m1, m2, m3, m4]
```

Starting state per branch: `x0 = tXUi[1:11, 0]` — position from RRT root node, altitude -1.0m, velocity from trajectory heading.

### BC Perturbations

For each of 110 trajectories, the expert flies 4 perturbed repetitions:
```
Ntp = rate(10) x duration(2.0) = 20 time points per trajectory
For each of reps=4:
    For each of 20 time points:
        Perturb: position +/-0.40m, velocity +/-0.40m/s, quaternion +/-0.20
        MPC expert flies 2.0s from perturbed state back toward reference
```
Result: 110 x 4 x 20 = 8,800 short rollouts per object.

### Validation branches
`generate-rollouts --validation-mode` produces 11 branches per object using the same RRT process but different random trees. These are never used in training.

---

## DAgger Iteration Details

```
ITERATION i:
|
+-- 1. RESET TO BEST MODEL (if i > 0)
|   Load model_best_staging.pth -> pilot
|
+-- 2. COLLECT DATA (n_rollouts_per_object per object)
|   For each object:
|     For each sampled branch:
|       Perturb start +/-start_pos_noise (position only)
|       MixedPolicy = beta*expert + (1-beta)*pilot
|       Run full trajectory (3-10s):
|         u_expert = MPC.control(state)
|         u_pilot, xnn = Pilot.OODA(state)
|         Record annotation: {xnn, state, u_expert}
|         Execute: u_expert (prob beta) or u_pilot (prob 1-beta)
|     Filter annotations
|
+-- 3. RETRAIN COMMANDER
|   Convert annotations -> observations_dagger.pt
|   Train on ALL BC files (~330) + 1 DAgger file
|   For Nep_dagger epochs: MSE loss, backprop
|
+-- 4. EVALUATE
|   Pilot flies alone on seeded branches
|   Track success_rate, collision_rate per object
|   If better -> save as best
|   If worse -> restore previous best
|
+-- 5. EARLY STOPPING
    If patience exceeded without improvement -> stop
```

### DAgger Observation File

Location: `observation_data/{pilot}/dagger/observations_dagger.pt`

Generated at start of each iteration's retrain phase:
```python
For each annotation {xnn, x, u_expert}:
    Xnn.append(xnn)
    Ynn.append({
        "unn": u_expert,
        "mfn": [0.3, 0.3],
        "onn": x_state
    })
```

### Annotation Filtering

```
1. HARD CUTOFF at first collision step
2. SKIP if deviation > max_deviation_dist (8.0m)
3. SKIP if goal_dist > max_annotation_goal_dist (50.0m)
4. ALWAYS KEEP if goal_dist < close_approach_dist (5.0m)
5. KEEP if deviation > deviation_filter_dist (0.3m)
6. SKIP otherwise (on-trajectory, BC handles this)
```

### DAgger vs BC Perturbation Comparison

|  | BC Rollouts | DAgger Rollouts | Evaluation/Benchmark |
|--|-------------|-----------------|----------------------|
| Perturbed | Pos +/-0.4m, vel +/-0.4m/s, quat +/-0.2 | Position only +/-0.5m | Nothing (deterministic) |
| Source | 110 training branches | 5 branches sampled per iter | 50 sampled from 121 |
| Who flies | MPC expert only | beta*expert + (1-beta)*pilot | Pilot only |
| Duration | 2.0s segments | Full trajectory (3-10s) | Full branch |
| Runs/object | ~8,800 | 5 per iteration | 40 (eval) or 50 (benchmark) |
| Seed | None | None | 42 (fixed) |

---

## Simulation Architecture (Benchmark & Training)

```
Simulator.simulate() loop (200 Hz sim, 20 Hz control):
  For each control step (20 Hz):
    1. Render: gsplat.render_rgb(camera, T_c2w, query)
       -> image_dict["semantic"] (turbo-colormapped similarity heatmap)
    2. Add sensor noise: xsn = xcr + N(mu_sn, std_sn)
    3. Pilot.control(tcr, xsn, ucm, obj, icr, zcr):
       observe() -> process_image(icr) -> _compute_centroid() -> [bearing, elev, size]
       orient()  -> update DxU history
       decide()  -> model.extract_inputs(tx, Obj, Img, DxU)
       act()     -> Commander(tx_com, obj_com, dxu_par, zimg) -> unn
    4. ACADOS solver: xcr = solver.simulate(x=xcr, u=uin)
    5. Add model noise: xcr += N(mu_md, std_md)
```

Key: `obj=np.zeros((18,1))` passed to pilot. Centroid features overwrite indices [0,1,2] from vision inference.

### Success Criteria (_evaluate_run)

1. Drone enters within 2.3m (r1 + 2*r2 = 2.0 + 0.3) of object centroid
2. No collision before first entry:
   - Point-cloud collision: any Gaussian center < 0.15m from drone
   - Out-of-bounds: drone exits minbound/maxbound
3. FOV check: logged but NOT part of success criteria

---

## Centroid Feature Computation

```
_compute_centroid() flow:
  icr (ImageNet-normalized, 3x224x224)
    -> heat = mean over channels (224x224)
    -> heat /= 255.0
    -> V9:  mask = heat > percentile(heat, 75)  (~25% of pixels)
      V10: mask = heat > median(heat)           (~50% of pixels)
    -> centroid = weighted_average(pixel_coords, weights=heat[mask])
    -> bearing   = 2 * cx - 1   ([-1, 1])
    -> elevation = 2 * cy - 1   ([-1, 1])
    -> apparent_size:
        V9:  mask.sum() / total_pixels  (always ~0.25, known bug)
        V10: (heat > 0.3).sum() / total_pixels
```

**CRITICAL**: centroid_version in pilot config must match training version. V9 model with V10 centroid = ~0% success.

---

## Configuration Reference

### Experiment Config

```yaml
cohort: "SSV_DAGGER_CENTROID_V9"     # Experiment name, data under cohorts/{cohort}/
method: "rrt"                         # RRT* path planning
review: false                         # false=run, true=review existing
bc_cohort: "ssv_BC_6S"               # Copy BC model from here (skip steps 1-5)

# BC Training
Nep_his: 100                          # HistoryEncoder epochs
Nep_com: 150                          # Commander (BC) epochs

# DAgger Core
expert_type: "mpc"                    # MPC expert for corrections
n_iterations: 8                       # Max DAgger iterations
Nep_dagger: 8                         # Epochs per DAgger retrain
n_eval_per_iter: 20                   # Eval trajectories per object per iteration
n_benchmark: 50                       # Final benchmark trajectories per object
dagger_lr: 0.00003                    # DAgger learning rate (lower than BC's 1e-4)

# DAgger Strategy
aggregate_dagger: false               # false=online (recommended), true=cumulative
reset_to_best: true                   # Restore best model before each iteration
eval_seed: 42                         # Fixed seed for reproducible evaluation

# Data Collection
n_rollouts_per_object: 5              # Branches per object per iteration
start_pos_noise: 0.5                  # Position perturbation +/-0.5m

# Annotation Filtering
deviation_filter_dist: 0.3            # Keep if deviated >0.3m from reference
close_approach_dist: 5.0              # Always keep within 5m of goal
max_annotation_goal_dist: 50.0        # Discard if >50m from goal
max_deviation_dist: 8.0               # Discard if >8m deviation

# Scene
flights:
  - ["flightroom_ssv_exp", "flightroom_ssv_exp"]
roster:
  - "InstinctJester"
```

### Scene Config

```yaml
queries: ["green clock", "green and pink leafblower", "yellow handheld cordless drill on two boxes"]
radii:
  - [2.0, 0.4]         # [outer_radius, inner_radius]
altitudes:
  - -1.0                # Flight height (negative = above ground)
similarities:
  - [0.90, 0.025]       # CLIPSeg [threshold, margin]
nbranches:
  - 110                  # RRT branches per object
algorithm: 'RRT*'
N: 2500                  # RRT iterations
step_size: 1.0
minbound: [-3.7, -8, -2]
maxbound: [3.7, 8, 0]
```

---

## Reproducing a Benchmark

```python
from sousvide.instruct.train_dagger import (
    _run_benchmark_pilot, _get_scene, _preload_bc_trajectories, Pilot, DEVICE
)
_get_scene(scene_name, scenes_cfg_dir, frame_name, rollout_name)
_preload_bc_trajectories(bc_cohort_name, flights, scenes_cfg_dir)
pilot = Pilot(cohort_name, pilot_name)
pilot.set_mode("deploy")
pilot.model.to(DEVICE)
metrics = _run_benchmark_pilot(pilot=pilot, model_path=model_path, ...)
```

---

## Where to Find Results

| What | Location |
|------|----------|
| Deploy-ready model | `cohorts/{cohort}/roster/{pilot}/model.pth` |
| Pre-DAgger BC model | `cohorts/{cohort}/roster/{pilot}/model_before_dagger.pth` |
| Best DAgger checkpoint | `cohorts/{cohort}/dagger_data/{pilot}/benchmark_*/model_best_dagger.pth` |
| Benchmark JSON | `cohorts/{cohort}/dagger_data/{pilot}/benchmark_*/benchmark_results.json` |
| DAgger annotations | `cohorts/{cohort}/dagger_data/{pilot}/dagger_iter_*.pt` |
| DAgger observations | `cohorts/{cohort}/observation_data/{pilot}/dagger/observations_dagger.pt` |
| BC observations | `cohorts/{cohort}/observation_data/{pilot}/{course}/observations*.pt` |
| Simulation videos | `cohorts/{cohort}/simulation_data/{timestamp}/videos/*.mp4` |
| Trajectory plots | `cohorts/{cohort}/dagger_data/{pilot}/plots/*.png` |
| Seen/unseen benchmark | `cohorts/{cohort}/benchmark_seen_unseen/` |
