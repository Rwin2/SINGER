# SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones

SINGER trains a neural drone pilot (InstinctJester) to navigate to language-described targets in 3D Gaussian Splat environments. The training pipeline uses Behavioral Cloning (BC) from an MPC expert, followed by DAgger (Dataset Aggregation) for iterative refinement.

---

## Table of Contents

1. [Installation](#installation)
2. [Concepts & Terminology](#concepts--terminology)
3. [Architecture Overview](#architecture-overview)
4. [Quick Start — Full Pipeline](#quick-start--full-pipeline)
5. [Pipeline Steps in Detail](#pipeline-steps-in-detail)
6. [DAgger Deep Dive](#dagger-deep-dive)
7. [Configuration Reference](#configuration-reference)
8. [Deploying & Visualizing Results](#deploying--visualizing-results)
9. [Where to Find Results](#where-to-find-results)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)

---

## Installation

SINGER requires [FiGS-Standalone](https://github.com/StanfordMSL/FiGS-Standalone) as its simulator and renderer.

### Option A: Docker (original setup)

```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
git clone https://github.com/StanfordMSL/SINGER.git
cd FiGS-Standalone && docker-compose build
cd ../SINGER && docker-compose run singer
```

### Option B: Conda (coruscant server)

```bash
conda activate FiGS
export ACADOS_SOURCE_DIR=/path/to/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/FiGS-Standalone/acados/lib
cd /path/to/SINGER
```

---

## Concepts & Terminology

### What is an Observation?

An **observation** is a single training sample for the neural pilot. It consists of:

- **Xnn** (neural network input): What the pilot "sees" at one timestep
  - `tx_com`: Current time + drone state (position, velocity, orientation)
  - `obj_com`: Goal/target information (where to fly)
  - `dxu_par`: Rolling history of deltas (how the drone has been moving over the last 20 steps)
  - `img_vis`: Image/perception features from the onboard camera (CLIPSeg embeddings)
  - `tx_vis`: Visual state information

- **Ynn** (expert label): What the expert MPC would have commanded at this timestep
  - `unn`: Expert action `[thrust, wx, wy, wz]` (4 values — collective thrust + 3 angular velocities)
  - `mfn`: Vehicle parameters `[mass, force_normalized]` (2 values)
  - `onn`: Observed state `[x,y,z, vx,vy,vz, qx,qy,qz,qw]` (10 values)

**How observations are generated (Step 3)**: For each timestep of each expert rollout, we run the pilot's OODA loop (`Pilot.OODA()`) to extract what the pilot would "see" (Xnn), and pair it with what the expert actually commanded (Ynn). This creates `{Xnn, Ynn}` pairs — the pilot learns to predict Ynn given Xnn.

### Epoch vs Iteration

- **Epoch**: One complete pass through all training files. During BC training (`Nep_com=150`), the model sees all ~330 observation files 150 times. During DAgger retraining (`Nep_dagger=8`), the model sees all BC files + 1 DAgger file 8 times.

- **Iteration** (DAgger only): One full cycle of: collect data → filter → retrain → evaluate. Each iteration has `Nep_dagger` epochs inside it. So with `n_iterations=8, Nep_dagger=8`, there are 8 iterations × 8 epochs = 64 total training epochs during DAgger.

### HistoryEncoder vs Commander

The neural pilot has two main components:

- **HistoryEncoder** (trained in Step 4): Processes temporal sequences — how the drone has been moving over the last 20 timesteps. Produces a compressed feature vector from the rolling history buffer `DxU (11×20)` and feature history `Znn (Nz×20)`.

- **Commander** (trained in Step 5, refined in Step 6): Takes all inputs (state, goal, image, history features) and outputs the action command `[thrust, wx, wy, wz]`. This is the part that DAgger fine-tunes.

### What is DAgger?

**DAgger** (Dataset Aggregation) solves the "distribution shift" problem in Behavioral Cloning:

- **Problem**: BC trains on expert data (where the expert flies). But at deploy time, the pilot makes small mistakes, ending up in states the expert never visited. The pilot has no training data for these states → it makes bigger mistakes → cascading failure.

- **Solution**: Fly the pilot (with some expert guidance), observe where it deviates from the expert, and retrain on those specific states. Repeat until the pilot can handle its own mistakes.

### Cohort

A **cohort** is a self-contained experiment directory under `cohorts/`. It holds all data for one training run: rollout data, observations, trained models, DAgger data, and simulation results. Different experiments use different cohort names.

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────────────────┐
                    │              SINGER Training Pipeline                │
                    └──────────────────────────────────────────────────────┘

  Step 1-2: Generate Rollouts          Step 3: Generate Observations
  ┌─────────────────────────┐          ┌──────────────────────────────┐
  │ MPC expert flies RRT    │          │ Pilot.OODA() processes each  │
  │ trajectories in GS sim  │──────────│ rollout timestep → (Xnn,Ynn) │
  │ Records (T,X,U) + video │          │ observation files            │
  └─────────────────────────┘          └──────────────┬───────────────┘
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
  Step 4: Train History                Step 5: Train Commander (BC)
  ┌─────────────────────────┐          ┌──────────────────────────────┐
  │ Train HistoryEncoder on │          │ Train Commander network on   │
  │ observation sequences   │──────────│ observation data (MSE loss)  │
  │ (temporal features)     │          │ → model.pth (BC baseline)    │
  └─────────────────────────┘          └──────────────┬───────────────┘
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
  Step 6: DAgger (Iterative Refinement)
  ┌──────────────────────────────────────────────────────────────────┐
  │ For each iteration:                                              │
  │   1. MixedPolicy = β·expert + (1-β)·pilot flies trajectories    │
  │   2. Record where pilot deviates from expert (annotations)       │
  │   3. Filter annotations (deviation, goal proximity, collisions)  │
  │   4. Retrain Commander on BC data + DAgger annotations           │
  │   5. Evaluate → track best model → early stopping               │
  └──────────────────────────────────────────────────────────────────┘
```

### Neural Pilot — OODA Loop (20 Hz)

```
Pilot.OODA(upr, tcr, xcr, obj, icr, zcr):
  observe() → Store drone state (tcr, xcr), goal (obj), camera image (icr)
  orient()  → Compute DxU (11×20 delta history) and Znn (feature history)
  decide()  → xnn = model.extract_inputs(state, goal, image, history)
  act()     → unn = Commander(xnn)  →  [thrust, wx, wy, wz]

Arguments:
  upr: Previous action (for history tracking)
  tcr: Current time
  xcr: Current state [x,y,z, vx,vy,vz, qx,qy,qz,qw] (10D)
  obj: Goal/target info (18D)
  icr: Camera image
  zcr: Feature input (unused, always zeros)
```

---

## Quick Start — Full Pipeline

### 1. Create the experiment config

See [Configuration Reference](#configuration-reference) for all parameters.

### 2. Run the full pipeline

```bash
export CUDA_VISIBLE_DEVICES=0
CFG=configs/experiment/ssv_dagger_full_campaign.yml
ENTRY=notebooks/ssv_muilti3dgs_campaign_coruscant.py
RUN="conda run --no-capture-output -n FiGS python $ENTRY"

# Step 1: Generate expert rollouts (MPC flies RRT trajectories)
$RUN generate-rollouts --config-file $CFG 2>&1 | tee logs/step1_rollouts.log

# Step 2: Generate validation rollouts
$RUN generate-rollouts --config-file $CFG --validation-mode 2>&1 | tee logs/step2_val.log

# Step 3: Generate observations (Pilot.OODA on rollout data)
$RUN generate-observations --config-file $CFG 2>&1 | tee logs/step3_obs.log

# Step 4: Train HistoryEncoder
$RUN train-history --config-file $CFG 2>&1 | tee logs/step4_history.log

# Step 5: Train Commander (Behavioral Cloning)
$RUN train-command --config-file $CFG 2>&1 | tee logs/step5_command.log

# Step 6: DAgger (iterative refinement) + auto-simulate for videos
$RUN dagger --config-file $CFG --run-simulate 2>&1 | tee logs/step6_dagger.log
```

W&B logging is **enabled by default** for `dagger`. Disable with `--no-use-wandb`.

### Using a pre-trained BC model (skip steps 1-5)

If you already have a trained BC model in another cohort, set `bc_cohort` in your config:

```yaml
bc_cohort: "ssv_CLIPSEG_NORMAL"   # Copy model + symlink observations from here
```

Then run only step 6. DAgger copies the model and symlinks observation data automatically.

---

## Pipeline Steps in Detail

### Step 1-2: Generate Rollouts

An MPC expert controller (VehicleRateMPC, using ACADOS optimal control) flies RRT*-planned trajectories through the 3D Gaussian Splat environment.

#### How RRT* trajectory branches are created

For each target object, RRT* builds a random exploration tree through the obstacle field:
1. **RRT* sampling**: 2500 random nodes are placed in the 2D environment bounds `[-3.7,3.7] × [-8,8]`, connected while avoiding obstacles. RRT* rewires connections to shorten paths.
2. **Branch extraction**: From the tree, `nbranches=110` distinct paths are extracted per object — each is a different collision-free route from the start region to the target.
3. **Parameterization**: Each 2D path is converted to a smooth 3D trajectory at the configured altitude (-1.0m), producing an 18×N `tXUi` array with time, state, and control columns.

**No explicit seed** — RRT* is inherently stochastic (random node placement), so each `generate-rollouts` call produces different branches.

Result: **110 trajectory branches per object** × 3 objects = **330 reference trajectories**.

#### How BC perturbations work (during rollout generation)

For each of the 110 trajectories, the expert flies **multiple perturbed rollouts**:

```
For each trajectory branch (110 per object):
    Ntp = rate(10) × duration(2.0) = 20 time points per trajectory
    For each of reps=4 repetitions:
        For each of the 20 time points:
            1. Pick a reference state at that time on the trajectory
            2. ADD PERTURBATION (uniform random noise):
                  position (x,y,z):    ±0.40 m
                  velocity (vx,vy,vz): ±0.40 m/s
                  quaternion (4 vals):  ±0.20
            3. Start MPC expert from this perturbed state
            4. Expert flies for 2.0s back toward the reference path
            5. Record state + images at each 20Hz timestep
```

This means: 110 branches × 4 reps × 20 time points = **8,800 short rollouts per object**,
each starting from a different perturbed state along the trajectory.

**Why perturb in BC?** The expert always flies perfectly on the reference path. Without perturbation, the neural pilot would only learn "what to do on the path" but never "how to recover when off the path." The perturbations force the expert to demonstrate recovery behavior.

**tXUi format** (the RRT reference trajectory, 18×N array):
```
Row 0:      time vector
Rows 1-10:  state [x, y, z, vx, vy, vz, qx, qy, qz, qw]
Rows 11-13: angular velocity [wx, wy, wz]
Row 14:     unused
Rows 15-17: motor commands [m1, m2, m3, m4]
```

**Output**: `cohorts/{cohort}/rollout_data/{course}/`
- `trajectories_*.pt` — Time (T), state (X), control (U) arrays
- `imgdata_*.pt` — Camera images at each timestep
- `video_*.mp4` — Rendered flight videos

Step 2 generates a validation split (same process, fewer reps).

### Step 3: Generate Observations

For each timestep of each rollout, runs the pilot's OODA loop to extract training samples.

**What this does concretely**: Takes the expert's rollout data (state + images at each timestep) and processes it through the pilot's perception and feature extraction pipeline. The output is what the neural network would "see" as input (Xnn) paired with what the expert commanded as output (Ynn).

**Output**: `cohorts/{cohort}/observation_data/{pilot}/{course}/observations*.pt`

Each `.pt` file = **one trajectory branch's observations** (all perturbed rollouts on that branch combined):
```python
{
    "data": [{
        "Xnn": [xnn_dict_0, xnn_dict_1, ...],   # ~160 NN inputs (one per timestep)
        "Ynn": [ynn_dict_0, ynn_dict_1, ...],    # ~160 expert labels
        "Ndata": int,                              # Number of timesteps
        "rollout_id": int,
        "course": str,
    }],
    "set": "", "Nobs": int, "course": str
}
```

With 110 branches × 3 objects = ~330 observation files, each with ~160 samples = **~52,800 total training samples**.

### Step 4: Train HistoryEncoder

Trains the temporal feature extractor to compress the rolling history buffer (how the drone has been moving over the last 20 timesteps) into useful features.

- **Loss**: MSE between predicted and target history encodings
- **Epochs**: `Nep_his` (default: 100)
- **What freezes**: Only the HistoryEncoder parameters are trainable; Commander is frozen.

### Step 5: Train Commander (Behavioral Cloning)

Trains the command network to predict expert actions from observations.

**The training loop (per epoch)**:
```
For each observation file (train split):
    dataset = load observations from .pt file
    For each batch (size=64, shuffled):
        prediction = model(xnn)            # pilot predicts [thrust, wx, wy, wz]
        label = ynn["unn"]                 # expert's action [thrust, wx, wy, wz]
        loss = MSELoss(prediction, label)  # mean squared error
        loss.backward() → optimizer.step() → optimizer.zero_grad()

For each observation file (test split):
    Same loop but NO gradient update (evaluation only)
```

- **Train loss**: MSE averaged over all training observation files
- **Test loss**: MSE on the held-out test file (no gradient)
- **Optimizer**: Adam with lr=1e-4, batch_size=64
- **Epochs**: `Nep_com` (default: 150)
- **Data split**: All observation `.pt` files are sorted alphabetically.
  `all_files[:-1]` → train (~329 files), `all_files[-1]` → test (1 file).
  The "held-out test file" is simply the last file in alphabetical order.
  During DAgger, the same split applies: BC files + `observations_dagger.pt`,
  and the last file in the combined sorted list becomes the test file.

---

## DAgger Deep Dive

### Why DAgger is needed

After BC training, the pilot can fly reasonably well on the expert's trajectories. But in deployment, the pilot makes small errors that compound. DAgger addresses this by:

1. Flying the pilot (with some expert help) to encounter realistic error states
2. Asking the expert "what would you do here?" at each state
3. Retraining the pilot on these corrections

### The DAgger Observation File (`observations_dagger.pt`)

**Where**: `cohorts/{cohort}/observation_data/{pilot}/dagger/observations_dagger.pt`

**When generated**: At the start of each DAgger iteration's retraining phase, inside `_retrain_commander()`.

**How generated**:
1. MixedPolicy flies a trajectory → collects raw annotations at each timestep
2. Annotations are filtered (see annotation filtering below)
3. Filtered annotations are converted to BC observation format:
   ```python
   For each annotation {xnn, x, u_expert}:
       Xnn.append(xnn)           # Same format as BC observations
       Ynn.append({
           "unn": u_expert,      # Expert's action at this state
           "mfn": [0.3, 0.3],    # Default mass/force
           "onn": x_state        # Drone state at this timestep
       })
   ```
4. Saved as a standard observation .pt file in `observation_data/{pilot}/dagger/`

**What it contains**: The same format as BC observations, but the states come from the pilot's actual flight path (with expert corrections), not from the expert's original rollouts.

### What happens during a DAgger Iteration

**Per iteration: 1 full-trajectory rollout per object** (3 rollouts total with 3 objects).
Unlike BC (which flies thousands of short 2s segments), DAgger flies the **entire trajectory
from start to goal**, recording annotations at every 20Hz timestep.

```
ITERATION i:
│
├── 1. RESET TO BEST MODEL (if i > 0)
│   Load model_best_staging.pth → pilot, restore to model.pth
│
├── 2. COLLECT DATA (1 rollout per object)
│   For each object (clock, leafblower, drill):
│   │   Build fresh RRT trajectory from (perturbed) start → goal
│   │   Perturb start position by ±start_pos_noise (position only, no vel/quat)
│   │   Build MPC expert tracking this RRT reference
│   │   MixedPolicy = β·expert + (1-β)·pilot
│   │   Run simulator on FULL trajectory (3-10s, ~60-190 timesteps):
│   │     At each 20Hz timestep:
│   │       u_expert = MPC.control(state)      # Expert's command
│   │       u_pilot, xnn = Pilot.OODA(state)   # Pilot's command + features
│   │       Record annotation: {xnn, state, u_expert}
│   │       Execute: u_expert (prob β) or u_pilot (prob 1-β)
│   │
│   └── Filter annotations (~60-180 raw → ~100-300 kept per iteration)
│
├── 3. RETRAIN COMMANDER
│   │   Restore best model to model.pth
│   │   Convert annotations → observations_dagger.pt
│   │   train_roster() trains on:
│   │     - ALL BC observation files (~330 files, ~52,800 samples)
│   │     - 1 DAgger file (~300 samples, ~0.6% of total)
│   │   For Nep_dagger epochs:
│   │     For each file:
│   │       loss = MSE(pilot(xnn), expert_action)
│   │       loss.backward()
│   │
│   └── Result: updated model.pth
│
├── 4. EVALUATE
│   Run pilot on 2nd half of tXUi (unseen during BC)
│   Measure: success_rate, collision_rate, goal_distance per object
│   If better → save as new best → model.pth
│   If worse → restore previous best → model.pth
│
└── 5. EARLY STOPPING
    If 2 consecutive iterations without improvement → stop
```

### DAgger Train/Test Loss

During DAgger retraining, the train and test loss are computed the same way as BC:

```
train_loss = MSE(pilot_prediction, expert_action)
             averaged over ALL training files (BC + DAgger)

test_loss  = MSE(pilot_prediction, expert_action)
             on the held-out test file (no gradient)
```

The DAgger file is ~0.6% of total training data (300 vs 52,800 samples). So the loss is dominated by BC data. The DAgger samples have higher individual loss (novel states the model hasn't seen) and contribute disproportionate gradient signal. This is by design — it's a small, targeted correction.

### The MPC Expert during DAgger

The MPC expert (VehicleRateMPC) uses the pre-computed **RRT reference trajectory** (`tXUi`) stored in `configs/scenes/flightroom_ssv_exp_{object}.pkl`.

The expert receives:
- The full reference trajectory to track
- The drone's current state from the simulator

It uses ACADOS (an optimal control solver) to compute: "given where I am now and where the trajectory says I should be, what control command minimizes tracking error?" Output: `[thrust, wx, wy, wz]`.

The expert doesn't "think" or "plan" in real-time — it simply tracks a pre-planned path. This is why DAgger works: the expert can provide corrections for ANY drone state by computing the optimal action to return to the reference path.

### Annotation Filtering

Not all timesteps are useful for training. The filtering logic per timestep:

```
1. HARD CUTOFF at first collision step
   (post-crash physics are garbage — drone tumbles uncontrollably)

2. SKIP if deviation > max_deviation_dist (8.0m)
   (drone flew into ceiling or out of scene — extreme excursion)

3. SKIP if goal_dist > max_annotation_goal_dist (50.0m)
   (drone is hopelessly lost — these labels won't help)

4. ALWAYS KEEP if goal_dist < close_approach_dist (5.0m)
   (final approach is the hardest part — needs lots of data)

5. KEEP if deviation > deviation_filter_dist (0.3m)
   (pilot deviated from reference — this is where it needs help)

6. SKIP otherwise
   (on-trajectory, BC already handles this well)
```

### How Evaluation and Benchmarking Work

Both per-iteration evaluation and the final benchmark use the **same BC reference trajectories** (`tXUi`) — they do NOT generate new RRT paths.

#### Per-iteration evaluation (`_eval_full_trajectories`)

After each DAgger iteration, the pilot is tested:
```
For each object:
    tXUi = load pre-computed BC reference trajectory
    Sample n_eval_per_iter=20 start indices from the 2ND HALF of tXUi
       (the 1st half was used during BC training → 2nd half is held-out)
    Seed with eval_seed=42 → same 20 start points every iteration
    For each start index:
        Extract state at that point on the reference trajectory
        Pilot flies alone (NO expert, NO perturbation) from that state to goal
        Measure: goal_distance, collision (yes/no)
        Success = goal_dist < 2.0m AND no collision
```

**Why seed=42?** It ensures every iteration is evaluated on the **identical 20 start positions**. Without this, random variation in start positions would make it impossible to tell if the model actually improved or just got easier starts. The pilot never sees these exact states during DAgger data collection (which uses random perturbations without this seed).

#### Final benchmark (`_run_benchmark_pilot`)

Run twice — once with the BC model ("before") and once with the DAgger model ("after"):
```
For each object:
    tXUi = same BC reference trajectory
    Seed with benchmark_seed=42
    Sample n_benchmark=50 start indices from the 2ND HALF of tXUi
    For each start:
        Pilot flies alone from that state to goal
        Same success criteria as per-iter eval
```

The seed ensures before/after use **identical start conditions** for a fair comparison.

#### Why this is NOT "cheating"

- **DAgger rollout data** uses `start_pos_noise` (random ±0.5m perturbation, no seed) on **full trajectories** with mixed expert/pilot control
- **Evaluation** uses fixed seeds on the **2nd half of tXUi** (held out from BC training) with **pilot-only** control
- The two distributions are different — the pilot must genuinely generalize, not memorize
- The fixed seed only ensures reproducibility across iterations, not overlap with training data

### DAgger vs BC Perturbation Summary

| | BC Rollouts | DAgger Rollouts | Evaluation/Benchmark |
|---|---|---|---|
| **What's perturbed** | Position ±0.4m, velocity ±0.4m/s, quaternion ±0.2 | Position only ±0.5m | Nothing (fixed start indices) |
| **Trajectory source** | 110 RRT branches (pre-computed) | 1 fresh RRT per object per iter | Same tXUi from BC (2nd half) |
| **Who flies** | MPC expert only | β·expert + (1-β)·pilot | Pilot only |
| **Duration** | 2.0s segments at sampled time points | Full trajectory (3-10s) | Full trajectory from sampled start |
| **Runs per object** | ~8,800 (110 × 4 reps × 20 time pts) | 1 per iteration | 20 (eval) or 50 (benchmark) |
| **Seed** | None (global random) | None (random each iter) | 42 (fixed for reproducibility) |

---

## Configuration Reference

### Experiment Config (`configs/experiment/*.yml`)

```yaml
# ─── Identity ────────────────────────────────────────
cohort: "SSV_DAGGER_FULL_CAMPAIGN"
  # Unique name for this experiment. All data saved under cohorts/{cohort}/

method: "rrt"
  # Trajectory planning method. "rrt" uses RRT* for path planning.

review: false
  # false = run the pipeline. true = review/analyze existing results.

# ─── BC Training ─────────────────────────────────────
Nep_his: 100
  # Number of epochs for HistoryEncoder training (Step 4).
  # Higher = better temporal feature extraction, but diminishing returns past ~100.

Nep_com: 150
  # Number of epochs for Commander (BC) training (Step 5).
  # The model needs enough epochs to converge on the expert's behavior.
  # Original ssv_CLIPSEG_NORMAL used 150 + cumulative DAgger retrain = ~400 total.

# ─── BC Shortcut ─────────────────────────────────────
bc_cohort: "ssv_CLIPSEG_NORMAL"
  # If set, DAgger copies the model from this cohort and symlinks its observation
  # data. Allows skipping steps 1-5 when a good BC model already exists.
  # If null/omitted, steps 1-5 must produce the model in this cohort.

# ─── DAgger Core ─────────────────────────────────────
expert_type: "mpc"
  # Type of expert controller for DAgger.
  # "mpc" = VehicleRateMPC (ACADOS optimal control tracking RRT reference)
  # This expert can provide corrections for any drone state.

n_iterations: 8
  # Maximum number of DAgger iterations.
  # Each iteration: collect data → retrain → evaluate.
  # Early stopping may terminate before reaching this limit.

Nep_dagger: 8
  # Training epochs per DAgger iteration.
  # Kept small (8 vs 150 for BC) because DAgger is a gentle correction,
  # not a full retrain. Too many epochs → overfits to DAgger data.

n_eval_per_iter: 20
  # Number of trajectories per object for quick per-iteration evaluation.
  # Used to track progress and select the best model.

n_benchmark: 50
  # Number of trajectories per object for the final before/after benchmark.
  # Higher = more statistically robust comparison.

lim_sv: 5
  # Save model checkpoint every N epochs during retraining.

dagger_lr: 0.00003
  # Learning rate for DAgger retraining.
  # Lower than BC's 1e-4 to avoid destroying what BC learned.

# ─── DAgger Strategy ─────────────────────────────────
aggregate_dagger: false
  # false (ONLINE/RECOMMENDED): Train on BC + THIS iteration's annotations only.
  #   Previous DAgger annotations are discarded each iteration.
  #   Pros: Fresh data each time, no stale annotations.
  #   Combined with reset_to_best, each iteration is an independent correction attempt.
  #
  # true (CUMULATIVE): Accumulate ALL past DAgger annotations.
  #   Train on BC + annotations from iter 0,1,...,N.
  #   Pros: More DAgger data. Cons: Old annotations become stale/redundant.
  #   Can cause data dilution and training instability.

reset_to_best: true
  # Before each iteration:
  #   1. Load the best model (by success_rate) for rollout collection
  #   2. Restore best model to model.pth before retraining
  # WHY: Prevents catastrophic cascade — if iteration 2 degrades the model,
  # iteration 3 starts from the best (iter 0 or 1), not the degraded version.
  # Without this, one bad iteration can destroy all progress.

eval_seed: 42
  # Fixed random seed for evaluation start positions.
  # Ensures before/after benchmarks use IDENTICAL start conditions.
  # Without this, random variation can mask real improvements.

# ─── DAgger Data Collection ──────────────────────────
start_pos_noise: 0.5
  # Random perturbation (±0.5m in each axis) added to trajectory start position.
  # POSITION ONLY — no velocity or quaternion noise (would make MPC solver diverge).
  # Creates diversity — without this, every iteration flies the same path
  # and DAgger can't discover new failure modes.
  # NOTE: Currently 1 rollout per object per iteration (hardcoded in the DAgger loop).
  # Total DAgger rollouts = n_iterations × n_objects (e.g., 8 × 3 = 24).

# ─── Annotation Filtering ────────────────────────────
# After MixedPolicy flies, we filter annotations to keep only useful ones:

deviation_filter_dist: 0.3
  # Keep annotation if pilot deviated MORE than 0.3m from reference trajectory.
  # These are states where the pilot was wrong → exactly what DAgger should fix.
  # Smaller value = keep more annotations (more conservative corrections).

close_approach_dist: 5.0
  # ALWAYS keep annotations when drone is within 5m of goal.
  # Final approach is the hardest/most important part of navigation.
  # These annotations are kept regardless of deviation.

max_annotation_goal_dist: 50.0
  # DISCARD annotations where drone is more than 50m from goal.
  # The drone is completely lost — expert corrections for these states
  # are meaningless (the model can't recover from 50m away).

max_deviation_dist: 8.0
  # DISCARD annotations where drone deviated more than 8m from reference.
  # Extreme excursions (flew into ceiling, out of bounds) produce
  # unrealistic physics and garbage training data.

# ─── Scene + Pilot ───────────────────────────────────
flights:
  - ["flightroom_ssv_exp", "flightroom_ssv_exp"]
  # List of [scene_name, course_name] pairs.
  # The scene defines the 3D environment and target objects.

roster:
  - "InstinctJester"
  # List of pilot names to train. InstinctJester is the main deep-learned pilot.
```

### Scene Config (`configs/scenes/*.yml`)

```yaml
queries: ["green clock", "green and pink leafblower", "yellow handheld cordless drill on two boxes"]
  # Natural language descriptions of target objects.
  # CLIPSeg uses these to locate objects in the Gaussian Splat scene.

radii:
  - [2.0, 0.4]    # [outer_radius, inner_radius] per object
altitudes:
  - -1.0           # Flight height per object (negative = above ground)
similarities:
  - [0.90, 0.025]  # CLIPSeg [threshold, margin] per object
nbranches:
  - 110             # Number of RRT branches per object
algorithm: 'RRT*'
N: 2500             # RRT iterations
step_size: 1.0
minbound: [-3.7, -8, -2]  # Environment bounds [x, y, z]
maxbound: [3.7, 8, 0]
```

---

## Deploying & Visualizing Results

### Simulate a trained model (generates videos)

Run the best model on RRT trajectories and record MP4 videos:

```bash
conda run --no-capture-output -n FiGS python notebooks/ssv_muilti3dgs_campaign_coruscant.py \
    simulate --config-file configs/experiment/ssv_dagger_full_campaign.yml
```

Or use `--run-simulate` with the `dagger` command to auto-run after DAgger completes.

**Output**:
```
cohorts/{cohort}/simulation_data/{timestamp}/
├── rrt_planning/       # RRT trees and filtered paths (.pkl)
├── trajectories/       # Simulation state data (.pt)
└── videos/             # MP4 videos at 20 FPS
    ├── sim_video_{scene}_{object}_{pilot}_semantic.mp4
    └── sim_video_{scene}_{object}_{pilot}_rgb.mp4
```

**What the simulate command does**:
1. Loads the Gaussian Splat scene
2. Plans RRT trajectories (same as in BC)
3. Runs both the MPC expert AND the neural pilot on each trajectory
4. Renders camera views and saves as MP4 videos

### Cross-benchmark multiple models

```bash
conda run --no-capture-output -n FiGS python notebooks/ssv_muilti3dgs_campaign_coruscant.py \
    cross-benchmark \
    --config-file configs/experiment/ssv_dagger_full_campaign.yml \
    --cohort-before ssv_CLIPSEG_NORMAL \
    --cohort-rrt SSV_DAGGER_FULL_CAMPAIGN \
    --max-trajectories 50 --benchmark-seed 123
```

---

## Where to Find Results

| What | Location |
|------|----------|
| Training logs | `logs/step{1-6}_*.log` |
| **Deploy-ready model** | `cohorts/{cohort}/roster/{pilot}/model.pth` (always the best) |
| Pre-DAgger BC model | `cohorts/{cohort}/roster/{pilot}/model_before_dagger.pth` |
| Archival best model | `cohorts/{cohort}/dagger_data/{pilot}/benchmark_*/model_best_dagger.pth` |
| Benchmark JSON | `cohorts/{cohort}/dagger_data/{pilot}/benchmark_*/benchmark_results.json` |
| DAgger annotations | `cohorts/{cohort}/dagger_data/{pilot}/dagger_iter_*.pt` |
| DAgger observations | `cohorts/{cohort}/observation_data/{pilot}/dagger/observations_dagger.pt` |
| BC observations | `cohorts/{cohort}/observation_data/{pilot}/{course}/observations*.pt` |
| Simulation videos | `cohorts/{cohort}/simulation_data/{timestamp}/videos/*.mp4` |
| Trajectory plots | `cohorts/{cohort}/dagger_data/{pilot}/plots/*.png` |
| W&B dashboard | `https://wandb.ai/{team}/singer-dagger` |

**Model management**: `model.pth` is always the current best model. When DAgger starts, the BC model is backed up as `model_before_dagger.pth`. After each iteration, if the retrained model improves, it becomes the new `model.pth`. If not, the previous best is restored. `simulate` always uses the best model without manual copying.

---

## Project Structure

```
SINGER/
├── ssv_muilti3dgs_campaign.py          # Root entry point
├── notebooks/
│   └── ssv_muilti3dgs_campaign_coruscant.py  # CLI (Typer) — all commands
│
├── configs/
│   ├── experiment/                     # Experiment configs
│   │   ├── ssv_multi3dgs.yml           # BC-only baseline config
│   │   └── ssv_dagger_full_campaign.yml # Full BC + DAgger config
│   ├── method/rrt.json                 # Method config (frame=carl, policy=vrmpc_rrt)
│   ├── frame/carl.json                 # Vehicle frame parameters
│   ├── policy/vrmpc_rrt.json           # MPC policy parameters
│   └── scenes/                         # Scene configs + RRT trajectory .pkl files
│       ├── flightroom_ssv_exp.yml      # Flight room scene definition
│       └── flightroom_ssv_exp_*.pkl    # Pre-computed RRT trajectories per object
│
├── src/sousvide/
│   ├── control/pilot.py                # Pilot class (OODA loop, NN inference)
│   ├── instruct/
│   │   ├── train_policy.py             # train_roster(), train_student() — training loop
│   │   ├── train_dagger.py             # DAgger: MixedPolicy, annotation, retraining
│   │   └── synthesized_data.py         # Dataset loading, observation file discovery
│   ├── synthesize/
│   │   ├── rollout_generator.py        # Step 1-2: MPC expert rollout generation
│   │   └── observation_generator.py    # Step 3: OODA observation extraction
│   └── flight/
│       └── deploy_ssv.py              # Simulation + video recording
│
├── cohorts/                            # All experiment data
│   └── {cohort}/
│       ├── rollout_data/{course}/      # Step 1-2 outputs
│       ├── observation_data/{pilot}/   # Step 3 + DAgger observations
│       ├── roster/{pilot}/             # Model files
│       ├── dagger_data/{pilot}/        # DAgger iteration data + benchmarks
│       └── simulation_data/            # Simulate command outputs
├── logs/                               # Training logs
└── docs/                               # Architecture documentation
```

---

## Troubleshooting

### ACADOS errors / solver=None

```bash
export ACADOS_SOURCE_DIR=/path/to/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/FiGS-Standalone/acados/lib
```

### DAgger not improving

- Check that BC model has reasonable loss (< 0.01 train). DAgger refines, it can't fix a broken BC.
- Verify `aggregate_dagger: false` + `reset_to_best: true` + `eval_seed: 42`.
- Look at per-object metrics — some objects may improve while others regress.
- Check W&B dashboard for per-iteration trends.

### Training loss slightly increases during DAgger retraining

This is **normal and expected**. The DAgger observation file contains novel states the model hasn't seen during BC training. These have higher loss. The overall loss (dominated by BC data, ~99.4%) fluctuates slightly but stays close to the BC baseline. What matters is the simulation metrics (success_rate, collision_rate), not the training loss.

### Out of GPU memory

- The Gaussian Splat renderer uses significant VRAM
- Ensure only one training process per GPU
- Reduce `n_eval_per_iter` or `n_benchmark` if needed

### Pipeline step fails

Each step depends on the previous. Check `logs/step{N}_*.log` and verify the cohort directory has expected data.
