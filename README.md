# SINGER: Onboard Vision-Language Navigation Policy for Drones

<div align="center">
  <video src="assets/leafblower_rgb.mp4" width="400" autoplay loop muted playsinline></video>
  <video src="assets/leafblower_semantic.mp4" width="400" autoplay loop muted playsinline></video>
  <p><em>"Go to the green leafblower" — RGB (left) and CLIPSeg semantic similarity (right).</em><br>
  <em>The drone navigates autonomously using only its onboard camera.</em><br>
  <a href="https://youtu.be/R6zd46fFNQ0">Full demo video on YouTube</a></p>
</div>

> **Given a natural language instruction like "go to the green leafblower," the drone autonomously identifies and navigates to the target — collision-free.**
>
> The video shows the drone's onboard view: RGB (left) and CLIPSeg semantic similarity field (right). The system encodes the language instruction via CLIP, localizes the target using CLIPSeg segmentation, and outputs real-time control commands to navigate through a cluttered indoor 3D Gaussian Splatting environment while avoiding obstacles.
>
> The control policy is a lightweight neural network trained via Behavioral Cloning from an ACADOS MPC expert, refined with a full **DAgger pipeline** (mixed-policy rollouts, expert annotation filtering, iterative retraining with best-model checkpointing). A key contribution is the introduction of explicit **geometric centroid features** (bearing + elevation) extracted from the CLIPSeg heatmap, providing the policy with a direct spatial signal for goal-directed control.
>
> **Results: 88% success rate (up from 52%), collision rate reduced from 20% to 8%, with generalization to unseen trajectories (91%).**
>
> *Tech stack: PyTorch, 3D Gaussian Splatting (gsplat), CLIPSeg, ACADOS optimal control, CUDA*

---

## Installation

SINGER requires [FiGS-Standalone](https://github.com/StanfordMSL/FiGS-Standalone) as its simulator/renderer.

### Docker

```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
git clone https://github.com/StanfordMSL/SINGER.git
cd FiGS-Standalone && docker-compose build
cd ../SINGER && docker-compose run singer
```

### Conda (coruscant)

```bash
conda activate FiGS
export ACADOS_SOURCE_DIR=/path/to/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/FiGS-Standalone/acados/lib
cd /path/to/SINGER
```

---

## Architecture

### Neural Pilot — OODA Loop (20 Hz)

At each control step, the pilot receives **only**:

| Input | Source | Description |
|-------|--------|-------------|
| `tx_com` | IMU/state | Time + position + velocity + quaternion (10D) |
| `obj_com` | **Vision inference** | Bearing, elevation, apparent size — computed from CLIPSeg heatmap on the RGB image |
| `dxu_par` | Flight history | Rolling 20-step delta buffer (how the drone has been moving) |
| `img_vis` | Onboard camera | Processed RGB image (CLIPSeg embeddings) |

The pilot does **NOT** receive: object 3D position, goal coordinates, expert trajectory, waypoints, or any ground truth.

```
Pilot.OODA(upr, tcr, xcr, obj, icr, zcr):
  observe() → process image → _compute_centroid(image) → [bearing, elevation, size]
  orient()  → update DxU history buffer (11x20)
  decide()  → model.extract_inputs(state, centroid, image, history)
  act()     → Commander(inputs) → [thrust, wx, wy, wz]
```

### Two-Stage Training

```
  BC (Behavioral Cloning)                    DAgger (Dataset Aggregation)
  ┌────────────────────────────┐             ┌──────────────────────────────────────┐
  │ 1. MPC expert flies 110    │             │ For each iteration:                  │
  │    RRT branches per object │             │   1. Mixed policy flies trajectories │
  │ 2. Extract (Xnn, Ynn) obs │──model.pth──│   2. Record expert corrections       │
  │ 3. Train HistoryEncoder    │             │   3. Retrain Commander on BC+DAgger  │
  │ 4. Train Commander (MSE)   │             │   4. Evaluate → keep best model      │
  └────────────────────────────┘             └──────────────────────────────────────┘
```

### Network Architecture

- **HistoryEncoder**: Compresses temporal sequences (DxU 11x20) into feature vectors. Trained once during BC (Step 4), frozen during DAgger.
- **VisionMLP**: Processes image features. Frozen during DAgger retraining.
- **CommanderSV**: Takes state + centroid + image + history features → outputs `[thrust, wx, wy, wz]`. This is the only component fine-tuned by DAgger.

### Project Structure

```
SINGER/
├── ssv_muilti3dgs_campaign.py              # Root entry point
├── notebooks/
│   └── ssv_muilti3dgs_campaign_coruscant.py  # CLI (Typer) — all commands
├── configs/
│   ├── experiment/                         # Experiment configs (.yml)
│   ├── pilots/InstinctJester.json          # Pilot config (centroid_version, network indices)
│   ├── scenes/                             # Scene configs + RRT trajectory caches
│   └── method/rrt.json                     # Method config (frame=carl, policy=vrmpc_rrt)
├── src/sousvide/
│   ├── control/pilot.py                    # Pilot OODA loop, centroid computation
│   ├── control/policies/svnet.py           # Network forward pass, input extraction
│   ├── control/policies/ComponentNetworks.py # CommanderSV, HistoryEncoder definitions
│   ├── instruct/train_dagger.py            # DAgger engine, benchmark, evaluation
│   ├── instruct/train_policy.py            # train_roster(), training loop
│   ├── instruct/synthesized_data.py        # Dataset loading, observation file discovery
│   ├── synthesize/rollout_generator.py     # MPC expert rollout generation
│   ├── synthesize/observation_generator.py # OODA observation extraction
│   └── flight/deploy_ssv.py               # Simulation + video recording
├── scripts/
│   ├── benchmark_seen_unseen.py            # Seen/unseen generalization benchmark
│   └── compare_trajectories_3d.py          # Interactive 3D Plotly trajectory visualization
├── cohorts/                                # All experiment data
│   └── {cohort}/
│       ├── rollout_data/                   # Expert rollouts (T, X, U, images, video)
│       ├── observation_data/{pilot}/       # Training observations (Xnn, Ynn pairs)
│       ├── roster/{pilot}/model.pth        # Best trained model
│       ├── dagger_data/{pilot}/            # DAgger iteration data (annotations, obs)
│       ├── training_benchmarks/{timestamp}/ # Benchmarks during DAgger training
│       │   ├── benchmark_results.json      #   Before/after metrics
│       │   ├── model_before_dagger.pth     #   BC model snapshot
│       │   └── model_best_dagger.pth       #   Best DAgger model snapshot
│       ├── post_training_benchmarks/{timestamp}/ # Post-training benchmarks
│       │   ├── results_*.json              #   Raw metrics
│       │   ├── summary_*.txt               #   Human-readable summary
│       │   ├── plots/*.html                #   Interactive 3D Plotly per object
│       │   └── videos/*.mp4                #   Per-run onboard camera (if enabled)
│       ├── simulation_data/                # Simulate command videos
│       └── visualizations/                 # compare_trajectories_3d output
├── docs/                                   # Detailed technical reference (gitignored)
└── logs/                                   # Training logs
```

---

## Full Pipeline

```bash
export CUDA_VISIBLE_DEVICES=0
CFG=configs/experiment/ssv_dagger_centroid_v9.yml
RUN="conda run --no-capture-output -n FiGS python ssv_muilti3dgs_campaign.py"

# Step 1: Generate expert rollouts (MPC flies 110 RRT branches per object)
$RUN generate-rollouts --config-file $CFG

# Step 2: Generate validation rollouts (11 held-out branches per object)
$RUN generate-rollouts --config-file $CFG --validation-mode

# Step 3: Generate observations (extract Xnn/Ynn pairs from rollouts)
$RUN generate-observations --config-file $CFG

# Step 4: Train HistoryEncoder (temporal feature extraction)
$RUN train-history --config-file $CFG

# Step 5: Train Commander — Behavioral Cloning
$RUN train-command --config-file $CFG

# Step 6: DAgger (iterative refinement)
$RUN dagger --config-file $CFG
```

### Skip BC (use pre-trained model)

Set `bc_cohort` in your experiment config to copy a model from another cohort, then run only Step 6:

```yaml
bc_cohort: "ssv_BC_6S"  # Copies model + symlinks observation data
```

### Simulate (generate videos)

```bash
$RUN simulate --config-file $CFG
# Or auto-simulate after DAgger:
$RUN dagger --config-file $CFG --run-simulate
```

### Seen/Unseen Benchmark

```bash
conda run --no-capture-output -n FiGS python scripts/benchmark_seen_unseen.py
```

---

## Training Principles

### Behavioral Cloning

- **110 RRT branches per object**, each flown with **4 perturbed repetitions** (position ±0.4m, velocity ±0.4m/s, quaternion ±0.2)
- Perturbations teach recovery behavior — without them, the pilot only learns on-trajectory actions
- ~52,800 training samples across 330 observation files
- HistoryEncoder trained first (100 epochs), then Commander (150 epochs, MSE loss, Adam lr=1e-4)

### DAgger

- Solves **distribution shift**: BC trains on expert states, but at deploy time the pilot visits novel states where it has no training data
- Each iteration: fly mixed policy (β·expert + (1-β)·pilot) → record expert corrections at pilot-visited states → retrain Commander
- **Only CommanderSV is retrained** — VisionMLP and HistoryEncoder stay frozen
- **Annotation filtering**: keep deviations > 0.3m from reference, always keep within 5m of goal, discard extreme excursions (> 8m deviation or > 50m from goal)
- **`reset_to_best: true`**: Each iteration starts from the best model so far, preventing catastrophic cascading from one bad iteration
- **`aggregate_dagger: false`** (recommended): Train on BC + current iteration's annotations only, not accumulated stale data

### Centroid Features (V9)

The pilot locates objects via CLIPSeg semantic segmentation on the onboard RGB image:

```
RGB image → CLIPSeg similarity heatmap → threshold at percentile(75)
  → weighted centroid of top-25% pixels
  → bearing [-1,1], elevation [-1,1], apparent_size [0,1]
```

These are the **only goal-related inputs** to the network. No ground truth position is ever provided. The centroid version must match the training version (`centroid_version: "v9"` in pilot config).

### What the Model Does NOT Have Access To

- Object 3D position or distance
- Goal coordinates or waypoints
- Expert trajectory at test time
- Collision proximity information
- Any oracle or ground truth signal

The model must learn to navigate purely from: what it sees (RGB image → centroid), where it is (IMU state), and how it's been flying (history buffer).

---

## Benchmark Results (V9 DAgger)

### Training Benchmark (50 runs/object, seed=42)

| Phase | Clock | Leafblower | Boxes | **Avg Success** | **Collision** |
|-------|-------|------------|-------|-----------------|---------------|
| BC baseline | 86% | 76% | 80% | **80.7%** | 13.3% |
| **After DAgger** | **90%** | **84%** | **90%** | **88.0%** | **8.0%** |

### Seen vs Unseen Generalization (seed=42)

| Set | Model | Clock | Leafblower | Boxes | **Avg Success** | **Collision** |
|-----|-------|-------|------------|-------|-----------------|---------------|
| Seen (50/obj) | BC | 82% | 76% | 74% | 77.3% | 16.0% |
| Seen (50/obj) | DAgger | 90% | 84% | 84% | **86.0%** | 8.0% |
| Unseen (11/obj) | BC | 91% | 73% | 73% | 78.8% | 15.2% |
| Unseen (11/obj) | DAgger | **100%** | 91% | 82% | **90.9%** | 9.1% |

**Key findings:**
- DAgger improves over BC by +8.7pp (seen) and +12.1pp (unseen)
- Collision rate halved across the board
- **Unseen > Seen** (90.9% vs 86.0%) — the model generalizes, it does not memorize training paths
- Clock is effectively solved (100% on unseen)

### What "Seen" vs "Unseen" Means

- **Seen**: 50 branches sampled from the 110 RRT paths used during BC training. The model was trained on expert demonstrations along these paths.
- **Unseen**: All 11 validation branches per object, generated separately via `generate-rollouts --validation-mode`. Different RRT random trees, different waypoints, different starting positions. Never used in any training.
- Trajectories are **not regenerated** at benchmark time — they are loaded from disk
- Fixed seed ensures BC and DAgger are evaluated on **identical branches** for fair comparison

### Current Model

| File | Description |
|------|-------------|
| `cohorts/SSV_DAGGER_CENTROID_V9/roster/InstinctJester/model.pth` | Best model (DAgger V9) |
| `cohorts/SSV_DAGGER_CENTROID_V9/roster/InstinctJester/model_before_dagger.pth` | BC-only baseline |
| `configs/pilots/InstinctJester.json` | Pilot config (`centroid_version: "v9"`) |
| `configs/experiment/ssv_dagger_centroid_v9.yml` | V9 experiment config |

---

## Visualizations

### Simulation Videos

Best model (V9 DAgger) flying to each object — RGB, depth, and semantic views:

```
cohorts/SSV_DAGGER_CENTROID_V9/simulation_data/20260330_220330/videos/
  sim_video_..._green clock_InstinctJester_rgb.mp4
  sim_video_..._green and pink leafblower_InstinctJester_rgb.mp4
  sim_video_..._yellow handheld cordless drill on two boxes_InstinctJester_rgb.mp4
```

18 videos total: 3 objects x 2 pilots (expert + InstinctJester) x 3 render modes (rgb, depth, semantic).

### DAgger Training Trajectory Plots

Per-iteration 2D trajectory plots showing pilot path vs reference branch:

```
cohorts/SSV_DAGGER_CENTROID_V9/dagger_data/InstinctJester/plots/
  iter000_green_clock_br068.png
  iter008_green_and_pink_leafblower_br033.png
  ...
```

### Benchmark Trajectory Plots & Videos

The benchmark script (`scripts/benchmark_seen_unseen.py`) can save per-object interactive Plotly HTML plots and per-run MP4 videos. Controlled by two flags at the top of the script:

```python
SAVE_PLOTS  = True     # Interactive 3D Plotly HTML with point cloud + trajectories (~10MB each)
SAVE_VIDEOS = False    # MP4 videos per run — ~1MB each, but 600 runs = ~600MB
```

Output:
```
cohorts/{cohort}/benchmark_seen_unseen/
  plots/
    SEEN_DAGGER_green_clock.html              # Interactive 3D: point cloud + all runs + success zone
    SEEN_DAGGER_green_and_pink_leafblower.html
    SEEN_DAGGER_yellow_handheld_cordless_drill.html
    ...
  videos/
    SEEN_DAGGER_green_run000_rgb.mp4          # Per-run onboard camera views
    SEEN_DAGGER_green_run000_semantic.mp4
    SEEN_DAGGER_green_run000_depth.mp4
    ...
```

Plots reuse `create_comparison_figure()` from `scripts/compare_trajectories_3d.py`. Videos use the same `imageio` pattern as `deploy_ssv.py`. Videos are off by default — flip `SAVE_VIDEOS = True` for a specific subset, or reduce `max_traj`.

### Regenerate Simulation Videos

```bash
conda run --no-capture-output -n FiGS python ssv_muilti3dgs_campaign.py \
    simulate --config-file configs/experiment/ssv_dagger_centroid_v9.yml
```

Use `review: true` in the config to reuse existing RRT trajectories instead of regenerating.

---

## Detailed Reference

For in-depth implementation details (observation format, RRT generation, DAgger iteration internals, annotation filtering, config reference, simulation architecture), see [`docs/detailed_reference.md`](docs/detailed_reference.md).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ACADOS errors / solver=None | `export ACADOS_SOURCE_DIR=... && export LD_LIBRARY_PATH+=:...` |
| Benchmark success ~0% (centroid mismatch) | Ensure `centroid_version` in pilot config matches training (v9=percentile75, v10=median) |
| DAgger not improving | Verify `aggregate_dagger: false`, `reset_to_best: true`, `eval_seed: 42`. Check BC loss < 0.01. |
| Training loss rises slightly during DAgger | Normal — DAgger introduces novel states with higher loss. Judge by success_rate, not loss. |
| Out of GPU memory | One process per GPU. Reduce `n_eval_per_iter` or `n_benchmark`. |
