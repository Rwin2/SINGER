# SINGER: Onboard Vision-Language Navigation Policy for Drones

> **Given a natural language instruction like "go to the green leafblower," the drone autonomously identifies and navigates to the target — collision-free.**
>
> The control policy is a lightweight neural network trained via Behavioral Cloning from an ACADOS MPC expert, refined with **DAgger** (pure learner rollout + expert relabeling). Geometric **centroid features** (bearing + elevation) extracted from the CLIPSeg heatmap provide the policy with a direct spatial signal for goal-directed control.
>
> **Results: 88% success rate, 8% collision rate, 91% generalization to unseen trajectories.**
>
> *Tech stack: PyTorch, 3D Gaussian Splatting (gsplat), CLIPSeg, ACADOS optimal control, CUDA*

---

## Installation

SINGER requires [FiGS-Standalone](https://github.com/StanfordMSL/FiGS-Standalone) as its simulator/renderer.

```bash
conda activate FiGS
export ACADOS_SOURCE_DIR=/path/to/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/FiGS-Standalone/acados/lib
cd /path/to/SINGER
```

---

## Architecture

### Neural Pilot — OODA Loop (20 Hz)

The pilot receives **only** onboard sensor data, no ground truth:

| Input | Source | Description |
|-------|--------|-------------|
| `tx_com` | IMU/state | `[z, vx, vy, vz, qx, qy, qz, qw]` — altitude + velocity + orientation. **No x,y position.** |
| `obj_com` | CLIPSeg | `[bearing, elevation, size]` — from semantic heatmap centroid |
| `dxu_par` | History | Rolling 20-step delta buffer (11x20) |
| `img_vis` | Camera | CLIPSeg semantic embedding → VisionMLP features |

### Network Architecture

- **HistoryEncoder**: Compresses DxU temporal sequences → feature vector. Frozen during DAgger.
- **VisionMLP**: Processes CLIPSeg image features via SqueezeNet backbone. Frozen during DAgger.
- **CommanderSV**: `(state + centroid + history + vision) → [thrust, ωx, ωy, ωz]`. Fine-tuned by DAgger.

### Two-Stage Training

```
  BC (Behavioral Cloning)                    DAgger (Dataset Aggregation)
  ┌────────────────────────────┐             ┌──────────────────────────────────────────┐
  │ 1. MPC expert flies 110    │             │ For each round:                          │
  │    RRT branches per object │             │   1. Learner flies ALL training branches │
  │ 2. Extract (Xnn, Ynn) obs │──model.pth──│   2. Expert relabels at learner states   │
  │ 3. Train HistoryEncoder    │             │      (inline, same sim step)             │
  │ 4. Train Commander (MSE)   │             │   3. Reset to BC, retrain on BC+DAgger   │
  └────────────────────────────┘             │   4. Next round: focus on failures       │
                                             └──────────────────────────────────────────┘
```

DAgger retrains from BC weights each round on the growing dataset (textbook DAgger, Ross et al. 2011). The best model across rounds is saved for deployment.

---

## CLI Commands

```bash
RUN="conda run --no-capture-output -n FiGS python ssv_muilti3dgs_campaign.py"
CFG=configs/experiment/ssv_dagger_centroid_v9.yml
```

### BC Pipeline (Steps 1-5)

```bash
$RUN generate-rollouts --config-file $CFG              # MPC flies 110 branches/object
$RUN generate-rollouts --config-file $CFG --validation-mode  # 11 held-out branches
$RUN generate-observations --config-file $CFG           # Extract (Xnn, Ynn) pairs
$RUN train-history --config-file $CFG                   # Train HistoryEncoder
$RUN train-command --config-file $CFG                   # Train Commander (BC)
```

### DAgger (Step 6)

```bash
$RUN dagger --config-file $CFG --n-rounds 10 --n-epochs 30
```

Config fields: `bc_cohort`, `n_rounds`, `n_epochs_per_round`, `dagger_lr`, `seed`, `max_success_per_obj`.

Algorithm per round:
1. Eval+collect on ALL training branches in single pass (DAggerPolicy: pilot flies, expert relabels inline)
2. Track best model by success rate
3. Retrain from BC weights on BC + all accumulated DAgger annotations
4. Next round focuses on failure branches + few random successes

### Simulate (Deploy on Trajectories + Videos)

Deploy the trained pilot (or MPC expert) on RRT trajectories with video recording:

```bash
# Generate fresh RRT trajectories and simulate
$RUN simulate --config-file configs/experiment/ssv_simulate_hw1.yml

# Config example (configs/experiment/ssv_simulate_hw1.yml):
# cohort: "SSV_DAGGER_HW1_COMPREHENSIVE"
# method: "rrt_6s"
# review: false          # false = fresh RRT, true = reuse BC rollout trajectories
# flights:
#   - ["flightroom_ssv_exp", "flightroom_ssv_exp"]
# roster:
#   - "InstinctJester"
```

Simulates both **expert** (MPC) and **InstinctJester** (learned pilot) on the same trajectory. Early stopping on success (goal zone + FOV) or collision. Outputs:

```
cohorts/{cohort}/simulation_data/{timestamp}/
├── videos/          # MP4 per pilot × object × channel (semantic, rgb, depth)
├── trajectories/    # .pt state/control data
└── rrt_planning/    # RRT tree data
```

### Benchmark (Evaluate Models)

```bash
$RUN benchmark --config-file $CFG \
    --models "BC:ssv_BC_CENTROID_V9/InstinctJester, DAgger:SSV_DAGGER_CENTROID_V9/InstinctJester" \
    --branches seen --max-trajectories 50 --seed 42 --include-expert
```

All artifacts on by default: plotly HTMLs (toggleable per trajectory), MP4 videos, JSON analysis with clearance/FOV/yaw error/normalized distance. Supports: `--branches seen|unseen|both`, `--seeds 42,123,456` (multi-seed), `--overlay` (multi-model overlay).

Output per branch ID:
```
cohorts/{cohort}/benchmark_results/{timestamp}/
├── plots/           # Interactive 3D Plotly with toggleable trajectories per branch
├── videos/          # {model}_{object}_br{id}_{semantic,rgb}.mp4
├── analysis/        # Detailed per-run performance (clearance, yaw, FOV, normalized dist)
├── benchmark_results_seed{N}.json
└── diagnostics_seed{N}.json
```

---

## Results

### Training Benchmark (50 runs/object, seed=42)

| Phase | Clock | Leafblower | Boxes | **Avg Success** | **Collision** |
|-------|-------|------------|-------|-----------------|---------------|
| BC baseline | 86% | 76% | 80% | **80.7%** | 13.3% |
| **After DAgger** | **90%** | **84%** | **90%** | **88.0%** | **8.0%** |

### Seen vs Unseen Generalization

| Set | Model | **Avg Success** | **Collision** |
|-----|-------|-----------------|---------------|
| Seen (50/obj) | DAgger | **86.0%** | 8.0% |
| Unseen (11/obj) | DAgger | **90.9%** | 9.1% |

The model generalizes: unseen > seen (90.9% vs 86.0%).

---

## Centroid Features

The pilot locates objects via the gsplat semantic similarity map:

```
Semantic heatmap → threshold at percentile(75)
  → weighted centroid → bearing [-1,1], elevation [-1,1], apparent_size [0,1]
```

These are the **only goal-related inputs**. No ground truth position is ever provided.

---

## Project Structure

```
SINGER/
├── ssv_muilti3dgs_campaign.py              # Entry point
├── notebooks/
│   └── ssv_muilti3dgs_campaign_coruscant.py  # CLI (Typer)
├── configs/
│   ├── experiment/                         # Experiment configs (.yml)
│   ├── pilots/InstinctJester.json          # Pilot config (centroid_version)
│   ├── scenes/flightroom_ssv_exp.yml       # Scene: queries, radii, bounds
│   └── method/rrt_6s.json                  # Method: frame, policy, rollout
├── src/sousvide/
│   ├── control/pilot.py                    # Pilot OODA loop + centroid
│   ├── control/policies/svnet.py           # Network forward pass
│   ├── control/policies/ComponentNetworks.py # CommanderSV, HistoryEncoder, VisionMLP
│   ├── instruct/train_dagger.py            # DAgger training (single-pass eval+collect)
│   ├── instruct/benchmark.py              # Unified benchmark (plots, videos, analysis)
│   ├── instruct/train_policy.py            # BC training loop
│   ├── flight/deploy_ssv.py               # Simulation + video recording
│   └── visualize/analyze_simulated_experiments.py  # Performance analysis
└── cohorts/                                # Experiment data
    └── {cohort}/
        ├── rollout_data/{scene}/           # Expert trajectories (.pt)
        ├── roster/{pilot}/
        │   ├── model.pth                  # Best model (for deployment)
        │   ├── model_best.pth             # Best across DAgger rounds
        │   └── model_current.pth          # Latest round
        ├── dagger_data/{pilot}/            # DAgger annotations
        ├── checkpoints/                    # Crash-resilient round checkpoints
        └── benchmark_results/{timestamp}/  # Benchmark output
```

---

## Simulation Timing

Per-step cost breakdown (after torch compile warmup):

| Component | Time |
|-----------|------|
| gsplat semantic rendering | ~660ms |
| MPC expert solve | ~0.5ms |
| Pilot OODA (SqueezeNet + Commander) | ~2ms |
| **Total per step** | **~664ms** |

The bottleneck is gsplat rendering with `compute_semantics=True`. The pilot and expert add negligible overhead.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ACADOS errors / solver=None | Set `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` |
| Centroid mismatch | Ensure `centroid_version: "v9"` in pilot config (percentile 75) |
| Out of GPU memory | One process per GPU |
| Disk full during DAgger | Delete `checkpoints/`, `observation_data/`, `dagger_data/` from old cohorts |
