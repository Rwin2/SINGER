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
> The control policy is a lightweight neural network trained via Behavioral Cloning from an ACADOS MPC expert, refined with **DAgger** (pure learner rollout + expert relabeling on all training branches). A key contribution is the introduction of explicit **geometric centroid features** (bearing + elevation) extracted from the CLIPSeg heatmap, providing the policy with a direct spatial signal for goal-directed control.
>
> **Results: 88% success rate, 8% collision rate, with 91% generalization to unseen trajectories.**
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
| `tx_com` | IMU/state | Position + velocity + quaternion (10D) |
| `obj_com` | **CLIPSeg** | Bearing, elevation, apparent size from heatmap centroid |
| `dxu_par` | History | Rolling 20-step delta buffer |
| `img_vis` | Camera | CLIPSeg semantic embedding |

### Network Architecture

- **HistoryEncoder**: Compresses DxU (11x20) temporal sequences. Trained once during BC, frozen during DAgger.
- **VisionMLP**: Processes CLIPSeg image features. Frozen during DAgger.
- **CommanderSV**: State + centroid + image + history → `[thrust, wx, wy, wz]`. Fine-tuned by DAgger.

### Two-Stage Training

```
  BC (Behavioral Cloning)                    DAgger (Dataset Aggregation)
  ┌────────────────────────────┐             ┌──────────────────────────────────────────┐
  │ 1. MPC expert flies 110    │             │ For each round:                          │
  │    RRT branches per object │             │   1. Learner flies ALL training branches │
  │ 2. Extract (Xnn, Ynn) obs │──model.pth──│   2. Expert relabels at learner states   │
  │ 3. Train HistoryEncoder    │             │   3. Reset to BC, retrain on BC+DAgger   │
  │ 4. Train Commander (MSE)   │             │   4. Evaluate on ALL branches            │
  └────────────────────────────┘             │   5. Round 2+: focus on failure branches │
                                             └──────────────────────────────────────────┘
```

### Project Structure

```
SINGER/
├── ssv_muilti3dgs_campaign.py              # Entry point (delegates to CLI)
├── notebooks/
│   └── ssv_muilti3dgs_campaign_coruscant.py  # CLI (Typer) — all commands
├── configs/
│   ├── experiment/                         # Experiment configs (.yml)
│   ├── pilots/InstinctJester.json          # Pilot config (centroid_version)
│   ├── scenes/flightroom_ssv_exp.yml       # Scene: queries, radii, bounds
│   └── method/rrt_6s.json                  # Method: frame, policy, rollout
├── src/sousvide/
│   ├── control/pilot.py                    # Pilot OODA loop + centroid
│   ├── control/policies/svnet.py           # Network forward pass
│   ├── control/policies/ComponentNetworks.py # CommanderSV, HistoryEncoder
│   ├── instruct/train_dagger.py            # DAgger training loop
│   ├── instruct/benchmark.py              # Unified benchmark (plots, videos, analysis)
│   ├── instruct/train_policy.py            # BC training loop
│   ├── instruct/synthesized_data.py        # Dataset loading
│   ├── synthesize/rollout_generator.py     # Expert rollout generation
│   ├── synthesize/observation_generator.py # Observation extraction
│   ├── flight/deploy_ssv.py               # Simulation + video recording
│   └── visualize/analyze_simulated_experiments.py  # Performance analysis
└── cohorts/                                # Experiment data (gitignored)
    └── {cohort}/
        ├── rollout_data/{scene}/           # Expert trajectories (.pt)
        ├── observation_data/{pilot}/       # Training observations (Xnn, Ynn)
        ├── roster/{pilot}/model.pth        # Trained model
        ├── dagger_data/{pilot}/            # DAgger annotations
        ├── checkpoints/                    # Crash-resilient round checkpoints
        ├── benchmark_results/{timestamp}/  # Benchmark output
        │   ├── plots/*.html               #   Interactive 3D Plotly
        │   ├── videos/*.mp4               #   Semantic + RGB per run
        │   ├── analysis/*.json            #   Detailed performance analysis
        │   ├── benchmark_results_seed*.json
        │   └── diagnostics_seed*.json
        └── simulation_data/{timestamp}/    # Simulate command output
            └── videos/*.mp4
```

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

DAgger algorithm:
- Round 1: fly ALL training branches with learner, expert relabels
- Round 2+: fly ALL failure branches + 5 random successes
- Each round: reset to BC weights, retrain on BC + all aggregated DAgger data
- Per-round checkpoints for crash resilience

### Simulate (Deploy + Videos)

```bash
$RUN simulate --config-file $CFG                       # Fresh RRT trajectories
$RUN simulate --config-file $CFG  # review=true in yml  # Reuse BC trajectories
```

Early stopping on success (goal zone + FOV) or collision.

### Benchmark (Evaluate Models)

```bash
$RUN benchmark --config-file $CFG \
    --models "BC:ssv_BC_CENTROID_V9/InstinctJester, DAgger:SSV_DAGGER_CENTROID_V9/InstinctJester" \
    --branches seen --max-trajectories 50 --seed 42 --include-expert
```

All artifacts on by default: plotly HTMLs, MP4 videos, JSON analysis, multi-model overlay.
Supports: `--branches seen|unseen|both`, `--seeds 42,123,456` (multi-seed), `--overlay`.

---

## Results

### Training Benchmark (50 runs/object, seed=42)

| Phase | Clock | Leafblower | Boxes | **Avg Success** | **Collision** |
|-------|-------|------------|-------|-----------------|---------------|
| BC baseline | 86% | 76% | 80% | **80.7%** | 13.3% |
| **After DAgger** | **90%** | **84%** | **90%** | **88.0%** | **8.0%** |

### Seen vs Unseen Generalization

| Set | Model | Clock | Leafblower | Boxes | **Avg Success** | **Collision** |
|-----|-------|-------|------------|-------|-----------------|---------------|
| Seen (50/obj) | DAgger | 90% | 84% | 84% | **86.0%** | 8.0% |
| Unseen (11/obj) | DAgger | **100%** | 91% | 82% | **90.9%** | 9.1% |

The model generalizes: unseen > seen (90.9% vs 86.0%).

---

## Centroid Features

The pilot locates objects via CLIPSeg on the onboard RGB image:

```
RGB image → CLIPSeg heatmap → threshold at percentile(75)
  → weighted centroid → bearing [-1,1], elevation [-1,1], apparent_size [0,1]
```

These are the **only goal-related inputs**. No ground truth position is ever provided.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ACADOS errors / solver=None | Set `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` |
| Centroid mismatch | Ensure `centroid_version: "v9"` in pilot config (percentile 75) |
| Out of GPU memory | One process per GPU |
