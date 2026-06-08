# SINGER — CS224R Code Submission

**Title:** Enhancing SINGER: Onboard Visual Drone Navigation through Iterative Imitation Learning and Reinforcement Fine-Tuning  
**Author:** Erwin Poussi  
**Course:** CS224R Deep Reinforcement Learning, Stanford, Spring 2026

## Overview

This codebase extends SINGER (Semantic In-situ Navigation and Guidance for Embodied Robots) with goal-conditioned centroid features, DAgger training, and offline IQL fine-tuning for language-conditioned drone navigation in 3D Gaussian Splatting simulation.

## Execution Path

### BC Pipeline (stages run sequentially)

```
ssv_muilti3dgs_campaign.py
  -> notebooks/ssv_muilti3dgs_campaign_coruscant.py

1. generate-rollouts     -> rg.generate_rollout_data()          # RRT* expert trajectories
2. generate-observations -> og.generate_observation_data()       # render frames, extract features
3. train-history         -> tp.train_roster(..., "Parameter")    # train HistoryEncoder
4. train-command         -> tp.train_roster(..., "Commander")    # train Commander head (BC)
```

### DAgger

```
5. dagger -> train_dagger.train_dagger_policy()
   |-- _get_scene()              # load GS simulator + scene
   |-- _load_training_branches() # get RRT branches from BC cohort
   |-- Per round:
   |   |-- _select_branches_for_round()   # pick branches (failures + some successes)
   |   |-- _collect_and_evaluate()         # rollout pilot, query MPC expert, collect annotations
   |   |   |-- DAggerPolicy               # wraps pilot for rollout
   |   |   |-- _evaluate_run()             # success/collision/timeout classification
   |   |   +-- _save_traj_plot()           # 3D plotly visualization
   |   |-- _aggregate_dagger_dataset()     # merge new + old annotations
   |   |-- _retrain_commander()            # retrain Commander head only (VisionMLP frozen)
   |   +-- _save_model_checkpoint()        # save best model
   +-- _wandb_log_round()
```

### IQL (separate scripts, not in main campaign CLI)

```
6. scripts/collect_rl_transitions.py -> main()
   |-- XnnCapturePilot              # wraps pilot to capture Commander inputs
   |-- ExpertWithPerception         # MPC expert with centroid features
   |-- extract_commander_input()    # extract 147-d obs from xnn logs
   +-- build_transitions()          # (s, a, r, s', done) tuples

7. scripts/train_iql.py -> train() / main()
   |-- LinearGaussianActor          # actor matching Commander architecture
   |-- SingerIQLAgent               # IQL agent (actor + Q + V networks)
   |-- load_commander_weights()     # init actor from DAgger best model
   |-- load_transitions()           # load collected data
   +-- export_actor_to_model()      # write trained actor back to model.pth

8. scripts/benchmark_iql_r8.py      # evaluate on the 17 training branches
```

### Benchmark (used for all evaluations)

```
benchmark -> benchmark.run_unified_benchmark()
   |-- _resolve_branches()          # seen vs unseen branch selection
   |-- evaluate_branches()          # run pilot on branches, score each
   |   +-- _evaluate_run()          # per-trajectory success criteria
   |-- _save_overlay_plotly()       # multi-model trajectory overlay
   +-- JSON results saved
```

## Key Files

| File | Description |
|------|-------------|
| `ssv_muilti3dgs_campaign.py` | Root entry point |
| `notebooks/ssv_muilti3dgs_campaign_coruscant.py` | Main CLI with all commands |
| `src/sousvide/instruct/train_dagger.py` | DAgger training loop |
| `src/sousvide/instruct/benchmark.py` | Unified benchmark system |
| `src/sousvide/instruct/synthesized_data.py` | Data loading and splitting |
| `src/sousvide/instruct/train_policy.py` | BC training (HistoryEncoder + Commander) |
| `src/sousvide/control/pilot.py` | Pilot class (wraps policy for deployment) |
| `src/sousvide/control/policies/` | Network architectures (SqueezeNet, Commander, VisionMLP) |
| `scripts/train_iql.py` | Offline IQL training |
| `scripts/collect_rl_transitions.py` | RL transition collection from pilot + expert |
| `scripts/benchmark_iql_r8.py` | IQL benchmark on training branches |
| `scripts/train_gaussian_bc.py` | Gaussian BC variant training |
| `configs/experiment/` | All experiment configuration files |

## Dependencies

- Python 3.10, PyTorch, FiGS-Standalone (3D Gaussian Splatting simulator)
- CLIPSeg for semantic segmentation
- ACADOS for MPC controller
- wandb for experiment tracking

## Experiment Configs

All experiment configurations are YAML files in `configs/experiment/`. Key configs:
- `ssv_bc_centroid_v9.yml` — Centroid BC training
- `ssv_dagger_centroid_v9.yml` — DAgger V9 (best model)
- `ssv_iql_v2.yml` / `ssv_iql_v3_frozen.yml` — IQL fine-tuning
- `ssv_gaussian_bc_v1.yml` — Gaussian BC variant
