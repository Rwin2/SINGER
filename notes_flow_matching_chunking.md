# SINGER: Flow Matching & Action Chunking Analysis

**Date**: 2026-04-04
**Context**: After completing CS224R HW1 (BC + Flow Matching + DAgger on Flappy Bird), exploring whether these techniques can improve SINGER's drone navigation policy.

---

## 1. SINGER Current Architecture

### How BC Works Today
- **Loss**: MSELoss (mean reduction)
- **Action**: Single-step, 4D vector `[thrust, wx, wy, wz]` at 20Hz
- **No action chunking**: one action per forward pass, purely reactive
- **Network**: 3-stage pipeline:
  - HistoryEncoder: 11×20 → [64,32] → 8D (frozen during DAgger)
  - VisionMLP: SqueezeNet → [512,256] → 128D (frozen during DAgger)
  - CommanderSV: 147D input → [100,100] → 4D output (only part retrained in DAgger)
- **Training data**: ~52K samples from RRT expert (110 branches × 4 perturbations × ~120 steps × 3 objects)

### Current Results (V9)
| Phase | Success | Collision |
|-------|---------|-----------|
| BC baseline | 80.7% | 13.3% |
| DAgger (iter 8) | 88.0% | 8.0% |

### Key Files
| File | Purpose |
|------|---------|
| `src/sousvide/instruct/train_policy.py:36` | MSELoss declaration |
| `src/sousvide/control/policies/svnet.py:170-195` | SVNet forward pass |
| `src/sousvide/control/policies/ComponentNetworks.py:208-265` | CommanderSV class |
| `src/sousvide/instruct/train_dagger.py:275` | DAgger MSELoss |
| `configs/pilots/InstinctJester.json` | Network dimensions |

---

## 2. Does SINGER Have a Multimodality Problem?

**Maybe, but less obvious than Flappy Bird.**

In Flappy Bird, multimodality is clear: two gaps → two valid actions → MSE averages → crash.

In SINGER, possible multimodality:
- **Multiple approach paths**: drone could go left or right around an obstacle to reach an object
- **Different approach angles**: multiple valid entry angles to reach a goal
- **Hover vs approach**: near the goal, expert sometimes adjusts position vs stays still

**However**: SINGER's RRT expert generates specific branches, and the drone follows one branch at a time. The expert data may not be strongly multimodal — each state maps to roughly one correct action given the current branch context.

**Verdict**: Flow matching could help, but the gain may be smaller than in Flappy Bird. The bigger bottleneck is likely **distribution shift** (already addressed by DAgger) and **temporal coherence** (not addressed — action chunking would help here).

---

## 3. Flow Matching for SINGER

### What Would Change
1. **Commander architecture**: Add time embedding input (sinusoidal, ~32D)
   - Current: 147D → [100,100] → 4D
   - New: 147D + 32D (time) + 4D (noisy action) = 183D → [100,100] → 4D (velocity)
2. **Loss function**: MSE on velocity prediction instead of direct action MSE
3. **Inference**: 10 Euler steps from noise to action (instead of 1 forward pass)

### Implementation Sketch
```python
# Training (in train_policy.py)
t = torch.rand(B, 1, device=device)           # flow time
noise = torch.randn_like(action_label)         # a0 ~ N(0,I)
a_t = t * action_label + (1 - t) * noise      # interpolate
target_v = action_label - noise                # target velocity

pred_v = commander(state_input, a_t, t)        # predict velocity
loss = F.mse_loss(pred_v, target_v)

# Inference (in pilot.py act())
a = torch.randn(1, 4, device=device)           # start from noise
for i in range(10):
    t = torch.tensor([i / 10.0], device=device)
    v = commander(state_input, a, t)
    a = a + v / 10.0
a = a.clamp(-1, 1)                             # action bounds
```

### Pros
- Handles multimodal expert data without averaging
- Drop-in replacement for MSE — same data pipeline
- Can combine with DAgger (retrain velocity field with DAgger data)

### Cons
- **10x slower inference**: 10 forward passes instead of 1 (at 20Hz: 10 passes × ~0.1ms = 1ms — still fast enough)
- **More complex training**: need time conditioning, noise sampling
- **Uncertain gain**: SINGER's expert may not be strongly multimodal
- **Requires Commander architecture change** (new input dimensions)

---

## 4. Action Chunking for SINGER

### What Would Change
1. **Commander output**: 4D → 4×H D (H = chunk horizon, e.g., 10 steps = 40D)
2. **Training data**: window sequential actions into chunks of H
3. **Execution**: predict H actions, execute first K < H, then re-query

### Recommended Parameters (inspired by π₀)
- **Chunk horizon (H)**: 10 steps = 0.5s at 20Hz (conservative start)
- **Execute steps (K)**: 3-5 steps = 0.15-0.25s
- **Action dim**: 4 × 10 = 40D output

### Implementation Sketch
```python
# Data preparation: window sequential expert actions
# From trajectory [a0, a1, a2, ..., a119] → chunks
# State s_i maps to action chunk [a_i, a_{i+1}, ..., a_{i+H-1}]

# Commander output: 4D → 40D
# At inference: execute first K=5 actions, then re-query

# In pilot.py act():
if self.chunk_buf is None or self.chunk_step >= K:
    chunk = commander(state_input)       # → 40D = 10 × 4D
    self.chunk_buf = chunk.view(10, 4)   # reshape to (H, 4)
    self.chunk_step = 0
action = self.chunk_buf[self.chunk_step]
self.chunk_step += 1
```

### Pros
- **Temporal consistency**: smooth trajectories, less jitter between steps
- **Fewer decision points**: reduces compounding error
- **Encodes intent**: a chunk says "turn right then stabilize" vs single-step "turn right"
- **Simple to implement**: just change output dim + add windowing to data pipeline

### Cons
- **Less reactive**: if something unexpected happens mid-chunk, drone can't adapt for K steps
- **Larger output space**: 40D instead of 4D — needs more training data or bigger network
- **Data pipeline change**: need to create chunk pairs from trajectories

---

## 5. π₀ Inspiration — What's Applicable to SINGER

### π₀ Architecture Summary
- VLM backbone (PaliGemma 2B) + action expert (transformer, ~300M params)
- Flow matching with 10 Euler steps
- Action chunk: 50 steps at 5Hz (10s lookahead)
- Trained on 10K+ hours of multi-robot data
- Fine-tuned with ~50-100 demos per task

### What SINGER Can Borrow
| π₀ Feature | SINGER Adaptation | Difficulty |
|------------|-------------------|------------|
| Flow matching loss | Replace MSE in Commander training | **Medium** — new loss + time conditioning |
| Action chunking (50 steps) | 10-step chunks (0.5s at 20Hz) | **Easy** — change output dim + data windowing |
| Sinusoidal time embedding | Add 32D time input to Commander | **Easy** — standard module |
| 10 Euler inference steps | Same | **Easy** — simple loop |
| Cross-attention VLM+actions | Too heavy for SINGER | **Hard** — overkill for this scale |
| LoRA fine-tuning | Not needed — Commander is small enough to fully retrain | **N/A** |

### What SINGER Should NOT Borrow
- PaliGemma / large VLM backbone — SINGER uses SqueezeNet + CLIPSeg, which is appropriate for its scale
- 50-step action chunks — too long for drone at 20Hz, 2.5s of blind flying is dangerous
- Multi-robot pretraining — not applicable (single drone platform)

---

## 6. Recommendation: What to Try First

### Priority Order

**1. Action Chunking (EASY, HIGH IMPACT)**
- Simplest change: modify Commander output dim + data windowing
- Directly addresses temporal jitter in drone trajectories
- Can reuse existing MSE loss — no other changes needed
- Start with H=10 (0.5s), K=3 (0.15s execute)

**2. Flow Matching + Action Chunking (MEDIUM, POTENTIALLY HIGH IMPACT)**
- Combine both: predict velocity field over action chunks
- This is what π₀ does (minus the VLM)
- Addresses both temporal coherence AND multimodality
- Architecture: Commander takes (state, noisy_chunk, t) → predicts velocity over chunk

**3. Flow Matching alone (MEDIUM, UNCERTAIN IMPACT)**
- Single-step flow matching on 4D actions
- May not help much if expert data isn't strongly multimodal
- Try this only if action chunking reveals multimodality issues

### Experiment Plan

```
Phase 1: Action Chunking BC
  - Modify CommanderSV output: 4D → 40D (10-step chunks)
  - Window training data into chunks
  - Train BC with MSE on chunks
  - Benchmark vs current single-step BC (target: smoother trajectories)

Phase 2: Action Chunking + DAgger
  - Same chunking, run DAgger on top
  - Compare vs V9 DAgger (target: >88% success)

Phase 3: Flow Matching + Chunking
  - Replace MSE with flow matching loss
  - Add time embedding to Commander (147D + 32D + 40D → velocity)
  - 10 Euler steps at inference
  - Train BC, then DAgger
  - Compare vs Phase 2 (target: better multimodal handling)
```

---

## 7. Comparison: Flow Matching vs DAgger

| Aspect | Flow Matching | DAgger | Both Combined |
|--------|---------------|--------|---------------|
| Solves multimodality | Yes (learns distribution) | Yes (deterministic expert) | Yes |
| Solves distribution shift | No | Yes (visits policy states) | Yes |
| Needs expert at test time | No | No (only during training) | No |
| Inference speed | Slower (N forward passes) | Same as BC | Slower |
| Training complexity | Medium (time conditioning) | Medium (rollout + relabel) | High |
| Data efficiency | Same as BC | Needs more data (online) | Needs more data |
| **Best for SINGER** | If multiple valid paths exist | Already proven (+7.3pp) | Ideal but complex |

**Bottom line**: DAgger already works for SINGER. Action chunking is the lowest-hanging fruit. Flow matching is worth trying if chunking reveals multimodality issues, or as a combined approach for maximum performance.

---

## 8. DreamZero — World Action Model (arXiv 2602.15922)

**Paper**: "World Action Models are Zero-shot Policies" (NVIDIA GEAR Lab, Feb 2026)
**Code**: https://github.com/dreamzero0/dreamzero

### Core Idea
DreamZero jointly predicts **future video frames AND actions** using a single DiT (Diffusion Transformer). The video prediction acts as a **world model** — by predicting how the world will look, the network implicitly learns physics/dynamics, which regularizes the action prediction.

### What's "State Chunking"?
DreamZero uses **block-wise causal chunking**: the sequence is organized as paired blocks of `[image, action, state]`. Each block:
- 1 video frame (latent space)
- 32 action tokens
- 1 state token (proprioception)

Causal attention ensures block `i` can only see blocks `0..i`, making it autoregressive over time. This is different from π₀ which denoises the full action chunk at once.

### Dynamic Loss (Gaussian Timestep Weighting)
Standard flow matching samples `t ~ Uniform(0,1)`. DreamZero uses a **Gaussian-shaped weight**:

```
w(t) = exp(-2 * ((t - T/2) / T)^2)
```

This **upweights intermediate noise levels** and downweights extremes (too clean or too noisy). Intuition: the network learns most from medium-noise samples, not from trivial cases.

Total loss = `weighted_video_loss + weighted_action_loss`

### Architecture
- **Base**: Wan2.1-I2V-14B (video generation model, 40-layer DiT, 5120-dim)
- **Fine-tuning**: LoRA (rank 4) — base weights frozen
- **Inference**: 4 denoising steps (UniPC scheduler) — very fast
- **Scale**: 14B params, 7Hz on 2× GB200 GPUs

### What SINGER Can Borrow

| DreamZero Feature | SINGER Adaptation | Value |
|-------------------|-------------------|-------|
| **Joint state+action prediction** | Predict future bearing/elevation + thrust/angular vel | **HIGH** — dynamics regularizer prevents catastrophic forgetting |
| **Gaussian timestep weighting** | Apply to flow matching loss | **MEDIUM** — easy to implement, may improve training stability |
| **Block-wise causal chunking** | Predict state-action blocks autoregressively | **LOW** — overkill for 4D actions |
| **Video generation backbone** | Too heavy | **SKIP** — SINGER uses SqueezeNet, not a 14B DiT |

### Key Insight for SINGER: Dynamics Loss as Regularizer

The most transferable idea: **predict future states alongside actions**. For SINGER:

```python
# Instead of just predicting action:
#   Commander(state) → [thrust, wx, wy, wz]  (4D)
#
# Predict action + next state:
#   Commander(state) → [thrust, wx, wy, wz, next_bearing, next_elevation]  (6D)
#
# Loss = MSE_action + λ * MSE_state_prediction
```

Why this helps:
1. **Anti-forgetting**: The state prediction loss anchors the network's understanding of dynamics, making it resistant to catastrophic forgetting during DAgger
2. **Better representations**: The network must understand physics to predict future states, which improves action quality
3. **Cheap to implement**: We already have ground-truth future states in the trajectory data — no new data needed

This is similar to **auxiliary task learning** — the state prediction is a free regularizer.

---

## 9. Updated Recommendation: Priority Order

### Phase 1: Action Chunking + Dynamics Loss (EASY)
- Commander output: 4D → 40D actions + 6D state prediction = 46D
- Window training data into 10-step chunks
- Loss = MSE_action_chunk + 0.1 * MSE_state_prediction
- **Why first**: simplest changes, biggest expected impact on trajectory smoothness

### Phase 2: Flow Matching + Chunking + Dynamics (MEDIUM)
- Replace MSE with flow matching loss (on action chunk only)
- Add time embedding (32D) to Commander
- Gaussian timestep weighting from DreamZero
- 10 Euler steps at inference
- **Why second**: addresses potential multimodality

### Phase 3: DAgger on Phase 2 Model (MEDIUM)
- Run DAgger with flow matching loss instead of MSE
- The dynamics loss should reduce catastrophic forgetting during DAgger
- **Why third**: DAgger already works; this tests if flow matching + dynamics improves DAgger further

---

## 10. Open Questions

1. **Is SINGER's expert data actually multimodal?** → Analyze RRT branches: do different branches suggest different actions for similar states?
2. **How reactive does the drone need to be?** → If obstacles appear suddenly, long chunks are dangerous. Test with K=3 first.
3. **Can Commander handle 40D output?** → May need larger hidden layers ([256, 128] instead of [100, 100])
4. **How does flow matching interact with HistoryEncoder?** → History provides temporal context already — does chunking make it redundant?
5. **V10/V11 changes**: Should flow matching/chunking build on V9 or wait for V11 (fixed apparent_size)?
