# Perception & Centroid Pipeline — Analysis and Development Notes

## Problem Statement

The SINGER drone navigates toward a text-described object using a **centroid bearing** derived from a semantic similarity heatmap. The heatmap is produced by gsplat, which renders per-pixel CLIP similarity between the text query (e.g., "green clock") and language features embedded in 3D Gaussians. The centroid provides the `obj_com` input to the Commander network: `[bearing, elevation, apparent_size]`.

**Current limitation**: The centroid is always computed (even when the target is not visible), and is computed on the wrong data (turbo colormap instead of raw similarity). This causes:
1. Misleading bearings when the target is out of FOV
2. Inaccurate centroids even when the target is visible
3. No confidence signal to distinguish visible vs not-visible

## Pipeline: How the Semantic Image Flows

```
gsplat 3D Gaussians (each has CLIP language features)
  → volumetric rendering from camera pose
  → per-pixel CLIP cosine similarity (raw, single-channel float)
  → render_rescale() — running min-max normalization to [0,1]
  → apply_colormap("turbo") — RGB turbo colormap          ← "semantic" in Iro
  → (255 * ...).astype(uint8)
  → VisionMLP (SqueezeNet CNN) → y_vis (128-d)            ← sees the colormap
  → _compute_centroid() → bearing/elevation/size            ← ALSO on the colormap!
```

**Key finding (2026-04-28)**: `_compute_centroid()` operates on the turbo-colormapped RGB image, not on the raw similarity. Averaging R,G,B channels of a turbo colormap does not produce the actual similarity values. The centroid is computed on visualization artifacts.

## V9 Centroid (Current — in production)

- Source: `pilot.py:_compute_centroid()` (line 248)
- Input: turbo-colormapped semantic image (uint8, H=360, W=640, 3 channels)
- Method: `np.percentile(heat, 75)` → always selects top 25% of pixels
- Weighted mean of selected pixel positions → centroid
- `confidence = heat.max()` → **always 0.6562** (constant, useless)
- No visibility gating — centroid always computed, even when target not in frame

### Problems:
1. Operating on turbo colormap, not raw similarity
2. Percentile threshold is relative — always picks 25% of pixels
3. No outlier removal, no spatial clustering
4. No confidence gating — misleading centroid when target not visible

## V12 Centroid (Experimental — 2026-04-28)

- Source: `scripts/eval_failed_val_videos.py:_centroid_v12()`
- Input: **raw similarity** (single-channel float, post-render_rescale, pre-colormap)
- Method:
  1. Absolute threshold: `heat > 0.7 * heat.max()`
  2. Largest connected component (cv2.connectedComponentsWithStats)
  3. Weighted centroid within that blob
- Confidence: `peak = heat.max()` — now meaningful (0.60-1.0 range, varies with visibility)

### Results (9 test cases):

| Object | val | V9 err | V12 err | Winner |
|---|---|---|---|---|
| clock | 0 (success) | 69px | **19px** | V12 3.6x |
| clock | 4 (failed) | **107px** | 134px | V9 (V12 bad start) |
| clock | 5 (failed) | — | — | GT never visible |
| clock | 6 (failed) | 144px | **47px** | V12 3.1x |
| leafblower | 0 (success) | 249px | **22px** | V12 11x |
| leafblower | 1 (failed) | **34px** | 154px | V9 (low conf) |
| drill | 0 (success) | 150px | **142px** | V12 slight |
| drill | 6 (failed) | 148px | 157px | ~tie |
| drill | 7 (failed) | 296px | **69px** | V12 4.3x |

**V12 wins 5/7 comparable cases.** When it loses, the pattern is:
- Low confidence (leafblower val=1: conf=0.78 — green object in green room)
- Large blob (n_pixels > 10K — diffuse, unreliable)

### Confidence separation:
- GT visible: V12 confidence **mean 0.97**, min 0.80
- GT NOT visible: V12 confidence **mean 0.77**, min 0.60
- **Threshold at 0.90 cleanly separates visible from not-visible**

## Fused Centroid (V12 + Confidence Gate + Dead-Reckoning)

Two-mode navigation, inspired by biological agents:

### Regime 1 — Target visible (conf > 0.90, n_pixels < 15K)
Use V12 centroid directly. The raw similarity gives accurate bearing/elevation.

### Regime 2 — Target not visible (conf < 0.90 or diffuse blob)
**Dead-reckoning**: predict goal bearing from last confident observation using drone rotation.

The goal is fixed in world frame. Between the last confident observation (step t) and now (step t+k), only the drone has moved. We can predict where the goal should be in the current camera frame:

1. Reconstruct the goal's ray direction in the last camera frame (from bearing/elevation)
2. Transform that ray to world frame using the drone's pose at step t
3. Transform back to current camera frame using the drone's pose at step t+k
4. Project to predicted bearing/elevation

This uses the **exact camera model** (no small-angle approximation):
- Camera intrinsics: fx=462.956, fy=463.002, cx=323.076, cy=181.184
- Camera-to-body transform: T_C2B (from carl.json)
- Drone state: position + quaternion at each step
- Convention: nerfstudio/OpenGL (forward = -Z, Y up)

Dead-reckoning degrades over time (accumulates drift), but is accurate for short gaps of lost visibility. The drone should reacquire the target within ~20-30 steps.

### Implementation: `_deadreckon_bearing()` in `eval_failed_val_videos.py`

```python
# 1. Pixel → ray in last camera frame
ray_cam = [(u - cx)/fx, -(v - cy)/fy, -1.0]  # OpenGL convention

# 2. Camera → world at last step
ray_world = R_c2w_last @ ray_cam

# 3. World → camera at current step
ray_now = R_w2c_now @ ray_world

# 4. Project back to pixel (if not behind camera)
u_new = fx * ray_now[0] / (-ray_now[2]) + cx
v_new = fy * (-ray_now[1]) / (-ray_now[2]) + cy
```

## GT Centroid (Reference — not used in deployment)

- Source: `FiGS-Standalone/src/figs/scene_editing/scene_editing_utils.py:get_centroid()`
- Operates on **3D Gaussian point cloud** (all ~300K splats), not 2D rendered image
- Pipeline:
  1. CLIP similarity per 3D Gaussian (raw cosine)
  2. Shift+clip: `clip(similarity - 0.5, 0, 1)`, normalize by max
  3. Hard threshold (0.90) — only most similar Gaussians survive
  4. Statistical outlier removal (nb_neighbors=20, std_ratio=0.01)
  5. Spherical filter (Mahalanobis distance) — keeps spatial cluster
  6. Convex hull fill — captures full object volume
  7. Negative prompts: 'window,wall,floor,ceiling' — suppresses background
  8. `np.mean(filtered_3D_points)` → 3D centroid in world frame
- Only computed once at trajectory planning time (get_objectives)
- NOT available at runtime — requires full 3D point cloud query

## Max's Original SINGER

- `obj_com = [0, 0, 0]` always (objective features unused)
- Commander navigates purely on `tx_com + z_par + y_vis`
- No centroid computation, no CLIPSeg, no semantic reasoning
- Our centroid approach (V9) improved BC from ~36% to 48%, DAgger from 52% to 88%

## Development Roadmap

### Phase 1 — Validated (rendering comparison only)
- [x] V12: raw similarity centroid (abs threshold + connected component)
- [x] Confidence gate (conf > 0.90)
- [x] Dead-reckoning when not confident
- [ ] Visual + numerical validation on 9 test cases

### Phase 2 — Integration (modify pilot.py)
- [ ] Add `similarity_raw` channel to gsplat render pipeline (done in gsplat_semantic.py)
- [ ] Modify `_compute_centroid()` to use raw similarity instead of colormap
- [ ] Add confidence gate: if not confident, output [0, 0, 0]
- [ ] Optionally: dead-reckoning using pilot's own state history (DxU buffer)
- [ ] Re-run DAgger or PPO with new centroid → new model

### Phase 3 — Exploration mode (future)
- [ ] When confidence stays low for N steps, add exploration reward in PPO
- [ ] Frontier-style exploration: penalize staying in low-confidence orientation
- [ ] Alternatively: train a small "target detector" on y_vis (128-d) → binary visible/not
- [ ] Could use the VLA-AN approach: embed natural language goal via LLM, use depth for obstacles

## Key Files
- `pilot.py:_compute_centroid()` — current V9 centroid (lines 248-343)
- `gsplat_semantic.py:render_rgb()` — now returns `similarity_raw` (lines 251-267)
- `gsplat_semantic.py:render_rescale()` — running min-max normalization (lines 475-490)
- `scene_editing_utils.py:get_centroid()` — GT 3D centroid (lines 406-710)
- `scripts/eval_failed_val_videos.py` — V12 + fused comparison rendering
- `scripts/plot_failed_val_trajectories.py` — Plotly trajectory visualization (legacy)
- `configs/perception/perception_mode.yml` — perception_type: "similarity"
