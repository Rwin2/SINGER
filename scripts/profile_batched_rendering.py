#!/usr/bin/env python3
"""
Profile batched GemSplat rendering: render N camera views in one rasterization call.

Usage:
    cd /data/erwinpi/FiGS-Standalone
    CUDA_VISIBLE_DEVICES=0 conda run -n FiGS python /data/erwinpi/SINGER/scripts/profile_batched_rendering.py

Measures wall-clock time and GPU memory for batch sizes 1..100.
"""

import os, sys, time, json
import numpy as np
import torch
import subprocess

# ── Paths ──────────────────────────────────────────────────────────────
FIGS_ROOT = "/data/erwinpi/FiGS-Standalone"
SINGER_ROOT = "/data/erwinpi/SINGER"
sys.path.insert(0, os.path.join(FIGS_ROOT, "src"))
sys.path.insert(0, os.path.join(FIGS_ROOT, "gemsplat"))
os.chdir(FIGS_ROOT)

# acados
os.environ["ACADOS_SOURCE_DIR"] = os.path.join(FIGS_ROOT, "acados")
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.path.join(FIGS_ROOT, "acados/lib")

from figs.simulator import Simulator
from figs.render.gsplat_semantic import GSplat
from nerfstudio.cameras.cameras import Cameras, CameraType
from gsplat.rendering import rasterization
from gemsplat.gemsplat import get_viewmat


def get_gpu_stats():
    """Get current GPU memory and power from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits", f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"],
            text=True
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "mem_used_mb": float(parts[0]),
            "mem_total_mb": float(parts[1]),
            "power_w": float(parts[2]),
            "gpu_util_pct": float(parts[3]),
        }
    except Exception:
        return {}


def generate_random_poses(n, center=np.array([0, 0, 1.5]), radius=5.0, seed=42):
    """Generate n random camera poses looking roughly toward center."""
    rng = np.random.RandomState(seed)
    poses = []
    for _ in range(n):
        # Random position on sphere around center
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(np.pi / 6, np.pi / 3)  # 30-60 deg elevation
        r = rng.uniform(radius * 0.5, radius * 1.5)
        pos = center + r * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])

        # Look at center
        forward = center - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        T = np.eye(4)
        T[:3, 0] = right
        T[:3, 1] = -up       # camera convention: y points down
        T[:3, 2] = forward   # z points forward
        T[:3, 3] = pos
        poses.append(T)
    return poses


def run_batched_rasterization(model, camera, poses_c2g, batch_sizes, n_warmup=3, n_repeats=5):
    """
    Profile batched rasterization at different batch sizes.

    model: GemSplatModel instance
    camera: nerfstudio Camera template
    poses_c2g: list of [3,4] torch tensors (camera-to-gsplat, already transformed)
    """
    device = model.device

    # Pre-extract gaussian parameters (shared across all cameras)
    means = model.means
    quats = model.quats / model.quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities).squeeze(-1)
    colors = torch.cat((model.features_dc[:, None, :], model.features_rest), dim=1)
    sh_degree = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)

    # Get single intrinsics and tile them
    K_single = camera.get_intrinsics_matrices().to(device)  # [1, 3, 3]
    W, H = int(camera.width.item()), int(camera.height.item())

    results = []

    for bs in batch_sizes:
        if bs > len(poses_c2g):
            print(f"  Skipping batch_size={bs} (only {len(poses_c2g)} poses)")
            break

        # Stack poses and build viewmats
        c2w_batch = torch.stack([p.unsqueeze(0) for p in poses_c2g[:bs]], dim=0).squeeze(1).to(device)  # [bs, 3, 4]
        viewmat = get_viewmat(c2w_batch)  # [bs, 4, 4]
        K_batch = K_single.expand(bs, -1, -1)  # [bs, 3, 3]

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                render, alpha, info = rasterization(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                    viewmats=viewmat, Ks=K_batch,
                    width=W, height=H,
                    tile_size=16, packed=False,
                    near_plane=0.01, far_plane=1e10,
                    render_mode="RGB+ED", sh_degree=sh_degree,
                    sparse_grad=False, absgrad=False,
                    rasterize_mode=model.config.rasterize_mode,
                )
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                render, alpha, info = rasterization(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                    viewmats=viewmat, Ks=K_batch,
                    width=W, height=H,
                    tile_size=16, packed=False,
                    near_plane=0.01, far_plane=1e10,
                    render_mode="RGB+ED", sh_degree=sh_degree,
                    sparse_grad=False, absgrad=False,
                    rasterize_mode=model.config.rasterize_mode,
                )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        gpu_stats = get_gpu_stats()
        mem_allocated = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6

        avg_time = np.mean(times)
        per_cam = avg_time / bs
        result = {
            "batch_size": bs,
            "total_time_ms": round(avg_time * 1000, 1),
            "per_camera_ms": round(per_cam * 1000, 2),
            "fps": round(bs / avg_time, 1),
            "render_shape": list(render.shape),
            "torch_mem_alloc_mb": round(mem_allocated, 1),
            "torch_mem_reserved_mb": round(mem_reserved, 1),
            **{f"gpu_{k}": v for k, v in gpu_stats.items()},
        }
        results.append(result)

        print(f"  batch={bs:3d}  total={avg_time*1000:7.1f}ms  per_cam={per_cam*1000:6.2f}ms  "
              f"fps={bs/avg_time:6.1f}  mem={mem_allocated:7.0f}MB  "
              f"GPU_util={gpu_stats.get('gpu_util_pct','?')}%  power={gpu_stats.get('power_w','?')}W")

        # Safety: if we exceed 40GB, stop
        if mem_allocated > 40_000:
            print(f"  ⚠ Memory limit reached ({mem_allocated:.0f}MB), stopping.")
            break

        # Free memory for next round
        del render, alpha, info
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("Batched GemSplat Rendering Profiler")
    print("=" * 70)

    # ── Load scene ─────────────────────────────────────────────────────
    print("\nLoading scene (flightroom_ssv_exp)...")
    sim = Simulator("flightroom_ssv_exp")
    sim.load_frame("carl")
    gsplat = sim.gsplat
    model = gsplat.pipeline.model
    model.eval()
    device = model.device
    print(f"  Device: {device}")
    print(f"  Gaussians: {model.means.shape[0]:,}")

    # ── Camera config ──────────────────────────────────────────────────
    cam_cfg = sim.conFiG["drone"]["camera"]
    camera = gsplat.generate_output_camera(cam_cfg)
    W, H = cam_cfg["width"], cam_cfg["height"]
    print(f"  Camera: {W}x{H}")

    # ── Generate poses ─────────────────────────────────────────────────
    # Use object centroid from scene as target
    n_max = 150
    poses_world = generate_random_poses(n_max, center=np.array([3.0, 3.0, 1.0]), radius=5.0)

    # Transform to gsplat coordinate frame
    poses_c2g = []
    for T_c2w in poses_world:
        if gsplat.name.startswith("sv_") and "sfm_to_mocap_T" in gsplat.transforms_nerf:
            dp_scale = gsplat.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
            dp_transform = gsplat.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
            sfm2mocap_T = np.asarray(gsplat.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
            T_w2g = gsplat.T_w2g
            S = np.eye(4); S[:3, :3] *= 1.0 / dp_scale
            T_dp = np.eye(4); T_dp[:3, :] = dp_transform
            inv_dp = np.linalg.inv(T_dp)
            M = sfm2mocap_T @ inv_dp
            A = T_w2g @ M @ S
            A_inv = np.linalg.inv(A)
            T_c2g = A_inv @ T_c2w
            P_c2g = torch.from_numpy(T_c2g[:3, :]).float()
        else:
            T_c2g = gsplat.dataparser_scale * (gsplat.T_w2g @ T_c2w)
            P_c2g = torch.tensor(T_c2g[:3, :]).float()
            P_c2g[:3, :3] *= 1.0 / gsplat.dataparser_scale
        poses_c2g.append(P_c2g)

    print(f"  Generated {n_max} random camera poses")

    # ── Profile ────────────────────────────────────────────────────────
    batch_sizes = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]

    baseline_stats = get_gpu_stats()
    print(f"\n  Baseline GPU: mem={baseline_stats.get('mem_used_mb','?')}MB  "
          f"power={baseline_stats.get('power_w','?')}W")

    print(f"\nProfiling {len(batch_sizes)} batch sizes (RGB+Depth, no CLIP)...")
    print("-" * 90)

    results = run_batched_rasterization(model, camera, poses_c2g, batch_sizes)

    print("-" * 90)

    # ── Save results ───────────────────────────────────────────────────
    out_path = os.path.join(SINGER_ROOT, "batch_rendering_profile.json")
    with open(out_path, "w") as f:
        json.dump({
            "scene": "flightroom_ssv_exp",
            "resolution": f"{W}x{H}",
            "n_gaussians": model.means.shape[0],
            "gpu": baseline_stats,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary ────────────────────────────────────────────────────────
    if results:
        single = results[0]["per_camera_ms"]
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Single-camera: {single:.2f} ms/frame")
        for r in results:
            speedup = single / r["per_camera_ms"] if r["per_camera_ms"] > 0 else 0
            print(f"  Batch {r['batch_size']:3d}: {r['per_camera_ms']:6.2f} ms/cam  "
                  f"({speedup:.1f}x speedup)  mem={r['torch_mem_alloc_mb']:.0f}MB")


if __name__ == "__main__":
    main()
