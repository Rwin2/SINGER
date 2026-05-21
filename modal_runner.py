"""
SINGER + FiGS — Complete Modal Environment
==========================================

This replicates the coruscant FiGS conda environment on Modal,
with GPU support, all code from GitHub, and weights from the Modal volume.

Usage:
    # Interactive shell (like SSH-ing into coruscant):
    modal shell modal_runner.py::shell_fn

    # Run a specific command:
    modal run modal_runner.py --cmd "cd /root/SINGER && python ssv_muilti3dgs_campaign.py --help"

    # Run detached (long jobs, survives terminal close):
    modal run --detach modal_runner.py --cmd "cd /root/SINGER && python ssv_muilti3dgs_campaign.py dagger ..."
"""
import modal

app = modal.App("singer-env")

# ---------------------------------------------------------------------------
# 1) Persistent volume — holds weights, splat checkpoints, scene data
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("singer-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# 2) Image — mirrors the FiGS conda env on coruscant
# ---------------------------------------------------------------------------
image = (
    # Start from NVIDIA CUDA 11.8 + Ubuntu 22.04 (matches coruscant)
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    # System dependencies
    .apt_install(
        "git",
        "cmake",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "ffmpeg",
        "wget",
        "unzip",
    )
    # PyTorch + CUDA 11.8 (must match exactly)
    .pip_install(
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    # gsplat (pre-built wheel for torch 2.1 + cu118)
    .pip_install("gsplat==1.4.0+pt21cu118", extra_index_url="https://docs.gsplat.studio/whl")
    # nerfstudio + core ML deps
    .pip_install(
        "nerfstudio==1.1.5",
        "transformers==4.38.2",
        "scipy",
        "numpy",
        "plotly",
        "open3d==0.19.0",
        "opencv-python-headless",
        "tqdm",
        "matplotlib",
        "imageio[ffmpeg]",
        "albumentations",
        "qpsolvers",
        "einops",
        "ftfy",
        "regex",
        "timm",
        "gdown",
        "ninja",
        "ipykernel",
        "cython",
        "pytorch-msssim",
    )
    # tiny-cuda-nn (builds from source, needs CUDA devel headers — already in base image)
    .pip_install(
        "git+https://github.com/NVlabs/tiny-cuda-nn/@2e757bbe781db59c4980d389d7dccbf5edc09669#subdirectory=bindings/torch",
    )
    # CLIP
    .pip_install("git+https://github.com/openai/CLIP.git")
    # Clone and install repos from GitHub
    .run_commands(
        # SINGER (your latest branch)
        "git clone --branch feature/rl-finetuning https://github.com/Rwin2/SINGER.git /root/SINGER",
        "cd /root/SINGER && pip install -e .",
        # FiGS-Standalone
        "git clone https://github.com/Rwin2/FiGS-Standalone.git /root/FiGS-Standalone",
        "cd /root/FiGS-Standalone && pip install -e .",
        # gemsplat (submodule inside FiGS)
        "cd /root/FiGS-Standalone && git submodule update --init --recursive",
        "cd /root/FiGS-Standalone/gemsplat && pip install -e .",
        # Hierarchical-Localization
        "cd /root/FiGS-Standalone/Hierarchical-Localization && pip install -e .",
        # acados — build from source
        "cd /root/FiGS-Standalone/acados && mkdir -p build && cd build"
        " && cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_EXAMPLES=OFF"
        " && make -j$(nproc) install",
        "cd /root/FiGS-Standalone/acados/interfaces/acados_template && pip install -e .",
    )
    # Set environment variables (acados + paths)
    .env({
        "ACADOS_SOURCE_DIR": "/root/FiGS-Standalone/acados",
        "LD_LIBRARY_PATH": "/root/FiGS-Standalone/acados/lib:/usr/local/cuda/lib64",
        "PYTHONPATH": "/root/SINGER/src:/root/FiGS-Standalone/src:/root/FiGS-Standalone/gemsplat",
    })
)

# ---------------------------------------------------------------------------
# 3) Symlink volume data into the expected paths at container start
# ---------------------------------------------------------------------------
SETUP_CMD = """
# Link weights from volume into the repo checkpoints
if [ -d /vol/SINGER/cohorts ]; then
    for cohort in /vol/SINGER/cohorts/*/; do
        name=$(basename "$cohort")
        target="/root/SINGER/cohorts/$name"
        mkdir -p "$target"
        # Link roster (weights)
        if [ -d "$cohort/roster" ]; then
            ln -sfn "$cohort/roster" "$target/roster"
        fi
        # Link JSONs
        for f in "$cohort"/*.json; do
            [ -f "$f" ] && ln -sf "$f" "$target/"
        done
    done
fi

# Link 3dgs data
if [ -d /vol/FiGS-Standalone/3dgs ]; then
    ln -sfn /vol/FiGS-Standalone/3dgs /root/FiGS-Standalone/3dgs
fi

echo "Environment ready. Repos at /root/SINGER and /root/FiGS-Standalone"
echo "Volume data at /vol/"
echo "Weights linked into /root/SINGER/cohorts/*/roster/"
"""


# ---------------------------------------------------------------------------
# 4) Functions
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10G",           # good balance of cost/performance; change to "A100" for heavy training
    timeout=86400,        # 24h max
    memory=32768,         # 32GB RAM
)
def run_command(cmd: str) -> str:
    """Run an arbitrary command in the SINGER environment."""
    import subprocess
    import os

    # Run setup to link volume data
    subprocess.run(["bash", "-c", SETUP_CMD], check=True)

    # Run the actual command
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        cwd="/root/SINGER",
        env={
            **os.environ,
            "ACADOS_SOURCE_DIR": "/root/FiGS-Standalone/acados",
            "LD_LIBRARY_PATH": "/root/FiGS-Standalone/acados/lib:/usr/local/cuda/lib64",
        },
    )
    output = result.stdout + result.stderr
    print(output)
    return output


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A10G",
    timeout=86400,
    memory=32768,
)
def shell_fn():
    """Placeholder for `modal shell modal_runner.py::shell_fn`"""
    import subprocess
    subprocess.run(["bash", "-c", SETUP_CMD], check=True)
    print("Ready! Volume at /vol, repos at /root/SINGER and /root/FiGS-Standalone")


@app.local_entrypoint()
def main(cmd: str = "echo 'SINGER Modal env ready!' && python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}\")'"):
    result = run_command.remote(cmd)
    print(result)
