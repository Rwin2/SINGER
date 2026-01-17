# SOUS VIDE

## Installation
1) Clone repository and load the submodules.
```
git clone https://github.com/StanfordMSL/SousVide.git
git submodule update --recursive --init
```
2) Build ACADOS locally.
```
# Navigate to acados folder
cd <repository-path>/SousVide/FiGS/acados/

# Compile
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
3) Set up conda environment (in the main directory)
 ```
cd <repository-path>/SousVide-Semantic/

conda create --name <env-name> -y python=3.10

conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate <env-name>

python -m pip install --upgrade pip

pip install numpy==1.26.3

conda deactivate
```
4) Install Zed SDK and pyzed (Zed Python bindings)
Follow instructions on Zed website for this part, including the bindings.

5) Install pyzed in <env-name> and rest of dependencies
```
conda activate <env-name>

python3 -m pip install --no-deps \
--ignore-installed /usr/local/zed/pyzed-5.0-cp310-cp310-linux_aarch64.whl

conda deactivate

conda env update <env-name> environment_x86.yml

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio==1.1.1
pip uninstall gsplat
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 pip install git+https://github.com/nerfstudio-project/gsplat.git@c7b0a383657307a13dff56cb2f832e3ab7f029fd

cd ~/StanfordMSL/SousVide-Semantic

git submodule add -b semantic_field_v1 https://github.com/StanfordMSL/gemsplat.git ~/data/StanfordMSL/SousVide-Semantic/gemsplat

cd gemsplat
pip install -e .

conda install conda-forge::albumentations==2.0.5 --freeze-installed

ns-install-cli (this might fail)
```
## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sv_shakedown</b>: Use this notebook to verify all components before running other SV notebooks. It does not produce a usable policy
  3. <b>sv_robustness</b>: Produces the policies used in Section VI.A and VI.B.
  4. <b>sv_extended</b>: Produces the policy for the extended trajectory in Section VI.C.
  5. <b>sv_cluttered</b>: Produces the policy for the cluttered environment trajectory in Section VI.C.

## [COMING SOON (2025)] Deploy SOUS VIDE in the Real World
Deploy SOUS VIDE policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware). Tutorial and code coming soon!
