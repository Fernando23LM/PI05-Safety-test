#!/bin/bash
# One-time Isaac Sim + sim-evals setup on a Gilbreth LOGIN node (no GPU needed). ~1-2 h, ~60 GB in /scratch.
#
#   ACCEPT_NVIDIA_EULA=1 bash isaac/setup_isaac_gilbreth.sh
#
# Running Isaac Sim requires accepting the NVIDIA Omniverse / Isaac Sim license agreement
# (https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
# Setting ACCEPT_NVIDIA_EULA=1 means you have read and accept it; the jobs then pass
# OMNI_KIT_ACCEPT_EULA=YES to Isaac Sim.
#
# Why a container: Isaac Sim 5.0 pip wheels need glibc >= 2.35 (Ubuntu 22.04). The script checks
# the node's glibc; the container (NVIDIA's Isaac Sim image) supplies the right system libraries,
# while the Python env itself lives in /scratch and is built with uv, as sim-evals expects.
set -euo pipefail

if [ "${ACCEPT_NVIDIA_EULA:-}" != "1" ]; then
  echo "Isaac Sim needs the NVIDIA Omniverse License Agreement accepted. Read it, then rerun with ACCEPT_NVIDIA_EULA=1."
  exit 1
fi

SCR=/scratch/gilbreth/$USER
ALIGN=$SCR/pi05-align
SIF=$SCR/containers/isaac-sim-5.0.0.sif
ISAAC_HOME=$SCR/isaac-home          # $HOME inside the container: Isaac Sim caches (shaders, extensions) go here
export APPTAINER_CACHEDIR=$SCR/.apptainer-cache APPTAINER_TMPDIR=$SCR/.apptainer-tmp
mkdir -p "$SCR/containers" "$SCR/bin" "$ISAAC_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
source ~/.bashrc.pi05align   # uv + scratch cache paths from setup_gilbreth.sh
module load apptainer 2>/dev/null || true

echo "host glibc: $(ldd --version | head -1)"

# 1. Container with Isaac Sim's system libraries (~20 GB, 20-60 min). Public on NGC; if the pull asks
#    for credentials, create an NGC API key and run:
#    export APPTAINER_DOCKER_USERNAME='$oauthtoken' APPTAINER_DOCKER_PASSWORD=<your NGC key>
[ -f "$SIF" ] || apptainer pull "$SIF" docker://nvcr.io/nvidia/isaac-sim:5.0.0
cp "$(command -v uv)" "$SCR/bin/uv"   # static binary, usable inside the container

# 2. sim-evals (DROID scenes for Isaac Lab) + its assets from Hugging Face
[ -d "$SCR/sim-evals" ] || git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git "$SCR/sim-evals"
[ -d "$SCR/sim-evals/assets" ] || (cd "$SCR/sim-evals" && \
  uvx --from huggingface_hub hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets)

# 3. sim-evals Python env (Isaac Lab 2.2 + Isaac Sim 5.0 wheels, ~15 GB). Built on the host (which has
#    git for sim-evals' GitHub dependencies) but targeting the container's glibc 2.35, then run in the
#    container; /scratch is mounted at the same path in both, so the venv works unchanged.
cd "$SCR/sim-evals"
uv sync --python-platform x86_64-manylinux_2_35
uv pip install --python .venv/bin/python --python-platform x86_64-manylinux_2_35 opencv-python-headless imageio imageio-ffmpeg
apptainer exec --cleanenv --home "$ISAAC_HOME" --bind "$SCR" "$SIF" \
  "$SCR/sim-evals/.venv/bin/python" -c "import cv2, mediapy, openpi_client, isaaclab; print('sim-evals env OK inside container')" \
  || echo "WARNING: import test failed on the login node (no GPU here); the CHECK=1 job is the real test"

# 4. DROID joint-position checkpoints for the inspector (openpi 'polaris' configs), ~12 GB each
cd "$SCR/openpi"
if ! uv run --no-sync python -c "from openpi.training import config as c; c.get_config('pi05_droid_jointpos_polaris'); c.get_config('pi0_fast_droid_jointpos_polaris')"; then
  echo "Your openpi checkout predates the DROID-sim configs; updating it."
  git pull && GIT_LFS_SKIP_SMUDGE=1 uv sync
fi
for c in pi05_droid_jointpos_polaris pi0_fast_droid_jointpos_polaris; do
  uv run --no-sync python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/polaris/$c'))"
done

touch "$ISAAC_HOME/.eula_accepted"
echo "Isaac setup done. Next: sbatch --export=ALL,CHECK=1 isaac/run_isaac.slurm"
