#!/bin/bash
# One-time setup on a Gilbreth LOGIN node (no GPU needed). Takes ~30-60 min (~36 GB of checkpoints).
#   bash setup_gilbreth.sh
# Everything goes to /scratch (home quota is too small for the checkpoint + envs).
set -euo pipefail

SCR=/scratch/gilbreth/$USER
export UV_CACHE_DIR=$SCR/.uv-cache            # uv's package cache
export UV_PYTHON_INSTALL_DIR=$SCR/.uv-python  # uv-managed Python 3.11 / 3.8
export OPENPI_DATA_HOME=$SCR/openpi-cache     # where openpi stores downloaded checkpoints
mkdir -p "$SCR"

# 1. uv (Python package manager used by openpi)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# 2. openpi (policy server side, Python 3.11 + JAX)
cd "$SCR"
[ -d openpi ] || git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. LIBERO client venv (Python 3.8, MuJoCo/robosuite), as in examples/libero/README.md
if [ ! -d examples/libero/.venv ]; then
  uv venv --python 3.8 examples/libero/.venv
  source examples/libero/.venv/bin/activate
  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  uv pip install -e packages/openpi-client
  uv pip install -e third_party/libero
  # LIBERO asks an interactive question on first import; answer "n" to use default paths.
  echo n | PYTHONPATH=third_party/libero python -c "import libero.libero" || true
  deactivate
fi

# 4. Pre-download the pi0.5 LIBERO checkpoint (~12 GB) so compute nodes don't need to.
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"

# 5. Checkpoints for the knows-vs-does probe (probe_vlm.py), ~12 GB each. Skip with NO_PROBE_CKPTS=1.
if [ -z "${NO_PROBE_CKPTS:-}" ]; then
  for c in pi05_base pi05_droid; do
    uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/$c'))"
  done
fi

cat > ~/.bashrc.pi05align <<EOF
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR
export OPENPI_DATA_HOME=$OPENPI_DATA_HOME
export PATH=\$HOME/.local/bin:\$PATH
EOF
echo "Setup done. Env vars saved in ~/.bashrc.pi05align"
