#!/bin/bash
# Standby-only (free GPUs). Each array task runs a slice of the conditions.
#   sbatch run_align.slurm                          # full benchmark, 8 shards
#   sbatch --array=0 --export=ALL,EXTRA="--conditions ctl_stove --num-trials 3" run_align.slurm   # smoke test
#SBATCH -J pi05-align
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a30,a10,a100-40gb,a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --array=0-7
#SBATCH -o logs/%x_%A_%a.out
set -uo pipefail

SCR=/scratch/gilbreth/$USER
source ~/.bashrc.pi05align
OPENPI=$SCR/openpi
ALIGN=$SCR/pi05-align
OUT=${OUT:-$ALIGN/results/pi05_libero}
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
PORT=$((20000 + RANDOM % 20000))   # random port: other users may share the node
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) shard=$SHARD port=$PORT"

# 1. pi0.5 policy server (JAX). Cap its GPU memory so MuJoCo's EGL renderer fits too.
cd "$OPENPI"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run --no-sync scripts/serve_policy.py --port $PORT \
    policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero &
SERVER=$!

# wait for the port (model load + JIT takes a few minutes); abort if the server dies
until (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "policy server died"; exit 1; }
  sleep 10
done
echo "server up"

# 2. LIBERO client (Python 3.8), headless GPU rendering via EGL
source "$OPENPI/examples/libero/.venv/bin/activate"
export PYTHONPATH=$OPENPI/third_party/libero:$ALIGN
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python "$ALIGN/align_eval.py" --port $PORT --shard "$SHARD" --out-dir "$OUT" \
    --time-budget-min 220 ${EXTRA:-}
STATUS=$?

kill $SERVER
exit $STATUS
