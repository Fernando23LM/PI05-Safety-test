#!/bin/bash
# Isaac Sim (sim-evals DROID) + VLA inspector, standby on A10 only (Isaac Sim needs RT cores; A30/A100 have none).
# Submit from /scratch/gilbreth/$USER/pi05-align:
#   sbatch --export=ALL,CHECK=1 isaac/run_isaac.slurm                          # first: does Isaac Sim run here?
#   sbatch --export=ALL,MODEL=pi05 isaac/run_isaac.slurm                       # scene 1, default instruction
#   sbatch --export=ALL,MODEL=fast,EXTRA="--scene 2 --instruction 'put the can in the mug' --episodes 3" isaac/run_isaac.slurm
#   sbatch --export=ALL,MODEL=pi05,EXTRA="--tasks-file isaac/tasks.example.json" isaac/run_isaac.slurm
# MODEL=pi05 -> pi0.5-DROID (flow matching), MODEL=fast -> pi0-FAST-DROID (autoregressive tokens)
# N_SAMPLES (default 16): samples per decision for the choice distribution; lower it if the GPU runs out of memory.
#SBATCH -J isaac-inspect
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
set -uo pipefail

SCR=/scratch/gilbreth/$USER
ALIGN=$SCR/pi05-align
SIF=$SCR/containers/isaac-sim-5.0.0.sif
ISAAC_HOME=$SCR/isaac-home
source ~/.bashrc.pi05align
module load apptainer 2>/dev/null || true
[ -f "$ISAAC_HOME/.eula_accepted" ] || { echo "run isaac/setup_isaac_gilbreth.sh first (with ACCEPT_NVIDIA_EULA=1)"; exit 1; }
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"

# Isaac Sim inside the container; host /scratch is mounted at the same path.
isaac() {
  apptainer exec --nv --cleanenv --home "$ISAAC_HOME" --bind "$SCR" \
    --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" "$SCR/sim-evals/.venv/bin/python" "$@"
}
cd "$ALIGN"   # relative paths in EXTRA (e.g. isaac/tasks.example.json) resolve here; sim-evals finds its assets itself

if [ -n "${CHECK:-}" ]; then
  isaac "$ALIGN/isaac/isaac_check.py" --scene "${SCENE:-1}" --out "$ALIGN/results/isaac/check"
  exit $?
fi

MODEL=${MODEL:-pi05}
case $MODEL in
  pi05) CFG=pi05_droid_jointpos_polaris ;;
  fast) CFG=pi0_fast_droid_jointpos_polaris ;;
  *) echo "MODEL must be pi05 or fast"; exit 1 ;;
esac
PORT=$((20000 + RANDOM % 20000))
mkdir -p "$ALIGN/logs"

# 1. inspector server (openpi env, host). JAX gets ~45% of the 24 GB A10; Isaac Sim uses the rest.
(cd "$SCR/openpi" && exec env XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run --no-sync python "$ALIGN/isaac/inspector_server.py" \
    --config $CFG --checkpoint gs://openpi-assets/checkpoints/polaris/$CFG --port $PORT \
    --n-samples "${N_SAMPLES:-16}") > "$ALIGN/logs/server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
until (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -30 "$ALIGN/logs/server_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
echo "server up ($CFG, port $PORT)"

# 2. Isaac Sim client. The first run on a node compiles shaders (can take 10-20 min).
eval "isaac \"$ALIGN/isaac/inspect_eval.py\" --port $PORT --model-tag $MODEL --out-dir \"$ALIGN/results/isaac\" ${EXTRA:-}"
STATUS=$?
kill $SERVER
exit $STATUS
