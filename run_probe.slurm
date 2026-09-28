#!/bin/bash
# Knows-vs-does probe: asks each pi0.5 checkpoint's own VLM whether the instructions are OK.
# Needs camera frames from at least a few rollouts (align_eval.py saves them in results/.../frames).
#   sbatch run_probe.slurm                                              # 3 checkpoints, images
#   sbatch --export=ALL,EXTRA="--text-only" run_probe.slurm             # language-only variant
#   sbatch --export=ALL,EXTRA="--random-init --checkpoints pi05_libero --frames-per-condition 1" run_probe.slurm  # smoke test
#SBATCH -J pi05-probe
#SBATCH -A shey
#SBATCH -q standby
#SBATCH -p a30,a10,a100-40gb,a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail

SCR=/scratch/gilbreth/$USER
source ~/.bashrc.pi05align
ALIGN=$SCR/pi05-align
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$SCR/openpi"
uv run --no-sync python "$ALIGN/probe_vlm.py" \
    --frames-dir "$ALIGN/results/pi05_libero/frames" \
    --out "$ALIGN/results/vlm_probe.jsonl" ${EXTRA:-}
