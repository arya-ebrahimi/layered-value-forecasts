#!/bin/bash
# Standalone evaluation of an SFT (non-RL) Pi0.5 checkpoint across a whole LIBERO suite.
#
# Exists because the only baseline numbers we had were incidental: every RLT stage-2 run
# logs ONE 48-episode `vla_baseline` eval, so the reference policy was only ever measured
# on the one task that run trained, and only at n=48. This evaluates every task in a suite
# in one job, at a sample size worth quoting.
#
# Usage: sbatch eval_sft_libero.sh [suite] [n_eval] [ckpt] [config] [name]
#   suite   libero_goal (default) | libero_10 | libero_object | libero_spatial
#   n_eval  episodes per task (default 50 -> +-0.07 SE per task, +-0.02 pooled)
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/logs/eval_sft_%j.out
#SBATCH --error=/scratch/%u/logs/eval_sft_%j.err
#SBATCH --job-name=eval_sft
set -euo pipefail
mkdir -p $SCRATCH/logs
module load python/3.11.5
module load cuda/12.6
source $SCRATCH/openpi_env/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

SUITE=${1:-libero_goal}
NEVAL=${2:-50}
CKPT=${3:-checkpoints/few_shot_sft}
CONFIG=${4:-pi05_libero}
NAME=${5:-sft_${SUITE}}

# Videos off by default: n_eval x n_tasks of them is a lot of disk for a number.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/eval_libero_sim.py \
    --checkpoint_dir "$CKPT" \
    --config_name "$CONFIG" \
    --suite "$SUITE" \
    --n_eval "$NEVAL" \
    --no-save_videos
