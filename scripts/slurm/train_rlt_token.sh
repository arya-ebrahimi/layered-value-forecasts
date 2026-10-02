#!/bin/bash
# RLT stage 1: train the RL-token encoder-decoder on frozen-VLA embeddings.
# Uses the STOCK pi05_libero checkpoint and the short-horizon suites
# (config pi05_rlt_only_libero_base; the old SFT/libero_10 config is
# pi05_rlt_only_libero).
#
# The stock checkpoint must be mirrored locally first (compute nodes cannot reach
# storage.googleapis.com). Once, on the login node:
#   uv run python -c "import gcsfs; fs = gcsfs.GCSFileSystem(token='anon'); \
#     fs.get('openpi-assets/checkpoints/pi05_libero', 'checkpoints/pi05_libero_base', recursive=True)"
#
# Usage: sbatch train_rlt_token.sh [suite] [global_task_index]
#   suite:             libero_goal (default) | libero_object | libero_spatial
#   global_task_index: optional single task, GLOBAL 0-39 index
#                      (goal=10-19, object=20-29, spatial=30-39).
#                      Omit to train on the whole suite.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/rlt_token_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_token_%j.err
#SBATCH --job-name=rlt_token

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

SUITE=${1:-libero_goal}
TASK=${2:-}
# CONFIG env var overrides the TrainConfig (e.g. pi05_rlt_only_libero_fewshot for the
# few-shot SFT VLA — the tokenizer must be retrained per VLA checkpoint).
CONFIG=${CONFIG:-pi05_rlt_only_libero_base}
if [ -n "$TASK" ]; then
    TASK_ARGS="--data.libero-task-indices $TASK"
    EXP_NAME="rlt_${SUITE}_task${TASK}"
else
    TASK_ARGS=""
    EXP_NAME="rlt_${SUITE}"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train.py "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --overwrite \
    --data.libero-suite "$SUITE" \
    $TASK_ARGS
