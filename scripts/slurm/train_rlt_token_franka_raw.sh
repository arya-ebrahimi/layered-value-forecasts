#!/bin/bash
# RLT stage 1: train the RL-token encoder-decoder on frozen-VLA embeddings, using the
# franka_raw SFT checkpoint (checkpoints/pi05_franka_raw/book/19999, trained on the 100-episode
# "pick up the book and place it in the book holder" dataset). Config: pi05_rlt_only_franka_raw
# in src/openpi/training/config.py.
#
# Everything except the `rlt` module is frozen, so this only needs 1 GPU (unlike the 4-GPU full
# SFT fine-tune) -- the frozen backbone forward pass doesn't need FSDP-scale optimizer memory.
#
# Stage 2 (online TD3 on the frozen token) would need a franka_raw analogue of
# scripts/train_rlt_libero.py -- not yet ported (that script is LIBERO-env-specific for rollouts).
#
# Usage: sbatch scripts/slurm/train_rlt_token_franka_raw.sh [exp_name]
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/rlt_token_franka_raw_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_token_franka_raw_%j.err
#SBATCH --job-name=rlt_token_franka_raw
set -euo pipefail

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot
export WANDB_API_KEY=$(cat $SCRATCH/.hf_secrets/wandb_token)

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

EXP_NAME="${1:-rlt_franka_raw}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train.py pi05_rlt_only_franka_raw \
  --exp-name "$EXP_NAME" \
  --overwrite
