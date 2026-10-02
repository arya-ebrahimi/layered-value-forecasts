#!/bin/bash
# RLT stage 1: train the RL-token encoder-decoder on frozen-VLA embeddings, using the
# hammer SFT checkpoint (checkpoints/pi05_hammer/hammer_sft2/19999, trained by the
# pi05_hammer config). Config: pi05_rlt_only_hammer
# in src/openpi/training/config.py.
#
# Everything except the `rlt` module is frozen, so this only needs 1 GPU (unlike the 4-GPU full
# SFT fine-tune) -- the frozen backbone forward pass doesn't need FSDP-scale optimizer memory.
#
# Stage 2 (online TD3 on the frozen token) is scripts/train_rlt_libero.py --real_robot_ports,
# with --config_name pi05_rlt_only_hammer and a hammer --task_prompts.
#
# Usage: sbatch scripts/slurm/train_rlt_token_hammer.sh [exp_name]
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/rlt_token_hammer_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_token_hammer_%j.err
#SBATCH --job-name=rlt_token_hammer
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

EXP_NAME="${1:-rlt_hammer}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train.py pi05_rlt_only_hammer \
  --exp-name "$EXP_NAME" \
  --overwrite
