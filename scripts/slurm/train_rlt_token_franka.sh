#!/bin/bash
# RLT stage 1 (generalized): train the RL-token encoder-decoder on frozen-VLA embeddings for
# any franka task. Generalizes train_rlt_token_{franka_raw,hammer}.sh -- pass the config name.
#
# Usage:
#   CONFIG_NAME=pi05_rlt_only_ram sbatch --export=ALL \
#     scripts/slurm/train_rlt_token_franka.sh [exp_name]
#
# The config's weight_loader must point at a COMPLETED SFT checkpoint (its .../19999/params),
# and its AssetsConfig at that checkpoint's own assets dir -- so this must run after the SFT.
#
# Everything except the `rlt` module is frozen (Pi0Config.get_freeze_filter), so this only needs
# 1 GPU (unlike the 4-GPU full SFT fine-tune) -- the frozen backbone forward pass doesn't need
# FSDP-scale optimizer memory.
#
# Stage 2 (online TD3 on the frozen token) is scripts/train_rlt_libero.py --real_robot_ports,
# with --config_name $CONFIG_NAME and a matching --task_prompts.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/rlt_token_franka_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_token_franka_%j.err
#SBATCH --job-name=rlt_token_franka
set -euo pipefail

: "${CONFIG_NAME:?set CONFIG_NAME, e.g. pi05_rlt_only_ram}"

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot
export WANDB_API_KEY=$(cat $SCRATCH/.hf_secrets/wandb_token)

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

EXP_NAME="${1:-${CONFIG_NAME#pi05_rlt_only_}}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --overwrite
