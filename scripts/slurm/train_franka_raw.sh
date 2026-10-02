#!/bin/bash
# SFT fine-tune pi05_base on the custom franka_raw dataset (100 episodes, single-arm Franka
# Panda, "pick up the book and place it in the book holder"). See examples/franka_raw/ for the
# hdf5 -> LeRobot conversion and src/openpi/training/config.py:pi05_franka_raw for the data config.
#
# One-time setup (already done as of this script's authoring, but kept here for reproducibility):
#   1. Mirror the base checkpoint locally -- compute nodes can't reach storage.googleapis.com.
#      On the login node:
#        uv run python -c "import gcsfs; fs = gcsfs.GCSFileSystem(token='anon'); \
#          fs.get('openpi-assets/checkpoints/pi05_base', 'checkpoints/pi05_base', recursive=True)"
#   2. Compute norm stats for this config (needs the dataset at $HF_LEROBOT_HOME/franka_raw):
#        HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot \
#          uv run scripts/compute_norm_stats.py --config-name pi05_franka_raw
#
# Full pi05 fine-tune (~2.3B params) needs 4 GPUs with fsdp_devices=4 (set in the config) to fit
# on 48GB L40S cards at batch_size=32 -- params+optimizer state are REPLICATED, not sharded,
# unless fsdp_devices matches the device count. Verified via smoke test (20 steps, checkpoint
# save/restore all correct) before this was pointed at a real training run.
#
# Usage: sbatch scripts/slurm/train_franka_raw.sh [exp_name] [num_train_steps]
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/train_franka_raw_%j.out
#SBATCH --error=/scratch/%u/logs/train_franka_raw_%j.err
#SBATCH --job-name=train_franka_raw
set -euo pipefail

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
# Resolves LeRobotDataset("franka_raw") to the dataset written by
# examples/franka_raw/convert_franka_raw_to_lerobot.py --root .../data/lerobot/franka_raw
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot
# wandb needs SDK >=0.22.3 for this account's longer API key format (already upgraded in the venv).
export WANDB_API_KEY=$(cat $SCRATCH/.hf_secrets/wandb_token)

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

EXP_NAME="${1:-franka_raw_sft}"
NUM_TRAIN_STEPS="${2:-}"

CMD=(uv run --active scripts/train.py pi05_franka_raw --exp-name "$EXP_NAME" --overwrite)
[ -n "$NUM_TRAIN_STEPS" ] && CMD+=(--num-train-steps "$NUM_TRAIN_STEPS")

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "${CMD[@]}"
