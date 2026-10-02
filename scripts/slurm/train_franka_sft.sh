#!/bin/bash
# Generalized version of train_franka_raw.sh: SFT fine-tune pi05_base on a converted franka
# hdf5 dataset (see convert_franka_hdf5.sh for the hdf5 -> LeRobot conversion).
#
# Usage:
#   sbatch --export=ALL,CONFIG_NAME=pi05_hammer scripts/slurm/train_franka_sft.sh [exp_name] [steps]
#
# Prerequisites (both one-time per dataset):
#   1. checkpoints/pi05_base mirrored locally -- compute nodes can't reach storage.googleapis.com.
#      On the login node:
#        uv run python -c "import gcsfs; fs = gcsfs.GCSFileSystem(token='anon'); \
#          fs.get('openpi-assets/checkpoints/pi05_base', 'checkpoints/pi05_base', recursive=True)"
#   2. Norm stats for $CONFIG_NAME (convert_franka_hdf5.sh DO_NORM=1, or directly:
#        HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot \
#          uv run scripts/compute_norm_stats.py --config-name $CONFIG_NAME)
#
# Full pi05 fine-tune (~2.3B params) needs 4 GPUs with fsdp_devices=4 (set in the config) to fit
# on 48GB L40S cards at batch_size=32 -- params+optimizer state are REPLICATED, not sharded,
# unless fsdp_devices matches the device count.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/train_franka_sft_%j.out
#SBATCH --error=/scratch/%u/logs/train_franka_sft_%j.err
#SBATCH --job-name=train_franka_sft
set -euo pipefail

: "${CONFIG_NAME:?set CONFIG_NAME, e.g. pi05_hammer}"

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
# Resolves LeRobotDataset("<name>") to data/lerobot/<name> written by the converter.
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot
export WANDB_API_KEY=$(cat $SCRATCH/.hf_secrets/wandb_token)

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

EXP_NAME="${1:-${CONFIG_NAME}_sft}"
NUM_TRAIN_STEPS="${2:-}"

CMD=(uv run --active scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --overwrite)
[ -n "$NUM_TRAIN_STEPS" ] && CMD+=(--num-train-steps "$NUM_TRAIN_STEPS")

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "${CMD[@]}"
