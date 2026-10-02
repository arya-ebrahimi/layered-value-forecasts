#!/bin/bash
# Download <hf-user>/franka_raw (hdf5, ~43GB/100 episodes) from the Hugging Face Hub and
# convert it to LeRobot dataset v2.0 format (parquet + mp4 videos) under /project.
#
# Usage: sbatch scripts/slurm/convert_franka_raw.sh
#
# Requires $SCRATCH/.hf_secrets/token (chmod 600, HF token, not committed to git).
#SBATCH --account=def-CHANGE_ME
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/logs/convert_franka_raw_%j.out
#SBATCH --error=/scratch/%u/logs/convert_franka_raw_%j.err
#SBATCH --job-name=convert_franka_raw
set -euo pipefail

mkdir -p $SCRATCH/logs

module load python/3.11.5

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_TOKEN=$(cat $SCRATCH/.hf_secrets/token)
export XDG_CACHE_HOME=$SCRATCH
# Resolves LeRobotDataset("franka_raw") for the norm-stats step below.
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

RAW_DIR=$SCRATCH/datasets/franka_raw_hdf5
LEROBOT_ROOT=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot/franka_raw
# held: latch the sparse gripper trigger into a persistent state command (LIBERO's
# convention). See the converter's module docstring for why the raw "trigger" encoding
# breaks quantile norm, the RLT ActionSpace calibration, and the BC target.
GRIPPER_ENCODING=${GRIPPER_ENCODING:-held}

# The raw hdf5 is the INPUT to the conversion and is never modified by it -- re-encoding
# the gripper column happens while reading, so an existing complete download is reusable
# as-is. Skipped automatically when all 100 episodes are already present; force with
# SKIP_DOWNLOAD=0.
N_RAW=$(find "$RAW_DIR" -maxdepth 1 -name 'episode_*.hdf5' 2>/dev/null | wc -l)
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-$([ "$N_RAW" -ge 100 ] && echo 1 || echo 0)}
if [ "$SKIP_DOWNLOAD" = "1" ]; then
  echo "=== skipping download ($N_RAW episodes already in $RAW_DIR) ==="
else
  echo "=== downloading raw hdf5 episodes ($N_RAW present) ==="
  : "${REPO_ID:?set REPO_ID, the HF dataset holding the raw hdf5 episodes}"
  python examples/franka_raw/download_franka_raw.py --repo-id "$REPO_ID" --local-dir "$RAW_DIR"
fi

# NOTE the converter rmtree's $LEROBOT_ROOT first -- the existing dataset is destroyed and
# fully rebuilt from $RAW_DIR (which is why the raw hdf5 must stay around).
echo "=== converting to LeRobot format (gripper encoding: $GRIPPER_ENCODING) ==="
python examples/franka_raw/convert_franka_raw_to_lerobot.py \
  --raw-dir "$RAW_DIR" \
  --repo-id franka_raw \
  --root "$LEROBOT_ROOT" \
  --task "pick up the book and place it in the book holder" \
  --gripper-encoding "$GRIPPER_ENCODING" \
  --fps 10

# Norm stats MUST be recomputed whenever the action encoding changes -- the gripper column's
# mean/std/quantiles are exactly what the encoding alters.
echo "=== computing norm stats ==="
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi05_franka_raw

echo "Done. LeRobot dataset written to: $LEROBOT_ROOT"
du -sh "$LEROBOT_ROOT"
echo "Next: retrain SFT (train_franka_raw.sh), then RLT stage 1 (train_rlt_token_franka_raw.sh)."
