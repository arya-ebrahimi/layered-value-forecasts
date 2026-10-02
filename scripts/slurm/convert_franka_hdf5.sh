#!/bin/bash
# Generalized version of convert_franka_raw.sh: download a raw franka hdf5 dataset from the
# Hugging Face Hub and convert it to LeRobot v2.0 format (parquet + mp4) under /project.
#
# Usage -- export the variables in the SUBMITTING shell, then plain `--export=ALL`:
#   REPO_ID=<hf-user>/hammer DATASET_NAME=hammer TASK="pick up the hammer, then ..." \
#     sbatch --export=ALL scripts/slurm/convert_franka_hdf5.sh
#
# Do NOT use `sbatch --export=ALL,TASK="a, b"`. Slurm splits that list on commas, so a TASK
# containing a comma is silently TRUNCATED at the first one and the remainder is dropped as a
# bogus assignment -- the dataset then gets a half prompt baked into meta/tasks.jsonl with no
# error anywhere. (Hit for real on the hammer dataset; repaired in-place afterwards.)
#
# Stages are individually skippable (DO_DOWNLOAD / DO_CONVERT / DO_NORM, each 0 or 1).
# DO_NORM requires a TrainConfig named "pi05_$DATASET_NAME" to already exist in
# src/openpi/training/config.py -- it is off by default.
#
# Requires $SCRATCH/.hf_secrets/token (chmod 600, HF token, not committed to git).
#SBATCH --account=def-CHANGE_ME
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/logs/convert_franka_hdf5_%j.out
#SBATCH --error=/scratch/%u/logs/convert_franka_hdf5_%j.err
#SBATCH --job-name=convert_franka_hdf5
set -euo pipefail

: "${REPO_ID:?set REPO_ID, e.g. <hf-user>/hammer}"
: "${DATASET_NAME:?set DATASET_NAME, e.g. hammer}"

DO_DOWNLOAD=${DO_DOWNLOAD:-1}
DO_CONVERT=${DO_CONVERT:-1}
DO_NORM=${DO_NORM:-0}
# See convert_franka_raw_to_lerobot.py: "held" latches the sparse gripper trigger into a
# persistent state command. Required for RL fine-tuning -- do not change without reading
# that module docstring.
GRIPPER_ENCODING=${GRIPPER_ENCODING:-held}
FPS=${FPS:-10}
N_EXPECTED=${N_EXPECTED:-100}

mkdir -p $SCRATCH/logs

module load python/3.11.5
source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_TOKEN=$(cat $SCRATCH/.hf_secrets/token)
export XDG_CACHE_HOME=$SCRATCH
export HF_LEROBOT_HOME=${SLURM_SUBMIT_DIR:-$PWD}/data/lerobot

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

RAW_DIR=$SCRATCH/datasets/${DATASET_NAME}_hdf5
LEROBOT_ROOT=$HF_LEROBOT_HOME/$DATASET_NAME

if [ "$DO_DOWNLOAD" = "1" ]; then
  # mkdir first: `find` on a missing dir exits 1, which under `set -o pipefail` propagates
  # through the command substitution and kills the script with no message (stderr is /dev/null).
  mkdir -p "$RAW_DIR"
  N_RAW=$(find "$RAW_DIR" -maxdepth 1 -name 'episode_*.hdf5' 2>/dev/null | wc -l) || N_RAW=0
  if [ "$N_RAW" -ge "$N_EXPECTED" ]; then
    echo "=== skipping download ($N_RAW episodes already in $RAW_DIR) ==="
  else
    echo "=== downloading $REPO_ID ($N_RAW/$N_EXPECTED present) ==="
    # snapshot_download is idempotent/resumable: only missing or changed files are fetched.
    python examples/franka_raw/download_franka_raw.py \
      --repo-id "$REPO_ID" --local-dir "$RAW_DIR"
  fi
  du -sh "$RAW_DIR"
fi

if [ "$DO_CONVERT" = "1" ]; then
  : "${TASK:?set TASK, the natural-language prompt baked into the LeRobot dataset}"
  # NOTE the converter rmtree's $LEROBOT_ROOT first -- any existing dataset there is destroyed
  # and fully rebuilt from $RAW_DIR (which is why the raw hdf5 must stay around).
  echo "=== converting to LeRobot ($DATASET_NAME, gripper: $GRIPPER_ENCODING, task: '$TASK') ==="
  python examples/franka_raw/convert_franka_raw_to_lerobot.py \
    --raw-dir "$RAW_DIR" \
    --repo-id "$DATASET_NAME" \
    --root "$LEROBOT_ROOT" \
    --task "$TASK" \
    --gripper-encoding "$GRIPPER_ENCODING" \
    --fps "$FPS"
  du -sh "$LEROBOT_ROOT"
fi

if [ "$DO_NORM" = "1" ]; then
  # Norm stats MUST be recomputed whenever the dataset or action encoding changes.
  echo "=== computing norm stats (pi05_$DATASET_NAME) ==="
  JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name "pi05_$DATASET_NAME"
fi

echo "Done. LeRobot dataset: $LEROBOT_ROOT"
