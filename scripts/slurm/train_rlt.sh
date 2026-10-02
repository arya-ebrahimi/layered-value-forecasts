#!/bin/bash
# RLT stage 2: online TD3 on the frozen RL token (scripts/train_rlt_libero.py).
# Usage: sbatch train_rlt.sh [suite] [task_id] [bc_coef] [name]
#   suite:   libero_goal (default) | libero_object | libero_spatial | libero_10
#   task_id: SUITE-LOCAL task index 0-9 (default 0)
#   bc_coef: beta of the BC regularizer (default 0.5; sweep {0.3, 0.5, 1.0})
#   name:    optional run name (default rlt_<suite>_t<task>_b<beta>)
# checkpoint_dir defaults to the stage-1 suite-wide run for <suite>; override with
# CKPT_DIR=... sbatch ... for a per-task stage-1 run.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/logs/rlt_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_%j.err
#SBATCH --job-name=rlt

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
TASK=${2:-0}
BETA=${3:-0.5}
NAME=${4:-rlt_${SUITE}_t${TASK}_b${BETA}}
# CONFIG_NAME/CKPT_DIR env vars override the stage-1 config + checkpoint (e.g. the
# few-shot SFT variant: CONFIG_NAME=pi05_rlt_only_libero_fewshot CKPT_DIR=...).
# EXTRA_ARGS: appended verbatim to the driver command, e.g.
#   EXTRA_ARGS="--tau 0.01 --critic_lr 1e-3 --max_updates_per_iter 2000"
CONFIG_NAME=${CONFIG_NAME:-pi05_rlt_only_libero_base}
CKPT_DIR=${CKPT_DIR:-checkpoints/${CONFIG_NAME}/rlt_${SUITE}}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ASP_SRC: seed <run dir>/action_space.npz from a DEMO-FITTED box before the driver
# starts. Without it the run fits its own box from the warm-up rollouts, and with
# --total_num_envs 1 that fit is a handful of episodes -- a 5-episode fit once put 50
# of 70 reference dims outside [-1,1], and collect() clips to +-1 before from_rl, so
# every executed action was silently corrupted. Sharing ONE file across arms also
# removes a confound: each arm then has an identical action map instead of its own.
# The driver loads the file when present and skips the warm-up fit entirely.
ASP_SRC=${ASP_SRC:-}
if [ -n "$ASP_SRC" ]; then
    RUN_DIR="checkpoints/rlt_${CONFIG_NAME}/${SUITE}/${NAME}"
    mkdir -p "$RUN_DIR"
    if [ ! -f "$RUN_DIR/action_space.npz" ]; then
        if [ ! -f "$ASP_SRC" ]; then
            echo "ERROR: ASP_SRC=$ASP_SRC not found — run compute_action_space.py first" >&2
            exit 1
        fi
        cp "$ASP_SRC" "$RUN_DIR/action_space.npz"
        echo "seeded $RUN_DIR/action_space.npz from $ASP_SRC (demo-fitted)"
    fi
fi

# MEM_FRACTION 0.5: stage 2 is single-device JAX (~7GB bf16 weights + small batches)
# sharing the GPU with the LIBERO EGL render contexts — 0.75 starves the CUDA driver
# of memory for loading compiled kernels (CUDA_ERROR_OUT_OF_MEMORY at first forward).
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run --active scripts/train_rlt_libero.py \
    --checkpoint_dir "$CKPT_DIR" \
    --config_name "$CONFIG_NAME" \
    --suite "$SUITE" \
    --task_id "$TASK" \
    --bc_coef "$BETA" \
    --total_num_envs 1 \
    --action_chunk 10 \
    --subsample_stride 2 \
    --num_iterations 300 \
    --eval_interval 10 \
    --save_interval 100 \
    --name "$NAME" \
    $EXTRA_ARGS
