#!/bin/bash
# RLT stage 2 on the REAL ROBOT, book_placement task.
#
# Trainer side only. The robot desktop runs examples/franka_real/train_pi05_real.py and connects to
# $PORT over the RL bridge (README.md, "Real robot pipeline"); during training the operator uses
#   ENTER     episode end, SUCCESS
#   BACKSPACE episode end, manual FAILURE
#   SPACE     pause / unpause the arm
#   g         grasp GVF event (+1)
#   c         collision GVF event (-1)
#
# Arch mirrors the last real book run (book_gvf_lam0.5): 5 critics, dual-encoder
# critic (--critic_input_norm), grasp+coll GVF channels at lam 0.5 in `critic`
# mode. What changed: the stage-1 tokenizer / frozen VLA is now the book_placement
# one (prompt "pick black book and place it in book holder"), and bc_coef is 0.2.
#
# Usage:  bash scripts/slurm/train_rlt_book.sh [name] [bc_coef] [port]
#   (run it inside salloc, or sbatch it — but the robot desktop must be able to
#    reach the allocated node on $PORT, so an interactive salloc + tunnel is the
#    usual path.)
# Env overrides: CONFIG_NAME, CKPT_DIR, PROMPT, ASP_SRC, FRAMES_EVERY (0=off), EXTRA_ARGS.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/rlt_book_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_book_%j.err
#SBATCH --job-name=rlt_book

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

# rl1 diverged on the self-loop bug (fixed in _fill_tables); its buffer, labels and
# m=2.0 action_space.npz are all stale, so start a NEW run rather than resume it.
NAME=${1:-book_placement_rl2}
# 0.5, not the 0.2 originally chosen: the rl1 log shows beta=0.5 held force_ratio
# (BC force / Q force) at 1.14 through it 16 and the policy was succeeding 6 of 10
# iterations. The anchor only lost once Q diverged and dQ/da grew with it, which the
# _fill_tables fix addresses at the source -- so there is no evidence 0.5 was too
# strong, and weakening it would confound the fix.
BETA=${2:-0.2}
PORT=${3:-8000}

CONFIG_NAME=${CONFIG_NAME:-pi05_rlt_only_book_placement}
CKPT_DIR=${CKPT_DIR:-checkpoints/pi05_rlt_only_book_placement/rlt_book_placement/9999}
# Must match the prompt pi05_book_placement was fine-tuned with, or the frozen VLA
# is conditioned off-distribution and z_rl encodes a task it never saw.
PROMPT=${PROMPT:-pick black book and place it in book holder}
EXTRA_ARGS=${EXTRA_ARGS:-}

# The RL action box comes from the 80 DEMO episodes, not from the warm-up rollouts:
# on hardware the warm-up is a handful of episodes and its 0.5/99.5 percentile fit is
# a lottery (see scripts/compute_action_space.py's docstring — a 5-episode fit put 50
# of 70 reference dims outside [-1,1], and collect() clips to +-1 before from_rl, so
# every executed action was corrupted for three hours in silence). train_rlt_libero
# loads <output_dir>/action_space.npz when present and skips the warm-up fit entirely.
# assets/action_space/book_placement_demo_m1.5.npz is the checked-in demo fit (all 80
# episodes). Regenerate with:
#   uv run scripts/compute_action_space.py --config-name $CONFIG_NAME \
#       --output $ASP_SRC --action_env_dim 7 --action_chunk 10 --max_frames 30000 --margin 1.5
#
# MARGIN 1.5, not the 2.0 the first fit used. The margin sets how much wider the RL box
# is than the demos, so RL +-1 -- which is where action_clip=1.0 saturates -- commands
# `margin` x the demos' full range. At 2.0 a saturated actor asked the arm for twice the
# largest motion any human demo contained. Measured against 57,990 stored reference rows
# from the rl1 run, tightening to 1.5 costs almost nothing: the fraction of reference
# components pushed outside the box goes 0.00031 -> 0.00041, and dim 5 (which dominates
# it) was already outside at 2.0. Do not go below 1.5 without re-measuring -- the VLA
# extrapolates past its training distribution at rollout, and a reference the box cannot
# hold is clipped at execution AND unreachable for the tanh actor.
# Reusing ONE file across arms also removes a confound: every arm then shares an
# identical action map instead of each fitting its own.
# Camera frames: OFF by default. They are written synchronously after each episode
# ends (deliberately — a PNG encode inside the step loop would stretch the control
# period), so they land squarely in the gap between episodes. Measured on this task:
# 640x480 PNGs at ~0.11 s each, 2 cameras x ~365 steps = 83 s and 450 MB PER EPISODE,
# i.e. ~7 h and ~135 GB over a 300-iteration run. Set FRAMES_EVERY=N to enable at a
# stride of N control steps (5 -> ~17 s and ~90 MB per episode, still 2 Hz against a
# 10 Hz control loop — enough to re-time a mislabeled g/c event against video).
FRAMES_EVERY=${FRAMES_EVERY:-0}
if [ "$FRAMES_EVERY" -gt 0 ]; then
    FRAME_ARGS="--save_frames_dir $SCRATCH/rlt_frames/${NAME} --save_frames_every $FRAMES_EVERY"
else
    FRAME_ARGS=""
fi

ASP_SRC=${ASP_SRC:-assets/action_space/book_placement_demo_m1.5.npz}
RUN_DIR="checkpoints/rlt_${CONFIG_NAME}/libero_10/${NAME}"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/action_space.npz" ]; then
    if [ ! -f "$ASP_SRC" ]; then
        echo "ERROR: no demo-fitted action space at $ASP_SRC — run compute_action_space.py first" >&2
        exit 1
    fi
    cp "$ASP_SRC" "$RUN_DIR/action_space.npz"
    echo "seeded $RUN_DIR/action_space.npz from $ASP_SRC (demo-fitted)"
fi

# No LIBERO env pool on the real-robot path, so the JAX pool gets the whole card.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train_rlt_libero.py \
    --checkpoint_dir "$CKPT_DIR" \
    --config_name "$CONFIG_NAME" \
    --real_robot_ports "$PORT" \
    --task_prompts "0:${PROMPT}" \
    --bc_coef "$BETA" \
    --total_num_envs 1 \
    --action_chunk 10 \
    --subsample_stride 2 \
    --critic_input_norm \
    --gvf_channels 'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1' \
    --gvf_mode critic \
    --num_iterations 300 \
    --eval_interval 0 \
    --save_interval 50 \
    --dump_rollouts \
    $FRAME_ARGS \
    --name "$NAME" \
    $EXTRA_ARGS
