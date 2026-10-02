#!/bin/bash
# RLT stage 2 on the REAL ROBOT, book_extended task (TWO-STAGE: place the black book in
# the holder, then move the purple book from the holder into the next slot).
#
# Twin of train_rlt_book.sh -- same arch, same knobs, same bridge protocol. Only the
# task-specific pieces differ, and each difference is called out below.
#
# Trainer side only. The robot desktop runs examples/franka_real/train_pi05_real.py and connects to $PORT over
# the RL bridge (README.md, "Real robot pipeline"); during training the operator uses
#   ENTER     episode end, SUCCESS (both books placed)
#   BACKSPACE episode end, manual FAILURE
#   SPACE     pause / unpause the arm
#   g / c     grasp / collision GVF event for the BLACK book  (+1 / -1)
#   h / v     grasp / collision GVF event for the PURPLE book (+1 / -1)
# `p` and `u` are reserved by the labeler itself (pause, undo).
#
# Usage:  bash scripts/slurm/train_rlt_book_extended.sh [name] [bc_coef] [port]
#   (run it inside salloc, or sbatch it -- but the robot desktop must be able to reach
#    the allocated node on $PORT, so an interactive salloc + tunnel is the usual path.)
# Env overrides: CONFIG_NAME, CKPT_DIR, PROMPT, ASP_SRC, FRAMES_EVERY (0=off),
#                GVF_CHANNELS, PHASE_CHANNELS, MAX_EP_STEPS, EXTRA_ARGS.
#SBATCH --account=def-CHANGE_ME
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
# 12 h, not train_rlt_book.sh's 8: episodes here are ~1.7x longer (see MAX_EP_STEPS),
# and book_placement already measured ~71 s/iteration at ~357 steps. Scaling the arm and
# table-fill portions puts this near ~105 s/iteration, i.e. ~9 h for 300 iterations.
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/rlt_book_ext_%j.out
#SBATCH --error=/scratch/%u/logs/rlt_book_ext_%j.err
#SBATCH --job-name=rlt_book_ext

mkdir -p $SCRATCH/logs

module load python/3.11.5
module load cuda/12.6

source $SCRATCH/openpi_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

NAME=${1:-book_extended_rl1}
BETA=${2:-0.2}
PORT=${3:-8000}

CONFIG_NAME=${CONFIG_NAME:-pi05_rlt_only_book_extended}
CKPT_DIR=${CKPT_DIR:-checkpoints/pi05_rlt_only_book_extended/rlt_book_extended/9999}
# Must match the prompt pi05_book_extended was fine-tuned with, or the frozen VLA is
# conditioned off-distribution and z_rl encodes a task it never saw.
PROMPT=${PROMPT:-pick black book and place it in book holder, then pick purple book from book holder and place it in the next slot in book holder}
EXTRA_ARGS=${EXTRA_ARGS:-}

# EPISODE LENGTH. The default 480 is a book_placement number and is WRONG here: the
# book_extended demos average 585 steps (46,787 frames / 80 episodes at 10 fps) because
# the task is two manipulations, not one. At 480 the cap would fire before the second
# book could be placed on a majority of episodes -- every one of them entering the buffer
# as a truncation with reward 0, and the policy never seeing a success to learn from.
# 800 keeps roughly the 1.4x headroom over the demo mean that 480 gives book_placement.
# Must stay EVEN so the stride-2 grid lands on the final step.
MAX_EP_STEPS=${MAX_EP_STEPS:-800}

# GVF CHANNELS. Four channels, two per stage: a grasp predictor and a collision
# predictor for the black book, then the same pair for the purple book.
#   grasp  (g)  the BLACK book is grasped
#   coll   (c)  collision while handling the black book
#   grasp1 (h)  the PURPLE book is grasped
#   coll1  (v)  collision while handling the purple book
# g/c/h/v are all free: the bridge reserves ENTER, backspace and space, and the labeler
# reserves `p` (pause) and `u` (undo).
#
# PHASE_CHANNELS must name channels that EXIST in GVF_CHANNELS -- naming one that does
# not is the ValueError this script hit -- and every name advances the episode one phase,
# so N names give phases 0..N. `grasp` alone gives two phases, which is what the gating
# above uses (every channel is gated to phase 0 or phase 1, nothing references a phase 2).
#
# NOTE the purple-stage channels are currently gated the same way as the black-stage ones
# (grasp1 -> phase 0, coll1 -> phase 1). If grasp1 is meant to predict the PURPLE grasp,
# phase 0 is the wrong gate: phase 0 ends at the first (black) grasp, so the channel is
# switched off exactly during the stage it describes. Two consistent alternatives:
#   two phases, purple channels live in stage 2:
#     'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1;grasp1:h:0.95:0.3:+1:1;coll1:v:0.95:0.3:-1:1'
#     PHASE_CHANNELS=grasp
#   three phases, one per grasp:
#     'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1,2;grasp1:h:0.95:0.3:+1:1;coll1:v:0.95:0.3:-1:2'
#     PHASE_CHANNELS=grasp,grasp1
# lam 0.3 and the 0.95 channel discount match train_rlt_book.sh.
GVF_CHANNELS=${GVF_CHANNELS:-'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1;grasp1:h:0.95:0.3:+1:0;coll1:v:0.95:0.3:-1:1'}
PHASE_CHANNELS=${PHASE_CHANNELS:-grasp}

# Camera frames: OFF by default. They are written synchronously after each episode ends
# (a PNG encode inside the step loop would stretch the control period), so they land
# squarely in the gap between episodes -- measured at 83 s and 450 MB per 365-step
# episode, and this task's episodes are longer still. Set FRAMES_EVERY=N to enable at a
# stride of N control steps; 5 gives 2 Hz against a ~10 Hz loop, enough to re-time a
# mislabeled g/c/h/v event against video.
FRAMES_EVERY=${FRAMES_EVERY:-0}
if [ "$FRAMES_EVERY" -gt 0 ]; then
    FRAME_ARGS="--save_frames_dir $SCRATCH/rlt_frames/${NAME} --save_frames_every $FRAMES_EVERY"
else
    FRAME_ARGS=""
fi

# The RL action box comes from the 80 DEMO episodes, not from the warm-up rollouts: on
# hardware the warm-up is a handful of episodes and its 0.5/99.5 percentile fit is a
# lottery (see scripts/compute_action_space.py's docstring). train_rlt_libero loads
# <output_dir>/action_space.npz when present and skips the warm-up fit entirely.
#
# MARGIN 1.5, matching book_placement. The checked-in book_extended_demo.npz is margin
# 2.0 (back-solve the gripper column against the norm stats to verify) -- at 2.0 a
# saturated actor asks the arm for twice the largest motion any human demo contained.
# Regenerate with:
#   uv run scripts/compute_action_space.py --config-name $CONFIG_NAME \
#       --output $ASP_SRC --action_env_dim 7 --action_chunk 10 --max_frames 50000 --margin 1.5
ASP_SRC=${ASP_SRC:-assets/action_space/book_extended_demo_m1.5.npz}
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
    --max_episode_steps "$MAX_EP_STEPS" \
    --critic_input_norm \
    --gvf_channels "$GVF_CHANNELS" \
    --phase_channels "$PHASE_CHANNELS" \
    --gvf_mode critic \
    --num_iterations 300 \
    --eval_interval 0 \
    --save_interval 50 \
    --dump_rollouts \
    $FRAME_ARGS \
    --name "$NAME" \
    $EXTRA_ARGS
