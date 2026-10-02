# RLT + LVF: online RL fine-tuning of π₀.₅

This repository fine-tunes a frozen π₀.₅ vision-language-action model (VLA) with online
reinforcement learning, in LIBERO simulation and on a real Franka arm. It is built on
[openpi](https://github.com/Physical-Intelligence/openpi) and contains two methods:

- **RLT** — an implementation of *RL Token: Bootstrapping Online RL with
  Vision-Language-Action Models* (arXiv 2604.23073). A small transformer compresses the
  frozen VLA's image-token embeddings into a single **RL token** `z_rl`. A lightweight
  TD3 actor–critic then learns on the state `x = (z_rl, proprio)`. Its actor takes the
  VLA's reference action chunk as input and outputs a refined chunk, held near the
  reference by a behavior-cloning (BC) penalty.
- **LVF** — RLT plus **subgoal general value functions (GVFs)**. Extra critic heads
  predict task subgoal events (grasp, contact, drawer open, goal, collision), and their
  predictions are added to the TD3 critic's target as a dense, action-dependent reward
  bonus. This gives the actor credit at the few states that decide an episode, where the
  sparse success reward alone provides almost none. In simulation the events are derived
  automatically from the task's BDDL definition; on a real robot an operator labels them
  with a key press.

Both methods share one training script; LVF is RLT with GVF channels switched on.

**The pipeline is the same in simulation and on the real robot:**

| step | what | script |
|---|---|---|
| 1 | get demonstrations | LIBERO dataset (sim) / record + convert (robot) |
| 2 | fine-tune π₀.₅ on them (SFT) | `scripts/train.py pi05_…` |
| 3 | **stage 1:** train the RL token on the frozen SFT model | `scripts/train.py pi05_rlt_only_…` |
| 4 | fit the RL action box from the demos | `scripts/compute_action_space.py` |
| 5 | **stage 2:** online improvement with **RLT** or **LVF** | `scripts/train_rlt_libero.py` |
| 6 | evaluate | `EVAL` lines during training, `scripts/eval_rlt_real.py` |

---

## Contents

1. [How it works](#1-how-it-works)
2. [Installation](#2-installation)
3. [Simulation pipeline (LIBERO)](#3-simulation-pipeline-libero)
4. [Real robot pipeline (Franka)](#4-real-robot-pipeline-franka)
5. [Reading the logs](#5-reading-the-logs)
6. [Stage-2 flag reference](#6-stage-2-flag-reference)
7. [Repository layout](#7-repository-layout)
8. [Acknowledgements and license](#8-acknowledgements-and-license)

---

## 1. How it works

```
 demos ──► SFT π₀.₅ ──► Stage 1: RL-token tokenizer      (scripts/train.py, VLA frozen)
                              │  z_rl = g_φ(image-token embeddings)
                              ▼
                         Stage 2: online TD3              (scripts/train_rlt_libero.py)
                              state  x = (z_rl, proprio)
                              actor  μ(x, a_ref) → refined action chunk
                              critic Q(x, a)  [+ GVF heads ψ_k(x, a) for LVF]
```

**Stage 1** trains an encoder `g_φ` and a causal decoder on demonstration frames while
the VLA stays frozen. The encoder reads the VLA's final-layer image-token embeddings and
emits `z_rl`; the decoder reconstructs the embeddings from `z_rl` (paper Eq. 2). Only
the `rlt` sub-tree of the parameters is trained. A tokenizer is tied to the VLA
checkpoint it was trained on, so **retrain stage 1 whenever you change the VLA**.

**Stage 2** runs online in the environment. Each decision executes an action chunk of
`C = 10` steps. One frozen-VLA forward pass yields both `z_rl` and the reference chunk
`a_ref` (a deterministic ODE sample). TD3 learns a refinement on top of that:

- The actor loss is `E_{a~π}[-Q(x, a)] + β · ‖μ - a_ref‖²` (paper Eq. 5), where `β` is
  `--bc_coef`. During training the reference is dropped from the actor's input with
  probability 0.5.
- The critic is an ensemble of 5 Q-heads. Its target takes the minimum over a random
  2 of them (REDQ-style), which keeps twin-critic pessimism.
- Training proceeds in three phases:

| phase | what executes | what trains |
|---|---|---|
| warm-up (`--n_warmup_iters`, 5 iterations) | the raw VLA reference | nothing; fills the replay buffer |
| critic warm-up (`--critic_warmup_updates`, 3000 updates) | reference + exploration noise | the critic, plus the actor on BC only (it learns to imitate the reference) |
| RL | actor mean + exploration noise | full TD3, with periodic deterministic evaluation |

**LVF** adds one GVF "twin" head per subgoal channel `k`. Each head is trained by its
own TD loss on an event cumulant: 1 on the step the subgoal is first reached, after
which the channel terminates. With `--gvf_mode critic` (the default), the task critic's
target becomes

```
y = r + Σ_k sign_k · λ_k · phase_gate_k · GVF_k(x, a) + γⁿ · min_subset Q'(x', a')
```

while the actor loss stays the plain RLT loss. The GVF signal reaches the policy only
through `Q`. The terms are:

- `λ_k` is the per-channel weight.
- `sign_k` is -1 for collision and +1 otherwise.
- `phase_gate_k` restricts a channel to part of the episode (e.g. grasp before the
  object is held, collision after).

### RLT or LVF: what differs and how to choose

Both run with the same script (`scripts/train_rlt_libero.py`), the same stage-1
tokenizer and the same action box. The only switch is `--gvf_channels`:

| | **RLT** | **LVF** |
|---|---|---|
| `--gvf_channels` | omitted (the default, `''`) | `auto` (simulation) or a manual spec (real robot) |
| GVF heads | none are built | one twin head per channel |
| subgoal labels | none collected; `--auto_label` is ignored and no keyboard labeler starts | `--auto_label` (sim) or operator key presses (robot) |
| critic target | `r + γⁿ·min Q'` (sparse task reward only) | `r + Σ_k λ_k·GVF_k + γⁿ·min Q'` |
| actor loss | `-Q + β‖μ − a_ref‖²` | the same |

- **To run plain RLT, leave out `--gvf_channels`.** With channels empty, no heads,
  batch columns or labels exist, and the TD3 update is exactly the RLT baseline.
- **To run LVF, add `--gvf_channels`.** In simulation, also add `--auto_label`.
  Without event labels the heads train on all-zero targets and the bonus is zero.
- A third option, `--gvf_channels auto --gvf_auto_lam 0.0`, builds and trains the heads
  with λ = 0. The policy then learns exactly like RLT, but with the same parameter
  count, labels and compute as LVF. That is the RLT arm of our comparison (section 3,
  step 5b); use it for a controlled comparison and plain RLT otherwise.

---

## 2. Installation

### GPU machine (training)

Tested on Ubuntu 22.04 with Python 3.11 and NVIDIA L40S (48 GB) GPUs.

| job | GPUs used |
|---|---|
| π₀.₅ SFT (full fine-tune) | 4, with `fsdp_devices=4` |
| stage 1 | 1–2 |
| stage 2 | 1 (about 7 GB for the frozen bf16 VLA, plus LIBERO rendering) |

```bash
git clone --recurse-submodules <this-repo-url>
cd <repo>
git submodule update --init --recursive      # if you cloned without submodules

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl    # headless rendering for LIBERO
```

`uv sync` also installs the LIBERO dependencies (`robosuite`, `bddl`, `gym`, …). LIBERO
itself is the `third_party/libero` submodule. `stubs/` contains empty stand-ins for
`evdev` and `av`, which are imported transitively but not used.

Base π₀.₅ weights come from openpi's public bucket. Mirror them once if your compute
nodes have no internet access:

```bash
uv run python -c "import gcsfs; fs = gcsfs.GCSFileSystem(token='anon'); \
  fs.get('openpi-assets/checkpoints/pi05_base', 'checkpoints/pi05_base', recursive=True)"
```

**SLURM.** `scripts/slurm/` holds the launchers for every step. Before using them, set
`#SBATCH --account`, the `module load` lines and the venv path for your cluster. Submit
them from the repository root.

### Robot machine (real robot only)

This is the computer wired to the Franka (a real-time kernel, as libfranka requires) and
to two Intel RealSense cameras (one third-person, one on the wrist). It does not need a
GPU, and it does not need this whole repository: only the client scripts in
`examples/franka_real/` and the small `openpi-client` package.

```bash
pip install franky-control pyrealsense2 opencv-python numpy
pip install -e packages/openpi-client          # copy this folder to the robot machine
```

- [franky](https://github.com/TimSchneider42/franky) controls the arm. It bundles
  libfranka; pick a franky/libfranka version that matches your robot's system version,
  and follow franky's instructions for the real-time kernel and user permissions.
- The robot machine must reach the GPU machine on the bridge port (default `8000`). On a
  cluster this usually means an interactive allocation plus an SSH tunnel.

---

## 3. Simulation pipeline (LIBERO)

### Step 1 — data

The LIBERO demonstrations are the LeRobot dataset `physical-intelligence/libero`, which
the data loader downloads automatically. All four suites are in one dataset; tasks are
selected by **global** index:

| suite | global indices |
|---|---|
| `libero_10` | 0–9 |
| `libero_goal` | 10–19 |
| `libero_object` | 20–29 |
| `libero_spatial` | 30–39 |

For the SFT in step 2, our experiments used only **two demonstrations per task** from
this dataset (see the note in step 2).

### Step 2 — fine-tune π₀.₅ (SFT)

You can train the SFT model yourself or use openpi's public checkpoint.

**Option A — train it yourself** with openpi's `pi05_libero` config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name my_sft --overwrite
```

**Option B — use the public checkpoint.** It is what config
`pi05_rlt_only_libero_base` expects:

```bash
uv run python -c "import gcsfs; fs = gcsfs.GCSFileSystem(token='anon'); \
  fs.get('openpi-assets/checkpoints/pi05_libero', 'checkpoints/pi05_libero_base', recursive=True)"
```

Online RL helps most where the SFT policy is mediocre (roughly 30–80% success).
Measure it per task before choosing tasks:

```bash
uv run scripts/eval_libero_sim.py --config_name pi05_libero \
    --checkpoint_dir checkpoints/pi05_libero_base --suite libero_goal --n_eval 32
```

> **Our experiments used a few-shot SFT, not the full LIBERO dataset.** The π₀.₅ policy
> we improved with RLT and LVF was fine-tuned on only **two demonstrations per task**.
> The public `pi05_libero` checkpoint is trained on the full dataset and already solves
> most tasks, which leaves little room for online RL to show an effect. A two-demo SFT
> gives a policy that partly works and leaves room to improve.
>
> The few-shot checkpoint is not distributed; the configs
> `pi05_rlt_only_libero_fewshot`, `_libero_10_fewshot` and `_libero_spatial_t5/_t9`
> expect it at `checkpoints/few_shot_sft`. To reproduce the setting, fine-tune π₀.₅ on a
> copy of the LIBERO dataset reduced to two episodes per task. Any π₀.₅ LIBERO
> checkpoint works with the rest of the pipeline, as long as the stage-1 config points
> at it (`weight_loader` and `assets_dir`).

### Step 3 — stage 1: train the RL token

```bash
# a whole suite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_rlt_only_libero_base \
    --exp-name rlt_libero_goal --overwrite --data.libero-suite libero_goal

# or one task (global index; this overrides --data.libero-suite)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_rlt_only_libero_base \
    --exp-name rlt_libero_goal_task12 --overwrite --data.libero-task-indices 12

# SLURM:
sbatch scripts/slurm/train_rlt_token.sh libero_goal          # suite
sbatch scripts/slurm/train_rlt_token.sh libero_goal 12       # one task
```

- A run is 10k steps at batch 64, and its output goes to
  `checkpoints/<config>/<exp_name>/<step>/`. Stage 2 takes the run directory and uses
  the latest step.
- To use your own SFT checkpoint, copy `pi05_rlt_only_libero_base` in
  `src/openpi/training/config.py` and change its `weight_loader` and `assets_dir`.

### Step 4 — the RL action box

The actor works in an "RL-canonical" space: each action dimension is mapped affinely so
its useful range is `[-1, 1]`, and actions are clipped to `±1` before being mapped back.
The map is stored in `<run_dir>/action_space.npz`.

Without that file, stage 2 fits the map from its warm-up rollouts. That works with many
envs (e.g. 16), but with few envs the fit is unreliable: it can push reference actions
outside `[-1, 1]` and silently clip them. Fitting from the demonstrations is safer:

```bash
uv run scripts/compute_action_space.py \
    --config-name pi05_rlt_only_libero_base \
    --assets-dir checkpoints/pi05_libero_base/assets \
    --libero-task-index 12 --margin 1.5 \
    --output assets/action_space/libero_goal_task2_demo.npz

RUN=checkpoints/rlt_pi05_rlt_only_libero_base/libero_goal/my_run
mkdir -p $RUN && cp assets/action_space/libero_goal_task2_demo.npz $RUN/action_space.npz
```

`--libero-task-index` must be the task you train on (global index), and `--assets-dir`
must hold the normalization stats the policy runs with. The SLURM launcher does the copy
for you when you set `ASP_SRC=<file>`. The boxes in `assets/action_space/` were fit for
our two-demos-per-task SFT checkpoints.

### Step 5a — online improvement with RLT

This is plain RLT: there is no `--gvf_channels` flag, so no GVF heads are built (see
"RLT or LVF" in section 1).

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/train_rlt_libero.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_libero_base/rlt_libero_goal \
    --config_name pi05_rlt_only_libero_base \
    --suite libero_goal --task_id 2 \
    --bc_coef 0.5 \
    --total_num_envs 16 --action_chunk 10 --subsample_stride 2 \
    --num_iterations 300 --eval_interval 10 --eval_episodes 48 \
    --name my_run
```

- `--task_id` is **suite-local** (0–9).
- Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`. JAX shares the GPU with LIBERO's EGL render
  contexts, and larger fractions cause CUDA out-of-memory errors.
- `--total_num_envs` is the number of LIBERO subprocesses. `--rollout_num_envs`
  collects with fewer envs than evaluation uses.
- The output goes to `checkpoints/rlt_<config_name>/<suite>/<name>/`.
- SLURM: `sbatch scripts/slurm/train_rlt.sh <suite> <task_id> <bc_coef> <name>`, with the
  environment variables `CONFIG_NAME`, `CKPT_DIR`, `ASP_SRC` and `EXTRA_ARGS` (passed
  verbatim).
- Resume with `--resume_from <run_dir>/<step>` and the same `--name`. The networks and
  the `z_rl` normalizer are restored; the replay buffer and optimizers are not, so the
  warm-up iterations refill the buffer.

**Tuning:**

- `--bc_coef` (β) matters most; watch `force_ratio` (section 5).
- On low-success tasks, `--success_sample_frac 0.25` draws a quarter of each batch from
  successful episodes.
- `--rl_start_step K` lets the VLA run the first `K` steps and trains RL only on the
  rest; `K` must be a multiple of the chunk length.
- `--hidden 512 512 512` is the paper's larger network for hard tasks.

### Step 5b — online improvement with LVF

Add GVF channels. In simulation the subgoal events are read from the BDDL predicates on
MuJoCo state, which requires `--auto_label`:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/train_rlt_libero.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_libero_base/rlt_libero_goal \
    --config_name pi05_rlt_only_libero_base \
    --suite libero_goal --task_id 2 \
    --bc_coef 0.2 \
    --gvf_channels auto --gvf_auto_lam 0.3 --auto_label \
    --total_num_envs 16 --action_chunk 10 --subsample_stride 2 \
    --num_iterations 300 --eval_interval 10 --eval_episodes 48 \
    --name my_lvf_run
```

**What `--gvf_channels auto` creates.** It parses the task's `.bddl` file and creates
one channel per derived subgoal. It logs the resolved spec at startup.

| channel | meaning | sign | active phase |
|---|---|---|---|
| `grasp` | the manipulated object is grasped | + | before the grasp |
| `contact` | the gripper touches the object | + | before the grasp |
| `open` | the destination's open/on affordance | + | whole episode |
| `goal` | the goal conjunction holds | + | whole episode |
| `collide` | contact with fixtures the task never asks the arm to touch | − | after the grasp |

A predicate that the scene cannot evaluate becomes a constant-zero channel, with a
warning.

**Related flags:**

- `--gvf_auto_lam` sets λ for every auto channel, and `--gvf_auto_gamma` (default 0.95)
  sets each GVF's discount.
- For a manual spec, use `name:key:gamma:lam:sign:phases` entries separated by `;`, for
  example `--gvf_channels 'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1'`. Add
  `--phase_channels grasp` to name the channel whose first event advances the phase.

**What λ means.** The bonus is dense, while the task reward is non-zero on only a few
percent of transitions, so the bonus can dominate the value scale (up to about
λ/(1-γ)). Watch `gvf_bonus_frac` and `q_mean`. The GVF terms switch on one critic
warm-up's worth of updates after the critic warm-up ends; set `--gvf_actor_start_step`
to change that. The heads' TD losses train from the first update.

**Alternative mechanisms** (for comparison): `--gvf_mode actor`, `both` and
`lookahead`, plus `--gvf_freeze` / `--gvf_freeze_after_successes N`.

**Settings of our RLT-vs-LVF comparison.** All runs used the two-demos-per-task SFT, its stage-1
tokenizer and a demo-fitted box, with:

```
--total_num_envs 1 --action_chunk 10 --subsample_stride 2
--num_iterations 141 --eval_interval 45 --eval_episodes 48
```

The arms differ only in:

| arm | flags |
|---|---|
| RLT | `--bc_coef 0.1 --gvf_channels auto --gvf_auto_lam 0.0 --auto_label` |
| LVF | `--bc_coef 0.2 --gvf_channels auto --gvf_auto_lam 0.3 --auto_label` |

The RLT arm keeps the GVF heads with λ = 0, so its parameters, labels and compute match
LVF and only the shaping differs. Pool results over `--seed`s: one 48-episode
evaluation has a standard error of about ±0.07.

### Step 6 — evaluate

- During training, compare the `EVAL rlt success=…` lines (the deterministic actor mean)
  with the one-time `EVAL vla-baseline` line.
- From a saved checkpoint:

```bash
uv run scripts/eval_rlt_real.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_libero_base/rlt_libero_goal \
    --config_name pi05_rlt_only_libero_base \
    --td3_checkpoint checkpoints/rlt_pi05_rlt_only_libero_base/libero_goal/my_run/140 \
    --suite libero_goal --task_id 2 --eval_episodes 48 --total_num_envs 16 \
    --n_critics 5 --also_vla_baseline
```

`eval_rlt_real.py` has its own defaults (e.g. `--n_critics 2`). Pass every
architecture flag the run used: `--n_critics`, `--hidden`, `--critic_input_norm`,
`--gvf_channels`, and `--rlt_width` if set.

---

## 4. Real robot pipeline (Franka)

The real-robot path uses the same scripts as simulation. The differences are:

- you record and convert your own demonstrations;
- the SFT and stage-1 configs are per task;
- stage 2 runs `--real_robot_ports`, which replaces the LIBERO env pool with a websocket
  bridge to a client on the robot machine;
- success, failure and (for LVF) subgoal events come from a human operator.

The configs for our tasks (`book_placement`, `book_extended`, `franka_raw`, `hammer`,
`ram`) are included as templates.

**Conventions used throughout** (from `examples/franka_raw/convert_franka_raw_to_lerobot.py`
and `src/openpi/policies/franka_policy.py`):

- **state** (7 dims): end-effector position (3), axis-angle orientation (3), gripper
  width (1).
- **action** (7 dims): 6 arm deltas and 1 gripper command.
- **cameras:** a third-person camera and a wrist camera, at about 10 Hz.

### Step 1 — collect your demonstrations

Record teleoperated demonstrations of the task with your own teleoperation setup (the
recording code is not part of this repo). Write one HDF5 file per episode, named
`episode_*.hdf5`, with this layout (T = number of steps):

| key | shape | content |
|---|---|---|
| `action` | (T, 7) float32 | dims 0–5: arm motion command; dim 6: gripper trigger (-1 close, 0 no-op, +1 open) |
| `observations/qpos` | (T, 7) | joint angles (stored, not used as state) |
| `observations/qvel` | (T, 7) | joint velocities |
| `observations/gpos` | (T, 1) | gripper width in metres |
| `observations/ee_pos_t` | (T, 3) | end-effector position |
| `observations/ee_pos_q` | (T, 4) | end-effector quaternion, **(x, y, z, w)** order, as franky returns it |
| `observations/images/ext1` | (T, 480, 640, 3) uint8 | third-person camera |
| `observations/images/wrist` | (T, 480, 640, 3) uint8 | wrist camera |
| `tm` | (T, 1) | per-step dt |

Our tasks used 50–100 demonstrations each. Optionally push the raw episodes to a
Hugging Face dataset; `examples/franka_raw/download_franka_raw.py --repo-id <user>/<dataset>`
downloads them back (set `HF_TOKEN` for private datasets).

### Step 2a — convert to LeRobot format

```bash
export HF_LEROBOT_HOME=$PWD/data/lerobot
uv run examples/franka_raw/convert_franka_raw_to_lerobot.py \
    --raw-dir /path/to/my_task_hdf5 \
    --repo-id my_task \
    --root $HF_LEROBOT_HOME/my_task \
    --task "pick black book and place it in book holder" \
    --gripper-encoding held --fps 10
```

- `--task` is the language instruction. Use the **same string** in every later step:
  the SFT `default_prompt`, the stage-1 config and stage 2's `--task_prompts`.
- `--gripper-encoding held` (the default) turns the sparse gripper trigger into a held
  open/close command on every frame. Keep it. The sparse form breaks normalization,
  the RL action box and the BC target, so the actor would never learn to close the
  gripper.
- The converter **deletes and rebuilds** `--root`, so keep the raw HDF5 files.
- SLURM: `REPO_ID=<user>/<dataset> DATASET_NAME=my_task TASK="…" sbatch --export=ALL
  scripts/slurm/convert_franka_hdf5.sh`. Do not put the task string inside
  `--export=…,TASK=…`, because Slurm splits that list on commas.

### Step 2b — fine-tune π₀.₅ (SFT)

Add an SFT config to `src/openpi/training/config.py` by copying `pi05_book_placement`:

```python
TrainConfig(
    name="pi05_my_task",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotFrankaDataConfig(
        repo_id="my_task",                                   # = --repo-id from step 2a
        base_config=DataConfig(prompt_from_task=True),
        default_prompt="pick black book and place it in book holder",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi05_base/params"),
    num_train_steps=20_000,
    batch_size=32,
    fsdp_devices=4,
),
```

Then compute the normalization stats and train:

```bash
export HF_LEROBOT_HOME=$PWD/data/lerobot
uv run scripts/compute_norm_stats.py --config-name pi05_my_task
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_my_task --exp-name my_task_sft --overwrite
# SLURM (4 GPUs): CONFIG_NAME=pi05_my_task sbatch --export=ALL scripts/slurm/train_franka_sft.sh my_task_sft
```

Recompute the norm stats whenever the dataset or the gripper encoding changes. The
checkpoint lands in `checkpoints/pi05_my_task/my_task_sft/19999/`. Training logs to
Weights & Biases by default; run `wandb login` first, or add `wandb_enabled=False` to the
config.

**Check the SFT policy on the robot** before any RL. Serve it on the GPU machine:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_my_task --policy.dir=checkpoints/pi05_my_task/my_task_sft/19999
```

Then, on the robot machine, edit `SERVER_IP`, `FRANKA_IP`, the camera serials and
`TASK_INSTRUCTION` at the top of `examples/franka_raw/eval_franka_raw.py` and run it.
This uses the plain inference server, not the RL bridge.

### Step 3 — stage 1: train the RL token

Add a stage-1 config by copying `pi05_rlt_only_book_placement`. Point it at your SFT
checkpoint, and use the same `repo_id` and `default_prompt`:

```python
TrainConfig(
    name="pi05_rlt_only_my_task",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                               rlt_enabled=True, rlt_only=True),
    data=LeRobotFrankaDataConfig(
        repo_id="my_task",
        assets=AssetsConfig(assets_dir="checkpoints/pi05_my_task/my_task_sft/19999/assets",
                            asset_id="my_task"),
        base_config=DataConfig(prompt_from_task=True),
        default_prompt="pick black book and place it in book holder",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "checkpoints/pi05_my_task/my_task_sft/19999/params", missing_regex=".*rlt.*"),
    freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                                       rlt_enabled=True, rlt_only=True).get_freeze_filter(),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=1e-4,
                                               decay_steps=10_000, decay_lr=1e-5),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    num_train_steps=10_000, save_interval=2000, keep_period=2000, ema_decay=None,
    wandb_enabled=False,
),
```

```bash
export HF_LEROBOT_HOME=$PWD/data/lerobot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_rlt_only_my_task --exp-name my_task --overwrite
# SLURM (1 GPU): CONFIG_NAME=pi05_rlt_only_my_task sbatch --export=ALL scripts/slurm/train_rlt_token_franka.sh my_task
```

### Step 4 — the RL action box (required on hardware)

On a robot the warm-up is only a few episodes, too few to fit the action map reliably.
**Always fit it from the demonstrations:**

```bash
uv run scripts/compute_action_space.py --config-name pi05_rlt_only_my_task \
    --output assets/action_space/my_task_demo_m1.5.npz \
    --action_env_dim 7 --action_chunk 10 --max_frames 30000 --margin 1.5
```

Copy it to `<run_dir>/action_space.npz` before stage 2 starts. A real-robot run's
directory is `checkpoints/rlt_<config>/libero_10/<name>/`; the `libero_10` part comes
from the unused `--suite` default.

### Step 5 — online improvement with RLT or LVF

**1. On the GPU machine, start the trainer.** It listens on the bridge port and waits
for the robot client.

```bash
# plain RLT (no --gvf_channels)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_rlt_libero.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_my_task/my_task \
    --config_name pi05_rlt_only_my_task \
    --real_robot_ports 8000 \
    --task_prompts "0:pick black book and place it in book holder" \
    --bc_coef 0.2 \
    --total_num_envs 1 --action_chunk 10 --subsample_stride 2 \
    --critic_input_norm \
    --num_iterations 300 --eval_interval 0 --save_interval 50 \
    --dump_rollouts \
    --name my_task_rlt

# LVF: the same command, plus hand-labeled GVF channels
#   --gvf_channels 'grasp:g:0.95:0.3:+1:0;coll:c:0.95:0.3:-1:1' --gvf_mode critic
```

`scripts/slurm/train_rlt_book.sh` is the complete launcher for our book task; it also
seeds the action box. Run it inside an interactive allocation so the robot can reach
the node.

**2. On the robot machine, start the client.** Edit the constants at the top of
`examples/franka_real/train_pi05_real.py`:

- `SERVER_IP` / `SERVER_PORT`: the trainer (`localhost` if you use an SSH tunnel);
- `FRANKA_IP`;
- `BASE_CAMERA_SN` / `WRIST_CAMERA_SN`: to find the serials, uncomment
  `list_realsense_cameras()` in the script's `main()` and run it once, or use
  `rs-enumerate-devices`;
- `TASK_INSTRUCTION`.

Then run:

```bash
python examples/franka_real/train_pi05_real.py
```

The client connects, sends a "ready" ping, and from then on just executes the actions
the trainer sends and reports observations back. All learning happens on the GPU
machine.

**3. Operate.** During each episode, use these keys:

| where | key | effect |
|---|---|---|
| robot client terminal | `Enter` | end the episode as a **success** (reward 1) |
| robot client terminal | `Backspace` | end the episode as a **failure** |
| robot client terminal | `Space` | pause / unpause the arm |
| **trainer** terminal (LVF only) | channel key (`g`, `c`, …) | mark that subgoal event at the current step |

- LVF's keys come from `--gvf_channels`, so the trainer must run in an interactive
  terminal.
- In quick mode an event is stamped `--label_lookback_steps` (2) steps before the key
  press, to absorb reaction time.
- Labels are saved per episode in `<run_dir>/labels/`. Edit a label there to correct
  it, and rebuild the transitions with `--relabel_only`; this needs the run's
  `--dump_rollouts` data.

**Things to know on hardware:**

- `--sigma_explore_gripper` defaults to 0 because noise on the gripper flips it open or
  closed rather than exploring.
- If the robot hardware gets into a bad state mid-session (slipping grasps, rising
  latency), reboot it before suspecting the policy.
- `--save_frames_dir` records camera frames for videos and for re-timing labels.

### Step 6 — evaluate on the robot

```bash
# GPU machine
uv run scripts/eval_rlt_real.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_my_task/my_task \
    --config_name pi05_rlt_only_my_task \
    --td3_checkpoint checkpoints/rlt_pi05_rlt_only_my_task/libero_10/my_task_rlt/200 \
    --real_robot_ports 8000 --total_num_envs 1 --eval_episodes 20 \
    --task_prompts "0:pick black book and place it in book holder" \
    --n_critics 5 --critic_input_norm --also_vla_baseline

# robot machine
python examples/franka_real/eval_pi05_real.py --episodes 20 --server_ip <trainer-ip>
```

This runs the deterministic policy (actor mean, reference always given) with no
updates. `--also_vla_baseline` also evaluates the raw SFT policy, which gives the
before/after comparison. `--monitor_dir` adds a live plot of the critic's values. Pass
the training run's architecture flags, as in simulation.

---

## 5. Reading the logs

Every iteration prints one line:

```
[it 57] success=0.62 ep_len=301 … new=148 rew_rows=12 buffer=… env_steps=… episodes=…
        updates=740(actor 370) | critic_loss=… q_mean=… q_reward=… bc_loss=… gq_norm=… gbc_norm=…
        force_ratio=… [gvf_grasp_loss=… gvf_bonus_frac=…] | fire_grasp=… grasp_no_success=…
[it 60] EVAL rlt success=0.71 n=48
```

- **Judge progress by `EVAL rlt`** against `EVAL vla-baseline`. The per-iteration
  `success=` includes exploration noise.
- `force_ratio` is the BC force divided by the Q force. Values near 0.5–1 are healthy.
  If it stays well above 1, the anchor dominates; lower β. If it falls far below that
  range, the actor is chasing critic errors; raise β.
- `q_mean` / `target_q_mean` are the critic's value and its target. If `q_mean` keeps
  climbing past the largest achievable return, the critic is diverging.
- `q_reward` is Q on transitions that carry the success reward. If it stays low, the
  critic cannot fit the reward. If it is high while `q_mean` lags, the reward is
  propagating slowly down the bootstrap chain.
- `gq_norm` and `gbc_norm` are the action-gradient norms of the Q and BC terms. If
  `gq_norm` stays near 0, the critic ignores the action; try `--critic_input_norm`
  (set from the start of a run).
- LVF adds `gvf_<ch>_loss`, `gvf_<ch>_event_frac` (the label rate in each batch),
  `gvf_bonus_frac` (the shaping share of the reward) and `phase_frac_<p>`.
- `fire_<event>` / `count_<event>` are per-episode event rates (in simulation, for every
  run). `grasp_no_success` is the fraction of episodes that grasped but failed.

**Run directory contents:**

| path | contents |
|---|---|
| `<step>/` | TD3 networks plus `zrl_norm.npz` |
| `action_space.npz` | the RL action box |
| `eval_log.jsonl` | one line per evaluation, with raw success counts (for pooling seeds), event rates and the run's settings |
| `labels/<run_id>/<ep>.json` | per-episode event labels |
| `rollouts/` | rollout dumps (with `--dump_rollouts`) |

---

## 6. Stage-2 flag reference

Full docs: the `Config` dataclass in `scripts/train_rlt_libero.py`.

| flag | default | meaning |
|---|---|---|
| `--checkpoint_dir` / `--config_name` | — | stage-1 run directory and its config |
| `--suite` / `--task_id` | `libero_10` / 0 | LIBERO suite and suite-local task |
| `--real_robot_ports` / `--task_prompts` | — | real-robot bridge port(s) and `id:instruction` prompts |
| `--bc_coef` | 0.5 | β on `‖μ − a_ref‖²`, summed over the `C × 7` action dims |
| `--sigma_explore` / `--sigma_explore_gripper` | 0.1 / 0.0 | exploration noise (RL units); gripper noise is separate |
| `--critic_warmup_updates` | 3000 | updates during which the actor trains on BC only |
| `--utd` / `--max_updates_per_iter` | 5 / 1000 | updates per new transition, and the per-iteration cap |
| `--batch_size` / `--buffer_capacity` | 256 / 200k | replay settings |
| `--n_critics` / `--target_subset_size` | 5 / 2 | critic ensemble size and the REDQ min-subset |
| `--hidden` | 256 256 | actor/critic MLP widths |
| `--critic_input_norm` | off | dual-encoder critic: separate projections and LayerNorm for x and a |
| `--success_sample_frac` | 0.0 | fraction of each batch drawn from successful episodes |
| `--rl_start_step` | 0 | the VLA alone runs the first K steps |
| `--n_warmup_iters` / `--warmup_min_success` | 5 / 0.05 | warm-up length; abort if the base VLA succeeds less often than this |
| `--total_num_envs` / `--rollout_num_envs` | 16 / 0 | env pool size, and the envs used for collection (0 = all) |
| `--eval_interval` / `--eval_episodes` | 10 / 48 | evaluation cadence and size (0 = no evaluation) |
| `--save_interval` / `--keep_last_n` | 20 / 10 | checkpointing |
| `--gvf_channels` | `''` | `''`, `auto`, `default`, or a manual spec |
| `--gvf_auto_lam` / `--gvf_auto_gamma` | 0.3 / 0.95 | λ and γ for auto channels |
| `--gvf_mode` | `critic` | `critic` (LVF), `actor`, `both` or `lookahead` |
| `--auto_label` | off | label GVF events from simulator state (sim only) |
| `--label_lookback_steps` | 2 | key-press latency correction for manual labels |
| `--dump_rollouts` / `--relabel_only` | off | save rollout data / rebuild transitions after editing labels |
| `--save_frames_dir` / `--save_frames_every` | `''` / 1 | save camera frames |
| `--resume_from` | — | resume TD3 networks from a step directory |
| `--rlt_width` | 0 | encoder width override for stage-1 checkpoints trained at 512 |

---

## 7. Repository layout

| path | contents |
|---|---|
| `src/openpi/models/rl_token.py` | the RL-token encoder/decoder (stage 1) |
| `src/openpi/models/pi0.py`, `pi0_config.py` | π₀.₅ with the `rlt_enabled` / `rlt_only` options |
| `src/openpi/training/config.py` | all SFT and stage-1 configs (`pi05_*`, `pi05_rlt_only_*`) |
| `src/openpi/training/rlt/td3.py` | TD3 update, REDQ target, GVF shaping |
| `src/openpi/training/rlt/networks.py` | actor, critic ensemble, GVF twin heads |
| `src/openpi/training/rlt/labeling.py` | GVF channels, BDDL subgoals, auto/keyboard labeling, phases |
| `src/openpi/training/rlt/vla.py` | one frozen-VLA pass returning `(z_rl, a_ref)` |
| `src/openpi/training/rlt/envs.py` | LIBERO subprocess env pool, real-robot env, model loading |
| `src/openpi/training/rlt/robot_bridge.py` | websocket relay between the trainer and the robot client |
| `src/openpi/training/rlt/{replay,action_space,norm,events,live_monitor}.py` | replay buffer, action box, `z_rl` normalizer, event stats, live monitor |
| `src/openpi/policies/franka_policy.py` | Franka input/output transforms |
| `scripts/train.py` | SFT and stage 1 |
| `scripts/train_rlt_libero.py` | stage 2 (RLT and LVF, sim and robot) |
| `scripts/eval_rlt_real.py` | stand-alone stage-2 evaluation (sim and robot) |
| `scripts/compute_action_space.py` / `compute_norm_stats.py` | action box / normalization stats |
| `scripts/serve_policy.py`, `eval_libero_sim.py` | serve or evaluate a plain SFT policy |
| `scripts/slurm/` | cluster launchers for every step |
| `examples/franka_raw/` | download and convert robot HDF5 data; plain SFT robot evaluation |
| `examples/franka_real/` | robot-side clients for RL training and evaluation |
| `assets/action_space/` | demo-fitted action boxes from our experiments |
| `docs/rlt.md` | extra notes on stage 1 and stage 2 in simulation |

The rest of `src/openpi` is upstream openpi; see
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) for its
documentation.

---

## 8. Acknowledgements and license

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical
Intelligence. The LIBERO environment-pool layout follows
[RLinf](https://github.com/RLinf/RLinf). The RL-token method is from arXiv 2604.23073.

Apache 2.0 (`LICENSE`). Gemma-derived weights are subject to `LICENSE_GEMMA.txt`.
