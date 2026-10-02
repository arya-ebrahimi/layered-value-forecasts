# RLT training guide (RL Token, arXiv 2604.23073)

Two stages. Stage 1 trains the RL-token encoder–decoder on frozen-VLA embeddings from
LIBERO demos (`scripts/train.py` with an `rlt_only` config). Stage 2 freezes everything
from stage 1 and trains a lightweight TD3 actor–critic online in LIBERO
(`scripts/train_rlt_libero.py`). The tokenizer is tied to the VLA checkpoint it was
trained on — **retrain stage 1 whenever you switch VLA checkpoints**.

## Stage-1 configs

| config | VLA checkpoint | default data |
|---|---|---|
| `pi05_rlt_only_libero` | `checkpoints/29999_merged_sft` (local SFT) | `libero_10` |
| `pi05_rlt_only_libero_base` | `checkpoints/pi05_libero_base` (stock `gs://openpi-assets/checkpoints/pi05_libero`, mirrored locally) | `libero_goal`, all tasks |
| `pi05_rlt_only_libero_fewshot` | `checkpoints/few_shot_sft` (external few-shot SFT) | **goal task 5 only** (`libero_task_indices=(15,)`) |

Task selection: `--data.libero-task-indices` takes GLOBAL 0–39 indices
(libero_10=0–9, goal=10–19, object=20–29, spatial=30–39) and **wins over
`--data.libero-suite`**.

> **Gotcha:** because task indices beat the suite flag, running the fewshot config
> without explicit indices always trains on task 15 only — passing
> `--data.libero-suite libero_goal` does NOT widen it. To train suite-wide you must
> list all ten indices.

## Stage 1 — ways to run

```bash
# sbatch, base checkpoint, whole goal suite:
sbatch scripts/slurm/train_rlt_token.sh libero_goal

# sbatch, base checkpoint, single task (global index):
sbatch scripts/slurm/train_rlt_token.sh libero_goal 12

# sbatch, fewshot checkpoint, its default single task (15):
CONFIG=pi05_rlt_only_libero_fewshot sbatch scripts/slurm/train_rlt_token.sh libero_goal 15

# direct / interactive (inside salloc, venv active, from the repo root):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --active scripts/train.py pi05_rlt_only_libero_fewshot \
    --exp-name my_run --overwrite --data.libero-task-indices 10 11 12 13 14 15 16 17 18 19
```

Output: `checkpoints/<config_name>/<exp_name>/<step>/{params,assets}`; the final step
(default 9999) is what stage 2 consumes. 10k steps, batch 64; only the `rlt` subtree
trains (everything else frozen via the config's freeze filter).

## Stage 2 — ways to run

The sbatch wrapper (`scripts/slurm/train_rlt.sh`) takes positional args and env
overrides:

```bash
# base checkpoint, suite-wide stage-1 run, goal task 2, beta 0.5, named run:
sbatch scripts/slurm/train_rlt.sh libero_goal 2 0.5 my_run_name

# fewshot checkpoint (config + stage-1 checkpoint dir overridden):
CONFIG_NAME=pi05_rlt_only_libero_fewshot \
CKPT_DIR=checkpoints/pi05_rlt_only_libero_fewshot/rlt_libero_goal_task15 \
sbatch scripts/slurm/train_rlt.sh libero_goal 5 0.5 rlt_fs_t5

# extra driver flags pass through EXTRA_ARGS verbatim:
EXTRA_ARGS="--eval_episodes 64 --critic_warmup_updates 15000" \
sbatch scripts/slurm/train_rlt.sh libero_goal 5 0.5 rlt_fs_t5_long
```

Positional args: `[suite] [task_id(SUITE-LOCAL 0-9)] [bc_coef] [name]`.
Env overrides: `CONFIG_NAME`, `CKPT_DIR` (stage-1 step dir OR its parent run dir),
`EXTRA_ARGS`.

Direct / interactive (inside `salloc --gres=gpu:l40s:2 --mem=96G --cpus-per-task=8`,
after `module load python/3.11.5 cuda/12.6`, venv active, `MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl` exported):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run --active scripts/train_rlt_libero.py \
    --checkpoint_dir checkpoints/pi05_rlt_only_libero_fewshot/rlt_libero_goal_task15 \
    --config_name pi05_rlt_only_libero_fewshot \
    --suite libero_goal --task_id 5 --bc_coef 0.5 \
    --total_num_envs 16 --num_iterations 300 --name smoke
```

Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` — the JAX pool shares the GPU with the
LIBERO EGL render contexts; 0.75 OOMs at kernel-module load.

### Flags worth knowing (see `Config` in `scripts/train_rlt_libero.py` for all)

| flag | default | meaning |
|---|---|---|
| `--rl_start_step` | 0 | critical-phase handover K: VLA runs steps `< K`, RL policy + training only from `K` on (multiple of `action_chunk`; 0 = whole episode). Shrinks the bootstrap chain to `(ep_len−K)/C` hops |
| `--bc_coef` | 0.5 | β on the Eq.-5 BC norm `‖μ−ā‖²` (SUM over C·d dims) |
| `--sigma_explore` | 0.1 | fixed rollout noise std (RL-canonical units) |
| `--critic_warmup_updates` | 3000 | updates before Q-gradients reach the actor (actor trains BC-only); was 10k pre-`rl_start_step` |
| `--drop_partial_chunks` | off | store only full-chunk (n==C) windows instead of ref-tail padding terminal ones — ablation only; see the flag's docstring warning about lost success rewards |
| `--critic_input_norm` | off | per-block LayerNorm on the critic's (x, a) inputs — tests whether the critic's action-blindness (`gq_norm`≈0) is input-scale-driven (‖x‖≈47/2065 dims vs ‖a‖≈3/70 dims). Parameter-free; don't resume across a flip |
| `--success_sample_frac` | 0.0 | fraction of each batch drawn from success-episode rows (stratified replay). Counters the ~95%-failure-row imbalance on low-baseline tasks that drives the critic toward an action-independent fit. Try 0.25–0.5 |
| `--max_updates_per_iter` | 1000 | serial-loop cap; with ~360 boundary chunks/iter, UTD 5 saturates it |
| `--eval_interval` / `--eval_episodes` | 10 / 48 | deterministic eval cadence and episode count (multi-pass, n=16 is too noisy) |
| `--resume_from` | — | TD3 step dir; requires the run dir's `action_space.npz` |
| `--hidden` | 256 256 | `512 512 512` = the paper's hard-task variant |
| `--tau` / `--critic_lr` / `--actor_lr` | .005 / 3e-4 / 1e-4 | TD3 knobs |

### Phases and outputs

Iterations 1–5: warmup (execute raw VLA reference; measures `action_space.npz`).
Then critic warm-start until `critic_warmup_updates` (execute reference+noise; actor
BC-only). Then the actor phase (execute `μ(x, ā)+noise`; full TD3; eval every
`eval_interval`). Output dir: `checkpoints/rlt_<config_name>/<suite>/<name>/` —
TD3-nets step dirs + `action_space.npz`. The frozen VLA is referenced by path, not
copied.

Judge learning by the `EVAL rlt` vs the one-time `EVAL vla-baseline` line, not the
per-iteration rollout success (which includes exploration noise).

### Probing task baselines (which task to RL)

Pick tasks where the base VLA sits at ~30–80%. Probe with the plain SFT evaluator:

```bash
uv run scripts/eval_libero_sim.py --config_name pi05_libero \
    --checkpoint_dir checkpoints/pi05_libero_base --suite libero_goal --n_eval 32
```

## Recipes

1. **Stock VLA, one goal task end-to-end**
   `sbatch scripts/slurm/train_rlt_token.sh libero_goal` →
   `sbatch scripts/slurm/train_rlt.sh libero_goal 2 0.5 t2_run`
   (the suite-wide tokenizer at `checkpoints/pi05_rlt_only_libero_base/rlt_libero_goal`
   is the default `CKPT_DIR`, reusable for any goal task).
2. **Fewshot VLA, task 5 (current experiment)** — the two-command block under
   "Stage 2 — ways to run" above.
3. **Resume stage 2** — add
   `EXTRA_ARGS="--resume_from checkpoints/rlt_<config>/<suite>/<name>/<step>"`
   with the same name so it finds `action_space.npz` (buffer/optimizers re-init;
   warmup refills the buffer).
4. **β sweep** — same command, vary the third positional arg
   (`0.3 / 0.5 / 1.0`) with distinct names.
