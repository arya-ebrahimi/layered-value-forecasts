"""Fit the RLT ActionSpace from the DEMONSTRATIONS, not from warmup rollouts.

WHY THIS EXISTS. `ActionSpace.from_ref_samples` estimates the per-dimension RL action
box from the references collected during the warm-up iterations. That is a 0.5/99.5
percentile over however many episodes the warm-up happened to run, and at small
episode counts it is a lottery: measured on libero_goal task 3, a 5-episode (1 env)
fit produced a box under which 50 of 70 action dimensions of the REAL reference range
fell outside [-1, 1], the worst by 1.94x. Because `collect()` clips the executed
action to +-1 before `from_rl` (train_rlt_libero.py), those references are not merely
coarse, they are destroyed -- the rollout executes something the VLA never asked for.
Observed consequence: the raw-reference warm-up scored its usual 0.20 success (that
path bypasses the action space entirely) and then EVERY subsequent episode failed,
0/7 on reference+noise and 0/23 on the actor, for three hours, silently.

The natural fix in simulation is "collect more warm-up episodes", which is free with
16 parallel envs. On hardware it is not: episodes cost minutes of robot time and
supervision, and the run must start from a handful of them.

So take the estimate from the data instead. The distribution the reference policy
emits IS, by construction, the distribution of the demonstrations the VLA was
fine-tuned on -- hundreds of episodes, already on disk, task-conditioned, and
available before the robot moves at all. This script reads those actions through the
SAME transform stack the policy sees (repack -> data_transforms -> Normalize), so the
samples land in the identical VLA-normalized space as the rollout-time `ref_flat`,
and hands them to the same `from_ref_samples` estimator with the same margin and
discrete-column handling. Nothing about the box's SHAPE changes; only the sample it
is estimated from.

The alternative considered and rejected was to make the box the VLA's own normalized
action range (center 0, halfwidth ~margin), which can never clip. It destroys
resolution where it matters: on libero_goal task 3 the reference spans only +-0.046 /
+-0.050 / +-0.079 in normalized units on the three wrist dimensions, so a unit box is
21x / 20x / 13x too coarse there and `sigma_explore=0.1` in RL units becomes 2.7x that
dimension's entire useful range -- exploration would randomize wrist orientation. The
narrow per-dim box is real information; it just has to be estimated from enough data.

USAGE
    uv run scripts/compute_action_space.py <config_name> --output <run_dir>/action_space.npz

train_rlt_libero.main() loads `<output_dir>/action_space.npz` if it exists and skips
the warm-up fit entirely, so dropping the file into the run directory is the whole
integration. Writing one file and reusing it across arms also removes a confound: every
arm then shares one identical action map rather than each fitting its own.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.rlt.action_space import ActionSpace
import openpi.transforms as _transforms

logger = logging.getLogger("compute_action_space")


class _RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@dataclasses.dataclass
class Args:
    config_name: str
    """TrainConfig whose data pipeline (and norm stats) the VLA was fine-tuned under."""
    output: str
    """Where to write action_space.npz — normally <run output_dir>/action_space.npz."""
    assets_dir: str = ""
    """Override the norm-stats assets dir (e.g. a checkpoint's own assets/). Empty uses
    the config's. The stats MUST be the ones the policy runs with: they define the
    normalized space the box lives in."""
    libero_task_index: int = -1
    """Override the config's `libero_task_indices` (-1 = use the config's own).

    LOAD-BEARING, and easy to get wrong: the data config carries the task its VLA was
    fine-tuned on, so `pi05_rlt_only_libero_fewshot` reads GLOBAL task 15 =
    libero_goal task 5 no matter which checkpoint --assets-dir points at. Fitting a
    box for libero_goal task 3 therefore needs `--libero-task-index 13` explicitly, or
    it silently estimates from the wrong task's demonstrations. Global index is
    10 + suite-local for libero_goal."""
    action_env_dim: int = 7
    """Robot action dims actually commanded, matching train_rlt_libero's flag."""
    action_chunk: int = 10
    """Chunk length C. The box is per-dim and tiled to C*action_env_dim, so this only
    has to match the training run's --action_chunk."""
    max_frames: int = 20_000
    """Frames to sample. The estimator is per-dim quantiles, so a few thousand frames
    across many episodes is plenty; what matters is EPISODE diversity, not frame count."""
    batch_size: int = 256
    num_workers: int = 0
    margin: float = 1.5
    """Wider than from_ref_samples' 1.25, because a demo-fitted box has to cover
    actions the demos never contained: the VLA extrapolates a little beyond its
    training distribution at rollout. Measured on libero_goal task 3, the real
    reference range reached |to_rl| = 1.014 under a margin-1.25 demo box (10 of 70
    dims clipped); 1.5 covers it with 0/70 clipped. The cost is 1.5-2.7x coarser
    resolution on the narrow wrist dims than a well-conditioned 16-env warmup fit --
    against the 20x a norm-stats box would cost, which is the alternative this
    approach exists to avoid."""
    min_halfwidth: float = 0.05
    trigger_raw_scale: float = 1.0
    discrete_action_dims: str = ""
    """Same `dim:raw_full_scale[:eps]` spec as train_rlt_libero. Pass the SAME value the
    training run uses, or the trigger columns get calibrated differently."""


def _collect_actions(args: Args) -> np.ndarray:
    """[N, action_horizon, action_dim] demo actions in VLA-NORMALIZED space."""
    train_cfg = _config.get_config(args.config_name)
    if args.assets_dir:
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(
                train_cfg.data, assets=dataclasses.replace(train_cfg.data.assets, assets_dir=args.assets_dir)
            ),
        )
    if args.libero_task_index >= 0:
        train_cfg = dataclasses.replace(
            train_cfg, data=dataclasses.replace(train_cfg.data, libero_task_indices=(args.libero_task_index,))
        )
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    logger.info(f"task filter: {data_config.task_indices}")
    if data_config.repo_id is None:
        raise ValueError(f"config {args.config_name!r} has no repo_id — nothing to read demos from")
    if data_config.norm_stats is None:
        raise ValueError(
            f"config {args.config_name!r} has no norm stats under "
            f"{args.assets_dir or train_cfg.data.assets.assets_dir!r}. The box must be fitted in the "
            f"SAME normalized space the policy emits, so the stats are required — run "
            f"scripts/compute_norm_stats.py first, or point --assets_dir at the checkpoint's assets/."
        )

    dataset = _data_loader.create_torch_dataset(data_config, train_cfg.model.action_horizon, train_cfg.model)
    # repack -> data_transforms -> Normalize: exactly the chain data_loader.py builds,
    # minus model_transforms (tokenization/images), which do not touch `actions`.
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            _RemoveStrings(),
        ],
    )
    n = len(dataset)
    take = min(args.max_frames, n)
    # Strided rather than random: a contiguous head would sample a handful of episodes,
    # and EPISODE diversity is the whole point (it is what the warm-up fit lacks).
    idx = np.linspace(0, n - 1, take).astype(int) if take < n else np.arange(n)
    logger.info(f"{args.config_name}: {n} frames in {data_config.repo_id}, sampling {take}")
    out = [np.asarray(dataset[int(i)]["actions"], np.float32) for i in tqdm.tqdm(idx, desc="actions", unit="frame")]
    return np.stack(out)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    acts = _collect_actions(args)  # [N, H, action_dim]
    ad, chunk = args.action_env_dim, args.action_chunk
    if acts.shape[-1] < ad:
        raise ValueError(f"dataset actions have {acts.shape[-1]} dims < --action_env_dim {ad}")
    # from_ref_samples reshapes to per-dim internally, so the chunk length here only
    # has to be self-consistent; tile one frame's horizon into the [N, C*ad] it expects.
    per_dim = acts[..., :ad].reshape(-1, ad)
    refs = np.repeat(per_dim, chunk, axis=0).reshape(-1, chunk * ad)

    train_cfg = _config.get_config(args.config_name)
    ns = None
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.norm_stats is not None and "actions" in data_config.norm_stats:
        s = data_config.norm_stats["actions"]
        ns = (
            ("quantile", np.asarray(s.q01), np.asarray(s.q99))
            if data_config.use_quantile_norm
            else ("zscore", np.asarray(s.mean), np.asarray(s.std))
        )

    from train_rlt_libero import parse_discrete_action_dims

    discrete = parse_discrete_action_dims(args.discrete_action_dims, ad) if args.discrete_action_dims else None
    space = ActionSpace.from_ref_samples(
        refs,
        ad,
        margin=args.margin,
        action_norm=ns,
        trigger_raw_scale=args.trigger_raw_scale,
        min_halfwidth=args.min_halfwidth,
        discrete_dims={d: v[0] for d, v in discrete.items()} if discrete else None,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    space.save(out)

    # The diagnostic that would have caught the 1-env failure at fit time instead of
    # three hours in: how much of the demo action range this box can actually express.
    rl = space.to_rl(refs)
    frac = float(np.mean(np.abs(rl) > 1.0))
    logger.info(f"wrote {out}")
    logger.info(f"  halfwidth[:{ad}] = {np.round(space.halfwidth[:ad], 4)}")
    logger.info(f"  center[:{ad}]    = {np.round(space.center[:ad], 4)}")
    logger.info(f"  demo actions outside [-1,1] under this box: {100 * frac:.3f}%  (want ~0)")
    if frac > 0.01:
        logger.warning(
            f"{100 * frac:.1f}% of demo action components fall outside the box and would be CLIPPED at "
            f"execution — raise --margin or check --discrete_action_dims."
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(tyro.cli(Args))
