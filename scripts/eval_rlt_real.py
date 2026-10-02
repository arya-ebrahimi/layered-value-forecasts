"""Standalone evaluation of a trained RLT stage-2 policy — server side.

Runs the SAME deterministic deployment policy the training loop's periodic eval uses
(`train_rlt_libero.evaluate`: actor mean, no exploration noise, reference always
provided) but as a one-shot job against a saved checkpoint, with no rollout
collection, no replay buffer and no TD3 updates.

Works against either backend:
  * real robot  — pass --real_robot_ports, then run examples/franka_real/eval_pi05_real.py on the robot
    machine (it speaks the same bridge protocol as train_pi05_real.py).
  * LIBERO sim  — omit --real_robot_ports.

Every component is reused from train_rlt_libero.py rather than reimplemented, so the
number this prints is directly comparable to the `EVAL rlt` line in a training log.

THREE pieces of state must come from the run that produced the checkpoint, not from
defaults -- each is silent (no error) if wrong, and each invalidates the result:
  1. TD3 actor weights            <- <step_dir>/actor
  2. z_rl normalizer statistics   <- <step_dir>/zrl_norm.npz  (the actor consumes
     STANDARDIZED z_rl; a fresh RunningNorm is an identity map, feeding it raw z_rl
     with the tokenizer's ~46.5 offset)
  3. the RL-canonical ActionSpace <- <run_dir>/action_space.npz  (maps actor output
     back into the VLA's normalized action space; a mismatched map silently rescales
     every action, and the gripper column especially)
(2) and (3) are looked up automatically; --strict_state (default) aborts if either is
missing rather than reporting a meaningless success rate.

Example (real robot, 20 episodes):
    uv run scripts/eval_rlt_real.py \\
        --checkpoint_dir checkpoints/pi05_rlt_only_franka_raw/rlt_franka_raw/9999 \\
        --config_name pi05_rlt_only_franka_raw \\
        --td3_checkpoint checkpoints/rlt_pi05_rlt_only_franka_raw/libero_10/book_eef_rl2/200 \\
        --eval_episodes 20 --total_num_envs 1 --real_robot_ports 8000 \\
        --task_prompts "0:pick up the book and place it in the book holder" \\
        --critic_input_norm --also_vla_baseline
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
import sys
import time

# train_rlt_libero performs EGL/XLA setup at import time and MUST be imported before jax
# is touched -- which is why this block is deliberately not import-sorted. It also
# re-exports every helper used below, so eval shares one implementation with training.
sys.path.insert(0, str(Path(__file__).parent))
import train_rlt_libero as T  # noqa: I001, N812

import jax
import numpy as np
import tyro

from openpi.training.rlt.live_monitor import LiveMonitor

logger = logging.getLogger("eval_rlt_real")
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class EvalConfig:
    # ── what to load ─────────────────────────────────────────────────────────
    checkpoint_dir: str = ""
    """Stage-1 checkpoint (frozen VLA + trained RLTokenizer) -- same value the
    training run used for --checkpoint_dir."""
    config_name: str = "pi05_rlt_only_franka_raw"
    td3_checkpoint: str = ""
    """Stage-2 TD3 checkpoint to evaluate: either a step dir (.../book_eef_rl2/200)
    or the run dir (.../book_eef_rl2), in which case the latest step is used."""
    action_space_path: str = ""
    """Defaults to <run_dir>/action_space.npz (the run dir being td3_checkpoint's
    parent when a step dir was given)."""
    zrl_norm_path: str = ""
    """Defaults to <step_dir>/zrl_norm.npz."""
    strict_state: bool = True
    """Abort when the normalizer or ActionSpace is missing. Disable ONLY to inspect a
    checkpoint saved before those were persisted, and treat the number as invalid."""

    # ── what to run ──────────────────────────────────────────────────────────
    eval_episodes: int = 20
    also_vla_baseline: bool = False
    """Additionally evaluate the raw VLA reference (no RL) for a same-session baseline.
    Doubles the episode count -- on a real robot that is 2x the human resets."""
    vla_baseline_only: bool = False
    """Evaluate ONLY the raw VLA reference and skip the RLT pass entirely. This is the
    SFT checkpoint's own number: vla_only executes the reference chunk, so no actor,
    ActionSpace or z_rl normalizer is involved. --td3_checkpoint becomes optional --
    omit it to grade a bare stage-1 checkpoint. Prefer --also_vla_baseline when you
    are evaluating an RLT checkpoint anyway: same session and same scene makes the
    two arms paired rather than two runs on different days."""
    seed: int = 0
    """Eval trial seed. CHANGE IT between runs you intend to compare as independent
    samples; keep it fixed to re-run the identical trial sequence."""
    results_json: str = ""
    """Optional path to append a one-line JSON record of the result."""

    # ── live critic / z_rl monitoring ────────────────────────────────────────
    monitor: bool = True
    """Per-chunk critic value and z_rl summary: printed here, relayed to the robot
    terminal over the bridge, and saved to <monitor_dir>/eval_trace.npz. Purely
    observational — the executed action is unaffected, so the success rate is
    identical either way."""
    monitor_dir: str = ""
    """Where eval_trace.npz and the live PNG go. Defaults to <step_dir>/eval_monitor."""
    save_plots: bool = False
    """Keep one PNG per episode under <monitor_dir>/ep_<n>_<success|failure>.png (Q per
    chunk, the z_rl change rate, and the full z_rl heatmap).

    Independent of --live_plot, which only ever rewrites a single live_monitor.png. The
    figure is rendered at episode END on the monitor's own thread, so it costs one
    ~150 ms draw per episode and never lands between a chunk's observation and its
    action -- unlike --save_frames_dir, this does not perturb the control period."""
    live_plot: bool = False
    """Also rewrite <monitor_dir>/live_monitor.png every --plot_every chunks. Open it
    in any auto-reloading image viewer to watch Q and z_rl evolve during an episode."""
    plot_every: int = 1
    monitor_q_reduce: str = "min"
    save_frames_dir: str = ""
    """Write per-episode camera frames as PNGs here, SERVER-side (empty = off).

    The trainer is the server: the robot only relays obs over the bridge and keeps
    nothing, so this is where frames have to be captured. Layout matches a training run
    exactly, and the two arms land in separate directories automatically:
        <dir>/eval_it0000_{vla,rlt}_p<pass>_s<slot>_<success|failure>/step_<t>_<base|wrist>.png

    Frames are buffered during the episode and written at its end, the same as
    `collect()` does -- so encoding never sits between a chunk's observation and its
    action, and the stride costs disk rather than control-loop time. Memory is the
    trade: ~1.4 GB per episode per slot at stride 1, 800 steps, 640x480. Point it at
    $SCRATCH."""
    save_frames_every: int = 1
    """Stride in control steps between saved frames (1 = every step). Divides both the
    file count and the buffer, since only the steps that will be written are kept."""
    """'min' (TD3's own pessimistic estimate) or 'mean' over the critic ensemble."""

    # ── env (must match the training run) ─────────────────────────────────────
    suite: str = "libero_10"
    task_id: int = 0
    total_num_envs: int = 1
    max_episode_steps: int = 480
    action_chunk: int = 10
    action_env_dim: int = 7
    num_denoise_steps: int = 10
    include_velocity: bool = True
    subsample_stride: int = 2
    rl_start_step: int = 0
    real_robot_ports: list[int] = dataclasses.field(default_factory=list)
    task_prompts: str = "0:perform the task"

    # ── TD3 architecture (must match training, else the restore mismatches) ──
    hidden: tuple[int, ...] = (256, 256)
    n_critics: int = 2
    critic_input_norm: bool = False
    action_clip: float = 1.0
    rlt_width: int = 0
    """Mirrors train_rlt_libero's --rlt_width (0 = use the config's own). Training passes
    it to load_pi0_model; eval did not, so a run whose tokenizer width was overridden
    loaded at the config's width here and produced a different z_rl."""
    gvf_mode: str = "critic"
    """Mirrors --gvf_mode. Only "actor"/"both" change the DEPLOYED action (the GVF heads
    steer it); under "critic" the actor is plain -Q + BC and this is inert. Exposed so a
    steered run cannot be silently evaluated unsteered."""
    target_subset_size: int = 2
    """Mirrors --target_subset_size. Affects the target backup, not the deployed action,
    so it matters only for the monitor's Q readout -- synced for completeness."""
    from_run_config: bool = True
    """Read <run_dir>/run_config.json (written by train_rlt_libero.main) and adopt every
    field that defines the deployed policy, so eval reproduces training by DEFAULT.

    A flag you passed explicitly on the command line always wins; the mismatch is logged
    as a warning naming both values, so an intentional override is visible and an
    accidental one is not silent. Runs that predate the dump have no file and fall back
    to the flags, with a warning listing what you must set by hand."""
    gvf_channels: str = ""
    """Same spec string the training run used (see train_rlt_libero's --gvf_channels).

    REQUIRED when the run had GVF channels: the heads are part of the critic's param
    tree, so init_td3 must build the same number of `gvf_twins` or the restore sees a
    structure it cannot match. Only the COUNT has to be right for the restore -- the
    per-channel gamma/lam/sign/phases do not affect the deployed action under
    gvf_mode="critic", where the actor is plain -Q + BC -- but pass the training string
    verbatim so nothing depends on that being true.

    Recover the count from the checkpoint when the run dir does not record it (real-robot
    runs write no eval_log.jsonl): count the distinct `gvf_twins/<i>` indices in
    json.load(open('<step>/critic/_METADATA'))['tree_metadata'] -- one twin per channel.

    NOTE `gvf_mode` is still not exposed here and defaults to "critic". That is correct
    for every run so far, but a run trained with gvf_mode="actor" steers the deployed
    action with the GVF heads, and evaluating it through this path would silently
    deploy the unsteered policy. Add the field before evaluating such a run."""

    def to_train_config(self) -> T.Config:
        """Build the Config that train_rlt_libero's helpers expect. Only fields the
        eval path actually reads matter; the rest keep their defaults."""
        return T.Config(
            checkpoint_dir=self.checkpoint_dir,
            config_name=self.config_name,
            seed=self.seed,
            suite=self.suite,
            task_id=self.task_id,
            total_num_envs=self.total_num_envs,
            max_episode_steps=self.max_episode_steps,
            action_chunk=self.action_chunk,
            action_env_dim=self.action_env_dim,
            num_denoise_steps=self.num_denoise_steps,
            include_velocity=self.include_velocity,
            subsample_stride=self.subsample_stride,
            rl_start_step=self.rl_start_step,
            eval_episodes=self.eval_episodes,
            real_robot_ports=self.real_robot_ports,
            task_prompts=self.task_prompts,
            hidden=self.hidden,
            n_critics=self.n_critics,
            critic_input_norm=self.critic_input_norm,
            gvf_channels=self.gvf_channels,
            gvf_mode=self.gvf_mode,
            target_subset_size=self.target_subset_size,
            rlt_width=self.rlt_width,
            action_clip=self.action_clip,
            # Not used by the eval path, but Config.__post_init__ requires an
            # output_dir; point it at the checkpoint so nothing is ever written.
            name="eval",
        )


def _resolve_paths(cfg: EvalConfig) -> tuple[Path, Path, Path, Path]:
    """-> (step_dir, run_dir, action_space_path, zrl_norm_path), all None when there is
    no TD3 checkpoint to resolve (baseline-only mode)."""
    if not cfg.td3_checkpoint:
        if cfg.vla_baseline_only:
            logger.info("no --td3_checkpoint: grading the raw VLA reference only")
            return None, None, None, None
        raise ValueError("--td3_checkpoint is required")
    step_dir, step = T._resolve_td3_dir(cfg.td3_checkpoint)
    run_dir = step_dir.parent
    asp = Path(cfg.action_space_path) if cfg.action_space_path else run_dir / "action_space.npz"
    znorm = Path(cfg.zrl_norm_path) if cfg.zrl_norm_path else step_dir / "zrl_norm.npz"
    logger.info(f"TD3 checkpoint: {step_dir} (step {step})")
    return step_dir, run_dir, asp, znorm


def _write_results_json(cfg: EvalConfig, results: dict, step_dir: Path | None, step: int) -> None:
    if not cfg.results_json:
        return
    rec = {
        "step": step,
        "td3_checkpoint": str(step_dir) if step_dir is not None else "",
        "seed": cfg.seed,
        "eval_episodes": cfg.eval_episodes,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(cfg.results_json, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"appended result -> {cfg.results_json}")



# Fields that define the DEPLOYED policy or the environment it runs in. Everything else
# in EvalConfig is eval's own business (how many episodes, which checkpoint, monitoring,
# frames) and is deliberately NOT synced.
_PARITY_FIELDS = (
    "config_name",
    "checkpoint_dir",
    "rlt_width",
    "action_chunk",
    "action_env_dim",
    "subsample_stride",
    "max_episode_steps",
    "rl_start_step",
    "include_velocity",
    "num_denoise_steps",
    "hidden",
    "n_critics",
    "target_subset_size",
    "critic_input_norm",
    "action_clip",
    "gvf_channels",
    "gvf_mode",
    "suite",
    "task_id",
    "task_prompts",
)


def _explicit_on_cli(field: str) -> bool:
    """Did the user actually type this flag? tyro renders a bool as --f / --no-f and
    accepts both `--f v` and `--f=v`, and underscores and dashes interchangeably."""
    names = {f"--{field}", f"--no-{field}", f"--{field.replace('_', '-')}", f"--no-{field.replace('_', '-')}"}
    return any(a in names or any(a.startswith(n + "=") for n in names) for a in sys.argv[1:])


def _apply_run_config(cfg: EvalConfig, run_dir) -> EvalConfig:
    """Adopt the training run's own config so eval reproduces it by default.

    Precedence: an explicitly typed flag wins, but the disagreement is logged. Anything
    you did not type comes from the file. This is what makes "evaluate checkpoint X"
    a one-argument operation instead of a manual reconstruction from critic/_METADATA.
    """
    if not cfg.from_run_config or run_dir is None:
        return cfg
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        if not cfg.checkpoint_dir:
            # Without the file AND without the flag, the next failure is
            # Config.__post_init__'s "--checkpoint_dir is required", which says nothing
            # about the actual cause. Name it here instead.
            raise ValueError(
                f"no {path}, so nothing can be read back from the training run, and "
                f"--checkpoint_dir was not given either.\n"
                f"This run predates the config dump. Pass the training values by hand:\n"
                f"  --checkpoint_dir <stage-1 ckpt>  --config_name <config>\n"
                f"  --n_critics N  [--critic_input_norm]  --gvf_channels '<spec>'\n"
                f"  --max_episode_steps M  --task_prompts '0:<prompt>'\n"
                f"Recover the architecture from the checkpoint itself: count the distinct "
                f"critics/<i> keys (-> n_critics) and gvf_twins/<i> keys (-> one per GVF "
                f"channel), and look for x_enc_norm/a_enc_norm (-> --critic_input_norm), in "
                f"json.load(open('<step>/critic/_METADATA'))['tree_metadata']. "
                f"Runs launched after the dump was added need none of this."
            )
        logger.warning(
            f"no {path} — this run predates the config dump. The architecture cannot be "
            f"read back, so --n_critics / --critic_input_norm / --gvf_channels / "
            f"--max_episode_steps must be set BY HAND to the training values (count the "
            f"critics/<i> and gvf_twins/<i> keys in <step>/critic/_METADATA). Two of "
            f"those are silent when wrong."
        )
        return cfg
    train = json.loads(path.read_text())

    taken, overridden = {}, {}
    updates = {}
    for f in _PARITY_FIELDS:
        if f not in train:
            continue
        want = train[f]
        if isinstance(getattr(cfg, f), tuple) and isinstance(want, list):
            want = tuple(want)
        if getattr(cfg, f) == want:
            continue
        if _explicit_on_cli(f):
            overridden[f] = (getattr(cfg, f), want)
        else:
            updates[f] = want
            taken[f] = want
    if taken:
        logger.info(f"run_config.json -> matching training: {taken}")
    for f, (mine, theirs) in overridden.items():
        logger.warning(f"--{f}={mine!r} overrides the training value {theirs!r} — eval is NOT reproducing training")
    return dataclasses.replace(cfg, **updates) if updates else cfg


def main(cfg: EvalConfig) -> None:
    # Resolve paths FIRST: run_dir is where the training config lives, and t_cfg must be
    # built from the synced values, not the raw CLI ones.
    step_dir, _run_dir, asp_path, znorm_path = _resolve_paths(cfg)
    cfg = _apply_run_config(cfg, _run_dir)
    t_cfg = cfg.to_train_config()

    # ── the three pieces of run-specific state (see module docstring) ────────
    # None of it is read when only the VLA reference is executed, so a baseline-only
    # run must not be blocked by its absence.
    missing = [] if step_dir is None else [str(p) for p in (asp_path, znorm_path) if not p.exists()]
    if missing:
        msg = (
            f"missing state required to reproduce the trained policy: {missing}. "
            "Checkpoints saved before zrl_norm.npz was persisted cannot be evaluated "
            "faithfully -- the actor would run on unstandardized z_rl. Re-run training "
            "long enough to write a new checkpoint, or pass --no-strict_state to "
            "produce an INVALID number for debugging only."
        )
        if cfg.strict_state:
            raise RuntimeError(msg)
        logger.warning(msg)

    action_space = T.ActionSpace.load(asp_path) if asp_path is not None and asp_path.exists() else None
    logger.info(
        f"ActionSpace: gripper center={action_space.center[cfg.action_env_dim - 1]:+.4f} "
        f"halfwidth={action_space.halfwidth[cfg.action_env_dim - 1]:.4f}"
        if action_space is not None
        else "ActionSpace: MISSING"
    )

    # ── frozen VLA + RLT (mirrors train_rlt_libero.main) ─────────────────────
    rng = jax.random.PRNGKey(cfg.seed)
    graphdef, pi0_state, model_config, train_cfg = T.W.load_pi0_model(
        cfg.checkpoint_dir, cfg.config_name, rlt_width=(cfg.rlt_width or None)
    )
    if not getattr(model_config, "rlt_enabled", False):
        raise ValueError(f"config {cfg.config_name} has no RLT module (rlt_enabled=False)")
    rlt_state, rest_state = pi0_state.split(T.nnx_utils.PathRegex(".*rlt.*"), ...)
    rlt_state = jax.tree.map(lambda x: x.astype(T.jnp.float32), rlt_state)
    pi0_state = T.nnx.State.merge(rlt_state, rest_state)

    if train_cfg.data.assets.assets_dir and train_cfg.data.assets.assets_dir.startswith("gs://"):
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(
                train_cfg.data,
                assets=dataclasses.replace(train_cfg.data.assets, assets_dir=str(Path(cfg.checkpoint_dir) / "assets")),
            ),
        )
    data_config = train_cfg.data.create(train_cfg.assets_dirs, model_config)
    norm_stats = T._resolve_norm_stats(cfg.checkpoint_dir, data_config)
    state_norm = T.build_norm(norm_stats, data_config.use_quantile_norm, "state")
    action_norm = T.build_norm(norm_stats, data_config.use_quantile_norm, "actions")
    if state_norm is None or action_norm is None:
        raise RuntimeError("state/action norm stats missing — eval would be near-random")
    tok = T.PaligemmaTokenizer(max_len=model_config.max_token_len)

    # ── TD3 nets: init the same shapes, then restore the trained actor ───────
    proprio_dim = T.RAW_PROPRIO_DIM + (T.VEL_PROPRIO_DIM if cfg.include_velocity else 0)
    obs_dim = model_config.rlt_zrl_dim + proprio_dim
    chunk_action_dim = cfg.action_chunk * cfg.action_env_dim
    tcfg = T.td3_config(t_cfg)
    (actor_gd, critic_gd), td3_state = T._td3.init_td3(tcfg, obs_dim, chunk_action_dim, seed=cfg.seed)
    actor_apply = T._td3.build_actor_apply(actor_gd)
    critic_apply = T._td3.build_critic_apply(critic_gd)
    if step_dir is not None:
        td3_state, step = T.load_rlt_checkpoint(str(step_dir), td3_state)
    else:
        step = -1

    # ── z_rl normalizer: the piece that silently breaks everything ───────────
    if znorm_path is not None and znorm_path.exists():
        zrl_normalizer = T.RunningNorm.load(znorm_path)
        logger.info(
            f"RunningNorm restored: count={zrl_normalizer.count} "
            f"mean|.|={np.abs(zrl_normalizer.mean).mean():.3f} std|.|={zrl_normalizer.std.mean():.3f}"
        )
    else:
        zrl_normalizer = T.RunningNorm(model_config.rlt_zrl_dim)
        if not cfg.vla_baseline_only:
            logger.warning("RunningNorm MISSING -> identity; the reported number is NOT valid")
    zrl_normalizer.freeze()  # eval must never mutate the stats

    vla_forward = T.build_vla_forward(graphdef, model_config, num_steps=cfg.num_denoise_steps)

    # ── env backend ──────────────────────────────────────────────────────────
    if cfg.real_robot_ports:
        prompts = T.parse_task_prompts(cfg.task_prompts)
        channels = [T._BridgeChannel() for _ in cfg.real_robot_ports]
        for ch, port in zip(channels, cfg.real_robot_ports, strict=True):
            T.RobotBridgeServer(ch, port=port, metadata={"action_dim": model_config.action_dim}).start()
        logger.info(f"waiting for {len(channels)} robot connection(s) on {cfg.real_robot_ports} ...")
        for ch in channels:
            ch.connected.wait()
        logger.info("robot(s) connected — start eval_pi05_real.py on the robot machine")
        env_worker = T.W.RealRobotEnvWorker(
            t_cfg, tok, model_config.action_dim, state_norm, action_norm, channels, prompts
        )
    else:
        env_worker = T.W.EnvWorker(t_cfg, tok, model_config.action_dim, state_norm, action_norm)

    # ── server-side trajectory frames ────────────────────────────────────────
    frame_writer = (
        T._FrameWriter(Path(cfg.save_frames_dir), cfg.save_frames_every) if cfg.save_frames_dir else None
    )
    # ── live critic / z_rl monitor ───────────────────────────────────────────
    monitor = None
    if cfg.monitor and not cfg.vla_baseline_only:
        mdir = Path(cfg.monitor_dir) if cfg.monitor_dir else step_dir / "eval_monitor"
        monitor = LiveMonitor(
            n_envs=env_worker.batch_size,
            out_dir=mdir,
            live_plot=cfg.live_plot,
            plot_every=cfg.plot_every,
            q_reduce=cfg.monitor_q_reduce,
            save_plots=cfg.save_plots,
            print_fn=lambda s: print(s, flush=True),
        )
        logger.info(f"live monitor on -> {mdir}" + (" (live_monitor.png)" if cfg.live_plot else ""))

    # ── run ──────────────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    t0 = time.time()
    if cfg.also_vla_baseline or cfg.vla_baseline_only:
        rng, brng = jax.random.split(rng)
        results["vla_baseline"] = T.evaluate(
            t_cfg,
            env_worker,
            vla_forward,
            actor_apply,
            td3_state,
            pi0_state,
            brng,
            zrl_normalizer,
            vla_only=True,
            frame_writer=frame_writer,
        )
        r = results["vla_baseline"]
        print(f"[eval] VLA baseline success={r['success_rate']:.3f} n={r['n']}", flush=True)

    if cfg.vla_baseline_only:
        print(f"[eval] elapsed={time.time() - t0:.0f}s (VLA baseline only)", flush=True)
        _write_results_json(cfg, results, step_dir, step)
        env_worker.close()
        return

    rng, erng = jax.random.split(rng)
    results["rlt"] = T.evaluate(
        t_cfg,
        env_worker,
        vla_forward,
        actor_apply,
        td3_state,
        pi0_state,
        erng,
        zrl_normalizer,
        frame_writer=frame_writer,
        action_space=action_space,
        vla_only=False,
        critic_apply=critic_apply,
        monitor=monitor,
    )
    if monitor is not None:
        # Drain queued figures first: the render thread is a daemon, so the last
        # episode's plot would be lost when the process exits.
        monitor.flush()
        monitor.save_trace()
    r = results["rlt"]
    n = r["n"]
    p = r["success_rate"]
    # Binomial 95% CI -- at n=20 the interval is wide enough that small deltas
    # between evals are noise; report it so the number isn't over-read.
    half = 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))
    print(f"[eval] RLT success={p:.3f} n={n}  (95% CI +-{half:.3f})", flush=True)
    print(f"[eval] step={step} elapsed={time.time() - t0:.0f}s", flush=True)

    _write_results_json(cfg, results, step_dir, step)

    env_worker.close()


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
