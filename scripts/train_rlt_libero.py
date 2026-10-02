"""RLT online RL for LIBERO — TD3 on the frozen VLA's RL token (arXiv 2604.23073).

Stage 2 of the RL-Token recipe. Loads a pi05_rlt_only_libero checkpoint (frozen
SFT VLA + trained RLTokenizer), then trains a lightweight TD3 actor-critic whose
state is x = (z_rl, proprio) and whose actor refines the VLA's reference action
chunk (see src/openpi/training/rlt/). Single LIBERO task, sparse binary success
reward, no human interventions (sim adaptation of the paper's Alg. 1).

Loop per iteration:
  collect one env-pool rollout (warmup: execute the VLA reference; RL: actor +
  fixed exploration noise) -> post-episode batched VLA pass over the stride-2
  subsample grid -> assemble chunk transitions into the replay buffer ->
  G = utd * n_new_boundary_transitions TD3 updates (actor every policy_delay-th)
  -> periodic deterministic eval + checkpoint.

Rollouts and updates alternate serially: TD3 is off-policy (staleness-tolerant),
the env pool is already subprocess-parallel, and the MLP updates cost seconds per
iteration — the paper's async rollout/learner split is a real-robot constraint
that buys nothing in sim.

Example:
    uv run scripts/train_rlt_libero.py \\
        --checkpoint_dir checkpoints/pi05_rlt_only_libero/rlt_task0 \\
        --config_name pi05_rlt_only_libero --task_id 0 --total_num_envs 16
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import time

from openpi.training.rlt.envs import silence_deprecation_warnings

silence_deprecation_warnings()
# Stage 2 shares the GPU with the LIBERO EGL render contexts (one per env subprocess)
# and only needs ~7GB for the frozen bf16 VLA + small rollout batches — a large
# preallocated pool starves the CUDA driver (kernel-module loading OOMs).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
from openpi.training.rlt.envs import setup_egl

setup_egl()

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/libero"))

import openpi.models.model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.rlt import envs as W
from openpi.training.rlt import events as _events
from openpi.training.rlt import labeling as _lab
from openpi.training.rlt import td3 as _td3
from openpi.training.rlt.action_space import ActionSpace
from openpi.training.rlt.norm import RunningNorm
from openpi.training.rlt.replay import ReplayBuffer
from openpi.training.rlt.robot_bridge import RobotBridgeServer
from openpi.training.rlt.robot_bridge import _BridgeChannel
from openpi.training.rlt.vla import build_vla_forward

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
for _noisy in ("OpenGL", "OpenGL.acceleratesupport", "absl", "jax", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("train_rlt_libero")
logger.setLevel(logging.INFO)
logging.getLogger("rlt.envs").setLevel(logging.INFO)

RAW_PROPRIO_DIM = 8  # eef_pos(3) + eef axis-angle(3) + gripper_qpos(2), normalized
VEL_PROPRIO_DIM = 9  # robot0_joint_vel(7) + robot0_gripper_qvel(2), raw


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Config:
    # checkpoint / io
    checkpoint_dir: str = ""
    """Stage-1 checkpoint dir (pi05_rlt_only_libero run: frozen VLA + trained RLTokenizer)."""
    config_name: str = "pi05_rlt_only_libero_base"
    rlt_width: int = 0
    """Override Pi0Config.rlt_width when loading the stage-1 checkpoint (0 = use the
    config's value).

    Needed only for checkpoints trained BEFORE the default rose to 2048 (the paper's
    VLA embedding width): every stage-1 run currently on disk has a 512-wide encoder,
    and restoring one into a 2048-wide module dies at the first attention call with
    "Incompatible input dimension, got 512 but module expects 2048". The width is
    readable from the checkpoint without loading it — `params.rlt.zrl_proj.kernel.value`
    in `<step>/params/array_metadatas/*` has shape [width, zrl_dim].

    A FLAG rather than a config edit, deliberately: `rlt_width` lives on the shared
    TrainConfig, so changing it there re-shapes every run of that config name —
    including stage-1 tokenizer jobs already queued at the current default, which must
    keep producing 2048-wide encoders. This flag affects one stage-2 process only.
    Leave it at 0 for checkpoints trained after the default changed."""
    """Stage-1 TrainConfig name (pi05_rlt_only_libero_base = stock pi05_libero VLA,
    short-horizon suites; pi05_rlt_only_libero = the local SFT/libero_10 variant)."""
    output_dir: str = ""
    name: str = ""
    """Optional run name; checkpoints go to checkpoints/rlt_<config_name>/<suite>/<name>."""
    resume_from: str = ""
    """Resume TD3 nets from a saved step dir (optimizer states and the replay buffer are
    NOT persisted — optimizers re-init and warmup rollouts refill the buffer)."""
    seed: int = 42

    # env
    suite: str = "libero_10"
    task_id: int = 0
    """Single LIBERO task (index within the suite)."""
    total_num_envs: int = 16
    max_episode_steps: int = 480
    action_chunk: int = 10
    """RL chunk length C (executed sub-steps per decision; model action_horizon >= C)."""
    action_env_dim: int = 7
    group_size: int = 1  # read by EnvWorker; RLT has no grouping
    num_denoise_steps: int = 10
    """ODE steps for the VLA reference chunk (deployment sampler)."""

    rl_start_step: int = 0
    """Critical-phase handover K (paper Sec. V "full-task evaluation"): the raw VLA
    reference runs the first K env steps of every episode; the RL policy (and
    exploration noise) takes over from step K, and ONLY transitions starting at
    t >= K are stored/trained on. Concentrates the sparse terminal credit on the
    final segment: bootstrap depth shrinks from ep_len/C hops to (ep_len-K)/C, so
    early-segment Q values stop being gamma-crushed toward 0 and dQ/da carries
    signal. Episodes that succeed before K contribute nothing (the base policy
    already solved them - same as the paper). Must be a multiple of action_chunk;
    0 = whole-episode RL (the default)."""

    # RL state / data
    include_velocity: bool = True
    """Append raw joint+gripper velocities to the proprio part of the RL state (paper)."""
    subsample_stride: int = 2
    """Store transitions every `stride` env steps (paper: 2). Must divide action_chunk;
    stride=action_chunk disables subsampling. Cost: action_chunk/stride VLA forwards per
    executed chunk in the post-episode pass."""
    success_sample_frac: float = 0.0
    """Fraction of each TD3 batch drawn from success-episode rows (stratified replay
    sampling; 0 = uniform). At ~20% task success, failure episodes (truncated at 480)
    contribute ~5x more rows each, so ~95% of a uniform batch carries target ~0 and the
    critic's action-gradient collapses toward 0 everywhere. 0.25-0.5 keeps the
    value-carrying manifold represented. Targets unchanged; sampling weights only."""
    drop_partial_chunks: bool = False
    """Store only transitions whose action window is a full chunk (n == C), instead of
    padding terminal windows with the reference tail. WARNING: with stride 2 / C 10 this
    cuts reward-bearing rows ~5x, and for episodes whose length L is odd no stride-grid
    window satisfies t + C == L, so those successes' rewards are never stored at all.
    Kept as an option for ablation; the padded default is distribution-matched to the
    target actor (mu ~= ref) and empirically ignites the bootstrap chain."""
    buffer_capacity: int = 200_000
    forward_batch: int = 32
    """Micro-batch size of the post-episode VLA pass (fixed shape -> one extra jit trace)."""

    # TD3 (see rlt/td3.py). bc_coef/actor_lr and the two scheduling knobs below were
    # retuned after an observed collapse: at bc_coef=0.1 / actor_lr=3e-4 with a big
    # serial update block, the actor exploited the barely-trained critic's
    # extrapolation errors and left the VLA manifold within the first RL iteration
    # (bc_loss 3.4, q_pi 0.52 vs q_mean 0.03, success 0.69 -> 0.00).
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    utd: float = 5.0
    batch_size: int = 256
    sigma_explore: float = 0.1
    discrete_action_dims: str = ""
    """Action columns that are DISCRETE two-level commands rather than continuous
    controls, as `dim:raw_full_scale` entries separated by ',' — e.g. `5:0.1:0.02,6:1.0`
    for a +-0.1 rad yaw trigger and a +-1 gripper. Empty (the default) preserves the
    historical behaviour: the last column alone is treated as the gripper trigger.

    Declaring a column does two things, both for the same reason:

    * its ActionSpace calibration comes from the NORM STATS (RL +-1 <-> the raw
      full-scale command) instead of sample quantiles or the full-range fallback. A
      rare two-level command cannot be estimated from a finite warmup sample —
      measured on the ram insertion task, two refits of the same task gave dim 5
      ranges of [-6.628, +6.681] and [-6.305, +0.036], the latter yielding an
      asymmetric map under which a positive fire clipped to 12% of its command.
    * it is explored by HELD PULSES rather than Gaussian noise. The optional third
      field is the per-step probability of starting one: with probability `eps` the
      column is overridden with a full-scale command of random sign, held for
      ~`explore_discrete_hold` steps; otherwise the actor's own value passes through.
      Gaussian noise cannot explore such a column — reaching a level from rest needs
      sigma of order the level itself, which would wreck every continuous column
      sharing the scale, while a small sigma merely jitters a value that should be
      exactly 0 or exactly ±full-scale. Size `eps` against the demos — MEASURED, not
      guessed, because the pulse hold makes the occupancy ~eps*hold/(1+eps*hold) rather
      than eps. The ram rotation fires on 2.3% of steps in ~7 pulses per episode, and
      at hold=2 over 480 steps this knob gives:

          eps    0.005   0.010   0.012   0.020   0.030   0.050
          steps  0.94%   2.08%   2.51%   3.58%   5.81%  10.09%
          pulses   2.5     4.8     5.5     9.0    13.2    22.9

      so eps ~0.012 reproduces the demo rate and ~0.02 is a sensible slightly-richer
      default; 0.05 is 4x the demo rate and will dominate the channel. eps=0 (the
      default when the
      field is omitted) means the column is NEVER explored — right for a gripper you
      do not want flipped, wrong for a channel that decides the task.

    Find the levels in the checkpoint's norm stats: a discrete column shows up as a
    q01/q99 pair sitting at the command levels with almost no mass between them."""
    sigma_explore_discrete: float = 0.0
    """Gaussian exploration sigma for columns named in `discrete_action_dims`. 0 is the
    right default — explore those columns with the `eps` field instead, which perturbs
    them the way they are actually commanded."""
    explore_discrete_hold: float = 2.0
    """Mean length, in env steps, of an exploratory discrete pulse (geometric). Match it
    to the demos: the ram rotation pulses have median 2 / mean 2.08 steps, and a
    one-step blip at a delta controller barely moves the arm."""
    sigma_explore_gripper: float = 0.0
    """Exploration noise on the GRIPPER column only (the last of action_env_dim, at
    every chunk position). 0 = the gripper is never perturbed. Ignored for any column
    covered by `discrete_action_dims`, which takes precedence.

    Separate from `sigma_explore` because the gripper is not a continuous control. Under
    the `held` encoding the converter produces (see convert_franka_raw_to_lerobot.py) the
    channel is a SIGN, and the robot's +-0.5 threshold turns it into a discrete
    open/close command -- so Gaussian noise on a value sitting near the boundary does not
    explore, it flips the physical gripper state.

    Measured on the hammer task: executing `reference + noise` with a shared sigma of 0.1
    dropped success from 0.800 (raw reference, warmup) to 0.167 in the very next phase,
    before any actor or critic was involved. Book, whose SFT sits near the gripper sign
    boundary in 10.3% of chunks rather than hammer's 18.2%, only fell to 0.667. Set this
    to `sigma_explore` to restore the old behaviour."""
    target_noise: float = 0.1
    target_noise_clip: float = 0.3
    bc_coef: float = 0.5
    actor_q_eval: str = "sample"
    """Where the actor's Q term is evaluated (see TD3Config.actor_q_eval).

    "sample" (default) is the paper's Eq. 5, E_{a~pi}[-Q(x, a)] via a reparameterized
    sample. "mean" is the pre-Eq.5 -Q(x, mu) at the deterministic mean, kept as an
    ablation control; it reproduces the old update bit-exactly.
    """
    ref_dropout: float = 0.5
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    action_clip: float = 1.0
    hidden: tuple[int, ...] = (256, 256)
    """(512, 512, 512) for the hard-task variant (paper)."""
    n_critics: int = 5
    """Critic ensemble size (see TD3Config.n_critics).
    NOT parameter-compatible — do not resume across a change."""
    target_subset_size: int = 2
    """REDQ-style target: min over a random subset of this many heads (see
    TD3Config.target_subset_size). Keeps twin-min pessimism at n_critics=5."""
    critic_input_norm: bool = False
    """Dual-encoder critic architecture (see TD3Config.critic_input_norm /
    RltCritic in networks.py): fixes the V(s)-shortcut where the critic fits TD
    targets from state alone and dQ/da decays to ~0. NOT parameter-free — do not
    resume a run across a flip of this flag."""
    critic_warmup_updates: int = 3_000
    """Warm-start updates before Q-gradients reach the actor: the critic trains normally
    while the actor trains BC-ONLY (learning to imitate the reference, with dropout).
    Rollouts execute reference + exploration noise during this phase, so the buffer keeps
    collecting VLA-quality data while the sparse terminal reward propagates through the
    bootstrap chain. Not in the paper (their async streaming starts the actor
    immediately); it exists because the plain-MLP actor starts at mu=0. Was 10k when the
    whole-episode bootstrap chain was ~20 hops; with rl_start_step shortening the chain
    (and BC imitation converging within ~2k updates) 3k suffices."""
    max_updates_per_iter: int = 1000
    """Cap on gradient updates per iteration. The paper streams ~UTD updates per sample
    asynchronously (small policy drift between behavioral feedback); a serial loop must
    cap the block size or the actor moves thousands of steps against a static buffer."""

    # schedule
    n_warmup_iters: int = 5
    """Iterations executing the raw VLA reference chunk (buffer prefill, no updates)."""
    warmup_min_success: float = 0.05
    """Fail fast if the base VLA's warmup success rate is below this (no positive signal
    for TD3 to find). 0 disables the check."""
    min_buffer: int = 2000
    """Minimum transitions in the buffer before updates start."""
    num_iterations: int = 300
    eval_interval: int = 10
    rollout_num_envs: int = 0
    """Envs used for TRAINING rollouts (0 = all of --total_num_envs).

    Splits the two consumers of the env pool, which want different widths: eval is a
    measurement and wants many episodes in parallel, while collection is throttled by
    the VLA forward pass and by how fast TD3 can consume new transitions (utd * new
    rows per iteration), so extra rollout envs mostly deepen the buffer per iteration
    rather than speed anything up. Size --total_num_envs for EVAL and set this lower.

    Implemented as a slice of the one pool (make_pool's n_active), not a second pool:
    the idle subprocesses cost memory but no time, whereas building and tearing down
    envs per phase costs an EGL/MuJoCo rebuild every iteration."""
    eval_episodes: int = 48
    """Episodes per eval, run as ceil(eval_episodes / total_num_envs) pool passes with
    fresh random reset states. n=16 gives a +-0.10 std error on a ~0.7 success rate —
    too noisy to detect the <0.2 improvements this method targets; 48+ brings it to
    ~+-0.06 and the baseline (measured once) to a usable reference."""
    save_interval: int = 20
    keep_last_n: int = 10
    """Was 3, which pruned every step before 50 and left the CV[g]-vs-training-step
    trend (a paper figure) with only two surviving points."""
    dump_rollouts: bool = False
    """Write <output_dir>/rollouts/rollout_<iter>.npz each iteration: the per-episode
    stride-grid z_rl trace plus the success flag and ep_len. Lets the critical-phase
    peak detection be re-derived offline (it is just argmax of the summed |dz_rl|)
    without re-running rollouts — the expensive part on a real robot."""
    save_frames_dir: str = ""
    """Write per-episode camera frames as PNGs under this directory (empty = off).

    The trainer IS the server on the real-robot path: the robot desktop only relays
    obs over the bridge and keeps nothing, and the rollout npz stores z_rl / proprio /
    ref / actions — no pixels anywhere. So frames for figures (and for re-timing a
    mislabeled event against video) must be captured here, at run time; they cannot
    be recovered afterwards. Same on-disk layout as eval_pi05_real.py's
    --save_frames_dir, so the same downstream tooling reads both:
        <dir>/it<iter>_ep<ep_id>_<success|failure>/step_<t>_<base|wrist>.png
        <dir>/eval_it<iter>_<kind>_p<pass>_s<slot>_<success|failure>/...
    Cost is real — two PNGs per control step, ~800 per 400-step episode — so point
    this at $SCRATCH, not /project, and raise --save_frames_every in sim, where the
    whole env pool writes at once."""
    save_frames_every: int = 1
    """Stride in control steps between saved frames (1 = every step)."""

    # ── subgoal GVF critics + event labeling (see rlt/labeling.py, rlt/td3.py) ──
    gvf_channels: str = ""
    """Auxiliary event channels, as `name:key:gamma:lam:sign:phases` entries separated
    by ';' (phases comma-separated inside the entry), or the literal "default" for the
    paper's two: grasp (g, +1, phase 0) and coll (c, -1, phase 1). Empty (the default)
    disables the whole mechanism — no heads, no batch columns, and an update
    bit-identical to the baseline.

    A string rather than tyro's nested-dataclass form because tyro cannot populate a
    variable-length tuple of dataclasses from the command line (it parses the empty
    default fine and then rejects every --gvf-channels.N.* option).

    The literal "auto" derives the set from THIS task's BDDL goal — contact/grasp of
    the manipulated object, the destination's open/on affordance, the goal conjunction,
    and contact with the fixtures the task never asks the arm to touch. That is the
    companion to --auto_label_source predicate and needs no hand-written spec; it logs
    the resolved channels at startup. Use --gvf_auto_lam / --gvf_auto_gamma to set the
    coefficients, or paste the logged spec and edit it for per-channel control.

    Examples:
        --gvf_channels auto                             # derived from the task's BDDL
        --gvf_channels default
        --gvf_channels 'grasp:g:0.95:0.0:+1:0'          # lam 0 => auxiliary-only arm
        --gvf_channels 'grasp:g:0.95:0.3:+1:0;coll:c:0.9:0.2:-1:1'
    """
    gvf_auto_lam: float = 0.3
    """`lam` given to every channel that `--gvf_channels auto` derives."""
    gvf_auto_gamma: float = 0.95
    """`gamma` given to every channel that `--gvf_channels auto` derives."""
    gvf_mode: str = "critic"
    """How the GVF heads reach the policy (see TD3Config.gvf_mode).

        critic (default)  reward shaping: y = r + sum_k w_k*GVF_k(x, a) + discount*Q'.
                          The actor loss is the plain RLT baseline -Q + beta*||mu-ref||^2.
        actor             the original composite-steering term in the actor loss;
                          the reward is untouched.
        both              both, double-counting the same signal (diagnostic only).
        lookahead         look-ahead advice (Wiewiora et al. 2003): the critic backs up
                          F = discount*Phi(x', a') - Phi(x, a) and the actor optimizes
                          Q_0 + Phi, with Phi = sum_k w_k*GVF_k off the TARGET heads.
                          Policy-invariant for ANY lam -- the shaped and unshaped
                          optima coincide exactly -- while the actor still gets
                          d(GVF)/da directly. Needs a `next_phase` batch column, so it
                          cannot read buffers collected before that existed.

    `lam` is NOT comparable between the modes: under `critic` it prices a DENSE
    per-step bonus against a task reward that is nonzero on ~2% of rows, so the
    actor-mode default of 0.3 makes Q_task essentially a GVF predictor. Start ~0.01-0.03
    and watch the `gvf_bonus_frac` / `q_mean` log fields. Under `lookahead` the shaping
    telescopes and cannot accumulate on the value scale, so lam needs no such rescaling
    -- watch `la_F_abs_mean` and `q_eff_mean` instead."""
    gvf_ramp_steps: int = 0
    """Ramp the GVF gate linearly from 0 to 1 over this many updates past
    `gvf_actor_start_step`, instead of the default hard switch (0). Mainly for
    `--gvf_mode lookahead`, where the critic is fitting Q_task - Phi and needs time to
    re-learn that difference whenever Phi changes; a ramp bounds the transient bias."""
    gvf_actor_start_step: int = 0
    """Updates after the critic warm-up before the GVF terms switch on — the reward
    bonus under `--gvf_mode critic`, the actor steering under `actor` (see
    TD3Config.gvf_actor_start_step). 0 => derive as critic_warmup_updates, i.e. the end
    of the critic warm-up plus one more warm-up's worth of margin. The heads' TD losses
    train from the first update either way."""
    gvf_freeze_after_successes: int = 0
    """Train the GVF heads normally until this many RL-mode episode successes have been
    collected, then FREEZE them for the rest of the run (see TD3Config.gvf_freeze).

    0 (default) disables the switch. Warm-up episodes do not count: the base VLA is
    driving there and the heads have barely trained, so counting them would freeze on
    the reference policy's successes rather than the learner's.

    The point is a real-robot-shaped question: the event labels the heads fit come from
    the simulator's predicate detectors, which do not exist on hardware. This gives the
    heads a fixed budget of grounded supervision and then asks the run to carry on with
    the task reward plus a FIXED subgoal potential."""

    gvf_freeze: bool = False
    """Freeze the GVF heads (see TD3Config.gvf_freeze): their TD term is dropped, so
    they keep whatever a resumed checkpoint holds and the shaping bonus becomes a FIXED
    function of (state, action). The task reward is untouched. Only meaningful with
    --resume_from; from scratch it freezes randomly initialized heads."""
    phase_channels: str = "grasp"
    """Comma-separated ORDERED channel names whose first label each advance the episode
    one phase (see labeling.PHASE_CHANNELS). N names give phases 0..N, and a channel's
    `phases` field selects which of those it steers in.

    Default "grasp" is the two-phase book task. A three-stage task lists two:
        --phase_channels grasp,place
        --gvf_channels 'grasp:g:0.95:0.3:+1:0;place:l:0.95:0.3:+1:1;push:k:0.95:0.3:+1:2'
    Every name here must also appear in --gvf_channels, or the phase can never advance."""
    label_lookback_steps: int = 2
    """Quick-mode latency correction: a keypress at step t is recorded at
    max(0, t - label_lookback_steps). Pause-mode entries take no offset (the operator
    had time to look). The offset is echoed on every console label line so it can be
    calibrated against video."""
    auto_label: bool = False
    """Derive the events from sim state instead of a human (LIBERO only). Produces the
    identical structure to the keyboard path, so the whole pipeline runs with no human
    in the loop. WHICH source it uses is `--auto_label_source` below."""
    auto_label_source: str = "predicate"
    """Where `--auto_label` gets its events.

        predicate (default)  the task's own BDDL goal predicates, evaluated against
                             MuJoCo state inside the env subprocess and shipped out as
                             obs["subgoal"] (rlt/envs.build_subgoal_evaluator).
                             Exact contacts/containment/joint angles — no thresholds,
                             no per-suite tuning, and it picks up subgoals the goal
                             conjunct never mentions (the drawer-open step of
                             open_the_top_drawer_and_put_the_bowl_inside).
        proxy                the older proprio heuristics (labeling.auto_label_libero):
                             grasp = "gripper closed AND eef rose 2cm", coll =
                             "commanded but not moving". Kept because it is the only
                             option when the env exposes no parsed problem, and to
                             reproduce runs collected before the predicate path existed.

    Measured difference on libero_goal task 3: the proxy grasp put only 3-6% of buffer
    rows past the phase boundary, so a phase-1-gated channel was off almost everywhere.
    Falls back to `proxy` with a warning if no predicates were discovered."""
    relabel_only: bool = False
    """Rebuild the transition table from existing rollout dumps + label sidecars and
    exit — no env, no VLA, no training. This is what lets a mistimed label be fixed
    offline from video and the run continued without re-collecting hardware episodes.
    Requires the run to have been collected with --dump_rollouts."""

    real_robot_ports: list[int] = dataclasses.field(default_factory=list)
    """If non-empty, use RealRobotEnvWorker instead of the LIBERO EnvWorker — one
    port per physical robot. Also implies total_num_envs == len(real_robot_ports)."""
    task_prompts: str = ""
    """Comma-separated 'id:instruction' pairs, e.g. '0:pick up the red block'."""

    def __post_init__(self):
        if self.action_chunk % self.subsample_stride != 0:
            raise ValueError("subsample_stride must divide action_chunk (next-states must land on the grid)")
        if self.rl_start_step % self.action_chunk != 0 or not 0 <= self.rl_start_step < self.max_episode_steps:
            raise ValueError("rl_start_step must be a multiple of action_chunk in [0, max_episode_steps)")
        if not self.checkpoint_dir:
            raise ValueError("--checkpoint_dir (stage-1 RLT checkpoint) is required")
        if not self.output_dir:
            run = self.name or f"task{self.task_id}_b{self.bc_coef}"
            self.output_dir = f"checkpoints/rlt_{self.config_name}/{self.suite}/{run}"

    @property
    def task_ids(self) -> list[int]:
        return [self.task_id]

    @property
    def parsed_phase_channels(self) -> tuple[str, ...]:
        return tuple(n.strip() for n in self.phase_channels.split(",") if n.strip())

    @property
    def parsed_discrete_action_dims(self) -> dict[int, tuple[float, float]]:
        """`discrete_action_dims` parsed to {column: (raw_full_scale, explore_eps)}."""
        return parse_discrete_action_dims(self.discrete_action_dims, self.action_env_dim)

    @property
    def parsed_gvf_channels(self) -> tuple[_lab.GvfChannel, ...]:
        """The parsed form of the `gvf_channels` CLI string.

        `auto` resolves against THIS run's LIBERO task, by parsing its .bddl — no env,
        no GL, so it is available here, long before the env pool exists. That ordering
        matters: the channel set fixes the critic's GVF head count and TD3 is built
        before the pool.

        Deliberately NOT named `channels`: main() also binds `channels` to the list of
        _BridgeChannel websocket handles on the real-robot path, and the two silently
        swapped once already."""
        if (self.gvf_channels or "").strip() == "auto":
            return _lab.subgoal_channels(self.auto_subgoal_names, gamma=self.gvf_auto_gamma, lam=self.gvf_auto_lam)
        return parse_gvf_channels(self.gvf_channels)

    @property
    def auto_subgoal_names(self) -> tuple[str, ...]:
        """BDDL subgoal channel names for this run's task (empty on the real-robot path)."""
        if self.real_robot_ports:
            return ()
        return W.libero_subgoal_names(self.suite, self.task_id)

    @property
    def run_id(self) -> str:
        """Namespace for the label sidecars (labels/<run_id>/<ep_id>.json)."""
        return self.name or Path(self.output_dir).name


def parse_task_prompts(spec: str) -> dict[int, str]:
    """`"0:foo,1:bar"` -> `{0: "foo", 1: "bar"}`.

    Splits ONLY on commas that begin a new `<id>:` entry, so a prompt may itself contain
    commas. A plain `split(",")` cannot: the hammer instruction contains one ("after
    dropping the hammer, you should close your gripper..."), which split the entry in
    half and raised `dictionary update sequence element #1 has length 1` on a fragment
    with no colon.
    """
    import re

    entries = re.split(r",(?=\s*\d+\s*:)", spec.strip())
    out: dict[int, str] = {}
    for entry in entries:
        if not entry.strip():
            continue
        if ":" not in entry:
            raise ValueError(f"task_prompts entry {entry!r} has no '<id>:' prefix (full spec: {spec!r})")
        tid, prompt = entry.split(":", 1)
        out[int(tid.strip())] = prompt.strip()
    if not out:
        raise ValueError(f"task_prompts parsed to nothing from {spec!r}")
    return out


def parse_discrete_action_dims(spec: str, action_env_dim: int) -> dict[int, tuple[float, float]]:
    """`dim:raw_full_scale[:eps]` entries -> {dim: (raw_full_scale, eps)}.

    "5:0.1:0.05,6:1.0" -> {5: (0.1, 0.05), 6: (1.0, 0.0)}. Empty string -> {}.
    """
    out: dict[int, tuple[float, float]] = {}
    for entry in (e.strip() for e in spec.split(",") if e.strip()):
        parts = entry.split(":")
        if len(parts) > 3:
            raise ValueError(f"--discrete_action_dims entry {entry!r} has too many fields")
        try:
            d, scale = int(parts[0]), float(parts[1])
            eps = float(parts[2]) if len(parts) > 2 else 0.0
        except (ValueError, IndexError) as e:
            raise ValueError(f"--discrete_action_dims entry {entry!r} is not `dim:raw_full_scale[:eps]`") from e
        if not 0 <= d < action_env_dim:
            raise ValueError(f"--discrete_action_dims dim {d} out of range for action_env_dim={action_env_dim}")
        if scale <= 0:
            raise ValueError(f"--discrete_action_dims dim {d}: raw_full_scale must be > 0, got {scale}")
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"--discrete_action_dims dim {d}: eps must be in [0, 1], got {eps}")
        out[d] = (scale, eps)
    return out


def discrete_levels_rl(action_space, action_norm, discrete_dims: dict, action_env_dim: int) -> dict[int, float]:
    """{dim: |RL coordinate of that column's full-scale raw command|}.

    Under the norm-stats calibration this is exactly 1/margin (0.8) for every declared
    column, but it is computed rather than assumed so a hand-edited or legacy
    ActionSpace still explores at that column's real command level.
    """
    kind, p1, p2 = action_norm

    def _norm(raw: float, d: int) -> float:
        if kind == "quantile":
            return (raw - p1[d]) / (p2[d] - p1[d] + 1e-6) * 2.0 - 1.0
        return (raw - p1[d]) / (p2[d] + 1e-6)

    return {
        d: abs(_norm(scale, d) - _norm(-scale, d)) / 2.0 / float(action_space.halfwidth[d])
        for d, (scale, _eps) in discrete_dims.items()
    }


def parse_gvf_channels(spec: str) -> tuple[_lab.GvfChannel, ...]:
    """`name:key:gamma:lam:sign:phases` entries separated by ';'; "default" for
    labeling.DEFAULT_CHANNELS. Trailing fields may be omitted (GvfChannel defaults
    apply); `phases` is a comma-separated list and may be empty (= not phase-gated)."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    if spec == "default":
        return _lab.DEFAULT_CHANNELS
    out = []
    for raw in spec.split(";"):
        entry = raw.strip()
        if not entry:
            continue
        f = entry.split(":")
        if len(f) < 2:
            raise ValueError(f"gvf channel {entry!r}: need at least name:key")
        kw = {"name": f[0].strip(), "key": f[1].strip()}
        if len(f) > 2 and f[2].strip():
            kw["gamma"] = float(f[2])
        if len(f) > 3 and f[3].strip():
            kw["lam"] = float(f[3])
        if len(f) > 4 and f[4].strip():
            kw["sign"] = float(f[4])
        if len(f) > 5 and f[5].strip():
            kw["phases"] = tuple(int(x) for x in f[5].split(",") if x.strip())
        out.append(_lab.GvfChannel(**kw))
    return tuple(out)


def td3_config(cfg: Config) -> _td3.TD3Config:
    return _td3.TD3Config(
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_delay=cfg.policy_delay,
        utd=cfg.utd,
        batch_size=cfg.batch_size,
        sigma_explore=cfg.sigma_explore,
        target_noise=cfg.target_noise,
        target_noise_clip=cfg.target_noise_clip,
        bc_coef=cfg.bc_coef,
        ref_dropout=cfg.ref_dropout,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        action_clip=cfg.action_clip,
        hidden=tuple(cfg.hidden),
        n_critics=cfg.n_critics,
        target_subset_size=cfg.target_subset_size,
        critic_input_norm=cfg.critic_input_norm,
        gvf_channels=cfg.parsed_gvf_channels,
        actor_q_eval=cfg.actor_q_eval,
        gvf_mode=cfg.gvf_mode,
        gvf_freeze=cfg.gvf_freeze,
        # 0 => one further critic-warmup's worth of updates past the warm-up's end,
        # i.e. 2x critic_warmup_updates in absolute terms. A freshly initialized GVF
        # head emits noise gradients of ORDINARY magnitude, not small ones, so an
        # ungated lam feeds that noise into the actor at full strength -- or, under
        # gvf_mode="critic", straight into Q_task's regression target.
        gvf_actor_start_step=(cfg.gvf_actor_start_step or cfg.critic_warmup_updates),
        gvf_ramp_steps=cfg.gvf_ramp_steps,
    )


# ════════════════════════════════════════════════════════════════════════════
# Norm stats (resolved from the checkpoint assets dir)
# ════════════════════════════════════════════════════════════════════════════


def _resolve_norm_stats(checkpoint_dir: str, data_config):
    import openpi.training.checkpoints as _checkpoints

    ckpt = Path(checkpoint_dir)
    candidates = [ckpt / "assets"]
    if ckpt.name.isdigit() or ckpt.name.startswith("step_"):
        candidates.append(ckpt.parent / "assets")
    elif ckpt.is_dir():
        # run dir: scripts/train.py saves assets per step (<run>/<step>/assets)
        steps = sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit())
        if steps:
            candidates.append(ckpt / str(steps[-1]) / "assets")
    candidates.append(ckpt.parent / "assets")
    if data_config.asset_id is not None:
        for assets_dir in candidates:
            try:
                if (assets_dir / data_config.asset_id).exists():
                    ns = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
                    if ns is not None:
                        logger.info(f"norm stats from {assets_dir / data_config.asset_id}")
                        return ns
            except Exception as e:
                logger.warning(f"norm-stats load from {assets_dir} failed: {e}")
    if data_config.norm_stats is not None:
        logger.info("norm stats from config data_config (fallback)")
        return data_config.norm_stats
    return None


def build_norm(norm_stats, use_quantile, key):
    if norm_stats is None or key not in norm_stats:
        logger.warning(f"No {key} norm stats — skipping normalization")
        return None
    s = norm_stats[key]
    if use_quantile:
        return ("quantile", np.array(s.q01, np.float32), np.array(s.q99, np.float32))
    return ("zscore", np.array(s.mean, np.float32), np.array(s.std, np.float32))


# ════════════════════════════════════════════════════════════════════════════
# Obs helpers
# ════════════════════════════════════════════════════════════════════════════

_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_vel",
    "robot0_gripper_qvel",
    # BDDL subgoal predicates, evaluated in the env subprocess (the only side with
    # sim.data/contacts) and attached to obs there. Absent on the real-robot path and
    # when the env exposes no parsed problem, so `_trim_obs`'s `if k in o` drop is the
    # normal case, not an error.
    "subgoal",
)
_warned_missing_vel = [False]  # one-time warning latch


def _trim_obs(o: dict) -> dict:
    return {k: np.asarray(o[k]) for k in _OBS_KEYS if k in o}


def _rl_proprio(o: dict, action_dim: int, state_norm, *, include_velocity: bool) -> np.ndarray:
    """Paper proprio: the normalized 8-dim pose/gripper state (+ raw velocities)."""
    base = W._make_state(o, action_dim, state_norm)[:RAW_PROPRIO_DIM]
    if not include_velocity:
        return base
    if "robot0_joint_vel" not in o and not _warned_missing_vel[0]:
        logger.warning("robot0_joint_vel missing from obs — velocity proprio is zeros")
        _warned_missing_vel[0] = True
    jv = np.asarray(o.get("robot0_joint_vel", np.zeros(7)), np.float32)
    gv = np.asarray(o.get("robot0_gripper_qvel", np.zeros(2)), np.float32)
    return np.concatenate([base, jv, gv]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Trajectory frames (see Config.save_frames_dir)
# ════════════════════════════════════════════════════════════════════════════


class _FrameWriter:
    """Per-episode camera frames as PNGs, in eval_pi05_real.py's layout.

    An episode's directory is named by index while it is in flight (the outcome is
    not known until it ends) and stamped `_success` / `_failure` by finish(), so
    successes and failures sort apart when picking figure frames later.
    """

    _CAMS = (("base", "agentview_image"), ("wrist", "robot0_eye_in_hand_image"))

    def __init__(self, root: Path, every: int = 1):
        self.root = Path(root)
        self.every = max(int(every), 1)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving trajectory frames to {self.root} (every {self.every} step(s))")

    def save_step(self, ep_name: str, step: int, obs: dict) -> None:
        if step % self.every:
            return
        import imageio.v3 as iio
        from openpi_client import image_tools

        d = self.root / ep_name
        d.mkdir(parents=True, exist_ok=True)
        for cam, key in self._CAMS:
            img = obs.get(key)
            if img is None:
                continue
            # Rotate 180°, matching obs_multi_to_pi0 (and eval_libero_sim's videos):
            # these are the pixels the policy is handed, and the right way up.
            arr = image_tools.convert_to_uint8(np.ascontiguousarray(np.asarray(img)[::-1, ::-1]))
            iio.imwrite(d / f"step_{step:05d}_{cam}.png", arr)

    def save_episode(self, ep_name: str, obs_seq: list, *, success: bool) -> None:
        """Write a whole finished episode. Cheaper than save_step per control step for
        `collect`, whose records already hold every obs — it keeps PNG encoding out of
        the rollout loop, where it would sit between the arm's steps."""
        for t, obs in enumerate(obs_seq):
            self.save_step(ep_name, t, obs)
        self.finish(ep_name, success=success)

    def finish(self, ep_name: str, *, success: bool) -> None:
        d = self.root / ep_name
        if not d.exists():
            return
        target = d.with_name(f"{ep_name}_{'success' if success else 'failure'}")
        if target.exists():
            # A resumed run can reuse an episode index; keep the earlier capture.
            logger.warning(f"frame dir {target} already exists — leaving {d} unstamped")
            return
        d.rename(target)


# ════════════════════════════════════════════════════════════════════════════
# Rollout collection
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class _EpisodeRecord:
    obs_seq: list  # trimmed obs at step indices 0..L (L+1 entries)
    act_seq: list  # executed normalized [action_env_dim] actions, L entries
    rew_seq: list  # per-step 0/1 rewards, L entries
    success: bool
    tok_prompt: np.ndarray
    tok_mask: np.ndarray
    btab: dict  # boundary table t -> (zrl, ref_flat, proprio) computed during rollout
    ep_id: int = -1
    """Run-global episode index. Stable across iterations, so it names this episode's
    label sidecar (labels/<run_id>/<ep_id>.json) for offline re-labeling."""
    events: dict = dataclasses.field(default_factory=dict)
    """Subgoal channel labels: {channel name: [(step, +1/-1), ...]}. +1 = the event
    occurred here, -1 = it can no longer occur; BOTH terminate the channel and differ
    only in the terminal value (see rlt/labeling.py). Empty for every channel when the
    mechanism is off."""
    init_state: object = None
    """The trial init state this episode was reset to (`_EpState.init_state`): a
    MuJoCo state vector in sim, the bridge's {task_id, trial} dict on the real robot,
    None for a plain reset. Nothing in training reads it.

    It is recorded so an exported demonstration can be REPLAYED: LIBERO is
    deterministic given (init state, action sequence), but without the init state a
    replay starts from a different scene and open-loop playback of the stored actions
    reproduces nothing."""


def _finalize_labels(
    cfg: Config,
    records: list[_EpisodeRecord],
    channels: tuple[_lab.GvfChannel, ...],
    output_dir: Path | None,
    subgoal_names: tuple[str, ...] = (),
) -> None:
    """Resolve each episode's events, then persist them.

    Precedence is: an EXISTING SIDECAR wins over anything collected in memory. That is
    the whole point of the sidecar — it lets an operator fix a mistimed label offline
    from video and re-run training without re-collecting hardware episodes. Auto-labels
    fill in when no sidecar and no human input exist.

    Under `--auto_label_source predicate` (the default) the labels come from the BDDL
    goal predicates the env shipped in `obs["subgoal"]` — exact MuJoCo state, no
    thresholds. `proxy` restores the proprio heuristics of `auto_label_libero`, which
    are the only option when the env exposed no predicates at all.
    """
    if not channels:
        return
    use_predicates = cfg.auto_label_source == "predicate" and bool(subgoal_names)
    if cfg.auto_label and cfg.auto_label_source == "predicate" and not subgoal_names:
        logger.warning(
            "--auto_label_source predicate but the env pool exposed no BDDL predicates "
            "(real-robot path, or a task whose problem could not be parsed) — falling back "
            "to the proprio proxies of --auto_label_source proxy."
        )
    for rec in records:
        if cfg.auto_label and not any(rec.events.get(ch.name) for ch in channels):
            rec.events = (
                _lab.auto_label_libero_predicates(rec, channels, subgoal_names)
                if use_predicates
                else _lab.auto_label_libero(rec, channels)
            )
        side = _lab.sidecar_path(output_dir, cfg.run_id, rec.ep_id) if output_dir is not None else None
        existing = _lab.load_episode_labels(side) if side is not None else None
        if existing is not None:
            rec.events = existing["events"]
            logger.info(f"episode {rec.ep_id}: labels overridden by sidecar {side}")
        elif side is not None:
            _lab.save_episode_labels(
                side,
                ep_id=rec.ep_id,
                ep_len=len(rec.act_seq),
                label_lookback_steps=cfg.label_lookback_steps,
                events={ch.name: rec.events.get(ch.name, []) for ch in channels},
            )
        rec.events = {ch.name: sorted(rec.events.get(ch.name, [])) for ch in channels}
        # A successful episode necessarily passed the phase boundary. Warn only — the
        # timestep of a missed label is unknown, and a fabricated timestamp is worse
        # than a gap. Checked against THIS run's --phase_channels, not a hardcoded
        # "grasp": a push task has no grasp in any solution.
        warn = _lab.check_success_consistency(
            rec.events,
            success=rec.success,
            ep_id=rec.ep_id,
            phase_channels=cfg.parsed_phase_channels,
            source=cfg.auto_label_source if cfg.auto_label else "human",
        )
        if warn:
            logger.warning(warn)


def collect(
    cfg: Config,
    env_worker: W.EnvWorker,
    vla_forward,
    actor_apply,
    td3_state,
    pi0_state,
    rng,
    np_rng: np.random.Generator,
    zrl_normalizer: RunningNorm,
    *,
    policy: str,
    action_space: ActionSpace | None = None,
    desc: str = "rollout",
    labeler: _lab.KeyboardLabeler | None = None,
    ep_id_base: int = 0,
    frame_writer: _FrameWriter | None = None,
    iteration: int = 0,
) -> tuple[list[_EpisodeRecord], dict]:
    """policy selects the executed-action source (RLT Alg. 1 line 9):
    "ref"       — the raw VLA reference chunk (warmup buffer prefill);
    "ref_noise" — reference + exploration noise (critic/BC warm-start phase: the
                  plain-MLP actor hasn't learned to imitate the reference yet);
    "actor"     — actor mean + exploration noise (the RL policy).
    "ref_noise"/"actor" require action_space (noise/clip/actor run in RL space).
    Records store everything in VLA-normalized space; to_rl happens at assembly.
    """
    if policy not in ("ref", "ref_noise", "actor"):
        raise ValueError(f"unknown rollout policy {policy!r}")
    if policy != "ref" and action_space is None:
        raise ValueError(f"policy {policy!r} requires an ActionSpace")
    pool = env_worker.make_pool(train=True, n_active=(cfg.rollout_num_envs or None))
    B = len(pool)
    C = cfg.action_chunk
    ad = cfg.action_env_dim
    # Per-dim exploration sigma over the flattened chunk [C * ad]. The gripper is the
    # LAST of the action_env_dim columns, repeated at every chunk position, and gets its
    # own sigma (default 0) because noise on a sign-encoded discrete trigger flips the
    # physical command rather than exploring around it. See Config.sigma_explore_gripper.
    sigma_vec = np.full((C * ad,), cfg.sigma_explore, np.float32)
    sigma_vec[ad - 1 :: ad] = cfg.sigma_explore_gripper
    # Declared discrete columns override the gripper default: noise on a two-level
    # channel randomises which level is sent rather than exploring around it, and that
    # is true of every such column, not only the last one.
    discrete = cfg.parsed_discrete_action_dims
    for d in discrete:
        sigma_vec[d::ad] = cfg.sigma_explore_discrete
    # Held-pulse exploration for those columns (see Config.discrete_action_dims). Levels
    # come from the ActionSpace so a column always explores at ITS command magnitude;
    # policy == "ref" (the raw-reference warmup) never explores.
    levels = (
        discrete_levels_rl(action_space, env_worker.action_norm, discrete, ad) if discrete and policy != "ref" else {}
    )
    explore_dims = [d for d in levels if discrete[d][1] > 0.0]
    # Per-env pulse state, carried ACROSS chunk boundaries: a pulse starting at position
    # 8 must not be truncated by the chunk edge, or the sampled hold distribution is
    # silently clipped at the chunk length.
    pulse_left = np.zeros((B, len(explore_dims)), np.int32)
    pulse_val = np.zeros((B, len(explore_dims)), np.float32)
    p_end = 1.0 / max(cfg.explore_discrete_hold, 1.0)  # geometric hold, mean = the knob
    max_chunk_steps = cfg.max_episode_steps // C

    records = [
        _EpisodeRecord(
            obs_seq=[_trim_obs(s.obs)],
            act_seq=[],
            rew_seq=[],
            success=False,
            tok_prompt=s.tok_prompt,
            tok_mask=s.tok_mask,
            btab={},
            ep_id=ep_id_base + j,
            init_state=s.init_state,
        )
        for j, s in enumerate(pool)
    ]

    # The human labels ONE arm: keypresses have no env index, so they are attributed to
    # a single focused episode. With more than one env in the pool the rest go
    # unlabeled, which silently halves (or worse) the GVF training signal.
    focus = 0
    label_history: list[str] = []  # channel names, newest last — the undo stack
    if labeler is not None and labeler.enabled and B > 1:
        logger.warning(
            f"keyboard labeling with {B} envs in the pool: keypresses are attributed to env "
            f"{focus} only (one operator cannot watch {B} arms). The other {B - 1} episodes "
            f"will carry no labels — run one env per operator, or use --auto_label in sim."
        )

    def _apply_labels() -> None:
        """Drain the listener into the focused record. Steps were resolved AT PRESS
        TIME (quick mode applies -label_lookback_steps, pause mode is exact)."""
        if labeler is None:
            return
        rec = records[focus]
        labeler.note_step(pool[focus].steps)
        for name, value, step in labeler.drain():
            # `step` was resolved at press time and is <= the current step by
            # construction (the lookback only moves it earlier), so it is recorded
            # verbatim — silently rewriting a label's timestamp would defeat the
            # calibration the console echo exists for.
            rec.events.setdefault(name, []).append((int(step), int(value)))
            label_history.append(name)
        if labeler.pop_undo() and label_history:
            name = label_history.pop()
            if rec.events.get(name):
                dropped = rec.events[name].pop()
                logger.info(f"undo: dropped {name} {dropped[1]:+d} @ step {dropped[0]} (ep {rec.ep_id})")

    def _wait_if_paused() -> None:
        """Block BEFORE the next action is dispatched.

        A pause must leave no trace in the episode: no send_step, no recv_step, no
        transition appended, and `s.steps` does not advance. If the env kept stepping
        while the operator deliberated, a run of near-identical held-position
        transitions would be injected at precisely the bottleneck states, biasing the
        GVF cumulants toward "nothing happens here" exactly where something does.
        """
        if labeler is None or not labeler.paused:
            return
        while labeler.paused:
            _apply_labels()
            time.sleep(0.05)
        _apply_labels()

    pbar = tqdm.tqdm(range(max_chunk_steps), desc=desc, unit="chunk", leave=False, dynamic_ncols=True)
    for _t in pbar:
        active = [j for j in range(B) if not pool[j].done]
        pbar.set_postfix(active=f"{len(active)}/{B}", success=sum(s.success for s in pool))
        if not active:
            break

        # One VLA forward for the whole pool (fixed shape): z_rl + reference chunk.
        batched_obs = W.obs_multi_to_pi0(
            [s.obs for s in pool],
            [s.tok_prompt for s in pool],
            [s.tok_mask for s in pool],
            env_worker.action_dim,
            env_worker.state_norm,
        )
        rng, srng = jax.random.split(rng)
        zrl, a_full = vla_forward(pi0_state, srng, batched_obs)
        zrl = zrl_normalizer(np.asarray(zrl, dtype=np.float32))  # [B, zrl_dim], standardized
        a_full = np.asarray(a_full, dtype=np.float32)  # [B, H, action_dim]
        proprio = np.stack(
            [
                _rl_proprio(s.obs, env_worker.action_dim, env_worker.state_norm, include_velocity=cfg.include_velocity)
                for s in pool
            ]
        )
        ref_flat = a_full[:, :C, :ad].reshape(B, C * ad)

        if policy == "ref":
            exec_flat = ref_flat
        else:
            ref_rl = action_space.to_rl(ref_flat)
            if policy == "actor":
                x = np.concatenate([zrl, proprio], axis=-1)
                mean_rl = np.asarray(actor_apply(td3_state, jnp.asarray(x), jnp.asarray(ref_rl)))
            else:  # "ref_noise"
                mean_rl = ref_rl
            noise = (np_rng.standard_normal(mean_rl.shape) * sigma_vec).astype(np.float32)
            exec_rl = mean_rl + noise
            # ── held-pulse exploration on declared discrete columns ─────────
            # Gaussian noise cannot explore a two/three-level channel: reaching a
            # command level from rest needs sigma of order the level itself, which
            # would wreck every continuous column sharing the scale. Instead, with
            # probability eps per env-step, OVERRIDE the column with a full-scale
            # command of random sign and hold it for a geometric number of steps.
            # Outside a pulse the actor's own value passes through untouched, so this
            # buys exploration without biasing the policy being learned.
            for k, d in enumerate(explore_dims):
                eps_d, lvl = discrete[d][1], levels[d]
                for c in range(C):
                    start = (pulse_left[:, k] <= 0) & (np_rng.random(B) < eps_d)
                    n_start = int(start.sum())
                    if n_start:
                        pulse_left[start, k] = np_rng.geometric(p_end, n_start)
                        pulse_val[start, k] = lvl * np_rng.choice([-1.0, 1.0], n_start)
                    on = pulse_left[:, k] > 0
                    exec_rl[on, c * ad + d] = pulse_val[on, k]
                    pulse_left[on, k] -= 1
            exec_rl = np.clip(exec_rl, -cfg.action_clip, cfg.action_clip)
            exec_flat = action_space.from_rl(exec_rl)
            # Critical-phase handover: before rl_start_step the raw reference runs
            # (no noise, no actor) — the RL policy owns only the final segment.
            for j in active:
                if pool[j].steps < cfg.rl_start_step:
                    exec_flat[j] = ref_flat[j]
        exec_chunk = exec_flat.reshape(B, C, ad)

        for j in active:
            records[j].btab[pool[j].steps] = (zrl[j], ref_flat[j], proprio[j])

        # Unnormalize in full model action width (norm stats may cover > env dims):
        # the executed [:C, :ad] block is the actor's, the tail stays the VLA sample.
        exec_model = a_full[:, :C, :].copy()
        exec_model[:, :, :ad] = exec_chunk
        action_raws = [W.unnormalize_action(env_worker.action_norm, exec_model[j]) for j in range(B)]

        steps_in_chunk = [min(C, cfg.max_episode_steps - s.steps) for s in pool]
        max_sub = max(steps_in_chunk)
        for sub in range(max_sub):
            running = [j for j in range(B) if not pool[j].done and sub < steps_in_chunk[j]]
            # Both BEFORE the send fan-out: the pause halts the episode with nothing
            # in flight, and the drain timestamps labels against the step the operator
            # was actually watching.
            _apply_labels()
            _wait_if_paused()
            for j in running:
                pool[j].env.send_step(action_raws[j][sub, :ad])
            for j in running:
                s = pool[j]
                obs, done, info = s.env.recv_step()
                s.obs = obs
                s.steps += 1
                rec = records[j]
                # LIBERO's done fires exactly on task success and never populates
                # info["reward"]. The real-robot bridge overloads done for ANY
                # episode end (success keypress, manual-failure keypress, or an
                # auto-reported MotionAborted) and carries the real outcome in
                # reward instead -- gate success on it when present, or a failed
                # episode gets recorded (and trained on!) as reward=1.0/success.
                reward = info.get("reward") if info else None
                is_success = done and (reward is None or reward > 0)
                rec.act_seq.append(exec_chunk[j, sub])
                rec.rew_seq.append(1.0 if is_success else 0.0)
                rec.obs_seq.append(_trim_obs(obs))
                if is_success:
                    s.success = True
                    s.done = True
                    rec.success = True
                elif done or s.steps >= cfg.max_episode_steps:
                    s.done = True

    _apply_labels()

    # Frames are written here, after the pool has finished, rather than inside the
    # step loop: the records already hold every obs, and on the real robot a PNG
    # encode between send_step and recv_step would stretch the control period.
    if frame_writer is not None:
        for rec in records:
            frame_writer.save_episode(f"it{iteration:04d}_ep{rec.ep_id:04d}", rec.obs_seq, success=rec.success)

    # z_rl dispersion across the rollout's states: if the tokenizer's encoding is
    # (near-)constant over visited states, the critic cannot discriminate states and
    # values converge to a buffer-average — the "uninformative token" failure mode.
    # NOTE zrl here is POST zrl_normalizer (zero-mean/unit-std per dim once frozen),
    # so zrl_norm/zrl_rms no longer reflect the raw tokenizer offset (~46.5) and
    # zrl_disc's old "~0.8 is healthy" calibration doesn't carry over — it now
    # measures state-dispersion in standardized units, still ~0 for a collapsed
    # token vs ~1 for a fully state-dependent one.
    zrls = np.stack([entry[0] for rec in records for entry in rec.btab.values()])
    zrl_std = float(np.mean(np.std(zrls, axis=0)))
    zrl_rms = float(np.sqrt(np.mean(np.square(zrls))))  # per-dim RMS magnitude
    stats = {
        "success_rate": float(np.mean([s.success for s in pool])),
        "mean_ep_len": float(np.mean([len(r.act_seq) for r in records])),
        "n_episodes": B,
        "zrl_std": zrl_std,
        "zrl_norm": float(np.mean(np.linalg.norm(zrls, axis=-1))),
        # The uninformative-token check as one dimension/scale-independent number:
        # across-state dispersion over typical magnitude. ~0 = near-constant token
        # (critic regresses a buffer average, stage-1 failure); ~1 = fully
        # state-dependent.
        "zrl_disc": zrl_std / max(zrl_rms, 1e-8),
    }
    return records, stats


# ════════════════════════════════════════════════════════════════════════════
# Transition assembly (stride-grid subsampling, post-episode batched VLA pass)
# ════════════════════════════════════════════════════════════════════════════


def _fill_tables(
    cfg: Config,
    env_worker: W.EnvWorker,
    vla_forward,
    pi0_state,
    rng,
    records: list[_EpisodeRecord],
    zrl_normalizer: RunningNorm,
) -> list[dict]:
    """Per-env tables t -> (zrl, ref_flat, proprio) at every stride-grid index <= L.

    Reuses the boundary entries computed during rollout; the rest come from a
    fixed-size batched VLA pass over the stored per-step obs (one extra jit shape).
    """
    C, ad = cfg.action_chunk, cfg.action_env_dim
    tables: list[dict] = []
    todo: list[tuple[int, int]] = []  # (env_idx, t)
    for j, rec in enumerate(records):
        L = len(rec.act_seq)
        tab = dict(rec.btab)
        tables.append(tab)
        # The stride grid, PLUS L itself when the episode did not happen to end on it.
        # The final state is the bootstrap target of every chunk that reaches the
        # episode end, and _assemble_from_tables silently falls back to `next_t = t`
        # when it is missing -- a SELF-LOOP, next_state == state, with a live discount
        # (up to gamma^1 = 0.99). Q(x,a) <- bonus + 0.99*Q'(x, mu'(x)) then diverges
        # outright for any target actor that outbids the buffer action at all, and
        # those rows feed the whole chain behind them.
        #
        # Only ever bites off the grid, which in SIMULATION means only successes:
        # LIBERO failures truncate at exactly max_episode_steps (and stride | 480), and
        # a success has discount = 0, so the missing entry is multiplied away. On the
        # REAL ROBOT an episode ends when the operator presses a key, at an arbitrary
        # step -- 11 of the first 16 book_placement episodes ended on an odd step, each
        # contributing 5 self-looping rows. Costs at most one extra VLA forward per
        # episode; do not "optimize" it back to the bare range.
        start = cfg.rl_start_step
        grid = list(range(start, L + 1, cfg.subsample_stride))
        if L >= start and (not grid or grid[-1] != L):
            grid.append(L)
        todo.extend((j, t) for t in grid if t not in tab and t < len(rec.obs_seq))

    fb = cfg.forward_batch
    for start in tqdm.tqdm(range(0, len(todo), fb), desc="subsample fwd", unit="batch", leave=False):
        batch = todo[start : start + fb]
        pad = fb - len(batch)
        batch_padded = batch + [batch[0]] * pad  # fixed shape; padded rows discarded
        obs_list = [records[j].obs_seq[t] for j, t in batch_padded]
        batched_obs = W.obs_multi_to_pi0(
            obs_list,
            [records[j].tok_prompt for j, _ in batch_padded],
            [records[j].tok_mask for j, _ in batch_padded],
            env_worker.action_dim,
            env_worker.state_norm,
        )
        rng, srng = jax.random.split(rng)
        zrl, a_full = vla_forward(pi0_state, srng, batched_obs)
        zrl = zrl_normalizer(np.asarray(zrl, dtype=np.float32))
        ref_flat = np.asarray(a_full, dtype=np.float32)[:, :C, :ad].reshape(fb, C * ad)
        for i, (j, t) in enumerate(batch):
            prop = _rl_proprio(
                records[j].obs_seq[t],
                env_worker.action_dim,
                env_worker.state_norm,
                include_velocity=cfg.include_velocity,
            )
            tables[j][t] = (zrl[i], ref_flat[i], prop)
    return tables


def _dump_rollout_npz(
    cfg: Config,
    records: list[_EpisodeRecord],
    tables,
    dump_dir: Path,
    iteration: int,
    action_space: ActionSpace | None = None,
) -> None:
    """Persist this iteration's rollout traces for offline analysis.

    z_rl is stored POST-normalizer: exactly the vector the critic/actor consume, so
    any state-dependent quantity can be re-derived offline without re-running rollouts.

    `proprio` and `ref` complete the critic's input at every grid point: the critic
    consumes x = concat(zrl, proprio) and an action in RL space, so z_rl alone is not
    enough to re-evaluate Q (or its action-gradient dQ/da) offline. `ref` is stored
    POST action_space.to_rl — the same convention the replay buffer uses — so
    an offline analysis can feed it to the critic verbatim. If
    action_space is None (first warmup iteration, before the map is measured) the RL
    map is not yet defined and `ref` is written as an empty array.

    Episodes have different lengths, so the ragged per-step arrays are concatenated
    with an explicit `ep_index`/`grid_t` rather than stored as object arrays — the
    file then loads without allow_pickle. Per-episode scalars are indexed by j:
        zrl[ep_index == j], proprio[...], ref[...], grid_t[...], ep_len[j], ep_success[j]
    """
    zrl_rows: list[np.ndarray] = []
    prop_rows: list[np.ndarray] = []
    ref_rows: list[np.ndarray] = []
    grid_t: list[int] = []
    ep_index: list[int] = []
    ep_len: list[int] = []
    ep_success: list[bool] = []
    ep_id: list[int] = []
    act_rows: list[np.ndarray] = []
    rew_rows: list[float] = []
    act_ep_index: list[int] = []
    for j, rec in enumerate(records):
        L = len(rec.act_seq)
        for t in sorted(t for t in tables[j] if t <= L):
            zrl_t, ref_t_vla, prop_t = tables[j][t]
            zrl_rows.append(zrl_t)
            prop_rows.append(prop_t)
            ref_rows.append(ref_t_vla)
            grid_t.append(t)
            ep_index.append(j)
        # Executed actions and per-step rewards complete what --relabel_only needs to
        # rebuild the transition table offline: the tables above give the states, these
        # give the actions/returns, and the sidecars supply the (corrected) events.
        for t in range(L):
            act_rows.append(np.asarray(rec.act_seq[t], np.float32))
            rew_rows.append(float(rec.rew_seq[t]))
            act_ep_index.append(j)
        ep_len.append(L)
        ep_success.append(bool(rec.success))
        ep_id.append(int(rec.ep_id))

    def _stack(rows: list[np.ndarray]) -> np.ndarray:
        return np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), np.float32)

    ref = _stack(ref_rows)
    if action_space is not None and ref.size:
        ref = action_space.to_rl(ref)
    elif action_space is None:
        ref = np.zeros((0, 0), np.float32)

    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"rollout_{iteration:05d}.npz"
    np.savez_compressed(
        path,
        zrl=_stack(zrl_rows),
        proprio=_stack(prop_rows),
        ref=ref,
        grid_t=np.asarray(grid_t, np.int32),
        ep_index=np.asarray(ep_index, np.int32),
        ep_len=np.asarray(ep_len, np.int32),
        ep_success=np.asarray(ep_success, dtype=bool),
        ep_id=np.asarray(ep_id, np.int32),
        act=np.stack(act_rows).astype(np.float32) if act_rows else np.zeros((0, 0), np.float32),
        rew=np.asarray(rew_rows, np.float32),
        act_ep_index=np.asarray(act_ep_index, np.int32),
        iteration=np.int32(iteration),
        subsample_stride=np.int32(cfg.subsample_stride),
    )


def assemble_transitions(
    cfg: Config,
    env_worker: W.EnvWorker,
    vla_forward,
    pi0_state,
    rng,
    records: list[_EpisodeRecord],
    action_space: ActionSpace,
    zrl_normalizer: RunningNorm,
    *,
    channels: tuple[_lab.GvfChannel, ...] = (),
    dump_dir: Path | None = None,
    iteration: int | None = None,
) -> dict[str, np.ndarray]:
    """Chunk transitions for every stride-grid t in [rl_start_step, L).

    reward   = sum_{i<n} gamma^i r_{t+i},  n = min(C, L - t)
    discount = 0 if the episode SUCCEEDED inside (t, t+n], else gamma^n
               (n < C only at the truncation tail -> TRUE bootstrap from the final obs).

    action = executed sub-actions a_{t:t+n-1}, zero-padded to C (padded rows only occur
             on terminal transitions where discount=0 and the reward is already realized).
    Actions/refs are mapped to RL space here — records keep VLA-normalized values.
    """
    tables = _fill_tables(cfg, env_worker, vla_forward, pi0_state, rng, records, zrl_normalizer)
    if dump_dir is not None and iteration is not None:
        _dump_rollout_npz(cfg, records, tables, dump_dir, iteration, action_space)
    return _assemble_from_tables(cfg, records, tables, action_space, channels)


def _assemble_from_tables(
    cfg: Config,
    records: list[_EpisodeRecord],
    tables: list[dict],
    action_space: ActionSpace,
    channels: tuple[_lab.GvfChannel, ...] = (),
) -> dict[str, np.ndarray]:
    """The pure part of assemble_transitions: tables + records -> transition columns.

    Factored out so `--relabel_only` can rebuild the identical table from a rollout
    dump plus corrected sidecars, with no env, no VLA and no rollout."""
    C, ad = cfg.action_chunk, cfg.action_env_dim

    out: dict[str, list] = {
        k: []
        for k in (
            "zrl",
            "next_zrl",
            "proprio",
            "next_proprio",
            "action",
            "ref",
            "next_ref",
            "reward",
            "discount",
            "success",
            "grid_t",
            "ep_len",
            *_lab.batch_field_names(channels),
        )
    }
    gammas = cfg.gamma ** np.arange(C, dtype=np.float32)
    for j, rec in enumerate(records):
        L = len(rec.act_seq)
        tab = tables[j]
        acts = np.asarray(rec.act_seq, dtype=np.float32).reshape(L, ad) if L else np.zeros((0, ad), np.float32)
        rews = np.asarray(rec.rew_seq, dtype=np.float32)
        # Phase 0 until the grasp channel receives ANY label, 1 after (labeling.py).
        phases = _lab.derive_phases(rec.events, L, phase_channels=cfg.parsed_phase_channels) if channels else None
        # Transitions only from the RL segment (t >= rl_start_step; whole episode
        # when 0). Episodes that ended before the handover contribute nothing.
        for t in range(cfg.rl_start_step, L, cfg.subsample_stride):
            n = min(C, L - t)
            if cfg.drop_partial_chunks and n < C:
                continue
            reward = float(np.sum(gammas[:n] * rews[t : t + n]))
            success_inside = rec.success and (t + n == L)
            discount = 0.0 if success_inside else cfg.gamma**n
            zrl_t, ref_t_vla, prop_t = tab[t]
            # Pad unexecuted steps (episode ended inside the chunk) with the REFERENCE
            # tail, not zeros: these terminal rows are the value support of the whole
            # bootstrap chain, and the backup one hop earlier queries Q' at REALISTIC
            # actions — high values learned at zero-padded action coordinates are
            # invisible to it (observed: q_reward 0.95 with q_mean pinned at the
            # reward-rows-only level ~0.05, propagation knobs ineffective).
            action = ref_t_vla.copy()
            action[: n * ad] = acts[t : t + n].reshape(-1)
            # _fill_tables guarantees t + n is present (the grid always includes L), so
            # this fallback should be unreachable. It is kept because what it does when
            # reached is a SELF-LOOP -- next_state == state -- which is harmless at
            # discount 0 and divergent at any live discount, and that failure is
            # invisible in every logged quantity except q_mean walking off the return
            # scale. Warn rather than assert: losing one transition beats killing a
            # hardware episode mid-run.
            next_t = t + n if (t + n) in tab else t
            if next_t == t and discount > 0.0:
                logger.warning(
                    f"ep {rec.ep_id}: no state at t={t + n} (L={L}); chunk t={t} would bootstrap from "
                    f"ITSELF at discount {discount:.3f} -- dropping the row. This is a bug in the "
                    f"state grid, not a data problem; the critic diverges if these accumulate."
                )
                continue
            zrl_n, ref_n, prop_n = tab[next_t]
            out["zrl"].append(zrl_t)
            out["proprio"].append(prop_t)
            out["ref"].append(action_space.to_rl(ref_t_vla))
            out["action"].append(action_space.to_rl(action))
            out["reward"].append(reward)
            out["discount"].append(discount)
            out["next_zrl"].append(zrl_n)
            out["next_proprio"].append(prop_n)
            out["next_ref"].append(action_space.to_rl(ref_n))
            out["success"].append(np.float32(rec.success))
            # Bookkeeping only (no loss reads these): lets eps be plotted against where
            # in the trajectory it was spent. See the ReplayBuffer docstring.
            out["grid_t"].append(np.int32(t))
            out["ep_len"].append(np.int32(L))
            if channels:
                # Subgoal cumulants. Deliberately independent of `reward`/`discount`
                # above: Q_task stays a pure sparse-terminal critic, and the GVFs
                # enter training only through the actor gradient (rlt/td3.py). Each
                # channel discounts with its OWN gamma over its own ~5-10 chunk chain.
                for key, val in _lab.chunk_cumulants(channels, rec.events, t=t, n=n, ep_len=L).items():
                    out[key].append(np.float32(val))
                out["phase"].append(np.int32(phases[t]) if t < len(phases) else np.int32(0))
                # The phase of the state next_zrl/next_proprio describe, i.e. at
                # `next_t` -- NOT at t + C. With subsample_stride < C consecutive
                # transitions overlap and next_t is t + n on the stride grid, so taking
                # it from the same index the next-state columns came from is the only
                # way the two stay in step.
                #
                # CLAMPED to the last step, not defaulted to 0. The state table runs to
                # L (the final observation) while `phases` has one entry per EXECUTED
                # step, so next_t == L on any chunk that reaches the episode end. That
                # row is terminal only when the episode SUCCEEDED there; a truncated
                # episode's last chunk has discount = gamma^n != 0 and its Phi_next is
                # live, so writing phase 0 would tell the backup the episode rewound to
                # the start at its final state. derive_phases is monotone
                # non-decreasing and no event can be labeled at a step that was never
                # executed, so phases[L-1] IS the phase at L.
                out["next_phase"].append(np.int32(phases[min(next_t, len(phases) - 1)]) if len(phases) else np.int32(0))
    return {k: np.stack(v) if v else np.zeros((0,)) for k, v in out.items()}


def relabel_from_dumps(cfg: Config) -> list[Path]:
    """Rebuild the transition table from `<output_dir>/rollouts/*.npz` + the label
    sidecars, with no env, no VLA and no rollout, and write it to
    `<output_dir>/relabel/transitions_<iter>.npz`.

    This is the offline half of the labeling workflow: the operator fixes a mistimed
    label from video (edit `labels/<run_id>/<ep_id>.json`), re-runs this, and gets a
    corrected table without spending hardware episodes on a re-collection.

    Requires the run to have been collected with --dump_rollouts. `ref` in the dump is
    stored POST action_space.to_rl (the replay-buffer convention), so it is mapped back
    through from_rl here — an exact affine inverse up to float32 rounding.
    """
    output_dir = Path(cfg.output_dir)
    channels = cfg.parsed_gvf_channels
    if not channels:
        raise ValueError("--relabel_only needs --gvf_channels (there is nothing to relabel otherwise)")
    dumps = sorted((output_dir / "rollouts").glob("rollout_*.npz"))
    if not dumps:
        raise FileNotFoundError(
            f"no rollout dumps under {output_dir / 'rollouts'} — --relabel_only rebuilds from them, "
            f"so the run must have been collected with --dump_rollouts"
        )
    action_space = ActionSpace.load(output_dir / "action_space.npz")
    out_dir = output_dir / "relabel"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dump in dumps:
        z = np.load(dump)
        if "act" not in z.files:
            logger.warning(f"{dump.name} predates the act/rew columns --relabel_only needs — skipping")
            continue
        n_eps = len(z["ep_len"])
        records, tables = [], []
        ref_vla = action_space.from_rl(z["ref"]) if z["ref"].size else z["ref"]
        for j in range(n_eps):
            grid = z["ep_index"] == j
            steps = z["grid_t"][grid]
            tables.append(
                {int(t): (z["zrl"][grid][i], ref_vla[grid][i], z["proprio"][grid][i]) for i, t in enumerate(steps)}
            )
            am = z["act_ep_index"] == j
            ep_id = int(z["ep_id"][j]) if "ep_id" in z.files else j
            side = _lab.load_episode_labels(_lab.sidecar_path(output_dir, cfg.run_id, ep_id))
            records.append(
                _EpisodeRecord(
                    obs_seq=[],
                    act_seq=list(z["act"][am]),
                    rew_seq=list(z["rew"][am]),
                    success=bool(z["ep_success"][j]),
                    tok_prompt=np.zeros(0),
                    tok_mask=np.zeros(0),
                    btab={},
                    ep_id=ep_id,
                    events=(side or {}).get("events", {}),
                )
            )
        transitions = _assemble_from_tables(cfg, records, tables, action_space, channels)
        path = out_dir / f"transitions_{int(z['iteration']):05d}.npz"
        np.savez_compressed(path, **transitions)
        written.append(path)
        n_ev = {ch.name: sum(len(r.events.get(ch.name, [])) for r in records) for ch in channels}
        logger.info(f"relabel {dump.name} -> {path.name}: {len(transitions['reward'])} transitions, labels {n_ev}")
    return written


# ════════════════════════════════════════════════════════════════════════════
# Eval
# ════════════════════════════════════════════════════════════════════════════


def evaluate(
    cfg: Config,
    env_worker: W.EnvWorker,
    vla_forward,
    actor_apply,
    td3_state,
    pi0_state,
    rng,
    zrl_normalizer: RunningNorm,
    *,
    action_space: ActionSpace | None = None,
    vla_only: bool = False,
    critic_apply=None,
    monitor=None,
    frame_writer: _FrameWriter | None = None,
    iteration: int = 0,
    event_names: tuple[str, ...] = (),
    subgoal_names: tuple[str, ...] = (),
) -> dict:
    """Deterministic eval on distinct trials: actor mean, no noise, reference always
    provided (paper's deployment policy). vla_only=True executes the raw reference
    chunk instead (base-policy baseline; needs no action_space).

    critic_apply/monitor are the live-diagnostics hook (see rlt.live_monitor): when
    both are given, every chunk-step evaluates the critic on the state and the action
    ABOUT TO BE EXECUTED and hands the result to the monitor. Purely observational —
    the executed action is computed first and is not touched by this path, so eval
    numbers are identical with and without it.
    """
    if not vla_only and action_space is None:
        raise ValueError("actor eval requires an ActionSpace")
    desc = "eval(vla)" if vla_only else "eval(rlt)"

    successes: list[bool] = []
    ep_stats: list[dict[str, float]] = []
    n_passes = max(1, -(-cfg.eval_episodes // cfg.total_num_envs))
    for _p in range(n_passes):
        pool = env_worker.make_pool(task=cfg.task_id)  # fresh random reset states per pass
        rng, pass_rng = jax.random.split(rng)
        if monitor is not None:
            monitor.on_episode_start(_p, len(pool))
        sg_buf = _run_eval_pass(
            cfg,
            env_worker,
            vla_forward,
            actor_apply,
            td3_state,
            pi0_state,
            pass_rng,
            pool,
            action_space,
            vla_only,
            desc,
            zrl_normalizer,
            critic_apply=critic_apply,
            monitor=monitor,
            frame_writer=frame_writer,
            frame_prefix=f"eval_it{iteration:04d}_{'vla' if vla_only else 'rlt'}_p{_p}",
        )
        if monitor is not None:
            monitor.on_episode_end(_p, [bool(s.success) for s in pool])
        successes.extend(s.success for s in pool)
        # Event rates under the DEPLOYMENT policy. `success` comes from the env's
        # done/info["reward"], never from a reward array, so every arm grades itself on
        # the true task outcome.
        if event_names:
            ep_stats.extend(
                _events.episode_stats(
                    sg_buf[j],
                    subgoal_names,
                    event_names,
                    ep_len=max(len(sg_buf[j]) - 1, 0),
                    success=bool(s.success),
                )
                for j, s in enumerate(pool)
                if sg_buf[j]
            )
    return {
        "success_rate": float(np.mean(successes)),
        "n": len(successes),
        "events": _events.aggregate_stats(ep_stats),
    }


def _log_eval(output_dir: Path, iteration: int, kind: str, ev: dict, cfg: Config) -> None:
    """Append one eval to <output_dir>/eval_log.jsonl.

    Raw counts, not just the rate: a single 48-episode eval cannot resolve small effects
    (a 0.70 rate at n=48 carries a +-0.066 standard error), so comparisons pool
    successes/trials across seeds and iterations, and a pooled binomial needs the
    numerator and denominator, not an average of rates. The arm flags ride along so runs
    can be grouped without re-deriving them from the directory name.
    """
    import json

    n = int(ev["n"])
    rec = {
        "iteration": iteration,
        "kind": kind,
        "success_rate": float(ev["success_rate"]),
        "n_episodes": n,
        "n_success": round(ev["success_rate"] * n),
        # Per-event firing rates, occurrence counts and the fired-but-failed fraction
        # under the deployment policy. Emitted for EVERY arm, so the RLT and LVF runs
        # carry the same columns and the rates are directly comparable.
        "events": {k: float(v) for k, v in (ev.get("events") or {}).items()},
        "arm": {
            "bc_coef": cfg.bc_coef,
            "n_critics": cfg.n_critics,
            "gvf_mode": cfg.gvf_mode,
            "gvf_channels": cfg.gvf_channels,
            "gvf_ramp_steps": cfg.gvf_ramp_steps,
            "seed": cfg.seed,
            "task_id": cfg.task_id,
            "suite": cfg.suite,
        },
    }
    with (output_dir / "eval_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _run_eval_pass(
    cfg,
    env_worker,
    vla_forward,
    actor_apply,
    td3_state,
    pi0_state,
    rng,
    pool,
    action_space,
    vla_only,
    desc,
    zrl_normalizer: RunningNorm,
    critic_apply=None,
    monitor=None,
    frame_writer: _FrameWriter | None = None,
    frame_prefix: str = "",
) -> None:
    B = len(pool)
    C, ad = cfg.action_chunk, cfg.action_env_dim
    max_chunk_steps = cfg.max_episode_steps // C
    # Frames are BUFFERED and written at pass end, matching collect(): a PNG encode
    # between send_step and recv_step stretches the control period, and the arm is driven
    # by CartesianVelocityMotion with no duration, so distance per command scales with
    # that gap -- writing in-loop does not merely slow the eval, it changes the motion
    # being evaluated (measured: ~0.11 s per 640x480 PNG, twice per step, against a
    # ~0.12 s control period). Only the steps that will actually be written are kept, so
    # --save_frames_every divides the memory as well as the file count: at stride 1 and
    # 800 steps this is ~1.4 GB per episode per slot at 640x480, so raise the stride
    # rather than the env count if that is tight.
    ep_names = [f"{frame_prefix}_s{j}" for j in range(B)] if frame_writer is not None else []
    frame_buf: list[list[tuple[int, dict]]] = [[] for _ in range(B)] if frame_writer is not None else []
    if frame_writer is not None:
        for j, s_ in enumerate(pool):
            frame_buf[j].append((0, s_.obs))
    # Per-slot subgoal-predicate trace, so the event rates can be reported for the EVAL
    # policy and not only for the (noisy, exploring) rollout policy. Only the predicate
    # vector is kept, not the obs -- ~5 floats per step against the ~1.4 GB of frames a
    # full obs history would be. Shaped as a list of one-key obs dicts so
    # events.episode_stats consumes it exactly as it consumes rec.obs_seq.
    sg_buf: list[list[dict]] = [
        [{"subgoal": np.asarray(s_.obs["subgoal"])}] if "subgoal" in s_.obs else [] for s_ in pool
    ]
    for _t in tqdm.tqdm(range(max_chunk_steps), desc=desc, unit="chunk", leave=False, dynamic_ncols=True):
        if all(s.done for s in pool):
            break
        batched_obs = W.obs_multi_to_pi0(
            [s.obs for s in pool],
            [s.tok_prompt for s in pool],
            [s.tok_mask for s in pool],
            env_worker.action_dim,
            env_worker.state_norm,
        )
        rng, srng = jax.random.split(rng)
        zrl, a_full = vla_forward(pi0_state, srng, batched_obs)
        # Read-only: eval must never perturb the normalizer's frozen stats.
        zrl = zrl_normalizer.normalize(np.asarray(zrl, dtype=np.float32))
        a_full = np.asarray(a_full, dtype=np.float32)
        ref_flat = a_full[:, :C, :ad].reshape(B, C * ad)
        if vla_only:
            exec_flat = ref_flat
        else:
            proprio = np.stack(
                [
                    _rl_proprio(
                        s.obs, env_worker.action_dim, env_worker.state_norm, include_velocity=cfg.include_velocity
                    )
                    for s in pool
                ]
            )
            x = np.concatenate([zrl, proprio], axis=-1)
            ref_rl = action_space.to_rl(ref_flat)
            mu_rl = np.asarray(actor_apply(td3_state, jnp.asarray(x), jnp.asarray(ref_rl)))
            exec_flat = action_space.from_rl(np.clip(mu_rl, -cfg.action_clip, cfg.action_clip))
            # Same handover as training: the deployed policy is VLA until
            # rl_start_step, actor after.
            for j in range(B):
                if pool[j].steps < cfg.rl_start_step:
                    exec_flat[j] = ref_flat[j]

            # Live diagnostics. Q is evaluated on the action that is actually going
            # out (post-clip, in RL space — the critic's own input convention), so
            # this is the exact value of the executed chunk, not a reconstruction.
            if monitor is not None and critic_apply is not None:
                a_exec_rl = np.clip(mu_rl, -cfg.action_clip, cfg.action_clip)
                q = np.asarray(critic_apply(td3_state.critic, jnp.asarray(x), jnp.asarray(a_exec_rl)))
                for j in range(B):
                    if pool[j].done:
                        continue
                    tel = monitor.on_chunk(j, pool[j].steps, zrl[j], proprio[j], a_exec_rl[j], q[j])
                    # Relay to the robot-side operator when the backend supports it
                    # (bridge proxy only; the LIBERO subprocess env has no such hook).
                    if tel is not None and hasattr(pool[j].env, "set_telemetry"):
                        pool[j].env.set_telemetry(tel)
        exec_model = a_full[:, :C, :].copy()
        exec_model[:, :, :ad] = exec_flat.reshape(B, C, ad)
        action_raws = [W.unnormalize_action(env_worker.action_norm, exec_model[j]) for j in range(B)]

        steps_in_chunk = [min(C, cfg.max_episode_steps - s.steps) for s in pool]
        for sub in range(max(steps_in_chunk)):
            running = [j for j in range(B) if not pool[j].done and sub < steps_in_chunk[j]]
            for j in running:
                pool[j].env.send_step(action_raws[j][sub, :ad])
            for j in running:
                s = pool[j]
                obs, done, info = s.env.recv_step()
                s.obs = obs
                s.steps += 1
                if sg_buf[j] and "subgoal" in obs:
                    sg_buf[j].append({"subgoal": np.asarray(obs["subgoal"])})
                if frame_writer is not None and s.steps % frame_writer.every == 0:
                    # Store the TRUE step index: save_step filters on `step % every`, so
                    # buffering only multiples of it makes that filter a no-op rather
                    # than a second, compounding stride.
                    frame_buf[j].append((s.steps, obs))
                # See the matching comment in collect() -- done alone only means
                # success for the LIBERO sim path.
                reward = info.get("reward") if info else None
                is_success = done and (reward is None or reward > 0)
                if is_success:
                    s.success = True
                    s.done = True
                elif done or s.steps >= cfg.max_episode_steps:
                    s.done = True

    if frame_writer is not None:
        for j, s_ in enumerate(pool):
            for t, obs in frame_buf[j]:
                frame_writer.save_step(ep_names[j], t, obs)
            frame_buf[j].clear()  # ~GB per slot; do not hold it across passes
            frame_writer.finish(ep_names[j], success=bool(s_.success))
    return sg_buf


# ════════════════════════════════════════════════════════════════════════════
# Checkpointing (TD3 nets only; the frozen VLA+RLT checkpoint is referenced by path)
# ════════════════════════════════════════════════════════════════════════════

_NET_NAMES = ("actor", "actor_targ", "critic", "critic_targ")


def save_rlt_checkpoint(
    td3_state: _td3.TD3State,
    output_dir: Path,
    step: int,
    keep_last_n: int = 0,
    zrl_normalizer: RunningNorm | None = None,
) -> None:
    import orbax.checkpoint as ocp

    step_dir = (output_dir / str(step)).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        for name in _NET_NAMES:
            ckptr.save(
                str(step_dir / name),
                ocp.args.PyTreeSave(item={"params": jax.device_get(getattr(td3_state, name))}),
            )
    # The actor consumes STANDARDIZED z_rl, so the normalizer stats are part of the
    # policy -- without them a reloaded checkpoint silently runs on raw z_rl. Saved
    # per step dir so eval/resume can restore them (see RunningNorm.save).
    if zrl_normalizer is not None:
        step_dir.mkdir(parents=True, exist_ok=True)
        zrl_normalizer.save(step_dir / "zrl_norm.npz")
    logger.info(f"Saved TD3 checkpoint -> {step_dir}")

    if keep_last_n > 0:
        import shutil

        step_dirs = sorted(
            (d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)
        )
        for old_dir in step_dirs[:-keep_last_n]:
            shutil.rmtree(old_dir)


def load_rlt_checkpoint(resume_dir: str, td3_state: _td3.TD3State) -> tuple[_td3.TD3State, int]:
    step_dir, step = _resolve_td3_dir(resume_dir)
    replacements = {}
    for name in _NET_NAMES:
        state = getattr(td3_state, name)
        restored = _model.restore_params(str(step_dir / name), dtype=jnp.float32)
        state.replace_by_pure_dict(W._filter_params(restored, set(state.flat_state().keys())))
        replacements[name] = state
    logger.info(f"Restored TD3 nets from {step_dir} (iteration {step})")
    return td3_state._replace(**replacements), step


def _resolve_td3_dir(resume_dir: str) -> tuple[Path, int]:
    rdir = Path(resume_dir)
    if (rdir / "actor").exists():
        return rdir, (int(rdir.name) if rdir.name.isdigit() else 0)
    steps = sorted(int(p.name) for p in rdir.iterdir() if p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no checkpoint steps under {rdir}")
    return rdir / str(steps[-1]), steps[-1]


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The resolved config, so eval does not have to reconstruct it. A stage-2 run dir
    # otherwise records NOTHING about the architecture or the env: eval_log.jsonl is the
    # only place any of it appeared, and real-robot runs write none (eval_interval=0).
    # That left `n_critics`, `critic_input_norm`, `gvf_channels` and `max_episode_steps`
    # to be recovered by hand from critic/_METADATA before every evaluation, and two of
    # those are SILENT when wrong -- the eval reports a plausible number for a different
    # policy. scripts/eval_rlt_real.py reads this file and matches it automatically.
    # Rewritten on every launch, including a resume, so it always describes the process
    # that produced the checkpoints sitting next to it.
    (output_dir / "run_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    gvf_channels = cfg.parsed_gvf_channels
    # Fail at startup, not three hardware episodes in: a labeling key that shadows one
    # of the bridge's would be silent and expensive (see labeling.BRIDGE_KEYMAP).
    _lab.assert_no_key_collision(gvf_channels)
    _missing_phase = [n for n in cfg.parsed_phase_channels if n not in {c.name for c in gvf_channels}]
    if gvf_channels and _missing_phase:
        raise ValueError(
            f"--phase_channels names {_missing_phase} which is not in --gvf_channels "
            f"({[c.name for c in gvf_channels]}) — that phase boundary could never be reached, "
            f"so every later phase's GVF term would be permanently gated off."
        )
    if cfg.relabel_only:
        written = relabel_from_dumps(cfg)
        logger.info(f"relabel_only: wrote {len(written)} transition table(s) under {output_dir / 'relabel'}")
        return
    if gvf_channels:
        logger.info(
            "GVF channels: "
            + ", ".join(
                f"{ch.name}(key={ch.key} gamma={ch.gamma} lam={ch.lam} sign={ch.sign:+.0f} phases={ch.phases})"
                for ch in gvf_channels
            )
        )
        if (cfg.gvf_channels or "").strip() == "auto":
            # Echo the equivalent explicit spec: `auto` is a convenience, but a run
            # worth comparing against should pin the resolved set (and the checkpoints
            # record no config — see the `arm` dict in eval_log.jsonl).
            logger.info(
                "  --gvf_channels '"
                + ";".join(
                    f"{ch.name}:{ch.key}:{ch.gamma}:{ch.lam}:{ch.sign:+.0f}:" + ",".join(str(p) for p in ch.phases)
                    for ch in gvf_channels
                )
                + "'"
            )
    logger.info(
        f"RLT TD3 | task {cfg.task_id} ({cfg.suite}) | C={cfg.action_chunk} "
        f"stride={cfg.subsample_stride} bc_coef={cfg.bc_coef} | "
        f"gvf_mode={cfg.gvf_mode if gvf_channels else 'off (no channels)'}"
        + (f" ramp={cfg.gvf_ramp_steps}" if gvf_channels and cfg.gvf_ramp_steps else "")
        + f" | output {output_dir}"
    )

    rng = jax.random.PRNGKey(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    # ── frozen VLA + RLT ─────────────────────────────────────────────────────
    graphdef, pi0_state, model_config, train_cfg = W.load_pi0_model(
        cfg.checkpoint_dir, cfg.config_name, rlt_width=(cfg.rlt_width or None)
    )
    if not getattr(model_config, "rlt_enabled", False):
        raise ValueError(f"config {cfg.config_name} has no RLT module (rlt_enabled=False)")
    if cfg.action_chunk > model_config.action_horizon:
        raise ValueError("action_chunk must be <= the model action_horizon")
    # bf16 restore rounds the trained tokenizer; recast the rlt subtree to f32.
    rlt_state, rest_state = pi0_state.split(nnx_utils.PathRegex(".*rlt.*"), ...)
    rlt_state = jax.tree.map(lambda x: x.astype(jnp.float32), rlt_state)
    pi0_state = nnx.State.merge(rlt_state, rest_state)

    # ── norm stats (from the checkpoint's assets dir — critical) ─────────────
    if train_cfg.data.assets.assets_dir and train_cfg.data.assets.assets_dir.startswith("gs://"):
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(
                train_cfg.data,
                assets=dataclasses.replace(train_cfg.data.assets, assets_dir=str(Path(cfg.checkpoint_dir) / "assets")),
            ),
        )
    data_config = train_cfg.data.create(train_cfg.assets_dirs, model_config)
    norm_stats = _resolve_norm_stats(cfg.checkpoint_dir, data_config)
    state_norm = build_norm(norm_stats, data_config.use_quantile_norm, "state")
    action_norm = build_norm(norm_stats, data_config.use_quantile_norm, "actions")
    if state_norm is None or action_norm is None:
        logger.warning("state or action norm stats missing — performance will be near-random!")
    tok = PaligemmaTokenizer(max_len=model_config.max_token_len)

    # ── TD3 nets ─────────────────────────────────────────────────────────────
    proprio_dim = RAW_PROPRIO_DIM + (VEL_PROPRIO_DIM if cfg.include_velocity else 0)
    obs_dim = model_config.rlt_zrl_dim + proprio_dim
    chunk_action_dim = cfg.action_chunk * cfg.action_env_dim
    tcfg = td3_config(cfg)
    (actor_gd, critic_gd), td3_state = _td3.init_td3(tcfg, obs_dim, chunk_action_dim, seed=cfg.seed)
    update_fn = _td3.build_update_fn(tcfg, actor_gd, critic_gd)
    # Second trace with the GVF TD term dropped, selected at runtime once
    # --gvf_freeze_after_successes RL successes have been collected. Two compiled update
    # functions rather than a traced gate: the freeze is a ONE-WAY switch that fires at
    # most once per run, so one extra compile costs less than threading another argument
    # through every call site -- and it keeps the unfrozen path bit-identical.
    update_fn_frozen = (
        _td3.build_update_fn(dataclasses.replace(tcfg, gvf_freeze=True), actor_gd, critic_gd)
        if cfg.gvf_freeze_after_successes > 0 and not tcfg.gvf_freeze
        else None
    )
    actor_apply = _td3.build_actor_apply(actor_gd)
    logger.info(
        f"TD3: obs_dim={obs_dim} (zrl {model_config.rlt_zrl_dim} + proprio {proprio_dim}), "
        f"chunk_action_dim={chunk_action_dim}, hidden={tcfg.hidden}"
    )

    start_iter = 0
    if cfg.resume_from:
        td3_state, start_iter = load_rlt_checkpoint(cfg.resume_from, td3_state)

    buffer = ReplayBuffer(cfg.buffer_capacity, model_config.rlt_zrl_dim, proprio_dim, chunk_action_dim, gvf_channels)

    # ── env + VLA forward ────────────────────────────────────────────────────
    if cfg.real_robot_ports:
        task_prompts = parse_task_prompts(cfg.task_prompts)
        channels = [_BridgeChannel() for _ in cfg.real_robot_ports]
        for ch, port in zip(channels, cfg.real_robot_ports, strict=True):
            RobotBridgeServer(ch, port=port, metadata={"action_dim": model_config.action_dim}).start()
        logger.info(f"waiting for {len(channels)} robot connection(s) on ports {cfg.real_robot_ports}...")
        for ch in channels:
            ch.connected.wait()
        logger.info("all robots connected")
        env_worker = W.RealRobotEnvWorker(
            cfg, tok, model_config.action_dim, state_norm, action_norm, channels, task_prompts
        )
    else:
        env_worker = W.EnvWorker(cfg, tok, model_config.action_dim, state_norm, action_norm)
    # Column names for the predicate vector the env ships in obs["subgoal"]. Empty on
    # the real-robot path (no BDDL) and whenever the pool could not agree on one set.
    subgoal_names: tuple[str, ...] = tuple(getattr(env_worker, "subgoal_names", ()))
    if gvf_channels and cfg.auto_label and cfg.auto_label_source == "predicate" and subgoal_names:
        # A configured channel with no matching predicate column would be labeled by
        # nothing and train on an all-zero cumulant, which looks exactly like "the
        # event never happens" — say so rather than let it look like a null result.
        orphan = [ch.name for ch in gvf_channels if ch.name not in subgoal_names]
        if orphan:
            logger.warning(
                f"--gvf_channels names {orphan}, which this task's BDDL predicates "
                f"{list(subgoal_names)} do not provide — those channels get NO auto labels "
                f"and their heads will train on an all-zero cumulant. Use --gvf_channels auto, "
                f"or label them by hand/sidecar."
            )
    # Event diagnostics, resolved against the LIVE pool's predicate columns (a scene
    # whose affordance probe failed reports its column as permanently 0, and a pool
    # whose slots disagree reports none at all).
    event_names = tuple(n for n in subgoal_names if n not in _events.EXCLUDED_EVENTS)
    if event_names:
        logger.info(f"event diagnostics on predicates {list(event_names)}")
    vla_forward = build_vla_forward(graphdef, model_config, num_steps=cfg.num_denoise_steps)
    # Standardizes z_rl at extraction time (mean 0 / std 1 per dim), fixing the
    # tokenizer's normalize_targets-induced constant offset (||z_rl|| ~ 46.5,
    # near-constant across states) without retraining stage 1. Stats accumulate
    # during warmup rollouts (raw VLA reference, no RL updates yet) and freeze
    # once training starts, so the critic's input distribution stays stationary.
    zrl_normalizer = RunningNorm(model_config.rlt_zrl_dim)
    if cfg.resume_from:
        # The actor/critic were trained on z_rl STANDARDIZED by the run's own frozen
        # stats, so a resumed run must reuse them: a fresh RunningNorm (count == 0)
        # makes normalize() a no-op and feeds the loaded nets raw z_rl carrying the
        # tokenizer's ~46.5 offset — a distribution shift large enough to make the
        # restored policy useless while still running without error (see
        # RunningNorm.save). Older checkpoints predate zrl_norm.npz; those fall back
        # to re-accumulating over the resumed warmup rollouts.
        _norm_path = _resolve_td3_dir(cfg.resume_from)[0] / "zrl_norm.npz"
        if _norm_path.exists():
            zrl_normalizer = RunningNorm.load(_norm_path)
            logger.info(
                f"Restored z_rl normalizer from {_norm_path} (count={zrl_normalizer.count}, frozen={zrl_normalizer.frozen})"
            )
        else:
            logger.warning(f"{_norm_path} missing — z_rl stats will be re-accumulated over the resumed warmup rollouts")

    # ── event labeling ───────────────────────────────────────────────────────
    # ONE stdin owner for the whole run: the labeler handles its own keymap and
    # forwards everything else untouched. A second listener on the same terminal
    # drops keypresses nondeterministically. (The real-robot bridge's own Enter /
    # backspace / space handler lives in the OPERATOR-side process — train_pi05_real.py
    # on the robot desktop — and reaches this loop as `done`/`reward` over the
    # websocket, so it is unaffected either way.)
    labeler = _lab.KeyboardLabeler(gvf_channels, lookback_steps=cfg.label_lookback_steps) if gvf_channels else None
    # Continue the run-global episode numbering across a resume. Restarting at 0 would
    # make each resumed episode adopt the OLD sidecar of the same id (see
    # labeling.next_ep_id) -- labels from a different episode, applied silently.
    next_ep_id = _lab.next_ep_id(output_dir, cfg.run_id) if gvf_channels else 0
    if next_ep_id:
        logger.info(f"resuming episode numbering at ep_id {next_ep_id} (existing label sidecars found)")

    # ── loop ─────────────────────────────────────────────────────────────────
    vla_baseline_logged = False
    # Resumed runs are assumed past the critic warmup (the counter isn't persisted).
    total_updates = 0 if start_iter == 0 else cfg.critic_warmup_updates
    # RL-mode successes so far, and whether the GVF freeze has already fired. Warm-up
    # episodes are excluded (see Config.gvf_freeze_after_successes).
    rl_successes = 0
    gvf_frozen = False
    # RL-canonical action space, measured from the first rollout's reference chunks
    # and persisted so resumes/evals use the identical map.
    frame_writer = _FrameWriter(Path(cfg.save_frames_dir), cfg.save_frames_every) if cfg.save_frames_dir else None
    asp_path = output_dir / "action_space.npz"
    action_space: ActionSpace | None = None
    if asp_path.exists():
        action_space = ActionSpace.load(asp_path)
        logger.info(f"Loaded ActionSpace from {asp_path}")
    elif start_iter > 0:
        raise RuntimeError(f"--resume_from without {asp_path} — the RL action space would not match")
    # Aggregated over all warmup iterations (not just the last one) so the abort
    # gate below isn't a single-episode coin flip at total_num_envs=1 (e.g. real
    # robot, one env per robot port) — see warmup_min_success check.
    warmup_successes = 0
    warmup_episodes = 0
    # ── interaction budget ──────────────────────────────────────────────────
    # Cumulative ROLLOUT env steps and episodes, logged every iteration so methods can
    # be compared at matched INTERACTION budget rather than at matched iteration count.
    # Counts rollouts only (eval interaction is identical across arms by construction).
    # Both counters reset to 0 on --resume_from, since the pre-resume process's budget
    # is not knowable here -- sum across processes.
    total_env_steps = 0
    total_episodes = 0
    if labeler is not None:
        labeler.start()
    for it in range(start_iter + 1, cfg.num_iterations + 1):
        warmup = it <= start_iter + cfg.n_warmup_iters
        if not warmup:
            # Warmup rollouts (raw VLA reference) are done accumulating z_rl
            # stats — freeze so the critic's input distribution stays stationary
            # through RL training.
            zrl_normalizer.freeze()
        bc_phase = total_updates < cfg.critic_warmup_updates
        rollout_policy = "ref" if warmup else ("ref_noise" if bc_phase else "actor")
        t0 = time.time()
        rng, crng, arng = jax.random.split(rng, 3)
        records, roll_stats = collect(
            cfg,
            env_worker,
            vla_forward,
            actor_apply,
            td3_state,
            pi0_state,
            crng,
            np_rng,
            zrl_normalizer,
            policy=rollout_policy,
            action_space=action_space,
            desc=f"it{it}({rollout_policy})",
            labeler=labeler,
            ep_id_base=next_ep_id,
            frame_writer=frame_writer,
            iteration=it,
        )
        next_ep_id += len(records)
        total_env_steps += int(sum(len(r.act_seq) for r in records))
        total_episodes += len(records)
        _finalize_labels(cfg, records, gvf_channels, output_dir, subgoal_names)
        if action_space is None:
            refs = np.stack([entry[1] for rec in records for entry in rec.btab.values()])
            action_space = ActionSpace.from_ref_samples(
                refs,
                cfg.action_env_dim,
                action_norm=env_worker.action_norm,
                # ActionSpace is told the command magnitudes only; `eps` is an
                # exploration knob and has no business in the calibration.
                discrete_dims={d: scale for d, (scale, _eps) in cfg.parsed_discrete_action_dims.items()} or None,
            )
            action_space.save(asp_path)
        transitions = assemble_transitions(
            cfg,
            env_worker,
            vla_forward,
            pi0_state,
            arng,
            records,
            action_space,
            zrl_normalizer,
            channels=gvf_channels,
            dump_dir=(output_dir / "rollouts") if cfg.dump_rollouts else None,
            iteration=it,
        )
        buffer.add_batch(transitions)
        # Reward-signal guard: successes that finish BEFORE rl_start_step contribute
        # zero RL transitions — if most successes end pre-handover, the RL segment is
        # reward-free and the critic can only learn "everything fails" (observed with
        # rl_start_step=300 on a task whose successes finish at ~150-200 steps).
        # Terminal-reward rows, from the STORED OUTCOME FLAGS: `success & discount == 0`
        # is exactly _assemble_from_tables' own `success_inside`.
        n_rew_rows = (
            int(np.sum((transitions["success"] > 0.5) & (transitions["discount"] <= 0.0)))
            if len(transitions["reward"])
            else 0
        )
        n_succ = sum(r.success for r in records)
        if not warmup:
            rl_successes += n_succ
            if update_fn_frozen is not None and not gvf_frozen and rl_successes >= cfg.gvf_freeze_after_successes:
                gvf_frozen = True
                logger.info(
                    f"GVF heads FROZEN at iteration {it}: {rl_successes} RL successes "
                    f">= --gvf_freeze_after_successes {cfg.gvf_freeze_after_successes}. "
                    f"From here the critic backs up the task reward plus a FIXED subgoal "
                    f"potential; gvf_*_loss stays logged as a drift diagnostic."
                )
        n_succ_pre_k = sum(r.success and len(r.act_seq) <= cfg.rl_start_step for r in records)
        if cfg.rl_start_step > 0 and n_succ > 0 and n_succ_pre_k / n_succ > 0.5:
            logger.warning(
                f"{n_succ_pre_k}/{n_succ} successes ended before rl_start_step={cfg.rl_start_step} "
                f"— the RL segment is nearly reward-free; lower rl_start_step "
                f"(successful episodes run ~{np.mean([len(r.act_seq) for r in records if r.success]):.0f} steps)."
            )
        t_roll = time.time() - t0

        if warmup and cfg.warmup_min_success > 0:
            warmup_successes += n_succ
            warmup_episodes += roll_stats["n_episodes"]
            if it == start_iter + cfg.n_warmup_iters:
                agg_rate = warmup_successes / warmup_episodes if warmup_episodes else 0.0
                if agg_rate < cfg.warmup_min_success:
                    logger.warning(
                        f"WARMUP success {agg_rate:.2f} ({warmup_successes}/{warmup_episodes}) < "
                        f"{cfg.warmup_min_success} — the base VLA gives TD3 no positive signal on task "
                        f"{cfg.task_id}. Aborting after warmup would waste compute; check the checkpoint/task."
                    )
                    raise RuntimeError("base VLA success rate too low during warmup (see log)")

        # ── updates ──────────────────────────────────────────────────────────
        metrics_acc: dict[str, float] = {}
        n_updates = 0
        n_actor_updates = 0
        n_do_actor = 0
        if not warmup and len(buffer) >= cfg.min_buffer:
            # Exact count of transitions just added, rather than an rl_start_step-based
            # estimate of it.
            n_updates = min(int(cfg.utd * len(transitions["reward"])), cfg.max_updates_per_iter)
            for g in range(n_updates):
                batch = buffer.sample(cfg.batch_size, np_rng, success_frac=cfg.success_sample_frac)
                batch_j = {k: jnp.asarray(v) for k, v in batch.items()}
                rng, urng = jax.random.split(rng)
                do_actor = g % cfg.policy_delay == cfg.policy_delay - 1
                # BC warm-start: until the critic warm-up ends, actor steps imitate
                # the reference (no Q term) so the plain-MLP actor reaches
                # mu ~= a_ref before Q-gradients switch on.
                bc_only = total_updates + g < cfg.critic_warmup_updates
                n_do_actor += int(do_actor)
                n_actor_updates += int(do_actor and not bc_only)
                # Updates elapsed since the critic warm-up ended — drives the GVF
                # warm-up gate (see td3.gvf_gate).
                adapt_step = max(0, total_updates + g - cfg.critic_warmup_updates)
                ufn = update_fn_frozen if gvf_frozen else update_fn
                td3_state, m = ufn(td3_state, batch_j, urng, adapt_step, do_actor=do_actor, actor_bc_only=bc_only)
                for k, v in m.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + float(v)
            total_updates += n_updates
            # Average each metric over the steps that actually PRODUCED it. Actor
            # metrics are emitted only on do_actor steps (and the q_pi family only on
            # non-BC-only actor steps); dividing everything by n_updates diluted them
            # by policy_delay — the long-standing "q_pi_mean pinned at exactly
            # q_mean/2" mystery was THIS artifact, not critic geometry (true
            # q_pi ~= 0.93*q_mean once un-diluted).
            # NOTE these are checked FIRST below, so a key listed here wins over
            # _actor_keys: the q_eff_* family also arrives via gvf_actor_metric_keys
            # (it needs a zero placeholder on every update) but belongs in this group,
            # because like q_pi_mean it is zero on BC-only actor steps.
            _q_actor_keys = (
                "q_pi_mean",
                "q_pi_mean_det",
                "q_pi_ref",
                "q_pi_noref",
                "q_eff_pi_mean",
                "q_eff_pi_mean_val",
                "q_eff_pi_mean_det",
                "q_eff_pi_mean_det_val",
            )
            _actor_keys = (
                "actor_loss",
                "bc_loss",
                "bc_ref",
                "bc_noref",
                "gq_norm",
                "gq_val_norm",
                "gbc_norm",
                "beta",
                "force_ratio",
                # The GVF metrics read the post-update critic at mu_final, so like the
                # rest of this group they exist only on do_actor steps. (Under
                # gvf_mode="lookahead" the la_phi_*/la_F_*/q_eff_mean family is emitted
                # on EVERY update instead and is deliberately absent here, so it falls
                # through to the n_updates divisor with critic_loss.)
                *_td3.gvf_actor_metric_keys(gvf_channels, lookahead=(tcfg.gvf_mode == "lookahead")),
            )
            metrics_acc = {
                k: v
                / max(1, n_actor_updates if k in _q_actor_keys else (n_do_actor if k in _actor_keys else n_updates))
                for k, v in metrics_acc.items()
            }

        t_total = time.time() - t0

        mstr = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics_acc.items()))
        # Per-training-episode event diagnostics, over the ROLLOUT policy. Success is
        # rec.success (the env's own flag).
        ev_stats = (
            _events.aggregate_stats(
                [
                    _events.episode_stats(
                        r.obs_seq,
                        subgoal_names,
                        event_names,
                        ep_len=len(r.act_seq),
                        success=r.success,
                    )
                    for r in records
                ]
            )
            if event_names
            else {}
        )
        evstr = (" | " + " ".join(f"{k}={v:.3f}" for k, v in ev_stats.items())) if ev_stats else ""
        actor_note = " actor=BC-only(warm-start)" if (not warmup and n_updates and not n_actor_updates) else ""
        tqdm.tqdm.write(
            f"[it {it}] {'WARMUP ' if warmup else ''}success={roll_stats['success_rate']:.2f} "
            f"ep_len={roll_stats['mean_ep_len']:.0f} zrl_std={roll_stats['zrl_std']:.3f} "
            f"zrl_norm={roll_stats['zrl_norm']:.1f} zrl_disc={roll_stats['zrl_disc']:.2f} "
            f"new={len(transitions['reward'])} rew_rows={n_rew_rows} preK_succ={n_succ_pre_k}/{n_succ} "
            f"buffer={len(buffer)} env_steps={total_env_steps} episodes={total_episodes} "
            f"updates={n_updates}(actor {n_actor_updates}){actor_note} | {mstr}{evstr} | "
            f"roll {t_roll:.0f}s total {t_total:.0f}s"
        )

        # ── eval + checkpoint ────────────────────────────────────────────────
        # Skip actor eval during the BC warm-start: the plain-MLP actor is mid-imitation
        # and never executes in that phase, so its success rate is not meaningful.
        past_bc_phase = total_updates >= cfg.critic_warmup_updates
        if (
            not warmup
            and past_bc_phase
            and cfg.eval_interval > 0
            and (it - start_iter - cfg.n_warmup_iters) % cfg.eval_interval == 0
        ):
            rng, erng = jax.random.split(rng)
            if not vla_baseline_logged:
                base = evaluate(
                    cfg,
                    env_worker,
                    vla_forward,
                    actor_apply,
                    td3_state,
                    pi0_state,
                    erng,
                    zrl_normalizer,
                    vla_only=True,
                    frame_writer=frame_writer,
                    iteration=it,
                    event_names=event_names,
                    subgoal_names=subgoal_names,
                )
                tqdm.tqdm.write(f"[it {it}] EVAL vla-baseline success={base['success_rate']:.2f} n={base['n']}")
                _log_eval(output_dir, it, "vla_baseline", base, cfg)
                vla_baseline_logged = True
            rng, erng = jax.random.split(rng)
            ev = evaluate(
                cfg,
                env_worker,
                vla_forward,
                actor_apply,
                td3_state,
                pi0_state,
                erng,
                zrl_normalizer,
                action_space=action_space,
                frame_writer=frame_writer,
                iteration=it,
                event_names=event_names,
                subgoal_names=subgoal_names,
            )
            evstr = " ".join(f"{k}={v:.3f}" for k, v in (ev.get("events") or {}).items())
            tqdm.tqdm.write(
                f"[it {it}] EVAL rlt success={ev['success_rate']:.2f} n={ev['n']}" + (f" | {evstr}" if evstr else "")
            )
            _log_eval(output_dir, it, "rlt", ev, cfg)
        if cfg.save_interval > 0 and it % cfg.save_interval == 0:
            save_rlt_checkpoint(td3_state, output_dir, it, keep_last_n=cfg.keep_last_n, zrl_normalizer=zrl_normalizer)

    save_rlt_checkpoint(
        td3_state, output_dir, cfg.num_iterations, keep_last_n=cfg.keep_last_n, zrl_normalizer=zrl_normalizer
    )
    if labeler is not None:
        labeler.stop()
    env_worker.close()
    logger.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
