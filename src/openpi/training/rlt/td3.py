"""TD3 for RLT chunk-level transitions (RL Token paper, Sec. IV-B / Alg. 1).

Faithful recipe: twin critics with min-target and target networks (polyak tau),
chunk-level backup y = reward + discount * min_i Q'_i(x', a') where `reward` is
the within-chunk discounted sum and `discount` encodes gamma^n / terminal 0
(plus, under gvf_mode="critic", a subgoal-GVF shaping bonus on `reward` — see
TD3Config.gvf_mode);
Gaussian actor pi_theta = N(mu, sigma_explore^2 I) conditioned on the VLA reference chunk,
with the Q term evaluated under a reparameterized sample (paper Eq. 5) and BC
regularization beta*||mu - a_ref||^2 and 50% reference-action dropout (input
masking only — the BC target and the target-policy backup always use the true
reference); target policy smoothing noise; 2 critic updates per actor update.

Everything is pure-functional over nnx.State pytrees (the repo's split/merge
pattern); the polyak update mirrors scripts/train.py's EMA tree_map.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.training.rlt.labeling import GvfChannel
from openpi.training.rlt.networks import RltActor
from openpi.training.rlt.networks import RltCritic

_GVF_MODES = ("critic", "actor", "both", "lookahead")
_ACTOR_Q_EVAL = ("sample", "mean")


@dataclasses.dataclass(frozen=True)
class TD3Config:
    gamma: float = 0.99  # per-env-step discount (backup uses gamma^n via `discount`)
    tau: float = 0.005  # polyak coefficient for target networks
    policy_delay: int = 2  # critic updates per actor update (paper: 2)
    utd: float = 5.0  # grad updates per new boundary chunk-transition (paper: ~5)
    batch_size: int = 256
    sigma_explore: float = 0.1  # fixed policy std at rollout, normalized action units
    target_noise: float = 0.1  # TD3 target policy smoothing std
    target_noise_clip: float = 0.3
    # bc_coef/actor_lr retuned from 0.1/3e-4 after an observed collapse: the actor
    # exploited the young critic's extrapolation errors and left the VLA manifold
    # (bc_loss ~3.4) before the sparse reward had propagated through the bootstrap.
    # NOTE beta weighs the Eq.-5 SUM-over-dims norm ||mu - a_ref||^2 (not a per-dim
    # mean): per-dim deviation = sqrt(bc_loss / (C*d)), so bc_loss ~0.07 on a 70-dim
    # chunk is ~0.03 per dim.
    bc_coef: float = 0.5  # beta in L_pi = E_{a~pi}[-Q(x, a)] + beta*||mu - a_ref||^2
    # Where the actor's Q term is evaluated. The BC term is unaffected either way: for
    # fixed sigma, E_eps[||mu + sigma*eps - ref||^2] = ||mu - ref||^2 + sigma^2*C*d and the
    # extra term is constant in theta, so BC always uses the raw mu.
    #   "sample" (default) — paper Eq. 5, E_{a~pi_theta}[-Q(x, a)], via a reparameterized
    #       sample a = mu + sigma_explore*eps. Equals -Q(mu) - (sigma^2/2)*tr(grad^2_a Q)
    #       + O(sigma^4), i.e. it adds a critic-curvature penalty that keeps the actor off
    #       sharp Q ridges, and it queries the critic on the noisy action shell the replay
    #       buffer was actually filled from rather than at the OOD deterministic mean.
    #   "mean" — the pre-Eq.5 behavior, -Q(x, mu) at the deterministic mean. NOT the
    #       paper's objective; kept as an ablation control against "sample". Reproduces the
    #       legacy update bit-exactly (pinned by td3_test.test_mean_mode_matches_pre_eq5).
    actor_q_eval: str = "sample"
    ref_dropout: float = 0.5  # reference-action input dropout (paper: 50%)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    action_clip: float = 1.0  # actions clipped to +-action_clip (normalized space)
    hidden: tuple[int, ...] = (256, 256)  # (512, 512, 512) for the hard-task variant
    n_critics: int = 5
    """Ensemble size. Measured split-half reliability of the critic's action gradient
    at 2 heads was only Pearson 0.673 on 25k LIBERO grid points; Spearman-Brown projects
    ~0.91 at 5. NOT parameter-compatible -- do not resume a run across a change."""
    target_subset_size: int = 2
    """REDQ-style backup: the target is min over a random subset of this many heads
    (resampled per update), not min over all n_critics. Min over a 5-head ensemble is
    far more pessimistic than over 2 and would depress the whole value scale; the
    random subset keeps the twin-min pessimism level. subset >= n_critics degenerates
    to min-over-all, which is the exact twin-critic backup."""
    critic_input_norm: bool = False
    """Switches the critic to a dual-encoder architecture (RltCritic/_QMlp,
    networks.py): x and a are each linearly projected to hidden[0] and LayerNormed
    BEFORE the concat, instead of concatenated raw into a single first Linear.
    Fixes the V(s)-shortcut: ||x|| ~ 47 over ~2065 dims vs ||a|| ~ 2-3 over 70 dims
    means a shared first layer's fan-in gives the state ~97% of every
    preactivation's variance regardless of per-dim scale, so the critic fits TD
    targets as V(s) and dQ/da decays to ~0 (observed: gq_norm ~20x smaller than
    gbc_norm and falling). NOT parameter-free (adds x_enc/a_enc/out per critic
    head) — do not resume a run across a flip of this flag."""

    # ── subgoal GVF critics ──────────────────────────────────────────────────
    # Auxiliary General Value Functions predicting labeled subgoal EVENTS (grasp
    # achieved, holder collision), used to densify the credit the policy learns from.
    # Motivation: the task reward is sparse (~2% of rows) and
    # its credit chain is ~40 chunks long, so dQ_task/da is near-noise at the two
    # states that actually decide the episode. A subgoal channel is labeled on EVERY
    # episode (not just successes) and has a ~5-10 chunk chain, so it is both far
    # better estimated and far more action-sensitive.
    #
    # HOW they enter is `gvf_mode` below. In either mode the per-row, per-channel
    # weight is the same quantity,
    #
    #     w[i, k] = sign_k * lam_k * phase_gate(phase_i, phases_k) * warmup_gate
    #
    # so `sign` is applied at the point of USE and never folded into the cumulant --
    # a logged head value stays directly readable as "probability of this event".
    gvf_channels: tuple[GvfChannel, ...] = ()
    """Empty (the default) disables the whole mechanism: no heads are built, no batch
    keys are read, and the update is bit-identical to the task-only baseline."""
    gvf_mode: str = "critic"
    """Where the GVF heads' values are consumed.

    "critic" (default) -- REWARD SHAPING. The heads' conservatively reduced value at
      the transition's own (x, a) is added to the task reward inside the backup:

          bonus_i = sum_k w[i, k] * GVF_k(x_i, a_i)         (target net, stop_grad)
          y_i     = reward_i + bonus_i + discount_i * min_j Q'_j(x'_i, a'_i)

      The actor loss is then EXACTLY the RLT baseline, -Q_0(x, mu) + beta*||mu-ref||^2,
      and the GVF signal reaches the policy only through Q_task. Because the bonus is
      evaluated at the transition's own action, Q_task regresses onto an
      action-sensitive quantity and inherits d(GVF)/da at every step -- which is the
      point, given dQ_task/da is near-noise under the sparse terminal reward.

      Two consequences to hold onto:

      * NOT policy-invariant. This is a plain value bonus, not the potential-based
        gamma*Phi(x') - Phi(x) of Ng et al., so the shaped optimum can differ from the
        true task optimum. Under "actor" there was always an arm provably optimizing
        the real objective; under "critic" there is not. (A potential-based variant
        would also need Phi to be state-only -- Phi(x, a) breaks the theorem -- so it
        is a different construction, not a flag.)
      * `lam` MEANS SOMETHING ELSE HERE, and the actor-mode defaults are far too big.
        The bonus is DENSE (a GVF lives in [0, 1] at essentially every step) while the
        task reward is sparse (~2% of rows at 1.0, batch mean ~0.02). At lam=0.3 the
        shaping term is ~15x the mean task reward per step and, with gamma=0.99, adds
        ~lam/(1-gamma) = 30 to the value scale against a task Q of ~1 -- Q_task becomes
        a GVF predictor with a rounding error of task reward attached, and every
        quantity tuned against the old Q scale (bc_coef, the force ratio)
        is silently rescaled with it. Start around lam 0.01-0.03 and watch
        `gvf_bonus_frac` (the shaping share of the immediate signal) and `q_mean`.

    "actor" -- the original composite-steering path: the reward is untouched and the
      reduced GVFs are added to the actor's -Q term as sum_k w[i,k] * GVF_k(x, mu).
      Kept so the two mechanisms can be A/B'd on the same checkpoints.

    "both" -- shaping AND steering. Double-counts the same signal on purpose; only
      useful as a diagnostic.

    "lookahead" -- LOOK-AHEAD ADVICE (Wiewiora, Cottrell & Elkan, ICML 2003). The one
      construction here that is provably policy-invariant while still handing the actor
      d(GVF)/da directly. Define the state-ACTION potential from the same weights,

          Phi(x, a) = sum_k w[k] * gvf_reduced_k(x, a)          (target heads)

      and change BOTH sides of the update together:

          y     = reward + F + discount * min_subset Q'(x', a')
          F     = discount * Phi_next(x', a') - Phi(x, a)
          L_pi  = -mean( Q_0(x, a_pi) + Phi(x, a_pi) ) + BC

      At the fixed point the critic learns Q_Phi = Q_task - Phi (that is what the F
      term subtracts off), so the actor's objective Q_Phi + Phi IS the task Q exactly,
      for ANY lam. The optimal policy is therefore unchanged no matter how the channels
      are weighted -- the advice can only change the PATH the actor takes to it, never
      the destination. This is what "critic" mode cannot offer: a plain value bonus is
      not potential-based and moves the shaped optimum.

      The identity holds only if the SAME Phi appears on both sides, so every Phi in
      this mode is read off the TARGET heads (`state.critic_targ`) -- the actor's term
      included, unlike "actor" mode which reads the online heads. Phi_next uses the
      weights of the NEXT-state phase (`batch["next_phase"]`), not the current one.

      Unlike "critic", Phi is NOT accumulated over time: F telescopes along a
      trajectory, so the shaping contributes at most Phi(x_0, a_0) to any return and
      Q_Phi stays inside the task Q's scale plus |Phi| <= sum_k |lam_k| * |GVF_k|.
      The lam/(1-gamma) blow-up warned about under "critic" does NOT apply here, and
      `lam` needs no rescaling against the sparse task reward.

      Requires `next_phase` in the batch; missing it is an error, never a fallback to
      `phase` (a wrong Phi_next silently breaks the telescoping and the invariance).

    With `gvf_channels=()` this field is inert in every mode."""
    gvf_actor_start_step: int = 0
    """Warm-up gate on the GVF terms, in the SAME units as `adapt_step` (updates
    elapsed since the critic warm-up ended). Every channel's `lam` is multiplied by 0
    until `adapt_step >= gvf_actor_start_step`.

    REQUIRED, not tidiness. A freshly initialized GVF head does not produce small
    gradients, it produces NOISE gradients of ordinary magnitude, and with a constant
    lam that noise goes straight into the actor at full strength -- the same failure
    shape as the documented adaptive-beta collapse (a young critic's g is noise, and
    steering on noise is what kills a run). train_rlt_libero.py derives the default as
    one further critic-warmup's worth of updates, i.e. 2x the warm-up in absolute
    terms; the heads' TD losses are trained throughout that window regardless.

    The name predates `gvf_mode` and is now a misnomer under "critic": the gate covers
    whichever term the mode enables, and under "critic" it gates the REWARD BONUS. The
    argument for it is if anything stronger there -- a noise bonus does not merely
    misdirect one actor step, it is regressed into Q_task and then bootstrapped.

    Either way the heads' OWN TD losses train from step 0, which is what makes lam=0.0
    an "auxiliary heads only" arm. Note the heads share no weights with the task
    critic (networks._GvfTwin builds independent MLPs), so that arm is a null by
    construction -- it is a wiring check, not a representation-sharing ablation."""

    gvf_ramp_steps: int = 0
    """Length, in `adapt_step` units, of a LINEAR ramp of the GVF gate from 0 to 1
    starting at `gvf_actor_start_step`. 0 (the default) keeps the historical hard
    0 -> 1 switch exactly, so every existing run and every other mode is bit-identical.

    The reason it exists is specific to potential-based shaping: under gvf_mode
    "lookahead" the critic is regressing onto Q_task - Phi, so the moment Phi changes
    the critic's whole target surface moves and it needs time to re-learn the
    difference. A hard switch injects that entire discrepancy in one update, and the
    actor reads Q_Phi + Phi through the transient. Ramping lam in spreads the
    re-learning over `gvf_ramp_steps` updates and bounds the per-update bias. The
    invariance argument is unaffected: the gate multiplies `w` and therefore appears
    IDENTICALLY in the critic's F and in the actor's Phi, which is all the telescoping
    needs."""

    gvf_freeze: bool = False
    """Stop training the GVF heads: their TD term is dropped from the critic loss so
    they receive exactly zero gradient and stay at whatever a resumed checkpoint holds.

    The shaping bonus still reads them (off the TARGET heads, as always), so the reward
    keeps its subgoal term -- it just becomes a FIXED function of (state, action)
    instead of one that re-fits as the policy moves. That is the point: the event
    labels the heads train on come from the simulator's predicate detectors, which do
    not exist on a real robot. Freezing asks whether the learned GVFs are useful
    shaping on their own, with no further privileged grounding.

    Safe because `y` is already fully stop_gradient'd, so the bonus was never a
    gradient path into the twins -- the dropped TD term is the only one. Their polyak
    targets then sit still too (identical params in and out).

    The per-channel losses are still COMPUTED and logged, so `gvf_*_loss` remains a
    live read on how far the frozen heads have drifted from their own targets."""

    def __post_init__(self):
        if self.gvf_mode not in _GVF_MODES:
            raise ValueError(f"gvf_mode must be one of {sorted(_GVF_MODES)}, got {self.gvf_mode!r}")
        if self.actor_q_eval not in _ACTOR_Q_EVAL:
            raise ValueError(f"actor_q_eval must be one of {sorted(_ACTOR_Q_EVAL)}, got {self.actor_q_eval!r}")


class TD3State(NamedTuple):
    actor: Any  # nnx.State
    actor_targ: Any
    critic: Any
    critic_targ: Any
    actor_opt: Any  # optax states
    critic_opt: Any


def polyak(target, online, tau: float):
    return jax.tree.map(lambda t, o: (1.0 - tau) * t + tau * o, target, online)


def make_optimizers(cfg: TD3Config) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    return optax.adam(cfg.actor_lr), optax.adam(cfg.critic_lr)


def init_td3(cfg: TD3Config, obs_dim: int, chunk_action_dim: int, seed: int = 0):
    """Build actor/critic + targets + optimizer states. Returns (graphdefs, TD3State)."""
    actor = RltActor(obs_dim, chunk_action_dim, hidden=cfg.hidden, rngs=nnx.Rngs(seed))
    critic = RltCritic(
        obs_dim,
        chunk_action_dim,
        hidden=cfg.hidden,
        n_critics=cfg.n_critics,
        input_norm=cfg.critic_input_norm,
        gvf_signs=tuple(ch.sign for ch in cfg.gvf_channels),
        rngs=nnx.Rngs(seed + 1),
    )
    actor_gd, actor_state = nnx.split(actor)
    critic_gd, critic_state = nnx.split(critic)
    actor_opt, critic_opt = make_optimizers(cfg)
    state = TD3State(
        actor=actor_state,
        actor_targ=jax.tree.map(jnp.copy, actor_state),
        critic=critic_state,
        critic_targ=jax.tree.map(jnp.copy, critic_state),
        actor_opt=actor_opt.init(actor_state),
        critic_opt=critic_opt.init(critic_state),
    )
    return (actor_gd, critic_gd), state


def _rl_state(batch: dict[str, jax.Array], prefix: str = "") -> jax.Array:
    zrl = batch[f"{prefix}zrl"].astype(jnp.float32)
    return jnp.concatenate([zrl, batch[f"{prefix}proprio"]], axis=-1)


def phase_mask(phase: jax.Array, phases: tuple[int, ...]) -> jax.Array:
    """HARD 0/1 gate selecting the episode phases a channel's actor term is active in.

    Hard, not soft: a grasp GVF has nothing to say once the object is in hand, and a
    soft weight there would keep a stale head steering the actor through the whole
    second half of every episode. An EMPTY `phases` tuple means "not phase-gated"
    (active everywhere) -- the alternative reading, active nowhere, would silently
    zero a channel the operator had bothered to label.
    """
    if not phases:
        return jnp.ones(phase.shape, jnp.float32)
    m = jnp.zeros(phase.shape, jnp.float32)
    for p in phases:
        m = m + (phase == p).astype(jnp.float32)
    return jnp.clip(m, 0.0, 1.0)


def gvf_gate(cfg: TD3Config, adapt_step: jax.Array | int) -> jax.Array:
    """Warm-up gate multiplying every channel's `lam` -- a hard 0/1 switch at
    `gvf_actor_start_step`, or a linear ramp over `gvf_ramp_steps` updates past it.

    One definition for every mode and every term, which is load-bearing under
    gvf_mode="lookahead": the gate has to appear identically in the critic's F and in
    the actor's Phi or the two stop telescoping and the policy-invariance argument
    fails. `gvf_ramp_steps=0` reproduces the original expression EXACTLY (same branch,
    same float), so the other modes stay bit-identical.
    """
    step = jnp.asarray(adapt_step, jnp.float32)
    if cfg.gvf_ramp_steps <= 0:
        return (step >= cfg.gvf_actor_start_step).astype(jnp.float32)
    return jnp.clip((step - cfg.gvf_actor_start_step) / float(cfg.gvf_ramp_steps), 0.0, 1.0)


def _gvf_metric_zeros(channels: tuple[GvfChannel, ...], *, lookahead: bool = False) -> dict:
    """Placeholders so every GVF metric key exists on every update, the way the task
    actor metrics do -- the host divides each key by the number of steps that emit it
    (train_rlt_libero.py's _actor_keys), which needs the key present, not absent."""
    z: dict[str, jax.Array] = {}
    for ch in channels:
        z[f"gvf_{ch.name}_mean"] = jnp.zeros(())
        z[f"gvf_{ch.name}_pi"] = jnp.zeros(())
        z[f"g_gvf_{ch.name}_norm"] = jnp.zeros(())
        z[f"force_ratio_{ch.name}"] = jnp.zeros(())
        # Alignment probe, emitted in EVERY mode: la_cos_k is cos(d GVF_k/da, dQ_0/da)
        # and la_lam_bound_k is the largest lam for which adding lam*GVF_k still points
        # uphill on Q_0 (see the derivation at the point of use). These are the numbers
        # a lam=0 run exists to measure -- there Q_0 is an unshaped task critic, so the
        # bound is the honest one -- hence they are not gated on gvf_mode.
        z[f"la_cos_{ch.name}"] = jnp.zeros(())
        z[f"la_cos_{ch.name}_val"] = jnp.zeros(())
        z[f"la_lam_bound_{ch.name}"] = jnp.zeros(())
        z[f"la_lam_bound_{ch.name}_val"] = jnp.zeros(())
    if channels:
        z["force_ratio_total"] = jnp.zeros(())
    if channels and lookahead:
        for k in (
            "q_eff_pi_mean",
            "q_eff_pi_mean_det",
            "la_g_phi_norm",
            "la_g_resid_norm",
            "la_cos_phi_resid",
        ):
            z[k] = jnp.zeros(())
            z[f"{k}_val"] = jnp.zeros(())
    return z


def gvf_actor_metric_keys(channels: tuple[GvfChannel, ...], *, lookahead: bool = False) -> tuple[str, ...]:
    """Keys emitted on do_actor steps only (see _gvf_metric_zeros)."""
    return tuple(_gvf_metric_zeros(channels, lookahead=lookahead))


def build_update_fn(cfg: TD3Config, actor_gd, critic_gd):
    actor_opt, critic_opt = make_optimizers(cfg)

    @functools.partial(jax.jit, static_argnames=("do_actor", "actor_bc_only"))
    def update_fn(
        state: TD3State,
        batch: dict[str, jax.Array],
        rng: jax.Array,
        adapt_step: jax.Array | int = 0,
        *,
        do_actor: bool,
        actor_bc_only: bool = False,
    ):
        """adapt_step: updates elapsed since critic warmup ended, driving the GVF
        warm-up gate (see gvf_gate). Traced, not static -- it changes every call and a
        static value would retrigger compilation on every update."""
        noise_rng, drop_rng, diag_rng = jax.random.split(rng, 3)
        x = _rl_state(batch)
        x_next = _rl_state(batch, "next_")
        b = x.shape[0]
        ones = jnp.ones((b, 1), jnp.float32)

        # ── shared GVF weights ──────────────────────────────────────────────
        # w[i, k] = sign_k * lam_k * phase_gate_ik * warmup_gate, constant w.r.t. every
        # parameter being differentiated. Computed here, above the backup, because
        # gvf_mode="critic" needs it for the reward bonus and the actor (mode "actor")
        # needs the identical array later -- one definition, so the two modes can never
        # drift apart.
        #
        # `lam` is a FIXED hyperparameter: no gradient-norm normalization, no adaptive
        # scaling, no per-state modulation.
        # from fixed corrections.
        channels = cfg.gvf_channels
        shape_critic = bool(channels) and cfg.gvf_mode in ("critic", "both")
        steer_actor = bool(channels) and cfg.gvf_mode in ("actor", "both")
        lookahead = bool(channels) and cfg.gvf_mode == "lookahead"
        if channels:
            gvf_warm = gvf_gate(cfg, adapt_step)
            gates = jnp.stack([phase_mask(batch["phase"], ch.phases) for ch in channels], axis=-1)  # [B, K]
            lam_vec = jnp.asarray([ch.lam for ch in channels], jnp.float32) * gvf_warm  # [K]
            sign_vec = jnp.asarray([ch.sign for ch in channels], jnp.float32)  # [K]
            gvf_w = jax.lax.stop_gradient(sign_vec[None, :] * lam_vec[None, :] * gates)  # [B, K]
            targ = nnx.merge(critic_gd, state.critic_targ)
        else:
            gvf_w, targ = None, None
        if lookahead:
            # Phi_next is weighted by the NEXT state's phase, not this one's. The
            # potential is a function of (x, a) and the phase is part of what the
            # weights read off that state, so using `phase` here would evaluate Phi at
            # x' under the wrong channel set and break the telescoping F sums the whole
            # invariance argument rests on.
            if "next_phase" not in batch:
                raise KeyError(
                    "gvf_mode='lookahead' needs a `next_phase` column in the batch (the episode phase "
                    "at the transition's NEXT state). Buffers collected before it was added do not have "
                    "it; re-collect, or re-derive the table with --relabel_only. Falling back to `phase` "
                    "is deliberately NOT done -- it would silently break the telescoping of "
                    "F = discount*Phi(x', a') - Phi(x, a) and with it the policy-invariance guarantee."
                )
            gates_next = jnp.stack([phase_mask(batch["next_phase"], ch.phases) for ch in channels], axis=-1)
            gvf_w_next = jax.lax.stop_gradient(sign_vec[None, :] * lam_vec[None, :] * gates_next)  # [B, K]
        else:
            gvf_w_next = None

        def phi_of(module, x_: jax.Array, a_: jax.Array, w_: jax.Array) -> jax.Array:
            """Phi(x, a) = sum_k w[k] * gvf_reduced_k(x, a) -> [B]."""
            return jnp.sum(w_ * module.gvf_reduced(x_, a_), axis=-1)

        # ── Critic update: y = r + bonus + discount * min_i Q'_i(x', a') ────────
        a_next = nnx.merge(actor_gd, state.actor_targ)(x_next, batch["next_ref"], ones)
        smooth = jnp.clip(
            cfg.target_noise * jax.random.normal(noise_rng, a_next.shape),
            -cfg.target_noise_clip,
            cfg.target_noise_clip,
        )
        a_next = jnp.clip(a_next + smooth, -cfg.action_clip, cfg.action_clip)
        q_next = nnx.merge(critic_gd, state.critic_targ)(x_next, a_next)  # [B, n]
        if cfg.target_subset_size >= cfg.n_critics:
            # Degenerate case (and the legacy n_critics=2, subset=2 setting): min over
            # every head. Kept as a separate branch so it consumes no RNG and is
            # bit-identical to the pre-REDQ backup.
            q_targ = jnp.min(q_next, axis=-1)
        else:
            # REDQ: min over a random subset resampled per update. fold_in rather than
            # another split of `rng`, so the noise/dropout/diagnostic keys above keep
            # the exact values (and the exact consumption order) they had before this
            # branch existed.
            sub_idx = jax.random.permutation(jax.random.fold_in(rng, 0x5EED), cfg.n_critics)[: cfg.target_subset_size]
            q_targ = jnp.min(q_next[:, sub_idx], axis=-1)
        # ── GVF reward shaping (gvf_mode="critic"/"both") ──────────────────────
        # bonus_i = sum_k w[i, k] * GVF_k(x_i, a_i), read off the TARGET heads at the
        # transition's OWN (x, a) -- the same state-action the task backup is a target
        # for, which is what makes Q_task inherit d(GVF)/da rather than only the
        # dynamics-mediated sensitivity a next-state potential would give.
        #
        # Target net, not online: the bonus is part of y, and reading the online heads
        # would feed their own within-step noise straight into the regression target
        # they are bootstrapped from. (`y` is stop_gradient'd regardless, so nothing
        # backpropagates from the task loss into the GVF heads either way -- the heads
        # are trained ONLY by their own TD losses below.)
        bonus = jnp.zeros((b,), jnp.float32)
        gvf_bonus_per_ch = jnp.zeros((b, len(channels)), jnp.float32)
        if shape_critic:
            gvf_bonus_per_ch = jax.lax.stop_gradient(gvf_w * targ.gvf_reduced(x, batch["action"]))  # [B, K]
            bonus = jnp.sum(gvf_bonus_per_ch, axis=-1)
        # ── look-ahead advice (gvf_mode="lookahead") ───────────────────────────
        # F = discount * Phi(x', a') - Phi(x, a), the potential-based shaping of
        # Wiewiora et al. (2003) generalized to a state-ACTION potential. Because F
        # telescopes along any trajectory, adding it to the reward shifts every return
        # by exactly -Phi(x_0, a_0) and leaves the ARGMAX policy of the shaped MDP equal
        # to the task's -- for any lam -- provided the actor optimizes Q_Phi + Phi
        # rather than Q_Phi alone (it does; see the actor loss below).
        #
        # Three details the identity depends on:
        #   * a' is the SAME smoothed target-actor action the bootstrap uses, so Phi'
        #     and Q' are read at one point and no extra forward is spent.
        #   * `discount` is gamma^n, and 0 on terminal rows -- where F correctly
        #     collapses to -Phi(x, a) (no next state to carry the potential forward).
        #   * TARGET heads, not online. The bonus above has the same rule, and here it
        #     is stronger: the actor's Phi must be the numerically identical function,
        #     and only the target copy is stable within an update.
        # The min over critic heads commutes with F: F is a common additive constant
        # w.r.t. the head index, so min_j(Q'_j) + F == min_j(Q'_j + F) and no special
        # handling of the REDQ subset is needed.
        phi_cur = jnp.zeros((b,), jnp.float32)
        shaping_f = jnp.zeros((b,), jnp.float32)
        if lookahead:
            phi_cur = jax.lax.stop_gradient(phi_of(targ, x, batch["action"], gvf_w))
            phi_next = jax.lax.stop_gradient(phi_of(targ, x_next, a_next, gvf_w_next))
            shaping_f = batch["discount"] * phi_next - phi_cur
        y = jax.lax.stop_gradient(batch["reward"] + bonus + shaping_f + batch["discount"] * q_targ)

        # ── GVF backups: y_k = c_k + z_k * reduce_k Q'_k(x', a') ────────────────
        # Same a' (target actor + smoothing) as the task backup, so every head sees
        # the identical next-state action and no extra forward is spent on it. The
        # reduction is min for a MAXIMIZED channel and max for a MINIMIZED one (the
        # head owns that choice — see networks._GvfTwin), which is what keeps a
        # collision head pessimistic rather than optimistic about safety.
        if channels:
            gvf_next = targ.gvf_reduced(x_next, a_next)  # [B, K]
            y_gvf = jax.lax.stop_gradient(
                jnp.stack(
                    [
                        batch[ch.cumulant_key] + batch[ch.continuation_key] * gvf_next[:, k]
                        for k, ch in enumerate(channels)
                    ],
                    axis=-1,
                )
            )  # [B, K]

        def critic_loss_fn(critic_state):
            m = nnx.merge(critic_gd, critic_state)
            q = m(x, batch["action"])  # [B, n]
            task_loss = jnp.mean(jnp.square(q - y[:, None]))
            if not channels:
                return task_loss, (q, jnp.zeros((0,)), task_loss)
            # UNWEIGHTED sum: these are separate heads with their own targets, and
            # downweighting them only slows their fit. One optimizer over the whole
            # critic state, so `state.critic_targ` and the existing polyak cover them.
            g = m.gvf(x, batch["action"])  # [B, K, 2]
            if cfg.gvf_freeze:
                # The ONLY gradient path into the twins (`y` is already stop_gradient'd,
                # so the bonus never was one). Cutting it leaves them exactly as the
                # resumed checkpoint holds them, while per_ch stays live for logging.
                g = jax.lax.stop_gradient(g)
            per_ch = jnp.mean(jnp.square(g - y_gvf[:, :, None]), axis=(0, 2))  # [K]
            gvf_term = 0.0 if cfg.gvf_freeze else jnp.sum(per_ch)
            return task_loss + gvf_term, (q, per_ch, task_loss)

        (_, (q, gvf_losses, critic_loss)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic)
        critic_updates, critic_opt_state = critic_opt.update(critic_grads, state.critic_opt, state.critic)
        critic_state = optax.apply_updates(state.critic, critic_updates)

        # Q on reward-bearing transitions isolates "critic can't fit the reward"
        # (q_reward stays low -> state-representation problem) from "reward fits but
        # propagates slowly down the bootstrap chain" (q_reward ~reward scale while
        # q_mean lags).
        #
        # Detected from the STORED OUTCOME FLAGS, not from `reward > 0.1`, so the
        # statistic stays the same if the reward column ever carries a dense term.
        # `success` is the EPISODE outcome and `discount == 0`
        # marks the terminal chunk, so their conjunction is precisely the rows the
        # builder gave the task reward to (train_rlt_libero._assemble_from_tables:
        # `success_inside = rec.success and t + n == L`).
        #
        # `.get` because the batch is also built by hand in td3_test, which carries no
        # `success` column; the fallback is the old threshold on an unshaped reward.
        succ = batch.get("success")
        rew_mask = (
            (batch["reward"] > 0.1).astype(jnp.float32)
            if succ is None
            else ((succ > 0.5) & (batch["discount"] <= 0.0)).astype(jnp.float32)
        )
        q_reward = jnp.sum(jnp.mean(q, axis=-1) * rew_mask) / jnp.maximum(jnp.sum(rew_mask), 1.0)

        metrics = {
            "critic_loss": critic_loss,
            # Raw input scales at the critic: the state block vs the action block.
            # ~15-20x norm imbalance (and 2065-vs-70 dims) lets the critic fit TD
            # targets as V(s) while ignoring a — see TD3Config.critic_input_norm.
            "in_x_norm": jnp.mean(jnp.linalg.norm(x, axis=-1)),
            "in_a_norm": jnp.mean(jnp.linalg.norm(batch["action"], axis=-1)),
            "q_mean": jnp.mean(q),
            # twin-0 only, on the SAME buffer actions as q_mean (which averages both
            # twins). q_pi_mean below is also twin-0-only, but on the actor's own
            # sampled action a_pi = mu + sigma*eps instead of the buffer's real action. Comparing this to q_mean
            # isolates twin asymmetry (if q_mean_twin0 << q_mean, twin 0 is just a more
            # pessimistic network) from an actor extrapolation gap (if q_mean_twin0 ~=
            # q_mean but q_pi_mean is still ~half of it, the gap is really about mu).
            "q_mean_twin0": jnp.mean(q[:, 0]),
            "q_reward": q_reward,
            # What the bootstrap actually reads at next states (min-twin target critic
            # at the target-actor action). If q_reward is high but q_next stays at the
            # base level, high values live at action coordinates the backup never
            # queries (e.g. the zero-padding bug) and the chain cannot ignite.
            "q_next": jnp.mean(q_targ),
            "target_q_mean": jnp.mean(y),
            "actor_loss": jnp.zeros(()),
            "bc_loss": jnp.zeros(()),
            "q_pi_mean": jnp.zeros(()),
            "q_pi_mean_det": jnp.zeros(()),
            "q_pi_ref": jnp.zeros(()),
            "q_pi_noref": jnp.zeros(()),
            "bc_ref": jnp.zeros(()),
            "bc_noref": jnp.zeros(()),
            "gq_norm": jnp.zeros(()),
            "gq_val_norm": jnp.zeros(()),
            "gbc_norm": jnp.zeros(()),
            # Zero (not cfg.bc_coef) so the host's per-actor-step averaging is right:
            # this is emitted only on do_actor steps, and the training loop divides the
            # actor metrics by n_do_actor. Seeding it with bc_coef made it accumulate on
            # critic-only steps too and log exactly 2x the real coefficient.
            "beta": jnp.zeros(()),
            "force_ratio": jnp.zeros(()),
            **_gvf_metric_zeros(channels, lookahead=lookahead),
        }
        if channels:
            # Emitted on EVERY update (they are critic-side / batch statistics), so
            # the host averages them over n_updates like critic_loss.
            metrics["critic_loss_total"] = critic_loss + jnp.sum(gvf_losses)
            for k, ch in enumerate(channels):
                metrics[f"gvf_{ch.name}_loss"] = gvf_losses[k]
                # Share of batch rows carrying a +1 label inside the chunk. If this
                # sits near 0 the head has nothing to fit and its action-gradient is
                # noise; near 1 and the channel has stopped discriminating.
                metrics[f"gvf_{ch.name}_event_frac"] = jnp.mean((batch[ch.cumulant_key] > 0).astype(jnp.float32))
            # Per-phase occupancy, so the hard gating above can be verified against
            # the data rather than assumed (a channel gated to a phase the batch never
            # visits contributes exactly nothing and would otherwise look like a null).
            for ph in sorted({p for ch in channels for p in ch.phases} | {0}):
                metrics[f"phase_frac_{ph}"] = jnp.mean((batch["phase"] == ph).astype(jnp.float32))
            # ── shaping scale (gvf_mode="critic"/"both"; exactly 0 under "actor") ──
            # The pair to watch. The bonus is DENSE and the task reward is SPARSE, so
            # a `lam` carried over from actor-mode silently turns Q_task into a GVF
            # predictor: at lam=0.3 the bonus is ~15x the batch-mean task reward per
            # step and ~lam/(1-gamma) = 30 on the value scale against a task Q of ~1.
            # gvf_bonus_frac is the share of the immediate signal that is shaping
            # rather than task reward; if it sits near 1 the critic has stopped being
            # a task critic, and every quantity tuned against the old Q scale
            # (bc_coef, force_ratio) has been rescaled out from under you.
            abs_bonus = jnp.mean(jnp.abs(bonus))
            metrics["gvf_bonus_mean"] = jnp.mean(bonus)
            metrics["gvf_bonus_abs_mean"] = abs_bonus
            metrics["gvf_bonus_frac"] = abs_bonus / (jnp.mean(jnp.abs(batch["reward"])) + abs_bonus + 1e-8)
            for k, ch in enumerate(channels):
                metrics[f"gvf_{ch.name}_bonus"] = jnp.mean(gvf_bonus_per_ch[:, k])
        if lookahead:
            # Emitted on EVERY update (they are batch/critic-side quantities), so the
            # host averages them over n_updates like critic_loss.
            #
            # la_F_abs_mean against the batch-mean |reward| is the look-ahead analogue
            # of gvf_bonus_frac: it says how much of the immediate regression signal is
            # advice. Unlike the "critic" bonus it is NOT a value-scale hazard -- F
            # telescopes, so it cannot accumulate to lam/(1-gamma) -- but it still sets
            # how hard the critic has to work to represent Q_task - Phi.
            metrics["la_phi_mean"] = jnp.mean(phi_cur)
            metrics["la_phi_abs_mean"] = jnp.mean(jnp.abs(phi_cur))
            metrics["la_F_mean"] = jnp.mean(shaping_f)
            metrics["la_F_abs_mean"] = jnp.mean(jnp.abs(shaping_f))
            # The critic here fits Q_Phi = Q_task - Phi, so its raw q_mean is NOT on the
            # same scale as every other mode's. Adding Phi back at the buffer action
            # recovers the task value the other arms' q_mean reports, which is the only
            # way to read the three arms' value curves on one axis.
            metrics["q_eff_mean"] = jnp.mean(q + phi_cur[:, None])

        if not do_actor:
            # The critic target must track on EVERY critic update: during the
            # critic-warmup phase there are no actor steps, and a frozen random
            # target would pin the bootstrap to min-of-random-nets noise (observed:
            # q_mean stuck at -0.24 with tiny critic_loss) so the sparse reward
            # never propagates.
            return state._replace(
                critic=critic_state,
                critic_opt=critic_opt_state,
                critic_targ=polyak(state.critic_targ, critic_state, cfg.tau),
            ), metrics

        # ── Actor update: L = E_{a~pi}[-Q_1(x, a)] + beta*||mu - a_ref||^2 ───────────────
        # actor_bc_only drops the Q term (BC warm-start): the plain-MLP actor first
        # learns to imitate the reference (with dropout, so the no-ref pathway forms)
        # before Q-gradients switch on with the warmed-up critic.
        ref_mask = jax.random.bernoulli(drop_rng, 1.0 - cfg.ref_dropout, (b, 1)).astype(jnp.float32)

        # ── composite steering weights ──────────────────────────────────────
        # The actor's share of `gvf_w` (computed once above the backup). Under
        # gvf_mode="critic" this is exactly zero, so the actor loss below is the RLT
        # baseline -Q_0 + beta*||mu - ref||^2 and every GVF term drops out of the
        # gradient identically -- the shaping has already been spent inside `y`.
        # Kept as an all-zeros array rather than None so the diagnostics at the end of
        # this function (d_steer, force_ratio_total) stay shape-correct in every mode.
        steer_w = gvf_w if steer_actor else (jnp.zeros_like(gvf_w) if channels else None)

        def actor_loss_fn(actor_state):
            mu = nnx.merge(actor_gd, actor_state)(x, batch["ref"], ref_mask)
            # Paper Eq. 5: beta * ||mu - a_ref||^2 — the squared Euclidean NORM, i.e.
            # a SUM over all C*d action dims (batch-meaned). A per-dim mean here
            # divides the BC gradient by C*d (~70), silently weakening the anchor by
            # that factor relative to the Q term: observed equilibrium was the actor
            # parked at per-dim deviation ~= sigma_explore (the edge of the explored
            # action shell, where twin-min extrapolation optimism is largest) with
            # success ~0.15 below the VLA reference.
            bc_row = jnp.sum(jnp.square(mu - batch["ref"]), axis=-1)
            bc = jnp.mean(bc_row)
            if actor_bc_only:
                # No GVF terms during the BC warm-start: the path is unchanged, and
                # steering on heads that predate any usable critic is meaningless.
                return cfg.bc_coef * bc, (
                    jnp.zeros(b),
                    bc_row,
                    jnp.zeros(b),
                    jnp.zeros((b, len(channels))),
                    jnp.zeros(b),
                    jnp.zeros(b),
                )
            # Paper Eq. 5 takes the EXPECTATION over a_1:C ~ pi_theta = N(mu, sigma^2 I),
            # not the value at the mean: L_pi = E_a[-Q(x, a)] + beta*||a - a_ref||^2.
            # Reparameterized single sample, a = mu + sigma*eps, sigma fixed. Two
            # reasons this is not cosmetic:
            #   (1) E_eps[-Q(mu + sigma*eps)] = -Q(mu) - (sigma^2/2)*tr(grad^2_a Q) + O(sigma^4),
            #       so the expectation carries a critic-CURVATURE penalty that keeps the
            #       actor off sharp, narrow Q ridges. It is the actor-side analogue of the
            #       target-policy smoothing already applied to `a_next` in the critic backup.
            #   (2) the critic is trained almost exclusively on buffer actions collected as
            #       mu+explore-noise (train_rlt_libero.py's collect()), so Q at the bare
            #       deterministic mu is an out-of-distribution query; sampling evaluates the
            #       critic where it was actually fit.
            # Noise is added around the actor mean, matching what the rollout perturbs.
            # actor_q_eval="mean" pins a_pi to mu, restoring the pre-Eq.5 -Q(x, mu)
            # exactly: no normal() is drawn, and every downstream use of a_pi (the Q term
            # and the GVF steering) collapses onto mu. diag_rng is still split upstream,
            # and jax.random.normal is stateless, so skipping the draw shifts no other stream.
            a_pi = (
                jnp.clip(
                    mu + cfg.sigma_explore * jax.random.normal(diag_rng, mu.shape),
                    -cfg.action_clip,
                    cfg.action_clip,
                )
                if cfg.actor_q_eval == "sample"
                else mu
            )
            q_pi = nnx.merge(critic_gd, critic_state)(x, a_pi)[:, 0]
            # The BC term deliberately stays on mu, NOT on a_pi: for fixed sigma,
            # E_eps[||mu + sigma*eps - ref||^2] = ||mu - ref||^2 + sigma^2*C*d, and the extra
            # term is constant in theta. So mu gives the identical BC gradient at strictly
            # lower variance.
            # Diagnostic only (no gradient path back into actor_loss): Q at the deterministic
            # mu, which is exactly where the noise-free eval policy deploys. If q_pi_mean_det
            # sits far below q_pi_mean, the critic scores the deployed action worse than the
            # explored shell it was trained on.
            # Under actor_q_eval="mean" a_pi IS mu, so this is q_pi — alias rather than
            # pay a second critic forward for an identical number.
            q_pi_det = nnx.merge(critic_gd, critic_state)(x, mu)[:, 0] if cfg.actor_q_eval == "sample" else q_pi
            # Composite steering (gvf_mode="actor"/"both" only): the task Q plus each
            # channel's conservatively reduced GVF, evaluated at the SAME sampled action
            # a_pi as q_pi, so the auxiliary terms and q_pi are read at one point.
            phi_pi = jnp.zeros(b)
            phi_pi_det = jnp.zeros(b)
            if steer_actor:
                gvf_pi = nnx.merge(critic_gd, critic_state).gvf_reduced(x, a_pi)  # [B, K]
                steer = q_pi + jnp.sum(steer_w * gvf_pi, axis=-1)
            elif lookahead:
                # L = -mean(Q_0(x, a_pi) + Phi(x, a_pi)) + BC. The critic converges to
                # Q_Phi = Q_task - Phi under the F-shaped backup above, so this sum is
                # the TASK Q -- the actor is optimizing the unshaped objective while
                # receiving d(Phi)/da at full strength on every step, which is exactly
                # what a sparse task critic cannot supply. That cancellation is why lam
                # may be set for gradient quality alone and never traded against bias.
                #
                # TARGET heads, deliberately, unlike "actor" mode: the identity is
                # "the Phi the critic subtracted is the Phi the actor adds back", and
                # the critic subtracted the target copy. Reading the online heads here
                # would add a within-update mismatch that no lam can cancel.
                # Gradients still flow through Phi to a_pi and so to the actor params;
                # `targ` itself is a constant pytree here (only state.actor is
                # differentiated), so nothing propagates into the heads.
                gvf_pi = targ.gvf_reduced(x, a_pi)  # [B, K]
                phi_pi = jnp.sum(gvf_w * gvf_pi, axis=-1)
                steer = q_pi + phi_pi
                # Diagnostic twin of q_pi_det: the task value of the DEPLOYED action.
                phi_pi_det = jax.lax.stop_gradient(phi_of(targ, x, mu, gvf_w))
            else:
                # gvf_mode="critic" (and the no-channel baseline): not merely a zero
                # weight but no term at all, so the actor's compute graph is the
                # baseline's exactly.
                gvf_pi = jnp.zeros((b, 0))
                steer = q_pi
            return -jnp.mean(steer) + cfg.bc_coef * bc, (
                q_pi,
                bc_row,
                q_pi_det,
                gvf_pi,
                phi_pi,
                phi_pi_det,
            )

        (actor_loss, (q_pi, bc_row, q_pi_det, _gvf_pi, phi_pi, phi_pi_det)), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor)
        actor_updates, actor_opt_state = actor_opt.update(actor_grads, state.actor_opt, state.actor)
        actor_state = optax.apply_updates(state.actor, actor_updates)

        new_state = TD3State(
            actor=actor_state,
            actor_targ=polyak(state.actor_targ, actor_state, cfg.tau),
            critic=critic_state,
            critic_targ=polyak(state.critic_targ, critic_state, cfg.tau),
            actor_opt=actor_opt_state,
            critic_opt=critic_opt_state,
        )
        # Split diagnostics by the dropout mask: the deployed policy is the ref
        # pathway (mask=1); the no-ref pathway (mask=0) exists only as the paper's
        # independent-pathway regularizer. q_pi_mean averaging both can hide a
        # critic that scores one pathway at ~0 (observed: q_pi_mean pinned at
        # exactly q_mean/2 — the no-ref actions were rated worthless).
        m1 = ref_mask[:, 0]
        n1 = jnp.maximum(jnp.sum(m1), 1.0)
        n0 = jnp.maximum(jnp.sum(1.0 - m1), 1.0)
        # Per-row action-gradient norms of the two actor-loss terms, evaluated at mu.
        # If gbc_norm >> gq_norm throughout training, the BC anchor is the binding
        # constraint and beta is the knob; if gq_norm >> gbc_norm, the critic drives
        # and instability/OOD-chasing is the risk. Q_i depends only on a_i, so the
        # grad of the batch sum gives per-row dQ/da.
        mu_final = nnx.merge(actor_gd, actor_state)(x, batch["ref"], ref_mask)
        dq_da = jax.grad(lambda a: jnp.sum(nnx.merge(critic_gd, critic_state)(x, a)[:, 0]))(mu_final)
        dbc_da = 2.0 * cfg.bc_coef * (mu_final - batch["ref"])
        gq_rows = jnp.linalg.norm(dq_da, axis=-1)
        # gq on VALUE-carrying rows only (target y above the failure floor). The
        # batch-mean gq_norm is dominated by failure-episode rows where target ~= 0
        # and a zero action-gradient is CORRECT; this isolates whether the critic
        # has action-opinions where there is actually value to steer toward.
        val_mask = (y > 0.2).astype(jnp.float32)
        # BC-vs-Q force ratio gbc/gq: above ~1 the anchor dominates and the critic is
        # a passenger; far below 1 the actor chases critic extrapolation error.
        gbc_mean = jnp.mean(jnp.linalg.norm(dbc_da, axis=-1))
        force_ratio = gbc_mean / (jnp.mean(gq_rows) + 1e-8)
        metrics.update(
            beta=jnp.asarray(cfg.bc_coef, jnp.float32),
            force_ratio=force_ratio,
        )
        metrics.update(
            gq_norm=jnp.mean(gq_rows),
            gq_val_norm=jnp.sum(gq_rows * val_mask) / jnp.maximum(jnp.sum(val_mask), 1.0),
            gbc_norm=jnp.mean(jnp.linalg.norm(dbc_da, axis=-1)),
        )
        if channels:
            # All evaluated on the POST-update critic at mu_final, the same way
            # gq_norm/gbc_norm are, so the ratios below compare like with like.
            post_critic = nnx.merge(critic_gd, critic_state)
            gvf_buf = post_critic.gvf_reduced(x, batch["action"])  # on buffer actions
            gvf_at_mu = post_critic.gvf_reduced(x, mu_final)  # at the actor's own action
            n_val = jnp.maximum(jnp.sum(val_mask), 1.0)

            def _mmean(v: jax.Array) -> jax.Array:
                """Mean over value-carrying rows (y > 0.2), as gq_val_norm does. The
                batch mean is dominated by failure rows where dQ/da is ~0 and an
                undefined cosine; this is the sub-population the alignment condition
                is actually about."""
                return jnp.sum(v * val_mask) / n_val

            def _cos(u: jax.Array, v: jax.Array) -> jax.Array:
                return jnp.sum(u * v, axis=-1) / (jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1) + 1e-12)

            # d(total steering)/da: dQ_task/da plus every ACTIVE (phase-gated,
            # warm-up-gated) GVF term. This is the force the actor actually felt. Under
            # "lookahead" that is Q_0 + Phi with the TARGET heads -- the same expression
            # the actor loss maximizes -- not steer_w over the online heads.
            if lookahead:
                d_steer = jax.grad(lambda a: jnp.sum(post_critic(x, a)[:, 0] + phi_of(targ, x, a, gvf_w)))(mu_final)
            else:
                d_steer = jax.grad(
                    lambda a: jnp.sum(
                        post_critic(x, a)[:, 0] + jnp.sum(steer_w * post_critic.gvf_reduced(x, a), axis=-1)
                    )
                )(mu_final)
            gsteer_norm = jnp.mean(jnp.linalg.norm(d_steer, axis=-1))
            for k, ch in enumerate(channels):
                gk_vec = jax.grad(lambda a, _k=k: jnp.sum(post_critic.gvf_reduced(x, a)[:, _k]))(mu_final)
                g_k = jnp.linalg.norm(gk_vec, axis=-1)
                metrics[f"gvf_{ch.name}_mean"] = jnp.mean(gvf_buf[:, k])
                # The one that should MOVE if the mechanism works: the head's
                # probability at the actor's own action, versus at the buffer's.
                metrics[f"gvf_{ch.name}_pi"] = jnp.mean(gvf_at_mu[:, k])
                metrics[f"g_gvf_{ch.name}_norm"] = jnp.mean(g_k)
                # Against the CONFIGURED lam (not the warm-up-gated one), so the ratio
                # is readable as a preview during warm-up instead of collapsing to inf.
                metrics[f"force_ratio_{ch.name}"] = gbc_mean / (ch.lam * jnp.mean(g_k) + 1e-8)
                # ── alignment probe, emitted in every mode ──────────────────
                # Whether channel k's advice points the same way as the task critic,
                # and how much of it the task gradient can absorb. Moving along
                # d(Q_0 + lam*g_k)/da still increases Q_0 to first order iff
                #     <g_Q + lam*g_k, g_Q> > 0  <=>  lam > -||g_Q||^2 / <g_k, g_Q>,
                # and the symmetric statement for the advice not being overwhelmed by
                # its own curvature gives the familiar two-sided form: the step stays
                # ascent in Q_0 for lam up to
                #     lam_bound = 2 <g_k, g_Q> / ||g_k||^2.
                # Negative for an ANTI-aligned channel, i.e. no positive lam is safe
                # there. Read it in the lam=0 arm, where Q_0 is an unshaped task critic
                # and the bound is therefore the honest one; in a shaped run Q_0 has
                # already absorbed the advice and the number drifts toward 0.
                #
                # Deliberately on the ONLINE post-update heads in every mode (like
                # g_gvf_k_norm just above), not on `targ`: this is a read-only property
                # of the head function, and one convention keeps the number comparable
                # across the lam=0, critic and lookahead arms. The polyak lag between
                # the two copies is tau=0.005 per update.
                cos_k = _cos(gk_vec, dq_da)
                bound_k = 2.0 * jnp.sum(gk_vec * dq_da, axis=-1) / (jnp.sum(jnp.square(gk_vec), axis=-1) + 1e-12)
                metrics[f"la_cos_{ch.name}"] = jnp.mean(cos_k)
                metrics[f"la_cos_{ch.name}_val"] = _mmean(cos_k)
                metrics[f"la_lam_bound_{ch.name}"] = jnp.mean(bound_k)
                metrics[f"la_lam_bound_{ch.name}_val"] = _mmean(bound_k)
            # The aggregate to compare against force_ratio's known-good 0.45-0.8 band:
            # adding GVF terms raises the total steering force, and it is the total
            # that decides whether the actor stays on the VLA manifold. Under
            # gvf_mode="critic" steer_w is all zeros, so this collapses to force_ratio
            # by construction -- the shaping shows up in q_mean / gq_norm instead,
            # because there it IS the task critic that moved. Under "lookahead" it is
            # gbc / ||d(Q_0 + Phi)/da||, i.e. measured against the composite the actor
            # really optimizes, so the band stays readable across all three arms.
            metrics["force_ratio_total"] = gbc_mean / (gsteer_norm + 1e-8)
        if lookahead:
            # ── look-ahead decomposition of the actor's force ───────────────
            # Q_0 here is the RESIDUAL critic Q_Phi, so la_g_resid_norm is not a task
            # action-gradient -- it is whatever the task signal has left after Phi was
            # subtracted. la_cos_phi_resid is the quantity to watch: at convergence the
            # advice and the residual are the two halves of one task gradient, so a
            # strongly NEGATIVE cosine means Phi is fighting the critic rather than
            # carrying it, and lam is too large for this channel set.
            g_phi = jax.grad(lambda a: jnp.sum(phi_of(targ, x, a, gvf_w)))(mu_final)
            gphi_rows = jnp.linalg.norm(g_phi, axis=-1)
            cos_pr = _cos(g_phi, dq_da)
            metrics.update(
                la_g_phi_norm=jnp.mean(gphi_rows),
                la_g_phi_norm_val=_mmean(gphi_rows),
                la_g_resid_norm=jnp.mean(gq_rows),
                la_g_resid_norm_val=_mmean(gq_rows),
                la_cos_phi_resid=jnp.mean(cos_pr),
                la_cos_phi_resid_val=_mmean(cos_pr),
                # Task value of what the policy proposes, on the other arms' scale:
                # Q_Phi + Phi == Q_task at the fixed point, so these are the direct
                # counterparts of q_pi_mean / q_pi_mean_det elsewhere.
                q_eff_pi_mean=jnp.mean(q_pi + phi_pi),
                q_eff_pi_mean_val=_mmean(q_pi + phi_pi),
                q_eff_pi_mean_det=jnp.mean(q_pi_det + phi_pi_det),
                q_eff_pi_mean_det_val=_mmean(q_pi_det + phi_pi_det),
            )
        metrics.update(
            actor_loss=actor_loss,
            bc_loss=jnp.mean(bc_row),
            q_pi_mean=jnp.mean(q_pi),
            q_pi_mean_det=jnp.mean(q_pi_det),
            q_pi_ref=jnp.sum(q_pi * m1) / n1,
            q_pi_noref=jnp.sum(q_pi * (1.0 - m1)) / n0,
            bc_ref=jnp.sum(bc_row * m1) / n1,
            bc_noref=jnp.sum(bc_row * (1.0 - m1)) / n0,
        )
        return new_state, metrics

    return update_fn


def build_actor_apply(actor_gd):
    """Jitted deployed policy: (state, x, ref) -> the actor mean, with the reference
    always provided (the rollout/eval pathway)."""

    @jax.jit
    def actor_apply(state: TD3State, x: jax.Array, ref: jax.Array) -> jax.Array:
        ones = jnp.ones((x.shape[0], 1), jnp.float32)
        return nnx.merge(actor_gd, state.actor)(x, ref, ones)

    return actor_apply


def build_critic_apply(critic_gd):
    """Jitted ensemble Q: (critic_state, x, a) -> [B, n_critics].

    Read-only companion to build_actor_apply, for monitoring the value the critic
    assigns to what the actor is actually about to execute. The update path calls
    the module directly and does not use this.
    """

    @jax.jit
    def critic_apply(critic_state, x: jax.Array, a: jax.Array) -> jax.Array:
        return nnx.merge(critic_gd, critic_state)(x, a)

    return critic_apply
