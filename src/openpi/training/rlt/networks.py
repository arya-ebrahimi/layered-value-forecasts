"""Lightweight RLT actor/critic MLPs (RL Token paper, arXiv 2604.23073).

Both operate on the RL state x = concat(z_rl, proprio) and flattened action
chunks a in normalized model action space ([C * action_env_dim]). Paper sizes:
2-layer hidden 256 MLPs (3-layer hidden 512 for the hardest task).
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class RltActor(nnx.Module):
    """Paper-faithful actor mean mu_theta(x, a_ref) (RLT Eq. 4): a plain MLP.

    The (dropout-masked) reference chunk is an INPUT only — no residual base: the
    reference couples to the output solely through input conditioning, the BC term
    beta*||mu - a_ref||^2, and the 50% reference dropout, exactly as in the paper.
    The output is tanh-squashed to the normalized action space (TD3-canonical; the
    paper does not specify squashing). With the zero-initialized final layer, mu = 0
    at init — the driver therefore executes a_ref + noise and trains the actor
    BC-only until the critic warm-up ends (the paper's immediate async training
    gives it the same imitate-the-reference start for free). Exploration noise and
    clipping are applied by the caller.
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        dims = (obs_dim + chunk_action_dim, *hidden)
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        self.out = nnx.Linear(
            hidden[-1],
            chunk_action_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, ref: jax.Array, ref_mask: jax.Array) -> jax.Array:
        # x [B, obs_dim], ref [B, chunk_action_dim], ref_mask [B, 1] in {0, 1}.
        h = jnp.concatenate([x, ref * ref_mask], axis=-1)
        for layer in self.layers:
            h = nnx.relu(layer(h))
        return jnp.tanh(self.out(h))


class _QMlp(nnx.Module):
    """One Q(x, a) head.

    dual_encoder=False: x and a are concatenated raw and fed to a single first
    Linear. With standard fan-in init this weights each input UNIT equally, so a
    block's contribution to every preactivation scales with its DIMENSION COUNT,
    not its information content: x ([B, ~2065]) vs a ([B, ~70]) means x supplies
    ~2065/2135 ~= 97% of each preactivation's variance regardless of either
    block's per-dim scale. This is why per-block LayerNorm on the raw blocks
    alone (equalizing per-dim RMS) does NOT fix the V(s)-shortcut: the shared
    first layer still drowns the action in fan-in count.

    dual_encoder=True: x and a are each linearly projected to hidden[0] and
    LayerNormed BEFORE the concat, so the action gets an architecturally EQUAL
    (hidden[0] vs hidden[0]) share of the fused representation independent of
    obs_dim >> chunk_action_dim. Each critic in the ensemble gets its OWN x/a
    encoders (not shared across twins), so the twins stay independent function
    approximators (required for the min-Q pessimism trick).
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_action_dim: int,
        hidden: tuple[int, ...],
        *,
        dual_encoder: bool = False,
        rngs: nnx.Rngs,
    ):
        self.dual_encoder = dual_encoder
        if dual_encoder:
            self.x_enc = nnx.Linear(obs_dim, hidden[0], rngs=rngs)
            self.a_enc = nnx.Linear(chunk_action_dim, hidden[0], rngs=rngs)
            self.x_enc_norm = nnx.LayerNorm(hidden[0], use_scale=False, use_bias=False, rngs=rngs)
            self.a_enc_norm = nnx.LayerNorm(hidden[0], use_scale=False, use_bias=False, rngs=rngs)
            dims = (2 * hidden[0], *hidden)
        else:
            dims = (obs_dim + chunk_action_dim, *hidden)
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        self.out = nnx.Linear(hidden[-1], 1, rngs=rngs)

    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        if self.dual_encoder:
            h = jnp.concatenate([self.x_enc_norm(self.x_enc(x)), self.a_enc_norm(self.a_enc(a))], axis=-1)
        else:
            h = jnp.concatenate([x, a], axis=-1)
        for layer in self.layers:
            h = nnx.relu(layer(h))
        return self.out(h)[:, 0]


class _GvfTwin(nnx.Module):
    """One auxiliary GVF channel's head: a TWIN, like the task critic.

    The actor takes gradients through these heads (td3.py's composite steering
    term), so they need the same overestimation control TD3 gives Q_task -- a
    single head's extrapolation error would be steered into directly.

    The conservative direction depends on the channel's `sign` and is stored
    HERE, on the head, rather than being re-derived at each call site: the
    target reduction and the actor's read must never disagree.

      sign > 0 (the actor MAXIMIZES this GVF, e.g. grasp)  -> reduce with min
      sign < 0 (the actor MINIMIZES it, e.g. collision)    -> reduce with max

    Backwards, a min on a minimized channel makes the collision head OPTIMISTIC
    about safety -- exactly the wrong bias, and one that would look like a
    working run right up until the arm hit the holder.

    Output is a RAW LINEAR head, not a sigmoid. Targets are bounded in [0, 1] by
    construction (cumulant in {0, gamma**k}, continuation in [0, 1]), so the
    range comes for free, whereas a sigmoid saturates and kills d(gvf)/da -- the
    whole reason these heads exist.
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_action_dim: int,
        hidden: tuple[int, ...],
        *,
        dual_encoder: bool,
        sign: float,
        rngs: nnx.Rngs,
    ):
        self.sign = float(sign)
        self.heads = [_QMlp(obs_dim, chunk_action_dim, hidden, dual_encoder=dual_encoder, rngs=rngs) for _ in range(2)]

    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        return jnp.stack([h(x, a) for h in self.heads], axis=-1)  # [B, 2]

    def reduce(self, q: jax.Array) -> jax.Array:
        """Conservative reduction over the twin, per this head's own sign."""
        return jnp.min(q, axis=-1) if self.sign > 0 else jnp.max(q, axis=-1)


class RltCritic(nnx.Module):
    """Twin (ensemble) chunk-level Q functions: Q_i(x, a_1:C) -> [B, n_critics],
    plus optional auxiliary GVF twin heads sharing the same module (and therefore
    the same `state.critic_targ` / `polyak` / optimizer, with no change to either).

    input_norm=True switches every critic head to the dual-encoder architecture
    (see _QMlp) that gives the action an architecturally equal footing against
    the much higher-dimensional state block, fixing the V(s)-shortcut where the
    critic fits TD targets from state alone and the action gradient (dQ/da) decays
    to ~0 (observed: gq_norm ~20x smaller than gbc_norm and falling). It applies to
    the GVF heads identically -- they are read for their action-gradient, so the
    shortcut would defeat them even harder than it defeats Q_task.

    gvf_signs=() (the default) builds no GVF heads and consumes no extra rngs, so
    the parameter tree and the init draw are bit-identical to the task-only critic.
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        n_critics: int = 2,
        *,
        input_norm: bool = False,
        gvf_signs: tuple[float, ...] = (),
        rngs: nnx.Rngs,
    ):
        self.input_norm = input_norm
        self.critics = [
            _QMlp(obs_dim, chunk_action_dim, hidden, dual_encoder=input_norm, rngs=rngs) for _ in range(n_critics)
        ]
        # Built AFTER the task critics so the task params' rng draw is unchanged.
        self.gvf_twins = [
            _GvfTwin(obs_dim, chunk_action_dim, hidden, dual_encoder=input_norm, sign=s, rngs=rngs) for s in gvf_signs
        ]

    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        return jnp.stack([q(x, a) for q in self.critics], axis=-1)

    # `task` is an alias for __call__ so the composite actor loss in td3.py reads
    # symmetrically (critic.task(...) / critic.gvf(...)); every existing call site
    # keeps using the module directly.
    def task(self, x: jax.Array, a: jax.Array) -> jax.Array:
        return self(x, a)

    def gvf(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Per-channel twin outputs -> [B, n_gvf, 2]."""
        if not self.gvf_twins:
            return jnp.zeros((x.shape[0], 0, 2), jnp.float32)
        return jnp.stack([t(x, a) for t in self.gvf_twins], axis=1)

    def gvf_reduced(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Conservatively reduced per-channel GVF values -> [B, n_gvf]. This is what
        the actor loss and the TD target both read; the reduction direction is the
        head's own (min when maximized, max when minimized)."""
        if not self.gvf_twins:
            return jnp.zeros((x.shape[0], 0), jnp.float32)
        return jnp.stack([t.reduce(t(x, a)) for t in self.gvf_twins], axis=-1)

    def reduce_gvf(self, q: jax.Array) -> jax.Array:
        """Reduce an already-computed [B, n_gvf, 2] stack -> [B, n_gvf], so a caller
        that needs both the raw twins (for the TD loss) and their reduction (for the
        target) pays for only one forward pass."""
        if not self.gvf_twins:
            return jnp.zeros((q.shape[0], 0), jnp.float32)
        return jnp.stack([t.reduce(q[:, k, :]) for k, t in enumerate(self.gvf_twins)], axis=-1)
