"""Frozen-VLA rollout forward for RLT: one prefix pass -> (z_rl, reference chunk).

Mirrors Pi0.sample_actions' ODE sampler, but the prefix forward goes through
_prefix_cache (which returns prefix_out instead of discarding it) so the same
pass feeds both the RLTokenizer encoder and the denoise loop's KV cache.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask

_EXPECTED_IMAGE_ORDER = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _prefix_cache(model: Pi0, observation: _model.Observation):
    """Prefix forward -> (prefix_out, prefix_mask, kv_cache)."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return prefix_out, prefix_mask, kv_cache


def _velocity(model: Pi0, observation, x_t, time, prefix_mask, kv_cache):
    """One action-expert forward at (x_t, time) against the cached prefix -> v_t."""
    batch_size = x_t.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, jnp.broadcast_to(time, (batch_size,))
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def build_vla_forward(graphdef, model_config, *, num_steps: int = 10):
    """Jitted (pi0_state, rng, obs) -> (z_rl [B, zrl_dim] f32, a_ref [B, H, action_dim]).

    The reference chunk is a deterministic-ODE sample (the deployment sampler);
    stochasticity comes only from the initial noise drawn from `rng`.
    """
    num_image_tokens = model_config.rlt_num_input_cameras * 256

    @jax.jit
    def vla_forward(pi0_state, rng, obs: _model.Observation):
        # The image-token slice below assumes the canonical camera ordering.
        assert list(obs.images) == list(_EXPECTED_IMAGE_ORDER), list(obs.images)
        m = nnx.merge(graphdef, pi0_state)
        obs = _model.preprocess_observation(None, obs, train=False)

        prefix_out, prefix_mask, kv_cache = _prefix_cache(m, obs)
        z_rl = m.rlt.encode(prefix_out[:, :num_image_tokens].astype(jnp.float32))

        batch_size = obs.state.shape[0]
        dt = -1.0 / num_steps
        noise = jax.random.normal(rng, (batch_size, m.action_horizon, m.action_dim))

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        def step(carry):
            x_t, time = carry
            v_t = _velocity(m, obs, x_t, time, prefix_mask, kv_cache)
            return x_t + dt * v_t, time + dt

        (x_0, _) = jax.lax.while_loop(cond, step, (noise, 1.0))
        return z_rl, x_0

    return vla_forward
