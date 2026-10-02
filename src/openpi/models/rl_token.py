"""RL Token (RLT) — compact readout representation of a frozen VLA.

Implementation of the representation-learning stage of "RL Token: Bootstrapping
Online RL with Vision-Language-Action Models" (arXiv 2604.23073). A learned
embedding e_rl is appended to the VLA's final-layer prefix token embeddings
z_1:M; a small bidirectional encoder transformer g_phi produces the RL token
z_rl at the special-token position (Eq. 1). A causal decoder d_phi with linear
head h_phi is trained to autoregressively reconstruct the stop-gradient
embeddings from [z_rl, z_bar_1:i-1] (teacher forcing, Eq. 2):

    L_ro = E_D[ sum_i || h_phi(d_phi([z_rl, z_bar_1:i-1]))_i - z_bar_i ||^2 ]

The paper drops language-token embeddings (fixed per-task instruction), so the
input here is the image-token slice of `prefix_out` — the caller selects it.

Trained on demo data with the VLA frozen (see the `pi05_rlt_only_libero`
config), then frozen itself; online RL (src/openpi/training/rlt/) treats
`encode()` as a fixed feature extractor.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class TransformerBlock(nnx.Module):
    """Pre-LN transformer block: LN -> MHA -> residual; LN -> MLP(4x, gelu) -> residual."""

    def __init__(self, width: int, num_heads: int, *, rngs: nnx.Rngs):
        self.attn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=width,
            decode=False,
            rngs=rngs,
        )
        self.mlp_norm = nnx.LayerNorm(width, rngs=rngs)
        self.mlp_in = nnx.Linear(width, 4 * width, rngs=rngs)
        self.mlp_out = nnx.Linear(4 * width, width, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # mask: broadcastable to [b, num_heads, s, s]; None = full bidirectional.
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, mask=mask, deterministic=True, decode=False)
        h = self.mlp_norm(x)
        return x + self.mlp_out(nnx.gelu(self.mlp_in(h)))


class RLTokenizer(nnx.Module):
    """Encoder-decoder that compresses VLA prefix embeddings into one RL token.

    encode():           z_1:M [b, M, feature_dim] -> z_rl [b, zrl_dim]
    compute_rlt_loss(): the autoregressive reconstruction loss L_ro (Eq. 2).
    """

    def __init__(
        self,
        *,
        feature_dim: int = 2048,  # VLM prefix_out width (PaliGemma)
        zrl_dim: int = 2048,  # paper: 1x2048
        # Residual-stream width of g_phi/d_phi. The paper keeps this at the VLA
        # embedding width (2048), compressing only in sequence length (M -> 1).
        # width < zrl_dim makes zrl_proj an up-projection: z_rl then lives on a
        # width-dimensional subspace of R^zrl_dim, and in the reconstruction path
        # dec_in_proj immediately maps it back down, so the extra dimensions are
        # invisible to L_ro and only inflate the stage-2 RL state.
        width: int = 2048,
        enc_depth: int = 3,
        dec_depth: int = 3,
        num_heads: int = 8,
        num_tokens: int = 512,  # M (e.g. 2 cameras x 256 SigLIP tokens)
        normalize_targets: bool = True,
        rngs: nnx.Rngs,
    ):
        self.num_tokens = num_tokens
        self.normalize_targets = normalize_targets
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth

        init = nnx.initializers.normal(stddev=0.02)
        # NOTE: blocks live in an nnx.Dict with STRING keys, not a Python list —
        # list entries get integer path keys, which break the sep="/" flatten in
        # weight_loaders._merge_params (and orbax key roundtrips) when this module
        # is embedded in Pi0.
        # Encoder.
        self.in_proj = nnx.Linear(feature_dim, width, rngs=rngs)
        self.e_rl = nnx.Param(init(rngs.params(), (1, 1, width)))
        self.enc_pos = nnx.Param(init(rngs.params(), (1, num_tokens + 1, width)))
        self.enc_blocks = nnx.Dict(
            {f"block_{i}": TransformerBlock(width, num_heads, rngs=rngs) for i in range(enc_depth)}
        )
        self.enc_norm = nnx.LayerNorm(width, rngs=rngs)
        self.zrl_proj = nnx.Linear(width, zrl_dim, rngs=rngs)
        # Decoder (causal, teacher-forced).
        self.dec_in_proj = nnx.Linear(zrl_dim, width, rngs=rngs)
        self.dec_tok_proj = nnx.Linear(feature_dim, width, rngs=rngs)
        self.dec_pos = nnx.Param(init(rngs.params(), (1, num_tokens, width)))
        self.dec_blocks = nnx.Dict(
            {f"block_{i}": TransformerBlock(width, num_heads, rngs=rngs) for i in range(dec_depth)}
        )
        self.dec_norm = nnx.LayerNorm(width, rngs=rngs)
        self.head = nnx.Linear(width, feature_dim, rngs=rngs)

    @at.typecheck
    def _prep_targets(self, tokens: at.Float[at.Array, "b m f"]) -> at.Float[at.Array, "b m f"]:
        """f32 + stop_gradient (+ optional per-token RMS normalization) of z_1:M."""
        z_bar = jax.lax.stop_gradient(tokens.astype(jnp.float32))
        if self.normalize_targets:
            rms = jnp.sqrt(jnp.mean(jnp.square(z_bar), axis=-1, keepdims=True) + 1e-6)
            z_bar = z_bar / rms
        return z_bar

    @at.typecheck
    def encode(self, tokens: at.Float[at.Array, "b m f"]) -> at.Float[at.Array, "b z"]:
        """z_rl = g_phi([z_1:M, e_rl])_{M+1} (Eq. 1)."""
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {tokens.shape[1]}")
        z_bar = self._prep_targets(tokens)
        b = z_bar.shape[0]
        e_rl = jnp.broadcast_to(self.e_rl.value, (b, 1, self.e_rl.value.shape[-1]))
        x = jnp.concatenate([self.in_proj(z_bar), e_rl], axis=1) + self.enc_pos.value
        for i in range(self.enc_depth):
            x = self.enc_blocks[f"block_{i}"](x)
        return self.zrl_proj(self.enc_norm(x)[:, -1])

    @at.typecheck
    def compute_rlt_loss(self, tokens: at.Float[at.Array, "b m f"]) -> at.Float[at.Array, ""]:
        """Autoregressive reconstruction loss L_ro (Eq. 2), teacher-forced."""
        z_bar = self._prep_targets(tokens)
        z_rl = self.encode(tokens)
        # Decoder inputs: [z_rl, z_bar_1, ..., z_bar_{M-1}]; targets: z_bar_1:M.
        dec_in = jnp.concatenate([self.dec_in_proj(z_rl)[:, None, :], self.dec_tok_proj(z_bar[:, :-1])], axis=1)
        x = dec_in + self.dec_pos.value
        s = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, None, :, :]
        for i in range(self.dec_depth):
            x = self.dec_blocks[f"block_{i}"](x, mask=causal_mask)
        pred = self.head(self.dec_norm(x))
        return jnp.mean(jnp.square(pred - z_bar))
