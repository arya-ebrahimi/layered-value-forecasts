#!/usr/bin/env python3
"""
Convert a PyTorch / safetensors checkpoint back to OpenPi JAX/Orbax format.

This is the INVERSE of convert_openpi_jax_to_python.py.

Usage:
    # From safetensors (output of convert_openpi_jax_to_python.py):
    python convert_pt_to_openpi_jax.py \
        --input_path  /path/to/pytorch_checkpoint \
        --output_path /path/to/orbax_output \
        --config_name pi05_droid

    # From raw .pt state dict (e.g. full_weights.pt from RLinf):
    python convert_pt_to_openpi_jax.py \
        --input_path  /path/to/full_weights.pt \
        --output_path /path/to/orbax_output \
        --config_name pi05_droid \
        --from_pt

Example:
    python convert_pt_to_openpi_jax.py \
        --input_path  /path/to/global_step_250/actor/model_state_dict/full_weights.pt \
        --output_path /path/to/global_step_250/actor/orbax_checkpoint \
        --config_name pi05_droid \
        --from_pt
"""

import os
import pathlib
import shutil
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.gemma
import openpi.models.pi0_config
import openpi.training.config as _config
import orbax.checkpoint as ocp
import safetensors.torch
from safetensors import safe_open
import torch
import tyro
from flax.nnx import traversals
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a float32 numpy array."""
    return tensor.detach().float().cpu().numpy()


def load_state_dict(input_path: str, from_pt: bool) -> dict[str, torch.Tensor]:
    """Load weights from either a .pt file or a safetensors directory."""
    if from_pt:
        print(f"Loading PyTorch state dict from {input_path}")
        sd = torch.load(input_path, map_location="cpu")
        # Unwrap common wrappers
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    else:
        st_path = os.path.join(input_path, "model.safetensors")
        print(f"Loading safetensors from {st_path}")
        # sd = safetensors.torch.load_file(st_path)
        index_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

            sd = {}
            for weight_file in set(index["weight_map"].values()):
                path = os.path.join(input_path, weight_file)
                with safe_open(path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        sd[k] = f.get_tensor(k)
        else:
            sd = safetensors.torch.load_file(os.path.join(input_path, "model.safetensors"))

    # Safetensors deduplicates tied weights — restore the embedding from lm_head
    embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if embed_key not in sd and lm_head_key in sd:
        print(f"Restoring tied weight: {embed_key} <- {lm_head_key}")
        sd[embed_key] = sd[lm_head_key]

    return sd


# ---------------------------------------------------------------------------
# Reverse PaliGemma conversion
# ---------------------------------------------------------------------------

def unslice_paligemma_state_dict(
    state_dict: dict[str, torch.Tensor],
    config,
    pi05: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Inverse of slice_paligemma_state_dict.
    Returns (paligemma_jax_params, expert_jax_params).
    """
    jax_dict: dict[str, np.ndarray] = {}

    # ---- patch embeddings ----
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    # forward: .transpose(3, 2, 0, 1)  →  inverse: .transpose(2, 3, 1, 0)
    jax_dict["img/embedding/kernel"] = to_np(state_dict.pop(pt_key).permute(2, 3, 1, 0))

    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    jax_dict["img/embedding/bias"] = to_np(state_dict.pop(pt_key))

    # ---- positional embeddings ----
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    # forward: .reshape(-1, hidden_size)  →  inverse: add back leading dim of 1
    jax_dict["img/pos_embedding"] = to_np(state_dict.pop(pt_key)).reshape(
        1, -1, config.vision_config.hidden_size
    )

    # ---- vision encoder layers ----
    n_vis = config.vision_config.num_hidden_layers
    hidden = config.vision_config.hidden_size

    ln0_scale  = np.zeros((n_vis, hidden), dtype=np.float32)
    ln0_bias   = np.zeros((n_vis, hidden), dtype=np.float32)
    ln1_scale  = np.zeros((n_vis, hidden), dtype=np.float32)
    ln1_bias   = np.zeros((n_vis, hidden), dtype=np.float32)

    int_size   = config.vision_config.intermediate_size
    mlp_d0_k   = np.zeros((n_vis, hidden, int_size), dtype=np.float32)
    mlp_d0_b   = np.zeros((n_vis, int_size),          dtype=np.float32)
    mlp_d1_k   = np.zeros((n_vis, int_size, hidden),  dtype=np.float32)
    mlp_d1_b   = np.zeros((n_vis, hidden),             dtype=np.float32)

    n_heads    = config.vision_config.num_attention_heads
    head_dim   = hidden // n_heads
    # JAX shape: (layers, hidden, heads, head_dim)
    attn_shape = (n_vis, hidden, n_heads, head_dim)

    key_k   = np.zeros(attn_shape, dtype=np.float32)
    key_b   = np.zeros((n_vis, n_heads, head_dim), dtype=np.float32)
    val_k   = np.zeros(attn_shape, dtype=np.float32)
    val_b   = np.zeros((n_vis, n_heads, head_dim), dtype=np.float32)
    qry_k   = np.zeros(attn_shape, dtype=np.float32)
    qry_b   = np.zeros((n_vis, n_heads, head_dim), dtype=np.float32)
    out_k   = np.zeros((n_vis, n_heads, head_dim, hidden), dtype=np.float32)
    out_b   = np.zeros((n_vis, hidden), dtype=np.float32)

    pfx = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers"
    for i in range(n_vis):
        # forward: scale[i].transpose()  →  inverse: .transpose() (self-inverse for 1-D → effectively no-op, but kept for explicitness)
        ln0_scale[i] = to_np(state_dict.pop(f"{pfx}.{i}.layer_norm1.weight")).reshape(hidden)
        ln0_bias[i]  = to_np(state_dict.pop(f"{pfx}.{i}.layer_norm1.bias"))
        ln1_scale[i] = to_np(state_dict.pop(f"{pfx}.{i}.layer_norm2.weight")).reshape(hidden)
        ln1_bias[i]  = to_np(state_dict.pop(f"{pfx}.{i}.layer_norm2.bias"))

        # forward: kernel[i].transpose()  →  inverse: .T
        mlp_d0_k[i] = to_np(state_dict.pop(f"{pfx}.{i}.mlp.fc1.weight")).T
        mlp_d0_b[i] = to_np(state_dict.pop(f"{pfx}.{i}.mlp.fc1.bias"))
        mlp_d1_k[i] = to_np(state_dict.pop(f"{pfx}.{i}.mlp.fc2.weight")).T
        mlp_d1_b[i] = to_np(state_dict.pop(f"{pfx}.{i}.mlp.fc2.bias"))

        # forward: kernel[i].reshape(-1, hidden).transpose() → PT shape: (n_heads*head_dim, hidden)
        # inverse: .reshape(n_heads, head_dim, hidden).transpose(2,0,1) → JAX shape: (hidden, n_heads, head_dim)
        key_k[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.k_proj.weight")).reshape(n_heads, head_dim, hidden).transpose(2, 0, 1)
        key_b[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.k_proj.bias")).reshape(n_heads, head_dim)
        val_k[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.v_proj.weight")).reshape(n_heads, head_dim, hidden).transpose(2, 0, 1)
        val_b[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.v_proj.bias")).reshape(n_heads, head_dim)
        qry_k[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.q_proj.weight")).reshape(n_heads, head_dim, hidden).transpose(2, 0, 1)
        qry_b[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.q_proj.bias")).reshape(n_heads, head_dim)
        # out kernel: expected (heads, head_dim, hidden) — different axis order from k/q/v
        out_k[i] = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.out_proj.weight")).T.reshape(n_heads, head_dim, hidden)
        # out bias: expected (hidden,) flattened
        out_b[i]  = to_np(state_dict.pop(f"{pfx}.{i}.self_attn.out_proj.bias"))

    jax_dict["img/Transformer/encoderblock/LayerNorm_0/scale"] = ln0_scale
    jax_dict["img/Transformer/encoderblock/LayerNorm_0/bias"]  = ln0_bias
    jax_dict["img/Transformer/encoderblock/LayerNorm_1/scale"] = ln1_scale
    jax_dict["img/Transformer/encoderblock/LayerNorm_1/bias"]  = ln1_bias
    jax_dict["img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = mlp_d0_k
    jax_dict["img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"]   = mlp_d0_b
    jax_dict["img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = mlp_d1_k
    jax_dict["img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"]   = mlp_d1_b
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"]   = key_k
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"]     = key_b
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = val_k
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"]   = val_b
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = qry_k
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"]   = qry_b
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"]   = out_k
    jax_dict["img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"]     = out_b

    # ---- encoder norm ----
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    jax_dict["img/Transformer/encoder_norm/scale"] = to_np(state_dict.pop(pt_key)).reshape(hidden)

    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    jax_dict["img/Transformer/encoder_norm/bias"] = to_np(state_dict.pop(pt_key))

    # ---- multimodal projector ----
    pt_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    jax_dict["img/head/kernel"] = to_np(state_dict.pop(pt_key)).T

    pt_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    jax_dict["img/head/bias"] = to_np(state_dict.pop(pt_key))

    # ---- text (Gemma LLM) embeddings ----
    pt_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    jax_dict["llm/embedder/input_embedding"] = to_np(state_dict.pop(pt_key))

    # ---- LLM attention + MLP (stacked) ----
    n_text   = config.text_config.num_hidden_layers
    t_hidden = config.text_config.hidden_size
    t_heads  = config.text_config.num_attention_heads
    t_hdim   = config.text_config.head_dim

    llm_pfx = "paligemma_with_expert.paligemma.model.language_model.layers"

    # Infer intermediate size from actual weight shape (avoids // 2 assumption)
    t_inter = to_np(state_dict[f"{llm_pfx}.0.mlp.gate_proj.weight"]).shape[0]

    attn_vec = np.zeros((n_text, t_heads, t_hdim, t_hidden), dtype=np.float32)
    kv_ein   = np.zeros((n_text, 2, 1, t_hidden, t_hdim),   dtype=np.float32)
    q_ein    = np.zeros((n_text, t_heads, t_hidden, t_hdim), dtype=np.float32)
    gate_ein = np.zeros((n_text, 2, t_hidden, t_inter),      dtype=np.float32)
    lin_mlp  = np.zeros((n_text, t_inter, t_hidden),         dtype=np.float32)
    pre_attn = np.zeros((n_text, t_hidden), dtype=np.float32)
    pre_ffw  = np.zeros((n_text, t_hidden), dtype=np.float32)

    for i in range(n_text):
        # q: forward was .transpose(0,2,1).reshape(heads*hdim, hidden)
        # inverse: .reshape(heads, hdim, hidden).transpose(0,2,1)  → (heads, hidden, hdim)
        q_w = to_np(state_dict.pop(f"{llm_pfx}.{i}.self_attn.q_proj.weight"))
        q_ein[i] = q_w.reshape(t_heads, t_hdim, t_hidden).transpose(0, 2, 1)

        # k: forward was kv[i,0,0].transpose()  → inverse: .T then [None] to restore dims
        k_w = to_np(state_dict.pop(f"{llm_pfx}.{i}.self_attn.k_proj.weight"))
        kv_ein[i, 0, 0] = k_w.T

        v_w = to_np(state_dict.pop(f"{llm_pfx}.{i}.self_attn.v_proj.weight"))
        kv_ein[i, 1, 0] = v_w.T

        # o: forward was .transpose(2,0,1).reshape(heads*hdim, hidden)
        # inverse: .reshape(heads, hdim, hidden).transpose(1,2,0)
        o_w = to_np(state_dict.pop(f"{llm_pfx}.{i}.self_attn.o_proj.weight"))
        # attn_vec[i] = o_w.reshape(t_heads, t_hdim, t_hidden)
        attn_vec[i] = o_w.reshape(t_hidden, t_heads, t_hdim).transpose(1, 2, 0)

        gate_proj = to_np(state_dict.pop(f"{llm_pfx}.{i}.mlp.gate_proj.weight"))
        gate_ein[i, 0] = gate_proj.T
        up_proj = to_np(state_dict.pop(f"{llm_pfx}.{i}.mlp.up_proj.weight"))
        gate_ein[i, 1] = up_proj.T
        down_proj = to_np(state_dict.pop(f"{llm_pfx}.{i}.mlp.down_proj.weight"))
        lin_mlp[i] = down_proj.T

        pre_attn[i] = to_np(state_dict.pop(f"{llm_pfx}.{i}.input_layernorm.weight"))
        pre_ffw[i]  = to_np(state_dict.pop(f"{llm_pfx}.{i}.post_attention_layernorm.weight"))

    jax_dict["llm/layers/attn/attn_vec_einsum/w"] = attn_vec
    jax_dict["llm/layers/attn/kv_einsum/w"]        = kv_ein
    jax_dict["llm/layers/attn/q_einsum/w"]         = q_ein
    jax_dict["llm/layers/mlp/gating_einsum"]        = gate_ein
    jax_dict["llm/layers/mlp/linear"]               = lin_mlp
    jax_dict["llm/layers/pre_attention_norm/scale"] = pre_attn
    jax_dict["llm/layers/pre_ffw_norm/scale"]       = pre_ffw

    # ---- final LLM norm ----
    pt_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    jax_dict["llm/final_norm/scale"] = to_np(state_dict.pop(pt_key))

    # ---- expert keys (pass through for unslice_gemma to handle) ----
    expert_keys_pt = [k for k in list(state_dict.keys()) if "gemma_expert" in k]
    expert_dict = {k: state_dict.pop(k) for k in expert_keys_pt}

    return jax_dict, expert_dict


# ---------------------------------------------------------------------------
# Reverse Gemma (action expert) conversion
# ---------------------------------------------------------------------------

def unslice_gemma_state_dict(
    expert_dict: dict[str, torch.Tensor],
    config,
    pi05: bool,
) -> dict[str, np.ndarray]:
    """Inverse of slice_gemma_state_dict (num_expert=1)."""
    jax_dict: dict[str, np.ndarray] = {}

    n_layers = config.num_hidden_layers
    hidden   = config.hidden_size
    n_heads  = config.num_attention_heads
    head_dim = config.head_dim
    # intermediate size: gate_proj shape is (hidden, inter) after transpose
    # so inter = gate_proj.weight.shape[0]
    sample_key = "paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight"
    inter = to_np(expert_dict[sample_key]).shape[0]

    attn_vec = np.zeros((n_layers, n_heads, head_dim, hidden), dtype=np.float32)
    kv_ein   = np.zeros((n_layers, 2, 1, hidden, head_dim),   dtype=np.float32)
    q_ein    = np.zeros((n_layers, n_heads, hidden, head_dim), dtype=np.float32)
    gate_ein = np.zeros((n_layers, 2, hidden, inter),          dtype=np.float32)
    lin_mlp  = np.zeros((n_layers, inter, hidden),             dtype=np.float32)

    pfx = "paligemma_with_expert.gemma_expert.model.layers"

    if pi05:
        # Infer dense output dim from actual bias shape (may differ from hidden)
        _sample_bias = to_np(expert_dict[f"{pfx}.0.input_layernorm.dense.bias"])
        dense_out = _sample_bias.shape[0]
        # Infer dense input dim from kernel shape: kernel is (out, in) after .T in forward
        _sample_kernel = to_np(expert_dict[f"{pfx}.0.input_layernorm.dense.weight"])
        dense_in = _sample_kernel.shape[1]  # shape is (dense_out, dense_in) post-transpose

        pre_attn_bias   = np.zeros((n_layers, dense_out),           dtype=np.float32)
        pre_attn_kernel = np.zeros((n_layers, dense_in, dense_out), dtype=np.float32)
        pre_ffw_bias    = np.zeros((n_layers, dense_out),           dtype=np.float32)
        pre_ffw_kernel  = np.zeros((n_layers, dense_in, dense_out), dtype=np.float32)
    else:
        pre_attn = np.zeros((n_layers, hidden), dtype=np.float32)
        pre_ffw  = np.zeros((n_layers, hidden), dtype=np.float32)

    for i in range(n_layers):
        q_w = to_np(expert_dict.pop(f"{pfx}.{i}.self_attn.q_proj.weight"))
        q_ein[i] = q_w.reshape(n_heads, head_dim, hidden).transpose(0, 2, 1)

        k_w = to_np(expert_dict.pop(f"{pfx}.{i}.self_attn.k_proj.weight"))
        kv_ein[i, 0, 0] = k_w.T

        v_w = to_np(expert_dict.pop(f"{pfx}.{i}.self_attn.v_proj.weight"))
        kv_ein[i, 1, 0] = v_w.T

        # forward: .reshape(heads*hdim, hidden).transpose(1,0)
        # inverse: .T then .reshape(heads, hdim, hidden)
        o_w = to_np(expert_dict.pop(f"{pfx}.{i}.self_attn.o_proj.weight"))
        attn_vec[i] = o_w.T.reshape(n_heads, head_dim, hidden)

        gate_proj = to_np(expert_dict.pop(f"{pfx}.{i}.mlp.gate_proj.weight"))
        gate_ein[i, 0] = gate_proj.T
        up_proj = to_np(expert_dict.pop(f"{pfx}.{i}.mlp.up_proj.weight"))
        gate_ein[i, 1] = up_proj.T
        down_proj = to_np(expert_dict.pop(f"{pfx}.{i}.mlp.down_proj.weight"))
        lin_mlp[i] = down_proj.T

        if pi05:
            pre_attn_bias[i]   = to_np(expert_dict.pop(f"{pfx}.{i}.input_layernorm.dense.bias"))
            pre_attn_kernel[i] = to_np(expert_dict.pop(f"{pfx}.{i}.input_layernorm.dense.weight")).T
            pre_ffw_bias[i]    = to_np(expert_dict.pop(f"{pfx}.{i}.post_attention_layernorm.dense.bias"))
            pre_ffw_kernel[i]  = to_np(expert_dict.pop(f"{pfx}.{i}.post_attention_layernorm.dense.weight")).T
        else:
            pre_attn[i] = to_np(expert_dict.pop(f"{pfx}.{i}.input_layernorm.weight"))
            pre_ffw[i]  = to_np(expert_dict.pop(f"{pfx}.{i}.post_attention_layernorm.weight"))

    jax_dict["llm/layers/attn/attn_vec_einsum_1/w"] = attn_vec
    jax_dict["llm/layers/attn/kv_einsum_1/w"]        = kv_ein
    jax_dict["llm/layers/attn/q_einsum_1/w"]         = q_ein
    jax_dict["llm/layers/mlp_1/gating_einsum"]        = gate_ein
    jax_dict["llm/layers/mlp_1/linear"]               = lin_mlp

    if pi05:
        jax_dict["llm/layers/pre_attention_norm_1/Dense_0/bias"]   = pre_attn_bias
        jax_dict["llm/layers/pre_attention_norm_1/Dense_0/kernel"] = pre_attn_kernel
        jax_dict["llm/layers/pre_ffw_norm_1/Dense_0/bias"]         = pre_ffw_bias
        jax_dict["llm/layers/pre_ffw_norm_1/Dense_0/kernel"]        = pre_ffw_kernel
    else:
        jax_dict["llm/layers/pre_attention_norm_1/scale"] = pre_attn
        jax_dict["llm/layers/pre_ffw_norm_1/scale"]       = pre_ffw

    # final norm
    if pi05:
        jax_dict["llm/final_norm_1/Dense_0/bias"]   = to_np(expert_dict.pop("paligemma_with_expert.gemma_expert.model.norm.dense.bias"))
        jax_dict["llm/final_norm_1/Dense_0/kernel"] = to_np(expert_dict.pop("paligemma_with_expert.gemma_expert.model.norm.dense.weight")).T
    else:
        jax_dict["llm/final_norm_1/scale"] = to_np(expert_dict.pop("paligemma_with_expert.gemma_expert.model.norm.weight"))

    return jax_dict


# ---------------------------------------------------------------------------
# Unflatten flat JAX dict -> nested dict matching OpenPi param tree
# ---------------------------------------------------------------------------

def unflatten_jax_params(flat: dict[str, np.ndarray]) -> dict:
    """Convert flat slash-separated keys back to nested dict."""
    nested = {}
    for key, value in flat.items():
        parts = key.split("/")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_pt_to_jax(
    input_path: str,
    output_path: str,
    config_name: str,
    from_pt: bool,
    precision: str,
):
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    pi05 = model_config.pi05
    print(f"pi05 model: {pi05}")

    # Load weights
    state_dict = load_state_dict(input_path, from_pt)
    print(f"Loaded {len(state_dict)} keys")

    # Make all values float32 numpy for JAX
    state_dict = {k: v.clone() for k, v in state_dict.items()}  # detach

    # ---- Projection layers ----
    if pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = ["state_proj", "action_in_proj", "action_out_proj",
                     "action_time_mlp_in", "action_time_mlp_out"]

    proj_jax: dict[str, np.ndarray] = {}
    for key in proj_keys:
        w = to_np(state_dict.pop(f"{key}.weight"))
        b = to_np(state_dict.pop(f"{key}.bias"))
        # forward was: torch.from_numpy(weight).T  →  inverse: .T
        proj_jax[key] = {"kernel": w.T, "bias": b}

    # ---- PaliGemma + expert ----
    paligemma_config_obj = type("PaliGemmaConfig", (), {
        "vision_config": type("VC", (), {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 14,
            "projection_dim": 2048,
        })(),
        "text_config": type("TC", (), {
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "head_dim": 256,
            "intermediate_size": 16384,
        })(),
    })()

    action_expert_config = openpi.models.gemma.get_config("gemma_300m")
    # ensure attributes exist
    for attr, src in [("hidden_size", "width"), ("num_hidden_layers", "depth"),
                      ("num_attention_heads", "num_heads")]:
        if not hasattr(action_expert_config, attr):
            setattr(action_expert_config, attr, getattr(action_expert_config, src))

    paligemma_jax, expert_pt = unslice_paligemma_state_dict(
        state_dict, paligemma_config_obj, pi05=pi05
    )
    expert_jax = unslice_gemma_state_dict(expert_pt, action_expert_config, pi05=pi05)

    # ---- Merge all JAX params into flat dict ----
    all_flat = {**paligemma_jax, **expert_jax}

    # ---- Build nested param tree matching OpenPi's restore_params output ----
    # OpenPi nests under PaliGemma key + projection keys at top level
    paligemma_nested = unflatten_jax_params(all_flat)

    params = {"PaliGemma": paligemma_nested}
    for key, val in proj_jax.items():
        params[key] = val

    # ---- Convert to JAX arrays ----
    dtype = jnp.bfloat16 if precision == "bfloat16" else jnp.float32

    def to_jax_array(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x, dtype=dtype)
        return x

    params_jax = jax.tree_util.tree_map(to_jax_array, params)

    # Reference checkpoint stores every leaf as {"value": array}
    def wrap_value(x):
        return {"value": x}
    params_jax = jax.tree_util.tree_map(wrap_value, params_jax)

    # Wrap under "params" key — OpenPi's restore_params expects checkpoint["params"]
    save_tree = {"params": params_jax}

    # ---- Shape verification against reference checkpoint (optional but recommended) ----
    import os
    ref_dir = os.environ.get("OPENPI_REF_CHECKPOINT")
    if ref_dir:
        print(f"Verifying shapes against reference checkpoint: {ref_dir}")
        # Use metadata only — avoids GPU sharding issues when restoring
        ref_checkpointer = ocp.PyTreeCheckpointer()
        ref_meta = ref_checkpointer.metadata(f"{ref_dir}/params")

        def extract_shapes(tree, prefix=""):
            shapes = {}
            if hasattr(tree, "keys"):
                for k, v in tree.items():
                    shapes.update(extract_shapes(v, f"{prefix}['{k}']"))
            elif hasattr(tree, "shape"):
                shapes[prefix] = tuple(tree.shape)
            return shapes

        def extract_shapes_jax(tree, prefix=""):
            shapes = {}
            if isinstance(tree, dict):
                for k, v in tree.items():
                    shapes.update(extract_shapes_jax(v, f"{prefix}['{k}']"))
            elif hasattr(tree, "shape"):
                shapes[prefix] = tuple(tree.shape)
            return shapes

        ref_map  = extract_shapes(ref_meta)
        ours_map = extract_shapes_jax(save_tree)
        mismatches = 0
        for key in ref_map:
            if key not in ours_map:
                print(f"  MISSING: {key}")
                mismatches += 1
            elif ref_map[key] != ours_map[key]:
                print(f"  MISMATCH {key}: expected {ref_map[key]}, got {ours_map[key]}")
                mismatches += 1
        for key in ours_map:
            if key not in ref_map:
                print(f"  EXTRA:   {key}")
                mismatches += 1
        if mismatches == 0:
            print("  All shapes match!")
        else:
            print(f"  {mismatches} mismatches found — fix before serving.")

    # ---- Save with Orbax ----
    out_dir = pathlib.Path(output_path).resolve()
    params_dir = out_dir / "params"

    # Remove any previous partial run so Orbax doesn't complain
    if params_dir.exists():
        print(f"Removing existing {params_dir}")
        shutil.rmtree(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving Orbax checkpoint to {params_dir}")
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(params_dir, save_tree, force=True)
    checkpointer.wait_until_finished()

    # ---- Copy normstats if present alongside input ----
    pi_src = pathlib.Path(input_path) / "physical-intelligence"
    pi_dst = out_dir / "assets" / "physical-intelligence"

    if pi_src.exists():
        if pi_dst.exists():
            shutil.rmtree(pi_dst)

        shutil.copytree(pi_src, pi_dst)
        print(f"Copied assets from {pi_src}")

    print(f"\nDone! Orbax checkpoint saved to {output_path}")


def main(
    input_path: str,
    output_path: str,
    config_name: str,
    precision: Literal["float32", "bfloat16"] = "float32",
    *,
    from_pt: bool = False,
):
    """Convert a PyTorch checkpoint back to JAX/Orbax format for OpenPi serving.

    Args:
        input_path:   Path to .pt file (use --from_pt) or safetensors directory.
        output_path:  Where to write the Orbax checkpoint.
        config_name:  OpenPi config name (e.g. pi05_droid, pi0_aloha_sim).
        precision:    Output dtype (bfloat16 recommended).
        from_pt:      Set this flag when input_path is a raw .pt state dict file.
    """
    convert_pt_to_jax(input_path, output_path, config_name, from_pt, precision)


if __name__ == "__main__":
    tyro.cli(main)