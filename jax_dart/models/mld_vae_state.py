"""Torch-style state helpers for the JAX DART VAE.

The saved names mirror ``model.mld_vae.AutoMldVae.state_dict()``. Keeping this
layout makes parity checks, checkpoint inspection, and eventual checkpoint
conversion straightforward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


ArrayMap = Dict[str, np.ndarray]


def _to_numpy(variable) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(variable[...]), dtype=np.float32)


def _assign(variable, value: np.ndarray) -> None:
    import jax.numpy as jnp

    value = jnp.asarray(value, dtype=jnp.float32)
    try:
        variable[...] = value
        return
    except Exception:
        pass

    if hasattr(variable, "set_value"):
        variable.set_value(value)
    else:
        variable.value = value


def _collect_linear(out: ArrayMap, prefix: str, module) -> None:
    out[f"{prefix}.weight"] = _to_numpy(module.weight)
    out[f"{prefix}.bias"] = _to_numpy(module.bias)


def _load_linear(state: ArrayMap, prefix: str, module) -> None:
    _assign(module.weight, state[f"{prefix}.weight"])
    _assign(module.bias, state[f"{prefix}.bias"])


def _collect_layer_norm(out: ArrayMap, prefix: str, module) -> None:
    out[f"{prefix}.weight"] = _to_numpy(module.weight)
    out[f"{prefix}.bias"] = _to_numpy(module.bias)


def _load_layer_norm(state: ArrayMap, prefix: str, module) -> None:
    _assign(module.weight, state[f"{prefix}.weight"])
    _assign(module.bias, state[f"{prefix}.bias"])


def _collect_mha(out: ArrayMap, prefix: str, module) -> None:
    out[f"{prefix}.in_proj_weight"] = _to_numpy(module.in_proj_weight)
    out[f"{prefix}.in_proj_bias"] = _to_numpy(module.in_proj_bias)
    out[f"{prefix}.out_proj.weight"] = _to_numpy(module.out_proj_weight)
    out[f"{prefix}.out_proj.bias"] = _to_numpy(module.out_proj_bias)


def _load_mha(state: ArrayMap, prefix: str, module) -> None:
    _assign(module.in_proj_weight, state[f"{prefix}.in_proj_weight"])
    _assign(module.in_proj_bias, state[f"{prefix}.in_proj_bias"])
    _assign(module.out_proj_weight, state[f"{prefix}.out_proj.weight"])
    _assign(module.out_proj_bias, state[f"{prefix}.out_proj.bias"])


def _collect_encoder_layer(out: ArrayMap, prefix: str, module) -> None:
    _collect_mha(out, f"{prefix}.self_attn", module.self_attn)
    _collect_linear(out, f"{prefix}.linear1", module.linear1)
    _collect_linear(out, f"{prefix}.linear2", module.linear2)
    _collect_layer_norm(out, f"{prefix}.norm1", module.norm1)
    _collect_layer_norm(out, f"{prefix}.norm2", module.norm2)


def _load_encoder_layer(state: ArrayMap, prefix: str, module) -> None:
    _load_mha(state, f"{prefix}.self_attn", module.self_attn)
    _load_linear(state, f"{prefix}.linear1", module.linear1)
    _load_linear(state, f"{prefix}.linear2", module.linear2)
    _load_layer_norm(state, f"{prefix}.norm1", module.norm1)
    _load_layer_norm(state, f"{prefix}.norm2", module.norm2)


def _collect_decoder_layer(out: ArrayMap, prefix: str, module) -> None:
    _collect_mha(out, f"{prefix}.self_attn", module.self_attn)
    _collect_mha(out, f"{prefix}.multihead_attn", module.multihead_attn)
    _collect_linear(out, f"{prefix}.linear1", module.linear1)
    _collect_linear(out, f"{prefix}.linear2", module.linear2)
    _collect_layer_norm(out, f"{prefix}.norm1", module.norm1)
    _collect_layer_norm(out, f"{prefix}.norm2", module.norm2)
    _collect_layer_norm(out, f"{prefix}.norm3", module.norm3)


def _load_decoder_layer(state: ArrayMap, prefix: str, module) -> None:
    _load_mha(state, f"{prefix}.self_attn", module.self_attn)
    _load_mha(state, f"{prefix}.multihead_attn", module.multihead_attn)
    _load_linear(state, f"{prefix}.linear1", module.linear1)
    _load_linear(state, f"{prefix}.linear2", module.linear2)
    _load_layer_norm(state, f"{prefix}.norm1", module.norm1)
    _load_layer_norm(state, f"{prefix}.norm2", module.norm2)
    _load_layer_norm(state, f"{prefix}.norm3", module.norm3)


def _collect_skip_encoder(out: ArrayMap, prefix: str, module, num_layers: int) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _collect_encoder_layer(out, f"{prefix}.input_blocks.{index}", module.input_blocks[index])
        _collect_encoder_layer(out, f"{prefix}.output_blocks.{index}", module.output_blocks[index])
        _collect_linear(out, f"{prefix}.linear_blocks.{index}", module.linear_blocks[index])
    _collect_encoder_layer(out, f"{prefix}.middle_block", module.middle_block)
    _collect_layer_norm(out, f"{prefix}.norm", module.norm)


def _load_skip_encoder(state: ArrayMap, prefix: str, module, num_layers: int) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _load_encoder_layer(state, f"{prefix}.input_blocks.{index}", module.input_blocks[index])
        _load_encoder_layer(state, f"{prefix}.output_blocks.{index}", module.output_blocks[index])
        _load_linear(state, f"{prefix}.linear_blocks.{index}", module.linear_blocks[index])
    _load_encoder_layer(state, f"{prefix}.middle_block", module.middle_block)
    _load_layer_norm(state, f"{prefix}.norm", module.norm)


def _collect_skip_decoder(out: ArrayMap, prefix: str, module, num_layers: int) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _collect_decoder_layer(out, f"{prefix}.input_blocks.{index}", module.input_blocks[index])
        _collect_decoder_layer(out, f"{prefix}.output_blocks.{index}", module.output_blocks[index])
        _collect_linear(out, f"{prefix}.linear_blocks.{index}", module.linear_blocks[index])
    _collect_decoder_layer(out, f"{prefix}.middle_block", module.middle_block)
    _collect_layer_norm(out, f"{prefix}.norm", module.norm)


def _load_skip_decoder(state: ArrayMap, prefix: str, module, num_layers: int) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _load_decoder_layer(state, f"{prefix}.input_blocks.{index}", module.input_blocks[index])
        _load_decoder_layer(state, f"{prefix}.output_blocks.{index}", module.output_blocks[index])
        _load_linear(state, f"{prefix}.linear_blocks.{index}", module.linear_blocks[index])
    _load_decoder_layer(state, f"{prefix}.middle_block", module.middle_block)
    _load_layer_norm(state, f"{prefix}.norm", module.norm)


def collect_torch_style_vae_state(model, num_layers: int, arch: str) -> ArrayMap:
    out: ArrayMap = {}
    out["global_motion_token"] = _to_numpy(model.global_motion_token)
    out["latent_mean"] = _to_numpy(model.latent_mean)
    out["latent_std"] = _to_numpy(model.latent_std)
    out["query_pos_encoder.pe"] = _to_numpy(model.query_pos_encoder.pe)
    out["query_pos_decoder.pe"] = _to_numpy(model.query_pos_decoder.pe)

    _collect_skip_encoder(out, "encoder", model.encoder, num_layers)
    _collect_linear(out, "encoder_latent_proj", model.encoder_latent_proj)
    if arch == "all_encoder":
        _collect_skip_encoder(out, "decoder", model.decoder, num_layers)
    elif arch == "encoder_decoder":
        _collect_skip_decoder(out, "decoder", model.decoder, num_layers)
    else:
        raise ValueError(f"Unsupported VAE architecture {arch}.")
    _collect_linear(out, "decoder_latent_proj", model.decoder_latent_proj)
    _collect_linear(out, "skel_embedding", model.skel_embedding)
    _collect_linear(out, "final_layer", model.final_layer)
    return out


def load_torch_style_vae_state(model, state: ArrayMap, num_layers: int, arch: str) -> None:
    _assign(model.global_motion_token, state["global_motion_token"])
    _assign(model.latent_mean, state["latent_mean"])
    _assign(model.latent_std, state["latent_std"])
    _assign(model.query_pos_encoder.pe, state["query_pos_encoder.pe"])
    _assign(model.query_pos_decoder.pe, state["query_pos_decoder.pe"])

    _load_skip_encoder(state, "encoder", model.encoder, num_layers)
    _load_linear(state, "encoder_latent_proj", model.encoder_latent_proj)
    if arch == "all_encoder":
        _load_skip_encoder(state, "decoder", model.decoder, num_layers)
    elif arch == "encoder_decoder":
        _load_skip_decoder(state, "decoder", model.decoder, num_layers)
    else:
        raise ValueError(f"Unsupported VAE architecture {arch}.")
    _load_linear(state, "decoder_latent_proj", model.decoder_latent_proj)
    _load_linear(state, "skel_embedding", model.skel_embedding)
    _load_linear(state, "final_layer", model.final_layer)


def state_from_npz(path: str) -> ArrayMap:
    prefix = "state::"
    with np.load(path) as data:
        state = {
            key[len(prefix) :]: data[key].astype(np.float32, copy=False)
            for key in data.files
            if key.startswith(prefix)
        }
    if not state:
        raise ValueError(f"{path} does not contain state::* arrays.")
    return state


def save_vae_npz_checkpoint(
    path: str,
    model,
    *,
    num_layers: int,
    arch: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
) -> None:
    arrays: Dict[str, Any] = {
        "config_json": np.array(json.dumps(config, sort_keys=True)),
        "metrics_json": np.array(json.dumps(metrics or {}, sort_keys=True)),
    }
    if step is not None:
        arrays["step"] = np.array(step, dtype=np.int64)
    for name, value in collect_torch_style_vae_state(model, num_layers, arch).items():
        arrays[f"state::{name}"] = value

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
