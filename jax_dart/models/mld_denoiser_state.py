"""Torch-style state helpers for the JAX DART denoisers."""

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


def _collect_shared_timestep(out: ArrayMap, model) -> None:
    pe = _to_numpy(model.sequence_pos_encoder.pe)
    out["sequence_pos_encoder.pe"] = pe
    out["embed_timestep.sequence_pos_encoder.pe"] = pe
    _collect_linear(out, "embed_timestep.time_embed.0", model.embed_timestep.time_embed[0])
    _collect_linear(out, "embed_timestep.time_embed.2", model.embed_timestep.time_embed[1])


def _load_shared_timestep(state: ArrayMap, model) -> None:
    pe_key = "sequence_pos_encoder.pe"
    if pe_key not in state:
        pe_key = "embed_timestep.sequence_pos_encoder.pe"
    _assign(model.sequence_pos_encoder.pe, state[pe_key])
    _load_linear(state, "embed_timestep.time_embed.0", model.embed_timestep.time_embed[0])
    _load_linear(state, "embed_timestep.time_embed.2", model.embed_timestep.time_embed[1])


def _collect_mlp_state(out: ArrayMap, model) -> None:
    _collect_shared_timestep(out, model)
    _collect_linear(out, "input_project", model.input_project)
    for block_index, block in enumerate(model.mlp.layers):
        for layer_index, linear in enumerate(block.layers):
            _collect_linear(out, f"mlp.layers.{block_index}.layers.{layer_index}", linear)
    _collect_linear(out, "mlp.out_fc", model.mlp.out_fc)


def _load_mlp_state(state: ArrayMap, model) -> None:
    _load_shared_timestep(state, model)
    _load_linear(state, "input_project", model.input_project)
    for block_index, block in enumerate(model.mlp.layers):
        for layer_index, linear in enumerate(block.layers):
            _load_linear(state, f"mlp.layers.{block_index}.layers.{layer_index}", linear)
    _load_linear(state, "mlp.out_fc", model.mlp.out_fc)


def _collect_transformer_layer(out: ArrayMap, prefix: str, layer) -> None:
    _collect_mha(out, f"{prefix}.self_attn", layer.self_attn)
    _collect_linear(out, f"{prefix}.linear1", layer.linear1)
    _collect_linear(out, f"{prefix}.linear2", layer.linear2)
    _collect_layer_norm(out, f"{prefix}.norm1", layer.norm1)
    _collect_layer_norm(out, f"{prefix}.norm2", layer.norm2)


def _load_transformer_layer(state: ArrayMap, prefix: str, layer) -> None:
    _load_mha(state, f"{prefix}.self_attn", layer.self_attn)
    _load_linear(state, f"{prefix}.linear1", layer.linear1)
    _load_linear(state, f"{prefix}.linear2", layer.linear2)
    _load_layer_norm(state, f"{prefix}.norm1", layer.norm1)
    _load_layer_norm(state, f"{prefix}.norm2", layer.norm2)


def _collect_transformer_state(out: ArrayMap, model) -> None:
    _collect_shared_timestep(out, model)
    _collect_linear(out, "embed_text", model.embed_text)
    _collect_linear(out, "embed_history", model.embed_history)
    _collect_linear(out, "embed_noise", model.embed_noise)
    for index, layer in enumerate(model.seqTransEncoder.layers):
        _collect_transformer_layer(out, f"seqTransEncoder.layers.{index}", layer)
    _collect_linear(out, "output_process", model.output_process)


def _load_transformer_state(state: ArrayMap, model) -> None:
    _load_shared_timestep(state, model)
    _load_linear(state, "embed_text", model.embed_text)
    _load_linear(state, "embed_history", model.embed_history)
    _load_linear(state, "embed_noise", model.embed_noise)
    for index, layer in enumerate(model.seqTransEncoder.layers):
        _load_transformer_layer(state, f"seqTransEncoder.layers.{index}", layer)
    _load_linear(state, "output_process", model.output_process)


def collect_torch_style_denoiser_state(model) -> ArrayMap:
    out: ArrayMap = {}
    if hasattr(model, "seqTransEncoder"):
        _collect_transformer_state(out, model)
    else:
        _collect_mlp_state(out, model)
    return out


def load_torch_style_denoiser_state(model, state: ArrayMap) -> None:
    if hasattr(model, "seqTransEncoder"):
        _load_transformer_state(state, model)
    else:
        _load_mlp_state(state, model)


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


def config_from_npz(path: str) -> Dict[str, Any]:
    with np.load(path) as data:
        if "config_json" not in data:
            return {}
        return json.loads(str(data["config_json"].tolist()))


def save_denoiser_npz_checkpoint(
    path: str,
    model,
    *,
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
    for name, value in collect_torch_style_denoiser_state(model).items():
        arrays[f"state::{name}"] = value

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
