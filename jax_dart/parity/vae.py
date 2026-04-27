"""Compare the Torch and JAX DART VAE implementations on one shard batch.

This runner is intentionally deterministic: it uses the encoder mean as the
latent and compares the same intermediate tensors from both implementations.
It can run as one process when Torch/JAX share an environment, or in two steps
when the Torch DART env and JAX env are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata


ArrayMap = Dict[str, np.ndarray]
FORWARD_NAMES = ("mu", "logvar", "std", "latent", "future_pred")
GRAD_NAMES = ("loss", "rec", "kl", "grad_history", "grad_future")
TEMPORAL_NAMES = ("joints_delta", "transl_delta", "orient_delta", "temporal_smpl")
SMPL_JOINT_NAMES = ("smpl_joints_rec", "joints_consistency", "smpl_joint_loss")


def _parse_latent_dim(value: str) -> Tuple[int, int]:
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("latent dim must contain two integers, e.g. '1 256'")
    return parts[0], parts[1]


def _torch_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return _torch_numpy(value)


def _assign_nnx_value(variable, value: np.ndarray) -> None:
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


def _copy_linear(torch_state: Dict[str, Any], prefix: str, jax_linear) -> None:
    _assign_nnx_value(jax_linear.weight, _as_numpy(torch_state[f"{prefix}.weight"]))
    _assign_nnx_value(jax_linear.bias, _as_numpy(torch_state[f"{prefix}.bias"]))


def _copy_layer_norm(torch_state: Dict[str, Any], prefix: str, jax_norm) -> None:
    _assign_nnx_value(jax_norm.weight, _as_numpy(torch_state[f"{prefix}.weight"]))
    _assign_nnx_value(jax_norm.bias, _as_numpy(torch_state[f"{prefix}.bias"]))


def _copy_mha(torch_state: Dict[str, Any], prefix: str, jax_mha) -> None:
    _assign_nnx_value(
        jax_mha.in_proj_weight,
        _as_numpy(torch_state[f"{prefix}.in_proj_weight"]),
    )
    _assign_nnx_value(
        jax_mha.in_proj_bias,
        _as_numpy(torch_state[f"{prefix}.in_proj_bias"]),
    )
    _assign_nnx_value(
        jax_mha.out_proj_weight,
        _as_numpy(torch_state[f"{prefix}.out_proj.weight"]),
    )
    _assign_nnx_value(
        jax_mha.out_proj_bias,
        _as_numpy(torch_state[f"{prefix}.out_proj.bias"]),
    )


def _copy_encoder_layer(torch_state: Dict[str, Any], prefix: str, jax_layer) -> None:
    _copy_mha(torch_state, f"{prefix}.self_attn", jax_layer.self_attn)
    _copy_linear(torch_state, f"{prefix}.linear1", jax_layer.linear1)
    _copy_linear(torch_state, f"{prefix}.linear2", jax_layer.linear2)
    _copy_layer_norm(torch_state, f"{prefix}.norm1", jax_layer.norm1)
    _copy_layer_norm(torch_state, f"{prefix}.norm2", jax_layer.norm2)


def _copy_decoder_layer(torch_state: Dict[str, Any], prefix: str, jax_layer) -> None:
    _copy_mha(torch_state, f"{prefix}.self_attn", jax_layer.self_attn)
    _copy_mha(torch_state, f"{prefix}.multihead_attn", jax_layer.multihead_attn)
    _copy_linear(torch_state, f"{prefix}.linear1", jax_layer.linear1)
    _copy_linear(torch_state, f"{prefix}.linear2", jax_layer.linear2)
    _copy_layer_norm(torch_state, f"{prefix}.norm1", jax_layer.norm1)
    _copy_layer_norm(torch_state, f"{prefix}.norm2", jax_layer.norm2)
    _copy_layer_norm(torch_state, f"{prefix}.norm3", jax_layer.norm3)


def _copy_skip_encoder(
    torch_state: Dict[str, Any],
    prefix: str,
    jax_encoder,
    num_layers: int,
) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _copy_encoder_layer(
            torch_state,
            f"{prefix}.input_blocks.{index}",
            jax_encoder.input_blocks[index],
        )
        _copy_encoder_layer(
            torch_state,
            f"{prefix}.output_blocks.{index}",
            jax_encoder.output_blocks[index],
        )
        _copy_linear(
            torch_state,
            f"{prefix}.linear_blocks.{index}",
            jax_encoder.linear_blocks[index],
        )
    _copy_encoder_layer(torch_state, f"{prefix}.middle_block", jax_encoder.middle_block)
    _copy_layer_norm(torch_state, f"{prefix}.norm", jax_encoder.norm)


def _copy_skip_decoder(
    torch_state: Dict[str, Any],
    prefix: str,
    jax_decoder,
    num_layers: int,
) -> None:
    num_block = (num_layers - 1) // 2
    for index in range(num_block):
        _copy_decoder_layer(
            torch_state,
            f"{prefix}.input_blocks.{index}",
            jax_decoder.input_blocks[index],
        )
        _copy_decoder_layer(
            torch_state,
            f"{prefix}.output_blocks.{index}",
            jax_decoder.output_blocks[index],
        )
        _copy_linear(
            torch_state,
            f"{prefix}.linear_blocks.{index}",
            jax_decoder.linear_blocks[index],
        )
    _copy_decoder_layer(torch_state, f"{prefix}.middle_block", jax_decoder.middle_block)
    _copy_layer_norm(torch_state, f"{prefix}.norm", jax_decoder.norm)


def copy_torch_vae_to_jax(torch_model, jax_model, num_layers: int, arch: str) -> None:
    """Copy a Torch ``model.mld_vae.AutoMldVae`` state into the JAX NNX port."""
    copy_torch_state_to_jax(torch_model.state_dict(), jax_model, num_layers=num_layers, arch=arch)


def copy_torch_state_to_jax(
    torch_state: Dict[str, Any],
    jax_model,
    num_layers: int,
    arch: str,
) -> None:
    """Copy Torch-style state tensors into the JAX NNX VAE."""
    _assign_nnx_value(jax_model.global_motion_token, _as_numpy(torch_state["global_motion_token"]))
    _assign_nnx_value(jax_model.latent_mean, _as_numpy(torch_state["latent_mean"]))
    _assign_nnx_value(jax_model.latent_std, _as_numpy(torch_state["latent_std"]))
    _assign_nnx_value(jax_model.query_pos_encoder.pe, _as_numpy(torch_state["query_pos_encoder.pe"]))
    _assign_nnx_value(jax_model.query_pos_decoder.pe, _as_numpy(torch_state["query_pos_decoder.pe"]))

    _copy_skip_encoder(torch_state, "encoder", jax_model.encoder, num_layers)
    _copy_linear(torch_state, "encoder_latent_proj", jax_model.encoder_latent_proj)

    if arch == "all_encoder":
        _copy_skip_encoder(torch_state, "decoder", jax_model.decoder, num_layers)
    elif arch == "encoder_decoder":
        _copy_skip_decoder(torch_state, "decoder", jax_model.decoder, num_layers)
    else:
        raise ValueError(f"Unsupported VAE architecture {arch}.")

    _copy_linear(torch_state, "decoder_latent_proj", jax_model.decoder_latent_proj)
    _copy_linear(torch_state, "skel_embedding", jax_model.skel_embedding)
    _copy_linear(torch_state, "final_layer", jax_model.final_layer)


def _extract_model_state(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _snapshot_state_key(name: str) -> str:
    return f"state::{name}"


def _collect_torch_state_np(torch_model) -> ArrayMap:
    return {
        _snapshot_state_key(name): _torch_numpy(value)
        for name, value in torch_model.state_dict().items()
    }


def _prefix_torch_state_np(torch_state: ArrayMap) -> ArrayMap:
    return {
        key if key.startswith(_snapshot_state_key("")) else _snapshot_state_key(key): value
        for key, value in torch_state.items()
    }


def _load_torch_state_np(snapshot: Dict[str, np.ndarray]) -> ArrayMap:
    prefix = _snapshot_state_key("")
    return {
        key[len(prefix) :]: value.astype(np.float32, copy=False)
        for key, value in snapshot.items()
        if key.startswith(prefix)
    }


def _load_config(snapshot: Dict[str, np.ndarray]) -> Dict[str, Any]:
    raw = snapshot["config_json"]
    return json.loads(str(raw.tolist()))


def _load_snapshot(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _load_motion_normalization_np(root: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(Path(root) / "normalization.npz") as data:
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def _config_from_args(
    args: argparse.Namespace,
    feature_dim: int,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    config = {
        "feature_dim": int(feature_dim),
        "h_dim": int(args.h_dim),
        "ff_size": int(args.ff_size),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "latent_dim": [int(args.latent_dim[0]), int(args.latent_dim[1])],
        "arch": args.arch,
        "activation": args.activation,
        "position_embedding": args.position_embedding,
        "normalize_before": bool(args.normalize_before),
        "seed": int(args.seed),
        "weight_rec": float(args.weight_rec),
        "weight_kl": float(args.weight_kl),
        "include_temporal_smpl_loss": bool(args.include_temporal_smpl_loss),
        "weight_joints_delta": float(args.weight_joints_delta),
        "weight_transl_delta": float(args.weight_transl_delta),
        "weight_orient_delta": float(args.weight_orient_delta),
        "include_smpl_joint_loss": bool(args.include_smpl_joint_loss),
        "weight_smpl_joints_rec": float(args.weight_smpl_joints_rec),
        "weight_joints_consistency": float(args.weight_joints_consistency),
        "smpl_model_dir": args.smpl_model_dir,
        "smpl_gender": args.smpl_gender,
    }
    if metadata is not None:
        config["metadata_format"] = metadata.get("format")
        config["feature_slices"] = metadata["primitive"]["feature_slices"]
    return config


def load_torch_checkpoint(torch_model, checkpoint_path: str, device: str) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = _extract_model_state(checkpoint)
    if "latent_mean" not in model_state:
        model_state["latent_mean"] = torch.tensor(0.0, device=device)
    if "latent_std" not in model_state:
        model_state["latent_std"] = torch.tensor(1.0, device=device)
    torch_model.load_state_dict(model_state)


def deterministic_torch_forward(torch_model, history_np: np.ndarray, future_np: np.ndarray, device: str) -> ArrayMap:
    import torch

    torch_model.eval()
    history = torch.from_numpy(history_np).to(device=device, dtype=torch.float32)
    future = torch.from_numpy(future_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        batch_size = future.shape[0]
        x = torch.cat((history, future), dim=1)
        x = torch_model.skel_embedding(x)
        x = x.permute(1, 0, 2)
        dist_tokens = torch.tile(
            torch_model.global_motion_token[:, None, :],
            (1, batch_size, 1),
        )
        xseq = torch.cat((dist_tokens, x), dim=0)
        xseq = torch_model.query_pos_encoder(xseq)
        dist = torch_model.encoder(xseq)[: dist_tokens.shape[0]]
        dist = torch_model.encoder_latent_proj(dist)
        mu = dist[: torch_model.latent_size]
        logvar = torch.clamp(dist[torch_model.latent_size :], min=-10.0, max=10.0)
        std = logvar.exp().pow(0.5)
        latent = mu
        future_pred = torch_model.decode(latent, history, nfuture=future.shape[1])

    return {
        "mu": _torch_numpy(mu),
        "logvar": _torch_numpy(logvar),
        "std": _torch_numpy(std),
        "latent": _torch_numpy(latent),
        "future_pred": _torch_numpy(future_pred),
    }


def deterministic_jax_forward(jax_model, history_np: np.ndarray, future_np: np.ndarray) -> ArrayMap:
    import jax
    import jax.numpy as jnp

    history = jnp.asarray(history_np, dtype=jnp.float32)
    future = jnp.asarray(future_np, dtype=jnp.float32)
    latent, dist = jax_model.encode(future, history, rng=None)
    future_pred = jax_model.decode(latent, history, nfuture=future.shape[1])

    return {
        "mu": np.asarray(jax.device_get(dist.loc), dtype=np.float32),
        "logvar": np.asarray(jax.device_get(dist.logvar), dtype=np.float32),
        "std": np.asarray(jax.device_get(dist.scale), dtype=np.float32),
        "latent": np.asarray(jax.device_get(latent), dtype=np.float32),
        "future_pred": np.asarray(jax.device_get(future_pred), dtype=np.float32),
    }


def _torch_huber_loss(pred, target, delta: float = 1.0):
    import torch

    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _torch_slice_motion_features(motion, feature_slices: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: motion[..., int(bounds[0]) : int(bounds[1])]
        for name, bounds in feature_slices.items()
    }


def _temporal_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "weight_joints_delta": float(config.get("weight_joints_delta", 0.0)),
        "weight_transl_delta": float(config.get("weight_transl_delta", 0.0)),
        "weight_orient_delta": float(config.get("weight_orient_delta", 0.0)),
    }


def _smpl_joint_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "weight_smpl_joints_rec": float(config.get("weight_smpl_joints_rec", 0.0)),
        "weight_joints_consistency": float(config.get("weight_joints_consistency", 0.0)),
    }


def _future_betas_from_batch(metadata: dict, batch: Dict[str, np.ndarray]) -> np.ndarray:
    if "betas" not in batch:
        raise KeyError("SMPL joint losses require exported shard betas.")
    history_length = int(metadata["primitive"]["history_length"])
    future_length = int(metadata["primitive"]["future_length"])
    return batch["betas"][:, history_length : history_length + future_length, :].astype(
        np.float32,
        copy=False,
    )


def _torch_build_smplx_model(smpl_model_dir: str, device: str, gender: str = "male"):
    import smplx

    model = smplx.build_layer(
        smpl_model_dir,
        model_type="smplx",
        gender=gender,
        ext="npz",
        num_pca_comps=12,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _torch_temporal_smpl_feature_loss(
    history,
    future_pred,
    *,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    import torch
    from pytorch3d import transforms

    norm_mean = torch.as_tensor(norm_mean_np, device=future_pred.device, dtype=torch.float32)
    norm_std = torch.as_tensor(norm_std_np, device=future_pred.device, dtype=torch.float32)
    pred_motion = torch.cat([history[:, -1:, :], future_pred], dim=1)
    pred_motion = pred_motion.float() * norm_std + norm_mean
    features = _torch_slice_motion_features(pred_motion, feature_slices)

    transl = features["transl"]
    poses_6d = features["poses_6d"]
    pred_transl_delta = features["transl_delta"][:, :-1, :]
    pred_orient_delta = features["global_orient_delta_6d"][:, :-1, :]
    joints = features["joints"]
    pred_joints_delta = features["joints_delta"][:, :-1, :]

    calc_joints_delta = joints[:, 1:, :] - joints[:, :-1, :]
    calc_transl_delta = transl[:, 1:, :] - transl[:, :-1, :]
    pred_orient = transforms.rotation_6d_to_matrix(poses_6d[:, :, :6])
    calc_orient_delta_matrix = torch.matmul(
        pred_orient[:, 1:],
        pred_orient[:, :-1].permute(0, 1, 3, 2),
    )
    calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)

    terms = {
        "joints_delta": torch.mean(_torch_huber_loss(calc_joints_delta, pred_joints_delta)),
        "transl_delta": torch.mean(_torch_huber_loss(calc_transl_delta, pred_transl_delta)),
        "orient_delta": torch.mean(_torch_huber_loss(calc_orient_delta_6d, pred_orient_delta)),
    }
    terms["temporal_smpl"] = (
        weights["weight_joints_delta"] * terms["joints_delta"]
        + weights["weight_transl_delta"] * terms["transl_delta"]
        + weights["weight_orient_delta"] * terms["orient_delta"]
    )
    return terms


def deterministic_torch_temporal_smpl_outputs(
    history_np: np.ndarray,
    future_pred_np: np.ndarray,
    device: str,
    *,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> ArrayMap:
    import torch

    history = torch.from_numpy(history_np).to(device=device, dtype=torch.float32)
    future_pred = torch.from_numpy(future_pred_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        terms = _torch_temporal_smpl_feature_loss(
            history,
            future_pred,
            feature_slices=feature_slices,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=weights,
        )
    return {
        name: _torch_numpy(terms[name]).reshape(())
        for name in TEMPORAL_NAMES
    }


def _torch_smpl_joints_from_motion(
    body_model,
    motion,
    betas,
    *,
    feature_slices: Dict[str, Any],
):
    from pytorch3d import transforms

    features = _torch_slice_motion_features(motion, feature_slices)
    poses_6d = features["poses_6d"]
    transl = features["transl"]
    batch_size, frames = poses_6d.shape[:2]
    rot_mats = transforms.rotation_6d_to_matrix(poses_6d.reshape(-1, 22, 6))
    smplxout = body_model(
        return_verts=False,
        betas=betas.reshape(-1, 10),
        global_orient=rot_mats[:, 0],
        body_pose=rot_mats[:, 1:],
        transl=transl.reshape(-1, 3),
    )
    return smplxout.joints[:, :22, :].reshape(batch_size, frames, 22, 3)


def _torch_smpl_joint_loss(
    future_pred,
    future,
    *,
    betas,
    body_model,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    import torch

    norm_mean = torch.as_tensor(norm_mean_np, device=future_pred.device, dtype=torch.float32)
    norm_std = torch.as_tensor(norm_std_np, device=future_pred.device, dtype=torch.float32)
    pred_motion = future_pred.float() * norm_std + norm_mean
    gt_motion = future.float() * norm_std + norm_mean

    pred_features = _torch_slice_motion_features(pred_motion, feature_slices)
    pred_feature_joints = pred_features["joints"].reshape(*pred_motion.shape[:-1], 22, 3)
    pred_smpl_joints = _torch_smpl_joints_from_motion(
        body_model,
        pred_motion,
        betas,
        feature_slices=feature_slices,
    )
    with torch.no_grad():
        gt_smpl_joints = _torch_smpl_joints_from_motion(
            body_model,
            gt_motion,
            betas,
            feature_slices=feature_slices,
        )

    terms = {
        "smpl_joints_rec": torch.mean(_torch_huber_loss(pred_smpl_joints, gt_smpl_joints)),
        "joints_consistency": torch.mean(_torch_huber_loss(pred_feature_joints, pred_smpl_joints)),
    }
    terms["smpl_joint_loss"] = (
        weights["weight_smpl_joints_rec"] * terms["smpl_joints_rec"]
        + weights["weight_joints_consistency"] * terms["joints_consistency"]
    )
    return terms


def deterministic_torch_smpl_joint_outputs(
    future_pred_np: np.ndarray,
    future_np: np.ndarray,
    betas_np: np.ndarray,
    device: str,
    *,
    smpl_model_dir: str,
    smpl_gender: str,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> ArrayMap:
    import torch

    body_model = _torch_build_smplx_model(smpl_model_dir, device, gender=smpl_gender)
    future_pred = torch.from_numpy(future_pred_np).to(device=device, dtype=torch.float32)
    future = torch.from_numpy(future_np).to(device=device, dtype=torch.float32)
    betas = torch.from_numpy(betas_np).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        terms = _torch_smpl_joint_loss(
            future_pred,
            future,
            betas=betas,
            body_model=body_model,
            feature_slices=feature_slices,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=weights,
        )
    return {
        name: _torch_numpy(terms[name]).reshape(())
        for name in SMPL_JOINT_NAMES
    }


def _torch_deterministic_parts(torch_model, history, future):
    import torch

    batch_size = future.shape[0]
    x = torch.cat((history, future), dim=1)
    x = torch_model.skel_embedding(x)
    x = x.permute(1, 0, 2)
    dist_tokens = torch.tile(
        torch_model.global_motion_token[:, None, :],
        (1, batch_size, 1),
    )
    xseq = torch.cat((dist_tokens, x), dim=0)
    xseq = torch_model.query_pos_encoder(xseq)
    dist = torch_model.encoder(xseq)[: dist_tokens.shape[0]]
    dist = torch_model.encoder_latent_proj(dist)
    mu = dist[: torch_model.latent_size]
    logvar = torch.clamp(dist[torch_model.latent_size :], min=-10.0, max=10.0)
    std = logvar.exp().pow(0.5)
    latent = mu
    future_pred = torch_model.decode(latent, history, nfuture=future.shape[1])
    return future_pred, mu, logvar, std


def deterministic_torch_loss_and_input_grads(
    torch_model,
    history_np: np.ndarray,
    future_np: np.ndarray,
    device: str,
    weight_rec: float,
    weight_kl: float,
    feature_slices: Optional[Dict[str, Any]] = None,
    norm_mean_np: Optional[np.ndarray] = None,
    norm_std_np: Optional[np.ndarray] = None,
    temporal_weights: Optional[Dict[str, float]] = None,
    betas_np: Optional[np.ndarray] = None,
    smpl_model_dir: Optional[str] = None,
    smpl_gender: str = "male",
    smpl_joint_weights: Optional[Dict[str, float]] = None,
) -> ArrayMap:
    import torch

    torch_model.eval()
    torch_model.zero_grad(set_to_none=True)
    history = torch.from_numpy(history_np).to(device=device, dtype=torch.float32)
    future = torch.from_numpy(future_np).to(device=device, dtype=torch.float32)
    history.requires_grad_(True)
    future.requires_grad_(True)

    future_pred, mu, logvar, std = _torch_deterministic_parts(torch_model, history, future)
    rec = torch.mean(_torch_huber_loss(future_pred, future))
    kl = 0.5 * torch.mean(mu**2 + std**2 - 1.0 - logvar)
    loss = weight_rec * rec + weight_kl * kl
    temporal_terms = {}
    if temporal_weights is not None:
        if feature_slices is None or norm_mean_np is None or norm_std_np is None:
            raise ValueError("Temporal SMPL loss requires feature slices, normalization, and weights.")
        temporal_terms = _torch_temporal_smpl_feature_loss(
            history,
            future_pred,
            feature_slices=feature_slices,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=temporal_weights,
        )
        loss = loss + temporal_terms["temporal_smpl"]
    smpl_terms = {}
    if smpl_model_dir is not None:
        if feature_slices is None or norm_mean_np is None or norm_std_np is None:
            raise ValueError("SMPL joint loss requires feature slices and normalization.")
        if betas_np is None or smpl_joint_weights is None:
            raise ValueError("SMPL joint loss requires betas and weights.")
        body_model = _torch_build_smplx_model(smpl_model_dir, device, gender=smpl_gender)
        betas = torch.from_numpy(betas_np).to(device=device, dtype=torch.float32)
        smpl_terms = _torch_smpl_joint_loss(
            future_pred,
            future,
            betas=betas,
            body_model=body_model,
            feature_slices=feature_slices,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=smpl_joint_weights,
        )
        loss = loss + smpl_terms["smpl_joint_loss"]
    loss.backward()

    outputs = {
        "loss": _torch_numpy(loss).reshape(()),
        "rec": _torch_numpy(rec).reshape(()),
        "kl": _torch_numpy(kl).reshape(()),
        "grad_history": _torch_numpy(history.grad),
        "grad_future": _torch_numpy(future.grad),
    }
    for name in TEMPORAL_NAMES:
        if name in temporal_terms:
            outputs[name] = _torch_numpy(temporal_terms[name]).reshape(())
    for name in SMPL_JOINT_NAMES:
        if name in smpl_terms:
            outputs[name] = _torch_numpy(smpl_terms[name]).reshape(())
    return outputs


def deterministic_jax_loss_and_input_grads(
    jax_model,
    history_np: np.ndarray,
    future_np: np.ndarray,
    weight_rec: float,
    weight_kl: float,
    feature_slices: Optional[Dict[str, Any]] = None,
    norm_mean_np: Optional[np.ndarray] = None,
    norm_std_np: Optional[np.ndarray] = None,
    temporal_weights: Optional[Dict[str, float]] = None,
    betas_np: Optional[np.ndarray] = None,
    smplx_model=None,
    smpl_joint_weights: Optional[Dict[str, float]] = None,
) -> ArrayMap:
    import jax
    import jax.numpy as jnp

    from jax_dart.models.mld_vae import vae_reconstruction_kl_loss
    from jax_dart.models.smplx_joints import smpl_joint_loss_from_motion
    from jax_dart.models.temporal_smpl_loss import temporal_smpl_feature_loss

    history = jnp.asarray(history_np, dtype=jnp.float32)
    future = jnp.asarray(future_np, dtype=jnp.float32)
    norm_mean = None if norm_mean_np is None else jnp.asarray(norm_mean_np, dtype=jnp.float32)
    norm_std = None if norm_std_np is None else jnp.asarray(norm_std_np, dtype=jnp.float32)
    betas = None if betas_np is None else jnp.asarray(betas_np, dtype=jnp.float32)

    def loss_fn(history_value, future_value):
        latent, dist = jax_model.encode(future_value, history_value, rng=None)
        future_pred = jax_model.decode(latent, history_value, nfuture=future_value.shape[1])
        loss, terms = vae_reconstruction_kl_loss(
            future_pred,
            future_value,
            dist,
            weight_rec=weight_rec,
            weight_kl=weight_kl,
        )
        aux_terms = {"rec": terms["rec"], "kl": terms["kl"]}
        if temporal_weights is not None:
            if feature_slices is None or norm_mean is None or norm_std is None:
                raise ValueError("Temporal SMPL loss requires feature slices, normalization, and weights.")
            temporal_loss, temporal_terms = temporal_smpl_feature_loss(
                history_value,
                future_pred,
                feature_slices=feature_slices,
                norm_mean=norm_mean,
                norm_std=norm_std,
                weight_joints_delta=temporal_weights["weight_joints_delta"],
                weight_transl_delta=temporal_weights["weight_transl_delta"],
                weight_orient_delta=temporal_weights["weight_orient_delta"],
            )
            loss = loss + temporal_loss
            aux_terms.update(temporal_terms)
        if smplx_model is not None:
            if feature_slices is None or norm_mean is None or norm_std is None:
                raise ValueError("SMPL joint loss requires feature slices and normalization.")
            if betas is None or smpl_joint_weights is None:
                raise ValueError("SMPL joint loss requires betas and weights.")
            smpl_loss, smpl_terms = smpl_joint_loss_from_motion(
                smplx_model,
                pred_motion=future_pred,
                gt_motion=future_value,
                betas=betas,
                feature_slices=feature_slices,
                norm_mean=norm_mean,
                norm_std=norm_std,
                weight_smpl_joints_rec=smpl_joint_weights["weight_smpl_joints_rec"],
                weight_joints_consistency=smpl_joint_weights["weight_joints_consistency"],
            )
            loss = loss + smpl_loss
            aux_terms.update(smpl_terms)
        return loss, aux_terms

    (loss, terms), (grad_history, grad_future) = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(history, future)

    outputs = {
        "loss": np.asarray(jax.device_get(loss), dtype=np.float32).reshape(()),
        "rec": np.asarray(jax.device_get(terms["rec"]), dtype=np.float32).reshape(()),
        "kl": np.asarray(jax.device_get(terms["kl"]), dtype=np.float32).reshape(()),
        "grad_history": np.asarray(jax.device_get(grad_history), dtype=np.float32),
        "grad_future": np.asarray(jax.device_get(grad_future), dtype=np.float32),
    }
    for name in TEMPORAL_NAMES:
        if name in terms:
            outputs[name] = np.asarray(jax.device_get(terms[name]), dtype=np.float32).reshape(())
    for name in SMPL_JOINT_NAMES:
        if name in terms:
            outputs[name] = np.asarray(jax.device_get(terms[name]), dtype=np.float32).reshape(())
    return outputs


def deterministic_jax_temporal_smpl_outputs(
    history_np: np.ndarray,
    future_pred_np: np.ndarray,
    *,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> ArrayMap:
    import jax
    import jax.numpy as jnp

    from jax_dart.models.temporal_smpl_loss import temporal_smpl_feature_loss

    history = jnp.asarray(history_np, dtype=jnp.float32)
    future_pred = jnp.asarray(future_pred_np, dtype=jnp.float32)
    norm_mean = jnp.asarray(norm_mean_np, dtype=jnp.float32)
    norm_std = jnp.asarray(norm_std_np, dtype=jnp.float32)
    _, terms = temporal_smpl_feature_loss(
        history,
        future_pred,
        feature_slices=feature_slices,
        norm_mean=norm_mean,
        norm_std=norm_std,
        weight_joints_delta=weights["weight_joints_delta"],
        weight_transl_delta=weights["weight_transl_delta"],
        weight_orient_delta=weights["weight_orient_delta"],
    )
    return {
        name: np.asarray(jax.device_get(terms[name]), dtype=np.float32).reshape(())
        for name in TEMPORAL_NAMES
    }


def deterministic_jax_smpl_joint_outputs(
    future_pred_np: np.ndarray,
    future_np: np.ndarray,
    betas_np: np.ndarray,
    *,
    smplx_model,
    feature_slices: Dict[str, Any],
    norm_mean_np: np.ndarray,
    norm_std_np: np.ndarray,
    weights: Dict[str, float],
) -> ArrayMap:
    import jax
    import jax.numpy as jnp

    from jax_dart.models.smplx_joints import smpl_joint_loss_from_motion

    future_pred = jnp.asarray(future_pred_np, dtype=jnp.float32)
    future = jnp.asarray(future_np, dtype=jnp.float32)
    betas = jnp.asarray(betas_np, dtype=jnp.float32)
    norm_mean = jnp.asarray(norm_mean_np, dtype=jnp.float32)
    norm_std = jnp.asarray(norm_std_np, dtype=jnp.float32)
    _, terms = smpl_joint_loss_from_motion(
        smplx_model,
        pred_motion=future_pred,
        gt_motion=future,
        betas=betas,
        feature_slices=feature_slices,
        norm_mean=norm_mean,
        norm_std=norm_std,
        weight_smpl_joints_rec=weights["weight_smpl_joints_rec"],
        weight_joints_consistency=weights["weight_joints_consistency"],
    )
    return {
        name: np.asarray(jax.device_get(terms[name]), dtype=np.float32).reshape(())
        for name in SMPL_JOINT_NAMES
    }


def compare_arrays(
    name: str,
    torch_value: np.ndarray,
    jax_value: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    diff = np.abs(torch_value - jax_value)
    denom = np.maximum(np.abs(torch_value), atol)
    metrics = {
        "shape": torch_value.shape,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "max_rel": float(np.max(diff / denom)),
        "allclose": bool(np.allclose(torch_value, jax_value, atol=atol, rtol=rtol)),
    }
    print(
        f"{name:12s} shape={metrics['shape']} "
        f"max_abs={metrics['max_abs']:.6e} "
        f"mean_abs={metrics['mean_abs']:.6e} "
        f"max_rel={metrics['max_rel']:.6e} "
        f"allclose={metrics['allclose']}"
    )
    return metrics


def _save_snapshot(
    path: str,
    history: np.ndarray,
    future: np.ndarray,
    torch_outputs: ArrayMap,
    jax_outputs: Optional[ArrayMap],
    config: Dict[str, Any],
    torch_state: Optional[ArrayMap] = None,
    torch_grad_outputs: Optional[ArrayMap] = None,
    jax_grad_outputs: Optional[ArrayMap] = None,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    future_betas: Optional[np.ndarray] = None,
) -> None:
    arrays = {
        "config_json": np.array(json.dumps(config, sort_keys=True)),
        "history": history,
        "future": future,
    }
    for key, value in torch_outputs.items():
        arrays[f"torch_{key}"] = value
    if jax_outputs is not None:
        for key, value in jax_outputs.items():
            arrays[f"jax_{key}"] = value
    if torch_grad_outputs is not None:
        for key, value in torch_grad_outputs.items():
            arrays[f"torch_{key}"] = value
    if jax_grad_outputs is not None:
        for key, value in jax_grad_outputs.items():
            arrays[f"jax_{key}"] = value
    if torch_state is not None:
        arrays.update(_prefix_torch_state_np(torch_state))
    if norm_mean is not None:
        arrays["norm_mean"] = norm_mean.astype(np.float32, copy=False)
    if norm_std is not None:
        arrays["norm_std"] = norm_std.astype(np.float32, copy=False)
    if future_betas is not None:
        arrays["future_betas"] = future_betas.astype(np.float32, copy=False)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Torch/JAX DART VAE parity on one exported shard batch.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "torch-export", "jax-compare"],
        help="Run both frameworks, export a Torch snapshot, or compare JAX against a snapshot.",
    )
    parser.add_argument("--root", default=None, help="Export root containing metadata.json and shards/.")
    parser.add_argument("--batch-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional DART VAE .pt checkpoint to load first.")
    parser.add_argument("--snapshot", default=None, help="Torch-exported snapshot for --mode jax-compare.")
    parser.add_argument("--h-dim", type=int, default=256)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=_parse_latent_dim, default=(1, 256))
    parser.add_argument("--arch", default="all_encoder", choices=["all_encoder", "encoder_decoder"])
    parser.add_argument("--activation", default="gelu", choices=["gelu", "relu", "glu"])
    parser.add_argument("--position-embedding", default="learned", choices=["learned", "v3", "sine", "v2"])
    parser.add_argument("--normalize-before", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=3e-3)
    parser.add_argument("--grad-rtol", type=float, default=3e-3)
    parser.add_argument("--weight-rec", type=float, default=1.0)
    parser.add_argument("--weight-kl", type=float, default=1e-4)
    parser.add_argument("--weight-joints-delta", type=float, default=0.0)
    parser.add_argument("--weight-transl-delta", type=float, default=0.0)
    parser.add_argument("--weight-orient-delta", type=float, default=0.0)
    parser.add_argument("--weight-smpl-joints-rec", type=float, default=0.0)
    parser.add_argument("--weight-joints-consistency", type=float, default=0.0)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument("--smpl-gender", default="male", choices=["male"])
    parser.add_argument("--include-grads", action="store_true")
    parser.add_argument("--include-temporal-smpl-loss", action="store_true")
    parser.add_argument("--include-smpl-joint-loss", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--out-npz", default=None, help="Optional path for saving snapshot/output arrays.")
    return parser.parse_args()


def _require(value: Optional[str], message: str) -> str:
    if value is None:
        raise SystemExit(message)
    return value


def _load_batch_with_metadata(
    root: str,
    batch_samples: int,
) -> Tuple[dict, int, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    metadata = load_metadata(root)
    feature_dim = int(metadata["primitive"]["feature_dim"])
    batch = next(iter_vae_batches(root, batch_samples=batch_samples))
    history = batch["history"].astype(np.float32, copy=False)
    future = batch["future"].astype(np.float32, copy=False)
    return metadata, feature_dim, history, future, batch


def _load_batch(root: str, batch_samples: int) -> Tuple[int, np.ndarray, np.ndarray]:
    _, feature_dim, history, future, _ = _load_batch_with_metadata(root, batch_samples)
    return feature_dim, history, future


def _compare_outputs(torch_outputs: ArrayMap, jax_outputs: ArrayMap, atol: float, rtol: float) -> Dict[str, Any]:
    compared = {}
    for name in FORWARD_NAMES:
        compared[name] = compare_arrays(
            name,
            torch_outputs[name],
            jax_outputs[name],
            atol=atol,
            rtol=rtol,
        )
    return compared


def _load_torch_forward_outputs(snapshot: Dict[str, np.ndarray]) -> ArrayMap:
    return {
        name: snapshot[f"torch_{name}"].astype(np.float32, copy=False)
        for name in FORWARD_NAMES
    }


def _load_torch_grad_outputs(snapshot: Dict[str, np.ndarray]) -> Optional[ArrayMap]:
    if f"torch_{GRAD_NAMES[0]}" not in snapshot:
        return None
    return {
        name: snapshot[f"torch_{name}"].astype(np.float32, copy=False)
        for name in GRAD_NAMES
    }


def _load_torch_temporal_outputs(snapshot: Dict[str, np.ndarray]) -> Optional[ArrayMap]:
    if f"torch_{TEMPORAL_NAMES[0]}" not in snapshot:
        return None
    return {
        name: snapshot[f"torch_{name}"].astype(np.float32, copy=False)
        for name in TEMPORAL_NAMES
    }


def _load_torch_smpl_joint_outputs(snapshot: Dict[str, np.ndarray]) -> Optional[ArrayMap]:
    if f"torch_{SMPL_JOINT_NAMES[0]}" not in snapshot:
        return None
    return {
        name: snapshot[f"torch_{name}"].astype(np.float32, copy=False)
        for name in SMPL_JOINT_NAMES
    }


def _compare_grad_outputs(
    torch_grad_outputs: ArrayMap,
    jax_grad_outputs: ArrayMap,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    compared = {}
    for name in GRAD_NAMES:
        compared[name] = compare_arrays(
            name,
            torch_grad_outputs[name],
            jax_grad_outputs[name],
            atol=atol,
            rtol=rtol,
        )
    return compared


def _compare_temporal_outputs(
    torch_outputs: ArrayMap,
    jax_outputs: ArrayMap,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    compared = {}
    for name in TEMPORAL_NAMES:
        compared[name] = compare_arrays(
            name,
            torch_outputs[name],
            jax_outputs[name],
            atol=atol,
            rtol=rtol,
        )
    return compared


def _compare_smpl_joint_outputs(
    torch_outputs: ArrayMap,
    jax_outputs: ArrayMap,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    compared = {}
    for name in SMPL_JOINT_NAMES:
        compared[name] = compare_arrays(
            name,
            torch_outputs[name],
            jax_outputs[name],
            atol=atol,
            rtol=rtol,
        )
    return compared


def _maybe_fail(compared: Dict[str, Any]) -> None:
    failed = [name for name, metrics in compared.items() if not metrics["allclose"]]
    if failed:
        joined = ", ".join(failed)
        raise SystemExit(f"parity failed for: {joined}")


def run_torch_export(args: argparse.Namespace) -> None:
    import torch

    from model.mld_vae import AutoMldVae as TorchAutoMldVae

    root = _require(args.root, "--root is required for --mode torch-export")
    out_npz = _require(args.out_npz, "--out-npz is required for --mode torch-export")
    metadata, feature_dim, history, future, batch = _load_batch_with_metadata(root, args.batch_samples)
    config = _config_from_args(args, feature_dim, metadata=metadata)
    norm_mean_np = None
    norm_std_np = None
    temporal_weights = _temporal_weights_from_config(config)
    smpl_joint_weights = _smpl_joint_weights_from_config(config)
    future_betas = None
    if args.include_smpl_joint_loss:
        _require(args.smpl_model_dir, "--smpl-model-dir is required for --include-smpl-joint-loss")
        future_betas = _future_betas_from_batch(metadata, batch)
    if args.include_temporal_smpl_loss or args.include_smpl_joint_loss:
        norm_mean_np, norm_std_np = _load_motion_normalization_np(root)

    torch.manual_seed(args.seed)
    torch_device = args.torch_device
    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_model = TorchAutoMldVae(
        nfeats=feature_dim,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        arch=args.arch,
        normalize_before=args.normalize_before,
        activation=args.activation,
        position_embedding=args.position_embedding,
    ).to(torch_device)
    if args.checkpoint is not None:
        load_torch_checkpoint(torch_model, args.checkpoint, torch_device)

    torch_outputs = deterministic_torch_forward(torch_model, history, future, torch_device)
    if args.include_temporal_smpl_loss:
        torch_outputs.update(
            deterministic_torch_temporal_smpl_outputs(
                history,
                torch_outputs["future_pred"],
                torch_device,
                feature_slices=config["feature_slices"],
                norm_mean_np=norm_mean_np,
                norm_std_np=norm_std_np,
                weights=temporal_weights,
            )
        )
    if args.include_smpl_joint_loss:
        torch_outputs.update(
            deterministic_torch_smpl_joint_outputs(
                torch_outputs["future_pred"],
                future,
                future_betas,
                torch_device,
                smpl_model_dir=args.smpl_model_dir,
                smpl_gender=args.smpl_gender,
                feature_slices=config["feature_slices"],
                norm_mean_np=norm_mean_np,
                norm_std_np=norm_std_np,
                weights=smpl_joint_weights,
            )
        )
    torch_grad_outputs = None
    if args.include_grads:
        torch_grad_outputs = deterministic_torch_loss_and_input_grads(
            torch_model,
            history,
            future,
            torch_device,
            weight_rec=args.weight_rec,
            weight_kl=args.weight_kl,
            feature_slices=config["feature_slices"] if (args.include_temporal_smpl_loss or args.include_smpl_joint_loss) else None,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            temporal_weights=temporal_weights if args.include_temporal_smpl_loss else None,
            betas_np=future_betas,
            smpl_model_dir=args.smpl_model_dir if args.include_smpl_joint_loss else None,
            smpl_gender=args.smpl_gender,
            smpl_joint_weights=smpl_joint_weights if args.include_smpl_joint_loss else None,
        )
    _save_snapshot(
        out_npz,
        history,
        future,
        torch_outputs,
        jax_outputs=None,
        config=config,
        torch_state=_collect_torch_state_np(torch_model),
        torch_grad_outputs=torch_grad_outputs,
        norm_mean=norm_mean_np,
        norm_std=norm_std_np,
        future_betas=future_betas,
    )

    print("torch_device:", torch_device)
    print("history:", history.shape, history.dtype)
    print("future:", future.shape, future.dtype)
    for name, value in torch_outputs.items():
        print(f"torch_{name}:", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    if torch_grad_outputs is not None:
        for name, value in torch_grad_outputs.items():
            if name in torch_outputs:
                continue
            print(f"torch_{name}:", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    print("wrote:", out_npz)


def run_jax_compare(args: argparse.Namespace) -> None:
    import jax
    from flax import nnx

    from jax_dart.models.mld_vae import AutoMldVae as JaxAutoMldVae

    snapshot_path = _require(args.snapshot, "--snapshot is required for --mode jax-compare")
    snapshot = _load_snapshot(snapshot_path)
    config = _load_config(snapshot)
    history = snapshot["history"].astype(np.float32, copy=False)
    future = snapshot["future"].astype(np.float32, copy=False)
    torch_outputs = _load_torch_forward_outputs(snapshot)
    torch_grad_outputs = _load_torch_grad_outputs(snapshot)
    include_temporal = bool(args.include_temporal_smpl_loss or config.get("include_temporal_smpl_loss", False))
    include_smpl_joint = bool(args.include_smpl_joint_loss or config.get("include_smpl_joint_loss", False))
    torch_temporal_outputs = _load_torch_temporal_outputs(snapshot)
    torch_smpl_joint_outputs = _load_torch_smpl_joint_outputs(snapshot)
    norm_mean_np = None
    norm_std_np = None
    temporal_weights = _temporal_weights_from_config(config)
    smpl_joint_weights = _smpl_joint_weights_from_config(config)
    future_betas = None
    smpl_model_dir = args.smpl_model_dir or config.get("smpl_model_dir")
    if include_temporal or include_smpl_joint:
        if "feature_slices" not in config:
            raise SystemExit(f"{snapshot_path} does not contain feature slices.")
        if "norm_mean" not in snapshot or "norm_std" not in snapshot:
            raise SystemExit(f"{snapshot_path} does not contain normalization arrays.")
        norm_mean_np = snapshot["norm_mean"].astype(np.float32, copy=False)
        norm_std_np = snapshot["norm_std"].astype(np.float32, copy=False)
    if include_temporal:
        if torch_temporal_outputs is None:
            raise SystemExit(f"{snapshot_path} does not contain Torch temporal SMPL outputs.")
    if include_smpl_joint:
        if torch_smpl_joint_outputs is None:
            raise SystemExit(f"{snapshot_path} does not contain Torch SMPL joint loss outputs.")
        if "future_betas" not in snapshot:
            raise SystemExit(f"{snapshot_path} does not contain future betas.")
        future_betas = snapshot["future_betas"].astype(np.float32, copy=False)
        smpl_model_dir = _require(smpl_model_dir, "--smpl-model-dir is required unless the snapshot stores it.")
    if args.include_grads and torch_grad_outputs is None:
        raise SystemExit(f"{snapshot_path} does not contain Torch gradient outputs.")
    torch_state = _load_torch_state_np(snapshot)
    if not torch_state:
        raise SystemExit(f"{snapshot_path} does not contain Torch state arrays.")

    jax_model = JaxAutoMldVae(
        nfeats=int(config["feature_dim"]),
        latent_dim=tuple(config["latent_dim"]),
        h_dim=int(config["h_dim"]),
        ff_size=int(config["ff_size"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        arch=config["arch"],
        normalize_before=bool(config["normalize_before"]),
        activation=config["activation"],
        position_embedding=config["position_embedding"],
        rngs=nnx.Rngs(int(config["seed"])),
    )
    copy_torch_state_to_jax(
        torch_state,
        jax_model,
        num_layers=int(config["num_layers"]),
        arch=config["arch"],
    )
    smplx_model = None
    if include_smpl_joint:
        from jax_dart.models.smplx_joints import load_smplx_joints_model

        smplx_model = load_smplx_joints_model(
            smpl_model_dir,
            gender=config.get("smpl_gender", args.smpl_gender),
        )

    jax_outputs = deterministic_jax_forward(jax_model, history, future)
    jax_temporal_outputs = None
    if include_temporal:
        jax_temporal_outputs = deterministic_jax_temporal_smpl_outputs(
            history,
            jax_outputs["future_pred"],
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=temporal_weights,
        )
    jax_smpl_joint_outputs = None
    if include_smpl_joint:
        jax_smpl_joint_outputs = deterministic_jax_smpl_joint_outputs(
            jax_outputs["future_pred"],
            future,
            future_betas,
            smplx_model=smplx_model,
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=smpl_joint_weights,
        )

    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
    print("history:", history.shape, history.dtype)
    print("future:", future.shape, future.dtype)
    compared = _compare_outputs(torch_outputs, jax_outputs, args.atol, args.rtol)
    compared_temporal = {}
    if include_temporal:
        compared_temporal = _compare_temporal_outputs(
            torch_temporal_outputs,
            jax_temporal_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )
    compared_smpl = {}
    if include_smpl_joint:
        compared_smpl = _compare_smpl_joint_outputs(
            torch_smpl_joint_outputs,
            jax_smpl_joint_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )
    compared_grads = {}
    jax_grad_outputs = None
    if args.include_grads or torch_grad_outputs is not None:
        jax_grad_outputs = deterministic_jax_loss_and_input_grads(
            jax_model,
            history,
            future,
            weight_rec=float(config.get("weight_rec", args.weight_rec)),
            weight_kl=float(config.get("weight_kl", args.weight_kl)),
            feature_slices=config["feature_slices"] if (include_temporal or include_smpl_joint) else None,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            temporal_weights=temporal_weights if include_temporal else None,
            betas_np=future_betas,
            smplx_model=smplx_model,
            smpl_joint_weights=smpl_joint_weights if include_smpl_joint else None,
        )
        compared_grads = _compare_grad_outputs(
            torch_grad_outputs,
            jax_grad_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )

    if args.out_npz is not None:
        torch_save_outputs = dict(torch_outputs)
        jax_save_outputs = dict(jax_outputs)
        if include_temporal:
            torch_save_outputs.update(torch_temporal_outputs)
            jax_save_outputs.update(jax_temporal_outputs)
        if include_smpl_joint:
            torch_save_outputs.update(torch_smpl_joint_outputs)
            jax_save_outputs.update(jax_smpl_joint_outputs)
        _save_snapshot(
            args.out_npz,
            history,
            future,
            torch_save_outputs,
            jax_save_outputs,
            config=config,
            torch_state=torch_state,
            torch_grad_outputs=torch_grad_outputs,
            jax_grad_outputs=jax_grad_outputs,
            norm_mean=norm_mean_np,
            norm_std=norm_std_np,
            future_betas=future_betas,
        )
        print("wrote:", args.out_npz)

    if args.fail_on_mismatch:
        _maybe_fail(compared)
        _maybe_fail(compared_temporal)
        _maybe_fail(compared_smpl)
        _maybe_fail(compared_grads)


def run_both(args: argparse.Namespace) -> None:
    import jax
    import torch
    from flax import nnx

    from jax_dart.models.mld_vae import AutoMldVae as JaxAutoMldVae
    from model.mld_vae import AutoMldVae as TorchAutoMldVae

    root = _require(args.root, "--root is required for --mode both")
    metadata, feature_dim, history, future, batch = _load_batch_with_metadata(root, args.batch_samples)
    config = _config_from_args(args, feature_dim, metadata=metadata)
    norm_mean_np = None
    norm_std_np = None
    temporal_weights = _temporal_weights_from_config(config)
    smpl_joint_weights = _smpl_joint_weights_from_config(config)
    future_betas = None
    if args.include_smpl_joint_loss:
        _require(args.smpl_model_dir, "--smpl-model-dir is required for --include-smpl-joint-loss")
        future_betas = _future_betas_from_batch(metadata, batch)
    if args.include_temporal_smpl_loss or args.include_smpl_joint_loss:
        norm_mean_np, norm_std_np = _load_motion_normalization_np(root)

    torch.manual_seed(args.seed)
    torch_device = args.torch_device
    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_model = TorchAutoMldVae(
        nfeats=feature_dim,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        arch=args.arch,
        normalize_before=args.normalize_before,
        activation=args.activation,
        position_embedding=args.position_embedding,
    ).to(torch_device)
    if args.checkpoint is not None:
        load_torch_checkpoint(torch_model, args.checkpoint, torch_device)

    jax_model = JaxAutoMldVae(
        nfeats=feature_dim,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        arch=args.arch,
        normalize_before=args.normalize_before,
        activation=args.activation,
        position_embedding=args.position_embedding,
        rngs=nnx.Rngs(args.seed),
    )
    copy_torch_vae_to_jax(torch_model, jax_model, num_layers=args.num_layers, arch=args.arch)
    smplx_model = None
    if args.include_smpl_joint_loss:
        from jax_dart.models.smplx_joints import load_smplx_joints_model

        smplx_model = load_smplx_joints_model(args.smpl_model_dir, gender=args.smpl_gender)

    torch_outputs = deterministic_torch_forward(torch_model, history, future, torch_device)
    jax_outputs = deterministic_jax_forward(jax_model, history, future)
    torch_temporal_outputs = None
    jax_temporal_outputs = None
    if args.include_temporal_smpl_loss:
        torch_temporal_outputs = deterministic_torch_temporal_smpl_outputs(
            history,
            torch_outputs["future_pred"],
            torch_device,
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=temporal_weights,
        )
        jax_temporal_outputs = deterministic_jax_temporal_smpl_outputs(
            history,
            jax_outputs["future_pred"],
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=temporal_weights,
        )
    torch_smpl_joint_outputs = None
    jax_smpl_joint_outputs = None
    if args.include_smpl_joint_loss:
        torch_smpl_joint_outputs = deterministic_torch_smpl_joint_outputs(
            torch_outputs["future_pred"],
            future,
            future_betas,
            torch_device,
            smpl_model_dir=args.smpl_model_dir,
            smpl_gender=args.smpl_gender,
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=smpl_joint_weights,
        )
        jax_smpl_joint_outputs = deterministic_jax_smpl_joint_outputs(
            jax_outputs["future_pred"],
            future,
            future_betas,
            smplx_model=smplx_model,
            feature_slices=config["feature_slices"],
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            weights=smpl_joint_weights,
        )
    torch_grad_outputs = None
    jax_grad_outputs = None
    if args.include_grads:
        torch_grad_outputs = deterministic_torch_loss_and_input_grads(
            torch_model,
            history,
            future,
            torch_device,
            weight_rec=args.weight_rec,
            weight_kl=args.weight_kl,
            feature_slices=config["feature_slices"] if (args.include_temporal_smpl_loss or args.include_smpl_joint_loss) else None,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            temporal_weights=temporal_weights if args.include_temporal_smpl_loss else None,
            betas_np=future_betas,
            smpl_model_dir=args.smpl_model_dir if args.include_smpl_joint_loss else None,
            smpl_gender=args.smpl_gender,
            smpl_joint_weights=smpl_joint_weights if args.include_smpl_joint_loss else None,
        )
        jax_grad_outputs = deterministic_jax_loss_and_input_grads(
            jax_model,
            history,
            future,
            weight_rec=args.weight_rec,
            weight_kl=args.weight_kl,
            feature_slices=config["feature_slices"] if (args.include_temporal_smpl_loss or args.include_smpl_joint_loss) else None,
            norm_mean_np=norm_mean_np,
            norm_std_np=norm_std_np,
            temporal_weights=temporal_weights if args.include_temporal_smpl_loss else None,
            betas_np=future_betas,
            smplx_model=smplx_model,
            smpl_joint_weights=smpl_joint_weights if args.include_smpl_joint_loss else None,
        )

    print("torch_device:", torch_device)
    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
    print("history:", history.shape, history.dtype)
    print("future:", future.shape, future.dtype)
    compared = _compare_outputs(torch_outputs, jax_outputs, args.atol, args.rtol)
    compared_temporal = {}
    if args.include_temporal_smpl_loss:
        compared_temporal = _compare_temporal_outputs(
            torch_temporal_outputs,
            jax_temporal_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )
    compared_smpl = {}
    if args.include_smpl_joint_loss:
        compared_smpl = _compare_smpl_joint_outputs(
            torch_smpl_joint_outputs,
            jax_smpl_joint_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )
    compared_grads = {}
    if args.include_grads:
        compared_grads = _compare_grad_outputs(
            torch_grad_outputs,
            jax_grad_outputs,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )

    if args.out_npz is not None:
        torch_save_outputs = dict(torch_outputs)
        jax_save_outputs = dict(jax_outputs)
        if args.include_temporal_smpl_loss:
            torch_save_outputs.update(torch_temporal_outputs)
            jax_save_outputs.update(jax_temporal_outputs)
        if args.include_smpl_joint_loss:
            torch_save_outputs.update(torch_smpl_joint_outputs)
            jax_save_outputs.update(jax_smpl_joint_outputs)
        _save_snapshot(
            args.out_npz,
            history,
            future,
            torch_save_outputs,
            jax_save_outputs,
            config=config,
            torch_state=_collect_torch_state_np(torch_model),
            torch_grad_outputs=torch_grad_outputs,
            jax_grad_outputs=jax_grad_outputs,
            norm_mean=norm_mean_np,
            norm_std=norm_std_np,
            future_betas=future_betas,
        )
        print("wrote:", args.out_npz)

    if args.fail_on_mismatch:
        _maybe_fail(compared)
        _maybe_fail(compared_temporal)
        _maybe_fail(compared_smpl)
        _maybe_fail(compared_grads)


def main() -> None:
    args = parse_args()
    if args.mode == "torch-export":
        run_torch_export(args)
    elif args.mode == "jax-compare":
        run_jax_compare(args)
    else:
        run_both(args)


if __name__ == "__main__":
    main()
