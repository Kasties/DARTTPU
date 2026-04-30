"""Compare Torch and JAX DART DenoiserMLP plus diffusion ``q_sample``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata


ArrayMap = Dict[str, np.ndarray]
FORWARD_NAMES = ("x_t", "denoised")


def _parse_shape(value: str) -> Tuple[int, ...]:
    parts = [int(part) for part in value.replace(",", " ").split()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one integer")
    return tuple(parts)


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


def copy_torch_denoiser_to_jax(torch_model, jax_model) -> None:
    copy_torch_state_to_jax(torch_model.state_dict(), jax_model)


def copy_torch_state_to_jax(torch_state: Dict[str, Any], jax_model) -> None:
    """Copy Torch ``model.mld_denoiser.DenoiserMLP`` state into the JAX port."""
    pe_key = "sequence_pos_encoder.pe"
    if pe_key not in torch_state:
        pe_key = "embed_timestep.sequence_pos_encoder.pe"
    _assign_nnx_value(jax_model.sequence_pos_encoder.pe, _as_numpy(torch_state[pe_key]))
    _copy_linear(torch_state, "embed_timestep.time_embed.0", jax_model.embed_timestep.time_embed[0])
    _copy_linear(torch_state, "embed_timestep.time_embed.2", jax_model.embed_timestep.time_embed[1])
    _copy_linear(torch_state, "input_project", jax_model.input_project)
    for block_index, block in enumerate(jax_model.mlp.layers):
        for layer_index, linear in enumerate(block.layers):
            _copy_linear(
                torch_state,
                f"mlp.layers.{block_index}.layers.{layer_index}",
                linear,
            )
    _copy_linear(torch_state, "mlp.out_fc", jax_model.mlp.out_fc)


def _extract_model_state(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def load_torch_checkpoint(torch_model, checkpoint_path: str, device: str) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    torch_model.load_state_dict(_extract_model_state(checkpoint))


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


def _make_synthetic_batch(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed + 1000)
    batch_size = args.batch_size or 8
    history = rng.standard_normal(
        (batch_size, *args.history_shape),
        dtype=np.float32,
    )
    text = rng.standard_normal((batch_size, args.clip_dim), dtype=np.float32)
    return history, text


def _load_batch(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    if args.root is None:
        history, text = _make_synthetic_batch(args)
        return history, text, None
    metadata = load_metadata(args.root)
    batch = next(iter_vae_batches(args.root, batch_samples=args.batch_samples))
    history = batch["history"].astype(np.float32, copy=False)
    if "text_embedding" in batch:
        text = batch["text_embedding"].astype(np.float32, copy=False)
    else:
        rng = np.random.default_rng(args.seed + 1000)
        text = rng.standard_normal((history.shape[0], args.clip_dim), dtype=np.float32)
    if args.batch_size is not None:
        history = history[: args.batch_size]
        text = text[: args.batch_size]
    return history, text, metadata


def _config_from_args(
    args: argparse.Namespace,
    history: np.ndarray,
    text: np.ndarray,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    config = {
        "h_dim": int(args.h_dim),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "activation": args.activation,
        "cond_mask_prob": float(args.cond_mask_prob),
        "clip_dim": int(text.shape[-1]),
        "history_shape": [int(x) for x in history.shape[1:]],
        "noise_shape": [int(x) for x in args.noise_shape],
        "seed": int(args.seed),
        "diffusion_steps": int(args.diffusion_steps),
        "noise_schedule": args.noise_schedule,
        "scale_betas": float(args.scale_betas),
        "rescale_timesteps": bool(args.rescale_timesteps),
        "uncond": bool(args.uncond),
    }
    if metadata is not None:
        config["metadata_format"] = metadata.get("format")
    return config


def _make_noising_inputs(config: Dict[str, Any], batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(config["seed"]) + 2000)
    noise_shape = tuple(config["noise_shape"])
    x_start = rng.standard_normal((batch_size, *noise_shape), dtype=np.float32)
    noise = rng.standard_normal((batch_size, *noise_shape), dtype=np.float32)
    timesteps = rng.integers(
        0,
        int(config["diffusion_steps"]),
        size=(batch_size,),
        dtype=np.int64,
    )
    return x_start, noise, timesteps


def _torch_q_sample(
    x_start_np: np.ndarray,
    noise_np: np.ndarray,
    timesteps_np: np.ndarray,
    device: str,
    config: Dict[str, Any],
) -> np.ndarray:
    import torch

    from jax_dart.diffusion import get_named_beta_schedule

    betas = get_named_beta_schedule(
        config["noise_schedule"],
        int(config["diffusion_steps"]),
        float(config["scale_betas"]),
    )
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    x_start = torch.from_numpy(x_start_np).to(device=device, dtype=torch.float32)
    noise = torch.from_numpy(noise_np).to(device=device, dtype=torch.float32)
    timesteps = torch.from_numpy(timesteps_np).to(device=device, dtype=torch.long)

    def extract(arr):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(x_start.shape):
            res = res[..., None]
        return res.expand(x_start.shape)

    with torch.no_grad():
        x_t = (
            extract(sqrt_alphas_cumprod) * x_start
            + extract(sqrt_one_minus_alphas_cumprod) * noise
        )
    return _torch_numpy(x_t)


def deterministic_torch_forward(
    torch_model,
    history_np: np.ndarray,
    text_np: np.ndarray,
    x_start_np: np.ndarray,
    noise_np: np.ndarray,
    timesteps_np: np.ndarray,
    device: str,
    config: Dict[str, Any],
) -> ArrayMap:
    import torch

    torch_model.eval()
    x_t_np = _torch_q_sample(x_start_np, noise_np, timesteps_np, device, config)
    x_t = torch.from_numpy(x_t_np).to(device=device, dtype=torch.float32)
    timesteps = torch.from_numpy(timesteps_np).to(device=device, dtype=torch.long)
    model_timesteps = timesteps
    if config.get("rescale_timesteps", False):
        model_timesteps = timesteps.float() * (
            1000.0 / float(config["diffusion_steps"])
        )
    history = torch.from_numpy(history_np).to(device=device, dtype=torch.float32)
    text = torch.from_numpy(text_np).to(device=device, dtype=torch.float32)
    y = {
        "history_motion_normalized": history,
        "text_embedding": text,
    }
    if config.get("uncond", False):
        y["uncond"] = True
    with torch.no_grad():
        denoised = torch_model(x_t=x_t, timesteps=model_timesteps, y=y)
    return {
        "x_t": x_t_np,
        "denoised": _torch_numpy(denoised),
    }


def deterministic_jax_forward(
    jax_model,
    history_np: np.ndarray,
    text_np: np.ndarray,
    x_start_np: np.ndarray,
    noise_np: np.ndarray,
    timesteps_np: np.ndarray,
    config: Dict[str, Any],
) -> ArrayMap:
    import jax
    import jax.numpy as jnp

    from jax_dart.diffusion import create_gaussian_diffusion

    diffusion = create_gaussian_diffusion(
        diffusion_steps=int(config["diffusion_steps"]),
        noise_schedule=config["noise_schedule"],
        scale_betas=float(config["scale_betas"]),
        rescale_timesteps=bool(config["rescale_timesteps"]),
    )
    x_start = jnp.asarray(x_start_np, dtype=jnp.float32)
    noise = jnp.asarray(noise_np, dtype=jnp.float32)
    timesteps = jnp.asarray(timesteps_np, dtype=jnp.int32)
    history = jnp.asarray(history_np, dtype=jnp.float32)
    text = jnp.asarray(text_np, dtype=jnp.float32)
    x_t = diffusion.q_sample(x_start=x_start, t=timesteps, noise=noise)
    y = {
        "history_motion_normalized": history,
        "text_embedding": text,
    }
    if config.get("uncond", False):
        y["uncond"] = True
    denoised = jax_model(x_t, diffusion.scale_timesteps(timesteps), y)
    return {
        "x_t": np.asarray(jax.device_get(x_t), dtype=np.float32),
        "denoised": np.asarray(jax.device_get(denoised), dtype=np.float32),
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


def _compare_outputs(
    torch_outputs: ArrayMap,
    jax_outputs: ArrayMap,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    return {
        name: compare_arrays(name, torch_outputs[name], jax_outputs[name], atol, rtol)
        for name in FORWARD_NAMES
    }


def _load_torch_forward_outputs(snapshot: Dict[str, np.ndarray]) -> ArrayMap:
    return {
        name: snapshot[f"torch_{name}"].astype(np.float32, copy=False)
        for name in FORWARD_NAMES
    }


def _save_snapshot(
    path: str,
    history: np.ndarray,
    text: np.ndarray,
    x_start: np.ndarray,
    noise: np.ndarray,
    timesteps: np.ndarray,
    torch_outputs: ArrayMap,
    jax_outputs: Optional[ArrayMap],
    config: Dict[str, Any],
    torch_state: Optional[ArrayMap] = None,
) -> None:
    arrays = {
        "config_json": np.array(json.dumps(config, sort_keys=True)),
        "history": history,
        "text_embedding": text,
        "x_start": x_start,
        "noise": noise,
        "timesteps": timesteps,
    }
    for key, value in torch_outputs.items():
        arrays[f"torch_{key}"] = value
    if jax_outputs is not None:
        for key, value in jax_outputs.items():
            arrays[f"jax_{key}"] = value
    if torch_state is not None:
        arrays.update(_prefix_torch_state_np(torch_state))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def _maybe_fail(compared: Dict[str, Any]) -> None:
    failed = [name for name, metrics in compared.items() if not metrics["allclose"]]
    if failed:
        joined = ", ".join(failed)
        raise SystemExit(f"parity failed for: {joined}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Torch/JAX DART DenoiserMLP parity on one batch."
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "torch-export", "jax-compare"],
        help="Run both frameworks, export a Torch snapshot, or compare JAX against a snapshot.",
    )
    parser.add_argument("--root", default=None, help="Optional export root containing metadata.json and shards/.")
    parser.add_argument("--batch-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional DART DenoiserMLP .pt checkpoint to load.")
    parser.add_argument("--snapshot", default=None, help="Torch-exported snapshot for --mode jax-compare.")
    parser.add_argument("--h-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu", choices=["tanh", "relu", "sigmoid", "gelu", "lrelu"])
    parser.add_argument("--cond-mask-prob", type=float, default=0.0)
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--history-shape", type=_parse_shape, default=(2, 276))
    parser.add_argument("--noise-shape", type=_parse_shape, default=(1, 128))
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--noise-schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--scale-betas", type=float, default=1.0)
    parser.add_argument("--rescale-timesteps", action="store_true")
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow Torch TF32 matmuls. Disabled by default for tighter parity.",
    )
    parser.add_argument(
        "--jax-matmul-precision",
        default="highest",
        choices=["default", "high", "highest"],
        help="JAX matmul precision for compare modes.",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--out-npz", default=None, help="Optional path for saving snapshot/output arrays.")
    return parser.parse_args()


def _require(value: Optional[str], message: str) -> str:
    if value is None:
        raise SystemExit(message)
    return value


def _make_torch_model(config: Dict[str, Any], device: str, checkpoint: Optional[str] = None):
    from model.mld_denoiser import DenoiserMLP as TorchDenoiserMLP

    torch_model = TorchDenoiserMLP(
        h_dim=int(config["h_dim"]),
        n_blocks=int(config["n_blocks"]),
        dropout=float(config["dropout"]),
        activation=config["activation"],
        clip_dim=int(config["clip_dim"]),
        history_shape=tuple(config["history_shape"]),
        noise_shape=tuple(config["noise_shape"]),
        cond_mask_prob=float(config["cond_mask_prob"]),
    ).to(device)
    if checkpoint is not None:
        load_torch_checkpoint(torch_model, checkpoint, device)
    return torch_model


def _configure_torch_precision(torch, *, allow_tf32: bool) -> None:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _configure_jax_precision(precision: str) -> None:
    if precision == "default":
        return
    import jax

    jax.config.update("jax_default_matmul_precision", precision)


def _make_jax_model(config: Dict[str, Any]):
    from flax import nnx

    from jax_dart.models.mld_denoiser import DenoiserMLP as JaxDenoiserMLP

    return JaxDenoiserMLP(
        h_dim=int(config["h_dim"]),
        n_blocks=int(config["n_blocks"]),
        dropout=float(config["dropout"]),
        activation=config["activation"],
        clip_dim=int(config["clip_dim"]),
        history_shape=tuple(config["history_shape"]),
        noise_shape=tuple(config["noise_shape"]),
        cond_mask_prob=float(config["cond_mask_prob"]),
        rngs=nnx.Rngs(int(config["seed"])),
    )


def run_torch_export(args: argparse.Namespace) -> None:
    import torch

    _configure_torch_precision(torch, allow_tf32=args.allow_tf32)
    out_npz = _require(args.out_npz, "--out-npz is required for --mode torch-export")
    history, text, metadata = _load_batch(args)
    config = _config_from_args(args, history, text, metadata=metadata)
    x_start, noise, timesteps = _make_noising_inputs(config, history.shape[0])
    torch.manual_seed(args.seed)
    torch_device = args.torch_device
    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _make_torch_model(config, torch_device, checkpoint=args.checkpoint)
    torch_outputs = deterministic_torch_forward(
        torch_model,
        history,
        text,
        x_start,
        noise,
        timesteps,
        torch_device,
        config,
    )
    _save_snapshot(
        out_npz,
        history,
        text,
        x_start,
        noise,
        timesteps,
        torch_outputs,
        jax_outputs=None,
        config=config,
        torch_state=_collect_torch_state_np(torch_model),
    )
    print("torch_device:", torch_device)
    print("history:", history.shape, history.dtype)
    print("text_embedding:", text.shape, text.dtype)
    print("x_start:", x_start.shape, x_start.dtype)
    for name, value in torch_outputs.items():
        print(f"torch_{name}:", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    print("wrote:", out_npz)


def run_jax_compare(args: argparse.Namespace) -> None:
    _configure_jax_precision(args.jax_matmul_precision)
    import jax

    snapshot_path = _require(args.snapshot, "--snapshot is required for --mode jax-compare")
    snapshot = _load_snapshot(snapshot_path)
    config = _load_config(snapshot)
    history = snapshot["history"].astype(np.float32, copy=False)
    text = snapshot["text_embedding"].astype(np.float32, copy=False)
    x_start = snapshot["x_start"].astype(np.float32, copy=False)
    noise = snapshot["noise"].astype(np.float32, copy=False)
    timesteps = snapshot["timesteps"].astype(np.int64, copy=False)
    torch_outputs = _load_torch_forward_outputs(snapshot)
    torch_state = _load_torch_state_np(snapshot)
    if not torch_state:
        raise SystemExit(f"{snapshot_path} does not contain Torch state arrays.")
    jax_model = _make_jax_model(config)
    copy_torch_state_to_jax(torch_state, jax_model)
    jax_outputs = deterministic_jax_forward(
        jax_model,
        history,
        text,
        x_start,
        noise,
        timesteps,
        config,
    )
    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
    print("history:", history.shape, history.dtype)
    print("text_embedding:", text.shape, text.dtype)
    compared = _compare_outputs(torch_outputs, jax_outputs, args.atol, args.rtol)
    if args.out_npz is not None:
        _save_snapshot(
            args.out_npz,
            history,
            text,
            x_start,
            noise,
            timesteps,
            torch_outputs,
            jax_outputs,
            config=config,
            torch_state=torch_state,
        )
        print("wrote:", args.out_npz)
    if args.fail_on_mismatch:
        _maybe_fail(compared)


def run_both(args: argparse.Namespace) -> None:
    import torch

    _configure_torch_precision(torch, allow_tf32=args.allow_tf32)
    _configure_jax_precision(args.jax_matmul_precision)
    import jax

    history, text, metadata = _load_batch(args)
    config = _config_from_args(args, history, text, metadata=metadata)
    x_start, noise, timesteps = _make_noising_inputs(config, history.shape[0])
    torch.manual_seed(args.seed)
    torch_device = args.torch_device
    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _make_torch_model(config, torch_device, checkpoint=args.checkpoint)
    jax_model = _make_jax_model(config)
    copy_torch_denoiser_to_jax(torch_model, jax_model)
    torch_outputs = deterministic_torch_forward(
        torch_model,
        history,
        text,
        x_start,
        noise,
        timesteps,
        torch_device,
        config,
    )
    jax_outputs = deterministic_jax_forward(
        jax_model,
        history,
        text,
        x_start,
        noise,
        timesteps,
        config,
    )
    print("torch_device:", torch_device)
    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
    print("history:", history.shape, history.dtype)
    print("text_embedding:", text.shape, text.dtype)
    compared = _compare_outputs(torch_outputs, jax_outputs, args.atol, args.rtol)
    if args.out_npz is not None:
        _save_snapshot(
            args.out_npz,
            history,
            text,
            x_start,
            noise,
            timesteps,
            torch_outputs,
            jax_outputs,
            config=config,
            torch_state=_collect_torch_state_np(torch_model),
        )
        print("wrote:", args.out_npz)
    if args.fail_on_mismatch:
        _maybe_fail(compared)


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
