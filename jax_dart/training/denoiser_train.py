"""Shard-backed JAX/NNX trainer for DART's latent primitive denoiser.

This is the full training loop for benchmark runs on exported TPU shards. It
uses a frozen JAX VAE checkpoint, trains either the MLP or Transformer
denoiser on the same ``metadata.json``/``shards`` layout used by the parity
checks, and writes JSONL metrics plus Torch-style NPZ checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import (
    iter_vae_batches,
    load_metadata,
    load_text_embedding_table,
    text_embeddings_for_batch,
)
from jax_dart.diffusion import create_gaussian_diffusion
from jax_dart.models.mld_denoiser import DenoiserMLP, DenoiserTransformer
from jax_dart.models.mld_denoiser_state import (
    load_torch_style_denoiser_state,
    save_denoiser_npz_checkpoint,
    state_from_npz as denoiser_state_from_npz,
)
from jax_dart.models.mld_vae import AutoMldVae, huber_loss
from jax_dart.models.mld_vae_state import (
    load_torch_style_vae_state,
    state_from_npz as vae_state_from_npz,
)
from jax_dart.models.smplx_joints import load_smplx_joints_model, smpl_joint_loss_from_motion
from jax_dart.models.temporal_smpl_loss import (
    load_motion_normalization,
    temporal_smpl_feature_loss,
)
from jax_dart.training.vae_train import (
    _assert_finite,
    _betas_for_step,
    _jax_dtype,
    _parse_latent_dim,
    _to_float,
    _uses_smpl_joint_loss,
    _write_metrics,
)


def _cycle_batches(root: str, batch_samples: int) -> Iterator[Dict[str, np.ndarray]]:
    while True:
        yield from iter_vae_batches(root, batch_samples=batch_samples)


def _npz_json(path: str, key: str) -> Dict[str, Any]:
    with np.load(path) as data:
        if key not in data:
            return {}
        return json.loads(str(data[key].tolist()))


def _vae_config_value(config: Dict[str, Any], args: argparse.Namespace, key: str, attr: str, default):
    value = config.get(key, None)
    if value is not None:
        return value
    value = getattr(args, attr)
    return default if value is None else value


def _make_vae_model(args: argparse.Namespace, vae_config: Dict[str, Any], feature_dim: int):
    from flax import nnx

    latent_dim = tuple(
        int(x)
        for x in _vae_config_value(
            vae_config,
            args,
            "latent_dim",
            "vae_latent_dim",
            (1, 256),
        )
    )
    return AutoMldVae(
        nfeats=feature_dim,
        latent_dim=latent_dim,
        h_dim=int(_vae_config_value(vae_config, args, "h_dim", "vae_h_dim", 256)),
        ff_size=int(_vae_config_value(vae_config, args, "ff_size", "vae_ff_size", 1024)),
        num_layers=int(_vae_config_value(vae_config, args, "num_layers", "vae_num_layers", 7)),
        num_heads=int(_vae_config_value(vae_config, args, "num_heads", "vae_num_heads", 4)),
        dropout=float(_vae_config_value(vae_config, args, "dropout", "vae_dropout", 0.1)),
        arch=str(_vae_config_value(vae_config, args, "arch", "vae_arch", "all_encoder")),
        normalize_before=bool(
            _vae_config_value(vae_config, args, "normalize_before", "vae_normalize_before", False)
        ),
        activation=str(_vae_config_value(vae_config, args, "activation", "vae_activation", "gelu")),
        position_embedding=str(
            _vae_config_value(
                vae_config,
                args,
                "position_embedding",
                "vae_position_embedding",
                "learned",
            )
        ),
        rngs=nnx.Rngs(args.seed),
    )


def _vae_num_layers_arch(args: argparse.Namespace, vae_config: Dict[str, Any]) -> Tuple[int, str]:
    return (
        int(_vae_config_value(vae_config, args, "num_layers", "vae_num_layers", 7)),
        str(_vae_config_value(vae_config, args, "arch", "vae_arch", "all_encoder")),
    )


def _load_vae(args: argparse.Namespace, metadata: dict):
    feature_dim = int(metadata["primitive"]["feature_dim"])
    vae_config = _npz_json(args.vae_npz, "config_json")
    model = _make_vae_model(args, vae_config, feature_dim)
    num_layers, arch = _vae_num_layers_arch(args, vae_config)
    load_torch_style_vae_state(
        model,
        vae_state_from_npz(args.vae_npz),
        num_layers=num_layers,
        arch=arch,
    )
    return model, vae_config, (int(model.latent_size), int(model.latent_dim))


def _infer_text_dim(
    metadata: dict,
    args: argparse.Namespace,
    text_table: Optional[np.ndarray],
) -> int:
    if text_table is not None:
        return int(text_table.shape[-1])
    shape = metadata.get("arrays", {}).get("text_embedding")
    if shape:
        return int(shape[-1])
    if args.clip_dim > 0:
        return int(args.clip_dim)
    return 512


def _make_denoiser_model(
    args: argparse.Namespace,
    *,
    history_shape: Tuple[int, int],
    noise_shape: Tuple[int, int],
    clip_dim: int,
):
    from flax import nnx

    if args.model_type == "transformer":
        return DenoiserTransformer(
            h_dim=args.h_dim,
            ff_size=args.ff_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            activation=args.activation,
            clip_dim=clip_dim,
            history_shape=history_shape,
            noise_shape=noise_shape,
            cond_mask_prob=args.cond_mask_prob,
            rngs=nnx.Rngs(args.seed + 17),
        )
    return DenoiserMLP(
        h_dim=args.h_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        activation=args.activation,
        clip_dim=clip_dim,
        history_shape=history_shape,
        noise_shape=noise_shape,
        cond_mask_prob=args.cond_mask_prob,
        rngs=nnx.Rngs(args.seed + 17),
    )


def _make_optimizer(model, args: argparse.Namespace):
    import optax
    from flax import nnx

    if args.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=0.0,
            transition_steps=max(int(args.steps), 1),
        )
    else:
        learning_rate = args.learning_rate
    transforms = []
    if args.grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(args.grad_clip))
    transforms.append(optax.adamw(learning_rate=learning_rate, weight_decay=args.weight_decay))
    return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)


def _learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if not args.anneal_lr:
        return float(args.learning_rate)
    frac = max(0.0, 1.0 - (float(step - 1) / max(float(args.steps), 1.0)))
    return float(args.learning_rate * frac)


def _make_data_mesh(jax, device_count: int):
    devices = jax.local_devices()
    if device_count <= 0:
        device_count = len(devices)
    if device_count < 1:
        raise SystemExit("No local JAX devices are available.")
    if device_count > len(devices):
        raise SystemExit(f"Requested {device_count} devices, but only {len(devices)} are visible.")
    return jax.sharding.Mesh(np.asarray(devices[:device_count]), ("data",))


def _throughput_row(
    *,
    step_delta: int,
    elapsed_delta: float,
    batch_samples: int,
    global_batch: int,
) -> Dict[str, float]:
    if elapsed_delta <= 0.0 or step_delta <= 0:
        return {"steps_per_sec": 0.0, "samples_per_sec": 0.0, "examples_per_sec": 0.0}
    return {
        "steps_per_sec": float(step_delta / elapsed_delta),
        "samples_per_sec": float((step_delta * batch_samples) / elapsed_delta),
        "examples_per_sec": float((step_delta * global_batch) / elapsed_delta),
    }


def _global_batch_from_batch_samples(metadata: dict, batch_samples: int) -> int:
    return int(batch_samples) * int(metadata["primitive"]["num_primitive"])


def _check_global_batch(metadata: dict, batch_samples: int, device_count: int) -> None:
    global_batch = _global_batch_from_batch_samples(metadata, batch_samples)
    if global_batch % device_count != 0:
        raise SystemExit(
            f"Global denoiser batch {global_batch} is not divisible by {device_count} devices. "
            "Change --batch-samples or --data-parallel-devices."
        )


def _config_dict(
    args: argparse.Namespace,
    *,
    metadata: dict,
    vae_config: Dict[str, Any],
    history_shape: Tuple[int, int],
    noise_shape: Tuple[int, int],
    clip_dim: int,
    device_count: int,
    per_device_batch: int,
) -> Dict[str, Any]:
    return {
        "format": "jax_dart_denoiser_npz_v1",
        "source_format": metadata.get("format"),
        "root": args.root,
        "vae_npz": args.vae_npz,
        "vae_config": vae_config,
        "trainer": "spmd" if device_count > 1 else "single",
        "data_parallel_devices": int(device_count),
        "per_device_batch": int(per_device_batch),
        "batch_samples": int(args.batch_samples),
        "global_batch": int(_global_batch_from_batch_samples(metadata, args.batch_samples)),
        "dtype": args.dtype,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "anneal_lr": bool(args.anneal_lr),
        "weight_latent_rec": float(args.weight_latent_rec),
        "weight_feature_rec": float(args.weight_feature_rec),
        "weight_joints_delta": float(args.weight_joints_delta),
        "weight_transl_delta": float(args.weight_transl_delta),
        "weight_orient_delta": float(args.weight_orient_delta),
        "weight_smpl_joints_rec": float(args.weight_smpl_joints_rec),
        "weight_joints_consistency": float(args.weight_joints_consistency),
        "smpl_model_dir": args.smpl_model_dir,
        "smpl_gender": args.smpl_gender,
        "model_type": args.model_type,
        "h_dim": int(args.h_dim),
        "n_blocks": int(args.n_blocks),
        "ff_size": int(args.ff_size),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "activation": args.activation,
        "cond_mask_prob": float(args.cond_mask_prob),
        "clip_dim": int(clip_dim),
        "history_shape": [int(x) for x in history_shape],
        "noise_shape": [int(x) for x in noise_shape],
        "diffusion_steps": int(args.diffusion_steps),
        "noise_schedule": args.noise_schedule,
        "scale_betas": float(args.scale_betas),
        "rescale_timesteps": bool(args.rescale_timesteps),
        "rescale_latent": bool(args.rescale_latent),
        "sample_vae_latent": bool(args.sample_vae_latent),
        "seed": int(args.seed),
        "feature_slices": metadata["primitive"].get("feature_slices"),
        "rollout_history": "ground_truth_exported_shards",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JAX/NNX DART latent denoiser.")
    parser.add_argument("--root", required=True, help="Export root containing metadata.json and shards/.")
    parser.add_argument("--vae-npz", required=True, help="Frozen JAX VAE checkpoint/latest.npz.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-npz", default=None, help="Optional denoiser NPZ checkpoint to initialize from.")
    parser.add_argument("--batch-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=50000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--weight-latent-rec", type=float, default=1.0)
    parser.add_argument("--weight-feature-rec", type=float, default=1.0)
    parser.add_argument("--weight-joints-delta", type=float, default=1e4)
    parser.add_argument("--weight-transl-delta", type=float, default=1e4)
    parser.add_argument("--weight-orient-delta", type=float, default=1e4)
    parser.add_argument("--weight-smpl-joints-rec", type=float, default=0.0)
    parser.add_argument("--weight-joints-consistency", type=float, default=0.0)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument("--smpl-gender", default="male", choices=["male"])
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--h-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu", choices=["tanh", "relu", "sigmoid", "gelu", "lrelu"])
    parser.add_argument("--cond-mask-prob", type=float, default=0.1)
    parser.add_argument("--clip-dim", type=int, default=0, help="Override text embedding width; inferred by default.")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--noise-schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--scale-betas", type=float, default=1.0)
    parser.add_argument("--rescale-timesteps", action="store_true")
    parser.add_argument("--rescale-latent", dest="rescale_latent", action="store_true", default=True)
    parser.add_argument("--no-rescale-latent", dest="rescale_latent", action="store_false")
    parser.add_argument("--sample-vae-latent", dest="sample_vae_latent", action="store_true", default=True)
    parser.add_argument("--use-vae-mean", dest="sample_vae_latent", action="store_false")
    parser.add_argument("--data-parallel-devices", type=int, default=0)
    parser.add_argument("--jax-matmul-precision", default="highest", choices=["default", "high", "highest"])
    parser.add_argument("--fail-on-nonfinite", action="store_true")

    parser.add_argument("--vae-h-dim", type=int, default=256)
    parser.add_argument("--vae-ff-size", type=int, default=1024)
    parser.add_argument("--vae-num-layers", type=int, default=7)
    parser.add_argument("--vae-num-heads", type=int, default=4)
    parser.add_argument("--vae-dropout", type=float, default=0.1)
    parser.add_argument("--vae-latent-dim", type=_parse_latent_dim, default=(1, 256))
    parser.add_argument("--vae-arch", default="all_encoder", choices=["all_encoder", "encoder_decoder"])
    parser.add_argument("--vae-activation", default="gelu", choices=["gelu", "relu", "glu"])
    parser.add_argument("--vae-position-embedding", default="learned", choices=["learned", "v3", "sine", "v2"])
    parser.add_argument("--vae-normalize-before", action="store_true")
    return parser.parse_args()


def main() -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    args = parse_args()
    if args.rescale_timesteps:
        raise SystemExit("DART denoiser timestep embeddings expect integer timesteps; omit --rescale-timesteps.")
    if args.jax_matmul_precision != "default":
        jax.config.update("jax_default_matmul_precision", args.jax_matmul_precision)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    metadata = load_metadata(args.root)
    feature_dim = int(metadata["primitive"]["feature_dim"])
    history_length = int(metadata["primitive"]["history_length"])
    future_length = int(metadata["primitive"]["future_length"])
    feature_slices = metadata["primitive"]["feature_slices"]
    text_table = load_text_embedding_table(args.root)
    clip_dim = _infer_text_dim(metadata, args, text_table)
    history_shape = (history_length, feature_dim)

    vae_model, vae_config, noise_shape = _load_vae(args, metadata)
    dtype = _jax_dtype(args.dtype)
    diffusion = create_gaussian_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        scale_betas=args.scale_betas,
        rescale_timesteps=args.rescale_timesteps,
    )
    norm_mean_np, norm_std_np = load_motion_normalization(args.root)
    use_smpl_joint_loss = _uses_smpl_joint_loss(args)
    if use_smpl_joint_loss and args.smpl_model_dir is None:
        raise SystemExit("--smpl-model-dir is required when SMPL joint loss weights are nonzero.")

    mesh = _make_data_mesh(jax, args.data_parallel_devices)
    device_count = int(np.asarray(mesh.devices).size)
    _check_global_batch(metadata, args.batch_samples, device_count)
    global_batch = _global_batch_from_batch_samples(metadata, args.batch_samples)
    per_device_batch = global_batch // device_count
    data3_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
    data2_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    vector_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
    norm_mean = jax.device_put(jnp.asarray(norm_mean_np, dtype=jnp.float32), vector_sharding)
    norm_std = jax.device_put(jnp.asarray(norm_std_np, dtype=jnp.float32), vector_sharding)

    mesh_context = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else nullcontext()
    with mesh_context:
        smplx_model = None
        if use_smpl_joint_loss:
            smplx_model = load_smplx_joints_model(args.smpl_model_dir, gender=args.smpl_gender)

        denoiser_model = _make_denoiser_model(
            args,
            history_shape=history_shape,
            noise_shape=noise_shape,
            clip_dim=clip_dim,
        )
        if args.init_npz is not None:
            load_torch_style_denoiser_state(denoiser_model, denoiser_state_from_npz(args.init_npz))
        optimizer = _make_optimizer(denoiser_model, args)

        config = _config_dict(
            args,
            metadata=metadata,
            vae_config=vae_config,
            history_shape=history_shape,
            noise_shape=noise_shape,
            clip_dim=clip_dim,
            device_count=device_count,
            per_device_batch=per_device_batch,
        )
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

        @nnx.jit
        def train_step(denoiser_model, optimizer, vae_model, history, future, text, betas, rng):
            latent_key, timestep_key, noise_key, mask_key = jax.random.split(rng, 4)
            latent_rng = latent_key if args.sample_vae_latent else None
            latent_gt, _ = vae_model.encode(
                future,
                history,
                rng=latent_rng,
                scale_latent=args.rescale_latent,
            )
            latent_gt = jax.lax.stop_gradient(latent_gt)
            x_start = jnp.swapaxes(latent_gt, 0, 1)
            timesteps = jax.random.randint(
                timestep_key,
                (x_start.shape[0],),
                minval=0,
                maxval=diffusion.num_timesteps,
                dtype=jnp.int32,
            )
            noise = jax.random.normal(noise_key, x_start.shape, dtype=x_start.dtype)
            x_t = diffusion.q_sample(x_start=x_start, t=timesteps, noise=noise)

            def loss_fn(model):
                y = {"history_motion_normalized": history, "text_embedding": text}
                x_start_pred = model(
                    x_t,
                    diffusion.scale_timesteps(timesteps),
                    y,
                    training=True,
                    rng=mask_key,
                )
                latent_pred = jnp.swapaxes(x_start_pred, 0, 1)
                future_pred = vae_model.decode(
                    latent_pred,
                    history,
                    nfuture=future.shape[1],
                    scale_latent=args.rescale_latent,
                )
                latent_rec = jnp.mean(huber_loss(latent_pred, latent_gt))
                feature_rec = jnp.mean(huber_loss(future_pred, future))
                temporal_loss, temporal_terms = temporal_smpl_feature_loss(
                    history,
                    future_pred,
                    feature_slices=feature_slices,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    weight_joints_delta=args.weight_joints_delta,
                    weight_transl_delta=args.weight_transl_delta,
                    weight_orient_delta=args.weight_orient_delta,
                )
                if use_smpl_joint_loss:
                    smpl_loss, smpl_terms = smpl_joint_loss_from_motion(
                        smplx_model,
                        pred_motion=future_pred,
                        gt_motion=future,
                        betas=betas,
                        feature_slices=feature_slices,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        weight_smpl_joints_rec=args.weight_smpl_joints_rec,
                        weight_joints_consistency=args.weight_joints_consistency,
                    )
                else:
                    smpl_terms = {
                        "smpl_joints_rec": jnp.asarray(0.0, dtype=jnp.float32),
                        "joints_consistency": jnp.asarray(0.0, dtype=jnp.float32),
                        "smpl_joint_loss": jnp.asarray(0.0, dtype=jnp.float32),
                    }
                    smpl_loss = smpl_terms["smpl_joint_loss"]
                loss = (
                    args.weight_latent_rec * latent_rec
                    + args.weight_feature_rec * feature_rec
                    + temporal_loss
                    + smpl_loss
                )
                return loss, (
                    latent_rec,
                    feature_rec,
                    temporal_terms["joints_delta"],
                    temporal_terms["transl_delta"],
                    temporal_terms["orient_delta"],
                    temporal_terms["temporal_smpl"],
                    smpl_terms["smpl_joints_rec"],
                    smpl_terms["joints_consistency"],
                    smpl_terms["smpl_joint_loss"],
                )

            (loss, terms), grads = nnx.value_and_grad(loss_fn, has_aux=True)(denoiser_model)
            optimizer.update(denoiser_model, grads)
            return (loss, *terms)

        @nnx.jit
        def eval_step(denoiser_model, vae_model, history, future, text, betas, rng):
            timestep_key, noise_key = jax.random.split(rng, 2)
            latent_gt, _ = vae_model.encode(
                future,
                history,
                rng=None,
                scale_latent=args.rescale_latent,
            )
            latent_gt = jax.lax.stop_gradient(latent_gt)
            x_start = jnp.swapaxes(latent_gt, 0, 1)
            timesteps = jax.random.randint(
                timestep_key,
                (x_start.shape[0],),
                minval=0,
                maxval=diffusion.num_timesteps,
                dtype=jnp.int32,
            )
            noise = jax.random.normal(noise_key, x_start.shape, dtype=x_start.dtype)
            x_t = diffusion.q_sample(x_start=x_start, t=timesteps, noise=noise)
            y = {"history_motion_normalized": history, "text_embedding": text}
            x_start_pred = denoiser_model(
                x_t,
                diffusion.scale_timesteps(timesteps),
                y,
                training=False,
                rng=None,
            )
            latent_pred = jnp.swapaxes(x_start_pred, 0, 1)
            future_pred = vae_model.decode(
                latent_pred,
                history,
                nfuture=future.shape[1],
                scale_latent=args.rescale_latent,
            )
            latent_rec = jnp.mean(huber_loss(latent_pred, latent_gt))
            feature_rec = jnp.mean(huber_loss(future_pred, future))
            temporal_loss, temporal_terms = temporal_smpl_feature_loss(
                history,
                future_pred,
                feature_slices=feature_slices,
                norm_mean=norm_mean,
                norm_std=norm_std,
                weight_joints_delta=args.weight_joints_delta,
                weight_transl_delta=args.weight_transl_delta,
                weight_orient_delta=args.weight_orient_delta,
            )
            if use_smpl_joint_loss:
                smpl_loss, smpl_terms = smpl_joint_loss_from_motion(
                    smplx_model,
                    pred_motion=future_pred,
                    gt_motion=future,
                    betas=betas,
                    feature_slices=feature_slices,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    weight_smpl_joints_rec=args.weight_smpl_joints_rec,
                    weight_joints_consistency=args.weight_joints_consistency,
                )
            else:
                smpl_terms = {
                    "smpl_joints_rec": jnp.asarray(0.0, dtype=jnp.float32),
                    "joints_consistency": jnp.asarray(0.0, dtype=jnp.float32),
                    "smpl_joint_loss": jnp.asarray(0.0, dtype=jnp.float32),
                }
                smpl_loss = smpl_terms["smpl_joint_loss"]
            loss = (
                args.weight_latent_rec * latent_rec
                + args.weight_feature_rec * feature_rec
                + temporal_loss
                + smpl_loss
            )
            return (
                loss,
                latent_rec,
                feature_rec,
                temporal_terms["joints_delta"],
                temporal_terms["transl_delta"],
                temporal_terms["orient_delta"],
                temporal_terms["temporal_smpl"],
                smpl_terms["smpl_joints_rec"],
                smpl_terms["joints_consistency"],
                smpl_terms["smpl_joint_loss"],
            )

        def batch_to_device(batch: Dict[str, np.ndarray]):
            history = jax.device_put(jnp.asarray(batch["history"], dtype=dtype), data3_sharding)
            future = jax.device_put(jnp.asarray(batch["future"], dtype=dtype), data3_sharding)
            text_np = text_embeddings_for_batch(batch, text_embedding_table=text_table)
            text = jax.device_put(jnp.asarray(text_np, dtype=dtype), data2_sharding)
            betas_np = _betas_for_step(
                batch,
                history_length=history_length,
                future_length=future_length,
                batch_size=batch["future"].shape[0],
                require_betas=use_smpl_joint_loss,
            )
            betas = jax.device_put(jnp.asarray(betas_np, dtype=jnp.float32), data3_sharding)
            return history, future, text, betas

        def run_eval(step: int) -> Dict[str, float]:
            if args.eval_batches <= 0:
                return {}
            losses = []
            latent_recs = []
            feature_recs = []
            joints_deltas = []
            transl_deltas = []
            orient_deltas = []
            temporal_smpls = []
            smpl_joints_recs = []
            joints_consistencies = []
            smpl_joint_losses = []
            for index, batch in enumerate(iter_vae_batches(args.root, batch_samples=args.batch_samples)):
                if index >= args.eval_batches:
                    break
                history, future, text, betas = batch_to_device(batch)
                eval_key = jax.random.PRNGKey(args.seed + 1000003 + step * 997 + index)
                (
                    loss,
                    latent_rec,
                    feature_rec,
                    joints_delta,
                    transl_delta,
                    orient_delta,
                    temporal_smpl,
                    smpl_joints_rec,
                    joints_consistency,
                    smpl_joint_loss,
                ) = eval_step(denoiser_model, vae_model, history, future, text, betas, eval_key)
                losses.append(_to_float(loss))
                latent_recs.append(_to_float(latent_rec))
                feature_recs.append(_to_float(feature_rec))
                joints_deltas.append(_to_float(joints_delta))
                transl_deltas.append(_to_float(transl_delta))
                orient_deltas.append(_to_float(orient_delta))
                temporal_smpls.append(_to_float(temporal_smpl))
                smpl_joints_recs.append(_to_float(smpl_joints_rec))
                joints_consistencies.append(_to_float(joints_consistency))
                smpl_joint_losses.append(_to_float(smpl_joint_loss))
            row = {
                "step": int(step),
                "split": "eval",
                "loss": float(np.mean(losses)),
                "latent_rec": float(np.mean(latent_recs)),
                "feature_rec": float(np.mean(feature_recs)),
                "joints_delta": float(np.mean(joints_deltas)),
                "transl_delta": float(np.mean(transl_deltas)),
                "orient_delta": float(np.mean(orient_deltas)),
                "temporal_smpl": float(np.mean(temporal_smpls)),
                "smpl_joints_rec": float(np.mean(smpl_joints_recs)),
                "joints_consistency": float(np.mean(joints_consistencies)),
                "smpl_joint_loss": float(np.mean(smpl_joint_losses)),
                "data_parallel_devices": int(device_count),
                "per_device_batch": int(per_device_batch),
                "time": time.time(),
            }
            _write_metrics(metrics_path, row)
            print(
                f"eval step={step} loss={row['loss']:.6f} "
                f"latent_rec={row['latent_rec']:.6f} "
                f"feature_rec={row['feature_rec']:.6f} "
                f"temporal_smpl={row['temporal_smpl']:.6f} "
                f"smpl_joint_loss={row['smpl_joint_loss']:.6f}"
            )
            return row

        print("backend:", jax.default_backend())
        print("devices:", jax.devices())
        print("trainer:", "spmd" if device_count > 1 else "single")
        print("mesh:", mesh)
        print("data_parallel_devices:", device_count)
        print("batch_samples:", args.batch_samples)
        print("global_batch:", global_batch)
        print("per_device_batch:", per_device_batch)
        print("metadata_format:", metadata["format"])
        print("feature_dim:", feature_dim)
        print("history_shape:", history_shape)
        print("noise_shape:", noise_shape)
        print("clip_dim:", clip_dim)
        print("out_dir:", str(out_dir))

        batches = _cycle_batches(args.root, args.batch_samples)
        last_metrics: Dict[str, object] = run_eval(0)
        rng = jax.random.PRNGKey(args.seed)
        start_time = time.time()
        last_log_step = 0
        last_log_time = start_time
        for step in range(1, args.steps + 1):
            batch = next(batches)
            history, future, text, betas = batch_to_device(batch)
            rng, step_key = jax.random.split(rng)
            (
                loss,
                latent_rec,
                feature_rec,
                joints_delta,
                transl_delta,
                orient_delta,
                temporal_smpl,
                smpl_joints_rec,
                joints_consistency,
                smpl_joint_loss,
            ) = train_step(denoiser_model, optimizer, vae_model, history, future, text, betas, step_key)
            _assert_finite("loss", loss, args.fail_on_nonfinite)

            if step == 1 or step == args.steps or step % args.log_every == 0:
                now = time.time()
                elapsed = now - start_time
                interval = now - last_log_time
                step_delta = step - last_log_step
                throughput = _throughput_row(
                    step_delta=step_delta,
                    elapsed_delta=interval,
                    batch_samples=args.batch_samples,
                    global_batch=global_batch,
                )
                row = {
                    "step": int(step),
                    "split": "train",
                    "loss": _to_float(loss),
                    "latent_rec": _to_float(latent_rec),
                    "feature_rec": _to_float(feature_rec),
                    "joints_delta": _to_float(joints_delta),
                    "transl_delta": _to_float(transl_delta),
                    "orient_delta": _to_float(orient_delta),
                    "temporal_smpl": _to_float(temporal_smpl),
                    "smpl_joints_rec": _to_float(smpl_joints_rec),
                    "joints_consistency": _to_float(joints_consistency),
                    "smpl_joint_loss": _to_float(smpl_joint_loss),
                    "learning_rate": _learning_rate_for_step(args, step),
                    "data_parallel_devices": int(device_count),
                    "per_device_batch": int(per_device_batch),
                    "elapsed_sec": float(elapsed),
                    "steps_per_sec": throughput["steps_per_sec"],
                    "samples_per_sec": throughput["samples_per_sec"],
                    "examples_per_sec": throughput["examples_per_sec"],
                    "time": now,
                }
                _write_metrics(metrics_path, row)
                last_metrics = row
                print(
                    f"train step={step} loss={row['loss']:.6f} "
                    f"latent_rec={row['latent_rec']:.6f} "
                    f"feature_rec={row['feature_rec']:.6f} "
                    f"temporal_smpl={row['temporal_smpl']:.6f} "
                    f"smpl_joint_loss={row['smpl_joint_loss']:.6f} "
                    f"steps/s={row['steps_per_sec']:.3f} "
                    f"samples/s={row['samples_per_sec']:.1f} "
                    f"examples/s={row['examples_per_sec']:.1f} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_log_step = step
                last_log_time = now

            if args.eval_every > 0 and step % args.eval_every == 0:
                last_metrics = run_eval(step)

            should_save = args.save_every > 0 and step % args.save_every == 0
            if should_save or step == args.steps:
                checkpoint_path = out_dir / f"checkpoint_{step:08d}.npz"
                save_denoiser_npz_checkpoint(
                    str(checkpoint_path),
                    denoiser_model,
                    config=config,
                    metrics=last_metrics,
                    step=step,
                )
                save_denoiser_npz_checkpoint(
                    str(out_dir / "latest.npz"),
                    denoiser_model,
                    config=config,
                    metrics=last_metrics,
                    step=step,
                )
                print("saved:", checkpoint_path)

        print("optimizer_step:", optimizer.step[...])
        print("metrics:", str(metrics_path))
        print("latest:", str(out_dir / "latest.npz"))


if __name__ == "__main__":
    main()
