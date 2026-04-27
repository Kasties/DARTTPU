"""Minimal JAX/NNX trainer for the DART motion primitive VAE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata
from jax_dart.models.mld_vae import AutoMldVae, vae_reconstruction_kl_loss
from jax_dart.models.mld_vae_state import (
    load_torch_style_vae_state,
    save_vae_npz_checkpoint,
    state_from_npz,
)
from jax_dart.models.temporal_smpl_loss import (
    load_motion_normalization,
    temporal_smpl_feature_loss,
)


def _parse_latent_dim(value: str) -> Tuple[int, int]:
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("latent dim must contain two integers, e.g. '1 256'")
    return parts[0], parts[1]


def _jax_dtype(name: str):
    import jax.numpy as jnp

    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    if name == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {name}.")


def _cycle_vae_batches(root: str, batch_samples: int) -> Iterator[Dict[str, np.ndarray]]:
    while True:
        yield from iter_vae_batches(root, batch_samples=batch_samples)


def _make_model(args: argparse.Namespace, feature_dim: int):
    from flax import nnx

    return AutoMldVae(
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


def _make_optimizer(model, learning_rate: float, weight_decay: float):
    import optax
    from flax import nnx

    return nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        wrt=nnx.Param,
    )


def _load_initial_state(model, args: argparse.Namespace) -> None:
    if args.init_npz is None:
        return
    state = state_from_npz(args.init_npz)
    load_torch_style_vae_state(
        model,
        state,
        num_layers=args.num_layers,
        arch=args.arch,
    )


def _config_dict(args: argparse.Namespace, metadata: dict, feature_dim: int) -> Dict[str, object]:
    return {
        "format": "jax_dart_vae_npz_v1",
        "source_format": metadata.get("format"),
        "root": args.root,
        "feature_dim": int(feature_dim),
        "feature_slices": metadata["primitive"].get("feature_slices"),
        "batch_samples": int(args.batch_samples),
        "dtype": args.dtype,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "weight_rec": float(args.weight_rec),
        "weight_kl": float(args.weight_kl),
        "weight_joints_delta": float(args.weight_joints_delta),
        "weight_transl_delta": float(args.weight_transl_delta),
        "weight_orient_delta": float(args.weight_orient_delta),
        "latent_dim": [int(args.latent_dim[0]), int(args.latent_dim[1])],
        "h_dim": int(args.h_dim),
        "ff_size": int(args.ff_size),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "arch": args.arch,
        "activation": args.activation,
        "position_embedding": args.position_embedding,
        "normalize_before": bool(args.normalize_before),
        "seed": int(args.seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JAX/NNX DART primitive VAE.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-npz", default=None, help="Optional state::*.npz snapshot/checkpoint to initialize from.")
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--weight-rec", type=float, default=1.0)
    parser.add_argument("--weight-kl", type=float, default=1e-4)
    parser.add_argument("--weight-joints-delta", type=float, default=0.0)
    parser.add_argument("--weight-transl-delta", type=float, default=0.0)
    parser.add_argument("--weight-orient-delta", type=float, default=0.0)
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
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    return parser.parse_args()


def _to_float(value) -> float:
    import jax

    return float(np.asarray(jax.device_get(value)))


def _assert_finite(name: str, value, enabled: bool) -> None:
    if not enabled:
        return
    import jax.numpy as jnp

    if not bool(jnp.isfinite(value).all()):
        raise SystemExit(f"{name} is not finite: {value}")


def _write_metrics(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    metadata = load_metadata(args.root)
    feature_dim = int(metadata["primitive"]["feature_dim"])
    feature_slices = metadata["primitive"]["feature_slices"]
    norm_mean_np, norm_std_np = load_motion_normalization(args.root)
    norm_mean = jnp.asarray(norm_mean_np, dtype=jnp.float32)
    norm_std = jnp.asarray(norm_std_np, dtype=jnp.float32)
    dtype = _jax_dtype(args.dtype)
    config = _config_dict(args, metadata, feature_dim)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    model = _make_model(args, feature_dim)
    _load_initial_state(model, args)
    optimizer = _make_optimizer(model, args.learning_rate, args.weight_decay)

    @nnx.jit
    def train_step(model, optimizer, history, future):
        def loss_fn(model):
            latent, dist = model.encode(future, history, rng=None)
            future_pred = model.decode(latent, history, nfuture=future.shape[1])
            loss, terms = vae_reconstruction_kl_loss(
                future_pred,
                future,
                dist,
                weight_rec=args.weight_rec,
                weight_kl=args.weight_kl,
            )
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
            loss = loss + temporal_loss
            return loss, (
                terms["rec"],
                terms["kl"],
                temporal_terms["joints_delta"],
                temporal_terms["transl_delta"],
                temporal_terms["orient_delta"],
                temporal_terms["temporal_smpl"],
            )

        (loss, terms), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return (loss, *terms)

    @nnx.jit
    def eval_step(model, history, future):
        latent, dist = model.encode(future, history, rng=None)
        future_pred = model.decode(latent, history, nfuture=future.shape[1])
        loss, terms = vae_reconstruction_kl_loss(
            future_pred,
            future,
            dist,
            weight_rec=args.weight_rec,
            weight_kl=args.weight_kl,
        )
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
        loss = loss + temporal_loss
        return (
            loss,
            terms["rec"],
            terms["kl"],
            temporal_terms["joints_delta"],
            temporal_terms["transl_delta"],
            temporal_terms["orient_delta"],
            temporal_terms["temporal_smpl"],
        )

    def run_eval(step: int) -> Dict[str, float]:
        losses = []
        recs = []
        kls = []
        joints_deltas = []
        transl_deltas = []
        orient_deltas = []
        temporal_smpls = []
        for index, batch in enumerate(iter_vae_batches(args.root, batch_samples=args.batch_samples)):
            if index >= args.eval_batches:
                break
            history = jnp.asarray(batch["history"], dtype=dtype)
            future = jnp.asarray(batch["future"], dtype=dtype)
            loss, rec, kl, joints_delta, transl_delta, orient_delta, temporal_smpl = eval_step(model, history, future)
            losses.append(_to_float(loss))
            recs.append(_to_float(rec))
            kls.append(_to_float(kl))
            joints_deltas.append(_to_float(joints_delta))
            transl_deltas.append(_to_float(transl_delta))
            orient_deltas.append(_to_float(orient_delta))
            temporal_smpls.append(_to_float(temporal_smpl))
        row = {
            "step": int(step),
            "split": "eval",
            "loss": float(np.mean(losses)),
            "rec": float(np.mean(recs)),
            "kl": float(np.mean(kls)),
            "joints_delta": float(np.mean(joints_deltas)),
            "transl_delta": float(np.mean(transl_deltas)),
            "orient_delta": float(np.mean(orient_deltas)),
            "temporal_smpl": float(np.mean(temporal_smpls)),
            "time": time.time(),
        }
        _write_metrics(metrics_path, row)
        print(
            f"eval step={step} loss={row['loss']:.6f} "
            f"rec={row['rec']:.6f} kl={row['kl']:.6f} "
            f"temporal_smpl={row['temporal_smpl']:.6f}"
        )
        return row

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("metadata_format:", metadata["format"])
    print("feature_dim:", feature_dim)
    print("out_dir:", str(out_dir))

    batches = _cycle_vae_batches(args.root, args.batch_samples)
    run_eval(0)
    start_time = time.time()
    last_metrics: Dict[str, object] = {}
    for step in range(1, args.steps + 1):
        batch = next(batches)
        history = jnp.asarray(batch["history"], dtype=dtype)
        future = jnp.asarray(batch["future"], dtype=dtype)
        loss, rec, kl, joints_delta, transl_delta, orient_delta, temporal_smpl = train_step(
            model,
            optimizer,
            history,
            future,
        )
        _assert_finite("loss", loss, args.fail_on_nonfinite)

        if step == 1 or step == args.steps or step % args.log_every == 0:
            elapsed = time.time() - start_time
            row = {
                "step": int(step),
                "split": "train",
                "loss": _to_float(loss),
                "rec": _to_float(rec),
                "kl": _to_float(kl),
                "joints_delta": _to_float(joints_delta),
                "transl_delta": _to_float(transl_delta),
                "orient_delta": _to_float(orient_delta),
                "temporal_smpl": _to_float(temporal_smpl),
                "elapsed_sec": float(elapsed),
                "time": time.time(),
            }
            _write_metrics(metrics_path, row)
            last_metrics = row
            print(
                f"train step={step} loss={row['loss']:.6f} "
                f"rec={row['rec']:.6f} kl={row['kl']:.6f} "
                f"temporal_smpl={row['temporal_smpl']:.6f} "
                f"elapsed={elapsed:.1f}s"
            )

        if args.eval_every > 0 and step % args.eval_every == 0:
            last_metrics = run_eval(step)

        should_save = args.save_every > 0 and step % args.save_every == 0
        if should_save or step == args.steps:
            checkpoint_path = out_dir / f"checkpoint_{step:08d}.npz"
            save_vae_npz_checkpoint(
                str(checkpoint_path),
                model,
                num_layers=args.num_layers,
                arch=args.arch,
                config=config,
                metrics=last_metrics,
                step=step,
            )
            save_vae_npz_checkpoint(
                str(out_dir / "latest.npz"),
                model,
                num_layers=args.num_layers,
                arch=args.arch,
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
