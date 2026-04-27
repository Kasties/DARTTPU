"""Small JAX/NNX VAE training smoke test.

This is not the full DART VAE trainer yet. It proves that the JAX VAE can run a
real optimizer update on exported primitive shards, optionally initialized from
a Torch parity snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata
from jax_dart.models.mld_vae import AutoMldVae, vae_reconstruction_kl_loss
from jax_dart.models.smplx_joints import load_smplx_joints_model, smpl_joint_loss_from_motion
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


def _load_torch_state_from_snapshot(path: str) -> Dict[str, np.ndarray]:
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


def _maybe_load_snapshot(model, args: argparse.Namespace) -> None:
    if args.snapshot is None:
        return
    from jax_dart.parity.vae import copy_torch_state_to_jax

    torch_state = _load_torch_state_from_snapshot(args.snapshot)
    copy_torch_state_to_jax(
        torch_state,
        model,
        num_layers=args.num_layers,
        arch=args.arch,
    )


def _make_optimizer(model, learning_rate: float, weight_decay: float):
    import optax
    from flax import nnx

    return nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        wrt=nnx.Param,
    )


def _uses_smpl_joint_loss(args: argparse.Namespace) -> bool:
    return bool(args.weight_smpl_joints_rec or args.weight_joints_consistency)


def _future_betas_from_batch(
    batch: Dict[str, np.ndarray],
    *,
    history_length: int,
    future_length: int,
) -> np.ndarray:
    if "betas" not in batch:
        raise KeyError("SMPL joint losses require exported shard betas.")
    return batch["betas"][:, history_length : history_length + future_length, :].astype(
        np.float32,
        copy=False,
    )


def _betas_for_step(
    batch: Dict[str, np.ndarray],
    *,
    history_length: int,
    future_length: int,
    batch_size: int,
    require_betas: bool,
) -> np.ndarray:
    if require_betas:
        return _future_betas_from_batch(
            batch,
            history_length=history_length,
            future_length=future_length,
        )
    if "betas" in batch:
        return _future_betas_from_batch(
            batch,
            history_length=history_length,
            future_length=future_length,
        )
    return np.zeros((batch_size, future_length, 10), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny JAX/NNX DART VAE train smoke.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--snapshot", default=None, help="Optional Torch parity snapshot to initialize from.")
    parser.add_argument("--batch-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--weight-rec", type=float, default=1.0)
    parser.add_argument("--weight-kl", type=float, default=1e-4)
    parser.add_argument("--weight-joints-delta", type=float, default=0.0)
    parser.add_argument("--weight-transl-delta", type=float, default=0.0)
    parser.add_argument("--weight-orient-delta", type=float, default=0.0)
    parser.add_argument("--weight-smpl-joints-rec", type=float, default=0.0)
    parser.add_argument("--weight-joints-consistency", type=float, default=0.0)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument("--smpl-gender", default="male", choices=["male"])
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


def _assert_finite(name: str, value, enabled: bool) -> None:
    if not enabled:
        return
    import jax.numpy as jnp

    if not bool(jnp.isfinite(value).all()):
        raise SystemExit(f"{name} is not finite: {value}")


def main() -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    args = parse_args()
    metadata = load_metadata(args.root)
    feature_dim = int(metadata["primitive"]["feature_dim"])
    feature_slices = metadata["primitive"]["feature_slices"]
    history_length = int(metadata["primitive"]["history_length"])
    future_length = int(metadata["primitive"]["future_length"])
    norm_mean_np, norm_std_np = load_motion_normalization(args.root)
    norm_mean = jnp.asarray(norm_mean_np, dtype=jnp.float32)
    norm_std = jnp.asarray(norm_std_np, dtype=jnp.float32)
    dtype = _jax_dtype(args.dtype)
    use_smpl_joint_loss = _uses_smpl_joint_loss(args)
    if use_smpl_joint_loss and args.smpl_model_dir is None:
        raise SystemExit("--smpl-model-dir is required when SMPL joint loss weights are nonzero.")
    smplx_model = None
    if use_smpl_joint_loss:
        smplx_model = load_smplx_joints_model(args.smpl_model_dir, gender=args.smpl_gender)

    model = _make_model(args, feature_dim)
    _maybe_load_snapshot(model, args)
    optimizer = _make_optimizer(model, args.learning_rate, args.weight_decay)

    @nnx.jit
    def eval_step(model, history, future, betas):
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
            loss = loss + smpl_loss
        else:
            smpl_terms = {
                "smpl_joints_rec": jnp.asarray(0.0, dtype=jnp.float32),
                "joints_consistency": jnp.asarray(0.0, dtype=jnp.float32),
                "smpl_joint_loss": jnp.asarray(0.0, dtype=jnp.float32),
            }
        return (
            loss,
            terms["rec"],
            terms["kl"],
            temporal_terms["joints_delta"],
            temporal_terms["transl_delta"],
            temporal_terms["orient_delta"],
            temporal_terms["temporal_smpl"],
            smpl_terms["smpl_joints_rec"],
            smpl_terms["joints_consistency"],
            smpl_terms["smpl_joint_loss"],
        )

    @nnx.jit
    def train_step(model, optimizer, history, future, betas):
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
                loss = loss + smpl_loss
            else:
                smpl_terms = {
                    "smpl_joints_rec": jnp.asarray(0.0, dtype=jnp.float32),
                    "joints_consistency": jnp.asarray(0.0, dtype=jnp.float32),
                    "smpl_joint_loss": jnp.asarray(0.0, dtype=jnp.float32),
                }
            return loss, (
                terms["rec"],
                terms["kl"],
                temporal_terms["joints_delta"],
                temporal_terms["transl_delta"],
                temporal_terms["orient_delta"],
                temporal_terms["temporal_smpl"],
                smpl_terms["smpl_joints_rec"],
                smpl_terms["joints_consistency"],
                smpl_terms["smpl_joint_loss"],
            )

        (loss, terms), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return (loss, *terms)

    batches = _cycle_vae_batches(args.root, args.batch_samples)
    first_batch = next(batches)
    first_history = jnp.asarray(first_batch["history"], dtype=dtype)
    first_future = jnp.asarray(first_batch["future"], dtype=dtype)
    first_betas = jnp.asarray(
        _betas_for_step(
            first_batch,
            history_length=history_length,
            future_length=future_length,
            batch_size=first_future.shape[0],
            require_betas=use_smpl_joint_loss,
        ),
        dtype=jnp.float32,
    )

    (
        initial_loss,
        initial_rec,
        initial_kl,
        initial_joints_delta,
        initial_transl_delta,
        initial_orient_delta,
        initial_temporal_smpl,
        initial_smpl_joints_rec,
        initial_joints_consistency,
        initial_smpl_joint_loss,
    ) = eval_step(model, first_history, first_future, first_betas)
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("metadata_format:", metadata["format"])
    print("history:", first_history.shape, first_history.dtype)
    print("future:", first_future.shape, first_future.dtype)
    print("initial_loss:", initial_loss)
    print("initial_rec:", initial_rec)
    print("initial_kl:", initial_kl)
    print("initial_joints_delta:", initial_joints_delta)
    print("initial_transl_delta:", initial_transl_delta)
    print("initial_orient_delta:", initial_orient_delta)
    print("initial_temporal_smpl:", initial_temporal_smpl)
    print("initial_smpl_joints_rec:", initial_smpl_joints_rec)
    print("initial_joints_consistency:", initial_joints_consistency)
    print("initial_smpl_joint_loss:", initial_smpl_joint_loss)
    _assert_finite("initial_loss", initial_loss, args.fail_on_nonfinite)

    for step in range(1, args.steps + 1):
        if step == 1:
            batch = first_batch
        else:
            batch = next(batches)
        history = jnp.asarray(batch["history"], dtype=dtype)
        future = jnp.asarray(batch["future"], dtype=dtype)
        betas = jnp.asarray(
            _betas_for_step(
                batch,
                history_length=history_length,
                future_length=future_length,
                batch_size=future.shape[0],
                require_betas=use_smpl_joint_loss,
            ),
            dtype=jnp.float32,
        )
        (
            loss,
            rec,
            kl,
            joints_delta,
            transl_delta,
            orient_delta,
            temporal_smpl,
            smpl_joints_rec,
            joints_consistency,
            smpl_joint_loss,
        ) = train_step(
            model,
            optimizer,
            history,
            future,
            betas,
        )
        _assert_finite("loss", loss, args.fail_on_nonfinite)
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(
                f"step={step} loss={loss} rec={rec} kl={kl} "
                f"temporal_smpl={temporal_smpl} "
                f"smpl_joint_loss={smpl_joint_loss} "
                f"joints_delta={joints_delta} transl_delta={transl_delta} "
                f"orient_delta={orient_delta} "
                f"smpl_joints_rec={smpl_joints_rec} "
                f"joints_consistency={joints_consistency}"
            )

    (
        final_loss,
        final_rec,
        final_kl,
        final_joints_delta,
        final_transl_delta,
        final_orient_delta,
        final_temporal_smpl,
        final_smpl_joints_rec,
        final_joints_consistency,
        final_smpl_joint_loss,
    ) = eval_step(model, first_history, first_future, first_betas)
    _assert_finite("final_loss", final_loss, args.fail_on_nonfinite)
    print("final_loss_on_first_batch:", final_loss)
    print("final_rec_on_first_batch:", final_rec)
    print("final_kl_on_first_batch:", final_kl)
    print("final_joints_delta_on_first_batch:", final_joints_delta)
    print("final_transl_delta_on_first_batch:", final_transl_delta)
    print("final_orient_delta_on_first_batch:", final_orient_delta)
    print("final_temporal_smpl_on_first_batch:", final_temporal_smpl)
    print("final_smpl_joints_rec_on_first_batch:", final_smpl_joints_rec)
    print("final_joints_consistency_on_first_batch:", final_joints_consistency)
    print("final_smpl_joint_loss_on_first_batch:", final_smpl_joint_loss)
    print("loss_delta_on_first_batch:", final_loss - initial_loss)
    print("optimizer_step:", optimizer.step[...])


if __name__ == "__main__":
    main()
