"""Multi-device SPMD trainer for the JAX/NNX DART motion primitive VAE.

This keeps the same model, loss, and checkpoint format as ``vae_train.py`` but
places VAE batches on a data-parallel JAX mesh. It is the first TPU v5e-8
throughput path; the single-replica trainer remains available as a fallback.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata
from jax_dart.models.mld_vae import vae_reconstruction_kl_loss
from jax_dart.models.mld_vae_state import save_vae_npz_checkpoint
from jax_dart.models.smplx_joints import load_smplx_joints_model, smpl_joint_loss_from_motion
from jax_dart.models.temporal_smpl_loss import (
    load_motion_normalization,
    temporal_smpl_feature_loss,
)
from jax_dart.training.vae_train import (
    _assert_finite,
    _betas_for_step,
    _config_dict,
    _cycle_vae_batches,
    _jax_dtype,
    _load_initial_state,
    _make_model,
    _make_optimizer,
    _parse_latent_dim,
    _to_float,
    _uses_smpl_joint_loss,
    _write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JAX/NNX DART VAE with data SPMD.")
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
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=0,
        help="Number of local devices to use. Defaults to all local devices.",
    )
    return parser.parse_args()


def _make_data_mesh(jax, device_count: int):
    devices = jax.local_devices()
    if device_count <= 0:
        device_count = len(devices)
    if device_count < 1:
        raise SystemExit("No local JAX devices are available.")
    if device_count > len(devices):
        raise SystemExit(f"Requested {device_count} devices, but only {len(devices)} are visible.")
    return jax.sharding.Mesh(np.asarray(devices[:device_count]), ("data",))


def _device_put_batch(jax, jnp, sharding, batch: np.ndarray, dtype):
    return jax.device_put(jnp.asarray(batch, dtype=dtype), sharding)


def _global_batch_from_batch_samples(metadata: dict, batch_samples: int) -> int:
    return int(batch_samples) * int(metadata["primitive"]["num_primitive"])


def _throughput_row(
    *,
    step_delta: int,
    elapsed_delta: float,
    batch_samples: int,
    global_batch: int,
) -> Dict[str, float]:
    if elapsed_delta <= 0.0 or step_delta <= 0:
        return {
            "steps_per_sec": 0.0,
            "samples_per_sec": 0.0,
            "examples_per_sec": 0.0,
        }
    return {
        "steps_per_sec": float(step_delta / elapsed_delta),
        "samples_per_sec": float((step_delta * batch_samples) / elapsed_delta),
        "examples_per_sec": float((step_delta * global_batch) / elapsed_delta),
    }


def _check_global_batch(metadata: dict, batch_samples: int, device_count: int) -> None:
    global_batch = _global_batch_from_batch_samples(metadata, batch_samples)
    if global_batch % device_count != 0:
        raise SystemExit(
            f"Global VAE batch {global_batch} is not divisible by {device_count} devices. "
            "Change --batch-samples or --data-parallel-devices."
        )


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
    history_length = int(metadata["primitive"]["history_length"])
    future_length = int(metadata["primitive"]["future_length"])
    norm_mean_np, norm_std_np = load_motion_normalization(args.root)
    dtype = _jax_dtype(args.dtype)
    use_smpl_joint_loss = _uses_smpl_joint_loss(args)
    if use_smpl_joint_loss and args.smpl_model_dir is None:
        raise SystemExit("--smpl-model-dir is required when SMPL joint loss weights are nonzero.")

    mesh = _make_data_mesh(jax, args.data_parallel_devices)
    device_count = int(np.asarray(mesh.devices).size)
    _check_global_batch(metadata, args.batch_samples, device_count)
    global_batch = _global_batch_from_batch_samples(metadata, args.batch_samples)
    per_device_batch = global_batch // device_count
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
    vector_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
    norm_mean = jax.device_put(jnp.asarray(norm_mean_np, dtype=jnp.float32), vector_sharding)
    norm_std = jax.device_put(jnp.asarray(norm_std_np, dtype=jnp.float32), vector_sharding)

    mesh_context = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else nullcontext()
    with mesh_context:
        smplx_model = None
        if use_smpl_joint_loss:
            smplx_model = load_smplx_joints_model(args.smpl_model_dir, gender=args.smpl_gender)
        config = _config_dict(args, metadata, feature_dim)
        config["trainer"] = "spmd"
        config["data_parallel_devices"] = int(device_count)
        config["per_device_batch"] = int(per_device_batch)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

        model = _make_model(args, feature_dim)
        _load_initial_state(model, args)
        optimizer = _make_optimizer(model, args.learning_rate, args.weight_decay)

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

        def batch_to_device(batch: Dict[str, np.ndarray]):
            history = _device_put_batch(jax, jnp, data_sharding, batch["history"], dtype)
            future = _device_put_batch(jax, jnp, data_sharding, batch["future"], dtype)
            betas_np = _betas_for_step(
                batch,
                history_length=history_length,
                future_length=future_length,
                batch_size=batch["future"].shape[0],
                require_betas=use_smpl_joint_loss,
            )
            betas = _device_put_batch(jax, jnp, data_sharding, betas_np, jnp.float32)
            return history, future, betas

        def run_eval(step: int) -> Dict[str, float]:
            losses = []
            recs = []
            kls = []
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
                history, future, betas = batch_to_device(batch)
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
                ) = eval_step(model, history, future, betas)
                losses.append(_to_float(loss))
                recs.append(_to_float(rec))
                kls.append(_to_float(kl))
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
                "rec": float(np.mean(recs)),
                "kl": float(np.mean(kls)),
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
                f"rec={row['rec']:.6f} kl={row['kl']:.6f} "
                f"temporal_smpl={row['temporal_smpl']:.6f} "
                f"smpl_joint_loss={row['smpl_joint_loss']:.6f}"
            )
            return row

        print("backend:", jax.default_backend())
        print("devices:", jax.devices())
        print("trainer: spmd")
        print("mesh:", mesh)
        print("data_parallel_devices:", device_count)
        print("batch_samples:", args.batch_samples)
        print("global_batch:", global_batch)
        print("per_device_batch:", per_device_batch)
        print("metadata_format:", metadata["format"])
        print("feature_dim:", feature_dim)
        print("out_dir:", str(out_dir))

        batches = _cycle_vae_batches(args.root, args.batch_samples)
        run_eval(0)
        start_time = time.time()
        last_log_step = 0
        last_log_time = start_time
        last_metrics: Dict[str, object] = {}
        for step in range(1, args.steps + 1):
            batch = next(batches)
            history, future, betas = batch_to_device(batch)
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
            ) = train_step(model, optimizer, history, future, betas)
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
                    "rec": _to_float(rec),
                    "kl": _to_float(kl),
                    "joints_delta": _to_float(joints_delta),
                    "transl_delta": _to_float(transl_delta),
                    "orient_delta": _to_float(orient_delta),
                    "temporal_smpl": _to_float(temporal_smpl),
                    "smpl_joints_rec": _to_float(smpl_joints_rec),
                    "joints_consistency": _to_float(joints_consistency),
                    "smpl_joint_loss": _to_float(smpl_joint_loss),
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
                    f"rec={row['rec']:.6f} kl={row['kl']:.6f} "
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
