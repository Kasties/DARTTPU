"""Export JAX VAE reconstructions as DART visualization pickle files.

The exported ``.pkl`` files are intentionally close to the generated sequence
files consumed by ``visualize.vis_seq``. Arrays are stored as NumPy so this can
run from a JAX-only environment; the visualizer converts them to Torch tensors
when rendering meshes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata
from jax_dart.models.mld_vae import AutoMldVae
from jax_dart.models.mld_vae_state import (
    load_torch_style_vae_state,
    state_from_npz,
)
from jax_dart.models.temporal_smpl_loss import load_motion_normalization


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


def _load_checkpoint_config(path: str) -> Dict[str, Any]:
    with np.load(path) as data:
        if "config_json" not in data.files:
            return {}
        raw = data["config_json"]
        return json.loads(str(raw.item()))


def _config_value(args: argparse.Namespace, config: Mapping[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def _make_model(args: argparse.Namespace, config: Mapping[str, Any], feature_dim: int):
    from flax import nnx

    latent_dim = args.latent_dim
    if latent_dim is None:
        latent_dim = tuple(config.get("latent_dim", (1, 256)))
    return AutoMldVae(
        nfeats=feature_dim,
        latent_dim=latent_dim,
        h_dim=int(_config_value(args, config, "h_dim", 256)),
        ff_size=int(_config_value(args, config, "ff_size", 1024)),
        num_layers=int(_config_value(args, config, "num_layers", 7)),
        num_heads=int(_config_value(args, config, "num_heads", 4)),
        dropout=float(_config_value(args, config, "dropout", 0.1)),
        arch=str(_config_value(args, config, "arch", "all_encoder")),
        normalize_before=bool(_config_value(args, config, "normalize_before", False)),
        activation=str(_config_value(args, config, "activation", "gelu")),
        position_embedding=str(_config_value(args, config, "position_embedding", "learned")),
        rngs=nnx.Rngs(args.seed),
    )


def _load_state(model, checkpoint: str, args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    num_layers = int(_config_value(args, config, "num_layers", 7))
    arch = str(_config_value(args, config, "arch", "all_encoder"))
    load_torch_style_vae_state(
        model,
        state_from_npz(checkpoint),
        num_layers=num_layers,
        arch=arch,
    )


def _slice_features(motion: np.ndarray, feature_slices: Mapping[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {
        name: motion[..., int(bounds[0]) : int(bounds[1])]
        for name, bounds in feature_slices.items()
    }


def _gender_name(metadata: Mapping[str, Any], gender_id: int) -> str:
    gender_to_id = metadata.get("gender_to_id", {"female": 0, "male": 1})
    id_to_gender = {int(value): str(key) for key, value in gender_to_id.items()}
    return id_to_gender.get(int(gender_id), "male")


def _primitive_record(
    *,
    motion: np.ndarray,
    betas: np.ndarray,
    gender: str,
    fps: int,
    history_length: int,
    future_length: int,
    feature_slices: Mapping[str, Tuple[int, int]],
    text: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    features = _slice_features(motion, feature_slices)
    return {
        "gender": gender,
        "mocap_framerate": int(fps),
        "history_length": int(history_length),
        "future_length": int(future_length),
        "texts": [text],
        "transl": features["transl"].astype(np.float32, copy=False),
        "poses_6d": features["poses_6d"].astype(np.float32, copy=False),
        "joints": features["joints"].reshape(motion.shape[0], 22, 3).astype(np.float32, copy=False),
        "betas": betas[: motion.shape[0], :10].astype(np.float32, copy=False),
        "metadata": metadata,
    }


def _write_pkl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export JAX VAE reconstructions for DART visualization.")
    parser.add_argument("--root", required=True, help="Exported TPU primitive dataset root.")
    parser.add_argument("--checkpoint", required=True, help="JAX VAE checkpoint/latest.npz to load.")
    parser.add_argument("--out-dir", required=True, help="Directory to write visualization pkl files.")
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--h-dim", type=int, default=None)
    parser.add_argument("--ff-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--latent-dim", type=_parse_latent_dim, default=None)
    parser.add_argument("--arch", default=None, choices=["all_encoder", "encoder_decoder"])
    parser.add_argument("--activation", default=None, choices=["gelu", "relu", "glu"])
    parser.add_argument("--position-embedding", default=None, choices=["learned", "v3", "sine", "v2"])
    parser.add_argument("--normalize-before", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    import jax
    import jax.numpy as jnp

    args = parse_args()
    metadata = load_metadata(args.root)
    primitive = metadata["primitive"]
    feature_dim = int(primitive["feature_dim"])
    history_length = int(primitive["history_length"])
    future_length = int(primitive["future_length"])
    fps = int(primitive.get("fps", 30))
    feature_slices = primitive["feature_slices"]
    norm_mean, norm_std = load_motion_normalization(args.root)
    checkpoint_config = _load_checkpoint_config(args.checkpoint)

    model = _make_model(args, checkpoint_config, feature_dim)
    _load_state(model, args.checkpoint, args, checkpoint_config)

    batch = next(iter_vae_batches(args.root, batch_samples=args.batch_samples))
    end_index = args.start_index + args.num_seqs
    history_np = batch["history"][args.start_index:end_index].astype(np.float32, copy=False)
    future_np = batch["future"][args.start_index:end_index].astype(np.float32, copy=False)
    if history_np.shape[0] == 0:
        raise SystemExit(f"No sequences selected from start-index {args.start_index}.")
    if "betas" not in batch:
        raise SystemExit("Exported visualization requires shard betas.")

    dtype = _jax_dtype(args.dtype)
    history = jnp.asarray(history_np, dtype=dtype)
    future = jnp.asarray(future_np, dtype=dtype)
    latent, dist = model.encode(future, history, rng=None)
    future_pred = model.decode(latent, history, nfuture=future.shape[1])
    future_pred_np = np.asarray(jax.device_get(future_pred), dtype=np.float32)

    gt_motion_norm = np.concatenate([history_np, future_np], axis=1)
    pred_motion_norm = np.concatenate([history_np, future_pred_np], axis=1)
    gt_motion = gt_motion_norm * norm_std + norm_mean
    pred_motion = pred_motion_norm * norm_std + norm_mean
    future_error = future_pred_np - future_np
    rec_per_seq = 0.5 * np.minimum(np.abs(future_error), 1.0) ** 2
    rec_per_seq += np.maximum(np.abs(future_error) - 1.0, 0.0)
    rec_per_seq = rec_per_seq.mean(axis=(1, 2))

    out_dir = Path(args.out_dir)
    summary = []
    for local_index in range(history_np.shape[0]):
        batch_index = args.start_index + local_index
        gender = _gender_name(metadata, int(batch["gender"][batch_index]))
        betas = batch["betas"][batch_index].astype(np.float32, copy=False)
        common = {
            "batch_index": int(batch_index),
            "rec": float(rec_per_seq[local_index]),
            "checkpoint": args.checkpoint,
            "root": args.root,
        }
        pred_record = _primitive_record(
            motion=pred_motion[local_index],
            betas=betas,
            gender=gender,
            fps=fps,
            history_length=history_length,
            future_length=future_length,
            feature_slices=feature_slices,
            text=f"jax vae pred {batch_index} rec={rec_per_seq[local_index]:.6f}",
            metadata={**common, "kind": "pred"},
        )
        gt_record = _primitive_record(
            motion=gt_motion[local_index],
            betas=betas,
            gender=gender,
            fps=fps,
            history_length=history_length,
            future_length=future_length,
            feature_slices=feature_slices,
            text=f"ground truth {batch_index}",
            metadata={**common, "kind": "gt"},
        )
        pred_path = out_dir / f"sample_{batch_index:04d}_pred.pkl"
        gt_path = out_dir / f"sample_{batch_index:04d}_gt.pkl"
        _write_pkl(pred_path, pred_record)
        _write_pkl(gt_path, gt_record)
        summary.append(
            {
                "batch_index": int(batch_index),
                "rec": float(rec_per_seq[local_index]),
                "pred": str(pred_path),
                "gt": str(gt_path),
            }
        )
        print(f"wrote pred={pred_path} gt={gt_path} rec={rec_per_seq[local_index]:.6f}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
