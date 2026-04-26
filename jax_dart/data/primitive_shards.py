"""Load TPU-friendly primitive shards exported from DART.

The exported shard layout is:

    motion: [sample, primitive, frame, feature]

For VAE training we flatten the sample and primitive axes to create:

    history: [sample * primitive, history_length, feature]
    future:  [sample * primitive, future_length, feature]

This module is intentionally small and array-only so it can run unchanged on a
local GPU, Colab TPU, or Cloud TPU VM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


ArrayDict = Dict[str, np.ndarray]


def load_metadata(root: str) -> dict:
    root_path = Path(root)
    with (root_path / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def list_shards(root: str) -> List[Path]:
    paths = sorted((Path(root) / "shards").glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found under {Path(root) / 'shards'}")
    return paths


def load_shard(path: str) -> ArrayDict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def split_motion_to_vae(
    motion: np.ndarray,
    history_length: int,
    future_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``motion [S, P, T, D]`` to VAE ``history`` and ``future`` arrays."""
    if motion.ndim != 4:
        raise ValueError(f"Expected motion [sample, primitive, frame, feature], got {motion.shape}")
    sample_count, primitive_count, frame_count, feature_dim = motion.shape
    required_frames = history_length + future_length
    if frame_count < required_frames:
        raise ValueError(
            f"Motion has {frame_count} frames, but {required_frames} are required."
        )

    motion = motion[:, :, :required_frames, :]
    motion = motion.reshape(sample_count * primitive_count, required_frames, feature_dim)
    history = motion[:, :history_length, :]
    future = motion[:, history_length:required_frames, :]
    return history, future


def iter_vae_batches(
    root: str,
    batch_samples: Optional[int] = None,
) -> Iterator[ArrayDict]:
    """Yield VAE-ready batches from all shards.

    ``batch_samples`` splits each shard along its sample axis before flattening
    primitives. If omitted, each exported shard is yielded as one batch.
    """
    metadata = load_metadata(root)
    primitive = metadata["primitive"]
    history_length = int(primitive["history_length"])
    future_length = int(primitive["future_length"])

    for shard_path in list_shards(root):
        shard = load_shard(str(shard_path))
        motion = shard["motion"]
        sample_count = motion.shape[0]
        step = batch_samples or sample_count
        if sample_count % step != 0:
            raise ValueError(
                f"{shard_path} has {sample_count} samples, not divisible by {step}."
            )
        for start in range(0, sample_count, step):
            end = start + step
            history, future = split_motion_to_vae(
                motion[start:end],
                history_length=history_length,
                future_length=future_length,
            )
            batch = {
                "history": history,
                "future": future,
                "valid": np.repeat(shard["valid"][start:end], motion.shape[1]),
                "gender": np.repeat(shard["gender"][start:end], motion.shape[1]),
            }
            if "betas" in shard:
                betas = shard["betas"][start:end]
                batch["betas"] = betas.reshape(betas.shape[0] * betas.shape[1], *betas.shape[2:])
            if "text_id" in shard:
                text_id = shard["text_id"][start:end]
                batch["text_id"] = text_id.reshape(text_id.shape[0] * text_id.shape[1])
            if "text_embedding" in shard:
                text_embedding = shard["text_embedding"][start:end]
                batch["text_embedding"] = text_embedding.reshape(
                    text_embedding.shape[0] * text_embedding.shape[1],
                    text_embedding.shape[-1],
                )
            yield batch


def _jax_dtype(name: str):
    import jax.numpy as jnp

    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    if name == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {name}.")


def smoke_test(root: str, batch_samples: Optional[int] = None, dtype: str = "bf16"):
    import jax
    import jax.numpy as jnp

    metadata = load_metadata(root)
    batch = next(iter_vae_batches(root, batch_samples=batch_samples))
    jax_dtype = _jax_dtype(dtype)
    history = jnp.asarray(batch["history"], dtype=jax_dtype)
    future = jnp.asarray(batch["future"], dtype=jax_dtype)

    @jax.jit
    def smoke_step(history, future):
        history_mean = jnp.mean(history.astype(jnp.float32))
        future_loss = jnp.mean(future.astype(jnp.float32) ** 2)
        return history_mean, future_loss

    history_mean, future_loss = smoke_step(history, future)
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("metadata_format:", metadata["format"])
    print("history:", history.shape, history.dtype)
    print("future:", future.shape, future.dtype)
    print("history_mean:", history_mean)
    print("future_loss:", future_loss)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test exported DART primitive shards.")
    parser.add_argument("--root", required=True, help="Export root containing metadata.json and shards/.")
    parser.add_argument(
        "--batch-samples",
        type=int,
        default=None,
        help="Optional number of shard samples per yielded batch.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    args = parse_args()
    smoke_test(args.root, batch_samples=args.batch_samples, dtype=args.dtype)


if __name__ == "__main__":
    main()
