"""Export DART motion-primitive batches as static TPU-friendly shards.

This script is an offline bridge from the current PyTorch preprocessing stack
to JAX/TPU training. It consumes the existing processed DART dataset pickles
through the current dataset classes, performs canonicalization/feature building
once on CPU/GPU, and writes fixed-shape NumPy shards.

Example:
    python -m data_scripts.export_tpu_primitive_dataset \
        --data-dir ./data/seq_data_zero_male \
        --cfg-path ./config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml \
        --dataset mp_seq_v2 \
        --split train \
        --out-dir ./data/tpu_primitive/mp_h2_f8_r4/train \
        --num-samples 1048576 \
        --batch-size 256 \
        --shard-size 2048 \
        --text-mode none
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


GENDER_TO_ID = {"female": 0, "male": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fixed-shape DART primitive shards for JAX/TPU training."
    )
    parser.add_argument("--data-dir", required=True, help="Processed DART dataset dir.")
    parser.add_argument("--cfg-path", required=True, help="Motion primitive config YAML.")
    parser.add_argument(
        "--dataset",
        default="mp_seq_v2",
        choices=["mp_seq", "mp_seq_v2", "hml3d"],
        help="Dataset class matching the PyTorch training args.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to export.")
    parser.add_argument("--out-dir", required=True, help="Output directory for shards.")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--body-model-dir",
        default=None,
        help=(
            "Directory containing smplx/ and smplh/ body-model subfolders. "
            "Defaults to config_files.data_paths.body_model_dir."
        ),
    )
    parser.add_argument(
        "--skip-body-models",
        action="store_true",
        help=(
            "Patch smplx.build_layer with a dummy object. This is valid for "
            "mp_seq_v2/hml3d exports that use precomputed joints and pelvis_delta."
        ),
    )
    parser.add_argument("--body-type", default="smplx", choices=["smplx", "smplh"])
    parser.add_argument("--weight-scheme", default="uniform")
    parser.add_argument("--prob-static", type=float, default=0.0)
    parser.add_argument("--text-tolerance", type=float, default=0.0)
    parser.add_argument("--enforce-gender", default="male")
    parser.add_argument("--enforce-zero-beta", type=int, default=1)
    parser.add_argument(
        "--text-mode",
        default="none",
        choices=["none", "ids", "inline"],
        help=(
            "none: avoid CLIP and omit text arrays for VAE training; "
            "ids: save text_id shards plus a separate embedding table; "
            "inline: also duplicate text embeddings in every shard."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def maybe_patch_text_embedding(dataset_name: str):
    """Avoid loading CLIP when exporting VAE-only shards."""
    import data_loaders.humanml.data.dataset as dataset_mod

    def load_zero_clip(clip_version, device="cpu"):
        return object()

    def encode_zero_text(clip_model, raw_text, force_empty_zero=True):
        return torch.zeros((len(raw_text), 512), dtype=torch.float32)

    dataset_mod.load_and_freeze_clip = load_zero_clip
    dataset_mod.encode_text = encode_zero_text

    if dataset_name == "hml3d":
        import data_loaders.humanml.data.dataset_hml3d as hml3d_mod

        hml3d_mod.load_and_freeze_clip = load_zero_clip
        hml3d_mod.encode_text = encode_zero_text


def configure_body_model_dir(body_model_dir: Optional[str]):
    """Set and validate DART's body-model path before smpl_utils is imported."""
    import config_files.data_paths as data_paths

    resolved = Path(body_model_dir) if body_model_dir is not None else Path(data_paths.body_model_dir)
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved

    expected = [
        resolved / "smplx" / "SMPLX_MALE.npz",
        resolved / "smplx" / "SMPLX_FEMALE.npz",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "SMPL-X body model files were not found before importing DART's "
            "dataset stack.\n"
            f"Checked body_model_dir: {resolved}\n"
            f"Missing:\n{missing_text}\n"
            "Pass --body-model-dir pointing at the directory that contains "
            "the smplx/ and smplh/ subfolders, for example:\n"
            "  --body-model-dir ./data/smplx_lockedhead_20230207/models_lockedhead"
        )

    data_paths.body_model_dir = resolved
    return resolved


class _UnavailableBodyModel:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "A SMPL body model was called, but --skip-body-models was used. "
            "Use mp_seq_v2/hml3d data with precomputed joints/pelvis_delta, "
            "or rerun with --body-model-dir pointing to real SMPL files."
        )


def patch_body_model_loading():
    import smplx

    smplx.build_layer = lambda *args, **kwargs: _UnavailableBodyModel()


def get_dataset_class(dataset_name: str):
    from data_loaders.humanml.data.dataset import (
        WeightedPrimitiveSequenceDataset,
        WeightedPrimitiveSequenceDatasetV2,
    )
    from data_loaders.humanml.data.dataset_hml3d import HML3dDataset

    if dataset_name == "mp_seq_v2":
        return WeightedPrimitiveSequenceDatasetV2
    if dataset_name == "hml3d":
        return HML3dDataset
    return WeightedPrimitiveSequenceDataset


def torch_to_numpy(value: torch.Tensor):
    return value.detach().cpu().numpy()


def feature_slices(motion_repr: dict[str, int]):
    start = 0
    slices = OrderedDict()
    for name, width in motion_repr.items():
        end = start + int(width)
        slices[name] = [start, end]
        start = end
    return slices


def text_id_array(
    batch: list[dict[str, Any]],
    vocab: OrderedDict[str, int],
    embedding_by_text: dict[str, np.ndarray],
    dataset,
):
    primitive_ids = []
    for primitive in batch:
        ids = []
        for text in primitive["texts"]:
            if text not in vocab:
                vocab[text] = len(vocab)
                embedding = dataset.text_embedding_dict[text]
                embedding_by_text[text] = torch_to_numpy(embedding).astype(np.float32)
            ids.append(vocab[text])
        primitive_ids.append(np.asarray(ids, dtype=np.int32))
    return np.stack(primitive_ids, axis=1)


def batch_to_arrays(
    batch: list[dict[str, Any]],
    *,
    text_mode: str,
    vocab: OrderedDict[str, int],
    embedding_by_text: dict[str, np.ndarray],
    dataset,
):
    motions = []
    betas = []
    text_embeddings = []
    for primitive in batch:
        motion = primitive["motion_tensor_normalized"].squeeze(2).permute(0, 2, 1)
        motions.append(torch_to_numpy(motion).astype(np.float32))
        betas.append(torch_to_numpy(primitive["betas"]).astype(np.float32))
        if text_mode == "inline":
            text_embeddings.append(
                torch_to_numpy(primitive["text_embedding"]).astype(np.float32)
            )

    gender = np.asarray(
        [GENDER_TO_ID[item] for item in batch[0]["gender"]],
        dtype=np.int8,
    )

    arrays = {
        "motion": np.stack(motions, axis=1),
        "betas": np.stack(betas, axis=1),
        "gender": gender,
        "valid": np.ones((len(gender),), dtype=np.bool_),
    }
    if text_mode in ("ids", "inline"):
        arrays["text_id"] = text_id_array(batch, vocab, embedding_by_text, dataset)
    if text_mode == "inline":
        arrays["text_embedding"] = np.stack(text_embeddings, axis=1)
    return arrays


def concat_batches(batches: list[dict[str, np.ndarray]]):
    keys = batches[0].keys()
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def prepare_output_dir(out_dir: Path, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty. Pass --overwrite to write there anyway."
        )
    if overwrite:
        for path in [
            out_dir / "metadata.json",
            out_dir / "normalization.npz",
            out_dir / "text_embeddings.npz",
            out_dir / "text_vocab.json",
        ]:
            if path.exists():
                path.unlink()
        shard_dir = out_dir / "shards"
        if shard_dir.exists():
            for path in shard_dir.glob("shard_*.npz"):
                path.unlink()


def optional_string(value: Optional[str]):
    if value is None:
        return None
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def main():
    args = parse_args()
    if args.shard_size % args.batch_size != 0:
        raise ValueError("--shard-size must be divisible by --batch-size.")
    if args.num_samples < args.shard_size:
        raise ValueError("--num-samples must be at least --shard-size.")

    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    prepare_output_dir(out_dir, args.overwrite)
    shard_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.skip_body_models:
        patch_body_model_loading()
        print("body model loading: skipped")
    else:
        body_model_dir = configure_body_model_dir(args.body_model_dir)
        print(f"using body_model_dir: {body_model_dir}")

    if args.text_mode == "none":
        maybe_patch_text_embedding(args.dataset)

    dataset_class = get_dataset_class(args.dataset)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = dataset_class(
        dataset_path=args.data_dir,
        dataset_name=args.dataset,
        cfg_path=args.cfg_path,
        prob_static=args.prob_static,
        enforce_gender=optional_string(args.enforce_gender),
        enforce_zero_beta=args.enforce_zero_beta,
        body_type=args.body_type,
        split=args.split,
        device=device,
        weight_scheme=args.weight_scheme,
        text_tolerance=args.text_tolerance,
    )

    total_samples = (args.num_samples // args.shard_size) * args.shard_size
    dropped_samples = args.num_samples - total_samples
    batches_per_shard = args.shard_size // args.batch_size
    num_shards = total_samples // args.shard_size
    vocab: OrderedDict[str, int] = OrderedDict()
    embedding_by_text: dict[str, np.ndarray] = {}

    first_shapes = None
    for shard_idx in range(num_shards):
        batch_arrays = []
        for _ in range(batches_per_shard):
            batch = dataset.get_batch(args.batch_size)
            arrays = batch_to_arrays(
                batch,
                text_mode=args.text_mode,
                vocab=vocab,
                embedding_by_text=embedding_by_text,
                dataset=dataset,
            )
            batch_arrays.append(arrays)

        shard = concat_batches(batch_arrays)
        if first_shapes is None:
            first_shapes = {key: list(value.shape) for key, value in shard.items()}
        np.savez_compressed(shard_dir / f"shard_{shard_idx:06d}.npz", **shard)
        print(
            f"wrote shard {shard_idx + 1}/{num_shards}: "
            f"{args.shard_size} samples"
        )

    tensor_mean = torch_to_numpy(dataset.tensor_mean).astype(np.float32)
    tensor_std = torch_to_numpy(dataset.tensor_std).astype(np.float32)
    np.savez_compressed(out_dir / "normalization.npz", mean=tensor_mean, std=tensor_std)

    if args.text_mode in ("ids", "inline"):
        embeddings = np.zeros((len(vocab), 512), dtype=np.float32)
        for text, idx in vocab.items():
            embeddings[idx] = embedding_by_text[text]
        np.savez_compressed(out_dir / "text_embeddings.npz", embeddings=embeddings)
        write_json(out_dir / "text_vocab.json", list(vocab.keys()))

    metadata = {
        "format": "dart_primitive_tpu_npz_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_dir": args.data_dir,
            "cfg_path": args.cfg_path,
            "dataset": args.dataset,
            "split": args.split,
            "weight_scheme": args.weight_scheme,
            "body_type": args.body_type,
        },
        "export": {
            "num_samples_requested": args.num_samples,
            "num_samples": total_samples,
            "dropped_samples": dropped_samples,
            "num_shards": num_shards,
            "shard_size": args.shard_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "text_mode": args.text_mode,
        },
        "primitive": {
            "fps": int(dataset.target_fps),
            "history_length": int(dataset.history_length),
            "future_length": int(dataset.future_length),
            "primitive_length": int(dataset.primitive_length),
            "num_primitive": int(dataset.num_primitive),
            "feature_dim": int(sum(dataset.motion_repr.values())),
            "motion_repr": {key: int(value) for key, value in dataset.motion_repr.items()},
            "feature_slices": feature_slices(dataset.motion_repr),
        },
        "arrays": first_shapes or {},
        "gender_to_id": GENDER_TO_ID,
        "notes": [
            "motion is normalized and shaped [sample, primitive, frame, feature].",
            "betas is shaped [sample, primitive, frame, 10].",
            "gender is one id per sampled sequence.",
            "normalization.npz stores the mean/std used by the source dataset.",
        ],
    }
    write_json(out_dir / "metadata.json", metadata)
    print(f"export complete: {out_dir}")


if __name__ == "__main__":
    main()
