"""Torch/JAX parity runner for the minimal SMPL-X body-joint forward pass.

Run this in two steps when the Torch DART env and JAX env are separate:

    python -m jax_dart.parity.smplx_joints --mode torch-export ...
    python -m jax_dart.parity.smplx_joints --mode jax-compare ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from jax_dart.data.primitive_shards import list_shards, load_metadata, load_shard


ArrayMap = Dict[str, np.ndarray]
JOINT_NAMES = ("joints",)
GRAD_NAMES = ("loss", "grad_poses_6d", "grad_betas", "grad_transl")
IDENTITY_6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _torch_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Torch smplx joints with the JAX SMPL-X joint port.")
    parser.add_argument("--mode", default="torch-export", choices=["torch-export", "jax-compare"])
    parser.add_argument("--case", default="rest", choices=["rest", "random-betas", "random-pose", "real-shard"])
    parser.add_argument("--smpl-model-dir", default=None, help="Directory containing smplx/SMPLX_MALE.npz.")
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--root", default=None, help="Exported primitive shard root for --case real-shard.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--out-npz", default=None)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--grad-atol", type=float, default=3e-3)
    parser.add_argument("--grad-rtol", type=float, default=3e-3)
    parser.add_argument("--include-grads", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def _require(value: Optional[str], message: str) -> str:
    if value is None:
        raise SystemExit(message)
    return value


def _identity_poses(batch_size: int, frames: int) -> np.ndarray:
    pose = np.tile(IDENTITY_6D, 22).astype(np.float32)
    return np.broadcast_to(pose, (batch_size, frames, 22 * 6)).copy()


def _synthetic_inputs(args: argparse.Namespace) -> ArrayMap:
    rng = np.random.default_rng(args.seed)
    batch_size = int(args.batch_size)
    frames = int(args.frames)
    poses_6d = _identity_poses(batch_size, frames)
    betas = np.zeros((batch_size, frames, 10), dtype=np.float32)
    transl = np.zeros((batch_size, frames, 3), dtype=np.float32)

    if args.case == "random-betas":
        betas = rng.normal(scale=0.25, size=betas.shape).astype(np.float32)
        transl = rng.normal(scale=0.1, size=transl.shape).astype(np.float32)
    elif args.case == "random-pose":
        poses_6d = rng.normal(size=poses_6d.shape).astype(np.float32)
        betas = rng.normal(scale=0.25, size=betas.shape).astype(np.float32)
        transl = rng.normal(scale=0.1, size=transl.shape).astype(np.float32)

    return {"poses_6d": poses_6d, "betas": betas, "transl": transl}


def _real_shard_inputs(args: argparse.Namespace) -> ArrayMap:
    root = _require(args.root, "--root is required for --case real-shard")
    metadata = load_metadata(root)
    feature_slices = metadata["primitive"]["feature_slices"]
    shard = load_shard(str(list_shards(root)[0]))
    motion = shard["motion"]
    sample_count, primitive_count, frame_count, feature_dim = motion.shape
    motion = motion.reshape(sample_count * primitive_count, frame_count, feature_dim)
    betas = shard["betas"].reshape(sample_count * primitive_count, shard["betas"].shape[2], 10)
    gender = np.repeat(shard["gender"], primitive_count)

    gender_id = int(metadata.get("gender_to_id", {}).get(args.gender, 1 if args.gender == "male" else 0))
    keep = np.flatnonzero(gender == gender_id)
    if len(keep) == 0:
        raise SystemExit(f"No {args.gender} samples found in {root}.")
    keep = keep[: args.batch_size]

    frames = min(args.frames, frame_count, betas.shape[1])
    with np.load(Path(root) / "normalization.npz") as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)
    motion = motion[keep, :frames].astype(np.float32) * std + mean
    betas = betas[keep, :frames].astype(np.float32)

    poses_bounds = feature_slices["poses_6d"]
    transl_bounds = feature_slices["transl"]
    joints_bounds = feature_slices["joints"]
    return {
        "poses_6d": motion[..., poses_bounds[0] : poses_bounds[1]].astype(np.float32),
        "betas": betas,
        "transl": motion[..., transl_bounds[0] : transl_bounds[1]].astype(np.float32),
        "feature_joints": motion[..., joints_bounds[0] : joints_bounds[1]].reshape(len(keep), frames, 22, 3).astype(np.float32),
    }


def _make_inputs(args: argparse.Namespace) -> ArrayMap:
    if args.case == "real-shard":
        return _real_shard_inputs(args)
    return _synthetic_inputs(args)


def _torch_huber_loss(pred, target, delta: float = 1.0):
    import torch

    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _torch_smplx_joints(
    inputs: ArrayMap,
    *,
    smpl_model_dir: str,
    gender: str,
    device: str,
    require_grads: bool,
) -> Tuple[Any, Dict[str, Any]]:
    import smplx
    import torch
    from pytorch3d import transforms

    poses_6d = torch.as_tensor(inputs["poses_6d"], device=device, dtype=torch.float32)
    betas = torch.as_tensor(inputs["betas"], device=device, dtype=torch.float32)
    transl = torch.as_tensor(inputs["transl"], device=device, dtype=torch.float32)
    if require_grads:
        poses_6d.requires_grad_(True)
        betas.requires_grad_(True)
        transl.requires_grad_(True)

    batch_size, frames = poses_6d.shape[:2]
    body_model = smplx.build_layer(
        smpl_model_dir,
        model_type="smplx",
        gender=gender,
        ext="npz",
        num_pca_comps=12,
    ).to(device).eval()
    for parameter in body_model.parameters():
        parameter.requires_grad_(False)
    rot_mats = transforms.rotation_6d_to_matrix(poses_6d.reshape(-1, 22, 6))
    output = body_model(
        return_verts=False,
        betas=betas.reshape(-1, 10),
        global_orient=rot_mats[:, 0],
        body_pose=rot_mats[:, 1:],
        transl=transl.reshape(-1, 3),
    )
    joints = output.joints[:, :22].reshape(batch_size, frames, 22, 3)
    return joints, {"poses_6d": poses_6d, "betas": betas, "transl": transl}


def _torch_outputs(
    inputs: ArrayMap,
    *,
    smpl_model_dir: str,
    gender: str,
    device: str,
    include_grads: bool,
) -> Tuple[ArrayMap, Optional[ArrayMap]]:
    import torch

    joints, grad_inputs = _torch_smplx_joints(
        inputs,
        smpl_model_dir=smpl_model_dir,
        gender=gender,
        device=device,
        require_grads=include_grads,
    )
    outputs = {"joints": _torch_numpy(joints)}
    grad_outputs = None
    if include_grads:
        if "feature_joints" in inputs:
            feature_joints = torch.as_tensor(inputs["feature_joints"], device=device, dtype=torch.float32)
            loss = torch.mean(_torch_huber_loss(feature_joints, joints))
        else:
            loss = torch.mean(joints**2)
        loss.backward()
        grad_outputs = {
            "loss": _torch_numpy(loss).reshape(()),
            "grad_poses_6d": _torch_numpy(grad_inputs["poses_6d"].grad),
            "grad_betas": _torch_numpy(grad_inputs["betas"].grad),
            "grad_transl": _torch_numpy(grad_inputs["transl"].grad),
        }
    return outputs, grad_outputs


def _jax_outputs(
    inputs: ArrayMap,
    *,
    smpl_model_dir: str,
    gender: str,
    include_grads: bool,
) -> Tuple[ArrayMap, Optional[ArrayMap]]:
    import jax
    import jax.numpy as jnp

    from jax_dart.models.smplx_joints import load_smplx_joints_model, smplx_joints_from_6d
    from jax_dart.models.temporal_smpl_loss import huber_loss

    model = load_smplx_joints_model(smpl_model_dir, gender=gender)
    poses_6d = jnp.asarray(inputs["poses_6d"], dtype=jnp.float32)
    betas = jnp.asarray(inputs["betas"], dtype=jnp.float32)
    transl = jnp.asarray(inputs["transl"], dtype=jnp.float32)
    joints = smplx_joints_from_6d(model, poses_6d=poses_6d, betas=betas, transl=transl)
    outputs = {"joints": np.asarray(jax.device_get(joints), dtype=np.float32)}
    grad_outputs = None

    if include_grads:
        feature_joints = None
        if "feature_joints" in inputs:
            feature_joints = jnp.asarray(inputs["feature_joints"], dtype=jnp.float32)

        def loss_fn(poses_value, betas_value, transl_value):
            joints_value = smplx_joints_from_6d(
                model,
                poses_6d=poses_value,
                betas=betas_value,
                transl=transl_value,
            )
            if feature_joints is not None:
                return jnp.mean(huber_loss(feature_joints, joints_value))
            return jnp.mean(joints_value**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(poses_6d, betas, transl)
        grad_outputs = {
            "loss": np.asarray(jax.device_get(loss), dtype=np.float32).reshape(()),
            "grad_poses_6d": np.asarray(jax.device_get(grads[0]), dtype=np.float32),
            "grad_betas": np.asarray(jax.device_get(grads[1]), dtype=np.float32),
            "grad_transl": np.asarray(jax.device_get(grads[2]), dtype=np.float32),
        }
    return outputs, grad_outputs


def _save_snapshot(
    path: str,
    *,
    config: Dict[str, Any],
    inputs: ArrayMap,
    torch_outputs: ArrayMap,
    torch_grad_outputs: Optional[ArrayMap] = None,
    jax_outputs: Optional[ArrayMap] = None,
    jax_grad_outputs: Optional[ArrayMap] = None,
) -> None:
    arrays = {"config_json": np.array(json.dumps(config, sort_keys=True))}
    for name, value in inputs.items():
        arrays[name] = value
    for name, value in torch_outputs.items():
        arrays[f"torch_{name}"] = value
    if torch_grad_outputs is not None:
        for name, value in torch_grad_outputs.items():
            arrays[f"torch_{name}"] = value
    if jax_outputs is not None:
        for name, value in jax_outputs.items():
            arrays[f"jax_{name}"] = value
    if jax_grad_outputs is not None:
        for name, value in jax_grad_outputs.items():
            arrays[f"jax_{name}"] = value
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def _load_snapshot(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _load_config(snapshot: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return json.loads(str(snapshot["config_json"].tolist()))


def _snapshot_inputs(snapshot: Dict[str, np.ndarray]) -> ArrayMap:
    inputs = {
        "poses_6d": snapshot["poses_6d"].astype(np.float32, copy=False),
        "betas": snapshot["betas"].astype(np.float32, copy=False),
        "transl": snapshot["transl"].astype(np.float32, copy=False),
    }
    if "feature_joints" in snapshot:
        inputs["feature_joints"] = snapshot["feature_joints"].astype(np.float32, copy=False)
    return inputs


def _load_prefixed(snapshot: Dict[str, np.ndarray], prefix: str, names: Tuple[str, ...]) -> Optional[ArrayMap]:
    if f"{prefix}_{names[0]}" not in snapshot:
        return None
    return {
        name: snapshot[f"{prefix}_{name}"].astype(np.float32, copy=False)
        for name in names
    }


def _compare_arrays(name: str, torch_value: np.ndarray, jax_value: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
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
        f"{name:14s} shape={metrics['shape']} "
        f"max_abs={metrics['max_abs']:.6e} "
        f"mean_abs={metrics['mean_abs']:.6e} "
        f"max_rel={metrics['max_rel']:.6e} "
        f"allclose={metrics['allclose']}"
    )
    return metrics


def _compare_maps(torch_outputs: ArrayMap, jax_outputs: ArrayMap, names: Tuple[str, ...], atol: float, rtol: float):
    return {
        name: _compare_arrays(name, torch_outputs[name], jax_outputs[name], atol=atol, rtol=rtol)
        for name in names
    }


def _maybe_fail(compared: Dict[str, Any]) -> None:
    failed = [name for name, metrics in compared.items() if not metrics["allclose"]]
    if failed:
        raise SystemExit(f"SMPL-X parity failed for: {', '.join(failed)}")


def run_torch_export(args: argparse.Namespace) -> None:
    import torch

    smpl_model_dir = _require(args.smpl_model_dir, "--smpl-model-dir is required for --mode torch-export")
    out_npz = _require(args.out_npz, "--out-npz is required for --mode torch-export")
    device = args.torch_device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = _make_inputs(args)
    torch_outputs, torch_grad_outputs = _torch_outputs(
        inputs,
        smpl_model_dir=smpl_model_dir,
        gender=args.gender,
        device=device,
        include_grads=args.include_grads,
    )
    config = {
        "case": args.case,
        "gender": args.gender,
        "smpl_model_dir": smpl_model_dir,
        "batch_size": int(inputs["poses_6d"].shape[0]),
        "frames": int(inputs["poses_6d"].shape[1]),
        "seed": int(args.seed),
        "include_grads": bool(args.include_grads),
    }
    _save_snapshot(
        out_npz,
        config=config,
        inputs=inputs,
        torch_outputs=torch_outputs,
        torch_grad_outputs=torch_grad_outputs,
    )

    print("torch_device:", device)
    print("case:", args.case)
    for name, value in inputs.items():
        print(name + ":", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    for name, value in torch_outputs.items():
        print("torch_" + name + ":", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    if torch_grad_outputs is not None:
        for name, value in torch_grad_outputs.items():
            print("torch_" + name + ":", value.shape, value.dtype, "finite=", bool(np.isfinite(value).all()))
    print("wrote:", out_npz)


def run_jax_compare(args: argparse.Namespace) -> None:
    import jax

    snapshot_path = _require(args.snapshot, "--snapshot is required for --mode jax-compare")
    snapshot = _load_snapshot(snapshot_path)
    config = _load_config(snapshot)
    inputs = _snapshot_inputs(snapshot)
    smpl_model_dir = args.smpl_model_dir or config.get("smpl_model_dir")
    smpl_model_dir = _require(smpl_model_dir, "--smpl-model-dir is required unless the snapshot stores it.")
    include_grads = bool(args.include_grads or config.get("include_grads", False))

    torch_outputs = _load_prefixed(snapshot, "torch", JOINT_NAMES)
    torch_grad_outputs = _load_prefixed(snapshot, "torch", GRAD_NAMES)
    if torch_outputs is None:
        raise SystemExit(f"{snapshot_path} does not contain Torch joint outputs.")
    if include_grads and torch_grad_outputs is None:
        raise SystemExit(f"{snapshot_path} does not contain Torch gradient outputs.")

    jax_outputs, jax_grad_outputs = _jax_outputs(
        inputs,
        smpl_model_dir=smpl_model_dir,
        gender=config.get("gender", args.gender),
        include_grads=include_grads,
    )

    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
    print("case:", config.get("case"))
    print("joints:", jax_outputs["joints"].shape, jax_outputs["joints"].dtype)
    compared = _compare_maps(torch_outputs, jax_outputs, JOINT_NAMES, atol=args.atol, rtol=args.rtol)
    compared_grads = {}
    if include_grads:
        compared_grads = _compare_maps(
            torch_grad_outputs,
            jax_grad_outputs,
            GRAD_NAMES,
            atol=args.grad_atol,
            rtol=args.grad_rtol,
        )

    if args.out_npz is not None:
        _save_snapshot(
            args.out_npz,
            config=config,
            inputs=inputs,
            torch_outputs=torch_outputs,
            torch_grad_outputs=torch_grad_outputs,
            jax_outputs=jax_outputs,
            jax_grad_outputs=jax_grad_outputs,
        )
        print("wrote:", args.out_npz)

    if args.fail_on_mismatch:
        _maybe_fail(compared)
        _maybe_fail(compared_grads)


def main() -> None:
    args = _parse_args()
    if args.mode == "torch-export":
        run_torch_export(args)
    else:
        run_jax_compare(args)


if __name__ == "__main__":
    main()
