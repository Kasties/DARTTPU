"""Notebook-friendly entrypoint for full JAX denoiser TPU training.

Example Kaggle cell:

    !python jax_dart/kaggle/denoiser_train_run.py \
        --require-tpu --expect-devices 8 --platform tpu \
        --root /kaggle/input/dart-shards/train \
        --vae-npz /kaggle/working/vae/latest.npz \
        --out-dir /kaggle/working/denoiser_benchmark \
        --batch-samples 128 --steps 300000

Wrapper flags are consumed here. All remaining flags are forwarded directly to
``jax_dart.training.denoiser_train``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root


def _parse_wrapper_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Kaggle TPU wrapper for jax_dart.training.denoiser_train.",
        add_help=False,
    )
    parser.add_argument(
        "--require-tpu",
        action="store_true",
        help="Fail before training unless JAX's default backend is TPU.",
    )
    parser.add_argument(
        "--expect-devices",
        type=int,
        default=None,
        help="Fail unless at least this many JAX devices are visible.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=["tpu", "gpu", "cpu"],
        help="Set JAX_PLATFORMS before importing JAX if it is not already set.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print runtime and path checks; do not start training.",
    )
    parser.add_argument(
        "--kaggle-help",
        action="store_true",
        help="Show wrapper help, then exit.",
    )
    return parser.parse_known_args(argv)


def _arg_value(args: List[str], flag: str) -> Optional[str]:
    try:
        index = args.index(flag)
    except ValueError:
        return None
    next_index = index + 1
    if next_index >= len(args):
        return None
    return args[next_index]


def _check_path(label: str, path_text: Optional[str], *, expect_file: bool = False) -> None:
    if not path_text:
        return
    path = Path(path_text)
    ok = path.is_file() if expect_file else path.exists()
    status = "ok" if ok else "missing"
    print(f"{label}: {path} ({status})")
    if not ok:
        raise SystemExit(f"{label} not found: {path}")


def _runtime_check(require_tpu: bool, expect_devices: Optional[int]) -> None:
    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    print("jax_backend:", backend)
    print("jax_process_index:", jax.process_index())
    print("jax_process_count:", jax.process_count())
    print("jax_local_device_count:", jax.local_device_count())
    print("jax_devices:", devices)

    if require_tpu and backend != "tpu":
        raise SystemExit(
            "Expected JAX backend 'tpu', but got "
            f"'{backend}'. In a Kaggle notebook, make sure TPU is enabled before running this cell."
        )
    if expect_devices is not None and len(devices) < expect_devices:
        raise SystemExit(f"Expected at least {expect_devices} JAX devices, but found {len(devices)}.")


def main() -> None:
    wrapper_args, trainer_args = _parse_wrapper_args(sys.argv[1:])
    if wrapper_args.kaggle_help:
        print(__doc__.strip())
        print()
        print("Wrapper flags:")
        print("  --require-tpu")
        print("  --expect-devices N")
        print("  --platform {tpu,gpu,cpu}")
        print("  --check-only")
        print()
        print("All other flags are forwarded to jax_dart.training.denoiser_train.")
        return

    if wrapper_args.platform is not None and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = wrapper_args.platform

    repo_root = _ensure_repo_on_path()
    print("repo_root:", repo_root)
    print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

    root = _arg_value(trainer_args, "--root")
    out_dir = _arg_value(trainer_args, "--out-dir")
    vae_npz = _arg_value(trainer_args, "--vae-npz")
    init_npz = _arg_value(trainer_args, "--init-npz")
    smpl_model_dir = _arg_value(trainer_args, "--smpl-model-dir")

    _check_path("root", root)
    if root:
        _check_path("metadata", str(Path(root) / "metadata.json"), expect_file=True)
        _check_path("normalization", str(Path(root) / "normalization.npz"), expect_file=True)
        _check_path("shards", str(Path(root) / "shards"))
    _check_path("vae_npz", vae_npz, expect_file=True)
    _check_path("init_npz", init_npz, expect_file=True)
    _check_path("smpl_model_dir", smpl_model_dir)
    if out_dir:
        print("out_dir:", out_dir)

    _runtime_check(wrapper_args.require_tpu, wrapper_args.expect_devices)

    if wrapper_args.check_only:
        print("check_only: done")
        return
    if not trainer_args:
        raise SystemExit("No trainer arguments were provided.")

    sys.argv = ["jax_dart.training.denoiser_train"] + trainer_args
    from jax_dart.training.denoiser_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
