"""Notebook-friendly entrypoint for partial JAX VAE TPU runs.

This wrapper is designed for Kaggle cells such as:

    !python jax_dart/kaggle/vae_partial_run.py --require-tpu --root ... --out-dir ...

All arguments not owned by this wrapper are forwarded directly to
``jax_dart.training.vae_train`` or ``jax_dart.training.vae_train_spmd``.
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
        description="Kaggle TPU wrapper for jax_dart.training.vae_train.",
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
        "--trainer",
        default="single",
        choices=["single", "spmd"],
        help="Trainer backend: known-good single replica or data-sharded SPMD.",
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
    if expect_file:
        ok = path.is_file()
    else:
        ok = path.exists()
    status = "ok" if ok else "missing"
    print(f"{label}: {path} ({status})")
    if expect_file and not ok:
        raise SystemExit(f"{label} not found: {path}")
    if not expect_file and not ok:
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

    if len(devices) > 1:
        print(
            "note: use --trainer spmd for data-sharded multi-device training; "
            "--trainer single remains the fallback path."
        )


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
        print("  --trainer {single,spmd}")
        print()
        print("All other flags are forwarded to the selected trainer.")
        return

    if wrapper_args.platform is not None and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = wrapper_args.platform

    repo_root = _ensure_repo_on_path()
    print("repo_root:", repo_root)
    print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

    root = _arg_value(trainer_args, "--root")
    out_dir = _arg_value(trainer_args, "--out-dir")
    init_npz = _arg_value(trainer_args, "--init-npz")
    smpl_model_dir = _arg_value(trainer_args, "--smpl-model-dir")

    _check_path("root", root)
    if root:
        _check_path("metadata", str(Path(root) / "metadata.json"), expect_file=True)
        _check_path("normalization", str(Path(root) / "normalization.npz"), expect_file=True)
    _check_path("init_npz", init_npz, expect_file=True)
    _check_path("smpl_model_dir", smpl_model_dir)
    if out_dir:
        print("out_dir:", out_dir)

    _runtime_check(wrapper_args.require_tpu, wrapper_args.expect_devices)

    if wrapper_args.check_only:
        print("check_only: done")
        return
    if not trainer_args:
        raise SystemExit("No trainer arguments were provided. Pass the normal vae_train flags after wrapper flags.")

    if wrapper_args.trainer == "spmd":
        sys.argv = ["jax_dart.training.vae_train_spmd"] + trainer_args
        from jax_dart.training.vae_train_spmd import main as train_main
    else:
        sys.argv = ["jax_dart.training.vae_train"] + trainer_args
        from jax_dart.training.vae_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
