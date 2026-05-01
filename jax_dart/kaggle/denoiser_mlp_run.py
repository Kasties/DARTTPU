"""Notebook-friendly entrypoint for JAX denoiser TPU checks.

This wrapper is designed for Kaggle cells such as:

    !python jax_dart/kaggle/denoiser_mlp_run.py --require-tpu --task parity ...

Wrapper flags are consumed here. All remaining flags are forwarded to either
``jax_dart.parity.denoiser`` or ``jax_dart.models.mld_denoiser``.
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
        description="Kaggle TPU wrapper for JAX denoiser checks.",
        add_help=False,
    )
    parser.add_argument(
        "--require-tpu",
        action="store_true",
        help="Fail unless JAX's default backend is TPU.",
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
        help="Only print runtime and path checks; do not run a parity/smoke task.",
    )
    parser.add_argument(
        "--task",
        default="parity",
        choices=["parity", "smoke"],
        help="Run parity comparison or the standalone DenoiserMLP smoke.",
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
            f"'{backend}'. In a Kaggle notebook, enable TPU before running this cell."
        )
    if expect_devices is not None and len(devices) < expect_devices:
        raise SystemExit(f"Expected at least {expect_devices} JAX devices, but found {len(devices)}.")


def main() -> None:
    wrapper_args, task_args = _parse_wrapper_args(sys.argv[1:])
    if wrapper_args.kaggle_help:
        print(__doc__.strip())
        print()
        print("Wrapper flags:")
        print("  --require-tpu")
        print("  --expect-devices N")
        print("  --platform {tpu,gpu,cpu}")
        print("  --check-only")
        print("  --task {parity,smoke}")
        print()
        print("All other flags are forwarded to the selected task.")
        return

    if wrapper_args.platform is not None and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = wrapper_args.platform

    repo_root = _ensure_repo_on_path()
    print("repo_root:", repo_root)
    print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

    root = _arg_value(task_args, "--root")
    snapshot = _arg_value(task_args, "--snapshot")
    out_npz = _arg_value(task_args, "--out-npz")

    _check_path("root", root)
    if root:
        _check_path("metadata", str(Path(root) / "metadata.json"), expect_file=True)
        _check_path("shards", str(Path(root) / "shards"))
    _check_path("snapshot", snapshot, expect_file=True)
    if out_npz:
        print("out_npz:", out_npz)

    _runtime_check(wrapper_args.require_tpu, wrapper_args.expect_devices)

    if wrapper_args.check_only:
        print("check_only: done")
        return
    if not task_args:
        raise SystemExit("No task arguments were provided.")

    if wrapper_args.task == "smoke":
        sys.argv = ["jax_dart.models.mld_denoiser"] + task_args
        from jax_dart.models.mld_denoiser import main as task_main
    else:
        sys.argv = ["jax_dart.parity.denoiser"] + task_args
        from jax_dart.parity.denoiser import main as task_main

    task_main()


if __name__ == "__main__":
    main()
