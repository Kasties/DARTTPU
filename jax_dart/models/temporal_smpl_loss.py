"""Pure JAX temporal SMPL-feature consistency losses.

These losses mirror the temporal part of DART's VAE SMPL loss. They operate on
the exported normalized 276-D motion representation and do not require a SMPL-X
body-model forward pass.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import jax.numpy as jnp
import numpy as np

from .rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


def load_motion_normalization(root: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(f"{root}/normalization.npz") as data:
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def denormalize_motion(motion, mean, std):
    return motion.astype(jnp.float32) * jnp.asarray(std, dtype=jnp.float32) + jnp.asarray(
        mean,
        dtype=jnp.float32,
    )


def slice_motion_features(motion, feature_slices: Mapping[str, Tuple[int, int]]):
    return {
        name: motion[..., int(bounds[0]) : int(bounds[1])]
        for name, bounds in feature_slices.items()
    }


def huber_loss(pred, target, delta: float = 1.0):
    error = pred - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


def temporal_smpl_feature_losses(
    history_motion,
    future_motion_pred,
    *,
    feature_slices: Mapping[str, Tuple[int, int]],
    norm_mean,
    norm_std,
):
    pred_motion = jnp.concatenate([history_motion[:, -1:, :], future_motion_pred], axis=1)
    pred_motion = denormalize_motion(pred_motion, norm_mean, norm_std)
    features = slice_motion_features(pred_motion, feature_slices)

    transl = features["transl"]
    poses_6d = features["poses_6d"]
    transl_delta = features["transl_delta"][:, :-1, :]
    orient_delta = features["global_orient_delta_6d"][:, :-1, :]
    joints = features["joints"]
    joints_delta = features["joints_delta"][:, :-1, :]

    calc_joints_delta = joints[:, 1:, :] - joints[:, :-1, :]
    calc_transl_delta = transl[:, 1:, :] - transl[:, :-1, :]
    orient = rotation_6d_to_matrix(poses_6d[:, :, :6])
    calc_orient_delta_matrix = jnp.matmul(orient[:, 1:], jnp.swapaxes(orient[:, :-1], -1, -2))
    calc_orient_delta = matrix_to_rotation_6d(calc_orient_delta_matrix)

    return {
        "joints_delta": jnp.mean(huber_loss(calc_joints_delta, joints_delta)),
        "transl_delta": jnp.mean(huber_loss(calc_transl_delta, transl_delta)),
        "orient_delta": jnp.mean(huber_loss(calc_orient_delta, orient_delta)),
    }


def temporal_smpl_feature_loss(
    history_motion,
    future_motion_pred,
    *,
    feature_slices: Mapping[str, Tuple[int, int]],
    norm_mean,
    norm_std,
    weight_joints_delta: float = 0.0,
    weight_transl_delta: float = 0.0,
    weight_orient_delta: float = 0.0,
):
    terms = temporal_smpl_feature_losses(
        history_motion,
        future_motion_pred,
        feature_slices=feature_slices,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    loss = (
        weight_joints_delta * terms["joints_delta"]
        + weight_transl_delta * terms["transl_delta"]
        + weight_orient_delta * terms["orient_delta"]
    )
    terms["temporal_smpl"] = loss
    return loss, terms
