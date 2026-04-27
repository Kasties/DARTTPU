"""Minimal JAX SMPL-X joint forward pass.

This module ports only the part of SMPL-X needed by DART's VAE SMPL losses:
shape-blended joint regression plus the rigid body kinematic chain for the
first 22 body joints. It intentionally does not compute vertices, hands, face
landmarks, or SMPL-X expression terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .rotation_conversions import rotation_6d_to_matrix
from .temporal_smpl_loss import denormalize_motion, huber_loss, slice_motion_features


class SmplxJointsModel(NamedTuple):
    v_template: Any
    shapedirs: Any
    j_regressor: Any
    parents: Tuple[int, ...]
    num_betas: int


def resolve_smplx_model_path(model_dir: str, gender: str = "male") -> Path:
    root = Path(model_dir).expanduser()
    if root.is_file():
        return root

    gender_name = gender.upper()
    candidates = [
        root / "smplx" / f"SMPLX_{gender_name}.npz",
        root / f"SMPLX_{gender_name}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find SMPL-X {gender} model under {root}.\nChecked:\n{checked}"
    )


def _dense_array(value) -> np.ndarray:
    array = value
    if isinstance(array, np.ndarray) and array.dtype == object:
        array = array.item()
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array)


def _parents_from_kintree(kintree_table: np.ndarray) -> Tuple[int, ...]:
    kintree = np.asarray(kintree_table, dtype=np.int64)
    raw_parents = kintree[0]
    child_ids = kintree[1]
    id_to_index = {int(child_id): index for index, child_id in enumerate(child_ids)}

    parents = []
    for index, parent_id in enumerate(raw_parents):
        if index == 0:
            parents.append(-1)
        else:
            parents.append(int(id_to_index.get(int(parent_id), int(parent_id))))
    return tuple(parents)


def load_smplx_joints_model(
    model_dir: str,
    *,
    gender: str = "male",
    num_betas: int = 10,
) -> SmplxJointsModel:
    path = resolve_smplx_model_path(model_dir, gender=gender)
    with np.load(path, allow_pickle=True) as data:
        v_template = _dense_array(data["v_template"]).astype(np.float32)
        shapedirs = _dense_array(data["shapedirs"]).astype(np.float32)[..., :num_betas]
        j_regressor = _dense_array(data["J_regressor"]).astype(np.float32)
        parents = _parents_from_kintree(_dense_array(data["kintree_table"]))

    return SmplxJointsModel(
        v_template=jnp.asarray(v_template),
        shapedirs=jnp.asarray(shapedirs),
        j_regressor=jnp.asarray(j_regressor),
        parents=parents,
        num_betas=int(num_betas),
    )


def _as_flat_inputs(betas, global_orient, body_pose, transl=None):
    prefix_shape = tuple(global_orient.shape[:-2])
    flat_count = int(np.prod(prefix_shape))
    betas = jnp.asarray(betas, dtype=jnp.float32).reshape((flat_count, -1))
    global_orient = jnp.asarray(global_orient, dtype=jnp.float32).reshape((flat_count, 3, 3))
    body_pose = jnp.asarray(body_pose, dtype=jnp.float32).reshape((flat_count, -1, 3, 3))
    if transl is None:
        transl = jnp.zeros((flat_count, 3), dtype=jnp.float32)
    else:
        transl = jnp.asarray(transl, dtype=jnp.float32).reshape((flat_count, 3))
    return prefix_shape, betas, global_orient, body_pose, transl


def _batch_rigid_transform(rot_mats, joints, parents):
    rel_joints = joints
    if joints.shape[1] > 1:
        parent_indices = jnp.asarray(parents[1 : joints.shape[1]], dtype=jnp.int32)
        rel_children = joints[:, 1:, :] - joints[:, parent_indices, :]
        rel_joints = rel_joints.at[:, 1:, :].set(rel_children)

    batch_size, joint_count = joints.shape[:2]
    bottom_row = jnp.zeros((batch_size, joint_count, 1, 4), dtype=jnp.float32)
    bottom_row = bottom_row.at[:, :, 0, 3].set(1.0)
    transforms_mat = jnp.concatenate(
        [jnp.concatenate([rot_mats, rel_joints[..., None]], axis=-1), bottom_row],
        axis=-2,
    )

    chain = [transforms_mat[:, 0]]
    for joint_index in range(1, joint_count):
        chain.append(jnp.matmul(chain[parents[joint_index]], transforms_mat[:, joint_index]))
    transforms = jnp.stack(chain, axis=1)
    return transforms[:, :, :3, 3]


def smplx_joints_from_rotmat(
    model: SmplxJointsModel,
    *,
    betas,
    global_orient,
    body_pose,
    transl=None,
    joint_count: int = 22,
):
    """Return SMPL-X body joints shaped ``prefix + [joint_count, 3]``."""
    prefix_shape, betas, global_orient, body_pose, transl = _as_flat_inputs(
        betas,
        global_orient,
        body_pose,
        transl=transl,
    )
    betas = betas[:, : model.num_betas]
    v_shaped = model.v_template[None] + jnp.einsum("bn,vcn->bvc", betas, model.shapedirs)
    joints = jnp.einsum("jv,bvc->bjc", model.j_regressor, v_shaped)
    rot_mats = jnp.concatenate([global_orient[:, None], body_pose], axis=1)
    rot_mats = rot_mats[:, :joint_count]
    joints = joints[:, :joint_count]
    posed_joints = _batch_rigid_transform(rot_mats, joints, model.parents[:joint_count])
    posed_joints = posed_joints + transl[:, None, :]
    return posed_joints.reshape((*prefix_shape, joint_count, 3))


def smplx_joints_from_6d(
    model: SmplxJointsModel,
    *,
    poses_6d,
    betas,
    transl=None,
    joint_count: int = 22,
):
    """Return SMPL-X body joints from PyTorch3D-style 6D pose features."""
    poses_6d = jnp.asarray(poses_6d, dtype=jnp.float32)
    prefix_shape = poses_6d.shape[:-1]
    rot_mats = rotation_6d_to_matrix(poses_6d.reshape((*prefix_shape, 22, 6)))
    return smplx_joints_from_rotmat(
        model,
        betas=betas,
        global_orient=rot_mats[..., 0, :, :],
        body_pose=rot_mats[..., 1:, :, :],
        transl=transl,
        joint_count=joint_count,
    )


def smplx_joints_from_motion(
    model: SmplxJointsModel,
    motion,
    betas,
    *,
    feature_slices: Mapping[str, Tuple[int, int]],
    joint_count: int = 22,
):
    """Return SMPL-X joints from denormalized DART 276-D motion features."""
    features = slice_motion_features(motion, feature_slices)
    return smplx_joints_from_6d(
        model,
        poses_6d=features["poses_6d"],
        betas=betas,
        transl=features["transl"],
        joint_count=joint_count,
    )


def smpl_joint_loss_terms(
    *,
    pred_smpl_joints,
    gt_smpl_joints,
    pred_feature_joints,
):
    """DART VAE SMPL joint loss terms, using already-computed joints."""
    return {
        "smpl_joints_rec": jnp.mean(huber_loss(pred_smpl_joints, gt_smpl_joints)),
        "joints_consistency": jnp.mean(huber_loss(pred_feature_joints, pred_smpl_joints)),
    }


def smpl_joint_loss_from_motion(
    model: SmplxJointsModel,
    *,
    pred_motion,
    gt_motion,
    betas,
    feature_slices: Mapping[str, Tuple[int, int]],
    norm_mean,
    norm_std,
    weight_smpl_joints_rec: float = 0.0,
    weight_joints_consistency: float = 0.0,
):
    """Compute DART's differentiable SMPL-X joint loss terms.

    ``pred_motion`` and ``gt_motion`` are normalized future-motion tensors. The
    GT SMPL joints are explicitly stop-gradient, matching DART's
    ``torch.no_grad()`` target path.
    """
    pred_motion = denormalize_motion(pred_motion, norm_mean, norm_std)
    gt_motion = denormalize_motion(gt_motion, norm_mean, norm_std)

    pred_features = slice_motion_features(pred_motion, feature_slices)
    pred_feature_joints = pred_features["joints"].reshape((*pred_motion.shape[:-1], 22, 3))
    pred_smpl_joints = smplx_joints_from_motion(
        model,
        pred_motion,
        betas,
        feature_slices=feature_slices,
    )
    gt_smpl_joints = smplx_joints_from_motion(
        model,
        gt_motion,
        betas,
        feature_slices=feature_slices,
    )
    gt_smpl_joints = jax.lax.stop_gradient(gt_smpl_joints)

    terms = smpl_joint_loss_terms(
        pred_smpl_joints=pred_smpl_joints,
        gt_smpl_joints=gt_smpl_joints,
        pred_feature_joints=pred_feature_joints,
    )
    loss = (
        weight_smpl_joints_rec * terms["smpl_joints_rec"]
        + weight_joints_consistency * terms["joints_consistency"]
    )
    terms["smpl_joint_loss"] = loss
    return loss, terms
