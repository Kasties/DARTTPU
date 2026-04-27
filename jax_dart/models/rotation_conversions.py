"""JAX rotation conversion helpers matching PyTorch3D conventions."""

from __future__ import annotations

import jax.numpy as jnp


def _normalize(vector, eps: float = 1e-12):
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def rotation_6d_to_matrix(d6):
    """Convert 6D rotations to matrices using PyTorch3D's row convention."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = jnp.cross(b1, b2, axis=-1)
    return jnp.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix):
    """Convert rotation matrices to PyTorch3D-style first-two-row 6D rotations."""
    return matrix[..., :2, :].reshape((*matrix.shape[:-2], 6))
