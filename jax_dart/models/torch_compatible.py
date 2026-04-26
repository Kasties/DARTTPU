"""Small NNX layers with PyTorch-compatible parameter layouts.

The goal here is parity first. PyTorch ``nn.Linear`` stores weight as
``[out_features, in_features]`` and ``nn.MultiheadAttention`` stores a packed
``in_proj_weight`` as ``[3 * embed_dim, embed_dim]``. Mirroring those layouts
will make checkpoint comparison and conversion straightforward.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _uniform(rng, shape, bound):
    return jax.random.uniform(rng, shape, minval=-bound, maxval=bound)


def _xavier_uniform(rng, shape):
    fan_in, fan_out = shape[-1], shape[-2]
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    return _uniform(rng, shape, bound)


class TorchLinear(nnx.Module):
    """Linear layer using PyTorch's ``[out, in]`` weight layout."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(
            _uniform(rngs.params(), (out_features, in_features), bound)
        )
        self.bias = nnx.Param(_uniform(rngs.params(), (out_features,), bound))

    def __call__(self, x):
        return jnp.matmul(x, jnp.swapaxes(self.weight[...], -1, -2)) + self.bias[...]


class TorchLayerNorm(nnx.Module):
    """LayerNorm with PyTorch-style ``weight`` and ``bias`` names."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones((normalized_shape,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((normalized_shape,), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return x * self.weight[...] + self.bias[...]


class TorchMultiheadAttention(nnx.Module):
    """Batch-first self/cross attention with PyTorch MHA parameter names."""

    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nnx.Param(
            _xavier_uniform(rngs.params(), (3 * embed_dim, embed_dim))
        )
        self.in_proj_bias = nnx.Param(jnp.zeros((3 * embed_dim,), dtype=jnp.float32))
        self.out_proj_weight = nnx.Param(
            _xavier_uniform(rngs.params(), (embed_dim, embed_dim))
        )
        self.out_proj_bias = nnx.Param(jnp.zeros((embed_dim,), dtype=jnp.float32))

    def _project_qkv(self, query, key, value):
        q_weight, k_weight, v_weight = jnp.split(self.in_proj_weight[...], 3, axis=0)
        q_bias, k_bias, v_bias = jnp.split(self.in_proj_bias[...], 3, axis=0)
        q = jnp.matmul(query, q_weight.T) + q_bias
        k = jnp.matmul(key, k_weight.T) + k_bias
        v = jnp.matmul(value, v_weight.T) + v_bias
        return q, k, v

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return jnp.swapaxes(x, 1, 2)

    def _merge_heads(self, x):
        batch, _, seq_len, _ = x.shape
        x = jnp.swapaxes(x, 1, 2)
        return x.reshape(batch, seq_len, self.embed_dim)

    def __call__(self, query, key, value):
        q, k, v = self._project_qkv(query, key, value)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        scores = scores / math.sqrt(float(self.head_dim))
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = self._merge_heads(attended)
        return (
            jnp.matmul(attended, self.out_proj_weight[...].T)
            + self.out_proj_bias[...]
        )
