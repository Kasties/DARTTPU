"""NNX transformer blocks for the DART Motion Latent Diffusion VAE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .torch_compatible import (
    TorchLayerNorm,
    TorchLinear,
    TorchMultiheadAttention,
)


def _gelu_exact(x):
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _activation(name: str):
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return _gelu_exact
    if name == "glu":
        return jax.nn.glu
    raise ValueError(f"Unsupported activation {name}.")


class LearnedPositionEncoding1D(nnx.Module):
    """Equivalent to DART's ``PositionEmbeddingLearned1D``."""

    def __init__(self, d_model: int, max_len: int = 500, *, rngs: nnx.Rngs):
        self.pe = nnx.Param(jax.random.uniform(rngs.params(), (max_len, 1, d_model)))

    def __call__(self, x):
        return x + self.pe[: x.shape[1], 0, :][None, :, :]


class SinePositionEncoding1D(nnx.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        position = jnp.arange(max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * (-jnp.log(10000.0) / d_model)
        )
        pe = jnp.zeros((max_len, d_model), dtype=jnp.float32)
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: pe[:, 1::2].shape[-1]]))
        self.pe = nnx.Variable(pe)

    def __call__(self, x):
        return x + self.pe[: x.shape[1]][None, :, :]


def build_position_encoding(d_model: int, position_embedding: str, *, rngs: nnx.Rngs):
    if position_embedding in ("learned", "v3"):
        return LearnedPositionEncoding1D(d_model, rngs=rngs)
    if position_embedding in ("sine", "v2"):
        return SinePositionEncoding1D(d_model)
    raise ValueError(f"Unsupported position embedding {position_embedding}.")


class TransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        normalize_before: bool,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.self_attn = TorchMultiheadAttention(d_model, nhead, rngs=rngs)
        self.linear1 = TorchLinear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = TorchLinear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = TorchLayerNorm(d_model, rngs=rngs)
        self.norm2 = TorchLayerNorm(d_model, rngs=rngs)
        self.activation = _activation(activation)
        self.normalize_before = normalize_before
        self.dropout = dropout

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, src)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + src2)

    def forward_pre(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        return src + src2

    def __call__(self, src, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, pos=pos)
        return self.forward_post(src, pos=pos)


class SkipTransformerEncoder(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        normalize_before: bool,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        if num_layers % 2 != 1:
            raise ValueError("SkipTransformerEncoder requires an odd number of layers.")
        num_block = (num_layers - 1) // 2

        def make_layer():
            return TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                rngs=rngs,
            )

        self.input_blocks = nnx.List([make_layer() for _ in range(num_block)])
        self.middle_block = make_layer()
        self.output_blocks = nnx.List([make_layer() for _ in range(num_block)])
        self.linear_blocks = nnx.List(
            [TorchLinear(2 * d_model, d_model, rngs=rngs) for _ in range(num_block)]
        )
        self.norm = TorchLayerNorm(d_model, rngs=rngs)

    def __call__(self, src, pos=None):
        x = src
        xs = []
        for module in self.input_blocks:
            x = module(x, pos=pos)
            xs.append(x)

        x = self.middle_block(x, pos=pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = jnp.concatenate([x, xs.pop()], axis=-1)
            x = linear(x)
            x = module(x, pos=pos)

        return self.norm(x)


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        normalize_before: bool,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.self_attn = TorchMultiheadAttention(d_model, nhead, rngs=rngs)
        self.multihead_attn = TorchMultiheadAttention(d_model, nhead, rngs=rngs)
        self.linear1 = TorchLinear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = TorchLinear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = TorchLayerNorm(d_model, rngs=rngs)
        self.norm2 = TorchLayerNorm(d_model, rngs=rngs)
        self.norm3 = TorchLayerNorm(d_model, rngs=rngs)
        self.activation = _activation(activation)
        self.normalize_before = normalize_before
        self.dropout = dropout

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.multihead_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
        )
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        return self.norm3(tgt + tgt2)

    def forward_pre(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2)
        tgt = tgt + tgt2
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
        )
        tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return tgt + tgt2

    def __call__(self, tgt, memory, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, pos=pos, query_pos=query_pos)
        return self.forward_post(tgt, memory, pos=pos, query_pos=query_pos)


class SkipTransformerDecoder(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        normalize_before: bool,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        if num_layers % 2 != 1:
            raise ValueError("SkipTransformerDecoder requires an odd number of layers.")
        num_block = (num_layers - 1) // 2

        def make_layer():
            return TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                rngs=rngs,
            )

        self.input_blocks = nnx.List([make_layer() for _ in range(num_block)])
        self.middle_block = make_layer()
        self.output_blocks = nnx.List([make_layer() for _ in range(num_block)])
        self.linear_blocks = nnx.List(
            [TorchLinear(2 * d_model, d_model, rngs=rngs) for _ in range(num_block)]
        )
        self.norm = TorchLayerNorm(d_model, rngs=rngs)

    def __call__(self, tgt, memory, pos=None, query_pos=None):
        x = tgt
        xs = []
        for module in self.input_blocks:
            x = module(x, memory, pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, pos=pos, query_pos=query_pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = jnp.concatenate([x, xs.pop()], axis=-1)
            x = linear(x)
            x = module(x, memory, pos=pos, query_pos=query_pos)

        return self.norm(x)
